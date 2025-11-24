#!/usr/bin/env python3
"""
Stream Scribe - Transcriber Module
MLX Whisperによる文字起こしを提供するモジュール
"""

import queue
import threading
import time
from datetime import datetime
from typing import Callable

import mlx_whisper  # type: ignore[import-untyped]
import numpy as np
from colorama import Fore, Style  # type: ignore[import-untyped]

from stream_scribe.domain.constants import (
    SAMPLE_RATE,
    WHISPER_MODEL,
    WHISPER_PARAMS,
)
from stream_scribe.domain.models import TranscriptionSegment
from stream_scribe.infrastructure.ml.filters import HallucinationFilter
from stream_scribe.infrastructure.ml.transcription_strategy import (
    TranscriptionAction,
    TranscriptionRetryStrategy,
)


class Transcriber(threading.Thread):
    """
    MLX Whisperによる文字起こしスレッド

    Features:
    - キューベースの非同期処理
    - 幻覚フィルタリング
    """

    def __init__(
        self,
        on_segment: Callable[[TranscriptionSegment], None],
        on_error: Callable[[datetime, str, Exception | None], None],
        hallucination_filter: HallucinationFilter,
        model_name: str = WHISPER_MODEL,
    ) -> None:
        super().__init__(daemon=True)
        self.queue: queue.Queue[
            tuple[np.ndarray, datetime, datetime, Callable[[], None]]
        ] = queue.Queue()
        self.running = True
        self.model_name = model_name
        self.on_segment = on_segment  # コールバック関数
        self.on_error = on_error  # エラーコールバック
        self.hallucination_filter = hallucination_filter  # 幻覚フィルター

        print(f"{Fore.CYAN}Loading Whisper model: {model_name}{Style.RESET_ALL}")

        # モデルを事前ロード（ダミー音声で初期化）
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1秒の無音
        try:
            mlx_whisper.transcribe(
                dummy_audio, path_or_hf_repo=self.model_name, **WHISPER_PARAMS[0]
            )
            print(f"{Fore.GREEN}Whisper model ready.{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Model preload failed: {e}{Style.RESET_ALL}")
            print(
                f"{Fore.YELLOW}Model will be loaded on first transcription.{Style.RESET_ALL}\n"
            )

    def add_audio(
        self,
        audio: np.ndarray,
        start_time: datetime,
        end_time: datetime,
        completion_callback: Callable[[], None],
    ) -> None:
        """音声データをキューに追加"""
        self.queue.put((audio, start_time, end_time, completion_callback))

    def run(self) -> None:
        """文字起こしループ"""
        while self.running or not self.queue.empty():
            try:
                data = self.queue.get(timeout=0.5)
            except queue.Empty:
                # runningがFalseでキューも空なら終了
                if not self.running:
                    break
                continue

            # データを展開
            audio, recording_start, recording_end, completion_callback = data

            # リトライ戦略を使用した文字起こし処理
            self._process_audio(audio, recording_start, recording_end)

            # 処理完了コールバックを呼ぶ
            completion_callback()

    def _process_audio(
        self, audio: np.ndarray, recording_start: datetime, recording_end: datetime
    ) -> None:
        """
        リトライ戦略を使用した文字起こし処理

        Args:
            audio: 音声データ
            recording_start: 録音開始時刻
            recording_end: 録音終了時刻
        """
        strategy = TranscriptionRetryStrategy()
        processing_start = time.time()

        while True:
            params = strategy.get_current_params()

            # Whisper推論
            try:
                result = mlx_whisper.transcribe(
                    audio, path_or_hf_repo=self.model_name, **params
                )
            except Exception as e:
                # mlx_whisper自体のエラー（構造的な問題）は再試行しない
                if self.running:
                    self.on_error(recording_end, "Transcription failed", e)
                return

            # テキスト抽出と正規化
            text_raw = result.get("text", "")
            text = str(text_raw).strip() if text_raw else ""
            segments_raw = result.get("segments")
            segments = segments_raw if isinstance(segments_raw, list) else []

            # メトリクスを抽出（フィルタリング + 分析用）
            metrics = self.hallucination_filter.extract_metrics(segments)
            avg_logprob = metrics[0]

            # 幻覚検出（テキストパターン + 極端に低い信頼度）
            filter_reason = self.hallucination_filter.evaluate_transcription(
                text, avg_logprob
            )

            # 戦略に評価を委譲
            strategy_result = strategy.evaluate_result(text, filter_reason)

            if strategy_result.action == TranscriptionAction.ACCEPT:
                # 成功：処理時間を計算してセグメントを作成
                processing_time = time.time() - processing_start
                audio_duration = len(audio) / SAMPLE_RATE

                segment = TranscriptionSegment(
                    text=text,
                    start_time=recording_start,
                    end_time=recording_end,
                    audio_duration=audio_duration,
                    processing_time=processing_time,
                    avg_logprob=metrics[0],
                    compression_ratio=metrics[1],
                    no_speech_prob=metrics[2],
                )
                self.on_segment(segment)
                return

            elif strategy_result.action == TranscriptionAction.RETRY:
                # リトライ：エラーを通知して続行
                attempt, max_attempts = strategy.get_attempt_info()
                self.on_error(
                    recording_start,
                    f"Quality issue detected (attempt {attempt - 1}/{max_attempts}): "
                    f"{strategy_result.reason} | Retrying with stricter parameters...",
                    None,
                )
                # whileループで再試行

            else:  # TranscriptionAction.DISCARD
                # 破棄：無音以外の場合はエラーを通知
                if filter_reason:  # 無音ではない場合のみエラー表示
                    attempt, max_attempts = strategy.get_attempt_info()
                    self.on_error(
                        recording_start,
                        f"Quality issue filtered (attempt {attempt}/{max_attempts}): "
                        f"{strategy_result.reason} | Text: '{text[:50]}...'",
                        None,
                    )
                return

    def is_processing(self) -> bool:
        """
        現在処理中のタスクがあるかどうかを返す

        Returns:
            bool: キューにタスクがある場合True
        """
        return not self.queue.empty()

    def stop(self, wait_for_queue: bool = False) -> None:
        """
        スレッド停止

        Args:
            wait_for_queue: Trueの場合、キューが空になるまで処理を続ける
        """
        if not wait_for_queue:
            # キューをクリアして残タスクを破棄
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

        # スレッド停止フラグを立てる
        self.running = False
