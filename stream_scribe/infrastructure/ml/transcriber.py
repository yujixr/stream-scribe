#!/usr/bin/env python3
"""
Stream Scribe - Transcriber Module
MLX Whisperによる文字起こしを提供するモジュール
"""

import queue
import threading
import time
from datetime import datetime

import mlx_whisper  # type: ignore[import-untyped]
import numpy as np

from stream_scribe.domain import (
    MessageLevel,
    MessagePostedEvent,
    SegmentTranscribedEvent,
    TranscriptionSegment,
    message_posted,
    segment_transcribed,
)
from stream_scribe.domain.constants import (
    SAMPLE_RATE,
    WHISPER_MODEL,
    WHISPER_PARAMS,
)

from .filters import HallucinationFilter
from .transcription_strategy import TranscriptionAction, TranscriptionRetryStrategy


class Transcriber(threading.Thread):
    """
    MLX Whisperによる文字起こしスレッド

    機能:
    - キューベースの非同期処理
    - 幻覚フィルタリング
    - イベント駆動アーキテクチャ（Pub/Sub）
    """

    def __init__(
        self,
        hallucination_filter: HallucinationFilter,
        model_name: str = WHISPER_MODEL,
    ) -> None:
        super().__init__(daemon=True)
        self.queue: queue.Queue[tuple[np.ndarray, datetime, datetime]] = queue.Queue()
        self.running = True
        self._processing = False  # 現在処理中かどうか
        self.model_name = model_name
        self.hallucination_filter = hallucination_filter  # 幻覚フィルター

        message_posted.send(
            None,
            event=MessagePostedEvent(
                message=f"Loading Whisper model: {model_name}",
                level=MessageLevel.INFO,
            ),
        )

        # モデルを事前読み込み（仮音声で初期化）
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1秒の無音
        try:
            mlx_whisper.transcribe(
                dummy_audio, path_or_hf_repo=self.model_name, **WHISPER_PARAMS[0]
            )
            message_posted.send(
                None,
                event=MessagePostedEvent(
                    message="Whisper model ready.\n", level=MessageLevel.SUCCESS
                ),
            )
        except Exception as e:
            message_posted.send(
                None,
                event=MessagePostedEvent(
                    message=f"Warning: Model preload failed: {e}",
                    level=MessageLevel.WARNING,
                ),
            )
            message_posted.send(
                None,
                event=MessagePostedEvent(
                    message="Model will be loaded on first transcription.\n",
                    level=MessageLevel.WARNING,
                ),
            )

    def add_audio(
        self, audio: np.ndarray, start_time: datetime, end_time: datetime
    ) -> None:
        """音声データをキューに追加"""
        if self.running:
            self.queue.put((audio, start_time, end_time))

    def run(self) -> None:
        """文字起こしループ"""
        while self.running or not self.queue.empty():
            try:
                data = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # データを展開
            audio, recording_start, recording_end = data

            # リトライ戦略を使用した文字起こし処理
            self._processing = True
            try:
                self._process_audio(audio, recording_start, recording_end)
            finally:
                self._processing = False

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
                    event = MessagePostedEvent(
                        message=f"Transcription failed: {e}",
                        level=MessageLevel.ERROR,
                    )
                    message_posted.send(self, event=event)
                return

            # テキスト抽出と正規化
            text_raw = result.get("text", "")
            text = str(text_raw).strip() if text_raw else ""
            segments_raw = result.get("segments")
            segments = segments_raw if isinstance(segments_raw, list) else []

            # 音声の長さを計算
            audio_duration = len(audio) / SAMPLE_RATE

            # メトリクスを抽出（フィルタリングと分析用）
            metrics = self.hallucination_filter.extract_metrics(segments)
            avg_logprob = metrics[0]

            # 幻覚検出（テキストパターン、極端に低い信頼度、文脈なし挨拶）
            filter_reason = self.hallucination_filter.evaluate_transcription(
                text, avg_logprob, audio_duration
            )

            # 戦略に評価を委譲
            strategy_result = strategy.evaluate_result(text, filter_reason)

            if strategy_result.action == TranscriptionAction.ACCEPT:
                # 成功：処理時間を計算してセグメントを作成
                processing_time = time.time() - processing_start

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
                segment_event = SegmentTranscribedEvent(segment=segment)
                segment_transcribed.send(self, event=segment_event)
                return

            elif strategy_result.action == TranscriptionAction.RETRY:
                # リトライ：エラーを通知して続行
                attempt, max_attempts = strategy.get_attempt_info()
                error_event = MessagePostedEvent(
                    message=f"Quality issue detected (attempt {attempt - 1}/{max_attempts}): "
                    f"{strategy_result.reason} | Retrying with stricter parameters...",
                    level=MessageLevel.ERROR,
                    timestamp=recording_start,
                )
                message_posted.send(self, event=error_event)
                # whileループで再試行

            else:  # TranscriptionAction.DISCARD
                # 破棄：無音以外の場合はエラーを通知
                if filter_reason:  # 無音ではない場合のみエラー表示
                    attempt, max_attempts = strategy.get_attempt_info()
                    error_event = MessagePostedEvent(
                        message=f"Quality issue filtered (attempt {attempt}/{max_attempts}): "
                        f"{strategy_result.reason} | Text: '{text[:50]}...'",
                        level=MessageLevel.ERROR,
                        timestamp=recording_start,
                    )
                    message_posted.send(self, event=error_event)
                return

    @property
    def is_transcribing(self) -> bool:
        """
        文字起こし中かどうか（キュー待ち + 処理中）

        Returns:
            bool: キューにタスクがあるか、処理中の場合True
        """
        return self._processing or self.queue.qsize() > 0

    def stop(self, wait_for_queue: bool = False) -> None:
        """
        スレッド停止

        Args:
            wait_for_queue: Trueの場合、キューが空になるまで処理を続ける
        """
        # スレッド停止フラグを立てる（新規追加を防ぐ）
        self.running = False

        if not wait_for_queue:
            # キューをクリアして残タスクを破棄
            try:
                while True:
                    self.queue.get_nowait()
            except queue.Empty:
                pass
