#!/usr/bin/env python3
"""
Stream Scribe - Audio Stream Module
リアルタイム音声ストリーム管理を提供するモジュール
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from stream_scribe.domain import AudioRecordedEvent, audio_recorded
from stream_scribe.domain.constants import (
    AUDIO_STREAM_SHUTDOWN_TIMEOUT_SEC,
    MIN_SPEECH_CHUNKS,
    PREROLL_CHUNKS,
)

from .sources import AudioSource
from .vad_detector import VADDetector
from .vad_state_machine import VadAction, VadStateMachine


@dataclass
class AudioStreamStatus:
    """音声ストリームの状態"""

    probability: float  # VAD確率
    is_recording: bool  # 録音中かどうか
    recording_elapsed: float  # 録音経過時間（秒）
    speech_chunks: int  # 音声チャンク数


class AudioStream:
    """
    リアルタイム音声ストリーム管理

    機能:
    - リングバッファ（プリロール）
    - VAD検知とステートマシン
    - 音声ソースの抽象化（依存性注入）
    - 録音完了時のイベント発行（Pub/Sub）

    責務:
    - VADによる音声区間検出
    - 音声データのバッファリング
    - 録音完了イベントの発行
    """

    def __init__(
        self,
        vad: VADDetector,
        audio_source: AudioSource,
    ):
        self.vad = vad
        self.audio_source = audio_source

        # プリロール用リングバッファ（collections.deque）
        self.preroll_ring_buffer: deque[np.ndarray] = deque(maxlen=PREROLL_CHUNKS)

        # 録音バッファ
        self.recording_buffer: list[np.ndarray] = []

        # VAD状態遷移ロジック（委譲）
        self.state_machine = VadStateMachine()

        # 録音タイムスタンプ
        self.recording_start: datetime | None = None
        self.recording_start_mono: float | None = (
            None  # time.time()ベース（経過時間計算用）
        )

        # 現在のVAD確率（状態取得用）
        self._current_probability = 0.0

        # スレッド制御
        self._running = False
        self._thread: threading.Thread | None = None

    def get_status(self) -> AudioStreamStatus:
        """現在の状態を取得"""
        recording_elapsed = 0.0
        if self.state_machine.is_recording and self.recording_start_mono:
            recording_elapsed = time.time() - self.recording_start_mono

        return AudioStreamStatus(
            probability=self._current_probability,
            is_recording=self.state_machine.is_recording,
            recording_elapsed=recording_elapsed,
            speech_chunks=self.state_machine.speech_chunks,
        )

    def process_chunk(self, chunk: np.ndarray) -> None:
        """チャンク単位のVAD処理"""
        # VAD推論
        probability = self.vad(chunk)

        # 現在の確率を保存
        self._current_probability = probability

        # プリロールバッファに追加
        self.preroll_ring_buffer.append(chunk)

        # ステートマシンで状態遷移を処理
        action = self.state_machine.process(probability)

        # アクションに応じた処理
        if action == VadAction.START_RECORDING:
            self._start_recording()
        elif action == VadAction.STOP_RECORDING:
            self._stop_recording()
        elif action == VadAction.RESET_VAD_MODEL:
            self.vad.reset_states()

        # 録音中ならデータをバッファへ
        if self.state_machine.is_recording:
            self.recording_buffer.append(chunk)

    def _start_recording(self) -> None:
        """録音開始（プリロールを結合）"""
        self.recording_start = datetime.now()
        self.recording_start_mono = time.time()
        self.recording_buffer = list(self.preroll_ring_buffer)

    def _stop_recording(self) -> None:
        """録音終了（イベント発行）"""
        recording_end = datetime.now()

        if len(self.recording_buffer) > MIN_SPEECH_CHUNKS and self.recording_start:
            # numpy配列に変換
            audio = np.concatenate(self.recording_buffer)

            # イベント発行（Pub/Sub）
            event = AudioRecordedEvent(
                audio=audio, start_time=self.recording_start, end_time=recording_end
            )
            audio_recorded.send(self, event=event)

        # リセット
        self.recording_buffer = []
        self.recording_start = None
        self.recording_start_mono = None
        self.vad.reset_states()

    def _audio_processing_loop(self) -> None:
        """音声処理ループ（別スレッドで実行）"""
        for chunk in self.audio_source.stream():
            if self._running:
                self.process_chunk(chunk)
            else:
                break

        # ストリーム終了時に録音中の場合は停止
        if self.state_machine.is_recording:
            self._stop_recording()

    def start(self) -> None:
        """音声ストリーム開始"""
        if self._thread and self._thread.is_alive():
            return  # すでに起動済み

        self._running = True
        self.audio_source.start()

        # 音声処理ループを別スレッドで実行
        self._thread = threading.Thread(
            target=self._audio_processing_loop,
            daemon=True,
            name="AudioStreamThread",
        )
        self._thread.start()

    def pause(self) -> None:
        """音声ストリーム一時停止（スレッドは継続、処理のみ停止）"""
        self._running = False

    def resume(self) -> None:
        """音声ストリーム再開"""
        self._running = True

    def stop(self) -> None:
        """音声ストリーム停止"""
        self._running = False

        # 録音中なら停止
        if self.state_machine.is_recording:
            self._stop_recording()

        # スレッド終了を待つ
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=AUDIO_STREAM_SHUTDOWN_TIMEOUT_SEC)

        # AudioSourceをクリーンアップ
        self.audio_source.stop()

    def is_alive(self) -> bool:
        """音声処理スレッドが実行中かどうかを返す"""
        return self._thread.is_alive() if self._thread else False
