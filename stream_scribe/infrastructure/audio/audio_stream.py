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
from typing import Any

import numpy as np

from stream_scribe.domain.constants import (
    AUDIO_STREAM_SHUTDOWN_TIMEOUT_SEC,
    MIN_SPEECH_CHUNKS,
    PREROLL_CHUNKS,
)
from stream_scribe.domain.events import EventHandler
from stream_scribe.infrastructure.audio.sources import AudioSource
from stream_scribe.infrastructure.audio.vad_detector import VADDetector
from stream_scribe.infrastructure.audio.vad_state_machine import (
    VadAction,
    VadStateMachine,
)


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
    - 録音完了時のイベント発行

    責務:
    - VADによる音声区間検出
    - 音声データのバッファリング
    - EventHandlerへの録音完了通知
    """

    def __init__(
        self,
        vad: VADDetector,
        event_handler: EventHandler,
        audio_source: AudioSource,
    ):
        self.vad = vad
        self._event_handler = event_handler
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
        """録音終了（EventHandlerに通知）"""
        recording_end = datetime.now()

        if len(self.recording_buffer) > MIN_SPEECH_CHUNKS and self.recording_start:
            # numpy配列に変換
            audio = np.concatenate(self.recording_buffer)

            # EventHandlerに音声録音完了を通知
            self._event_handler.on_audio(audio, self.recording_start, recording_end)

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

    def __enter__(self) -> "AudioStream":
        """コンテキストマネージャ開始"""
        self._running = True
        self.audio_source.__enter__()

        # 音声処理ループを別スレッドで実行
        self._thread = threading.Thread(
            target=self._audio_processing_loop,
            daemon=True,
            name="AudioStreamThread",
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """コンテキストマネージャ終了"""
        self._running = False

        # 録音中なら停止
        if self.state_machine.is_recording:
            self._stop_recording()

        # スレッド終了を待つ
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=AUDIO_STREAM_SHUTDOWN_TIMEOUT_SEC)

        # AudioSourceをクリーンアップ
        self.audio_source.__exit__(exc_type, exc_val, exc_tb)

    def is_alive(self) -> bool:
        """音声処理スレッドが実行中かどうかを返す"""
        return self._thread.is_alive() if self._thread else False
