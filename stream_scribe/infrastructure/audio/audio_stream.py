#!/usr/bin/env python3
"""
Stream Scribe - Audio Stream Module
リアルタイム音声ストリーム管理を提供するモジュール
"""

import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Callable

import numpy as np

from stream_scribe.domain.constants import (
    AUDIO_BLOCK_SEC,
    MAX_SILENCE_CHUNKS,
    MIN_SPEECH_CHUNKS,
    PREROLL_CHUNKS,
    VAD_END_THRESHOLD,
    VAD_IDLE_RESET_CHUNKS,
    VAD_START_THRESHOLD,
)
from stream_scribe.domain.models import AudioStreamStatusEvent
from stream_scribe.infrastructure.audio.sources import AudioSource
from stream_scribe.infrastructure.audio.vad_detector import VADDetector
from stream_scribe.infrastructure.ml.transcriber import Transcriber


class AudioStream:
    """
    リアルタイム音声ストリーム管理

    Features:
    - リングバッファ (プリロール)
    - VAD検知とステートマシン
    - リアルタイム可視化
    - 音声ソースの抽象化 (Dependency Injection)
    """

    def __init__(
        self,
        vad: VADDetector,
        transcriber: Transcriber,
        on_status_update: Callable[[AudioStreamStatusEvent], None],
        audio_source: AudioSource,
    ):
        self.vad = vad
        self.transcriber = transcriber
        self.on_status_update = on_status_update
        self.audio_source = audio_source

        # プリロール用リングバッファ (collections.deque)
        self.preroll_ring_buffer: deque[np.ndarray] = deque(maxlen=PREROLL_CHUNKS)

        # 録音バッファと状態管理
        self.recording_buffer: list[np.ndarray] = []
        self.is_recording = False
        self.is_transcribing = False
        self.silence_chunks = 0
        self.speech_chunks = 0
        self.idle_silence_chunks = 0  # 待機中の無音カウンタ（LSTM状態リセット用）
        self.recording_start: datetime | None = None
        self.recording_start_mono: float | None = (
            None  # time.time()ベース（経過時間計算用）
        )

        # スレッド制御
        self._running = False
        self._thread: threading.Thread | None = None

    def process_chunk(self, chunk: np.ndarray) -> None:
        """チャンク単位のVAD処理（日本語最適化版：ヒステリシス制御）"""
        # 1. VAD推論
        probability = self.vad(chunk)

        # 2. 録音経過時間を計算
        recording_elapsed = 0.0
        if self.is_recording and self.recording_start_mono:
            recording_elapsed = time.time() - self.recording_start_mono

        # 3. ステータスイベントを発行
        status_event = AudioStreamStatusEvent(
            probability=probability,
            is_recording=self.is_recording,
            is_transcribing=self.is_transcribing,
            recording_elapsed=recording_elapsed,
            speech_chunks=self.speech_chunks,
        )
        self.on_status_update(status_event)

        # 4. プリロールバッファに追加
        self.preroll_ring_buffer.append(chunk)

        # === 5. ステートマシン (ヒステリシス制御) ===

        # 現在の状態に応じて閾値を切り替える
        if self.is_recording:
            # 録音中は「終わらせない」ために低い閾値を使う（語尾保護）
            is_speech = probability >= VAD_END_THRESHOLD
        else:
            # 待機中は「誤検知しない」ために高い閾値を使う（ノイズ対策）
            is_speech = probability >= VAD_START_THRESHOLD

        if is_speech:
            self.silence_chunks = 0
            self.speech_chunks += 1
            self.idle_silence_chunks = 0  # 音声検出時は待機中カウンタをリセット

            # 録音開始判定
            if not self.is_recording and self.speech_chunks >= MIN_SPEECH_CHUNKS:
                self.start_recording()

            # 録音中ならデータをバッファへ
            if self.is_recording:
                self.recording_buffer.append(chunk)

        else:
            # 音声ではないと判定された場合
            self.speech_chunks = 0

            if self.is_recording:
                self.silence_chunks += 1
                self.recording_buffer.append(chunk)

                # 録音終了判定 (無音が指定期間続いたら終了)
                if self.silence_chunks >= MAX_SILENCE_CHUNKS:
                    self.stop_recording()
            else:
                # 待機中の無音カウンタを更新
                self.idle_silence_chunks += 1

                # 長時間無音が続いたらLSTM状態をリセット（検出感度の劣化を防止）
                if self.idle_silence_chunks >= VAD_IDLE_RESET_CHUNKS:
                    self.vad.reset_states()
                    self.idle_silence_chunks = 0

    def start_recording(self) -> None:
        """録音開始 (プリロールを結合)"""
        self.is_recording = True
        self.recording_start = datetime.now()
        self.recording_start_mono = time.time()
        self.recording_buffer = list(self.preroll_ring_buffer)

    def stop_recording(self) -> None:
        """録音終了 (Transcriberに送信)"""
        self.is_recording = False
        self.recording_start_mono = None
        recording_end = datetime.now()

        if len(self.recording_buffer) > MIN_SPEECH_CHUNKS and self.recording_start:
            # numpy配列に変換
            audio = np.concatenate(self.recording_buffer)

            self.is_transcribing = True

            # Transcriberに送信（録音開始/終了時刻と処理完了コールバックを含む）
            self.transcriber.add_audio(
                audio, self.recording_start, recording_end, self.complete_processing
            )

        # リセット
        self.recording_buffer = []
        self.silence_chunks = 0
        self.recording_start = None
        self.vad.reset_states()

    def complete_processing(self) -> None:
        """処理完了を通知（Transcriberのコールバックから呼ばれる）"""
        # キューが空の場合のみ is_transcribing をFalseにする
        if not self.transcriber.is_processing():
            self.is_transcribing = False

    def _audio_processing_loop(self) -> None:
        """音声処理ループ（別スレッドで実行）"""
        for chunk in self.audio_source.stream():
            if not self._running:
                break
            self.process_chunk(chunk)

        # ストリーム終了時に録音中の場合は停止
        if self.is_recording:
            self.stop_recording()

        # 音声入力終了後も文字起こし処理中はステータス更新を継続
        while self._running and self.transcriber.is_processing():
            status_event = AudioStreamStatusEvent(
                probability=0.0,
                is_recording=False,
                is_transcribing=self.is_transcribing,
                recording_elapsed=0.0,
                speech_chunks=0,
            )
            self.on_status_update(status_event)
            time.sleep(AUDIO_BLOCK_SEC)

    @contextmanager
    def start(self) -> Generator["AudioStream", None, None]:
        """
        ストリーム開始

        コンテキストマネージャとして使用:
        ```
        with audio_stream.start():
            # 音声処理が別スレッドで実行される
            wait_for_exit_signal()
        ```
        """
        self._running = True

        # AudioSource をコンテキストマネージャで開始
        with self.audio_source:
            # 音声処理ループを別スレッドで実行
            self._thread = threading.Thread(
                target=self._audio_processing_loop,
                daemon=True,
                name="AudioStreamThread",
            )
            self._thread.start()

            try:
                yield self
            finally:
                self._running = False
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=2.0)

    def on_exit(self) -> None:
        """終了時の処理"""
        self._running = False
        if self.is_recording:
            self.stop_recording()

    def wait_for_completion(self) -> None:
        """スレッド終了を待機（Transcriberキューが空になるまで待機済み）"""
        if self._thread and self._thread.is_alive():
            self._thread.join()
