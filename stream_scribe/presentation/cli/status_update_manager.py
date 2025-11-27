#!/usr/bin/env python3
"""
Stream Scribe - Status Update Manager
ステータス更新を管理するモジュール
"""

import threading
import time

from stream_scribe.domain.constants import (
    STATUS_UPDATE_INTERVAL_SEC,
    STATUS_UPDATE_MANAGER_SHUTDOWN_TIMEOUT_SEC,
)
from stream_scribe.infrastructure.ai import RealtimeSummarizer
from stream_scribe.infrastructure.audio import AudioStream
from stream_scribe.infrastructure.ml import Transcriber

from .display import StatusDisplay


class StatusUpdateManager:
    """
    ステータス更新マネージャー

    責務:
    - 各コンポーネントから状態を定期的に取得
    - StatusDisplayに更新を指示
    - 音声入力終了後も書き起こし処理中は継続
    """

    def __init__(
        self,
        audio_stream: AudioStream,
        transcriber: Transcriber,
        display: StatusDisplay,
        summarizer: RealtimeSummarizer | None = None,
    ):
        self.audio_stream = audio_stream
        self.transcriber = transcriber
        self.display = display
        self.summarizer = summarizer

        # スレッド制御
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """ステータス更新スレッドを開始"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="StatusUpdateThread",
        )
        self._thread.start()

    def stop(self) -> None:
        """ステータス更新スレッドを停止"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=STATUS_UPDATE_MANAGER_SHUTDOWN_TIMEOUT_SEC)

    def _update_loop(self) -> None:
        """ステータス更新ループ（別スレッドで実行）"""
        while self._running:
            # AudioStreamから状態を取得
            audio_status = self.audio_stream.get_status()

            # StatusDisplayに更新を指示
            self.display.update_status(
                probability=audio_status.probability,
                is_recording=audio_status.is_recording,
                is_transcribing=self.transcriber.is_transcribing,
                is_summarizing=self.summarizer.is_summarizing
                if self.summarizer
                else False,
                recording_elapsed=audio_status.recording_elapsed,
                speech_chunks=audio_status.speech_chunks,
                summary_buffer_count=self.summarizer.buffer_char_count
                if self.summarizer
                else 0,
                summary_threshold=self.summarizer.trigger_threshold
                if self.summarizer
                else 0,
            )

            time.sleep(STATUS_UPDATE_INTERVAL_SEC)
