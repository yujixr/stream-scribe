#!/usr/bin/env python3
"""
Stream Scribe - CLI View
CLIのView層。Signalをsubscribeして表示を担当する。
"""

from stream_scribe.domain import (
    MessagePostedEvent,
    SegmentTranscribedEvent,
    SummaryGeneratedEvent,
    message_posted,
    segment_transcribed,
    summary_generated,
)
from stream_scribe.infrastructure.ai import RealtimeSummarizer
from stream_scribe.infrastructure.audio import AudioStream
from stream_scribe.infrastructure.ml import Transcriber

from .display import StatusDisplay
from .status_update_manager import StatusUpdateManager


class CLIView:
    """
    CLI View層

    StatusDisplayとStatusUpdateManagerを統合し、Signalベースで表示を行う。
    各Signalをsubscribeして、適切なCLI表示を実行する。
    """

    def __init__(self) -> None:
        """
        CLIViewの初期化とSignalサブスクリプション設定

        Note:
            StatusUpdateManagerはstart()で設定・起動する（遅延初期化）
        """
        self._display = StatusDisplay()
        self._status_manager: StatusUpdateManager | None = None

        # Signalサブスクリプション設定
        segment_transcribed.connect(self._on_segment_transcribed)
        summary_generated.connect(self._on_summary_generated)
        message_posted.connect(self._on_message_posted)

    # ========== Signalハンドラ ==========

    def _on_segment_transcribed(
        self, _sender: object, event: SegmentTranscribedEvent
    ) -> None:
        """文字起こしセグメント表示ハンドラ"""
        self._display.show_segment(event.segment)

    def _on_summary_generated(
        self, _sender: object, event: SummaryGeneratedEvent
    ) -> None:
        """サマリー表示ハンドラ"""
        self._display.show_summary(event.summary)

    def _on_message_posted(self, _sender: object, event: MessagePostedEvent) -> None:
        """ステータスメッセージ表示ハンドラ"""
        self._display.show_message(event)

    # ========== ライフサイクル制御 ==========

    def clear(self) -> None:
        """表示をクリア"""
        self._display.clear()

    def start(
        self,
        audio_stream: AudioStream,
        transcriber: Transcriber,
        summarizer: RealtimeSummarizer | None,
    ) -> None:
        """
        UI更新を開始

        Args:
            audio_stream: AudioStreamインスタンス
            transcriber: Transcriberインスタンス
            summarizer: RealtimeSummarizerインスタンス（Noneの場合はサマリー無効）
        """
        # ステータス更新マネージャーの設定と開始
        self._status_manager = StatusUpdateManager(
            audio_stream=audio_stream,
            transcriber=transcriber,
            display=self._display,
            summarizer=summarizer,
        )
        self._status_manager.start()

    def stop(self) -> None:
        """UI更新を停止"""
        if self._status_manager:
            self._status_manager.stop()
