#!/usr/bin/env python3
"""
Stream Scribe - Core Application
プレゼンテーション層：StreamScribeAppコアロジック（CLI/Web共通）
"""

import time

from stream_scribe.domain import (
    AudioRecordedEvent,
    MessageLevel,
    MessagePostedEvent,
    SegmentTranscribedEvent,
    Settings,
    SummaryGeneratedEvent,
    TranscriptionError,
    TranscriptionSession,
    audio_recorded,
    message_posted,
    segment_transcribed,
    summary_generated,
)
from stream_scribe.infrastructure.ai import LLMClient, RealtimeSummarizer
from stream_scribe.infrastructure.audio import (
    AudioSource,
    AudioStream,
    VADDetector,
)
from stream_scribe.infrastructure.ml import HallucinationFilter, Transcriber
from stream_scribe.infrastructure.persistence import SessionJsonExporter


class StreamScribeApp:
    """
    Stream Scribe共通コアアプリケーション（CLI/Web共通）

    責務:
    - コンポーネントの初期化と依存性注入
    - イベントサブスクリプションの設定（Pub/Sub）
    - セッションのライフサイクル管理

    Note:
    - イベント駆動アーキテクチャを採用（blinker使用）
    - 各コンポーネントはイベントバス経由で疎結合に連携
    - UI層は各Signalを直接subscribeして表示を行う
    """

    def __init__(
        self,
        llm_client: LLMClient | None,
        audio_source: AudioSource,
        settings: Settings,
    ):
        """
        StreamScribeAppの初期化

        Args:
            llm_client: LLMクライアント（Noneの場合はサマリー機能無効）
            audio_source: 音声入力ソース
            settings: アプリケーション設定
        """
        self.settings = settings
        self.is_file_mode = not audio_source.is_realtime

        # 1. VAD初期化
        self.vad = VADDetector(
            core_settings=settings.core, vad_model_settings=settings.vad.model
        )

        # 2. セッション初期化
        self.session = TranscriptionSession()

        # 3. RealtimeSummarizer初期化（llm_clientが存在する場合のみ）
        self.summarizer = (
            RealtimeSummarizer(llm_client=llm_client, settings=settings.summary)
            if llm_client
            else None
        )

        # 4. HallucinationFilter初期化
        hallucination_filter = HallucinationFilter(settings=settings.hallucination)

        # 5. Transcriber初期化
        self.transcriber = Transcriber(
            hallucination_filter=hallucination_filter,
            core_settings=settings.core,
            whisper_settings=settings.whisper,
        )

        # 6. AudioStream初期化
        self.audio_stream = AudioStream(
            vad=self.vad,
            audio_source=audio_source,
            vad_detection_settings=settings.vad.detection,
            core_settings=settings.core,
        )

        # 7. イベントサブスクリプション設定（Pub/Sub）
        self._setup_event_subscriptions()

    # ========== イベントサブスクリプション設定 ==========

    def _setup_event_subscriptions(self) -> None:
        """
        イベントサブスクリプションを設定（Pub/Sub）

        各イベントに対するハンドラを登録する。
        blinkerのSignalシステムを使用して疎結合なイベント処理を実現。
        """
        # 音声録音完了イベント → Transcriberへ転送
        audio_recorded.connect(self._on_audio_recorded)

        # 文字起こし完了イベント → セッション記録 + 要約送信
        segment_transcribed.connect(self._on_segment_transcribed)

        # 要約生成イベント → セッション記録
        summary_generated.connect(self._on_summary_generated)

        # メッセージ投稿イベント → ERRORレベルのみセッション記録
        message_posted.connect(self._on_message_posted)

    # ========== イベントハンドラ（Pub/Sub） ==========

    def _on_audio_recorded(self, _sender: object, event: AudioRecordedEvent) -> None:
        """
        音声録音完了時のイベントハンドラ

        AudioStreamからVAD検知により録音された音声を受け取り、
        Transcriberに転送する。

        Args:
            _sender: イベント送信元オブジェクト（未使用）
            event: AudioRecordedEvent（audio, start_time, end_timeを含む）
        """
        self.transcriber.add_audio(event.audio, event.start_time, event.end_time)

    def _on_segment_transcribed(
        self, _sender: object, event: SegmentTranscribedEvent
    ) -> None:
        """
        セグメント完了時のイベントハンドラ

        以下の処理を順次実行:
        1. セッションへの記録
        2. 要約スレッドへの送信（有効時のみ）

        Args:
            _sender: イベント送信元オブジェクト（未使用）
            event: SegmentTranscribedEvent（segmentを含む）
        """
        # 1. セッションに記録
        self.session.add_segment(event.segment)

        # 2. 要約スレッドに送信（有効時のみ）
        if self.summarizer:
            self.summarizer.add_segment(event.segment)

    def _on_summary_generated(
        self, _sender: object, event: SummaryGeneratedEvent
    ) -> None:
        """
        要約生成時のイベントハンドラ（中間サマリ・終了時サマリ共通）

        処理:
        1. セッションに保存

        Args:
            _sender: イベント送信元オブジェクト（未使用）
            event: SummaryGeneratedEvent
        """
        self.session.add_summary(event.summary, is_final=event.is_final)

    def _on_message_posted(self, _sender: object, event: MessagePostedEvent) -> None:
        """
        メッセージ投稿時のイベントハンドラ

        ERRORレベルのメッセージのみセッションにエラーとして記録する。

        Args:
            _sender: イベント送信元オブジェクト（未使用）
            event: MessagePostedEvent（message, level, timestampを含む）
        """
        # ERRORレベルのメッセージのみセッションに記録
        if event.level == MessageLevel.ERROR:
            error = TranscriptionError(timestamp=event.timestamp, message=event.message)
            self.session.add_error(error)

    # ========== 録音制御 ==========

    def start_recording(self) -> None:
        """
        録音開始（ワーカースレッド起動 + AudioStream開始）

        Note:
        - 初回のみワーカースレッド（Transcriber, Summarizer）を起動
        - AudioStreamを開始
        """
        # ワーカースレッドを起動（初回のみ）
        if not self.transcriber.is_alive():
            self.transcriber.start()

        if self.summarizer and not self.summarizer.is_alive():
            self.summarizer.start()

        # AudioStreamを開始
        self.audio_stream.start()

    def pause_recording(self) -> None:
        """
        録音一時停止（ワーカースレッドは継続、AudioStreamのみ停止）
        """
        self.audio_stream.pause()

    def resume_recording(self) -> None:
        """
        録音再開（AudioStreamを再開）
        """
        self.audio_stream.resume()

    # ========== セッション管理 ==========

    def shutdown(self, graceful: bool = True) -> None:
        """
        セッションの終了処理（録音停止 + ワーカー停止 + 保存）

        Args:
            graceful: Trueなら残り処理を完了させてから保存、Falseなら即座に終了
        """
        # 1. AudioStream停止
        self.audio_stream.stop()

        # 2. Transcriber/Summarizer停止
        self._stop_workers(graceful)

        # 3. セッション保存
        self._save_session()

    def _stop_workers(self, graceful: bool) -> None:
        """
        ワーカースレッドの停止

        Args:
            graceful: Trueなら残り処理を完了させてから停止
        """

        if graceful:
            # 残りのキューを処理（進捗を表示）
            if self.transcriber.is_transcribing:
                message_posted.send(
                    None,
                    event=MessagePostedEvent(
                        message="Processing remaining audio...", level=MessageLevel.INFO
                    ),
                )
                last_remaining = -1
                while self.transcriber.is_transcribing:
                    remaining = self.transcriber.queue.qsize()
                    if remaining > 0 and remaining != last_remaining:
                        message_posted.send(
                            None,
                            event=MessagePostedEvent(
                                message=f"  Transcribing... ({remaining} segments remaining)",
                                level=MessageLevel.WARNING,
                            ),
                        )
                        last_remaining = remaining
                    time.sleep(
                        self.settings.app.transcription_progress_poll_interval_sec
                    )

            self.transcriber.stop(wait_for_queue=True)
            self.transcriber.join(timeout=self.settings.whisper.shutdown_timeout_sec)

            if self.transcriber.is_alive():
                message_posted.send(
                    None,
                    event=MessagePostedEvent(
                        message="Warning: Transcriber thread did not stop cleanly",
                        level=MessageLevel.WARNING,
                    ),
                )

            # 終了時サマリの生成
            if self.summarizer:
                message_posted.send(
                    None,
                    event=MessagePostedEvent(
                        message="Generating final summary...", level=MessageLevel.INFO
                    ),
                )
                # リアルタイムサマリ処理を破棄し、終了時サマリを生成
                # Note: stop()内のシグナル送信は同期的なため、復帰時点でセッションに追加済み
                self.summarizer.stop(session=self.session)
                self.summarizer.join(timeout=self.settings.summary.shutdown_timeout_sec)
        else:
            self.transcriber.stop(wait_for_queue=False)
            if self.summarizer:
                # サマリ生成せずに即座に終了
                self.summarizer.stop(session=None)
                self.summarizer.join(timeout=1.0)
            self.transcriber.join(timeout=1.0)

    def _save_session(self) -> None:
        """
        セッションの保存（JSON出力）
        """
        if not self.settings.app.save_json or self.session.get_total_segments() == 0:
            return

        output_path = SessionJsonExporter.save_to_file(self.session)
        message_posted.send(
            None,
            event=MessagePostedEvent(
                message=f"Transcription saved to: {output_path}",
                level=MessageLevel.SUCCESS,
            ),
        )
