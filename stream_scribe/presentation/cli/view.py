#!/usr/bin/env python3
"""
Stream Scribe - CLI View
CLIのView層：Signal購読とコンソール表示の統合管理
"""

import os
import re
import sys
import threading
import time

import wcwidth  # type: ignore[import-untyped]
from colorama import Fore, Style  # type: ignore[import-untyped]

from stream_scribe import __version__
from stream_scribe.domain import (
    MessageLevel,
    MessagePostedEvent,
    SegmentTranscribedEvent,
    Settings,
    SummaryGeneratedEvent,
    TranscriptionSegment,
    message_posted,
    segment_transcribed,
    summary_generated,
)
from stream_scribe.infrastructure.ai import LLMClient, RealtimeSummarizer
from stream_scribe.infrastructure.audio import AudioStream
from stream_scribe.infrastructure.ml import Transcriber


class CLIView:
    """
    CLI View層

    責務:
    - Signalサブスクリプションとイベント駆動表示
    - コンソール表示のフォーマッティング
    - ステータスバーのリアルタイム更新
    - スレッドセーフな表示管理
    """

    # ANSIエスケープコード削除用パターン（コンパイル済み）
    _ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, settings: Settings) -> None:
        """
        CLIViewの初期化とSignalサブスクリプション設定

        Args:
            settings: アプリケーション設定
        """
        self.settings = settings
        self.lock = threading.Lock()  # スレッド間の同期用ロック
        self.session_start_time = time.time()  # セッション開始時刻

        # ステータス更新スレッド制御
        self._running = False
        self._update_thread: threading.Thread | None = None
        self._audio_stream: AudioStream | None = None
        self._transcriber: Transcriber | None = None
        self._summarizer: RealtimeSummarizer | None = None

        # Signalサブスクリプション設定
        segment_transcribed.connect(self._on_segment_transcribed)
        summary_generated.connect(self._on_summary_generated)
        message_posted.connect(self._on_message_posted)

    # ========== Signalハンドラ ==========

    def _on_segment_transcribed(
        self, _sender: object, event: SegmentTranscribedEvent
    ) -> None:
        """文字起こしセグメント表示ハンドラ"""
        self._show_segment(event.segment)

    def _on_summary_generated(
        self, _sender: object, event: SummaryGeneratedEvent
    ) -> None:
        """サマリー表示ハンドラ"""
        self._show_summary(event.summary)

    def _on_message_posted(self, _sender: object, event: MessagePostedEvent) -> None:
        """ステータスメッセージ表示ハンドラ"""
        self._show_message(event)

    # ========== ライフサイクル制御 ==========

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
        self._audio_stream = audio_stream
        self._transcriber = transcriber
        self._summarizer = summarizer

        # ステータス更新スレッドを開始
        if self._running:
            return

        self._running = True
        self._update_thread = threading.Thread(
            target=self._status_update_loop,
            daemon=True,
            name="StatusUpdateThread",
        )
        self._update_thread.start()

    def stop(self) -> None:
        """UI更新を停止して表示をクリア"""
        self._running = False
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(
                timeout=self.settings.app.status_update_manager_shutdown_timeout_sec
            )

        # 表示をクリア
        with self.lock:
            sys.stdout.write("\r\033[K\n")
            sys.stdout.flush()

    # ========== ステータス更新ループ ==========

    def _status_update_loop(self) -> None:
        """ステータス更新ループ（別スレッドで実行）"""
        while self._running:
            if self._audio_stream and self._transcriber:
                # AudioStreamから状態を取得
                audio_status = self._audio_stream.get_status()

                # ステータスバーを更新
                self._update_status_bar(
                    probability=audio_status.probability,
                    is_recording=audio_status.is_recording,
                    is_transcribing=self._transcriber.is_transcribing,
                    is_summarizing=self._summarizer.is_summarizing
                    if self._summarizer
                    else False,
                    recording_elapsed=audio_status.recording_elapsed,
                    speech_chunks=audio_status.speech_chunks,
                    summary_buffer_count=self._summarizer.buffer_char_count
                    if self._summarizer
                    else 0,
                    summary_threshold=self._summarizer.settings.trigger_threshold
                    if self._summarizer
                    else 0,
                )

            time.sleep(self.settings.app.status_update_interval_sec)

    # ========== 表示メソッド ==========

    def _update_status_bar(
        self,
        probability: float,
        is_recording: bool,
        is_transcribing: bool,
        is_summarizing: bool,
        recording_elapsed: float,
        speech_chunks: int,
        summary_buffer_count: int,
        summary_threshold: int,
    ) -> None:
        """ステータスバーを更新"""
        # ロックが取得できない場合はスキップ（セグメント表示中）
        if not self.lock.acquire(blocking=False):
            return

        try:
            terminal_width = os.get_terminal_size().columns

            # VADセクション構築
            bar_width = 20
            probability_bar_length = int(probability * bar_width)
            probability_bar = "|" * probability_bar_length + "." * (
                bar_width - probability_bar_length
            )
            probability_color = (
                Fore.GREEN
                if probability >= self.settings.vad.detection.start_threshold
                else Fore.CYAN
            )
            vad_section = f"{probability_color}VAD:[{probability_bar}] {probability:.2f}{Style.RESET_ALL}"

            # ステータステキスト構築
            status_parts = []
            if is_recording:
                status_parts.append(
                    f"{Fore.RED}● REC [{recording_elapsed:.1f}s]{Style.RESET_ALL}"
                )
            elif speech_chunks > 0:
                speech_duration = speech_chunks * self.settings.core.chunk_ms / 1000.0
                status_parts.append(f"🎧 Listening (speech: {speech_duration:.2f}s)")
            else:
                status_parts.append("🎧 Listening (idle)")

            if is_transcribing:
                status_parts.append(f"{Fore.MAGENTA}⏳ Transcribing{Style.RESET_ALL}")
            if is_summarizing:
                status_parts.append(f"{Fore.YELLOW}📝 Summarizing...{Style.RESET_ALL}")

            status_text = " | ".join(status_parts)

            # 右側セクション構築
            session_elapsed = time.time() - self.session_start_time
            session_minutes = int(session_elapsed // 60)
            session_seconds = int(session_elapsed % 60)
            session_time_str = f"{session_minutes}m{session_seconds:02d}s"
            summary_progress = f"{summary_buffer_count}/{summary_threshold}"
            right_section = (
                f"{Fore.CYAN}Session: {session_time_str}{Style.RESET_ALL} | "
                f"{Fore.YELLOW}Buffer: {summary_progress}{Style.RESET_ALL}"
            )

            # 左側セクション構築（オーバーフロー対応）
            full_left = f"{vad_section} | {status_text}"
            left_width = self._get_display_width(full_left)
            right_width = self._get_display_width(right_section)
            overflow = left_width + right_width - terminal_width

            if overflow > 0:
                status_width = self._get_display_width(status_text)
                target_status_width = status_width - overflow - 3
                if target_status_width > 0:
                    status_text = (
                        self._truncate_text(status_text, target_status_width) + "..."
                    )
                else:
                    status_text = "..."
                left_section = f"{vad_section} | {status_text}"
            else:
                left_section = full_left

            # 最終ステータスライン構築
            left_width = self._get_display_width(left_section)
            right_width = self._get_display_width(right_section)
            padding_width = max(0, terminal_width - left_width - right_width)
            padding = " " * padding_width

            status_line = f"\r\033[K{left_section}{padding}{right_section}"
            sys.stdout.write(status_line)
            sys.stdout.flush()
        finally:
            self.lock.release()

    def _show_segment(self, segment: TranscriptionSegment) -> None:
        """セグメント結果を表示"""
        timestamp = segment.start_time.strftime("%H:%M:%S")
        time_info = f"{Fore.MAGENTA}(audio: {segment.audio_duration:.2f}s, proc: {segment.processing_time:.2f}s){Style.RESET_ALL}"

        with self.lock:
            # 現在のステータスバーをクリア
            sys.stdout.write("\r\033[K")
            sys.stdout.write(
                f"{Fore.GREEN}[{timestamp}]{Style.RESET_ALL} {segment.text} {time_info}\n"
            )
            sys.stdout.flush()

    def _show_message(self, event: MessagePostedEvent) -> None:
        """メッセージを表示"""
        # メッセージレベルに応じた色を選択
        color_map = {
            MessageLevel.INFO: Fore.CYAN,
            MessageLevel.SUCCESS: Fore.GREEN,
            MessageLevel.WARNING: Fore.YELLOW,
            MessageLevel.ERROR: Fore.RED,
        }
        color = color_map.get(event.level, Fore.WHITE)

        with self.lock:
            # 現在のステータスバーをクリア
            sys.stdout.write("\r\033[K")
            sys.stdout.write(f"{color}{event.message}{Style.RESET_ALL}\n")
            sys.stdout.flush()

    def _show_summary(self, summary_text: str) -> None:
        """リアルタイム議事録をダッシュボード形式で表示"""
        with self.lock:
            # 現在のステータスバーをクリア
            sys.stdout.write("\r\033[K")

            # 要約ヘッダーと内容を表示
            print(f"\n{Fore.CYAN}{'─' * 50}{Style.RESET_ALL}")
            print(summary_text)
            print(f"{Fore.CYAN}{'─' * 50}{Style.RESET_ALL}\n")

            sys.stdout.flush()

    def show_banner(self, llm_client: LLMClient | None) -> None:
        """
        起動バナーを表示

        Args:
            llm_client: LLMクライアント（Noneの場合はサマリー無効として表示）
        """
        # バージョン文字列の表示：.dev以降をカット
        version_display = (
            __version__.split(".dev")[0] if ".dev" in __version__ else __version__
        )

        # LLMバックエンド情報を取得
        llm_info = llm_client.get_backend_info() if llm_client else "Disabled"

        # 設定値を取得
        vad_start = self.settings.vad.detection.start_threshold
        vad_end = self.settings.vad.detection.end_threshold
        whisper_model = self.settings.whisper.model
        min_speech_chunks = self.settings.vad.detection.min_speech_chunks
        chunk_ms = self.settings.core.chunk_ms
        preroll_sec = self.settings.vad.detection.preroll_sec

        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════╗
║       Stream Scribe v{version_display:<18}  ║
║  Real-time Conversation Recorder         ║
╚══════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}Config:{Style.RESET_ALL}
  - VAD: Silero VAD v5 (ONNX) [Hysteresis: {vad_start}/{vad_end}]
  - Whisper: {whisper_model}
  - Structurer: {llm_info}
  - Min Speech: {min_speech_chunks} chunks ({min_speech_chunks * chunk_ms}ms)
  - Preroll: {preroll_sec}s

"""
        sys.stdout.write(banner)
        sys.stdout.flush()

    # ========== フォーマッティングメソッド ==========

    def _get_display_width(self, text: str) -> int:
        """ANSIエスケープコードを除いた実際の表示幅を取得"""
        plain_text = self._ANSI_ESCAPE_PATTERN.sub("", text)
        return int(wcwidth.wcswidth(plain_text))

    def _truncate_text(self, text: str, max_width: int) -> str:
        """テキストを指定された表示幅に切り詰める（ANSIカラーコードを考慮）"""
        plain_text = self._ANSI_ESCAPE_PATTERN.sub("", text)

        # プレーンテキストで何文字目まで含められるか計算
        width = 0
        for i, char in enumerate(plain_text):
            char_width = wcwidth.wcwidth(char)
            if char_width < 0:
                char_width = 0
            if width + char_width > max_width:
                # プレーン文字i文字目が上限に達した
                # 元の文字列でプレーン文字i文字目の位置を探して切り詰め
                plain_count, in_escape = 0, False
                for j, c in enumerate(text):
                    if c == "\x1b":
                        in_escape = True
                    elif in_escape:
                        if c == "m":
                            in_escape = False
                    else:
                        if plain_count == i:
                            return text[:j]
                        plain_count += 1
                return text
            width += char_width

        return text
