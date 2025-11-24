#!/usr/bin/env python3
"""
Stream Scribe - CLI Application
プレゼンテーション層：CLIアプリケーションの制御
"""

import argparse
import os
import sys
import traceback
from datetime import datetime

from colorama import Fore, Style  # type: ignore[import-untyped]
from colorama import init as colorama_init

from stream_scribe.domain.constants import (
    BANNED_PHRASES,
    CHUNK_MS,
    ENABLE_SUMMARY,
    FAST_SHUTDOWN_TIMEOUT_SEC,
    MIN_SPEECH_CHUNKS,
    PREROLL_SEC,
    SUMMARIZER_SHUTDOWN_TIMEOUT_SEC,
    SUMMARY_MODEL,
    TRANSCRIBER_SHUTDOWN_TIMEOUT_SEC,
    VAD_END_THRESHOLD,
    VAD_START_THRESHOLD,
    WHISPER_MODEL,
)
from stream_scribe.domain.models import (
    AudioStreamStatusEvent,
    TranscriptionError,
    TranscriptionSegment,
    TranscriptionSession,
)
from stream_scribe.infrastructure.ai.summarizer import RealtimeSummarizer
from stream_scribe.infrastructure.audio.audio_stream import AudioStream
from stream_scribe.infrastructure.audio.sources import (
    AudioSource,
    FileAudioSource,
    MicrophoneAudioSource,
)
from stream_scribe.infrastructure.audio.vad_detector import VADDetector
from stream_scribe.infrastructure.ml.filters import HallucinationFilter
from stream_scribe.infrastructure.ml.transcriber import Transcriber
from stream_scribe.infrastructure.persistence.json_exporter import SessionJsonExporter
from stream_scribe.presentation.display import DisplayFormatter, StatusDisplay
from stream_scribe.presentation.input_handler import InputHandler

# Colorama初期化
colorama_init(autoreset=True)


class StreamScribeApp:
    """
    Stream Scribe CLIアプリケーション

    責務:
    - コンポーネントの初期化と依存性注入
    - イベントハンドリング
    - セッションのライフサイクル管理
    - UI表示とユーザーインタラクション
    """

    def __init__(
        self,
        api_key: str,
        device_id: int | None = None,
        file_path: str | None = None,
    ):
        self.api_key = api_key
        self.device_id = device_id
        self.file_path = file_path

        # コンポーネント（run()で初期化）
        self.session: TranscriptionSession | None = None
        self.display: StatusDisplay | None = None
        self.vad: VADDetector | None = None
        self.transcriber: Transcriber | None = None
        self.summarizer: RealtimeSummarizer | None = None
        self.audio_stream: AudioStream | None = None

    def print_banner(self) -> None:
        """起動バナー表示"""
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════╗
║       Stream Scribe v1.0                 ║
║  Real-time Conversation Recorder         ║
╚══════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}Config:{Style.RESET_ALL}
  - VAD: Silero VAD v5 (ONNX) [Hysteresis: {VAD_START_THRESHOLD}/{VAD_END_THRESHOLD}]
  - Whisper: {WHISPER_MODEL}
  - Structurer: Claude ({SUMMARY_MODEL})
  - Min Speech: {MIN_SPEECH_CHUNKS} chunks ({MIN_SPEECH_CHUNKS * CHUNK_MS}ms)
  - Preroll: {PREROLL_SEC}s
"""
        print(banner)

    # ========== イベントハンドラ ==========

    def on_segment(self, segment: TranscriptionSegment) -> None:
        """
        セグメント完了時のイベントハンドラ

        以下の処理を順次実行:
        1. セッションへの記録
        2. 画面表示
        3. 要約スレッドへの送信（有効時のみ）
        4. ストリーム処理完了通知
        """
        if not self.session or not self.display:
            return

        # 1. セッションに記録
        self.session.add_segment(segment)

        # 2. 画面表示
        self.display.show_segment(segment)

        # 3. 要約スレッドに送信（有効時のみ）
        if self.summarizer:
            self.summarizer.add_segment(segment.text)

        # 4. ストリーム処理完了を通知
        if (
            self.audio_stream
            and self.transcriber
            and not self.transcriber.is_processing()
        ):
            self.audio_stream.is_transcribing = False

    def on_error(
        self, error_time: datetime, error_message: str, exception: Exception | None
    ) -> None:
        """エラー発生時のイベントハンドラ"""
        # セッションにエラーを記録
        if self.session:
            error = TranscriptionError(
                timestamp=error_time,
                message=error_message,
                exception_type=type(exception).__name__ if exception else None,
            )
            self.session.add_error(error)

        if self.display:
            self.display.show_error(error_time, error_message, exception)

    def on_audio_status_update(self, event: AudioStreamStatusEvent) -> None:
        """AudioStreamのステータス更新イベントハンドラ"""
        if not self.display:
            return

        self.display.update_status(
            probability=event.probability,
            is_recording=event.is_recording,
            is_transcribing=event.is_transcribing,
            is_summarizing=self.summarizer.is_summarizing if self.summarizer else False,
            recording_elapsed=event.recording_elapsed,
            speech_chunks=event.speech_chunks,
            summary_buffer_count=self.summarizer.buffer_char_count
            if self.summarizer
            else 0,
            summary_threshold=self.summarizer.trigger_threshold
            if self.summarizer
            else 0,
        )

    # ========== セッション管理 ==========

    def _cleanup_and_save(self, wait_for_processing: bool = True) -> str | None:
        """
        セッション終了時のクリーンアップと保存処理

        Args:
            wait_for_processing: 残りの文字起こし・サマリ生成を待つかどうか

        Returns:
            str | None: 最終サマリ
        """
        if not self.transcriber or not self.session:
            return None

        final_summary = None

        if wait_for_processing:
            # Transcriberのキューが空になるまで待機
            print(f"\n{Fore.CYAN}Processing remaining audio...{Style.RESET_ALL}")
            self.transcriber.stop(wait_for_queue=True)
            self.transcriber.join(timeout=TRANSCRIBER_SHUTDOWN_TIMEOUT_SEC)

            # タイムアウトした場合の警告
            if self.transcriber.is_alive():
                print(
                    f"\n{Fore.YELLOW}Warning: Transcriber thread did not stop cleanly{Style.RESET_ALL}"
                )

            # 最終サマリを生成してSummarizerを停止（有効時のみ）
            if self.summarizer:
                print(f"\n{Fore.CYAN}Generating final summary...{Style.RESET_ALL}")
                final_summary = self.summarizer.stop(wait_for_final=True)

                if final_summary:
                    # セッションに保存
                    self.session.set_structured_summary(final_summary)

                self.summarizer.join(timeout=SUMMARIZER_SHUTDOWN_TIMEOUT_SEC)

            # JSON保存（wait_for_processing=Trueの場合のみ）
            if self.session.get_total_segments() > 0:
                output_path = SessionJsonExporter.save_to_file(self.session)
                print(
                    f"\n{Fore.GREEN}Transcription saved to: {output_path}{Style.RESET_ALL}"
                )
        else:
            # 即座に停止（JSON保存なし）
            self.transcriber.stop(wait_for_queue=False)
            if self.summarizer:
                self.summarizer.stop(wait_for_final=False)
                self.summarizer.join(timeout=FAST_SHUTDOWN_TIMEOUT_SEC)
            self.transcriber.join(timeout=FAST_SHUTDOWN_TIMEOUT_SEC)

        return final_summary

    def run(self) -> None:
        """メインエントリーポイント"""
        self.print_banner()

        # 1. VAD初期化
        print(f"{Fore.CYAN}Initializing VAD...{Style.RESET_ALL}")
        self.vad = VADDetector()
        print(f"{Fore.GREEN}VAD ready.{Style.RESET_ALL}\n")

        # 2. セッション初期化
        self.session = TranscriptionSession()

        # 3. DisplayFormatter初期化
        formatter = DisplayFormatter()
        self.display = StatusDisplay(formatter)

        # 4. RealtimeSummarizer初期化（ENABLE_SUMMARYがTrueの場合のみ）
        if ENABLE_SUMMARY:
            self.summarizer = RealtimeSummarizer(
                on_summary_update=self.session.set_structured_summary,
                on_summary_display=self.display.show_summary,
                on_error=self.on_error,
                api_key=self.api_key,
            )
            self.summarizer.start()

        # 5. HallucinationFilter初期化
        hallucination_filter = HallucinationFilter(banned_phrases=BANNED_PHRASES)

        # 6. Transcriber初期化（selfのイベントハンドラを使用）
        self.transcriber = Transcriber(
            on_segment=self.on_segment,
            on_error=self.on_error,
            hallucination_filter=hallucination_filter,
        )
        self.transcriber.start()

        # 7. AudioSource初期化（ファイル入力またはマイク入力）
        audio_source: AudioSource
        if self.file_path:
            audio_source = FileAudioSource(file_path=self.file_path)
        else:
            audio_source = MicrophoneAudioSource(device_id=self.device_id)

        # 8. AudioStream初期化（selfのイベントハンドラを使用）
        self.audio_stream = AudioStream(
            vad=self.vad,
            transcriber=self.transcriber,
            on_status_update=self.on_audio_status_update,
            audio_source=audio_source,
        )

        # 8. ストリーム開始と入力監視
        print(
            f"{Fore.GREEN}🎙️  Listening... (Ctrl+C to stop, Ctrl+D for fast exit){Style.RESET_ALL}\n"
        )

        try:
            with self.audio_stream.start():
                InputHandler.wait_for_exit_signal()
        except KeyboardInterrupt:
            print(f"\n{Fore.GREEN}Goodbye!{Style.RESET_ALL}")

            # AudioStreamの録音を停止
            self.audio_stream.on_exit()
            self.display.clear()

            # セッション終了
            final_summary = self._cleanup_and_save(wait_for_processing=True)
            if final_summary:
                self.display.show_summary(final_summary)
        except EOFError:
            # Ctrl-D による高速終了（JSON保存なし）
            print(f"\n{Fore.YELLOW}Fast exit (Ctrl-D){Style.RESET_ALL}")

            # AudioStreamの録音を停止
            self.audio_stream.on_exit()
            self.display.clear()

            # 即座に終了（JSON保存スキップ）
            self._cleanup_and_save(wait_for_processing=False)
        except Exception as e:
            # エラー時は即座に終了
            print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する"""
    parser = argparse.ArgumentParser(
        prog="stream-scribe",
        description="Real-time speech transcription with VAD and Whisper",
    )
    parser.add_argument(
        "-l",
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=None,
        metavar="ID",
        help="Audio input device ID (use --list-devices to see available devices)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        metavar="PATH",
        help="Audio file path (mp3/wav) to transcribe instead of microphone input",
    )
    return parser.parse_args()


def print_audio_devices() -> None:
    """利用可能なオーディオ入力デバイス一覧を表示する"""
    devices = MicrophoneAudioSource.list_devices()

    print(f"\n{Fore.CYAN}Available audio input devices:{Style.RESET_ALL}\n")
    for device in devices:
        default_marker = (
            f" {Fore.GREEN}(default){Style.RESET_ALL}" if device.is_default else ""
        )
        print(f"  [{device.id}] {device.name}{default_marker}")
    print()


def main() -> None:
    """エントリーポイント"""
    # CLI引数解析
    args = parse_args()

    # デバイス一覧表示モード
    if args.list_devices:
        colorama_init(autoreset=True)
        print_audio_devices()
        return

    # APIキーの取得（サマリ生成が有効な場合のみ必須）
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if ENABLE_SUMMARY and not api_key:
        print(
            f"{Fore.RED}Error: ANTHROPIC_API_KEY environment variable is not set.{Style.RESET_ALL}"
        )
        sys.exit(1)

    app = StreamScribeApp(
        api_key=api_key or "",
        device_id=args.device,
        file_path=args.file,
    )
    app.run()


if __name__ == "__main__":
    main()
