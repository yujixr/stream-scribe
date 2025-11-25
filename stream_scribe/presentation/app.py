#!/usr/bin/env python3
"""
Stream Scribe - CLI Application
プレゼンテーション層：CLIアプリケーションの制御
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

from colorama import Fore, Style  # type: ignore[import-untyped]
from colorama import init as colorama_init

from stream_scribe import __version__
from stream_scribe.domain.constants import (
    BANNED_PHRASES,
    CHUNK_MS,
    FAST_SHUTDOWN_TIMEOUT_SEC,
    MIN_SPEECH_CHUNKS,
    PREROLL_SEC,
    SUMMARIZER_SHUTDOWN_TIMEOUT_SEC,
    SUMMARY_MODEL,
    TRANSCRIBER_SHUTDOWN_TIMEOUT_SEC,
    TRANSCRIPTION_PROGRESS_POLL_INTERVAL_SEC,
    VAD_END_THRESHOLD,
    VAD_START_THRESHOLD,
    WHISPER_MODEL,
)
from stream_scribe.domain.models import (
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
from stream_scribe.presentation.status_update_manager import StatusUpdateManager

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
        enable_summary: bool = True,
    ):
        self.file_path = file_path

        # 1. VAD初期化
        print(f"{Fore.CYAN}Initializing VAD...{Style.RESET_ALL}")
        self.vad = VADDetector()
        print(f"{Fore.GREEN}VAD ready.{Style.RESET_ALL}\n")

        # 2. セッション初期化
        self.session = TranscriptionSession()

        # 3. DisplayFormatter初期化
        formatter = DisplayFormatter()
        self.display = StatusDisplay(formatter)

        # 4. RealtimeSummarizer初期化（enable_summaryがTrueかつAPIキーが存在する場合のみ）
        self.summarizer: RealtimeSummarizer | None = None
        if enable_summary and api_key:
            self.summarizer = RealtimeSummarizer(
                on_summary=self.on_summary,
                on_error=self.on_error,
                api_key=api_key,
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
        if file_path:
            audio_source = FileAudioSource(file_path=file_path)
        else:
            audio_source = MicrophoneAudioSource(device_id=device_id)

        # 8. AudioStream初期化
        self.audio_stream = AudioStream(
            vad=self.vad,
            transcriber=self.transcriber,
            audio_source=audio_source,
        )

        # 9. StatusUpdateManager初期化と開始
        self.status_update_manager = StatusUpdateManager(
            audio_stream=self.audio_stream,
            transcriber=self.transcriber,
            display=self.display,
            summarizer=self.summarizer,
        )
        self.status_update_manager.start()

    def print_banner(self) -> None:
        """起動バナー表示"""
        # バージョン文字列の表示：.dev以降をカット
        version_display = (
            __version__.split(".dev")[0] if ".dev" in __version__ else __version__
        )
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════╗
║       Stream Scribe v{version_display:<18}  ║
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
        # 1. セッションに記録
        self.session.add_segment(segment)

        # 2. 画面表示
        self.display.show_segment(segment)

        # 3. 要約スレッドに送信（有効時のみ）
        if self.summarizer:
            self.summarizer.add_segment(segment.text)

    def on_summary(self, summary: str) -> None:
        """
        要約生成時のイベントハンドラ

        処理:
        1. セッションに保存
        2. 画面表示
        """
        self.session.set_structured_summary(summary)
        self.display.show_summary(summary)

    def on_error(
        self, error_time: datetime, error_message: str, exception: Exception | None
    ) -> None:
        """エラー発生時のイベントハンドラ"""
        # セッションにエラーを記録
        error = TranscriptionError(timestamp=error_time, message=error_message)
        self.session.add_error(error)

        self.display.show_error(error_time, error_message, exception)

    # ========== セッション管理 ==========

    def _shutdown(self, graceful: bool = True) -> None:
        """
        セッションの終了処理（コンポーネント停止 + 保存 + サマリ表示）

        Args:
            graceful: Trueなら残り処理を完了させてから保存、Falseなら即座に終了
        """
        # 1. ステータス更新マネージャーを停止
        self.status_update_manager.stop()

        # 2. ディスプレイをクリア
        self.display.clear()

        # 3. Transcriber/Summarizer停止
        final_summary = self._stop_workers(graceful)

        # 4. 最終サマリ表示
        if final_summary:
            self.display.show_summary(final_summary)

        # 5. セッション保存
        self._save_session(final_summary)

    def _stop_workers(self, graceful: bool) -> str | None:
        """
        ワーカースレッドの停止

        Args:
            graceful: Trueなら残り処理を完了させてから停止

        Returns:
            str | None: 最終サマリ（graceful=Trueの場合のみ）
        """
        final_summary = None

        if graceful:
            # 残りのキューを処理（進捗を表示）
            if self.transcriber.is_transcribing:
                print(f"{Fore.CYAN}Processing remaining audio...{Style.RESET_ALL}")
                last_remaining = -1
                while self.transcriber.is_transcribing:
                    remaining = self.transcriber.queue.qsize()
                    if remaining > 0 and remaining != last_remaining:
                        print(
                            f"{Fore.YELLOW}  Transcribing... ({remaining} segments remaining){Style.RESET_ALL}"
                        )
                        last_remaining = remaining
                    time.sleep(TRANSCRIPTION_PROGRESS_POLL_INTERVAL_SEC)

            self.transcriber.stop(wait_for_queue=True)
            self.transcriber.join(timeout=TRANSCRIBER_SHUTDOWN_TIMEOUT_SEC)

            if self.transcriber.is_alive():
                print(
                    f"{Fore.YELLOW}Warning: Transcriber thread did not stop cleanly{Style.RESET_ALL}"
                )

            # 終了時サマリの生成
            if self.summarizer:
                print(f"{Fore.CYAN}Generating final summary...{Style.RESET_ALL}")
                # リアルタイムサマリ処理を破棄し、終了時サマリを生成
                final_summary = self.summarizer.stop(session=self.session)
                self.summarizer.join(timeout=SUMMARIZER_SHUTDOWN_TIMEOUT_SEC)
        else:
            self.transcriber.stop(wait_for_queue=False)
            if self.summarizer:
                # サマリ生成せずに即座に終了
                self.summarizer.stop(session=None)
                self.summarizer.join(timeout=FAST_SHUTDOWN_TIMEOUT_SEC)
            self.transcriber.join(timeout=FAST_SHUTDOWN_TIMEOUT_SEC)

        return final_summary

    def _save_session(self, final_summary: str | None) -> None:
        """
        セッションの保存（サマリ設定 + JSON出力）

        Args:
            final_summary: 最終サマリ（Noneでなければセッションに設定）
        """
        if final_summary:
            self.session.set_structured_summary(final_summary)

        if self.session.get_total_segments() > 0:
            output_path = SessionJsonExporter.save_to_file(self.session)
            print(f"{Fore.GREEN}Transcription saved to: {output_path}{Style.RESET_ALL}")

    def run(self) -> None:
        """メインエントリーポイント"""
        self.print_banner()

        # ストリーム開始と入力監視
        print(
            f"{Fore.GREEN}🎙️  Listening... (Ctrl+C to stop, Ctrl+D for fast exit){Style.RESET_ALL}\n"
        )

        try:
            with self.audio_stream as stream:
                # ファイル/マイク共通：終了シグナルを待つ
                # ファイル入力時は処理完了も終了条件に含める
                # AudioStreamが終了 かつ Transcriberの処理も完了した時点で終了
                stop_condition = (
                    (
                        lambda: not stream.is_alive()
                        and not self.transcriber.is_transcribing
                    )
                    if self.file_path
                    else None
                )
                completed = InputHandler.wait_for_exit_signal(stop_condition)
                if completed:
                    # ファイル処理完了
                    print(f"\n{Fore.GREEN}File processing completed.{Style.RESET_ALL}")
                    self._shutdown(graceful=True)
                    return
        except KeyboardInterrupt:
            # Ctrl-C: 正常終了（残り処理を待って保存）
            print(f"\n{Fore.GREEN}Goodbye!{Style.RESET_ALL}")
            self._shutdown(graceful=True)
            return
        except EOFError:
            # Ctrl-D: 高速終了（保存なし）
            print(f"\n{Fore.YELLOW}Fast exit (Ctrl-D){Style.RESET_ALL}")
            self._shutdown(graceful=False)
            return
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
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable real-time summary generation",
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

    # サマリ生成の有効/無効を判定
    enable_summary = not args.no_summary

    # APIキーの取得
    api_key = os.getenv("ANTHROPIC_API_KEY")

    # サマリ生成が有効でAPIキーがない場合は警告を表示して無効化
    if enable_summary and not api_key:
        colorama_init(autoreset=True)
        print(
            f"{Fore.YELLOW}Warning: ANTHROPIC_API_KEY is not set. Summary generation disabled.{Style.RESET_ALL}"
        )
        enable_summary = False

    app = StreamScribeApp(
        api_key=api_key or "",
        device_id=args.device,
        file_path=args.file,
        enable_summary=enable_summary,
    )
    app.run()


if __name__ == "__main__":
    main()
