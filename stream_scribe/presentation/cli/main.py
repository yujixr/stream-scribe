#!/usr/bin/env python3
"""
Stream Scribe - CLI Main Entry Point
CLIアプリケーションのエントリーポイント
"""

import argparse
import os
import sys
import traceback

from colorama import Fore, Style  # type: ignore[import-untyped]
from colorama import init as colorama_init

from stream_scribe.domain import MessageLevel, MessagePostedEvent, message_posted
from stream_scribe.infrastructure.audio import (
    AudioSource,
    FileAudioSource,
    MicrophoneAudioSource,
)
from stream_scribe.presentation.app import StreamScribeApp

from .input_handler import InputHandler
from .view import CLIView


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

    # colorama初期化
    colorama_init(autoreset=True)

    # デバイス一覧表示モード
    if args.list_devices:
        print_audio_devices()
        return

    # CLIView作成（最初に作成してSignal受信準備）
    view = CLIView()

    # APIキーの取得（--no-summaryオプションの場合はNoneに設定）
    api_key = None if args.no_summary else os.getenv("ANTHROPIC_API_KEY")

    # APIキーがない場合は警告を表示
    if not args.no_summary and not api_key:
        message_posted.send(
            None,
            event=MessagePostedEvent(
                message="Warning: ANTHROPIC_API_KEY is not set. Summary generation disabled.",
                level=MessageLevel.WARNING,
            ),
        )

    # AudioSource作成（ファイル入力またはマイク入力）
    audio_source: AudioSource
    if args.file:
        audio_source = FileAudioSource(file_path=args.file)
    else:
        audio_source = MicrophoneAudioSource(device_id=args.device)

    # StreamScribeApp作成（初期化時にmessage_postedシグナルを発行）
    app = StreamScribeApp(
        api_key=api_key,
        audio_source=audio_source,
    )

    # UI更新開始（コンポーネント参照を渡してStatusUpdateManagerを起動）
    view.start(
        audio_stream=app.audio_stream,
        transcriber=app.transcriber,
        summarizer=app.summarizer,
    )

    # ストリーム開始と入力監視
    message_posted.send(
        None,
        event=MessagePostedEvent(
            message="🎙️  Listening... (Ctrl+C to stop, Ctrl+D for fast exit)\n",
            level=MessageLevel.SUCCESS,
        ),
    )

    is_file_mode = not audio_source.is_realtime

    try:
        with app.audio_stream as stream:
            # ファイル/マイク共通：終了シグナルを待つ
            # ファイル入力時は処理完了も終了条件に含める
            # AudioStreamが終了 かつ Transcriberの処理も完了した時点で終了
            stop_condition = (
                (lambda: not stream.is_alive() and not app.transcriber.is_transcribing)
                if is_file_mode
                else None
            )
            completed = InputHandler.wait_for_exit_signal(stop_condition)
            if completed:
                # ファイル処理完了
                message_posted.send(
                    None,
                    event=MessagePostedEvent(
                        message="\nFile processing completed.",
                        level=MessageLevel.SUCCESS,
                    ),
                )
                view.stop()
                view.clear()
                app._shutdown(graceful=True)
                return
    except KeyboardInterrupt:
        # Ctrl-C: 正常終了（残り処理を待って保存）
        message_posted.send(
            None,
            event=MessagePostedEvent(message="\nGoodbye!", level=MessageLevel.SUCCESS),
        )
        view.stop()
        view.clear()
        app._shutdown(graceful=True)
        return
    except EOFError:
        # Ctrl-D: 高速終了（保存なし）
        message_posted.send(
            None,
            event=MessagePostedEvent(
                message="\nFast exit (Ctrl-D)", level=MessageLevel.WARNING
            ),
        )
        view.stop()
        view.clear()
        app._shutdown(graceful=False)
        return
    except Exception as e:
        # エラー時は即座に終了
        message_posted.send(
            None,
            event=MessagePostedEvent(message=f"\nError: {e}", level=MessageLevel.ERROR),
        )
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
