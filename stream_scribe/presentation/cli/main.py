#!/usr/bin/env python3
"""
Stream Scribe - CLI Main Entry Point
CLIアプリケーションのエントリーポイント
"""

import argparse

from colorama import Fore, Style  # type: ignore[import-untyped]
from colorama import init as colorama_init

from stream_scribe.infrastructure.audio import MicrophoneAudioSource

from .controller import CLIController


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

    # colorama初期化
    colorama_init(autoreset=True)

    # デバイス一覧表示モード
    if args.list_devices:
        print_audio_devices()
        return

    # CLIController起動
    controller = CLIController(
        device_id=args.device,
        file_path=args.file,
    )
    controller.run()


if __name__ == "__main__":
    main()
