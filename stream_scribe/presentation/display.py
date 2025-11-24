#!/usr/bin/env python3
"""
Stream Scribe - Display Module
表示フォーマッティングとステータス表示を提供するモジュール
"""

import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime

import wcwidth  # type: ignore[import-untyped]
from colorama import Fore, Style  # type: ignore[import-untyped]

from stream_scribe.domain.constants import (
    CHUNK_MS,
    MAX_ERROR_DETAIL_LENGTH,
    MAX_TRACEBACK_LENGTH,
    VAD_START_THRESHOLD,
)
from stream_scribe.domain.models import TranscriptionSegment


class DisplayFormatter:
    """
    表示フォーマッティング専用クラス

    Features:
    - ANSIエスケープコードを考慮した表示幅計算
    - 全角・半角を考慮したテキスト切り詰め
    """

    # ANSIエスケープコード削除用パターン（コンパイル済み）
    _ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")

    def get_display_width(self, text: str) -> int:
        """ANSIエスケープコードを除いた実際の表示幅を取得"""
        plain_text = self._ANSI_ESCAPE_PATTERN.sub("", text)
        return int(wcwidth.wcswidth(plain_text))

    def truncate_text(self, text: str, max_width: int) -> str:
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


class StatusDisplay:
    """
    リアルタイム表示管理クラス

    Features:
    - VAD/録音/処理ステータスのリアルタイム更新
    - 文字起こし結果の表示
    - 構造化された会話記録の表示
    """

    def __init__(self, formatter: DisplayFormatter):
        self.lock = threading.Lock()  # スレッド間の同期用ロック
        self.session_start_time = time.time()  # セッション開始時刻
        self.formatter = formatter  # フォーマッター

    def update_status(
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

        terminal_width = os.get_terminal_size().columns

        # VADセクション構築
        bar_width = 20
        probability_bar_length = int(probability * bar_width)
        probability_bar = "|" * probability_bar_length + "." * (
            bar_width - probability_bar_length
        )
        probability_color = (
            Fore.GREEN if probability >= VAD_START_THRESHOLD else Fore.CYAN
        )
        vad_section = f"{probability_color}VAD:[{probability_bar}] {probability:.2f}{Style.RESET_ALL}"

        # ステータステキスト構築
        status_parts = []
        if is_recording:
            status_parts.append(
                f"{Fore.RED}● REC [{recording_elapsed:.1f}s]{Style.RESET_ALL}"
            )
        elif speech_chunks > 0:
            speech_duration = speech_chunks * CHUNK_MS / 1000.0
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
        left_width = self.formatter.get_display_width(full_left)
        right_width = self.formatter.get_display_width(right_section)
        overflow = left_width + right_width - terminal_width

        if overflow > 0:
            status_width = self.formatter.get_display_width(status_text)
            target_status_width = status_width - overflow - 3
            if target_status_width > 0:
                status_text = (
                    self.formatter.truncate_text(status_text, target_status_width)
                    + "..."
                )
            else:
                status_text = "..."
            left_section = f"{vad_section} | {status_text}"
        else:
            left_section = full_left

        # 最終ステータスライン構築
        left_width = self.formatter.get_display_width(left_section)
        right_width = self.formatter.get_display_width(right_section)
        padding_width = max(0, terminal_width - left_width - right_width)
        padding = " " * padding_width

        status_line = f"\r\033[K{left_section}{padding}{right_section}"
        sys.stdout.write(status_line)
        sys.stdout.flush()
        self.lock.release()

    def show_segment(self, segment: TranscriptionSegment) -> None:
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

    def show_summary(self, summary_text: str) -> None:
        """リアルタイム議事録をダッシュボード形式で表示"""
        with self.lock:
            # 現在のステータスバーをクリア
            sys.stdout.write("\r\033[K")

            # 要約ヘッダーと内容を表示
            print(f"\n{Fore.CYAN}{'─' * 50}{Style.RESET_ALL}")
            print(summary_text)
            print(f"{Fore.CYAN}{'─' * 50}{Style.RESET_ALL}\n")

            sys.stdout.flush()

    def show_error(
        self,
        error_time: datetime,
        error_message: str,
        exception: Exception | None = None,
    ) -> None:
        """エラーメッセージを表示"""
        time_str = error_time.strftime("%H:%M:%S")

        with self.lock:
            # 現在のステータスバーをクリア
            sys.stdout.write("\r\033[K")

            # エラーメッセージを表示
            sys.stdout.write(
                f"{Fore.RED}[{time_str}] ❌ {error_message}{Style.RESET_ALL}\n"
            )

            # 例外の詳細を表示（オプション）
            if exception:
                error_detail = str(exception)[:MAX_ERROR_DETAIL_LENGTH]
                sys.stdout.write(
                    f"{Fore.YELLOW}Details: {error_detail}{Style.RESET_ALL}\n"
                )

                # トレースバックを表示
                traceback_str = traceback.format_exc()[:MAX_TRACEBACK_LENGTH]
                if traceback_str and traceback_str != "NoneType: None\n":
                    sys.stdout.write(f"{Fore.YELLOW}{traceback_str}{Style.RESET_ALL}\n")

            sys.stdout.flush()

    def clear(self) -> None:
        """表示をクリア"""
        with self.lock:
            sys.stdout.write("\r\033[K\n")
            sys.stdout.flush()
