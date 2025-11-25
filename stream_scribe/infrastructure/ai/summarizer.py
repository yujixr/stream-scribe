#!/usr/bin/env python3
"""
Stream Scribe - Summarizer Module
インフラ層：Claude APIによるリアルタイム会話構造化
"""

import queue
import threading
import time
from datetime import datetime
from typing import Callable

from anthropic import Anthropic
from anthropic.types import TextBlock

from stream_scribe.domain.constants import (
    SUMMARY_MAX_TOKENS,
    SUMMARY_MODEL,
    SUMMARY_QUEUE_GET_TIMEOUT_SEC,
    SUMMARY_SILENCE_TIMEOUT_SEC,
    SUMMARY_TRIGGER_THRESHOLD,
)
from stream_scribe.infrastructure.ai.prompts import SummaryPromptBuilder


class RealtimeSummarizer(threading.Thread):
    """
    リアルタイム会話構造化スレッド

    責務:
    - テキストのバッファリング
    - Claude APIによる会話の構造化とツリー化
    - トピック抽出と時系列ログの自動生成

    Note: display依存を削除し、エラーはコールバック経由で通知
    """

    def __init__(
        self,
        on_summary: Callable[[str], None],
        on_error: Callable[[datetime, str, Exception | None], None],
        api_key: str,
        model: str = SUMMARY_MODEL,
    ) -> None:
        super().__init__(daemon=True)
        self.queue: queue.Queue[str] = queue.Queue()
        self.running = True
        self.model = model
        self.api_key = api_key
        self.on_summary = on_summary  # サマリ生成時のコールバック
        self.on_error = on_error  # エラーコールバック

        # 議事録の状態保持
        self.current_summary = ""
        self.is_summarizing = False  # LLM要約中フラグ

        # バッファリング設定
        self.text_buffer: list[str] = []
        self.trigger_threshold = SUMMARY_TRIGGER_THRESHOLD
        self.silence_timeout = SUMMARY_SILENCE_TIMEOUT_SEC
        self._buffer_lock = threading.Lock()  # バッファ操作の排他制御

        # 最後のセグメント追加時刻（無音検出用）
        self.last_segment_time: float | None = None

    def add_segment(self, text: str) -> None:
        """
        新しいテキストセグメントを追加

        Note: バッファ更新は即座に行い、処理トリガーをキューに送信
        これにより呼び出し元とバッファ状態が同期する
        """
        if self.running and text:
            with self._buffer_lock:
                self.text_buffer.append(text)
                self.last_segment_time = time.monotonic()

            # 処理トリガー用のシグナルをキューに送信
            self.queue.put("")

    @property
    def buffer_char_count(self) -> int:
        """バッファ内の文字数を取得"""
        with self._buffer_lock:
            return sum(len(text) for text in self.text_buffer)

    def _process_buffer(self) -> str | None:
        """
        バッファテキストを処理してClaude APIで要約を生成

        Returns:
            str | None: 生成された要約 or None
        """
        # バッファを取得してクリア
        with self._buffer_lock:
            if not self.text_buffer:
                return None

            chunk_text = " ".join(self.text_buffer)
            self.text_buffer = []

        self.is_summarizing = True
        try:
            summary = self._call_claude(chunk_text)
        except Exception as e:
            # Claude API呼び出しエラーをコールバック経由で通知
            self.on_error(datetime.now(), "Summary generation failed", e)
            summary = None
        finally:
            self.is_summarizing = False

        return summary

    def _call_claude(self, new_text_chunk: str) -> str | None:
        """
        Claude APIを呼び出して議事録を更新

        Args:
            new_text_chunk: 新しい発言テキスト

        Returns:
            str | None: 生成された要約 or None

        Raises:
            Exception: API呼び出しエラー（呼び出し元で処理）
        """
        client = Anthropic(api_key=self.api_key)

        # プロンプトビルダーを使用してプロンプトを構築
        system_prompt = SummaryPromptBuilder.SYSTEM_PROMPT
        user_content = SummaryPromptBuilder.build_user_prompt(
            current_summary=self.current_summary,
            new_text_chunk=new_text_chunk,
        )

        # Claude API呼び出し
        message = client.messages.create(
            model=self.model,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        # TextBlockの場合のみtextを取得
        if message.content and len(message.content) > 0:
            first_block = message.content[0]
            if isinstance(first_block, TextBlock):
                updated_summary = first_block.text.strip()
                # 内部状態を更新
                self.current_summary = updated_summary
                return updated_summary
        return None

    def run(self) -> None:
        while self.running:
            try:
                # 処理トリガーを待つ（終了フラグ確認のためタイムアウト付き）
                self.queue.get(timeout=SUMMARY_QUEUE_GET_TIMEOUT_SEC)
            except queue.Empty:
                pass  # タイムアウトチェックのため続行

            # バッファの状態をチェック（add_segmentで既に更新済み）
            char_count = self.buffer_char_count
            should_process_by_threshold = char_count >= self.trigger_threshold

            # 無音タイムアウトチェック（バッファにテキストがある場合のみ）
            should_process_by_timeout = False
            if char_count > 0 and self.last_segment_time is not None:
                elapsed = time.monotonic() - self.last_segment_time
                should_process_by_timeout = elapsed >= self.silence_timeout

            # 閾値超過または無音タイムアウトで要約を実行
            if should_process_by_threshold or should_process_by_timeout:
                summary = self._process_buffer()

                if summary:
                    self.on_summary(summary)

    def stop(self, wait_for_final: bool = False) -> str | None:
        """
        スレッド停止

        Args:
            wait_for_final: Trueの場合、バッファを処理して最終サマリを生成する

        Returns:
            str | None: wait_for_final=True の場合は最終サマリ、それ以外は None
        """
        self.running = False

        if not wait_for_final:
            return None

        # キューをクリア
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        # バッファに残っているテキストがあれば処理
        summary = self._process_buffer()
        if summary:
            return summary

        # バッファが空だった場合は現在のサマリを返す
        return self.current_summary if self.current_summary else None
