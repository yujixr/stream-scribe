#!/usr/bin/env python3
"""
Stream Scribe - Summarizer Module
インフラ層：Claude APIによるリアルタイム会話構造化
"""

import queue
import threading
from datetime import datetime
from typing import Callable

from anthropic import Anthropic
from anthropic.types import TextBlock

from stream_scribe.domain.constants import (
    SUMMARY_MAX_TOKENS,
    SUMMARY_MODEL,
    SUMMARY_QUEUE_GET_TIMEOUT_SEC,
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
        on_summary_update: Callable[[str], None],
        on_summary_display: Callable[[str], None],
        on_error: Callable[[datetime, str, Exception | None], None],
        api_key: str,
        model: str = SUMMARY_MODEL,
    ) -> None:
        super().__init__(daemon=True)
        self.queue: queue.Queue[str] = queue.Queue()
        self.running = True
        self.model = model
        self.api_key = api_key
        self.on_summary_update = on_summary_update  # サマリ更新時のコールバック
        self.on_summary_display = on_summary_display  # サマリ表示コールバック
        self.on_error = on_error  # エラーコールバック

        # 議事録の状態保持
        self.current_summary = ""
        self.is_summarizing = False  # LLM要約中フラグ

        # バッファリング設定
        self.text_buffer: list[str] = []
        self.buffer_char_count = 0
        self.trigger_threshold = SUMMARY_TRIGGER_THRESHOLD

    def add_segment(self, text: str) -> None:
        """新しいテキストセグメントを追加"""
        if self.running and text:
            self.queue.put(text)

    def _process_buffer(self) -> str | None:
        """
        バッファテキストを処理してClaude APIで要約を生成

        Returns:
            str | None: 生成された要約 or None
        """
        if not self.text_buffer:
            return None

        # バッファを結合
        chunk_text = " ".join(self.text_buffer)
        self.text_buffer = []

        self.is_summarizing = True
        try:
            summary = self._call_claude(chunk_text)
            return summary
        finally:
            self.is_summarizing = False

    def _call_claude(self, new_text_chunk: str) -> str | None:
        """
        Claude APIを呼び出して議事録を更新

        Args:
            new_text_chunk: 新しい発言テキスト

        Returns:
            str | None: 生成された要約 or None
        """
        try:
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
        except Exception as e:
            # エラーをコールバック経由で通知
            self.on_error(datetime.now(), "Summary generation failed", e)
            return None

    def run(self) -> None:
        while self.running:
            try:
                # 終了フラグ確認のためタイムアウト付きget
                text = self.queue.get(timeout=SUMMARY_QUEUE_GET_TIMEOUT_SEC)

                self.text_buffer.append(text)
                self.buffer_char_count += len(text)

                # 閾値を超えたら要約を実行
                if self.buffer_char_count >= self.trigger_threshold:
                    summary = self._process_buffer()
                    self.buffer_char_count = 0

                    if summary:
                        # 表示コールバック
                        self.on_summary_display(summary)
                        # セッション保存コールバック
                        self.on_summary_update(summary)

            except queue.Empty:
                continue
            except Exception:
                self.is_summarizing = False
                continue

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

        # キューに残っているテキストをバッファに移動
        while not self.queue.empty():
            try:
                text = self.queue.get_nowait()
                self.text_buffer.append(text)
            except queue.Empty:
                break

        # バッファに残っているテキストがあれば処理
        summary = self._process_buffer()
        if summary:
            return summary

        # バッファが空だった場合は現在のサマリを返す
        return self.current_summary if self.current_summary else None
