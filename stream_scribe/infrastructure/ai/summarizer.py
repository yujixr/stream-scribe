#!/usr/bin/env python3
"""
Stream Scribe - Summarizer Module
インフラ層：Claude APIによるリアルタイム会話構造化
"""

import threading
import time

from stream_scribe.domain.constants import (
    SUMMARY_MODEL,
    SUMMARY_QUEUE_GET_TIMEOUT_SEC,
    SUMMARY_SILENCE_TIMEOUT_SEC,
    SUMMARY_TRIGGER_THRESHOLD,
)
from stream_scribe.domain import (
    MessageLevel,
    MessagePostedEvent,
    SummaryGeneratedEvent,
    TranscriptionSession,
    message_posted,
    summary_generated,
)

from .claude_client import ClaudeClient
from .prompts import (
    FinalSummaryPromptStrategy,
    PromptStrategy,
    RealtimePromptStrategy,
)


class RealtimeSummarizer(threading.Thread):
    """
    リアルタイム会話構造化スレッド

    責務:
    - テキストのバッファリング
    - Claude APIによる会話の構造化とツリー化
    - トピック抽出と時系列ログの自動生成
    - イベント駆動アーキテクチャ（Pub/Sub）
    """

    def __init__(
        self,
        api_key: str,
        model: str = SUMMARY_MODEL,
        prompt_strategy: PromptStrategy | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.running = True

        # Claude APIクライアント
        self.claude_client = ClaudeClient(api_key=api_key, model=model)

        # プロンプト戦略（DI: デフォルトはリアルタイム要約）
        self.prompt_strategy = prompt_strategy or RealtimePromptStrategy()

        # 議事録の状態保持
        self.current_summary = ""
        self.is_summarizing = False  # LLM要約中フラグ

        # バッファリング設定
        self.text_buffer: list[str] = []
        self.trigger_threshold = SUMMARY_TRIGGER_THRESHOLD
        self.silence_timeout = SUMMARY_SILENCE_TIMEOUT_SEC
        self._buffer_lock = threading.Lock()  # バッファ操作の排他制御

        # 処理トリガー用イベント
        self._trigger_event = threading.Event()

        # 最後のセグメント追加時刻（無音検出用）
        self.last_segment_time: float | None = None

    def add_segment(self, text: str) -> None:
        """
        新しいテキストセグメントを追加
        """
        if self.running and text:
            with self._buffer_lock:
                self.text_buffer.append(text)
                self.last_segment_time = time.monotonic()

            # 処理トリガーイベントをセット
            self._trigger_event.set()

    @property
    def buffer_char_count(self) -> int:
        """バッファ内の文字数を取得"""
        with self._buffer_lock:
            return sum(len(text) for text in self.text_buffer)

    def _should_summarize(self) -> bool:
        """
        要約を実行すべきか判断

        Returns:
            bool: 要約を実行すべき場合True
        """
        char_count = self.buffer_char_count

        # 閾値超過チェック
        if char_count >= self.trigger_threshold:
            return True

        # 無音タイムアウトチェック
        if self.last_segment_time is not None:
            elapsed = time.monotonic() - self.last_segment_time
            return elapsed >= self.silence_timeout

        return False

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
            self.text_buffer.clear()

        # プロンプト戦略を使用してプロンプトを構築
        system_prompt = self.prompt_strategy.system_prompt
        user_content = self.prompt_strategy.build_user_prompt(
            current_summary=self.current_summary,
            new_text_chunk=chunk_text,
        )

        self.is_summarizing = True
        try:
            # Claude APIクライアント経由で生成
            updated_summary = self.claude_client(
                system_prompt=system_prompt,
                user_prompt=user_content,
                temperature=0.0,
            )

            # 内部状態を更新
            if updated_summary:
                self.current_summary = updated_summary

            return updated_summary
        except Exception as e:
            # Claude API呼び出しエラーをイベント経由で通知
            event = MessagePostedEvent(
                message=f"Summary generation failed: {e}",
                level=MessageLevel.ERROR,
            )
            message_posted.send(self, event=event)
            return None
        finally:
            self.is_summarizing = False

    def run(self) -> None:
        while self.running:
            # 処理トリガーを待つ（終了フラグ確認のためタイムアウト付き）
            self._trigger_event.wait(timeout=SUMMARY_QUEUE_GET_TIMEOUT_SEC)
            self._trigger_event.clear()

            # 要約実行判断
            if self._should_summarize():
                summary = self._process_buffer()

                if summary:
                    event = SummaryGeneratedEvent(summary=summary)
                    summary_generated.send(self, event=event)

    def stop(self, session: TranscriptionSession | None = None) -> None:
        """
        スレッド停止と終了時サマリの生成

        Args:
            session: 終了時サマリを生成する場合はTranscriptionSessionを渡す。
                    Noneの場合は即座に終了する。
        """
        self.running = False

        # バッファをクリア（現在の処理を破棄）
        with self._buffer_lock:
            self.text_buffer.clear()

        # イベントをセットして待機中のスレッドを起こす
        self._trigger_event.set()

        # セッションが渡されなければ即座に終了
        if session is None or not session.segments:
            return

        # 終了時サマリ用の戦略を使用
        final_strategy = FinalSummaryPromptStrategy()
        system_prompt = final_strategy.system_prompt
        user_content = final_strategy.build_user_prompt(session=session)

        # 終了時サマリを生成してシグナル送信
        try:
            final_summary = self.claude_client(
                system_prompt=system_prompt,
                user_prompt=user_content,
                temperature=0.0,
            )
            # 終了時サマリイベントを送信
            if final_summary:
                summary_event = SummaryGeneratedEvent(
                    summary=final_summary, is_final=True
                )
                summary_generated.send(self, event=summary_event)
        except Exception as e:
            error_event = MessagePostedEvent(
                message=f"Final summary generation failed: {e}",
                level=MessageLevel.ERROR,
            )
            message_posted.send(self, event=error_event)
