#!/usr/bin/env python3
"""
Stream Scribe - Summarizer Module
インフラ層：LLMによるリアルタイム会話構造化
"""

import threading
import time

from stream_scribe.domain import (
    MessageLevel,
    MessagePostedEvent,
    SummaryGeneratedEvent,
    SummarySettings,
    TranscriptionSegment,
    TranscriptionSession,
    message_posted,
    summary_generated,
)

from .llm_client import LLMClient
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
    - LLMによる会話の構造化とツリー化
    - トピック抽出と時系列ログの自動生成
    - イベント駆動アーキテクチャ（Pub/Sub）
    """

    def __init__(
        self,
        llm_client: LLMClient,
        settings: SummarySettings,
        prompt_strategy: PromptStrategy | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.running = True

        # LLMクライアント（DI）
        self.llm_client = llm_client
        self.settings = settings

        # プロンプト戦略（DI: デフォルトはリアルタイム要約）
        self.prompt_strategy = prompt_strategy or RealtimePromptStrategy()

        # 議事録の状態保持
        self.current_summary = ""
        self.is_summarizing = False  # LLM要約中フラグ

        # セグメント履歴の保持（要約済み + 未要約）
        self.summarized_segments: list[TranscriptionSegment] = []
        self.pending_segments: list[TranscriptionSegment] = []
        self._segment_lock = threading.Lock()  # セグメント操作の排他制御

        # 処理トリガー用イベント
        self._trigger_event = threading.Event()

        # 最後のセグメント追加時刻（無音検出用）
        self.last_segment_time: float | None = None

    def add_segment(self, segment: TranscriptionSegment) -> None:
        """
        新しいセグメントを追加
        """
        if self.running:
            with self._segment_lock:
                self.pending_segments.append(segment)
                self.last_segment_time = time.monotonic()

            # 処理トリガーイベントをセット
            self._trigger_event.set()

    @property
    def buffer_char_count(self) -> int:
        """未処理セグメントの文字数を取得"""
        with self._segment_lock:
            return sum(len(seg.text) for seg in self.pending_segments)

    def _should_summarize(self) -> bool:
        """
        要約を実行すべきか判断

        Returns:
            bool: 要約を実行すべき場合True
        """
        char_count = self.buffer_char_count

        # 閾値超過チェック
        if char_count >= self.settings.trigger_threshold:
            return True

        # 無音タイムアウトチェック
        if self.last_segment_time is not None:
            elapsed = time.monotonic() - self.last_segment_time
            return elapsed >= self.settings.silence_timeout_sec

        return False

    def _process_buffer(self) -> str | None:
        """
        未処理セグメントを処理してLLM APIで要約を生成

        Returns:
            str | None: 生成された要約 or None
        """
        # 未処理セグメントを取得してクリア
        with self._segment_lock:
            if not self.pending_segments:
                return None

            new_segments = self.pending_segments.copy()
            self.pending_segments.clear()

        # プロンプト戦略を使用してプロンプトを構築
        # 要約済みセグメントから直近N件を取得
        n = self.settings.recent_segments_for_context
        summarized_recent = (
            self.summarized_segments[-n:] if self.summarized_segments else None
        )

        system_prompt = self.prompt_strategy.system_prompt
        user_content = self.prompt_strategy.build_user_prompt(
            previous_summary=self.current_summary if self.current_summary else None,
            processed_segments=summarized_recent,
            new_segments=new_segments,
        )

        self.is_summarizing = True
        try:
            # LLMクライアント経由で生成
            updated_summary = self.llm_client(
                system_prompt=system_prompt,
                user_prompt=user_content,
                temperature=self.settings.realtime_temperature,
                top_p=self.settings.realtime_top_p,
            )

            # 内部状態を更新
            if updated_summary:
                self.current_summary = updated_summary
                # 要約済みセグメントリストに追加
                self.summarized_segments.extend(new_segments)
                # 直近N件のみ保持（メモリ節約）
                n = self.settings.recent_segments_for_context
                self.summarized_segments = self.summarized_segments[-n:]

            return updated_summary
        except Exception as e:
            # LLM API呼び出しエラーをイベント経由で通知
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
            self._trigger_event.wait(timeout=self.settings.queue_get_timeout_sec)
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

        # 未処理セグメントをクリア（現在の処理を破棄）
        with self._segment_lock:
            self.pending_segments.clear()

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
            final_summary = self.llm_client(
                system_prompt=system_prompt,
                user_prompt=user_content,
                temperature=self.settings.final_temperature,
                top_p=self.settings.final_top_p,
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
