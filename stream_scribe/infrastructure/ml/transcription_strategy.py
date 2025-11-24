#!/usr/bin/env python3
"""
Stream Scribe - Transcription Retry Strategy Module
文字起こしのリトライ戦略を管理するモジュール
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from stream_scribe.domain.constants import MAX_TRANSCRIPTION_RETRIES, WHISPER_PARAMS


class TranscriptionAction(Enum):
    """文字起こし結果に対するアクション"""

    ACCEPT = auto()  # 成功：結果を受け入れる
    RETRY = auto()  # リトライ：パラメータを変えて再試行
    DISCARD = auto()  # 破棄：最大リトライ到達で諦める


@dataclass(frozen=True)
class StrategyResult:
    """
    戦略評価の結果

    Attributes:
        action: 実行すべきアクション
        next_params: リトライ時の次のパラメータ（RETRY時のみ）
        reason: アクションの理由（RETRY/DISCARD時のみ）
    """

    action: TranscriptionAction
    next_params: dict[str, Any] | None = None
    reason: str | None = None


class TranscriptionRetryStrategy:
    """
    文字起こしリトライ戦略

    段階的にWhisperパラメータを厳格化しながらリトライを管理する。
    戦略: 「標準」→「ループ対策」→「バイアス排除」→「厳格化」→「最終フィルタ」
    """

    def __init__(self) -> None:
        self.current_attempt = 0

    def reset(self) -> None:
        """状態をリセット（新しい音声処理の開始時に呼ぶ）"""
        self.current_attempt = 0

    def get_current_params(self) -> dict[str, Any]:
        """現在の試行に対応するWhisperパラメータを返す"""
        return WHISPER_PARAMS[self.current_attempt]

    def get_attempt_info(self) -> tuple[int, int]:
        """
        現在の試行情報を返す

        Returns:
            tuple[int, int]: (現在の試行番号 (1-based), 最大試行回数)
        """
        return (self.current_attempt + 1, MAX_TRANSCRIPTION_RETRIES)

    def evaluate_result(self, text: str, filter_reason: str | None) -> StrategyResult:
        """
        文字起こし結果を評価し、次のアクションを決定する

        Args:
            text: 文字起こし結果のテキスト
            filter_reason: HallucinationFilterの検出理由（Noneなら問題なし）

        Returns:
            StrategyResult: 評価結果とアクション
        """
        # 成功ケース：テキストがあり、フィルタに引っかからない
        if text and not filter_reason:
            return StrategyResult(action=TranscriptionAction.ACCEPT)

        # 空テキストで理由もない場合：無音として破棄（リトライ不要）
        if not text and not filter_reason:
            return StrategyResult(
                action=TranscriptionAction.DISCARD,
                reason="Empty transcription (likely silence)",
            )

        # リトライ可能な場合
        if self.current_attempt < MAX_TRANSCRIPTION_RETRIES - 1:
            self.current_attempt += 1
            return StrategyResult(
                action=TranscriptionAction.RETRY,
                next_params=self.get_current_params(),
                reason=filter_reason,
            )

        # 最大リトライ到達：破棄
        return StrategyResult(
            action=TranscriptionAction.DISCARD,
            reason=f"Max retries reached. Last error: {filter_reason}",
        )
