#!/usr/bin/env python3
"""
Stream Scribe - LLM Clients Module
LLMクライアントの抽象化とアダプタパターンを提供するモジュール
"""

from abc import ABC, abstractmethod

from anthropic import Anthropic
from anthropic.types import TextBlock

from stream_scribe.domain.constants import CLAUDE_SUMMARY_MODEL, SUMMARY_MAX_TOKENS


class LLMClient(ABC):
    """
    LLMクライアントの抽象基底クラス

    テキスト生成APIへのインターフェースを統一し、
    異なるLLMプロバイダ（Claude、OpenAIなど）を
    同じインターフェースで扱えるようにする。
    """

    @abstractmethod
    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = SUMMARY_MAX_TOKENS,
    ) -> str | None:
        """
        LLM APIでテキストを生成

        Args:
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            temperature: 生成の確率性（0.0=決定論的、1.0=最大ランダム性）
            max_tokens: 最大トークン数

        Returns:
            str | None: 生成されたテキスト or None

        Raises:
            Exception: API呼び出しエラー
        """
        pass


class ClaudeClient(LLMClient):
    """
    Claude APIクライアント

    責務:
    - Claude APIへのリクエスト送信
    - レスポンスのパース
    - API通信の抽象化
    """

    def __init__(self, api_key: str, model: str = CLAUDE_SUMMARY_MODEL) -> None:
        """
        Args:
            api_key: Anthropic APIキー
            model: 使用するClaudeモデル名（デフォルト: CLAUDE_SUMMARY_MODEL）
        """
        self.api_key = api_key
        self.model = model
        self.client = Anthropic(api_key=api_key)

    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = SUMMARY_MAX_TOKENS,
    ) -> str | None:
        """
        Claude APIでテキストを生成

        Args:
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            temperature: 生成の確率性（0.0=決定論的、1.0=最大ランダム性）
            max_tokens: 最大トークン数

        Returns:
            str | None: 生成されたテキスト or None

        Raises:
            Exception: API呼び出しエラー
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # TextBlockの場合のみtextを取得
        if message.content and len(message.content) > 0:
            first_block = message.content[0]
            if isinstance(first_block, TextBlock):
                return first_block.text.strip()
        return None
