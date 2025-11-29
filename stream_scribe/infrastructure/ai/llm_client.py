#!/usr/bin/env python3
"""
Stream Scribe - LLM Clients Module
LLMクライアントの抽象化とアダプタパターンを提供するモジュール
"""

from abc import ABC, abstractmethod

from anthropic import Anthropic
from anthropic.types import TextBlock

from stream_scribe.domain import SummarySettings


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
        max_tokens: int | None = None,
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

    @abstractmethod
    def get_backend_info(self) -> str:
        """
        使用しているLLMバックエンドの情報を返す

        Returns:
            str: バックエンド情報（例: "Claude (claude-3-5-haiku-20241022)"）
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

    def __init__(self, api_key: str, settings: SummarySettings) -> None:
        """
        Args:
            api_key: Anthropic APIキー
            settings: サマリー設定（モデル名、最大トークン数など）
        """
        self.api_key = api_key
        self.settings = settings
        self.client = Anthropic(api_key=api_key)

    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
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
            model=self.settings.model,
            max_tokens=max_tokens or self.settings.max_tokens,
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

    def get_backend_info(self) -> str:
        """
        使用しているLLMバックエンドの情報を返す

        Returns:
            str: バックエンド情報（例: "Claude (claude-3-5-haiku-20241022)"）
        """
        return f"Claude ({self.settings.model})"
