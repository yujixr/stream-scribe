#!/usr/bin/env python3
"""
Stream Scribe - LLM Clients Module
LLMクライアントの抽象化とアダプタパターンを提供するモジュール
"""

import re
from abc import ABC, abstractmethod

from anthropic import Anthropic
from anthropic.types import TextBlock
from openai import OpenAI

from stream_scribe.domain import LLMBackend, SummarySettings


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
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> str | None:
        """
        LLM APIでテキストを生成

        Args:
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            temperature: 生成の確率性（0.0=決定論的、1.0=最大ランダム性）
            top_p: nucleus sampling（累積確率がtop_pになるまでのトークンから選択、0.0-1.0）
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

    def __init__(self, settings: SummarySettings) -> None:
        """
        Args:
            settings: サマリー設定（APIキー、モデル名、最大トークン数など）
        """

        # 設定検証済みのため、anthropic_api_keyは必ず存在する
        assert settings.anthropic_api_key is not None
        self.settings = settings
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> str | None:
        """
        Claude APIでテキストを生成

        Args:
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            temperature: 生成の確率性（0.0=決定論的、1.0=最大ランダム性）
            top_p: nucleus sampling（累積確率がtop_pになるまでのトークンから選択、0.0-1.0）
            max_tokens: 最大トークン数

        Returns:
            str | None: 生成されたテキスト or None

        Raises:
            Exception: API呼び出しエラー
        """
        message = self.client.messages.create(
            model=self.settings.claude_model,
            max_tokens=max_tokens or self.settings.max_tokens,
            temperature=temperature,
            top_p=top_p,
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
        return f"Claude ({self.settings.claude_model})"


class VLLMClient(LLMClient):
    """
    vLLM OpenAI互換APIクライアント

    責務:
    - vLLMサーバへのリクエスト送信（OpenAI互換API）
    - レスポンスのパース
    - API通信の抽象化
    """

    # <think>...</think> タグ削除用の事前コンパイル済み正規表現
    _THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
    # ```markdown ... ``` ブロック抽出用の事前コンパイル済み正規表現
    _MARKDOWN_BLOCK_PATTERN = re.compile(r"```markdown\s*\n(.*?)\n```", re.DOTALL)

    def __init__(self, settings: SummarySettings) -> None:
        """
        Args:
            settings: サマリー設定（vLLMサーバURL、モデル名、APIキーなど）
        """
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key or "EMPTY",
        )

    @classmethod
    def _extract_markdown_block(cls, text: str) -> str:
        """
        レスポンステキストから最後のmarkdownコードブロックを抽出

        vLLMが思考過程（<think>タグなど）を含む応答を返す場合、
        まず<think>タグを削除してから、最終的なmarkdownコードブロック
        （```markdown ... ```）のみを抽出する。

        Args:
            text: LLMの生成したテキスト

        Returns:
            str: 抽出されたmarkdownコンテンツ、または元のテキスト

        Examples:
            >>> text = "<think>...</think>\\n```markdown\\n# Summary\\n```"
            >>> VLLMClient._extract_markdown_block(text)
            '# Summary'
        """
        # 1. <think>タグを削除
        text_without_think = cls._THINK_TAG_PATTERN.sub("", text)

        # 2. markdownブロックを抽出
        matches = cls._MARKDOWN_BLOCK_PATTERN.findall(text_without_think)
        return matches[-1].strip() if matches else text_without_think.strip()

    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> str | None:
        """
        vLLM APIでテキストを生成

        Args:
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            temperature: 生成の確率性（0.0=決定論的、1.0=最大ランダム性）
            top_p: nucleus sampling（累積確率がtop_pになるまでのトークンから選択、0.0-1.0）
            max_tokens: 最大トークン数

        Returns:
            str | None: 生成されたテキスト or None

        Raises:
            Exception: API呼び出しエラー
        """

        # 設定検証済みのため、vllm_modelは必ず存在する
        assert self.settings.vllm_model is not None
        response = self.client.chat.completions.create(
            model=self.settings.vllm_model,
            max_tokens=max_tokens or self.settings.max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                # 思考過程（<think>タグなど）を除外し、markdownブロックのみ抽出
                return self._extract_markdown_block(content.strip())
        return None

    def get_backend_info(self) -> str:
        """
        使用しているLLMバックエンドの情報を返す

        Returns:
            str: バックエンド情報（例: "vLLM (Qwen/Qwen3-30B-A3B @ http://localhost:8000/v1)"）
        """
        return f"vLLM ({self.settings.vllm_model} @ {self.settings.vllm_base_url})"


# ========================================
# Factory Function
# ========================================
def create_llm_client(settings: SummarySettings) -> LLMClient:
    """
    設定に基づいてLLMクライアントを生成

    Args:
        settings: サマリー設定

    Returns:
        LLMClient: 設定されたバックエンドのクライアント

    Raises:
        ValueError: バックエンド設定が無効な場合（到達不可能：Pydantic検証済み）

    Note:
        設定検証は SummarySettings.validate_backend_config で実施済み
    """
    match settings.backend:
        case LLMBackend.CLAUDE:
            return ClaudeClient(settings=settings)
        case LLMBackend.VLLM:
            return VLLMClient(settings=settings)
        case _:  # 到達不可能（StrEnum + Pydantic検証済み）、防衛的プログラミング
            raise ValueError(
                f"Invalid summary.backend: '{settings.backend}'. "
                f"Must be '{LLMBackend.CLAUDE}' or '{LLMBackend.VLLM}'"
            )
