#!/usr/bin/env python3
"""
Stream Scribe - AI Infrastructure
AI関連のインフラストラクチャ層（Claude要約生成）
"""

# 要約生成
from .summarizer import RealtimeSummarizer

# Claudeクライアント
from .claude_client import ClaudeClient

# プロンプト
from . import prompts

__all__ = [
    # 要約生成
    "RealtimeSummarizer",
    # Claudeクライアント
    "ClaudeClient",
    # プロンプトモジュール
    "prompts",
]
