#!/usr/bin/env python3
"""
Stream Scribe - AI Infrastructure
AI関連のインフラストラクチャ層（Claude要約生成）
"""

# Summarizer
from .summarizer import RealtimeSummarizer

# Claude Client
from .claude_client import ClaudeClient

# Prompts
from . import prompts

__all__ = [
    # Summarizer
    "RealtimeSummarizer",
    # Claude Client
    "ClaudeClient",
    # Prompts module
    "prompts",
]
