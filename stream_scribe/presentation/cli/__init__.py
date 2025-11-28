#!/usr/bin/env python3
"""
Stream Scribe - CLI Presentation
CLIプレゼンテーション層
"""

# CLIコントローラー
from .controller import CLIController

# CLIビュー
from .view import CLIView

# CLIエントリーポイント
from .main import main

__all__ = [
    # コントローラー
    "CLIController",
    # ビュー
    "CLIView",
    # エントリーポイント
    "main",
]
