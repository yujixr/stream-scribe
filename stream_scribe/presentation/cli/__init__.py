#!/usr/bin/env python3
"""
Stream Scribe - CLI Presentation
CLIプレゼンテーション層
"""

# CLI View
from .view import CLIView

# CLI Entry Point
from .main import main

__all__ = [
    # View
    "CLIView",
    # Entry Point
    "main",
]
