#!/usr/bin/env python3
"""
Stream Scribe - Infrastructure Layer
インフラストラクチャ層: 外部I/O、永続化、設定読み込み
"""

from .config import load_settings

__all__ = [
    "load_settings",
]
