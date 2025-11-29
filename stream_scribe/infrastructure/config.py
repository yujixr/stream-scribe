#!/usr/bin/env python3
"""
Stream Scribe - Configuration Loader
設定の読み込み（TOML）
"""

import tomllib
from pathlib import Path

from stream_scribe.domain import Settings


def load_settings() -> Settings:
    """
    TOMLファイルから設定を読み込む

    読み込み順序:
    1. デフォルト値（domain/settings.py内）
    2. {project_root}/config.toml（存在する場合）

    Returns:
        Settingsインスタンス
    """
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config.toml"

    try:
        with config_path.open("rb") as f:
            return Settings(**tomllib.load(f))
    except FileNotFoundError:
        return Settings()
