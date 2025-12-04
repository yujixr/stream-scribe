#!/usr/bin/env python3
"""
Stream Scribe - Configuration Loader
設定の読み込み（TOML）
"""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stream_scribe.domain import Settings


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    2つの辞書を深くマージする（overrideが優先）

    Args:
        base: ベースとなる辞書
        override: 上書きする辞書

    Returns:
        マージされた辞書
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(dict(result[key]), dict(value))
        else:
            result[key] = value
    return result


def load_settings() -> Settings:
    """
    TOMLファイルから設定を読み込む

    読み込み順序（後勝ち）:
    1. デフォルト値（domain/settings.py内）
    2. {project_root}/config.toml（存在する場合）
    3. {project_root}/config.local.toml（存在する場合）

    Returns:
        Settingsインスタンス
    """
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config.toml"
    local_config_path = project_root / "config.local.toml"

    # config.tomlを読み込み
    config_data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("rb") as f:
            config_data = tomllib.load(f)

    # config.local.tomlを読み込んでマージ（存在する場合）
    if local_config_path.exists():
        with local_config_path.open("rb") as f:
            config_data = _deep_merge(config_data, tomllib.load(f))

    return Settings(**config_data) if config_data else Settings()
