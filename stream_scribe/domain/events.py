#!/usr/bin/env python3
"""
Stream Scribe - Domain Events
ドメイン層：イベントハンドラの抽象インターフェース
"""

from datetime import datetime
from typing import Protocol

import numpy as np

from stream_scribe.domain.models import TranscriptionSegment


class EventHandler(Protocol):
    """
    イベントハンドラの抽象インターフェース

    システム全体のイベントを処理するハンドラが実装すべきメソッドを定義する。

    イベント種別:
    - on_audio: VAD検知による音声録音完了
    - on_segment: 文字起こし完了
    - on_summary: 要約生成完了
    - on_error: エラー発生

    Protocol型を使用することで、明示的な継承なしに
    構造的部分型（Structural Subtyping）を実現する。
    """

    def on_audio(
        self, audio: np.ndarray, start_time: datetime, end_time: datetime
    ) -> None:
        """
        音声録音完了時のハンドラ

        Args:
            audio: 録音された音声データ（16kHz モノラル float32）
            start_time: 録音開始時刻
            end_time: 録音終了時刻
        """
        ...

    def on_segment(self, segment: TranscriptionSegment) -> None:
        """
        セグメント完了時のハンドラ

        Args:
            segment: 文字起こしされた音声セグメント
        """
        ...

    def on_summary(self, summary: str) -> None:
        """
        サマリ生成時のハンドラ

        Args:
            summary: 生成されたサマリテキスト
        """
        ...

    def on_error(
        self, error_time: datetime, error_message: str, exception: Exception | None
    ) -> None:
        """
        エラー発生時のハンドラ

        Args:
            error_time: エラー発生時刻
            error_message: エラーメッセージ
            exception: 例外オブジェクト（存在する場合）
        """
        ...
