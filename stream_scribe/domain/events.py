#!/usr/bin/env python3
"""
Stream Scribe - Events (Pub/Sub)
ドメイン層: イベント駆動アーキテクチャの中核
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
from blinker import Signal

from .models import TranscriptionSegment

# ========================================
# イベント名定数
# ========================================
EVENT_AUDIO_RECORDED = "audio_recorded"
EVENT_SEGMENT_TRANSCRIBED = "segment_transcribed"
EVENT_SUMMARY_GENERATED = "summary_generated"
EVENT_MESSAGE_POSTED = "message_posted"


# ========================================
# イベント型定義
# ========================================


class MessageLevel(str, Enum):
    """メッセージレベル"""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class AudioRecordedEvent:
    """
    音声録音完了イベント

    VAD検知により音声が録音完了した際に発行される。
    """

    audio: np.ndarray  # 録音された音声データ（16kHz モノラル float32）
    start_time: datetime  # 録音開始時刻
    end_time: datetime  # 録音終了時刻


@dataclass(frozen=True)
class SegmentTranscribedEvent:
    """
    文字起こし完了イベント

    音声セグメントの文字起こしが完了した際に発行される。
    """

    segment: TranscriptionSegment  # 文字起こしされた音声セグメント


@dataclass(frozen=True)
class SummaryGeneratedEvent:
    """
    要約生成イベント

    会話の構造化サマリが生成された際に発行される。
    """

    summary: str  # 生成されたサマリテキスト


@dataclass(frozen=True)
class MessagePostedEvent:
    """
    メッセージ投稿イベント

    システム状態の変化やユーザーへの通知メッセージを表示する際に発行される。
    timestampは省略時に自動的に現在時刻が設定される。
    """

    message: str  # 表示するメッセージ
    level: MessageLevel  # メッセージレベル（INFO/SUCCESS/WARNING/ERROR）
    timestamp: datetime = field(
        default_factory=datetime.now
    )  # メッセージタイムスタンプ（省略時は自動設定）


# イベント型のユニオン（型チェック用）
Event = (
    AudioRecordedEvent
    | SegmentTranscribedEvent
    | SummaryGeneratedEvent
    | MessagePostedEvent
)


# ========================================
# グローバルシグナル定義
# ========================================

# 各イベントに対応するシグナル
# blinkerのSignalは型安全でスレッドセーフ
audio_recorded = Signal(EVENT_AUDIO_RECORDED)  # AudioRecordedEvent
segment_transcribed = Signal(EVENT_SEGMENT_TRANSCRIBED)  # SegmentTranscribedEvent
summary_generated = Signal(EVENT_SUMMARY_GENERATED)  # SummaryGeneratedEvent
message_posted = Signal(EVENT_MESSAGE_POSTED)  # MessagePostedEvent
