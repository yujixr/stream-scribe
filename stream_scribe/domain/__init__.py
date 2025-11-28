#!/usr/bin/env python3
"""
Stream Scribe - Domain Layer
ドメイン層：ビジネスロジック、エンティティ、定数
"""

# モデルとデータ構造
from .models import (
    SummaryEntry,
    TranscriptionError,
    TranscriptionSegment,
    TranscriptionSession,
)

# イベント（Pub/Sub）
from .events import (
    AudioRecordedEvent,
    MessageLevel,
    MessagePostedEvent,
    SegmentTranscribedEvent,
    SummaryGeneratedEvent,
    audio_recorded,
    message_posted,
    segment_transcribed,
    summary_generated,
)

# 定数モジュール
from . import constants

__all__ = [
    # モデル
    "SummaryEntry",
    "TranscriptionError",
    "TranscriptionSegment",
    "TranscriptionSession",
    # イベント
    "AudioRecordedEvent",
    "MessageLevel",
    "MessagePostedEvent",
    "SegmentTranscribedEvent",
    "SummaryGeneratedEvent",
    "audio_recorded",
    "message_posted",
    "segment_transcribed",
    "summary_generated",
    # 定数モジュール
    "constants",
]
