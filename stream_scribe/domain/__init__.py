#!/usr/bin/env python3
"""
Stream Scribe - Domain Layer
ドメイン層：ビジネスロジック、エンティティ、設定
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

# 設定スキーマ（Pydantic）
from .settings import (
    AppSettings,
    AudioSettings,
    CoreSettings,
    HallucinationFilterSettings,
    LLMBackend,
    Settings,
    SummarySettings,
    VADDetectionSettings,
    VADModelSettings,
    VADSettings,
    WhisperParamsSettings,
    WhisperSettings,
)

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
    # 設定
    "AppSettings",
    "AudioSettings",
    "CoreSettings",
    "HallucinationFilterSettings",
    "LLMBackend",
    "Settings",
    "SummarySettings",
    "VADDetectionSettings",
    "VADModelSettings",
    "VADSettings",
    "WhisperParamsSettings",
    "WhisperSettings",
]
