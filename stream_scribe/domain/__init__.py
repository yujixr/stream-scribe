#!/usr/bin/env python3
"""
Stream Scribe - Domain Layer
ドメイン層：ビジネスロジック、エンティティ、定数
"""

# Models and Data Structures
from stream_scribe.domain.models import (
    SummaryEntry,
    TranscriptionError,
    TranscriptionSegment,
    TranscriptionSession,
)

# Events (Pub/Sub)
from stream_scribe.domain.events import (
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

# Constants
from stream_scribe.domain import constants

__all__ = [
    # Models
    "SummaryEntry",
    "TranscriptionError",
    "TranscriptionSegment",
    "TranscriptionSession",
    # Events
    "AudioRecordedEvent",
    "MessageLevel",
    "MessagePostedEvent",
    "SegmentTranscribedEvent",
    "SummaryGeneratedEvent",
    "audio_recorded",
    "message_posted",
    "segment_transcribed",
    "summary_generated",
    # Constants module
    "constants",
]
