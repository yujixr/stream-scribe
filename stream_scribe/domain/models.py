#!/usr/bin/env python3
"""
Stream Scribe - Domain Models
ドメイン層：ビジネスエンティティとルール（外部依存なし）
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TranscriptionSegment:
    """文字起こしされた音声セグメント"""

    text: str
    start_time: datetime
    end_time: datetime
    audio_duration: float
    processing_time: float
    # Whisperメトリクス（分析用）
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


@dataclass
class TranscriptionError:
    """文字起こしエラー/品質問題の記録"""

    timestamp: datetime
    message: str


@dataclass
class SummaryEntry:
    """要約エントリ（中間・終了時サマリ）"""

    timestamp: datetime
    content: str


class TranscriptionSession:
    """
    会話記録セッション（ドメインエンティティ）

    責務:
    - セグメントの集約
    - エラーの集約
    - セッション状態の管理
    - サマリ履歴の保持（中間サマリ + 終了時サマリ）

    Note: 永続化ロジックは infrastructure/persistence に分離
    """

    def __init__(self) -> None:
        self.segments: list[TranscriptionSegment] = []
        self.errors: list[TranscriptionError] = []
        self.summaries: list[SummaryEntry] = []
        self.session_start = datetime.now()
        self.final_summary: SummaryEntry | None = None

    def add_segment(self, segment: TranscriptionSegment) -> None:
        """セグメントを追加"""
        self.segments.append(segment)

    def add_error(self, error: TranscriptionError) -> None:
        """エラーを追加"""
        self.errors.append(error)

    def add_summary(self, summary: str, is_final: bool = False) -> None:
        """
        サマリを追加

        Args:
            summary: サマリ内容
            is_final: 終了時サマリかどうか（Trueの場合はsummariesではなくfinal_summaryに設定）
        """
        entry = SummaryEntry(timestamp=datetime.now(), content=summary)
        if is_final:
            self.final_summary = entry
        else:
            self.summaries.append(entry)

    def get_total_segments(self) -> int:
        """総セグメント数を取得"""
        return len(self.segments)

    def get_total_errors(self) -> int:
        """総エラー数を取得"""
        return len(self.errors)
