#!/usr/bin/env python3
"""
Stream Scribe - JSON Exporter
インフラ層：セッションデータのJSON永続化
"""

import json
from datetime import datetime
from pathlib import Path

from stream_scribe.domain.models import TranscriptionSession


class SessionJsonExporter:
    """
    TranscriptionSessionをJSON形式で永続化

    責務:
    - セッションデータのシリアライズ
    - ファイルシステムへの保存
    """

    @staticmethod
    def save_to_file(
        session: TranscriptionSession, output_path: Path | None = None
    ) -> Path:
        """
        書き起こし結果をJSONファイルに保存

        Args:
            session: 保存するセッション
            output_path: 出力先パス（Noneの場合は自動生成）

        Returns:
            Path: 保存されたファイルのパス
        """
        if output_path is None:
            # デフォルトファイル名: transcription_YYYYMMDD_HHMMSS.json
            timestamp = session.session_start.strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"transcription_{timestamp}.json")

        # セグメントをdictに変換
        segments_dict = []
        for seg in session.segments:
            segment_data = {
                "text": seg.text,
                "start_time": seg.start_time.isoformat(),
                "end_time": seg.end_time.isoformat(),
                "audio_duration": round(seg.audio_duration, 2),
                "processing_time": round(seg.processing_time, 2),
            }
            # Whisperメトリクス（存在する場合のみ追加）
            if seg.avg_logprob is not None:
                segment_data["avg_logprob"] = round(seg.avg_logprob, 3)
            if seg.compression_ratio is not None:
                segment_data["compression_ratio"] = round(seg.compression_ratio, 3)
            if seg.no_speech_prob is not None:
                segment_data["no_speech_prob"] = round(seg.no_speech_prob, 3)
            segments_dict.append(segment_data)

        # エラーをdictに変換
        errors_dict = [
            {
                "timestamp": err.timestamp.isoformat(),
                "message": err.message,
                "exception_type": err.exception_type,
            }
            for err in session.errors
        ]

        output_data = {
            "session_start": session.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_segments": session.get_total_segments(),
            "total_errors": session.get_total_errors(),
            "segments": segments_dict,
            "errors": errors_dict,
        }

        # 構造化サマリがあれば追加
        if session.structured_summary:
            output_data["structured_summary"] = session.structured_summary

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return output_path
