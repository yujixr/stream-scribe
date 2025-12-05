#!/usr/bin/env python3
"""
Stream Scribe - Prompt Templates
LLM API用のプロンプトテンプレートを管理するモジュール
"""

from typing import Protocol

from stream_scribe.domain import TranscriptionSegment, TranscriptionSession


def format_segments(segments: list[TranscriptionSegment]) -> str:
    """
    セグメントリストをタイムスタンプ付きテキストにフォーマット

    Args:
        segments: フォーマットするセグメントのリスト
        start_index: 開始インデックス番号

    Returns:
        str: フォーマットされたテキスト（改行区切り）
    """
    return "\n".join(
        f"[{seg.start_time.strftime('%H:%M:%S')}] {seg.text}" for seg in segments
    )


class PromptStrategy(Protocol):
    """
    プロンプト構築戦略の抽象インターフェース

    責務:
    - システムプロンプトの提供
    - ユーザープロンプトの構築
    """

    @property
    def system_prompt(self) -> str:
        """システムプロンプトを取得"""
        ...

    def build_user_prompt(
        self,
        **kwargs: str | None | list[TranscriptionSegment] | TranscriptionSession,
    ) -> str:
        """ユーザープロンプトを構築"""
        ...


class RealtimePromptStrategy:
    """
    リアルタイム要約用プロンプト戦略

    責務:
    - インテリジェント・スクライブとしてのシステムプロンプト提供
    - 現在の議事録と新しい発言を統合するユーザープロンプト構築
    """

    @property
    def system_prompt(self) -> str:
        """リアルタイム要約用システムプロンプト"""
        return """
リアルタイム会話を構造化し、議事録を更新してください。

# 制約
- 修正報告・挨拶・前置き・思考過程を出力しないこと
- 指定フォーマット以外のテキストを含めないこと

# ノイズ補正
音声認識の誤変換・フィラー（"えー"等）を文脈から判断して修正・削除してください。

# 構造化ルール
- アクティブな話題: 詳細に記録
- 完了した話題: 大トピックと結論のみ残す（圧縮）

# 出力（Markdown）
## 🚀 現在の焦点
(現在話されている内容を1行で)

## 🌳 トピック・ツリー
- **話題1 (完了)**
  - [結論] 〇〇
- **話題2 (進行中)**
  - 議論ポイントA
    - [ToDo] 担当者・内容

## ⏱️ 直近ログ
(補正済み発言を時系列で3件程度)
"""

    def build_user_prompt(
        self,
        previous_summary: str | None,
        processed_segments: list[TranscriptionSegment] | None,
        new_segments: list[TranscriptionSegment],
    ) -> str:
        """
        リアルタイム要約用のユーザープロンプトを構築

        Args:
            previous_summary: 前回生成した議事録（Noneの場合は初期状態）
            processed_segments: 処理済みセグメント（直近N件）
            new_segments: 未処理の新しいセグメント

        Returns:
            str: 構築されたユーザープロンプト
        """
        # 前回の議事録
        summary_text = (
            previous_summary if previous_summary else "(まだ議事録はありません)"
        )

        # 処理済みセグメントをタイムスタンプ付きでフォーマット
        if processed_segments:
            processed_text = format_segments(processed_segments)
            new_text = format_segments(new_segments)
            # 処理済み + 区切り線 + 未処理
            transcript = f"{processed_text}\n\n--- ここから新しい発言 ---\n\n{new_text}"
        else:
            # 処理済みがなければ未処理のみ
            transcript = format_segments(new_segments)

        return f"""
【現在の議事録】
{summary_text}

【直近の発言テキスト（音声認識生データ・誤字含む）】
{transcript}
"""


class FinalSummaryPromptStrategy:
    """
    終了時サマリ用プロンプト戦略

    責務:
    - アーカイブ・アナリストとしてのシステムプロンプト提供
    - 全発言テキストから包括的サマリを生成するユーザープロンプト構築
    """

    @property
    def system_prompt(self) -> str:
        """終了時サマリ用システムプロンプト"""
        return """
会話全体を俯瞰し、包括的なサマリを生成してください。

# 制約
- 修正報告・挨拶・前置き・思考過程を出力しないこと
- 指定フォーマット以外のテキストを含めないこと

# ノイズ補正
音声認識の誤変換・フィラーを文脈から判断して修正・削除してください。

# 構造化
会話の性質（会議/講義/雑談/インタビュー等）を推定し、適切に構造化してください。

# 出力（Markdown）
## 📋 会話の概要
(全体を2-3行で。性質も含む)

## 🌳 トピック・ツリー
- **メイントピック1**
  - サブトピック1-1
    - [結論/要点] 〇〇
    - [ToDo] 担当者・内容

## 💡 重要ポイント
- [決定] 〇〇
- [ToDo] 担当者・内容（期限）
- [疑問] 未解決事項

## 🔑 キーワード
`キーワード1`, `キーワード2`, ...（5-10個）
"""

    def build_user_prompt(self, session: TranscriptionSession) -> str:
        """
        終了時サマリ用のユーザープロンプトを構築

        Args:
            session: 文字起こしセッション（全セグメントを含む）

        Returns:
            str: 構築されたユーザープロンプト
        """
        # 全セグメントをタイムスタンプ付きで時系列順に結合
        full_transcript = format_segments(session.segments)

        return f"""
以下は、会話の全文です（音声認識生データ・誤字含む）。
会話全体を俯瞰して、包括的なサマリを生成してください。

【全発言テキスト】
{full_transcript}
"""
