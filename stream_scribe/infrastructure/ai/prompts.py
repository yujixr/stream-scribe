#!/usr/bin/env python3
"""
Stream Scribe - Prompt Templates
Claude API用のプロンプトテンプレートを管理するモジュール
"""

from typing import Protocol

from stream_scribe.domain import TranscriptionSession


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

    def build_user_prompt(self, **kwargs: str | TranscriptionSession) -> str:
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

    def build_user_prompt(self, current_summary: str, new_text_chunk: str) -> str:
        """
        リアルタイム要約用のユーザープロンプトを構築

        Args:
            current_summary: 現在の議事録（空文字列の場合は初期状態）
            new_text_chunk: 新しい発言テキスト

        Returns:
            str: 構築されたユーザープロンプト
        """
        # 現在の議事録がない場合の初期値
        context = current_summary if current_summary else "(まだ議事録はありません)"

        return f"""
【現在の議事録】
{context}

【新しい発言（音声認識生データ・誤字含む）】
{new_text_chunk}
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
        full_transcript = "\n".join(
            f"[{i + 1}] {segment.start_time.strftime('%H:%M:%S')} {segment.text}"
            for i, segment in enumerate(session.segments)
        )

        return f"""
以下は、会話の全文です（音声認識生データ・誤字含む）。
会話全体を俯瞰して、包括的なサマリを生成してください。

【全発言テキスト】
{full_transcript}
"""
