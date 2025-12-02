#!/usr/bin/env python3
"""
Tests for LLM Client Adapters
VLLMClientのMarkdown抽出機能のテスト
"""

from stream_scribe.infrastructure.ai.llm_client import VLLMClient


class TestVLLMMarkdownExtraction:
    """VLLMClientのMarkdown抽出ロジックのテスト"""

    def test_extracts_markdown_from_think_tags(self) -> None:
        """思考過程タグを含むレスポンスからMarkdownを抽出"""
        response = """<think>
Okay, let's start by looking at the user's query. They provided a conversation transcript.

First, I need to check the constraints. The user mentioned not to include any corrections.
</think>

```markdown
## 📋 会話の概要
簡潔な確認コメント。作業の成功を示す発言が中心。

## 🔑 キーワード
`成功確認`, `作業完了`, `進行状況`
```"""

        result = VLLMClient._extract_markdown_block(response)

        expected = """## 📋 会話の概要
簡潔な確認コメント。作業の成功を示す発言が中心。

## 🔑 キーワード
`成功確認`, `作業完了`, `進行状況`"""

        assert result == expected

    def test_extracts_last_markdown_block_when_multiple(self) -> None:
        """複数のMarkdownブロックがある場合、最後のブロックを抽出"""
        response = """```markdown
## First Block
This is the first block.
```

Some text in between.

```markdown
## Second Block
This is the last block.
```"""

        result = VLLMClient._extract_markdown_block(response)
        assert result == "## Second Block\nThis is the last block."

    def test_returns_original_text_when_no_markdown_block(self) -> None:
        """Markdownブロックがない場合、元のテキストを返す"""
        response = "This is a plain text response without code blocks."
        result = VLLMClient._extract_markdown_block(response)
        assert result == response

    def test_handles_empty_markdown_block(self) -> None:
        """空のMarkdownブロックを処理"""
        response = """```markdown

```"""
        result = VLLMClient._extract_markdown_block(response)
        assert result == ""

    def test_handles_markdown_block_with_extra_whitespace(self) -> None:
        """前後の空白を除去"""
        response = """```markdown

## Title
Content here

```"""
        result = VLLMClient._extract_markdown_block(response)
        assert result == "## Title\nContent here"

    def test_preserves_internal_markdown_formatting(self) -> None:
        """Markdown内部のフォーマットを保持"""
        response = """```markdown
## Title

- Item 1
- Item 2

**Bold** and *italic* text.

`code snippet`
```"""
        result = VLLMClient._extract_markdown_block(response)
        expected = """## Title

- Item 1
- Item 2

**Bold** and *italic* text.

`code snippet`"""
        assert result == expected

    def test_handles_multiline_think_tags(self) -> None:
        """複数行の思考タグを含むレスポンスを処理"""
        response = """<think>
Line 1
Line 2
Line 3
</think>

Other text outside markdown.

```markdown
## Summary
Final content
```

More text after."""

        result = VLLMClient._extract_markdown_block(response)
        assert result == "## Summary\nFinal content"

    def test_removes_think_tags_before_markdown_extraction(self) -> None:
        """<think>タグを削除してからmarkdownブロックを抽出"""
        response = """<think>
Analyzing the conversation...
Checking context...
</think>

```markdown
## 📋 会話の概要
簡潔な確認コメント。
```"""

        result = VLLMClient._extract_markdown_block(response)
        assert result == "## 📋 会話の概要\n簡潔な確認コメント。"
        # <think>タグが結果に含まれていないことを確認
        assert "<think>" not in result
        assert "</think>" not in result

    def test_handles_multiple_think_tags(self) -> None:
        """複数の<think>タグを含むレスポンスを処理"""
        response = """<think>
First thought
</think>

Some text

<think>
Second thought
</think>

```markdown
## Result
Final output
```"""

        result = VLLMClient._extract_markdown_block(response)
        assert result == "## Result\nFinal output"
        assert "<think>" not in result

    def test_returns_cleaned_text_when_no_markdown_block_with_think_tags(self) -> None:
        """markdownブロックがなく<think>タグのみの場合、タグを削除したテキストを返す"""
        response = """<think>
Thinking process
</think>

This is the actual response without markdown blocks."""

        result = VLLMClient._extract_markdown_block(response)
        # <think>タグが削除されていることを確認
        assert "<think>" not in result
        assert "</think>" not in result
        # 実際のレスポンステキストが含まれていることを確認
        assert "This is the actual response without markdown blocks." in result
