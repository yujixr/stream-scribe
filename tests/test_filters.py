"""Tests for HallucinationFilter."""

import pytest

from stream_scribe.infrastructure.ml.filters import HallucinationFilter


@pytest.fixture
def filter_instance() -> HallucinationFilter:
    """Create a HallucinationFilter with common banned phrases."""
    banned_phrases = [
        "ご視聴ありがとうございました",
        "チャンネル登録",
        "高評価",
    ]
    return HallucinationFilter(banned_phrases)


class TestBannedPhrases:
    """Tests for banned phrase detection."""

    def test_detects_banned_phrase(self, filter_instance: HallucinationFilter) -> None:
        result = filter_instance.evaluate_transcription("ご視聴ありがとうございました")
        assert result is not None
        assert "Banned phrase" in result

    def test_detects_banned_phrase_in_middle(
        self, filter_instance: HallucinationFilter
    ) -> None:
        result = filter_instance.evaluate_transcription(
            "今日の動画はここまでです。ご視聴ありがとうございました。また次回"
        )
        assert result is not None
        assert "Banned phrase" in result

    def test_passes_normal_text(self, filter_instance: HallucinationFilter) -> None:
        result = filter_instance.evaluate_transcription("これは普通の文章です")
        assert result is None


class TestCharacterRepetition:
    """Tests for character repetition detection."""

    def test_detects_repeated_characters(
        self, filter_instance: HallucinationFilter
    ) -> None:
        result = filter_instance.evaluate_transcription("ああああああああああ")
        assert result is not None
        assert "Character repetition" in result

    def test_detects_repeated_katakana(
        self, filter_instance: HallucinationFilter
    ) -> None:
        result = filter_instance.evaluate_transcription("ンンンンンンンンンン")
        assert result is not None
        assert "Character repetition" in result

    def test_passes_short_repetition(
        self, filter_instance: HallucinationFilter
    ) -> None:
        result = filter_instance.evaluate_transcription("ああああ")  # Only 4
        assert result is None

    def test_passes_varied_text(self, filter_instance: HallucinationFilter) -> None:
        result = filter_instance.evaluate_transcription("あいうえおかきくけこ")
        assert result is None


class TestShortPatternRepetition:
    """Tests for short pattern repetition detection."""

    def test_detects_pattern_repetition(
        self, filter_instance: HallucinationFilter
    ) -> None:
        # Needs at least 20 chars and pattern must cover 50%+ of text
        result = filter_instance.evaluate_transcription(
            "ピリピリピリピリピリピリピリピリピリピリピリピリ"
        )
        assert result is not None
        assert "Pattern repetition" in result

    def test_detects_word_repetition(
        self, filter_instance: HallucinationFilter
    ) -> None:
        # Needs at least 20 chars and pattern must cover 50%+ of text
        result = filter_instance.evaluate_transcription(
            "はいはいはいはいはいはいはいはいはいはいはいはい"
        )
        assert result is not None
        assert "Pattern repetition" in result

    def test_passes_short_text(self, filter_instance: HallucinationFilter) -> None:
        result = filter_instance.evaluate_transcription("ピリピリ")
        assert result is None


class TestLongPatternRepetition:
    """Tests for long pattern repetition detection."""

    def test_detects_long_phrase_repetition(
        self, filter_instance: HallucinationFilter
    ) -> None:
        phrase = "私たちの意味が好きな話題について、"
        result = filter_instance.evaluate_transcription(phrase * 4)
        assert result is not None
        assert "Long phrase repetition" in result

    def test_passes_non_repetitive_long_text(
        self, filter_instance: HallucinationFilter
    ) -> None:
        text = (
            "今日は天気が良いです。明日は雨が降るかもしれません。週末は晴れるでしょう。"
        )
        result = filter_instance.evaluate_transcription(text)
        assert result is None


class TestTokenRepetition:
    """Tests for token repetition detection."""

    def test_detects_token_repetition_at_end(
        self, filter_instance: HallucinationFilter
    ) -> None:
        result = filter_instance.evaluate_transcription(
            "はい。はい。はい。はい。はい。"
        )
        assert result is not None
        assert "Token repetition" in result

    def test_passes_varied_tokens(self, filter_instance: HallucinationFilter) -> None:
        result = filter_instance.evaluate_transcription(
            "はい。いいえ。多分。そうですね。分かりました。"
        )
        assert result is None


class TestExtremeConfidence:
    """Tests for extreme low confidence detection."""

    def test_detects_extreme_low_confidence(
        self, filter_instance: HallucinationFilter
    ) -> None:
        result = filter_instance.evaluate_transcription("テスト", avg_logprob=-2.0)
        assert result is not None
        assert "Extreme low confidence" in result

    def test_passes_normal_confidence(
        self, filter_instance: HallucinationFilter
    ) -> None:
        result = filter_instance.evaluate_transcription("テスト", avg_logprob=-0.5)
        assert result is None

    def test_passes_none_confidence(self, filter_instance: HallucinationFilter) -> None:
        result = filter_instance.evaluate_transcription("テスト", avg_logprob=None)
        assert result is None


class TestEmptyInput:
    """Tests for empty input handling."""

    def test_empty_string(self, filter_instance: HallucinationFilter) -> None:
        result = filter_instance.evaluate_transcription("")
        assert result is None

    def test_whitespace_only(self, filter_instance: HallucinationFilter) -> None:
        result = filter_instance.evaluate_transcription("   ")
        assert result is None


class TestExtractMetrics:
    """Tests for metrics extraction."""

    def test_extracts_metrics(self, filter_instance: HallucinationFilter) -> None:
        segments = [
            {"avg_logprob": -0.5, "compression_ratio": 1.2, "no_speech_prob": 0.1},
            {"avg_logprob": -0.7, "compression_ratio": 1.5, "no_speech_prob": 0.2},
        ]
        avg_logprob, compression_ratio, no_speech_prob = (
            filter_instance.extract_metrics(segments)
        )

        assert avg_logprob == pytest.approx(-0.6, rel=1e-2)
        assert compression_ratio == 1.5  # max
        assert no_speech_prob == 0.2  # max

    def test_handles_empty_segments(self, filter_instance: HallucinationFilter) -> None:
        avg_logprob, compression_ratio, no_speech_prob = (
            filter_instance.extract_metrics([])
        )
        assert avg_logprob is None
        assert compression_ratio is None
        assert no_speech_prob is None

    def test_handles_none_segments(self, filter_instance: HallucinationFilter) -> None:
        avg_logprob, compression_ratio, no_speech_prob = (
            filter_instance.extract_metrics(None)
        )
        assert avg_logprob is None
        assert compression_ratio is None
        assert no_speech_prob is None

    def test_handles_partial_metrics(
        self, filter_instance: HallucinationFilter
    ) -> None:
        segments = [
            {"avg_logprob": -0.5},
            {"compression_ratio": 1.5},
        ]
        avg_logprob, compression_ratio, no_speech_prob = (
            filter_instance.extract_metrics(segments)
        )
        assert avg_logprob == -0.5
        assert compression_ratio == 1.5
        assert no_speech_prob is None
