"""HallucinationFilterのテスト"""

import sys
from unittest.mock import MagicMock

import pytest

# mlx が利用できない環境（Linux CI等）ではモックする
if "mlx" not in sys.modules:
    sys.modules["mlx"] = MagicMock()
    sys.modules["mlx.core"] = MagicMock()
if "mlx_whisper" not in sys.modules:
    sys.modules["mlx_whisper"] = MagicMock()

from stream_scribe.infrastructure.ml.filters import HallucinationFilter


@pytest.fixture
def filter_instance() -> HallucinationFilter:
    """一般的な禁止フレーズを持つHallucinationFilterを作成"""
    banned_phrases = [
        "ご視聴ありがとうございました",
        "チャンネル登録",
        "高評価",
    ]
    return HallucinationFilter(banned_phrases)


class TestBannedPhrases:
    """禁止フレーズ検出のテスト"""

    def test_detects_banned_phrase(self, filter_instance: HallucinationFilter) -> None:
        """禁止フレーズを検出する"""
        result = filter_instance.evaluate_transcription("ご視聴ありがとうございました")
        assert result is not None
        assert "Banned phrase" in result

    def test_detects_banned_phrase_in_middle(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """文中の禁止フレーズを検出する"""
        result = filter_instance.evaluate_transcription(
            "今日の動画はここまでです。ご視聴ありがとうございました。また次回"
        )
        assert result is not None
        assert "Banned phrase" in result

    def test_passes_normal_text(self, filter_instance: HallucinationFilter) -> None:
        """通常のテキストは通過する"""
        result = filter_instance.evaluate_transcription("これは普通の文章です")
        assert result is None


class TestCharacterRepetition:
    """文字繰り返し検出のテスト"""

    def test_detects_repeated_characters(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """繰り返し文字を検出する"""
        result = filter_instance.evaluate_transcription("ああああああああああ")
        assert result is not None
        assert "Character repetition" in result

    def test_detects_repeated_katakana(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """繰り返しカタカナを検出する"""
        result = filter_instance.evaluate_transcription("ンンンンンンンンンン")
        assert result is not None
        assert "Character repetition" in result

    def test_passes_short_repetition(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """短い繰り返しは通過する"""
        result = filter_instance.evaluate_transcription("ああああ")  # 4文字のみ
        assert result is None

    def test_passes_varied_text(self, filter_instance: HallucinationFilter) -> None:
        """多様なテキストは通過する"""
        result = filter_instance.evaluate_transcription("あいうえおかきくけこ")
        assert result is None


class TestShortPatternRepetition:
    """短パターン繰り返し検出のテスト"""

    def test_detects_pattern_repetition(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """パターンの繰り返しを検出する（20文字以上かつ50%以上がパターン）"""
        result = filter_instance.evaluate_transcription(
            "ピリピリピリピリピリピリピリピリピリピリピリピリ"
        )
        assert result is not None
        assert "Pattern repetition" in result

    def test_detects_word_repetition(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """単語の繰り返しを検出する（20文字以上かつ50%以上がパターン）"""
        result = filter_instance.evaluate_transcription(
            "はいはいはいはいはいはいはいはいはいはいはいはい"
        )
        assert result is not None
        assert "Pattern repetition" in result

    def test_passes_short_text(self, filter_instance: HallucinationFilter) -> None:
        """短いテキストは通過する"""
        result = filter_instance.evaluate_transcription("ピリピリ")
        assert result is None


class TestLongPatternRepetition:
    """長パターン繰り返し検出のテスト"""

    def test_detects_long_phrase_repetition(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """長いフレーズの繰り返しを検出する"""
        phrase = "私たちの意味が好きな話題について、"
        result = filter_instance.evaluate_transcription(phrase * 4)
        assert result is not None
        assert "Long phrase repetition" in result

    def test_passes_non_repetitive_long_text(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """繰り返しのない長いテキストは通過する"""
        text = (
            "今日は天気が良いです。明日は雨が降るかもしれません。週末は晴れるでしょう。"
        )
        result = filter_instance.evaluate_transcription(text)
        assert result is None


class TestTokenRepetition:
    """トークン繰り返し検出のテスト"""

    def test_detects_token_repetition_at_end(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """末尾のトークン繰り返しを検出する"""
        result = filter_instance.evaluate_transcription(
            "はい。はい。はい。はい。はい。"
        )
        assert result is not None
        assert "Token repetition" in result

    def test_passes_varied_tokens(self, filter_instance: HallucinationFilter) -> None:
        """多様なトークンは通過する"""
        result = filter_instance.evaluate_transcription(
            "はい。いいえ。多分。そうですね。分かりました。"
        )
        assert result is None


class TestExtremeConfidence:
    """極端な低信頼度検出のテスト"""

    def test_detects_extreme_low_confidence(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """極端に低い信頼度を検出する"""
        result = filter_instance.evaluate_transcription("テスト", avg_logprob=-2.0)
        assert result is not None
        assert "Extreme low confidence" in result

    def test_passes_normal_confidence(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """通常の信頼度は通過する"""
        result = filter_instance.evaluate_transcription("テスト", avg_logprob=-0.5)
        assert result is None

    def test_passes_none_confidence(self, filter_instance: HallucinationFilter) -> None:
        """信頼度がNoneの場合は通過する"""
        result = filter_instance.evaluate_transcription("テスト", avg_logprob=None)
        assert result is None


class TestEmptyInput:
    """空入力処理のテスト"""

    def test_empty_string(self, filter_instance: HallucinationFilter) -> None:
        """空文字列の処理"""
        result = filter_instance.evaluate_transcription("")
        assert result is None

    def test_whitespace_only(self, filter_instance: HallucinationFilter) -> None:
        """空白のみの文字列の処理"""
        result = filter_instance.evaluate_transcription("   ")
        assert result is None


class TestContextlessGreeting:
    """文脈なし挨拶フィルタのテスト"""

    def test_detects_greeting_with_low_confidence(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """低信頼度の挨拶を検出する"""
        result = filter_instance.evaluate_transcription(
            "おやすみなさい", avg_logprob=-0.9, audio_duration=2.0
        )
        assert result is not None
        assert "Contextless greeting with low confidence" in result
        assert "おやすみなさい" in result

    def test_detects_greeting_in_long_audio(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """長尺音声中の短い挨拶を検出する"""
        result = filter_instance.evaluate_transcription(
            "ありがとう", avg_logprob=-0.3, audio_duration=6.0
        )
        assert result is not None
        assert "Contextless greeting in long audio" in result
        assert "ありがとう" in result

    def test_passes_greeting_with_normal_confidence_and_short_audio(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """通常の信頼度かつ短い音声の挨拶は通過する"""
        result = filter_instance.evaluate_transcription(
            "おはよう", avg_logprob=-0.3, audio_duration=2.0
        )
        assert result is None

    def test_passes_greeting_in_long_context(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """長い文脈中の挨拶は通過する"""
        result = filter_instance.evaluate_transcription(
            "今日はいい天気ですね。おはようございます。",
            avg_logprob=-0.9,
            audio_duration=3.0,
        )
        assert result is None

    def test_passes_greeting_without_audio_duration(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """audio_durationなしで低信頼度のみチェック"""
        # 低信頼度で検出される
        result = filter_instance.evaluate_transcription(
            "こんにちは", avg_logprob=-0.9, audio_duration=None
        )
        assert result is not None
        assert "low confidence" in result

        # 通常の信頼度では通過
        result = filter_instance.evaluate_transcription(
            "こんにちは", avg_logprob=-0.3, audio_duration=None
        )
        assert result is None

    def test_detects_greeting_with_punctuation(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """句読点付きの挨拶も検出する"""
        result = filter_instance.evaluate_transcription(
            "おやすみなさい。", avg_logprob=-0.9, audio_duration=2.0
        )
        assert result is not None
        assert "Contextless greeting" in result


class TestExtractMetrics:
    """メトリクス抽出のテスト"""

    def test_extracts_metrics(self, filter_instance: HallucinationFilter) -> None:
        """メトリクスを正しく抽出する"""
        segments = [
            {"avg_logprob": -0.5, "compression_ratio": 1.2, "no_speech_prob": 0.1},
            {"avg_logprob": -0.7, "compression_ratio": 1.5, "no_speech_prob": 0.2},
        ]
        avg_logprob, compression_ratio, no_speech_prob = (
            filter_instance.extract_metrics(segments)
        )

        assert avg_logprob == pytest.approx(-0.6, rel=1e-2)
        assert compression_ratio == 1.5  # 最大値
        assert no_speech_prob == 0.2  # 最大値

    def test_handles_empty_segments(self, filter_instance: HallucinationFilter) -> None:
        """空のセグメントリストを処理する"""
        avg_logprob, compression_ratio, no_speech_prob = (
            filter_instance.extract_metrics([])
        )
        assert avg_logprob is None
        assert compression_ratio is None
        assert no_speech_prob is None

    def test_handles_none_segments(self, filter_instance: HallucinationFilter) -> None:
        """Noneのセグメントを処理する"""
        avg_logprob, compression_ratio, no_speech_prob = (
            filter_instance.extract_metrics(None)
        )
        assert avg_logprob is None
        assert compression_ratio is None
        assert no_speech_prob is None

    def test_handles_partial_metrics(
        self, filter_instance: HallucinationFilter
    ) -> None:
        """部分的なメトリクスを処理する"""
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
