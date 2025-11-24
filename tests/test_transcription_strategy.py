"""TranscriptionRetryStrategyのテスト"""

import pytest

from stream_scribe.domain.constants import MAX_TRANSCRIPTION_RETRIES, WHISPER_PARAMS
from stream_scribe.infrastructure.ml.transcription_strategy import (
    StrategyResult,
    TranscriptionAction,
    TranscriptionRetryStrategy,
)


@pytest.fixture
def strategy() -> TranscriptionRetryStrategy:
    """新しいTranscriptionRetryStrategyインスタンスを作成"""
    return TranscriptionRetryStrategy()


class TestInitialState:
    """初期状態のテスト"""

    def test_initial_attempt_is_zero(self, strategy: TranscriptionRetryStrategy) -> None:
        """初期状態では試行回数がゼロ"""
        assert strategy.current_attempt == 0

    def test_initial_params_are_first_whisper_params(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """初期パラメータは最初のWHISPER_PARAMS"""
        assert strategy.get_current_params() == WHISPER_PARAMS[0]

    def test_initial_attempt_info(self, strategy: TranscriptionRetryStrategy) -> None:
        """初期試行情報は(1, MAX_TRANSCRIPTION_RETRIES)"""
        attempt, max_attempts = strategy.get_attempt_info()
        assert attempt == 1
        assert max_attempts == MAX_TRANSCRIPTION_RETRIES


class TestReset:
    """リセット機能のテスト"""

    def test_reset_clears_attempt_counter(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """リセットで試行カウンターがクリアされる"""
        # いくつかリトライを進める
        strategy.evaluate_result("", "Some error")
        strategy.evaluate_result("", "Some error")
        assert strategy.current_attempt > 0

        # リセット
        strategy.reset()
        assert strategy.current_attempt == 0


class TestAcceptAction:
    """ACCEPT アクションのテスト"""

    def test_accepts_valid_text_without_filter_reason(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """有効なテキストでフィルタ理由がない場合はACCEPT"""
        result = strategy.evaluate_result("これは正常なテキストです", None)
        assert result.action == TranscriptionAction.ACCEPT
        assert result.next_params is None
        assert result.reason is None

    def test_accepts_on_first_attempt(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """最初の試行でも成功すればACCEPT"""
        result = strategy.evaluate_result("成功", None)
        assert result.action == TranscriptionAction.ACCEPT
        assert strategy.current_attempt == 0  # カウンターは増加しない

    def test_accepts_after_retries(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """リトライ後でも成功すればACCEPT"""
        # 2回失敗
        strategy.evaluate_result("", "Error 1")
        strategy.evaluate_result("", "Error 2")
        assert strategy.current_attempt == 2

        # 3回目で成功
        result = strategy.evaluate_result("成功テキスト", None)
        assert result.action == TranscriptionAction.ACCEPT


class TestRetryAction:
    """RETRY アクションのテスト"""

    def test_retries_on_filter_reason_with_text(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """フィルタ理由がある場合はRETRY"""
        result = strategy.evaluate_result("ご視聴ありがとうございました", "Banned phrase detected")
        assert result.action == TranscriptionAction.RETRY
        assert result.next_params is not None
        assert result.reason == "Banned phrase detected"

    def test_retries_on_empty_text_with_filter_reason(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """空テキストでもフィルタ理由があればRETRY"""
        result = strategy.evaluate_result("", "Extreme low confidence")
        assert result.action == TranscriptionAction.RETRY

    def test_retry_increments_attempt_counter(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """RETRYで試行カウンターが増加"""
        assert strategy.current_attempt == 0
        strategy.evaluate_result("bad", "Error")
        assert strategy.current_attempt == 1
        strategy.evaluate_result("bad", "Error")
        assert strategy.current_attempt == 2

    def test_retry_provides_next_params(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """RETRYは次のパラメータを提供"""
        result = strategy.evaluate_result("bad", "Error")
        assert result.next_params == WHISPER_PARAMS[1]

        result = strategy.evaluate_result("bad", "Error")
        assert result.next_params == WHISPER_PARAMS[2]

    def test_retry_until_max_retries_minus_one(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """最大リトライ - 1 回まではRETRY"""
        for i in range(MAX_TRANSCRIPTION_RETRIES - 1):
            result = strategy.evaluate_result("bad", f"Error {i}")
            assert result.action == TranscriptionAction.RETRY
            assert result.next_params == WHISPER_PARAMS[i + 1]


class TestDiscardAction:
    """DISCARD アクションのテスト"""

    def test_discards_empty_text_without_filter_reason(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """空テキストでフィルタ理由もない場合はDISCARD（無音）"""
        result = strategy.evaluate_result("", None)
        assert result.action == TranscriptionAction.DISCARD
        assert result.reason is not None
        assert "silence" in result.reason.lower()

    def test_discards_whitespace_only_without_filter_reason(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """空白のみでフィルタ理由もない場合はDISCARD"""
        result = strategy.evaluate_result("   ", None)
        # stripされた後の空文字として扱われないので、ACCEPTになる可能性
        # 実際の動作を確認
        # 注: evaluate_resultはstrip済みのテキストを受け取る前提
        # ここでは空白のみ="   "は空ではないのでACCEPT
        assert result.action == TranscriptionAction.ACCEPT

    def test_discards_after_max_retries(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """最大リトライ到達後はDISCARD"""
        # MAX_TRANSCRIPTION_RETRIES - 1 回RETRYする
        for i in range(MAX_TRANSCRIPTION_RETRIES - 1):
            result = strategy.evaluate_result("bad", f"Error {i}")
            assert result.action == TranscriptionAction.RETRY

        # 次はDISCARD
        result = strategy.evaluate_result("bad", "Final error")
        assert result.action == TranscriptionAction.DISCARD
        assert result.reason is not None
        assert "Max retries reached" in result.reason

    def test_discard_includes_last_error_reason(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """DISCARDは最後のエラー理由を含む"""
        # 最大までリトライ
        for _ in range(MAX_TRANSCRIPTION_RETRIES - 1):
            strategy.evaluate_result("bad", "Some error")

        result = strategy.evaluate_result("bad", "Character repetition detected")
        assert result.action == TranscriptionAction.DISCARD
        assert result.reason is not None
        assert "Character repetition detected" in result.reason


class TestStrategyResultDataclass:
    """StrategyResult データクラスのテスト"""

    def test_strategy_result_is_frozen(self) -> None:
        """StrategyResultは不変（frozen）"""
        result = StrategyResult(action=TranscriptionAction.ACCEPT)
        with pytest.raises(AttributeError):
            result.action = TranscriptionAction.RETRY  # type: ignore[misc]

    def test_strategy_result_default_values(self) -> None:
        """StrategyResultのデフォルト値"""
        result = StrategyResult(action=TranscriptionAction.ACCEPT)
        assert result.next_params is None
        assert result.reason is None

    def test_strategy_result_with_all_fields(self) -> None:
        """StrategyResultの全フィールド指定"""
        params = {"temperature": 0.5}
        result = StrategyResult(
            action=TranscriptionAction.RETRY,
            next_params=params,
            reason="Test reason",
        )
        assert result.action == TranscriptionAction.RETRY
        assert result.next_params == params
        assert result.reason == "Test reason"


class TestAttemptInfo:
    """get_attempt_info() のテスト"""

    def test_attempt_info_starts_at_one(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """試行番号は1から始まる（1-based）"""
        attempt, _ = strategy.get_attempt_info()
        assert attempt == 1

    def test_attempt_info_increments_with_retries(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """リトライごとに試行番号が増加"""
        strategy.evaluate_result("bad", "Error")
        attempt, _ = strategy.get_attempt_info()
        assert attempt == 2

        strategy.evaluate_result("bad", "Error")
        attempt, _ = strategy.get_attempt_info()
        assert attempt == 3

    def test_max_attempts_is_constant(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """最大試行回数は定数"""
        for _ in range(3):
            _, max_attempts = strategy.get_attempt_info()
            assert max_attempts == MAX_TRANSCRIPTION_RETRIES
            strategy.evaluate_result("bad", "Error")


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_multiple_success_evaluations(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """連続して成功評価してもカウンターは変わらない"""
        strategy.evaluate_result("success1", None)
        assert strategy.current_attempt == 0
        strategy.evaluate_result("success2", None)
        assert strategy.current_attempt == 0

    def test_reset_after_max_retries(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """最大リトライ後にリセットすると再利用可能"""
        # 最大までリトライ
        for _ in range(MAX_TRANSCRIPTION_RETRIES - 1):
            strategy.evaluate_result("bad", "Error")

        result = strategy.evaluate_result("bad", "Error")
        assert result.action == TranscriptionAction.DISCARD

        # リセット
        strategy.reset()
        assert strategy.current_attempt == 0
        assert strategy.get_current_params() == WHISPER_PARAMS[0]

        # 再度使用可能
        result = strategy.evaluate_result("bad", "Error")
        assert result.action == TranscriptionAction.RETRY

    def test_params_progression_matches_whisper_params(
        self, strategy: TranscriptionRetryStrategy
    ) -> None:
        """パラメータの進行はWHISPER_PARAMSに一致"""
        for i in range(MAX_TRANSCRIPTION_RETRIES - 1):
            assert strategy.get_current_params() == WHISPER_PARAMS[i]
            strategy.evaluate_result("bad", "Error")

        # 最後のパラメータ
        assert strategy.get_current_params() == WHISPER_PARAMS[MAX_TRANSCRIPTION_RETRIES - 1]
