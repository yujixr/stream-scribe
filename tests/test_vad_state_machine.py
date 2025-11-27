"""VadStateMachineのテスト"""

import sys
from unittest.mock import MagicMock

import pytest

# sounddevice が利用できない環境（Linux CI等）ではモックする
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = MagicMock()

from stream_scribe.domain.constants import (
    MAX_SILENCE_CHUNKS,
    MIN_SPEECH_CHUNKS,
    VAD_END_THRESHOLD,
    VAD_IDLE_RESET_CHUNKS,
    VAD_START_THRESHOLD,
)
from stream_scribe.infrastructure.audio.vad_state_machine import (
    VadAction,
    VadStateMachine,
)


@pytest.fixture
def state_machine() -> VadStateMachine:
    """新しいVadStateMachineインスタンスを作成"""
    return VadStateMachine()


class TestInitialState:
    """初期状態のテスト"""

    def test_initial_state_not_recording(self, state_machine: VadStateMachine) -> None:
        """初期状態では録音していない"""
        assert state_machine.is_recording is False

    def test_initial_counters_zero(self, state_machine: VadStateMachine) -> None:
        """初期状態ではカウンターがゼロ"""
        assert state_machine.speech_chunks == 0
        assert state_machine.silence_chunks == 0
        assert state_machine.idle_silence_chunks == 0


class TestHysteresisThresholds:
    """ヒステリシス閾値の動作テスト"""

    def test_uses_high_threshold_when_not_recording(
        self, state_machine: VadStateMachine
    ) -> None:
        """録音していない時は高い閾値を使用"""
        # 開始閾値未満は音声としてカウントされない
        state_machine.process(VAD_START_THRESHOLD - 0.01)
        assert state_machine.speech_chunks == 0

        # 開始閾値以上は音声としてカウント
        state_machine.process(VAD_START_THRESHOLD)
        assert state_machine.speech_chunks == 1

    def test_uses_low_threshold_when_recording(
        self, state_machine: VadStateMachine
    ) -> None:
        """録音中は低い閾値を使用"""
        # まず録音を開始
        for _ in range(MIN_SPEECH_CHUNKS):
            state_machine.process(VAD_START_THRESHOLD + 0.1)
        assert state_machine.is_recording is True

        # 終了閾値と開始閾値の間は、録音中なら音声としてカウント
        prob_between = (VAD_END_THRESHOLD + VAD_START_THRESHOLD) / 2
        state_machine.process(prob_between)
        # silence_chunksは増加しないはず
        assert state_machine.silence_chunks == 0

    def test_below_end_threshold_counts_silence_when_recording(
        self, state_machine: VadStateMachine
    ) -> None:
        """録音中、終了閾値未満は無音としてカウント"""
        # 録音開始
        for _ in range(MIN_SPEECH_CHUNKS):
            state_machine.process(VAD_START_THRESHOLD + 0.1)
        assert state_machine.is_recording is True

        # 終了閾値未満
        state_machine.process(VAD_END_THRESHOLD - 0.01)
        assert state_machine.silence_chunks == 1


class TestRecordingStart:
    """録音開始トリガーのテスト"""

    def test_starts_recording_after_min_speech_chunks(
        self, state_machine: VadStateMachine
    ) -> None:
        """MIN_SPEECH_CHUNKS後に録音開始"""
        high_prob = VAD_START_THRESHOLD + 0.1

        # MIN_SPEECH_CHUNKS - 1 チャンクを処理
        for _ in range(MIN_SPEECH_CHUNKS - 1):
            action = state_machine.process(high_prob)
            assert action == VadAction.NONE
            assert state_machine.is_recording is False

        # MIN_SPEECH_CHUNKS目で開始をトリガー
        action = state_machine.process(high_prob)
        assert action == VadAction.START_RECORDING
        assert state_machine.is_recording is True

    def test_resets_speech_counter_on_silence(
        self, state_machine: VadStateMachine
    ) -> None:
        """無音で音声カウンターがリセットされる"""
        high_prob = VAD_START_THRESHOLD + 0.1
        low_prob = VAD_START_THRESHOLD - 0.1

        # 音声チャンクを蓄積
        for _ in range(MIN_SPEECH_CHUNKS - 1):
            state_machine.process(high_prob)
        assert state_machine.speech_chunks == MIN_SPEECH_CHUNKS - 1

        # 1回の無音でカウンターリセット
        state_machine.process(low_prob)
        assert state_machine.speech_chunks == 0
        assert state_machine.is_recording is False

    def test_subsequent_speech_returns_none(
        self, state_machine: VadStateMachine
    ) -> None:
        """録音開始後の音声はNONEを返す"""
        high_prob = VAD_START_THRESHOLD + 0.1

        # 録音開始
        for _ in range(MIN_SPEECH_CHUNKS):
            state_machine.process(high_prob)

        # 追加の音声はNONEを返すべき
        action = state_machine.process(high_prob)
        assert action == VadAction.NONE
        assert state_machine.is_recording is True


class TestRecordingStop:
    """録音終了トリガーのテスト"""

    def test_stops_recording_after_max_silence_chunks(
        self, state_machine: VadStateMachine
    ) -> None:
        """MAX_SILENCE_CHUNKS後に録音終了"""
        high_prob = VAD_START_THRESHOLD + 0.1
        low_prob = VAD_END_THRESHOLD - 0.1

        # 録音開始
        for _ in range(MIN_SPEECH_CHUNKS):
            state_machine.process(high_prob)
        assert state_machine.is_recording is True

        # MAX_SILENCE_CHUNKS - 1 無音チャンクを処理
        for _ in range(MAX_SILENCE_CHUNKS - 1):
            action = state_machine.process(low_prob)
            assert action == VadAction.NONE
            assert state_machine.is_recording is True

        # MAX_SILENCE_CHUNKS目で停止をトリガー
        action = state_machine.process(low_prob)
        assert action == VadAction.STOP_RECORDING
        assert state_machine.is_recording is False

    def test_speech_during_silence_resets_silence_counter(
        self, state_machine: VadStateMachine
    ) -> None:
        """無音中の音声で無音カウンターがリセットされる"""
        high_prob = VAD_START_THRESHOLD + 0.1
        medium_prob = VAD_END_THRESHOLD + 0.05  # 終了閾値以上
        low_prob = VAD_END_THRESHOLD - 0.1

        # 録音開始
        for _ in range(MIN_SPEECH_CHUNKS):
            state_machine.process(high_prob)

        # 無音を蓄積
        for _ in range(MAX_SILENCE_CHUNKS - 2):
            state_machine.process(low_prob)
        assert state_machine.silence_chunks == MAX_SILENCE_CHUNKS - 2

        # 音声が無音を中断
        state_machine.process(medium_prob)
        assert state_machine.silence_chunks == 0
        assert state_machine.is_recording is True

    def test_state_reset_after_stop(self, state_machine: VadStateMachine) -> None:
        """停止後に状態がリセットされる"""
        high_prob = VAD_START_THRESHOLD + 0.1
        low_prob = VAD_END_THRESHOLD - 0.1

        # 録音開始→停止
        for _ in range(MIN_SPEECH_CHUNKS):
            state_machine.process(high_prob)
        for _ in range(MAX_SILENCE_CHUNKS):
            state_machine.process(low_prob)

        # 状態がリセットされているはず
        assert state_machine.is_recording is False
        assert state_machine.silence_chunks == 0
        assert state_machine.speech_chunks == 0


class TestVadModelReset:
    """VADモデルリセットトリガーのテスト"""

    def test_triggers_reset_after_idle_silence(
        self, state_machine: VadStateMachine
    ) -> None:
        """待機中の無音後にリセットをトリガー"""
        low_prob = VAD_START_THRESHOLD - 0.1

        # 録音していない状態で待機中の無音を蓄積
        for _ in range(VAD_IDLE_RESET_CHUNKS - 1):
            action = state_machine.process(low_prob)
            assert action == VadAction.NONE

        # リセットをトリガーすべき
        action = state_machine.process(low_prob)
        assert action == VadAction.RESET_VAD_MODEL
        assert state_machine.idle_silence_chunks == 0

    def test_no_reset_during_recording(self, state_machine: VadStateMachine) -> None:
        """録音中はリセットしない"""
        high_prob = VAD_START_THRESHOLD + 0.1
        low_prob = VAD_END_THRESHOLD + 0.05  # 終了閾値以上で録音継続

        # 録音開始
        for _ in range(MIN_SPEECH_CHUNKS):
            state_machine.process(high_prob)

        # 録音中はidle_silence_chunksが増加しないはず
        initial_idle = state_machine.idle_silence_chunks
        for _ in range(100):
            state_machine.process(low_prob)

        assert state_machine.idle_silence_chunks == initial_idle

    def test_speech_resets_idle_counter(self, state_machine: VadStateMachine) -> None:
        """音声で待機中カウンターがリセットされる"""
        low_prob = VAD_START_THRESHOLD - 0.1
        high_prob = VAD_START_THRESHOLD + 0.1

        # 待機中の無音を蓄積
        for _ in range(VAD_IDLE_RESET_CHUNKS // 2):
            state_machine.process(low_prob)
        assert state_machine.idle_silence_chunks > 0

        # 音声でリセットされるべき
        state_machine.process(high_prob)
        assert state_machine.idle_silence_chunks == 0


class TestComplexScenarios:
    """複雑な実世界シナリオのテスト"""

    def test_intermittent_speech_pattern(self, state_machine: VadStateMachine) -> None:
        """短い休止を含む発話パターンをシミュレート"""
        high_prob = VAD_START_THRESHOLD + 0.2
        medium_prob = VAD_END_THRESHOLD + 0.1
        low_prob = VAD_END_THRESHOLD - 0.1

        # 発話開始
        for _ in range(MIN_SPEECH_CHUNKS):
            state_machine.process(high_prob)
        assert state_machine.is_recording is True

        # 短い休止を含む発話（停止には不十分）
        for _ in range(3):
            # 発話
            for _ in range(5):
                state_machine.process(medium_prob)
            # 短い休止
            for _ in range(MAX_SILENCE_CHUNKS // 3):
                state_machine.process(low_prob)

        # まだ録音中のはず
        assert state_machine.is_recording is True

    def test_noise_spike_ignored(self, state_machine: VadStateMachine) -> None:
        """単発のノイズスパイクは録音を開始しない"""
        high_prob = VAD_START_THRESHOLD + 0.3
        low_prob = VAD_START_THRESHOLD - 0.2

        # 単発の高確率スパイク
        state_machine.process(high_prob)
        assert state_machine.speech_chunks == 1
        assert state_machine.is_recording is False

        # 続いて無音
        state_machine.process(low_prob)
        assert state_machine.speech_chunks == 0
        assert state_machine.is_recording is False

    def test_full_conversation_cycle(self, state_machine: VadStateMachine) -> None:
        """開始→発話→停止の完全なサイクルをシミュレート"""
        high_prob = VAD_START_THRESHOLD + 0.2
        low_prob = VAD_END_THRESHOLD - 0.1

        actions: list[VadAction] = []

        # フェーズ1: 録音開始
        for _ in range(MIN_SPEECH_CHUNKS):
            actions.append(state_machine.process(high_prob))
        assert VadAction.START_RECORDING in actions

        # フェーズ2: 発話継続
        actions.clear()
        for _ in range(20):
            actions.append(state_machine.process(high_prob))
        assert all(a == VadAction.NONE for a in actions)
        assert state_machine.is_recording is True

        # フェーズ3: 録音停止
        actions.clear()
        for _ in range(MAX_SILENCE_CHUNKS):
            actions.append(state_machine.process(low_prob))
        assert VadAction.STOP_RECORDING in actions
        assert state_machine.is_recording is False

        # フェーズ4: 長い待機期間でリセットをトリガー
        actions.clear()
        for _ in range(VAD_IDLE_RESET_CHUNKS):
            actions.append(state_machine.process(low_prob))
        assert VadAction.RESET_VAD_MODEL in actions


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_exact_threshold_values(self, state_machine: VadStateMachine) -> None:
        """閾値境界での動作をテスト"""
        # 開始閾値ちょうどは音声としてカウント
        state_machine.process(VAD_START_THRESHOLD)
        assert state_machine.speech_chunks == 1

        # 録音開始
        for _ in range(MIN_SPEECH_CHUNKS - 1):
            state_machine.process(VAD_START_THRESHOLD)

        # 録音中、終了閾値ちょうどは音声としてカウント
        state_machine.process(VAD_END_THRESHOLD)
        assert state_machine.silence_chunks == 0

    def test_probability_zero(self, state_machine: VadStateMachine) -> None:
        """確率ゼロでのテスト"""
        action = state_machine.process(0.0)
        assert action == VadAction.NONE
        assert state_machine.speech_chunks == 0
        assert state_machine.idle_silence_chunks == 1

    def test_probability_one(self, state_machine: VadStateMachine) -> None:
        """確率最大でのテスト"""
        for _ in range(MIN_SPEECH_CHUNKS):
            state_machine.process(1.0)
        assert state_machine.is_recording is True

    def test_rapid_state_transitions(self, state_machine: VadStateMachine) -> None:
        """音声と無音の急速な遷移をテスト"""
        high_prob = VAD_START_THRESHOLD + 0.2
        low_prob = VAD_START_THRESHOLD - 0.2

        # 高低を交互に
        for _ in range(50):
            state_machine.process(high_prob)
            state_machine.process(low_prob)

        # 交互パターンでは録音開始しないはず
        assert state_machine.is_recording is False
