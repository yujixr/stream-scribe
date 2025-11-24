#!/usr/bin/env python3
"""
Stream Scribe - VAD State Machine Module
VAD（音声活動検出）の状態遷移ロジックを管理するモジュール
"""

from enum import Enum, auto

from stream_scribe.domain.constants import (
    MAX_SILENCE_CHUNKS,
    MIN_SPEECH_CHUNKS,
    VAD_END_THRESHOLD,
    VAD_IDLE_RESET_CHUNKS,
    VAD_START_THRESHOLD,
)


class VadAction(Enum):
    """VAD状態遷移によって発生するアクション"""

    NONE = auto()
    START_RECORDING = auto()
    STOP_RECORDING = auto()
    RESET_VAD_MODEL = auto()


class VadStateMachine:
    """
    VAD状態遷移を管理するステートマシン

    ヒステリシス制御により発話区間を安定検出:
    - 録音開始: 高い閾値 (VAD_START_THRESHOLD) で誤検知防止
    - 録音終了: 低い閾値 (VAD_END_THRESHOLD) で語尾切れ防止
    """

    def __init__(self) -> None:
        self.is_recording = False
        self.speech_chunks = 0
        self.silence_chunks = 0
        self.idle_silence_chunks = 0

    def process(self, probability: float) -> VadAction:
        """
        VAD確率を処理し、状態遷移に基づくアクションを返す

        Args:
            probability: VADモデルから出力された音声確率 (0.0-1.0)

        Returns:
            VadAction: 実行すべきアクション
        """
        is_speech = self._evaluate_threshold(probability)
        if is_speech:
            return self._handle_speech()
        else:
            return self._handle_silence()

    def _evaluate_threshold(self, probability: float) -> bool:
        """ヒステリシス制御で閾値を切り替え"""
        if self.is_recording:
            # 録音中は低い閾値（語尾保護）
            return probability >= VAD_END_THRESHOLD
        # 待機中は高い閾値（ノイズ対策）
        return probability >= VAD_START_THRESHOLD

    def _handle_speech(self) -> VadAction:
        """音声検出時の処理"""
        self.silence_chunks = 0
        self.idle_silence_chunks = 0
        self.speech_chunks += 1

        if not self.is_recording and self.speech_chunks >= MIN_SPEECH_CHUNKS:
            self.is_recording = True
            return VadAction.START_RECORDING

        return VadAction.NONE

    def _handle_silence(self) -> VadAction:
        """無音検出時の処理"""
        self.speech_chunks = 0

        if self.is_recording:
            self.silence_chunks += 1
            if self.silence_chunks >= MAX_SILENCE_CHUNKS:
                self._reset_recording_state()
                return VadAction.STOP_RECORDING
        else:
            self.idle_silence_chunks += 1
            if self.idle_silence_chunks >= VAD_IDLE_RESET_CHUNKS:
                self.idle_silence_chunks = 0
                return VadAction.RESET_VAD_MODEL

        return VadAction.NONE

    def _reset_recording_state(self) -> None:
        """録音状態をリセット"""
        self.is_recording = False
        self.silence_chunks = 0
        self.speech_chunks = 0
