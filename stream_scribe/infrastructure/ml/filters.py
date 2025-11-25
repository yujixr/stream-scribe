#!/usr/bin/env python3
"""
Stream Scribe - Hallucination Filter
幻覚フィルタリングを提供するモジュール（日本語最適化版）
"""

import re
from collections.abc import Sequence
from typing import Any


class HallucinationFilter:
    """
    幻覚フィルタリング専用クラス（日本語最適化版）

    Features:
    - ブラックリストフレーズ検出
    - 文字レベルの繰り返し検出
    - パターン繰り返し検出
    - トークンレベルの繰り返し検出

    日本語環境での音声認識幻覚に特化した検出ロジック
    """

    # 検出閾値（定数化）
    MIN_CHAR_REPETITION = 10  # 文字の最小連続回数
    MIN_SHORT_PATTERN_REPETITION = 5  # 短いパターンの最小繰り返し回数
    MIN_LONG_PATTERN_REPETITION = 3  # 長いパターンの最小繰り返し回数
    MIN_TOKEN_REPETITION = 5  # トークンの最小連続回数
    SHORT_PATTERN_MAX_LENGTH = 10  # 短いパターンの最大長
    LONG_PATTERN_MIN_LENGTH = 11  # 長いパターンの最小長
    LONG_PATTERN_MAX_LENGTH = 50  # 長いパターンの最大長
    PATTERN_SEARCH_START_POSITIONS = (
        50  # パターン探索の開始位置数（テキスト後半の繰り返しも検出）
    )
    REPETITION_RATIO_THRESHOLD = 0.5  # 繰り返しが占める割合の閾値

    # 極端に低いavg_logprobの閾値（明らかな幻覚のみフィルタ）
    EXTREME_LOW_LOGPROB_THRESHOLD = -1.7

    # 日本語句読点パターン（コンパイル済み）
    _JAPANESE_PUNCTUATION_PATTERN = re.compile(r"[。、！？\s]+")

    def __init__(self, banned_phrases: list[str]):
        """
        Args:
            banned_phrases: 禁止フレーズリスト
        """
        self.banned_phrases = banned_phrases

    def evaluate_transcription(
        self,
        text: str,
        avg_logprob: float | None = None,
        audio_duration: float | None = None,
    ) -> str | None:
        """
        文字起こし結果のテキストパターンを評価

        テキストベースの幻覚検出 + 極端に低い信頼度のフィルタを行う。

        Args:
            text: 文字起こしテキスト
            avg_logprob: セグメントの平均対数確率（オプション）
            audio_duration: 音声の長さ（秒）（オプション）

        Returns:
            str | None: 再試行が必要な場合は理由、そうでなければNone
        """
        # 空テキスト（無音など）- 再試行不要
        if not text or not text.strip():
            return None

        # 各検出メソッドを順次実行（早期リターン）
        if reason := self._check_banned_phrases(text):
            return reason

        if reason := self._check_character_repetition(text):
            return reason

        if reason := self._check_short_pattern_repetition(text):
            return reason

        if reason := self._check_long_pattern_repetition(text):
            return reason

        if reason := self._check_token_repetition(text):
            return reason

        if reason := self._check_contextless_greeting(
            text, avg_logprob, audio_duration
        ):
            return reason

        if reason := self._check_extreme_low_confidence(avg_logprob):
            return reason

        return None

    def _check_banned_phrases(self, text: str) -> str | None:
        """
        ブラックリストフレーズ検出

        Args:
            text: チェック対象のテキスト

        Returns:
            str | None: 破棄理由 or None
        """
        for phrase in self.banned_phrases:
            if phrase in text:
                return f"Banned phrase: '{phrase}'"
        return None

    def _check_character_repetition(self, text: str) -> str | None:
        """
        文字レベルの繰り返し検出（O(n)）

        同じ文字が連続して出現する場合を検出
        例: "ああああああああああ"

        Args:
            text: チェック対象のテキスト

        Returns:
            str | None: 破棄理由 or None
        """
        text_len = len(text)
        if text_len < self.MIN_CHAR_REPETITION:
            return None

        consecutive_count = 1
        prev_char = text[0]

        for char in text[1:]:
            if char == prev_char:
                consecutive_count += 1
                if consecutive_count >= self.MIN_CHAR_REPETITION:
                    return f"Character repetition: '{char}' x{consecutive_count}+"
            else:
                consecutive_count = 1
                prev_char = char

        return None

    def _check_short_pattern_repetition(self, text: str) -> str | None:
        """
        短いパターンの繰り返し検出（2-10文字）

        短いフレーズが繰り返される場合を検出
        例: "ピリピリピリピリピリ"

        Args:
            text: チェック対象のテキスト

        Returns:
            str | None: 破棄理由 or None
        """
        text_len = len(text)
        if text_len < 20:
            return None

        # パターン長を短いものから長いものへ
        for pattern_len in range(
            2, min(self.SHORT_PATTERN_MAX_LENGTH + 1, text_len // 3 + 1)
        ):
            # テキストの最初の数箇所をパターン候補として試行
            max_start = min(
                self.PATTERN_SEARCH_START_POSITIONS, text_len - pattern_len * 3 + 1
            )

            for start in range(max_start):
                pattern = text[start : start + pattern_len]

                # 空白やスペースだけのパターンはスキップ
                if not pattern.strip():
                    continue

                # パターンの出現回数をカウント
                count = text.count(pattern)
                if count >= self.MIN_SHORT_PATTERN_REPETITION:
                    # 繰り返しがテキスト全体の閾値以上を占めるかチェック
                    if (
                        pattern_len * count
                    ) >= text_len * self.REPETITION_RATIO_THRESHOLD:
                        return f"Pattern repetition: '{pattern[:30]}...' x{count}"

        return None

    def _check_long_pattern_repetition(self, text: str) -> str | None:
        """
        長いフレーズの繰り返し検出（11-50文字）

        より長いパターンの繰り返しを検出（計算量を抑える）
        例: "私たちの意味が好きな話題について、私たちの意味が好きな話題について、..."

        Args:
            text: チェック対象のテキスト

        Returns:
            str | None: 破棄理由 or None
        """
        text_len = len(text)
        if text_len < 60:
            return None

        max_pattern_len = min(self.LONG_PATTERN_MAX_LENGTH, text_len // 3)

        # 5文字刻みでチェック（計算量削減）
        for pattern_len in range(self.LONG_PATTERN_MIN_LENGTH, max_pattern_len + 1, 5):
            # 先頭から1箇所のみチェック
            pattern = text[0:pattern_len]
            if not pattern.strip():
                continue

            count = text.count(pattern)
            if count >= self.MIN_LONG_PATTERN_REPETITION:
                if (pattern_len * count) >= text_len * self.REPETITION_RATIO_THRESHOLD:
                    return f"Long phrase repetition: '{pattern[:30]}...' x{count}"

        return None

    def _check_token_repetition(self, text: str) -> str | None:
        """
        日本語トークンレベルの末尾繰り返し検出

        句読点で分割したトークンの末尾繰り返しを検出
        例: "はい。はい。はい。はい。はい。"

        Args:
            text: チェック対象のテキスト

        Returns:
            str | None: 破棄理由 or None
        """
        # 日本語句読点で分割
        tokens = self._JAPANESE_PUNCTUATION_PATTERN.split(text)

        # 空文字列を除去
        tokens = [t for t in tokens if t.strip()]

        # 末尾で同じトークンが連続する場合は幻覚
        if len(tokens) >= self.MIN_TOKEN_REPETITION:
            last_token = tokens[-1]
            if last_token and all(
                tokens[-i] == last_token
                for i in range(1, self.MIN_TOKEN_REPETITION + 1)
            ):
                return f"Token repetition at end: '{last_token}' x{self.MIN_TOKEN_REPETITION}+"

        return None

    def extract_metrics(
        self, segments: Sequence[dict[str, Any]] | None
    ) -> tuple[float | None, float | None, float | None]:
        """
        セグメントリストからWhisperメトリクスを抽出

        TranscriptionSegmentに記録して分析に使用する。

        Args:
            segments: Whisperの出力セグメント情報

        Returns:
            tuple: (avg_logprob, compression_ratio, no_speech_prob)
        """
        if not segments:
            return None, None, None

        # avg_logprob: 全セグメントの平均
        logprobs: list[float] = [
            s["avg_logprob"]
            for s in segments
            if "avg_logprob" in s and isinstance(s["avg_logprob"], (int, float))
        ]
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None

        # compression_ratio: 最大値（最も疑わしい値を採用）
        ratios: list[float] = [
            s["compression_ratio"]
            for s in segments
            if "compression_ratio" in s
            and isinstance(s["compression_ratio"], (int, float))
        ]
        compression_ratio = max(ratios) if ratios else None

        # no_speech_prob: 最大値（最も疑わしい値を採用）
        no_speech_probs: list[float] = [
            s["no_speech_prob"]
            for s in segments
            if "no_speech_prob" in s and isinstance(s["no_speech_prob"], (int, float))
        ]
        no_speech_prob = max(no_speech_probs) if no_speech_probs else None

        return avg_logprob, compression_ratio, no_speech_prob

    def _check_contextless_greeting(
        self,
        text: str,
        avg_logprob: float | None,
        audio_duration: float | None,
    ) -> str | None:
        """
        文脈なしで単独出現する挨拶を検出

        以下の条件で挨拶フレーズを幻覚と判定：
        1. テキストが挨拶フレーズのみ、または短文（閾値以下）
        2. かつ、以下のいずれか：
           - 低信頼度（avg_logprobが閾値以下）
           - 長尺音声中の短文（音声長が閾値以上かつテキストが短い）

        Args:
            text: チェック対象のテキスト
            avg_logprob: セグメントの平均対数確率
            audio_duration: 音声の長さ（秒）

        Returns:
            str | None: 破棄理由 or None
        """
        from stream_scribe.domain.constants import (
            CONTEXTLESS_GREETING_PHRASES,
            GREETING_LONG_AUDIO_THRESHOLD,
            GREETING_LOW_LOGPROB_THRESHOLD,
            GREETING_SHORT_TEXT_THRESHOLD,
        )

        # テキストから句読点・空白を除去して正規化
        normalized_text = self._JAPANESE_PUNCTUATION_PATTERN.sub("", text)

        # 挨拶フレーズが含まれているかチェック
        matched_greeting = None
        for phrase in CONTEXTLESS_GREETING_PHRASES:
            if phrase in normalized_text:
                matched_greeting = phrase
                break

        if not matched_greeting:
            return None

        # 条件1: テキストが短文かチェック
        if len(normalized_text) > GREETING_SHORT_TEXT_THRESHOLD:
            return None

        # 条件2: 低信頼度または長尺音声中の短文
        is_low_confidence = (
            avg_logprob is not None and avg_logprob < GREETING_LOW_LOGPROB_THRESHOLD
        )
        is_short_in_long_audio = (
            audio_duration is not None
            and audio_duration >= GREETING_LONG_AUDIO_THRESHOLD
            and len(normalized_text) <= GREETING_SHORT_TEXT_THRESHOLD
        )

        if is_low_confidence:
            return f"Contextless greeting with low confidence: '{matched_greeting}' (avg_logprob={avg_logprob:.2f})"
        elif is_short_in_long_audio:
            return f"Contextless greeting in long audio: '{matched_greeting}' (audio={audio_duration:.1f}s, text={len(normalized_text)} chars)"

        return None

    def _check_extreme_low_confidence(self, avg_logprob: float | None) -> str | None:
        """
        極端に低い信頼度を検出

        avg_logprobが閾値を下回る場合は明らかな幻覚と判定。
        WHISPER_PARAMSのlogprob_thresholdはトークン単位だが、
        これはセグメント全体の平均なので別の指標。

        Args:
            avg_logprob: セグメントの平均対数確率

        Returns:
            str | None: 破棄理由 or None
        """
        if avg_logprob is not None and avg_logprob < self.EXTREME_LOW_LOGPROB_THRESHOLD:
            return f"Extreme low confidence (avg_logprob={avg_logprob:.2f})"
        return None
