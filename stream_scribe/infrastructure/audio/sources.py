#!/usr/bin/env python3
"""
Stream Scribe - Audio Sources Module
音声入力の抽象化とアダプタパターンを提供するモジュール
"""

from __future__ import annotations

import queue
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]

from stream_scribe.domain.constants import (
    AUDIO_BLOCK_SEC,
    AUDIO_QUEUE_GET_TIMEOUT_SEC,
    CHUNK_SIZE,
    SAMPLE_RATE,
)


@dataclass(frozen=True)
class AudioDevice:
    """オーディオデバイス情報"""

    id: int
    name: str
    max_input_channels: int
    is_default: bool = False


class AudioSource(ABC):
    """
    音声入力の抽象基底クラス

    音声データのチャンク（np.ndarray, float32）をジェネレータとして
    yield するインターフェースを定義する。
    """

    @abstractmethod
    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        音声データのチャンクをジェネレータとして yield する

        Yields:
            np.ndarray: CHUNK_SIZE（512）サンプルのfloat32音声データ
        """
        pass

    @abstractmethod
    def __enter__(self) -> AudioSource:
        """コンテキストマネージャのエントリポイント"""
        pass

    @abstractmethod
    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """コンテキストマネージャのクリーンアップ"""
        pass

    @property
    @abstractmethod
    def is_realtime(self) -> bool:
        """リアルタイム入力ソースかどうか"""
        pass


class MicrophoneAudioSource(AudioSource):
    """
    マイク入力からの音声ソース

    sounddevice のコールバック方式を同期的なジェネレータ形式に変換し、
    AudioStream がプル型でデータを取得できるようにする。
    """

    @staticmethod
    def list_devices() -> list[AudioDevice]:
        """
        利用可能な入力デバイス一覧を取得する

        Returns:
            list[AudioDevice]: 入力可能なオーディオデバイスのリスト
        """
        raw_devices = sd.query_devices()

        # デバイスが存在しない場合
        if not isinstance(raw_devices, sd.DeviceList) or len(raw_devices) == 0:
            return []

        default_input_device_id: int | None = sd.default.device[0]

        return [
            AudioDevice(
                id=device_id,
                name=device_info["name"],
                max_input_channels=device_info["max_input_channels"],
                is_default=(device_id == default_input_device_id),
            )
            for device_id, device_info in enumerate(raw_devices)
            if device_info["max_input_channels"] > 0
        ]

    def __init__(self, device_id: int | None = None) -> None:
        """
        Args:
            device_id: 使用するオーディオデバイスのID（Noneの場合はデフォルトデバイス）
        """
        self._device_id = device_id
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._running: bool = False
        self._chunk_buffer: np.ndarray = np.array([], dtype=np.float32)

    def _audio_callback(
        self,
        indata: np.ndarray,
        _frames: int,
        _time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddeviceコールバック（非同期）"""
        if status:
            import sys

            from colorama import Fore, Style  # type: ignore[import-untyped]

            print(
                f"\n{Fore.RED}Audio Error: {status}{Style.RESET_ALL}", file=sys.stderr
            )

        # モノラル変換してキューに追加
        audio = indata[:, 0].copy()
        self._queue.put(audio)

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        マイクからの音声データをチャンク単位で yield する

        Yields:
            np.ndarray: CHUNK_SIZE（512）サンプルのfloat32音声データ
        """
        while self._running:
            try:
                # キューから音声データを取得（タイムアウト付き）
                audio_block = self._queue.get(timeout=AUDIO_QUEUE_GET_TIMEOUT_SEC)
                self._chunk_buffer = np.append(self._chunk_buffer, audio_block)

                # CHUNK_SIZE単位でyield
                while len(self._chunk_buffer) >= CHUNK_SIZE:
                    chunk = self._chunk_buffer[:CHUNK_SIZE]
                    self._chunk_buffer = self._chunk_buffer[CHUNK_SIZE:]
                    yield chunk

            except queue.Empty:
                continue

    def __enter__(self) -> MicrophoneAudioSource:
        """マイク入力ストリームを開始"""
        self._running = True
        self._stream = sd.InputStream(
            device=self._device_id,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=int(SAMPLE_RATE * AUDIO_BLOCK_SEC),
            callback=self._audio_callback,
        )
        self._stream.start()
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """マイク入力ストリームを停止"""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def is_realtime(self) -> bool:
        """マイク入力はリアルタイムソース"""
        return True


class FileAudioSource(AudioSource):
    """
    音声ファイルからの音声ソース

    mp3/wav 等の音声ファイルを読み込み、VAD/Whisper の要件に合わせて
    16,000Hz モノラルにリサンプリング/変換する。
    """

    def __init__(
        self,
        file_path: str,
        realtime_simulation: bool = False,
    ) -> None:
        """
        Args:
            file_path: 音声ファイルのパス (mp3/wav 等)
            realtime_simulation: True の場合、実時間に合わせて sleep を入れる
        """
        self.file_path = file_path
        self.realtime_simulation = realtime_simulation
        self._audio_data: np.ndarray | None = None
        self._duration: float = 0.0

    def _load_audio(self) -> np.ndarray:
        """
        音声ファイルを読み込み、16kHzモノラルに変換

        Returns:
            np.ndarray: 16kHzモノラルのfloat32音声データ
        """
        # soundfileで読み込み（ネイティブサンプルレート）
        audio_data, original_sr = sf.read(self.file_path, dtype="float32")

        # ステレオの場合はモノラルに変換
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # リサンプリングが必要な場合
        if original_sr != SAMPLE_RATE:
            # 線形補間によるリサンプリング
            original_length = len(audio_data)
            target_length = int(original_length * SAMPLE_RATE / original_sr)
            audio_data = np.interp(
                np.linspace(0, original_length - 1, target_length),
                np.arange(original_length),
                audio_data,
            ).astype(np.float32)

        # 音声の長さを記録
        self._duration = len(audio_data) / SAMPLE_RATE

        return np.asarray(audio_data, dtype=np.float32)

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        音声ファイルからのデータをチャンク単位で yield する

        Yields:
            np.ndarray: CHUNK_SIZE（512）サンプルのfloat32音声データ
        """
        if self._audio_data is None:
            return

        # チャンクごとの時間（秒）
        chunk_duration = CHUNK_SIZE / SAMPLE_RATE

        # CHUNK_SIZEごとにスライスしてyield
        total_samples = len(self._audio_data)
        for i in range(0, total_samples, CHUNK_SIZE):
            chunk = self._audio_data[i : i + CHUNK_SIZE]

            # 最後のチャンクが短い場合はゼロパディング
            if len(chunk) < CHUNK_SIZE:
                chunk = np.pad(
                    chunk, (0, CHUNK_SIZE - len(chunk)), mode="constant"
                ).astype(np.float32)

            yield chunk

            # リアルタイムシミュレーションの場合は待機
            if self.realtime_simulation:
                time.sleep(chunk_duration)

    def __enter__(self) -> FileAudioSource:
        """音声ファイルを読み込み"""
        self._audio_data = self._load_audio()
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """リソースのクリーンアップ"""
        self._audio_data = None

    @property
    def is_realtime(self) -> bool:
        """ファイル入力は非リアルタイムソース"""
        return False

    @property
    def duration(self) -> float:
        """音声ファイルの長さ（秒）"""
        return self._duration
