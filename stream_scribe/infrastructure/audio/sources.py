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

from stream_scribe.domain import (
    AudioSettings,
    CoreSettings,
    MessageLevel,
    MessagePostedEvent,
    message_posted,
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
    def start(self) -> None:
        """音声ソースを開始"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """音声ソースを停止"""
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

    def __init__(
        self,
        core_settings: CoreSettings,
        audio_settings: AudioSettings,
        device_id: int | None = None,
    ) -> None:
        """
        Args:
            core_settings: コア設定（サンプルレート、チャンクサイズ）
            audio_settings: オーディオ設定
            device_id: 使用するオーディオデバイスのID（Noneの場合はデフォルトデバイス）
        """
        self.core_settings = core_settings
        self.audio_settings = audio_settings
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
            message_posted.send(
                None,
                event=MessagePostedEvent(
                    message=f"Audio Error: {status}", level=MessageLevel.ERROR
                ),
            )

        # モノラル変換してキューに追加
        audio = indata[:, 0].copy()
        self._queue.put(audio)

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        マイクからの音声データをチャンク単位で yield する

        Yields:
            np.ndarray: chunk_sizeサンプルのfloat32音声データ
        """
        chunk_size = self.core_settings.chunk_size
        while self._running:
            try:
                # キューから音声データを取得（タイムアウト付き）
                audio_block = self._queue.get(
                    timeout=self.audio_settings.queue_get_timeout_sec
                )
                self._chunk_buffer = np.append(self._chunk_buffer, audio_block)

                # chunk_size単位でyield
                while len(self._chunk_buffer) >= chunk_size:
                    chunk = self._chunk_buffer[:chunk_size]
                    self._chunk_buffer = self._chunk_buffer[chunk_size:]
                    yield chunk

            except queue.Empty:
                continue

    def start(self) -> None:
        """マイク入力ストリームを開始"""
        self._running = True
        self._stream = sd.InputStream(
            device=self._device_id,
            samplerate=self.core_settings.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=int(
                self.core_settings.sample_rate * self.audio_settings.block_sec
            ),
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
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
        core_settings: CoreSettings,
        file_path: str,
        realtime_simulation: bool = False,
    ) -> None:
        """
        Args:
            core_settings: コア設定（サンプルレート、チャンクサイズ）
            file_path: 音声ファイルのパス (mp3/wav 等)
            realtime_simulation: True の場合、実時間に合わせて sleep を入れる
        """
        self.core_settings = core_settings
        self.file_path = file_path
        self.realtime_simulation = realtime_simulation
        self._audio_data: np.ndarray | None = None
        self._duration: float = 0.0

    def _load_audio(self) -> np.ndarray:
        """
        音声ファイルを読み込み、設定されたサンプルレートのモノラルに変換

        Returns:
            np.ndarray: 設定されたサンプルレートのモノラルfloat32音声データ
        """
        # soundfileで読み込み（ネイティブサンプルレート）
        audio_data, original_sr = sf.read(self.file_path, dtype="float32")

        # ステレオの場合はモノラルに変換
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # リサンプリングが必要な場合
        if original_sr != self.core_settings.sample_rate:
            # 線形補間によるリサンプリング
            original_length = len(audio_data)
            target_length = int(
                original_length * self.core_settings.sample_rate / original_sr
            )
            audio_data = np.interp(
                np.linspace(0, original_length - 1, target_length),
                np.arange(original_length),
                audio_data,
            ).astype(np.float32)

        # 音声の長さを記録
        self._duration = len(audio_data) / self.core_settings.sample_rate

        return np.asarray(audio_data, dtype=np.float32)

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        音声ファイルからのデータをチャンク単位で yield する

        Yields:
            np.ndarray: chunk_sizeサンプルのfloat32音声データ
        """
        if self._audio_data is None:
            return

        chunk_size = self.core_settings.chunk_size
        # チャンクごとの時間（秒）
        chunk_duration = chunk_size / self.core_settings.sample_rate

        # chunk_sizeごとにスライスしてyield
        total_samples = len(self._audio_data)
        for i in range(0, total_samples, chunk_size):
            chunk = self._audio_data[i : i + chunk_size]

            # 最後のチャンクが短い場合はゼロパディング
            if len(chunk) < chunk_size:
                chunk = np.pad(
                    chunk, (0, chunk_size - len(chunk)), mode="constant"
                ).astype(np.float32)

            yield chunk

            # リアルタイムシミュレーションの場合は待機
            if self.realtime_simulation:
                time.sleep(chunk_duration)

    def start(self) -> None:
        """音声ファイルを読み込み"""
        self._audio_data = self._load_audio()

    def stop(self) -> None:
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
