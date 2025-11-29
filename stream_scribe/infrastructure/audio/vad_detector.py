#!/usr/bin/env python3
"""
Stream Scribe - VAD Module
Silero VADによる音声検知を提供するモジュール
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import requests

from stream_scribe.domain import (
    CoreSettings,
    MessageLevel,
    MessagePostedEvent,
    VADModelSettings,
    message_posted,
)


class VADDetector:
    """
    Silero VAD（ONNX）によるリアルタイム音声検知

    機能:
    - ステートフルLSTM（h, c状態管理）
    - 512サンプル（32ms）チャンク推論
    - モデル自動ダウンロード
    """

    def __init__(
        self,
        core_settings: CoreSettings,
        vad_model_settings: VADModelSettings,
        auto_download: bool = True,
    ):
        """
        Args:
            core_settings: コア設定（サンプルレート）
            vad_model_settings: VADモデル設定
            auto_download: モデルが存在しない場合に自動ダウンロードするか
        """
        self.core_settings = core_settings
        self.vad_model_settings = vad_model_settings

        message_posted.send(
            None,
            event=MessagePostedEvent(
                message="Initializing VAD...", level=MessageLevel.INFO
            ),
        )

        # モデルの自動ダウンロード
        if auto_download and not vad_model_settings.model_path.exists():
            self._download_model(vad_model_settings.model_path, vad_model_settings.url)

        # ONNX Runtimeセッション（CPU最適化）
        self.session = ort.InferenceSession(
            str(vad_model_settings.model_path), providers=["CPUExecutionProvider"]
        )

        # LSTM内部状態の初期化（batch=1, hidden=128）
        self.reset_states()

        # サンプルレート（ONNXモデルは16000固定）
        self._sr = np.array(core_settings.sample_rate, dtype=np.int64)

        message_posted.send(
            None,
            event=MessagePostedEvent(
                message="VAD ready.\n", level=MessageLevel.SUCCESS
            ),
        )

    @staticmethod
    def _download_model(model_path: Path, url: str) -> None:
        """Silero VADモデルを自動ダウンロード"""
        message_posted.send(
            None,
            event=MessagePostedEvent(
                message="Downloading Silero VAD model...", level=MessageLevel.INFO
            ),
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        message_posted.send(
            None,
            event=MessagePostedEvent(
                message=f"Model downloaded: {model_path}", level=MessageLevel.SUCCESS
            ),
        )

    def reset_states(self) -> None:
        """LSTM状態をリセット"""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def __call__(self, audio_chunk: np.ndarray) -> float:
        """
        VAD推論を実行

        Args:
            audio_chunk: (512,) float32 audio data

        Returns:
            probability: 音声確率（0.0-1.0）
        """
        # 入力のリシェイプ（batch, samples）
        audio_input = audio_chunk.reshape(1, -1)

        # ONNX推論
        ort_inputs = {
            "input": audio_input,
            "state": self.state,
            "sr": self._sr,
        }

        output, self.state = self.session.run(None, ort_inputs)

        # 確率値を返す（output shape: (1, 1)）
        probability = float(output.squeeze().item())
        return probability
