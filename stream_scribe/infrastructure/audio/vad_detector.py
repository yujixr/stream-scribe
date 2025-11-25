#!/usr/bin/env python3
"""
Stream Scribe - VAD Module
Silero VADによる音声検知を提供するモジュール
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from colorama import Fore, Style  # type: ignore[import-untyped]

from stream_scribe.domain.constants import MODEL_PATH, SAMPLE_RATE, SILERO_MODEL_URL


class VADDetector:
    """
    Silero VAD（ONNX）によるリアルタイム音声検知

    機能:
    - ステートフルLSTM（h, c状態管理）
    - 512サンプル（32ms）チャンク推論
    - モデル自動ダウンロード
    """

    def __init__(self, model_path: Path = MODEL_PATH, auto_download: bool = True):
        """
        Args:
            model_path: ONNXモデルのパス
            auto_download: モデルが存在しない場合に自動ダウンロードするか
        """
        # モデルの自動ダウンロード
        if auto_download and not model_path.exists():
            self._download_model(model_path)

        # ONNX Runtimeセッション（CPU最適化）
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        # LSTM内部状態の初期化（batch=1, hidden=128）
        self.reset_states()

        # サンプルレート（ONNXモデルは16000固定）
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    @staticmethod
    def _download_model(model_path: Path) -> None:
        """Silero VADモデルを自動ダウンロード"""
        print(f"{Fore.YELLOW}Downloading Silero VAD model...{Style.RESET_ALL}")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        import requests

        response = requests.get(SILERO_MODEL_URL, stream=True)
        response.raise_for_status()

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"{Fore.GREEN}Model downloaded: {model_path}{Style.RESET_ALL}")

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
