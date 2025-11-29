# Stream Scribe

リアルタイム音声文字起こし & 会話構造化ツール

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> [!IMPORTANT]
> 本ツールは [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) を使用しており、**Apple Silicon (M1/M2/M3/M4) 搭載の Mac 専用**です。Linux、Windows、Intel Mac では動作しません。

## 概要

Stream Scribeは、マイクからの音声をリアルタイムで文字起こしし、Claude AIで会話を構造化するPythonアプリケーションです。日本語音声に最適化されています。

## アーキテクチャ

Clean Architectureに基づく3層構造と、blinkerによるイベント駆動設計（Pub/Sub）を採用しています。

```
stream_scribe/
├── domain/          # ビジネスロジック（モデル、定数、イベント定義）
├── infrastructure/  # 外部連携
│   ├── audio/       # 音声入力、VAD
│   ├── ml/          # Whisper文字起こし
│   ├── ai/          # Claude要約
│   └── persistence/ # JSONエクスポート
└── presentation/    # CLI表示
```

### 処理フロー

1. **AudioStream** - マイクから32msチャンクで音声を取得し、3秒間のリングバッファでプリロールを保持。ヒステリシス制御により発話区間を安定検出
2. **VAD (Silero)** - 各チャンクに対してVAD推論を実行し、発話開始/終了を判定
3. **Transcriber (Whisper)** - 録音完了した音声をMLX Whisperで文字起こし。5段階リトライ戦略でハルシネーションに対応
4. **Summarizer (Claude)** - 文字起こし結果を蓄積し、600文字ごと（または60秒の無音）にClaude APIで会話を構造化
5. **JSON出力** - セッション終了時に全セグメントと構造化結果を保存

## 音声処理パイプライン

### Voice Activity Detection (VAD)

Silero VAD v5（ONNX）を使用した発話検出：

- **チャンクサイズ**：32ms（512サンプル @ 16kHz）
- **ヒステリシス制御**：開始閾値0.5 / 終了閾値0.3で誤検出を防止
- **プリロールバッファ**：3秒間のリングバッファで発話開始前の音声も保持
- **発話判定**：最小3チャンク（96ms）の音声で発話として認識し、相槌も拾う
- **無音判定**：最大25チャンク（800ms）の無音で発話終了、語尾の余韻を保護
- **状態リセット**：32秒間の無音でLSTM状態をリセットし、長時間稼働時の精度低下を防止

### Whisper文字起こし

MLX Whisper（`whisper-large-v3-turbo`）による高速文字起こし：

**5段階リトライ戦略でハルシネーションに対応：**

| フェーズ | temperature | compression_ratio | logprob | 特徴 |
|---------|------------|------------------|---------|------|
| 1. Standard | 0.0 | 2.4 | -1.0 | 標準パラメータ（決定論的） |
| 2. Anti-loop | 0.2 | 2.0 | -1.0 | 圧縮率を厳格化+軽い確率性 |
| 3. No-prompt | 0.4 | 2.2 | -1.0 | initial_promptを除去+中探索 |
| 4. Strict | 0.6 | 1.8 | -0.6 | 両方を厳格化+高探索 |
| 5. Final | 0.8 | 1.4 | -0.4 | 最終ゲート+最大探索 |

**ハルシネーションフィルタ：**
- 禁止フレーズ検出（YouTube関連コメント等）
- 文字・パターン・トークン繰り返し検出
- 文脈なし挨拶検出（低信頼度または長尺音声中の短文挨拶を除外）
- 低信頼度（avg_logprob < -1.7）フィルタリング

### 会話構造化 (Claude)

リアルタイムで会話をMarkdown形式の議事録に構造化：

- **モデル**：Claude Haiku 4.5（claude-haiku-4-5-20251001）
- **トリガー**：600文字蓄積ごと、または60秒の無音タイムアウトでAPIを呼び出し
- **ASR誤り訂正**：文脈に基づいて音声認識エラーを自動補正
- **トピックツリー**：完了/進行中のトピックを階層表示
- **コンテキスト維持**：前回の要約を引き継いで一貫性を保持

## インストール・使い方

```bash
# 依存ライブラリのインストール
brew install portaudio ffmpeg

git clone git@github.com:yujixr/stream-scribe.git
cd stream-scribe
uv sync

# 設定（会話構造化機能を使用する場合）
# config.toml を編集して Anthropic API キーを設定
# [app]
# anthropic_api_key = "sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 実行
python -m stream_scribe                 # デフォルトマイクで録音開始
python -m stream_scribe -l              # 利用可能なデバイス一覧
python -m stream_scribe -d 0            # デバイスID指定
python -m stream_scribe -f audio.mp3    # ファイルから文字起こし

# 終了方法
# Ctrl+C: 正常終了（残り処理を完了してからJSON保存）
# Ctrl+D: 高速終了（即座に終了、保存なし）
```

## 設定

プロジェクトルートの `config.toml` で各種パラメータを調整できます。

設定可能な項目の詳細は [stream_scribe/domain/settings.py](stream_scribe/domain/settings.py) を参照してください。

```toml
# 要約機能の無効化
[summary]
enabled = false

# VAD感度の調整
[vad.detection]
start_threshold = 0.5  # 低いほど敏感
end_threshold = 0.3    # 低いほど長く録音
```

## 技術スタック

| コンポーネント | 技術 |
|--------------|------|
| 音声入力 | sounddevice |
| VAD | Silero VAD (ONNX Runtime) |
| 文字起こし | MLX Whisper |
| 会話構造化 | Anthropic Claude API |

## ライセンス

MIT License
