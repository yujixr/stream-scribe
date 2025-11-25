#!/usr/bin/env python3
"""
Stream Scribe - Configuration
設定定数を管理するモジュール
"""

from pathlib import Path

# ========================================
# Audio Settings
# ========================================
SAMPLE_RATE = 16000  # Whisper & VADの標準サンプルレート
CHUNK_MS = 32  # VAD推論チャンク (32ms)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 512 samples

# ========================================
# Voice Activity Detection (VAD) - Japanese Optimization
# ========================================
VAD_START_THRESHOLD = 0.5  # 録音開始閾値（高め=誤検知防止）
VAD_END_THRESHOLD = 0.3  # 録音終了閾値（低め=語尾切れ防止）
MIN_SPEECH_CHUNKS = 3  # 最小発話チャンク数（32ms*3=96ms、相槌を拾う）
MAX_SILENCE_CHUNKS = 25  # 最大無音チャンク数（32ms*25=800ms、語尾の余韻を保護）
VAD_IDLE_RESET_CHUNKS = 1000  # 待機中の無音リセット閾値（32ms*1000=約32秒）

# ========================================
# Pre-roll Buffer
# ========================================
PREROLL_SEC = 3.0  # プリロールバッファ (3.0秒)
PREROLL_CHUNKS = int(PREROLL_SEC * 1000 / CHUNK_MS)  # ~94 chunks

# ========================================
# Silero VAD Model
# ========================================
SILERO_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx"
)
MODEL_DIR = Path.home() / ".cache" / "silero-vad"
MODEL_PATH = MODEL_DIR / "silero_vad.onnx"

# ========================================
# Whisper Model
# ========================================
# whisper-large-v3-turbo: OpenAI公式の高速モデル
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
WHISPER_INITIAL_PROMPT = "句読点を含む正確な日本語で書き起こします。"

# Whisperパラメータ（5段階の戦略的再試行 - Japanese Optimized）
# 戦略: 「標準」→「ループ対策」→「バイアス（プロンプト）排除」→「厳格化」→「最終フィルタ」
WHISPER_PARAMS = [
    # --- Phase 1: Standard (標準) ---
    # 1回目: 最も標準的な設定。プロンプトにより文脈と句読点を整える。
    # ほとんどの正常な発話はここで通過するはず。
    {
        "language": "ja",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "initial_prompt": WHISPER_INITIAL_PROMPT,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    },
    # --- Phase 2: Anti-Loop (ループ対策) ---
    # 2回目: プロンプトは維持しつつ、繰り返しループ（「ご視聴...」等）を圧縮率で弾く。
    # Logprobは変えず、異常な繰り返しだけをターゲットにする。
    {
        "language": "ja",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "initial_prompt": WHISPER_INITIAL_PROMPT,
        "compression_ratio_threshold": 2.0,  # 標準(2.4)より厳格化
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    },
    # --- Phase 3: No-Prompt (バイアス排除) ---
    # 3回目: プロンプト起因の強力な幻覚（「チャンネル登録」等）を物理的に遮断。
    # ここが最も重要な転換点。プロンプトを消すことで、音声のみに集中させる。
    {
        "language": "ja",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "initial_prompt": None,  # プロンプト完全削除
        "compression_ratio_threshold": 2.2,  # プロンプトがない分、圧縮率は少し緩め直す
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    },
    # --- Phase 4: Strict (厳格化) ---
    # 4回目: プロンプトなしでも出てくる「自信のない出力」をカット。
    # ノイズを無理やり言語化しているケースを弾く。
    {
        "language": "ja",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "compression_ratio_threshold": 1.8,
        "logprob_threshold": -0.6,  # かなり厳しめ
        "no_speech_threshold": 0.5,  # 無音判定を敏感に
    },
    # --- Phase 5: Drop / Final Gate (最終手段) ---
    # 5回目: ほぼ確実な音声以外は全て捨てる。
    # これでも弾かれるなら、それは「言葉」ではない可能性が高い。
    {
        "language": "ja",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "compression_ratio_threshold": 1.4,  # 繰り返しを一切許容しない
        "logprob_threshold": -0.4,  # 高い確信度が必要
        "no_speech_threshold": 0.4,  # 積極的に無音とみなす
    },
]

# 再試行の最大回数
MAX_TRANSCRIPTION_RETRIES = len(WHISPER_PARAMS)

# ========================================
# Real-time Summarizer
# ========================================
SUMMARY_TRIGGER_THRESHOLD = (
    400  # 要約トリガーの文字数閾値（文脈をまとめるため少し長めに）
)
SUMMARY_MODEL = "claude-sonnet-4-5-20250929"  # 使用するClaudeモデル
SUMMARY_MAX_TOKENS = 4096  # Claude APIの最大トークン数

# ========================================
# Audio Stream
# ========================================
AUDIO_BLOCK_SEC = 0.1  # sounddeviceのブロックサイズ（100ms）

# ========================================
# Shutdown Timeouts
# ========================================
TRANSCRIBER_SHUTDOWN_TIMEOUT_SEC = 10.0  # Transcriberスレッド停止のタイムアウト
SUMMARIZER_SHUTDOWN_TIMEOUT_SEC = 2.0  # Summarizerスレッド停止のタイムアウト
FAST_SHUTDOWN_TIMEOUT_SEC = 1.0  # 高速終了時のタイムアウト

# ========================================
# Display
# ========================================
MAX_ERROR_DETAIL_LENGTH = 200  # エラー詳細の最大表示文字数
MAX_TRACEBACK_LENGTH = 500  # トレースバックの最大表示文字数

# ========================================
# Hallucination Filter - Banned Phrases
# ========================================
BANNED_PHRASES = [
    WHISPER_INITIAL_PROMPT,  # プロンプト自体が出力されるのを防止
    "書き起こします",  # プロンプト類似の幻覚
    # YouTube/動画系の幻覚
    "ご視聴ありがとうございました",
    "チャンネル登録",
    "高評価",
    "Thank you for watching",
    "次回予告",
    # 字幕・ニュース系の幻覚
    "MBC News",
    "Subtitles by",
    "字幕",
    # Whisper v3特有の幻覚（無音時）
    "ブーブー",
    "ブーバー",
    "BGM",
    "♪",
]

# ========================================
# Hallucination Filter - Contextless Greetings
# ========================================
# 文脈なしで単独出現すると幻覚の可能性が高い挨拶フレーズ
CONTEXTLESS_GREETING_PHRASES = [
    "おやすみなさい",
    "おやすみ",
    "おはようございます",
    "おはよう",
    "こんにちは",
    "こんばんは",
    "さようなら",
    "ありがとうございました",
    "ありがとうございます",
    "ありがとう",
    "お疲れ様でした",
    "お疲れ様です",
    "お疲れさまでした",
    "お疲れさまです",
    "いただきます",
    "ごちそうさまでした",
    "ごちそうさま",
    "行ってきます",
    "行ってらっしゃい",
    "ただいま",
    "おかえりなさい",
    "おかえり",
]

# 挨拶フィルタ用の閾値
GREETING_LOW_LOGPROB_THRESHOLD = -0.8  # 挨拶判定用の信頼度閾値
GREETING_LONG_AUDIO_THRESHOLD = 5.0  # 長尺音声と判定する秒数
GREETING_SHORT_TEXT_THRESHOLD = 15  # 短文と判定する文字数
