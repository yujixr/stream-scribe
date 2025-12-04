#!/usr/bin/env python3
"""
Stream Scribe - Settings Schema
設定のスキーマ定義（Pydanticモデル）
"""

from enum import StrEnum
from pathlib import Path

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings
from typing_extensions import Self

# ========================================
# Private Constants (used multiple times)
# ========================================
_WHISPER_INITIAL_PROMPT = "句読点を含む正確な日本語で書き起こします。"


# ========================================
# Helper Functions
# ========================================
def _default_whisper_params() -> list["WhisperParamsSettings"]:
    """Whisper再試行パラメータのデフォルト値を生成"""
    return [
        # フェーズ1: 標準
        WhisperParamsSettings(
            language="ja",
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=_WHISPER_INITIAL_PROMPT,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        ),
        # フェーズ2: ループ対策+軽探索
        WhisperParamsSettings(
            language="ja",
            temperature=0.2,
            condition_on_previous_text=False,
            initial_prompt=_WHISPER_INITIAL_PROMPT,
            compression_ratio_threshold=2.0,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        ),
        # フェーズ3: バイアス排除+中探索
        WhisperParamsSettings(
            language="ja",
            temperature=0.4,
            condition_on_previous_text=False,
            initial_prompt=None,
            compression_ratio_threshold=2.2,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        ),
        # フェーズ4: 厳格化+高探索
        WhisperParamsSettings(
            language="ja",
            temperature=0.6,
            condition_on_previous_text=False,
            initial_prompt=None,
            compression_ratio_threshold=1.8,
            logprob_threshold=-0.6,
            no_speech_threshold=0.5,
        ),
        # フェーズ5: 最終ゲート+最大探索
        WhisperParamsSettings(
            language="ja",
            temperature=0.8,
            condition_on_previous_text=False,
            initial_prompt=None,
            compression_ratio_threshold=1.4,
            logprob_threshold=-0.4,
            no_speech_threshold=0.4,
        ),
    ]


# ========================================
# Core Configuration
# ========================================
class CoreSettings(BaseSettings):
    """基礎パラメータ（全コンポーネントで共有）"""

    sample_rate: int = Field(
        default=16000,
        description="サンプルレート（Hz） - WhisperとVADの標準サンプルレート",
    )
    chunk_ms: int = Field(
        default=32,
        description="VAD推論チャンク（ミリ秒） - 32ミリ秒単位で音声を処理",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunk_size(self) -> int:
        """チャンクサイズ（サンプル数）"""
        return int(self.sample_rate * self.chunk_ms / 1000)


# ========================================
# Audio Configuration
# ========================================
class AudioSettings(BaseSettings):
    """Audio入力設定"""

    block_sec: float = Field(
        default=0.1,
        description="sounddeviceのブロックサイズ（秒）",
    )
    queue_get_timeout_sec: float = Field(
        default=0.5,
        description="キュー取得タイムアウト（秒）",
    )


# ========================================
# VAD Configuration
# ========================================
class VADModelSettings(BaseSettings):
    """Silero VADモデル設定"""

    url: str = Field(
        default="https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx",
        description="モデルダウンロードURL",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_dir(self) -> Path:
        """モデル保存ディレクトリ"""
        return Path.home() / ".cache" / "silero-vad"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_path(self) -> Path:
        """モデルファイルパス"""
        return self.model_dir / "silero_vad.onnx"


class VADDetectionSettings(BaseSettings):
    """VAD音声区間検出設定"""

    start_threshold: float = Field(
        default=0.5,
        description="録音開始閾値（高め=誤検知防止）",
    )
    end_threshold: float = Field(
        default=0.3,
        description="録音終了閾値（低め=語尾切れ防止）",
    )
    min_speech_chunks: int = Field(
        default=3,
        description="最小発話チャンク数（32ms*3=96ms、相槌を拾う）",
    )
    max_silence_chunks: int = Field(
        default=25,
        description="最大無音チャンク数（32ms*25=800ms、語尾の余韻を保護）",
    )
    idle_reset_chunks: int = Field(
        default=1000,
        description="待機中の無音リセット閾値（32ms*1000=約32秒）",
    )
    preroll_sec: float = Field(
        default=3.0,
        description="プリロールバッファ（秒） - 発話開始前の音声を含める時間（3.0秒、約94チャンク）",
    )
    stream_shutdown_timeout_sec: float = Field(
        default=2.0,
        description="AudioStreamスレッド停止タイムアウト（秒）",
    )

    def preroll_chunks(self, chunk_ms: int) -> int:
        """プリロールバッファのチャンク数を計算"""
        return int(self.preroll_sec * 1000 / chunk_ms)


class VADSettings(BaseSettings):
    """VAD全体設定"""

    model: VADModelSettings = Field(default_factory=VADModelSettings)
    detection: VADDetectionSettings = Field(default_factory=VADDetectionSettings)


# ========================================
# Whisper Configuration
# ========================================
class WhisperParamsSettings(BaseSettings):
    """Whisperパラメータ（単一フェーズ）"""

    language: str = Field(description="言語コード")
    temperature: float = Field(
        description="温度パラメータ - 0.0で決定論的、高いほど確率的に多様な出力"
    )
    condition_on_previous_text: bool = Field(
        description="前のテキストを条件とするか - Falseでループ防止"
    )
    initial_prompt: str | None = Field(description="初期プロンプト")
    compression_ratio_threshold: float = Field(
        description="圧縮率閾値 - 低いほど繰り返しループを厳格に検出"
    )
    logprob_threshold: float = Field(
        description="対数確率閾値 - 高いほど高い確信度を要求（自信のない出力を排除）"
    )
    no_speech_threshold: float = Field(
        description="無音判定閾値 - 低いほど積極的に無音とみなす"
    )


class WhisperSettings(BaseSettings):
    """Whisperモデル設定"""

    model: str = Field(
        default="mlx-community/whisper-large-v3-turbo",
        description="Whisperモデル名 - OpenAI公式の高速モデル",
    )
    shutdown_timeout_sec: float = Field(
        default=10.0,
        description="書き起こしスレッド停止タイムアウト（秒）",
    )
    params: list[WhisperParamsSettings] = Field(
        default_factory=_default_whisper_params,
        description="再試行パラメータリスト - 5段階の戦略的再試行（標準→ループ対策+軽探索→バイアス排除+中探索→厳格化+高探索→最終ゲート+最大探索）",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_transcription_retries(self) -> int:
        """最大再試行回数"""
        return len(self.params)


# ========================================
# Hallucination Filter Configuration
# ========================================
class HallucinationFilterSettings(BaseSettings):
    """幻覚フィルタ設定"""

    banned_phrases: list[str] = Field(
        default=[
            _WHISPER_INITIAL_PROMPT,
            "書き起こします",
            "ご視聴ありがとうございました",
            "チャンネル登録",
            "高評価",
            "Thank you for watching",
            "次回予告",
            "MBC News",
            "Subtitles by",
            "字幕",
            "ブーブー",
            "ブーバー",
            "♪",
        ],
        description="禁止フレーズリスト - プロンプト出力、YouTube/動画系、字幕・ニュース系、Whisper特有の幻覚を検出",
    )
    contextless_greeting_phrases: list[str] = Field(
        default=[
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
        ],
        description="文脈なし挨拶フレーズリスト - 短い音声で出現した場合、幻覚の可能性が高い挨拶",
    )

    # 繰り返し検出閾値
    min_char_repetition: int = Field(
        default=10,
        description="文字の最小連続回数",
    )
    min_short_pattern_repetition: int = Field(
        default=5,
        description="短いパターンの最小繰り返し回数",
    )
    min_long_pattern_repetition: int = Field(
        default=3,
        description="長いパターンの最小繰り返し回数",
    )
    min_token_repetition: int = Field(
        default=5,
        description="トークンの最小連続回数",
    )
    short_pattern_max_length: int = Field(
        default=10,
        description="短いパターンの最大長",
    )
    long_pattern_min_length: int = Field(
        default=11,
        description="長いパターンの最小長",
    )
    long_pattern_max_length: int = Field(
        default=50,
        description="長いパターンの最大長",
    )
    pattern_search_start_positions: int = Field(
        default=50,
        description="パターン探索の開始位置数",
    )
    repetition_ratio_threshold: float = Field(
        default=0.5,
        description="繰り返しが占める割合の閾値",
    )
    extreme_low_logprob_threshold: float = Field(
        default=-1.7,
        description="極端に低いavg_logprobの閾値",
    )

    # 挨拶フィルタ閾値
    low_logprob_threshold: float = Field(
        default=-0.8,
        description="挨拶判定用の信頼度閾値",
    )
    long_audio_threshold: float = Field(
        default=5.0,
        description="長尺音声と判定する秒数",
    )
    short_text_threshold: int = Field(
        default=15,
        description="短文と判定する文字数",
    )


# ========================================
# Summary Configuration
# ========================================
class LLMBackend(StrEnum):
    """LLMバックエンドの種類"""

    CLAUDE = "claude"
    VLLM = "vllm"


class SummarySettings(BaseSettings):
    """リアルタイム要約設定"""

    enabled: bool = Field(
        default=True,
        description="要約機能の有効/無効",
    )
    backend: LLMBackend = Field(
        default=LLMBackend.CLAUDE,
        description="LLMバックエンド - claude または vllm",
    )
    # Claude固有設定
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic APIキー - backend='claude' の場合に必要。",
    )
    claude_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude APIモデル名",
    )
    # vLLM固有設定
    vllm_base_url: str | None = Field(
        default=None,
        description="vLLMサーバのベースURL（例: http://localhost:8000/v1）",
    )
    vllm_model: str = Field(
        default="Qwen/Qwen3-30B-A3B",
        description="vLLMモデル名（例: Qwen/Qwen3-30B-A3B）",
    )
    vllm_api_key: str | None = Field(
        default=None,
        description="vLLM APIキー（サーバが認証を要求する場合のみ必要）",
    )
    # 共通設定
    max_tokens: int = Field(
        default=4096,
        description="最大トークン数 - 全LLM共通の要約用最大トークン数",
    )
    trigger_threshold: int = Field(
        default=600,
        description="要約トリガーの文字数閾値",
    )
    silence_timeout_sec: float = Field(
        default=60.0,
        description="無音タイムアウト（最後のセグメント追加から経過秒数）",
    )
    queue_get_timeout_sec: float = Field(
        default=1.0,
        description="キュー取得タイムアウト（秒）",
    )
    shutdown_timeout_sec: float = Field(
        default=2.0,
        description="要約スレッド停止タイムアウト（秒）",
    )

    @model_validator(mode="after")
    def validate_backend_config(self) -> Self:
        """バックエンド固有の必須設定を検証"""
        if not self.enabled:
            return self

        if self.backend == LLMBackend.CLAUDE:
            if not self.anthropic_api_key:
                raise ValueError(
                    "summary.anthropic_api_key is required when backend='claude'"
                )
        elif self.backend == LLMBackend.VLLM:
            if not self.vllm_base_url:
                raise ValueError(
                    "summary.vllm_base_url is required when backend='vllm'"
                )

        return self


# ========================================
# Application Configuration
# ========================================
class AppSettings(BaseSettings):
    """アプリケーション全体設定"""

    save_json: bool = Field(
        default=True,
        description="セッション終了時にJSON形式で保存するかどうか",
    )
    fast_shutdown_timeout_sec: float = Field(
        default=1.0,
        description="高速終了時のタイムアウト（秒）",
    )
    status_update_manager_shutdown_timeout_sec: float = Field(
        default=1.0,
        description="ステータス更新マネージャー停止タイムアウト（秒）",
    )
    status_update_interval_sec: float = Field(
        default=0.1,
        description="ステータス更新間隔（秒）",
    )
    input_poll_interval_sec: float = Field(
        default=0.1,
        description="入力ポーリング間隔（秒）",
    )
    transcription_progress_poll_interval_sec: float = Field(
        default=0.5,
        description="文字起こし進捗ポーリング間隔（秒）",
    )
    max_error_detail_length: int = Field(
        default=200,
        description="エラー詳細の最大表示文字数",
    )
    max_traceback_length: int = Field(
        default=500,
        description="トレースバックの最大表示文字数",
    )


# ========================================
# Main Settings Class
# ========================================
class Settings(BaseSettings):
    """
    Stream Scribe全体設定

    設定の読み込み優先順位（後勝ち）:
    1. デフォルト値（各Settingsクラス内）
    2. config.toml（プロジェクトルート）
    """

    core: CoreSettings = Field(default_factory=CoreSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    hallucination: HallucinationFilterSettings = Field(
        default_factory=HallucinationFilterSettings
    )
    summary: SummarySettings = Field(default_factory=SummarySettings)
    app: AppSettings = Field(default_factory=AppSettings)
