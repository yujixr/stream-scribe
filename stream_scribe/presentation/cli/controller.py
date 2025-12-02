#!/usr/bin/env python3
"""
Stream Scribe - CLI Controller
CLIアプリケーションのコントローラー層：アプリケーションのライフサイクル管理
"""

import select
import sys
import time
import traceback
from collections.abc import Callable

from stream_scribe.domain import (
    MessageLevel,
    MessagePostedEvent,
    message_posted,
)
from stream_scribe.infrastructure.ai import create_llm_client
from stream_scribe.infrastructure.audio import (
    AudioSource,
    FileAudioSource,
    MicrophoneAudioSource,
)
from stream_scribe.infrastructure.config import load_settings
from stream_scribe.presentation.app import StreamScribeApp

from .view import CLIView


class CLIController:
    """
    CLIコントローラー

    責務:
    - AudioSourceの選択・生成
    - App/View初期化と配線
    - アプリケーションのライフサイクル管理（起動/終了）
    - 入力監視と終了シグナル処理
    """

    def __init__(self, device_id: int | None, file_path: str | None):
        """
        CLIControllerの初期化

        Args:
            device_id: オーディオデバイスID（Noneの場合はデフォルト）
            file_path: 音声ファイルパス（Noneの場合はマイク入力）
        """
        self.device_id = device_id
        self.file_path = file_path
        self.settings = load_settings()

        self.app: StreamScribeApp | None = None
        self.view: CLIView | None = None

    def run(self) -> None:
        """
        アプリケーションを実行

        Raises:
            SystemExit: エラー発生時
        """
        # 1. CLIView作成（Signal受信準備）
        self.view = CLIView(settings=self.settings)

        # 2. LLMクライアント初期化（設定検証済み）
        llm_client = (
            create_llm_client(settings=self.settings.summary)
            if self.settings.summary.enabled
            else None
        )

        # 3. バナー表示
        self.view.show_banner(llm_client)

        # 4. AudioSource生成
        audio_source = self._create_audio_source()

        # 5. StreamScribeApp作成
        self.app = StreamScribeApp(
            llm_client=llm_client, audio_source=audio_source, settings=self.settings
        )

        # 6. UI更新開始
        self.view.start(
            audio_stream=self.app.audio_stream,
            transcriber=self.app.transcriber,
            summarizer=self.app.summarizer,
        )

        # 7. 録音開始
        # 型の絞り込み: 初期化後、self.appは必ずStreamScribeAppインスタンスになる
        app = self.app
        assert app is not None

        app.start_recording()

        message_posted.send(
            None,
            event=MessagePostedEvent(
                message="🎙️  Listening... (Ctrl+C to stop, Ctrl+D for fast exit)\n",
                level=MessageLevel.SUCCESS,
            ),
        )

        is_file_mode = not audio_source.is_realtime

        try:
            # 終了シグナルを待機
            stop_condition = (
                (
                    lambda: not app.audio_stream.is_alive()
                    and not app.transcriber.is_transcribing
                )
                if is_file_mode
                else None
            )
            completed = self._wait_for_exit_signal(stop_condition)

            if completed:
                # ファイル処理完了
                message_posted.send(
                    None,
                    event=MessagePostedEvent(
                        message="\nFile processing completed.",
                        level=MessageLevel.SUCCESS,
                    ),
                )
                self._shutdown(graceful=True)
                return

        except KeyboardInterrupt:
            # Ctrl-C: 正常終了
            message_posted.send(
                None,
                event=MessagePostedEvent(
                    message="\nGoodbye!", level=MessageLevel.SUCCESS
                ),
            )
            self._shutdown(graceful=True)
            return

        except EOFError:
            # Ctrl-D: 高速終了
            message_posted.send(
                None,
                event=MessagePostedEvent(
                    message="\nFast exit (Ctrl-D)", level=MessageLevel.WARNING
                ),
            )
            self._shutdown(graceful=False)
            return

        except Exception as e:
            # エラー時は即座に終了
            message_posted.send(
                None,
                event=MessagePostedEvent(
                    message=f"\nError: {e}", level=MessageLevel.ERROR
                ),
            )
            traceback.print_exc()
            sys.exit(1)

    def _create_audio_source(self) -> AudioSource:
        """
        CLI引数に基づいてAudioSourceを生成

        Returns:
            AudioSource: ファイル入力またはマイク入力
        """
        if self.file_path:
            return FileAudioSource(
                core_settings=self.settings.core, file_path=self.file_path
            )
        else:
            return MicrophoneAudioSource(
                core_settings=self.settings.core,
                audio_settings=self.settings.audio,
                device_id=self.device_id,
            )

    def _wait_for_exit_signal(
        self, stop_condition: Callable[[], bool] | None = None
    ) -> bool:
        """
        終了シグナルを待機

        Args:
            stop_condition: 終了条件を判定する関数。Trueを返すとループ終了。

        Returns:
            bool: stop_conditionがTrueで終了した場合True、それ以外False

        Raises:
            KeyboardInterrupt: Ctrl-C が押された場合
            EOFError: Ctrl-D が押された場合
        """
        while stop_condition is None or not stop_condition():
            # 標準入力の監視（Ctrl-D検出用）
            if sys.stdin.isatty():
                ready, _, _ = select.select(
                    [sys.stdin], [], [], self.settings.app.input_poll_interval_sec
                )
                if ready:
                    try:
                        if not sys.stdin.read(1):
                            # EOF (Ctrl-D)
                            raise EOFError
                    except EOFError:
                        raise
            else:
                time.sleep(self.settings.app.input_poll_interval_sec)

        return True

    def _shutdown(self, graceful: bool) -> None:
        """
        アプリケーションの終了処理

        Args:
            graceful: Trueなら残り処理を完了させてから保存、Falseなら即座に終了
        """
        if self.view:
            self.view.stop()

        if self.app:
            self.app.shutdown(graceful=graceful)
