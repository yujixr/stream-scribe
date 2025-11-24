#!/usr/bin/env python3
"""
Stream Scribe - Input Handler
プレゼンテーション層：ユーザー入力の処理
"""

import select
import sys
import time
from typing import Callable


class InputHandler:
    """
    CLI入力ハンドラー

    責務:
    - Ctrl-C (KeyboardInterrupt) の処理
    - Ctrl-D (EOF) の検出
    - 標準入力の監視
    """

    @staticmethod
    def wait_for_exit_signal(
        stop_condition: Callable[[], bool] | None = None,
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
        while True:
            # 終了条件のチェック
            if stop_condition is not None and stop_condition():
                return True

            # 標準入力の監視（Ctrl-D検出用）
            if sys.stdin.isatty():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    # 標準入力が読み取り可能
                    try:
                        if not sys.stdin.read(1):
                            # EOF (Ctrl-D)
                            raise EOFError
                    except EOFError:
                        raise
            else:
                time.sleep(0.1)
