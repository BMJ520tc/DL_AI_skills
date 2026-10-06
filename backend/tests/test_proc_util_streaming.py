"""proc_util.run_command 的逐行回调（任务看板「实时进度」的底座）。

为什么需要：原先 run_command 是 `communicate()` 一次性取全部输出，长任务（训练/复现）
跑到一半界面什么都看不到——只能干等。加 on_line 后脚本的 stdout 逐行上报，
返回值仍是完整输出（语义不变）。
"""
from __future__ import annotations

import asyncio
import sys

from app.services import proc_util


def test_run_command_streams_lines_and_keeps_full_output():
    lines: list[str] = []
    code = "for i in range(3):\n    print(f'line {i}', flush=True)"
    rc, out = asyncio.run(
        proc_util.run_command([sys.executable, "-c", code], on_line=lines.append)
    )
    assert rc == 0
    assert lines == ["line 0", "line 1", "line 2"]  # 逐行回调，顺序与内容一致
    assert out.count("line ") == 3                   # 完整输出照常返回


def test_run_command_on_line_error_does_not_break_command():
    """上报回调自身抛异常，不能影响被跑的命令。"""
    def boom(_line: str) -> None:
        raise RuntimeError("进度上报炸了")

    rc, out = asyncio.run(
        proc_util.run_command([sys.executable, "-c", "print('ok')"], on_line=boom)
    )
    assert rc == 0 and "ok" in out
