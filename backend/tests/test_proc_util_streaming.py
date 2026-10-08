"""proc_util.run_command 的逐行回调（任务看板「实时进度」的底座）。

为什么需要：原先 run_command 是 `communicate()` 一次性取全部输出，长任务（训练/复现）
跑到一半界面什么都看不到——只能干等。加 on_line 后脚本的 stdout 逐行上报，
返回值仍是完整输出（语义不变）。
"""
from __future__ import annotations

import asyncio
import sys
import time

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


def test_on_line_makes_child_unbuffered_so_lines_arrive_live():
    """用 on_line 时必须让子进程**行缓冲**：Python 连管道默认块缓冲，不 flush 的 print 要写满 8KB
    才吐——长任务（训练）的进度会一直空着，超时被杀后连日志都没有。这里用**时序**断言：
    首行必须在进程退出**之前**就到（块缓冲下首行会拖到进程结束才一起到）。"""
    t0 = time.time()
    stamps: list[tuple[str, float]] = []

    def cb(line: str) -> None:
        stamps.append((line, time.time()))

    code = "import time\nprint('early')\ntime.sleep(1.2)\nprint('late')"
    rc, out = asyncio.run(proc_util.run_command([sys.executable, "-c", code], on_line=cb))
    total = time.time() - t0
    assert rc == 0 and "early" in out and "late" in out
    early = next(t for line, t in stamps if line == "early")
    assert early - t0 < total / 2, (
        f"首行 {early - t0:.2f}s 才到、总耗时 {total:.2f}s —— 子进程没被设成行缓冲（块缓冲）")


def test_script_timeout_error_carries_message():
    """脚本级超时的 TimeoutError 必须**带原因**：裸 TimeoutError 的 str() 是空串，会让失败记录
    一片空白、任务级又把它误报成「任务超时」（见 task_manager._execute 的修复）。"""
    try:
        asyncio.run(proc_util.run_command([sys.executable, "-c", "import time; time.sleep(30)"],
                                          timeout=0.4))
    except TimeoutError as e:      # asyncio.TimeoutError 在 3.11+ 即内置 TimeoutError
        assert "超时" in str(e) and "0.4" in str(e)
    else:                          # pragma: no cover
        raise AssertionError("应当抛 TimeoutError")
