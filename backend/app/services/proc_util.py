"""子进程执行与「进程树」终止（任务取消/超时用）。

为什么需要：原先一律 `asyncio.to_thread(subprocess.run, ...)`，取消只作用于 await 点，
线程与子进程会继续跑——任务已被取消（或超时判 failed）后，僵尸进程仍在写同一个环境目录，
此时用户 retry 会与它并发（同一 env_dir 上两套 pip install），把环境写坏。

本模块用 asyncio 子进程 + 取消/超时即杀**进程树**（Windows `taskkill /T`、POSIX 进程组），
让「取消」真的停下工作，handler 协程随即结束。
"""
import asyncio
import os
import signal
import subprocess
from typing import Optional


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """杀掉子进程及其后代（Windows 用 taskkill /T；POSIX 杀进程组）。"""
    if proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    finally:
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass


async def run_command(
    cmd: list[str],
    *,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    env: Optional[dict] = None,
) -> tuple[int, str]:
    """运行命令并返回 (returncode, 合并输出)。

    - 超时：杀进程树并抛 `TimeoutError`（不区分大小写，调用方按超时处理）。
    - 被取消（任务 cancel）：杀进程树后把 CancelledError 继续抛给上层。
    - Windows 下命令恒为 list、不经 shell（既有约定）。
    """
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        **kwargs,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await _kill_tree(proc)
        raise
    except asyncio.CancelledError:
        await _kill_tree(proc)
        raise
    return proc.returncode or 0, (out or b"").decode("utf-8", errors="replace")
