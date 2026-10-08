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
from typing import Callable, Optional

from app.config import DATA_DIR


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
    on_line: Optional[Callable[[str], None]] = None,
) -> tuple[int, str]:
    """运行命令并返回 (returncode, 合并输出)。

    - 超时：杀进程树并抛 `TimeoutError`（不区分大小写，调用方按超时处理）。
    - 被取消（任务 cancel）：杀进程树后把 CancelledError 继续抛给上层。
    - `on_line`：**逐行回调**（可选），用于把长任务（训练/复现）的 stdout 实时上报进度。
      返回值仍是完整输出，语义不变；回调自身抛异常不影响被跑的命令。
    - Windows 下命令恒为 list、不经 shell（既有约定）。
    """
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    # numba（dcor/scanpy 等依赖会用）默认把 JIT 缓存写进 site-packages，编译产物文件名极长；
    # 项目工作区路径本就深，容易突破 Windows MAX_PATH(260) → FileNotFoundError。
    # 统一把缓存指到项目内的短目录（可用 NUMBA_CACHE_DIR 覆盖）。
    run_env = {**os.environ, **(env or {})}
    # 逐行回调要求**子进程行缓冲**：Python 连到管道时 stdout 是**块缓冲**（要写满 8KB 才吐），
    # 于是 print 出来的行根本到不了这里 —— 长任务（训练/复现）的进度会一直停在「进行中」，
    # 超时/取消被杀后更是连一行日志都没留下（实测：训练一小时无任何 [epoch] 行）。设
    # PYTHONUNBUFFERED=1 让子进程行缓冲（只在本函数**用了 on_line** 时设，不影响其它调用）。
    if on_line is not None and not run_env.get("PYTHONUNBUFFERED"):
        run_env["PYTHONUNBUFFERED"] = "1"
    if os.name == "nt" and not run_env.get("NUMBA_CACHE_DIR"):
        short_cache = DATA_DIR / "numba_cache"
        try:
            short_cache.mkdir(parents=True, exist_ok=True)
            run_env["NUMBA_CACHE_DIR"] = str(short_cache)
        except OSError:
            pass
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env=run_env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        **kwargs,
    )

    async def _drain() -> str:
        if on_line is None:
            out, _ = await proc.communicate()
            return (out or b"").decode("utf-8", errors="replace")
        chunks: list[str] = []
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace")
            chunks.append(line)
            try:
                on_line(line.rstrip("\r\n"))
            except Exception:  # noqa: BLE001 —— 进度上报失败不能影响被跑的命令
                pass
        await proc.wait()
        return "".join(chunks)

    try:
        text = await asyncio.wait_for(_drain(), timeout=timeout)
    except asyncio.TimeoutError:
        await _kill_tree(proc)
        # **带消息**重抛：裸 TimeoutError 的 str() 是空串，会让上层落库的失败记录 error 字段一片
        # 空白、任务级又把它误报成「任务超时」（见 task_manager._execute）。这里给足原因。
        limit = f"{timeout:g}s" if timeout else "上限"
        raise TimeoutError(f"脚本执行超时（超过 {limit}）") from None
    except asyncio.CancelledError:
        await _kill_tree(proc)
        raise
    return proc.returncode or 0, text
