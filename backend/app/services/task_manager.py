"""任务管理：长任务状态机与后台串行队列（模块详细设计 2.1，D2）。

状态机: queued → running → success/failed; running → cancelled; failed → retry → queued。
职责边界: 任务表只存调度状态，运行结果/报错归 run_record（数据设计五.2）。
"""
import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from app.db.connection import get_connection

# 任务执行函数签名: (params: dict, task_id: str) -> None（结果由 handler 自行写 run_record）
Handler = Callable[[dict, str], Awaitable[None]]

_handlers: dict[str, Handler] = {}
_queue: Optional[asyncio.Queue] = None
_worker_task: Optional[asyncio.Task] = None
_running_tasks: dict[str, asyncio.Task] = {}

# 按任务类型的超时上限（秒），2.1 异常边界「超时→按任务类型超时上限终止」
TASK_TIMEOUTS: dict[str, float] = {
    "env_create": 3600,
    "agent_task": 900,
    "verify": 600,
    "analyze": 600,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_handler(task_type: str, handler: Handler) -> None:
    """注册某类任务的后台执行函数。"""
    _handlers[task_type] = handler


def create_task(task_type: str, project_id: Optional[str] = None, params: Optional[dict] = None) -> str:
    task_id = uuid.uuid4().hex
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO task(task_id, task_type, project_id, params, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'queued', ?, ?)",
            (task_id, task_type, project_id, json.dumps(params or {}, ensure_ascii=False), _now(), _now()),
        )
        conn.commit()
    finally:
        conn.close()
    if _queue is not None:
        _queue.put_nowait(task_id)
    return task_id


def get_task(task_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM task WHERE task_id = ?", (task_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def list_tasks(limit: int = 100, offset: int = 0) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM task ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def cancel_task(task_id: str) -> bool:
    """取消任务：queued 直接置 cancelled；running 取消其执行 Task（协作式，2.1 running→cancelled）。"""
    task = get_task(task_id)
    if task is None:
        return False
    if task["status"] == "queued":
        _set_status(task_id, "cancelled")
        return True
    if task["status"] == "running":
        t = _running_tasks.get(task_id)
        if t is not None:
            t.cancel()
            return True
    return False


def retry_task(task_id: str) -> bool:
    task = get_task(task_id)
    if task is None or task["status"] != "failed":
        return False
    _set_status(task_id, "queued")
    if _queue is not None:
        _queue.put_nowait(task_id)
    return True


def update_progress(task_id: str, progress: dict) -> None:
    """更新任务进度（供 handler 上报，2.1 进度展示）。"""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE task SET progress = ?, updated_at = ? WHERE task_id = ?",
            (json.dumps(progress, ensure_ascii=False), _now(), task_id),
        )
        conn.commit()
    finally:
        conn.close()


def _set_status(task_id: str, status: str, error: Optional[str] = None) -> None:
    conn = get_connection()
    try:
        if error is not None:
            conn.execute(
                "UPDATE task SET status = ?, error = ?, updated_at = ? WHERE task_id = ?",
                (status, error, _now(), task_id),
            )
        else:
            conn.execute(
                "UPDATE task SET status = ?, updated_at = ? WHERE task_id = ?",
                (status, _now(), task_id),
            )
        conn.commit()
    finally:
        conn.close()


async def start() -> None:
    """启动后台 worker（由 main lifespan 调用）。"""
    global _queue, _worker_task
    if _queue is None:
        _queue = asyncio.Queue()
    if _worker_task is None:
        _worker_task = asyncio.create_task(_worker())


async def stop() -> None:
    global _worker_task
    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None


async def _worker() -> None:
    while True:
        task_id = await _queue.get()
        try:
            await _execute(task_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            # _execute 内部已捕获并落库，此处兜底防止 worker 崩溃
            pass
        finally:
            _queue.task_done()


async def _execute(task_id: str) -> None:
    task = get_task(task_id)
    if task is None or task["status"] == "cancelled":
        return
    _set_status(task_id, "running")

    handler = _handlers.get(task["task_type"])
    if handler is None:
        _set_status(task_id, "failed", error=f"no handler for task_type={task['task_type']}")
        return

    try:
        params = json.loads(task["params"] or "{}")
        timeout = TASK_TIMEOUTS.get(task["task_type"], 600)
        t = asyncio.create_task(handler(params, task_id))
        _running_tasks[task_id] = t
        try:
            await asyncio.wait_for(t, timeout=timeout)
        except asyncio.TimeoutError:
            _set_status(task_id, "failed", error=f"timeout after {timeout}s")
            t.cancel()
            return
        except asyncio.CancelledError:
            # 被 cancel_task 取消（协作式），不重新抛出以免传播到 worker
            _set_status(task_id, "cancelled")
            return
        finally:
            _running_tasks.pop(task_id, None)
        _set_status(task_id, "success")
    except asyncio.CancelledError:
        _set_status(task_id, "cancelled")
        return
    except Exception as e:  # noqa: BLE001 —— 任务级异常需落库并继续
        _set_status(task_id, "failed", error=str(e))
