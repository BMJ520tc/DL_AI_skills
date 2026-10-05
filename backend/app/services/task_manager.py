"""任务管理：长任务状态机与后台串行队列（模块详细设计 2.1，D2）。

状态机: queued → running → success/failed; running → cancelled; failed → retry → queued。
职责边界: 任务表只存调度状态，运行结果/报错归 run_record（数据设计五.2）。
"""
import asyncio
import contextlib
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
    "extract_addresses": 1800,
    "preprocess": 1800,
    "baseline": 3600,
    "align": 1800,
    "compare": 900,
    "pdf_parse": 1800,
    "extract_items": 1800,
    "reproduce": 7200,
    "conclusion": 900,
    "decompose": 3900,  # agent 重试总预算 3600s + 余量（不能与单次 agent 超时相等，否则任务先被取消）
    "decompose_trace": 900,  # 脚本 600s + 余量
    "decompose_verify": 2400,  # 脚本 1800s + 余量
    "module_ingest": 600,
    "network_train": 3600,  # 训练脚本 1800s + 余量
    "multi_model": 900,     # 多模型综合分析（含分歧归因 agent）
    "paper_distill": 3600,  # 论文蒸馏（逐篇 agent 起草，多篇可能耗时）
    "knowledge_distill": 600,  # 任务后蒸馏（单次 agent 起草）
    "network_autotune": 3600,  # 自动调参（逐候选训练，每个训练上限见 TRAIN_TIMEOUT_S）
    "assistant_chat": 900,  # 前端 AI 助手对话（单次只读 agent 会话）
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_handler(task_type: str, handler: Handler) -> None:
    """注册某类任务的后台执行函数。"""
    _handlers[task_type] = handler


# 任务终态钩子（success/failed 后触发）：供「任务后蒸馏入库」等旁路使用（模块详细设计 8.3）。
_finish_hooks: list[Callable[[str, str], Awaitable[None]]] = []


def register_on_finish(hook: Callable[[str, str], Awaitable[None]]) -> None:
    """注册终态钩子：签名 (task_id, status) -> Awaitable[None]，status ∈ {success, failed}。"""
    _finish_hooks.append(hook)


def _schedule_finish_hooks(task_id: str, status: str) -> None:
    """调度终态钩子（fire-and-forget）：不阻塞 worker 处理下一个任务，钩子自身异常只吞掉。

    蒸馏是旁路——它慢（要跑 agent），绝不能拖住任务队列，也不能因它失败而影响主任务状态。
    """
    for hook in _finish_hooks:
        async def _run(h=hook):
            try:
                await h(task_id, status)
            except Exception:  # noqa: BLE001 —— 旁路失败不影响主流程
                pass
        try:
            asyncio.create_task(_run())
        except RuntimeError:  # 无运行事件循环（同步调用场景）→ 跳过
            pass


def create_task(task_type: str, project_id: Optional[str] = None, params: Optional[dict] = None) -> str:
    """创建任务并入队。**相同任务（类型+项目+参数）已在排队或执行中则复用**，不重复入队。

    重复入队的代价很高（同一项目跑两遍 agent 拆解 / 两遍 pip 安装），
    而前端重复点击、脚本重跑都很容易触发；已结束（success/failed/cancelled）的不算重复。
    """
    params_json = json.dumps(params or {}, ensure_ascii=False, sort_keys=True)
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT task_id FROM task WHERE task_type = ? AND IFNULL(project_id, '') = IFNULL(?, '') "
            "AND params = ? AND status IN ('queued', 'running') ORDER BY created_at DESC LIMIT 1",
            (task_type, project_id, params_json),
        ).fetchone()
        if existing is not None:
            return existing["task_id"]
        task_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO task(task_id, task_type, project_id, params, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'queued', ?, ?)",
            (task_id, task_type, project_id, params_json, _now(), _now()),
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
    stale = reconcile_stale_tasks()
    if stale:
        print(f"[task] 启动收敛：{stale} 个残留 running 任务已置为 failed（服务重启中断，可重试）")
    if _queue is None:
        _queue = asyncio.Queue()
    if _worker_task is None:
        _worker_task = asyncio.create_task(_worker())


def reconcile_stale_tasks() -> int:
    """启动时收敛残留 running 任务，返回被收敛的条数。

    状态机（2.1）只有 queued → running → success/failed/cancelled，没有「进程崩溃」迁移边：
    服务重启后，上次进程中被中断的任务会永久停在 running——worker 不会拾取它（队列是内存态），
    它也无法经 retry（要求 failed）重新入队。此处统一置 failed 并写明原因，
    使状态可自洽、可用现有 retry 通道重新入队（failed → queued）。
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE task SET status = 'failed', error = ?, updated_at = ? WHERE status = 'running'",
            ("服务重启中断：任务未执行完（残留 running 状态由启动收敛置为 failed，可重试）", _now()),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


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
        _schedule_finish_hooks(task_id, "failed")
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
            # 必须等 handler 真正收尾（它在收尾时会杀掉子进程树、释放环境目录）再返回，
            # 否则 worker 立刻处理 retry 入队的新任务，会与尚未退出的旧执行并发写同一环境。
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
            _schedule_finish_hooks(task_id, "failed")
            return
        except asyncio.CancelledError:
            # 被 cancel_task 取消（协作式）：同样等收尾后再返回（不重新抛出以免传播到 worker）
            _set_status(task_id, "cancelled")
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
            return
        finally:
            _running_tasks.pop(task_id, None)
        _set_status(task_id, "success")
        _schedule_finish_hooks(task_id, "success")
    except asyncio.CancelledError:
        _set_status(task_id, "cancelled")
        return
    except Exception as e:  # noqa: BLE001 —— 任务级异常需落库并继续
        _set_status(task_id, "failed", error=str(e))
        _schedule_finish_hooks(task_id, "failed")
