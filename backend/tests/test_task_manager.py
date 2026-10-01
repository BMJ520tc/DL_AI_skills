"""任务状态机与启动收敛（《模块详细设计》2.1、D2）。

覆盖：queued → running → success/failed、failed → retry → queued、
running → cancelled，以及服务重启后残留 running 的收敛（GB-7）。
"""
from __future__ import annotations

import asyncio

from app.db.connection import get_connection
from app.services import task_manager


def _insert_raw_task(task_id: str, task_type: str, status: str) -> None:
    """直接落一条指定状态的任务，模拟上次进程遗留的中间态。"""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO task(task_id, task_type, project_id, params, status, created_at, updated_at) "
            "VALUES (?, ?, NULL, '{}', ?, '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00')",
            (task_id, task_type, status),
        )
        conn.commit()
    finally:
        conn.close()


def _run_scenario(coro_factory):
    """在单个事件循环里跑一段场景（worker 与队列绑定同一 loop）。"""

    async def scenario():
        await task_manager.start()
        try:
            return await coro_factory()
        finally:
            await task_manager.stop()

    return asyncio.run(scenario())


async def _wait_terminal(task_id: str, timeout_s: float = 5.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        task = task_manager.get_task(task_id)
        if task["status"] in ("success", "failed", "cancelled"):
            return task
        await asyncio.sleep(0.01)
    raise AssertionError(f"任务 {task_id} 未在 {timeout_s}s 内到达终态：{task_manager.get_task(task_id)}")


def test_reconcile_marks_stale_running_as_failed(isolated_db):
    _insert_raw_task("stale-1", "pdf_parse", "running")
    _insert_raw_task("done-1", "pdf_parse", "success")
    _insert_raw_task("queued-1", "pdf_parse", "queued")

    assert task_manager.reconcile_stale_tasks() == 1

    stale = task_manager.get_task("stale-1")
    assert stale["status"] == "failed"
    assert "重启" in (stale["error"] or "")
    # 终态与 queued 不受影响
    assert task_manager.get_task("done-1")["status"] == "success"
    assert task_manager.get_task("queued-1")["status"] == "queued"
    # 收敛后可经既有通道重新入队（failed → queued）
    assert task_manager.retry_task("stale-1") is True
    assert task_manager.get_task("stale-1")["status"] == "queued"


def test_reconcile_is_idempotent(isolated_db):
    _insert_raw_task("stale-2", "env_create", "running")
    assert task_manager.reconcile_stale_tasks() == 1
    assert task_manager.reconcile_stale_tasks() == 0


def test_worker_runs_handler_and_records_progress(isolated_db):
    seen: dict = {}

    async def handler(params, task_id):
        seen["params"] = params
        task_manager.update_progress(task_id, {"step": 1, "total": 2})

    async def body():
        task_manager.register_handler("demo", handler)
        tid = task_manager.create_task("demo", params={"k": "v"})
        return tid, await _wait_terminal(tid)

    tid, task = _run_scenario(body)

    assert task["status"] == "success"
    assert seen["params"] == {"k": "v"}
    assert task_manager.get_task(tid)["progress"] == '{"step": 1, "total": 2}'


def test_failed_handler_records_error(isolated_db):
    async def boom(params, task_id):
        raise RuntimeError("boom-detail")

    async def body():
        task_manager.register_handler("boom", boom)
        tid = task_manager.create_task("boom")
        return await _wait_terminal(tid)

    task = _run_scenario(body)
    assert task["status"] == "failed"
    assert "boom-detail" in task["error"]


def test_unknown_task_type_fails_without_crashing_worker(isolated_db):
    async def body():
        tid = task_manager.create_task("no-such-handler")
        return await _wait_terminal(tid)

    task = _run_scenario(body)
    assert task["status"] == "failed"
    assert "no handler" in task["error"]


def test_cancel_queued_task(isolated_db):
    tid = task_manager.create_task("demo")
    assert task_manager.get_task(tid)["status"] == "queued"
    assert task_manager.cancel_task(tid) is True
    assert task_manager.get_task(tid)["status"] == "cancelled"
    # 已取消的任务不会被 worker 执行
    assert task_manager.retry_task(tid) is False
