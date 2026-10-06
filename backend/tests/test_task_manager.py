"""任务状态机与启动收敛（《模块详细设计》2.1、D2）。

覆盖：queued → running → success/failed、failed → retry → queued、
running → cancelled，以及服务重启后残留 **running / queued** 的收敛（GB-7）。
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


def test_reconcile_marks_stale_running_and_queued_as_failed(isolated_db):
    """残留 running **与 queued** 都要收敛——队列是内存态，重启后 queued 永远不会被拾取。"""
    _insert_raw_task("stale-1", "pdf_parse", "running")
    _insert_raw_task("done-1", "pdf_parse", "success")
    _insert_raw_task("queued-1", "pdf_parse", "queued")

    assert task_manager.reconcile_stale_tasks() == 2      # running + queued

    stale = task_manager.get_task("stale-1")
    assert stale["status"] == "failed"
    assert "重启" in (stale["error"] or "")
    stuck = task_manager.get_task("queued-1")
    assert stuck["status"] == "failed"
    assert "重启" in (stuck["error"] or "")
    # 终态不受影响
    assert task_manager.get_task("done-1")["status"] == "success"
    # 收敛后可经既有通道重新入队（failed → queued）
    assert task_manager.retry_task("stale-1") is True
    assert task_manager.get_task("stale-1")["status"] == "queued"


def test_reconcile_unblocks_create_task_dedupe(isolated_db):
    """残留 queued 会让 `create_task` 的「同参数复用」永久命中它 → 提交再也跑不起来。

    2026-10-05 实测：昨天遗留的 `decompose_trace` 卡在 queued，用户点「补形状」后端明明收到
    `POST …/decompose/trace`，却只把那个死 id 返回、什么都不执行（界面一直显示「在跑」）。
    收敛后必须能建出**新**任务。
    """
    _insert_raw_task("stale-q", "decompose_trace", "queued")     # params='{}'、project NULL

    # 收敛前：同参数提交被去重到那个死 id（真 bug 的表现）
    assert task_manager.create_task("decompose_trace") == "stale-q"

    assert task_manager.reconcile_stale_tasks() == 1
    assert task_manager.create_task("decompose_trace") != "stale-q"   # 不再被死任务吃掉


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


# --------------------------- 看板口径：排序与 project_name ---------------------------


def _insert_task_at(task_id: str, task_type: str, status: str, created_at: str,
                    project_id: str | None = None) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO task(task_id, task_type, project_id, params, status, created_at, updated_at) "
            "VALUES (?, ?, ?, '{}', ?, ?, ?)",
            (task_id, task_type, project_id, status, created_at, created_at),
        )
        conn.commit()
    finally:
        conn.close()


def test_list_tasks_board_order_puts_active_first(isolated_db):
    """看板口径：执行中 → 排队中 → 其余按创建时间倒序。

    纯时间倒序时，一个跑半小时的长任务会被后来创建的一堆短任务挤到看不见的地方——
    这正是「看不到现在在跑什么」的来源。
    """
    _insert_task_at("old-run", "reproduce", "running", "2026-10-01T00:00:01+00:00")
    _insert_task_at("mid-queued", "network_train", "queued", "2026-10-01T00:00:05+00:00")
    _insert_task_at("new-failed", "extract_items", "failed", "2026-10-01T00:00:08+00:00")
    _insert_task_at("new-success", "pdf_parse", "success", "2026-10-01T00:00:09+00:00")

    assert [t["task_id"] for t in task_manager.list_tasks(order="board")] == [
        "old-run", "mid-queued", "new-success", "new-failed",
    ]
    # 默认仍是纯时间倒序（既有行为不变）
    assert [t["task_id"] for t in task_manager.list_tasks()] == [
        "new-success", "new-failed", "mid-queued", "old-run",
    ]


def test_list_tasks_includes_project_name(isolated_db):
    """列表带回 project_name，看板不必再逐条查项目。"""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO project(project_id, project_type, name, status, workspace_path, "
            "created_at, updated_at, schema_version) VALUES "
            "('p1', 'original', '我的项目', 'ready', 'C:/x', '2026-10-01T00:00:00+00:00', "
            "'2026-10-01T00:00:00+00:00', '1.0')"
        )
        conn.commit()
    finally:
        conn.close()
    _insert_task_at("t1", "reproduce", "running", "2026-10-01T00:00:01+00:00", project_id="p1")
    _insert_task_at("t2", "pdf_parse", "success", "2026-10-01T00:00:02+00:00")

    rows = {t["task_id"]: t for t in task_manager.list_tasks(order="board")}
    assert rows["t1"]["project_name"] == "我的项目"
    assert rows["t2"]["project_name"] is None  # 无项目任务不报错
