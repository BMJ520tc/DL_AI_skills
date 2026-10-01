"""测试夹具：把知识库指向临时 SQLite，并复位进程内任务管理器状态。

后端所有持久化都经 `app.db.connection.get_connection()`，而 `DB_PATH` 是
`app.db.connection` 的模块级常量，因此用 monkeypatch 改写该常量即可完全隔离，
测试绝不触碰 `data/index.db`（真实运行数据）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db import connection  # noqa: E402


def _reset_task_manager() -> None:
    """清空任务管理器的内存态（队列/worker/handler），避免测试相互串扰。"""
    from app.services import task_manager

    task_manager._queue = None
    task_manager._worker_task = None
    task_manager._running_tasks.clear()
    task_manager._handlers.clear()


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """临时知识库（建表完成），yield 其路径。"""
    db_path = tmp_path / "index.db"
    monkeypatch.setattr(connection, "DB_PATH", db_path)
    connection.init_db()
    try:
        yield db_path
    finally:
        _reset_task_manager()


@pytest.fixture()
def app_client(isolated_db):
    """FastAPI TestClient：走 lifespan（含启动收敛与路由挂载），数据库为临时库。"""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client
