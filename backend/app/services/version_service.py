"""版本管理底座（阶段4 4d-1，模块详细设计 7.6，D8）。

结构化项目工作区即 git 仓库：创建时初始化（或首次保存时懒初始化，覆盖 4d-1 之前的
老项目），「保存即提交」「运行即提交」把 graph.json 与 network_version.json 逐版本入库。

约定：
- 提交身份用**仓库级**配置（git config，不带 --global），不依赖本机全局 git 配置；
- 全部走列表参数 subprocess（不经 shell），Windows 无 shell 特性依赖；
- 命令失败抛 RuntimeError 透出，由调用方决定降级方式（API 响应带 version_error、
  服务内部记日志）——不静默；
- git 是短平快操作（毫秒级），同步 subprocess.run 即可；长任务不经过本模块。
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.services import project_manager

logger = logging.getLogger(__name__)

VERSION_FILENAME = "network_version.json"
SCHEMA_VERSION = 1
GIT_USER_NAME = "DL-AI-skills"
GIT_USER_EMAIL = "dl-ai-skills@local"
GIT_TIMEOUT_S = 30.0

# 同一项目工作区的 git 命令互斥（保存与训练完成可能并发提交，防 index.lock 冲突）
_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


def _lock_for(ws: Path) -> threading.Lock:
    key = str(ws).lower()
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(ws: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", str(ws), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=GIT_TIMEOUT_S,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[-300:]
        raise RuntimeError(f"git {' '.join(args)} 失败（exit {proc.returncode}）：{detail}")
    return proc


def init_repo(ws: Path) -> None:
    """git init + 仓库级提交身份；已初始化则只补身份（幂等）。"""
    if not (ws / ".git").exists():
        _run_git(ws, "init")
    _run_git(ws, "config", "user.name", GIT_USER_NAME)
    _run_git(ws, "config", "user.email", GIT_USER_EMAIL)


def _version_payload(graph: dict, run_summary: Optional[dict] = None) -> dict:
    """network_version.json：输入/输出规格（DAG 级：无入边为输入、无出边为输出）+
    运行结果摘要 + 保存时间（4d-1 实施要点 2/3）。"""
    nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
    edges = [e for e in graph.get("edges", []) if isinstance(e, dict)]
    targets = {e.get("target") for e in edges}
    sources = {e.get("source") for e in edges}
    return {
        "schema_version": SCHEMA_VERSION,
        "saved_at": _now(),
        "input_spec": [{"node_id": n.get("id"), "type": n.get("type")}
                       for n in nodes if n.get("id") not in targets],
        "output_spec": [{"node_id": n.get("id"), "type": n.get("type")}
                        for n in nodes if n.get("id") not in sources],
        "run_summary": run_summary,
    }


def _write_version(ws: Path, graph: dict, run_summary: Optional[dict] = None) -> None:
    (ws / VERSION_FILENAME).write_text(
        json.dumps(_version_payload(graph, run_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _commit_staged(ws: Path, message: str) -> str:
    """提交暂存区并返回短提交号；saved_at 恒变化故每次保存都有内容（保存即提交）。"""
    _run_git(ws, "commit", "-m", message)
    return _run_git(ws, "rev-parse", "--short", "HEAD").stdout.strip()


def commit_graph(project_id: str, graph: dict) -> dict:
    """保存即提交：每次 PUT graph 都形成一次提交（graph.json + network_version.json，
    老项目在此懒初始化）。"""
    project = project_manager.get_project(project_id)
    if project is None:
        raise LookupError(f"project {project_id} not found")
    ws = Path(project["workspace_path"])
    with _lock_for(ws):
        init_repo(ws)
        _write_version(ws, graph)
        _run_git(ws, "add", "graph.json", VERSION_FILENAME)
        commit = _commit_staged(ws, "保存画布")
    return {"commit": commit}


def commit_run(project_id: str, task_id: str, metrics: dict) -> dict:
    """运行即提交：run_summary（指标摘要）写进 network_version.json 并提交。"""
    project = project_manager.get_project(project_id)
    if project is None:
        raise LookupError(f"project {project_id} not found")
    ws = Path(project["workspace_path"])
    with _lock_for(ws):
        init_repo(ws)
        graph_path = ws / "graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() \
            else {"nodes": [], "edges": []}
        _write_version(ws, graph, run_summary={
            "task_id": task_id, "metrics": metrics, "finished_at": _now(),
        })
        add_args = [VERSION_FILENAME] + (["graph.json"] if graph_path.exists() else [])
        _run_git(ws, "add", *add_args)
        commit = _commit_staged(ws, f"训练运行 {task_id}")
    return {"commit": commit}
