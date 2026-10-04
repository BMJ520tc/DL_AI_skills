"""版本管理（阶段4 4d-1/4d-2，模块详细设计 7.6，D8）。

结构化项目工作区即 git 仓库：创建时初始化（或首次保存时懒初始化，覆盖 4d-1 之前的
老项目），「保存即提交」「运行即提交」把 graph.json 与 network_version.json 逐版本入库
（需求五.3「每次运行」含失败运行：训练失败同样提交一条「训练失败 <task_id>」版本节点）；
4d-2 在此之上提供版本树（git 提交历史 + 元数据摘要 + 父子关系）、两版本对比
（代码差异 = 各自再生成代码的 unified diff + 参数差异表）、回退（检出目标版本图并
把回退动作本身记为一个新提交，需求五.3）。

约定：
- 提交身份用**仓库级**配置（git config，不带 --global），不依赖本机全局 git 配置；
- 全部走列表参数 subprocess（不经 shell），Windows 无 shell 特性依赖；
- 命令失败抛 RuntimeError 透出，由调用方决定降级方式（API 响应带 version_error、
  服务内部记日志）——不静默；
- git 是短平快操作（毫秒级），同步 subprocess.run 即可；长任务不经过本模块。
"""
from __future__ import annotations

import difflib
import json
import logging
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.services import network_export, project_manager

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


def _require_network(project_id: str) -> dict:
    """画布网络守卫（版本 API 只服务结构化项目；与 network_service 同口径）。"""
    try:
        return project_manager.require_type(project_id, {"structured"})
    except LookupError as e:
        raise LookupError(f"network not found: {e}") from e
    except PermissionError as e:
        raise PermissionError(f"不是结构化项目（画布网络）：{e}") from e


def _file_at(ws: Path, commit: str, filename: str) -> Optional[dict]:
    """某提交下的 JSON 文件内容；该提交没有此文件时返回 None。"""
    proc = _run_git(ws, "show", f"{commit}:{filename}", check=False)
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


def _resolve_commit(ws: Path, ref: str) -> str:
    """把短号/全号解析为完整提交号；不存在抛 LookupError（API 层映射 404）。"""
    proc = _run_git(ws, "rev-parse", "--verify", f"{ref}^{{commit}}", check=False)
    if proc.returncode != 0:
        raise LookupError(f"版本不存在：{ref}")
    return proc.stdout.strip()


def init_repo(ws: Path) -> None:
    """git init + 仓库级提交身份；已初始化则只补身份（幂等）。"""
    if not (ws / ".git").exists():
        _run_git(ws, "init")
    _run_git(ws, "config", "user.name", GIT_USER_NAME)
    _run_git(ws, "config", "user.email", GIT_USER_EMAIL)


def _version_payload(graph: dict, run_summary: Optional[dict] = None,
                     rollback_to: Optional[str] = None) -> dict:
    """network_version.json：输入/输出规格（DAG 级：无入边为输入、无出边为输出）+
    运行结果摘要 + 保存时间 + 回退来源（4d-1 实施要点 2/3；rollback_to 为 4d-2 增量字段）。"""
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
        "rollback_to": rollback_to,
    }


def _write_version(ws: Path, graph: dict, run_summary: Optional[dict] = None,
                   rollback_to: Optional[str] = None) -> None:
    (ws / VERSION_FILENAME).write_text(
        json.dumps(_version_payload(graph, run_summary, rollback_to), ensure_ascii=False, indent=2),
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


RUN_ERROR_SUMMARY_CHARS = 500


def commit_run(project_id: str, task_id: str, metrics: Optional[dict] = None, *,
               status: str = "success", error: Optional[str] = None) -> dict:
    """运行即提交：run_summary（指标摘要）写进 network_version.json 并提交。

    需求五.3「每次保存或运行时生成一个版本节点」：**失败也是一次运行**，故按 status 分两种提交
    （4d-1 成功口径不变，参数均为增量、向后兼容）：
    - status="success"（默认，与既有调用点/用例完全一致）：提交信息「训练运行 <task_id>」，
      run_summary = {status, task_id, metrics, finished_at}（status 为新增字段，未改名）；
    - status="failed"：提交信息「训练失败 <task_id>」，
      run_summary = {status: "failed", task_id, error: 失败原因前 RUN_ERROR_SUMMARY_CHARS 字符,
      finished_at}，无 metrics。
    """
    project = project_manager.get_project(project_id)
    if project is None:
        raise LookupError(f"project {project_id} not found")
    if status not in ("success", "failed"):
        raise ValueError(f"未知的运行状态：{status}")
    ws = Path(project["workspace_path"])
    with _lock_for(ws):
        init_repo(ws)
        graph_path = ws / "graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() \
            else {"nodes": [], "edges": []}
        if status == "failed":
            summary = {
                "status": "failed",
                "task_id": task_id,
                "error": (error or "")[:RUN_ERROR_SUMMARY_CHARS],
                "finished_at": _now(),
            }
            message = f"训练失败 {task_id}"
        else:
            summary = {
                "status": "success",
                "task_id": task_id,
                "metrics": metrics,
                "finished_at": _now(),
            }
            message = f"训练运行 {task_id}"
        _write_version(ws, graph, run_summary=summary)
        add_args = [VERSION_FILENAME] + (["graph.json"] if graph_path.exists() else [])
        _run_git(ws, "add", *add_args)
        commit = _commit_staged(ws, message)
    return {"commit": commit}


# ---------------------------------------------------------------------------
# 4d-2：版本树 / 对比 / 回退
# ---------------------------------------------------------------------------

def version_tree(project_id: str) -> dict:
    """版本树：git 提交历史（新→旧）＋ 每个版本的元数据摘要（节点/连线数、
    输入输出规格、运行摘要、回退来源）＋ 父子关系（4d-2 实施要点 1）。

    **--topo-order**：默认的 git log 按提交时间排序、同秒提交之间不保证拓扑次序
    （实测：同一秒里连提的版本会被排成「根版本在子版本之前」），而「树」的呈现前提是
    父提交出现在所有子提交之后，否则界面会把线性历史误画成分叉。故显式取拓扑序。

    空仓库（git 已初始化但尚无提交）返回空树，不算错误——此刻画布尚无任何版本。
    """
    ws = Path(_require_network(project_id)["workspace_path"])
    with _lock_for(ws):
        if not (ws / ".git").exists():
            return {"current": None, "versions": []}
        head = _run_git(ws, "rev-parse", "HEAD", check=False)
        if head.returncode != 0:
            return {"current": None, "versions": []}
        current = head.stdout.strip()[:7]  # 与 versions[].short 同口径（--short 歧义时会加长）
        log = _run_git(ws, "log", "--topo-order", "--format=%H%x09%P%x09%ct%x09%s").stdout.strip()
        versions = []
        for line in log.splitlines():
            if not line.strip():
                continue
            full, parents_raw, ct, subject = line.split("\t", 3)
            graph = _file_at(ws, full, "graph.json") or {"nodes": [], "edges": []}
            version = _file_at(ws, full, VERSION_FILENAME) or {}
            versions.append({
                "commit": full,
                "short": full[:7],
                "message": subject,
                "committed_at": datetime.fromtimestamp(int(ct), timezone.utc).isoformat(),
                "parents": [p for p in parents_raw.split(" ") if p],
                "meta": {
                    "saved_at": version.get("saved_at"),
                    "node_count": len(graph.get("nodes", [])),
                    "edge_count": len(graph.get("edges", [])),
                    "input_spec": version.get("input_spec", []),
                    "output_spec": version.get("output_spec", []),
                    "run_summary": version.get("run_summary"),
                    "rollback_to": version.get("rollback_to"),
                },
            })
        return {"current": current, "versions": versions}


# 参数差异只比 data 里的参数键：剔除展示/布局元数据与 __ 前缀的形状追踪元数据
PARAM_KEYS_EXCLUDED = ("label", "layout_hint")


def _node_params(data: object) -> dict:
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items()
            if k not in PARAM_KEYS_EXCLUDED and not k.startswith("__")}


def _param_diff(g1: dict, g2: dict) -> dict:
    """两版 graph.json 的参数差异表：节点参数逐项比对 + 节点/连线增删。"""
    n1 = {n.get("id"): n for n in g1.get("nodes", []) if isinstance(n, dict) and n.get("id")}
    n2 = {n.get("id"): n for n in g2.get("nodes", []) if isinstance(n, dict) and n.get("id")}
    nodes_added = [{"id": nid, "type": n.get("type")} for nid, n in n2.items() if nid not in n1]
    nodes_removed = [{"id": nid, "type": n.get("type")} for nid, n in n1.items() if nid not in n2]
    nodes_changed = []
    for nid in n1.keys() & n2.keys():
        p1 = _node_params(n1[nid].get("data"))
        p2 = _node_params(n2[nid].get("data"))
        changes = []
        for key in sorted(p1.keys() | p2.keys()):
            old, new = p1.get(key), p2.get(key)
            if old != new:
                changes.append({"key": key, "old": old, "new": new})
        if changes:
            nodes_changed.append(
                {"id": nid, "type": n1[nid].get("type"), "param_changes": changes})
    e1 = {e.get("id"): e for e in g1.get("edges", []) if isinstance(e, dict) and e.get("id")}
    e2 = {e.get("id"): e for e in g2.get("edges", []) if isinstance(e, dict) and e.get("id")}
    edges_added = [{"id": eid, "source": e.get("source"), "target": e.get("target")}
                   for eid, e in e2.items() if eid not in e1]
    edges_removed = [{"id": eid, "source": e.get("source"), "target": e.get("target")}
                     for eid, e in e1.items() if eid not in e2]
    return {
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "nodes_changed": nodes_changed,
        "edges_added": edges_added,
        "edges_removed": edges_removed,
    }


def compare_versions(project_id: str, v1: str, v2: str) -> dict:
    """两版本对比（4d-2 实施要点 2）：代码差异 = 各自画布图再生成代码的 unified diff
    （与导出/训练同源，导出即所训——比 graph.json 的 git diff 语义化）；
    参数差异 = 节点参数逐项比对表。版本号不存在抛 LookupError（404）。

    单边再生成失败不阻断对比：错误透出在 code_diff_error，参数差异照常给出。
    空画布（尚无节点）视为空代码，diff 呈现为纯新增。
    """
    ws = Path(_require_network(project_id)["workspace_path"])
    with _lock_for(ws):
        full1 = _resolve_commit(ws, v1)
        full2 = _resolve_commit(ws, v2)
        g1 = _file_at(ws, full1, "graph.json") or {"nodes": [], "edges": []}
        g2 = _file_at(ws, full2, "graph.json") or {"nodes": [], "edges": []}

        def _code_of(graph: dict) -> str:
            if not graph.get("nodes"):
                return ""
            return network_export.generate(graph)

        code_diff, code_diff_error = None, None
        error1 = error2 = None
        try:
            code1 = _code_of(g1)
        except Exception as e:  # noqa: BLE001 —— 对比是只读操作，失败必须透出不静默
            code1, error1 = None, f"v1 再生成代码失败：{e}"
        try:
            code2 = _code_of(g2)
        except Exception as e:  # noqa: BLE001
            code2, error2 = None, f"v2 再生成代码失败：{e}"
        if error1 or error2:
            code_diff_error = "；".join(e for e in (error1, error2) if e)
        else:
            code_diff = list(difflib.unified_diff(
                code1.splitlines(), code2.splitlines(),
                fromfile=f"{full1[:7]}", tofile=f"{full2[:7]}", lineterm="",
            ))
        return {
            "v1": full1[:7],
            "v2": full2[:7],
            "code_diff": code_diff,
            "code_diff_error": code_diff_error,
            "param_diff": _param_diff(g1, g2),
        }


def rollback(project_id: str, target_version: str) -> dict:
    """回退（4d-2 实施要点 3，需求五.3）：把 graph.json 检出到目标版本，
    回退动作本身记为「回退到 <short>」新提交。返回新提交号与恢复出的图
    （画布直接替换，一次往返）。目标即当前版本 → ValueError（400）。"""
    ws = Path(_require_network(project_id)["workspace_path"])
    with _lock_for(ws):
        init_repo(ws)
        target = _resolve_commit(ws, target_version)
        head = _run_git(ws, "rev-parse", "HEAD").stdout.strip()
        if target == head:
            raise ValueError("目标版本就是当前版本，无需回退")
        _run_git(ws, "checkout", target, "--", "graph.json")
        graph = json.loads((ws / "graph.json").read_text(encoding="utf-8"))
        _write_version(ws, graph, rollback_to=target)
        _run_git(ws, "add", "graph.json", VERSION_FILENAME)
        commit = _commit_staged(ws, f"回退到 {target[:7]}")
    return {"commit": commit, "target": target[:7], "graph": graph}
