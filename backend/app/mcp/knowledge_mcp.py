"""知识库 stdio MCP server（模块详细设计 2.6）。

以 stdio JSON-RPC 暴露 knowledge_search 工具，供 agent 会话每会话临时挂载。
运行: python -m app.mcp.knowledge_mcp（cwd 为 backend/）。
"""
import json
import sys

from app.db import connection
from app.db.connection import init_db
from app.services import knowledge_service, project_manager

SERVER_NAME = "knowledge-mcp"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

_KNOWLEDGE_SEARCH_TOOL = {
    "name": "knowledge_search",
    "description": (
        "检索知识库统一索引（四类数据 paper/run/knowledge/module 及数据集 dataset）。"
        "支持按类型、任务类型、模型、数据集过滤与关键词全文检索。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "数据类型，可选 paper/run/knowledge/module/dataset",
            },
            "task_type": {"type": "string", "description": "任务类型过滤"},
            "model": {"type": "string", "description": "模型名过滤"},
            "dataset": {"type": "string", "description": "数据集名过滤"},
            "q": {"type": "string", "description": "关键词全文检索"},
            "limit": {"type": "integer", "default": 20},
        },
    },
}


_LIST_PROJECTS_TOOL = {
    "name": "list_projects",
    "description": "列出平台项目（project_id / 名称 / 类型 original|structured / 状态）。回答「我有几个项目」这类问题用它。",
    "inputSchema": {
        "type": "object",
        "properties": {"project_type": {"type": "string", "description": "可选过滤：original 或 structured"}},
    },
}

_GET_PROJECT_TOOL = {
    "name": "get_project",
    "description": "按 project_id 取项目详情（名称/类型/状态/父项目/工作区），若已有结构分析报告则附报告摘要。",
    "inputSchema": {
        "type": "object",
        "properties": {"project_id": {"type": "string"}},
        "required": ["project_id"],
    },
}

_LIST_RUNS_TOOL = {
    "name": "list_runs",
    "description": "列出最近运行记录（run_type / status / 指标 / 时间），可按 project_id 过滤。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "可选，只看某项目"},
            "limit": {"type": "integer", "default": 20},
        },
    },
}

_PLATFORM_TOOLS = [_LIST_PROJECTS_TOOL, _GET_PROJECT_TOOL, _LIST_RUNS_TOOL]
_ALL_TOOLS = [_KNOWLEDGE_SEARCH_TOOL, *_PLATFORM_TOOLS]

# 全为**只读**工具：不写库、不发起任务、不改盘。
_READ_ALLOWED = {"knowledge_search", "list_projects", "get_project", "list_runs"}


def _report_summary(workspace_path: str | None) -> dict | None:
    try:
        from pathlib import Path
        p = Path(workspace_path or "") / "reports" / "structure_report.json"
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return {
            "entry_points": data.get("entry_points"),
            "model_files": data.get("model_files"),
            "module_hierarchy_len": len(data.get("module_hierarchy") or []),
            "dependencies": (data.get("dependencies") or [])[:20],
        }
    except Exception:
        return None


def _dispatch_tool(name: str, args: dict):
    if name == "knowledge_search":
        return knowledge_service.search(**args)
    if name == "list_projects":
        rows = project_manager.list_projects(args.get("project_type"))
        return [{"project_id": r["project_id"], "name": r.get("name"),
                 "project_type": r.get("project_type"), "status": r.get("status")} for r in rows]
    if name == "get_project":
        pid = args.get("project_id")
        proj = project_manager.get_project(pid)
        if not proj:
            raise ValueError(f"项目不存在：{pid}")
        out = {k: proj.get(k) for k in
               ("project_id", "name", "project_type", "status", "parent_project_id", "workspace_path")}
        summary = _report_summary(proj.get("workspace_path"))
        if summary:
            out["structure_report"] = summary
        return out
    if name == "list_runs":
        limit = int(args.get("limit") or 20)
        conn = connection.get_connection()
        try:
            if args.get("project_id"):
                rows = conn.execute(
                    "SELECT run_id, project_id, run_type, status, metrics, started_at FROM run_record"
                    " WHERE project_id = ? ORDER BY started_at DESC LIMIT ?",
                    (args["project_id"], limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_id, project_id, run_type, status, metrics, started_at FROM run_record"
                    " ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    raise ValueError(f"未知工具：{name}")


def _send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle(req: dict) -> dict | None:
    method = req.get("method")
    req_id = req.get("id")
    if req_id is None:
        return None  # notification（如 initialized），不响应

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _ALL_TOOLS}}
    if method == "tools/call":
        params = req.get("params", {}) or {}
        name = params.get("name", "")
        args = params.get("arguments", {}) or {}
        try:
            result = _dispatch_tool(name, args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]},
            }
        except Exception as e:  # noqa: BLE001
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}}

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"method not found: {method}"}}


def main() -> None:
    init_db()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(req)
        if resp is not None:
            _send(resp)


if __name__ == "__main__":
    main()
