"""知识库 stdio MCP server（模块详细设计 2.6）。

以 stdio JSON-RPC 暴露 knowledge_search 工具，供 agent 会话每会话临时挂载。
运行: python -m app.mcp.knowledge_mcp（cwd 为 backend/）。

③续：助手「可写」模式下再暴露**平台动作工具**（建原始项目 / 建环境 / 结构分析）。
动作工具跑在本 stdio 子进程里、拿不到后端内存，故**确认握手经 HTTP 回环**：本进程 POST
一条确认请求到后端 → 后端经 SSE 推给前端弹窗 → 用户点确认/取消 → 本进程轮询取决定 →
**用户确认后才**调后端既有 API 执行。未确认即不执行。
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from app.db import connection
from app.db.connection import init_db
from app.services import knowledge_service, project_manager

SERVER_NAME = "knowledge-mcp"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

# 会话上下文由后端挂载 MCP 时注入（见 agent_service._knowledge_mcp）。缺省 = 无动作能力。
_SESSION_KEY = os.getenv("ASSISTANT_SESSION_KEY") or ""
_MCP_MODE = os.getenv("ASSISTANT_MODE") or "read"
_BACKEND_URL = (os.getenv("DL_AI_BACKEND_URL") or "http://127.0.0.1:8000").rstrip("/")
_CONFIRM_TIMEOUT_S = float(os.getenv("ASSISTANT_CONFIRM_TIMEOUT_S", "300"))
_CONFIRM_POLL_S = float(os.getenv("ASSISTANT_CONFIRM_POLL_S", "1.0"))

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

# ---- 平台**动作**工具（③续）：写操作，仅在助手「可写」模式暴露，且**一律先经 UI 确认** ----

_CREATE_PROJECT_TOOL = {
    "name": "platform_create_project",
    "description": (
        "在平台上创建一个**原始项目**（project_type=original）。可选给 source_url（仓库地址或本地路径）"
        "以创建后加载源码。**属写操作**：会先弹窗请用户确认，用户同意后才真正创建；用户取消则不执行。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "项目名"},
            "source_url": {"type": "string", "description": "可选：仓库地址或本地路径，提供则创建后加载源码"},
        },
        "required": ["name"],
    },
}

_CREATE_ENV_TOOL = {
    "name": "platform_create_env",
    "description": (
        "为某个**原始项目**创建独立运行环境（装依赖，走后台任务队列）。"
        "**属写操作**：会先弹窗请用户确认，用户同意后才发起；用户取消则不执行。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {"project_id": {"type": "string", "description": "原始项目 id"}},
        "required": ["project_id"],
    },
}

_RUN_ANALYZE_TOOL = {
    "name": "platform_run_analyze",
    "description": (
        "对某个**原始项目**发起结构分析（读源码产出结构报告，走后台任务队列）。"
        "**属写操作**：会先弹窗请用户确认，用户同意后才发起；用户取消则不执行。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {"project_id": {"type": "string", "description": "原始项目 id"}},
        "required": ["project_id"],
    },
}

_ACTION_TOOLS = [_CREATE_PROJECT_TOOL, _CREATE_ENV_TOOL, _RUN_ANALYZE_TOOL]
# 工具名 → 内部动作名
_ACTION_DISPATCH = {
    "platform_create_project": "create_project",
    "platform_create_env": "create_env",
    "platform_run_analyze": "run_analyze",
}
_ACTION_TOOL_NAMES = set(_ACTION_DISPATCH)


def _tools_for_session() -> list:
    """本会话暴露的工具集：只读工具恒有；**动作工具仅在「可写」模式且挂了确认通道时**给。"""
    tools = [_KNOWLEDGE_SEARCH_TOOL, *_PLATFORM_TOOLS]
    if _MCP_MODE == "write" and _SESSION_KEY:
        tools = tools + _ACTION_TOOLS
    return tools


def _http_json(method: str, path: str, payload: dict | None = None) -> dict:
    """向后端发一个 JSON 请求（stdlib urllib，零依赖）。失败抛 RuntimeError（带后端 detail）。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{_BACKEND_URL}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            detail = ""
        raise RuntimeError(f"后端返回 {e.code}：{detail[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接后端（{_BACKEND_URL}）：{e.reason}")


def _request_confirmation(action: str, params: dict) -> bool:
    """经后端 HTTP 回环请用户确认；返回是否获准。未确认（取消/超时/无通道）一律 False。"""
    if not _SESSION_KEY:
        raise RuntimeError("此会话未挂载助手确认通道，动作已取消")
    rec = _http_json("POST", "/api/assistant/confirm",
                     {"session_key": _SESSION_KEY, "action": action, "params": params})
    confirm_id = rec.get("confirm_id")
    if not confirm_id:
        raise RuntimeError("后端未返回确认编号，动作已取消")
    deadline = time.monotonic() + _CONFIRM_TIMEOUT_S
    while time.monotonic() < deadline:
        st = _http_json("GET", f"/api/assistant/confirm/{confirm_id}")
        status = st.get("status")
        if status == "approved":
            return True
        if status in ("denied", "expired"):
            return False
        time.sleep(_CONFIRM_POLL_S)
    return False


def _run_action(action: str, args: dict) -> dict:
    """执行一个平台动作：先确认，获准后调后端既有 API。未获准返回 `executed=false`（非报错）。"""
    if action == "create_project":
        name = str(args.get("name") or "").strip()
        if not name:
            raise ValueError("缺少参数 name")
        source_url = str(args.get("source_url") or "").strip() or None
        params = {"name": name, "source_url": source_url}
        if not _request_confirmation(action, params):
            return {"executed": False, "reason": "用户未确认（取消或超时）"}
        result = _http_json("POST", "/api/projects",
                            {"project_type": "original", "name": name, "source_url": source_url})
        return {"executed": True, "action": action, "params": params, "result": result}

    if action in ("create_env", "run_analyze"):
        project_id = str(args.get("project_id") or "").strip()
        if not project_id:
            raise ValueError("缺少参数 project_id")
        params = {"project_id": project_id}
        if not _request_confirmation(action, params):
            return {"executed": False, "reason": "用户未确认（取消或超时）"}
        pid = urllib.parse.quote(project_id, safe="")
        path = f"/api/projects/{pid}/env" if action == "create_env" else f"/api/projects/{pid}/analyze"
        result = _http_json("POST", path)
        return {"executed": True, "action": action, "params": params, "result": result}

    raise ValueError(f"未知动作：{action}")


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
    if name in _ACTION_TOOL_NAMES:
        # 动作工具只在可写模式暴露；这里再挡一道（工具列表被绕开也不给执行）
        if name not in {t["name"] for t in _tools_for_session()}:
            raise ValueError(f"{name} 在当前模式不可用（平台动作工具仅在助手「可写」模式提供）")
        return _run_action(_ACTION_DISPATCH[name], args)
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
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _tools_for_session()}}
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
