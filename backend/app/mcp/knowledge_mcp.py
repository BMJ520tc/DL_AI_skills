"""知识库 stdio MCP server（模块详细设计 2.6）。

以 stdio JSON-RPC 暴露 knowledge_search 工具，供 agent 会话每会话临时挂载。
运行: python -m app.mcp.knowledge_mcp（cwd 为 backend/）。
"""
import json
import sys

from app.db.connection import init_db
from app.services import knowledge_service

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
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": [_KNOWLEDGE_SEARCH_TOOL]}}
    if method == "tools/call":
        args = req.get("params", {}).get("arguments", {}) or {}
        try:
            result = knowledge_service.search(**args)
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
