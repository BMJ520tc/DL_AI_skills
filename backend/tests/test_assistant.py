"""前端 AI 助手（《新增需求补充》补充 A + 流式/模式/常驻会话增强）。

agent 由假实现替代（不真调模型）：空消息守卫、prompt 组装（上下文 + 平台摘要 + "先查再答"）、
回复落进 progress.reply、**模式（只读/可写）**、**会话续接（conv_key）**、无内容时如实失败。
"""
from __future__ import annotations

import json
import time

import pytest

from app.mcp import knowledge_mcp
from app.services import agent_service, assistant_service


class _FakeQueue:
    """替代 asyncio.Queue：只收集 put_nowait 推入的事件（确认事件断言用）。"""

    def __init__(self) -> None:
        self.items: list = []

    def put_nowait(self, item) -> None:
        self.items.append(item)


def _wait(client, task_id: str, timeout_s: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in ("success", "failed", "cancelled"):
            return task
        time.sleep(0.05)
    raise AssertionError(f"任务未在 {timeout_s}s 内结束")


def _fake(result: dict, captured: dict):
    """替代 agent_service.run_client_stream（常驻会话流式）。"""
    async def fake(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["mode_key"] = kwargs.get("mode_key")
        captured["conv_key"] = kwargs.get("conv_key")
        on_event = kwargs.get("on_event")
        if on_event and result.get("result"):
            on_event("delta", {"text": result["result"][:4]})
        out = dict(result)
        out.setdefault("conv_key", kwargs.get("conv_key") or "conv-1")
        return out
    return fake


def test_chat_empty_message_rejected(app_client):
    r = app_client.post("/api/assistant/chat", json={"message": "   "})
    assert r.status_code == 400


def test_chat_success_reply_in_progress(app_client, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(agent_service, "run_client_stream",
                        _fake({"result": "这是助手的回答。", "session_id": "s1"}, captured))
    r = app_client.post("/api/assistant/chat", json={
        "message": "现在这个项目在干什么？",
        "context": {"page": "viewer", "project_id": "p1"},
    })
    assert r.status_code == 200
    task = _wait(app_client, r.json()["task_id"])
    assert task["status"] == "success"
    assert json.loads(task["progress"])["reply"] == "这是助手的回答。"
    # prompt 含"先查再答"约束 + 上下文 + 平台现状摘要
    assert "AI 助手" in captured["prompt"] and "先查再答" in captured["prompt"]
    assert "viewer" in captured["prompt"] and "p1" in captured["prompt"]
    assert "[平台现状]" in captured["prompt"] and "list_projects" in captured["prompt"]
    assert captured["mode_key"] == "read"   # 默认只读


def test_mode_selects_toolset(app_client, monkeypatch):
    """只读（默认）与可写各自透传 mode_key；只读工具集不含 Write/Edit/Bash。"""
    seen: dict = {}

    def make(tag: str):
        async def fake(prompt, **kwargs):
            seen[tag] = kwargs.get("mode_key")
            return {"result": "ok", "session_id": None, "conv_key": "c"}
        return fake

    monkeypatch.setattr(agent_service, "run_client_stream", make("read"))
    _wait(app_client, app_client.post("/api/assistant/chat",
                                      json={"message": "只读问一下"}).json()["task_id"])
    assert seen["read"] == "read"

    monkeypatch.setattr(agent_service, "run_client_stream", make("write"))
    _wait(app_client, app_client.post("/api/assistant/chat",
                                      json={"message": "帮我改", "mode": "write"}).json()["task_id"])
    assert seen["write"] == "write"

    assert assistant_service.READ_TOOLS == ["Read", "Glob", "Grep"]
    for banned in ("Write", "Edit", "Bash"):
        assert banned not in assistant_service.READ_TOOLS


def test_chat_resumes_conv(app_client, monkeypatch):
    """会话续接：请求带 conv_key（session_id 字段回传）时透传，回复回传同一 key。"""
    captured: dict = {}
    monkeypatch.setattr(agent_service, "run_client_stream",
                        _fake({"result": "好的。", "session_id": "cli-sid"}, captured))
    r = app_client.post("/api/assistant/chat", json={"message": "接着上一个问题", "session_id": "conv-abc"})
    task = _wait(app_client, r.json()["task_id"])
    assert captured["conv_key"] == "conv-abc"
    assert json.loads(task["progress"])["session_id"] == "conv-abc"


def test_chat_empty_reply_fails_honestly(app_client, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(agent_service, "run_client_stream",
                        _fake({"result": "", "session_id": None}, captured))
    r = app_client.post("/api/assistant/chat", json={"message": "在吗"})
    task = _wait(app_client, r.json()["task_id"])
    assert task["status"] == "failed"
    # 空回复 ≠ 凭证问题：文案不能误导用户去填凭证
    assert "凭证" not in (task["error"] or "")
    assert "没有产出内容" in (task["error"] or "")


def test_cli_auth_error_translated_to_hint(app_client, monkeypatch):
    """CLI 未登录原文（被当回复吐出来）要译制成「去设置页填凭证」的可操作指引。"""
    captured: dict = {}
    monkeypatch.setattr(agent_service, "run_client_stream",
                        _fake({"result": "Not logged in · Please run /login", "session_id": None}, captured))
    r = app_client.post("/api/assistant/chat", json={"message": "你好"})
    task = _wait(app_client, r.json()["task_id"])
    assert task["status"] == "failed"
    assert "设置" in (task["error"] or "") and "凭证" in (task["error"] or "")


def test_stale_session_retries_without_resume(app_client, monkeypatch):
    """续接目标失效（后端重启/会话过期）→ 自动丢掉续接重开一次，不把错误甩给用户。"""
    attempts: list = []

    async def fake(prompt, **kwargs):
        attempts.append(kwargs["options_factory"]().resume)
        raise RuntimeError(
            "Claude Code returned an error result: No conversation found with session ID: abc (exit code: 1)")

    monkeypatch.setattr(agent_service, "run_client_stream", fake)
    assistant_service._CONV["conv-x"] = "cli-sid"
    r = app_client.post("/api/assistant/chat", json={"message": "hi", "session_id": "conv-x"})
    task = _wait(app_client, r.json()["task_id"])
    assert attempts == ["cli-sid", None]   # 第一次带续接 → 失效后重开不带续接
    assert task["status"] == "failed"      # 两次都失败 → 如实失败（不伪造）


def test_path_gate_blocks_outside_roots(tmp_path):
    """路径闸门：读限白名单根、写只限 scratch、Bash 一律拒。"""
    import asyncio

    root = tmp_path / "repo"
    scratch = tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    gate = agent_service.path_gate_tool([root, scratch], [scratch])

    def decide(name, ti):
        return type(asyncio.run(gate(name, ti, None))).__name__

    assert decide("Read", {"file_path": str(root / "a.py")}) == "PermissionResultAllow"
    assert decide("Read", {"file_path": str(tmp_path / "outside.txt")}) == "PermissionResultDeny"
    assert decide("Grep", {"path": str(scratch)}) == "PermissionResultAllow"
    assert decide("Write", {"file_path": str(scratch / "n.txt")}) == "PermissionResultAllow"
    assert decide("Edit", {"file_path": str(root / "a.py")}) == "PermissionResultDeny"
    assert decide("Bash", {"command": "ls"}) == "PermissionResultDeny"
    # MCP 只读工具放行
    assert decide("mcp__knowledge__list_projects", {}) == "PermissionResultAllow"


def test_knowledge_mcp_read_tools_whitelisted():
    """只读平台工具进白名单（attach_knowledge 时）。"""
    rules = agent_service.allowed_tools(attach_knowledge=True)
    for t in ("knowledge_search", "list_projects", "get_project", "list_runs"):
        assert f"mcp__knowledge__{t}" in rules
    assert not any(r.startswith("mcp__knowledge__") for r in agent_service.allowed_tools(attach_knowledge=False))


# ---- ③续：平台动作工具 + UI 确认握手 ----

def test_mcp_tools_by_mode():
    """可写模式加动作工具；只读不加。"""
    read = set(assistant_service._mcp_tools_for_mode("read"))
    write = set(assistant_service._mcp_tools_for_mode("write"))
    assert "platform_create_project" not in read
    assert {"platform_create_project", "platform_create_env", "platform_run_analyze"} <= write
    assert {"knowledge_search", "list_projects"} <= read       # 只读工具两模式都有


def test_allowed_tools_can_include_action_tools():
    both = agent_service.KNOWLEDGE_MCP_READ_TOOLS + agent_service.KNOWLEDGE_MCP_ACTION_TOOLS
    rules = agent_service.allowed_tools(True, ["Read"], both)
    assert "mcp__knowledge__platform_create_project" in rules
    # 默认（只读）不含动作工具
    assert "mcp__knowledge__platform_create_project" not in agent_service.allowed_tools(True, ["Read"])


def test_knowledge_mcp_injects_session_env_only_when_keyed():
    cfg = agent_service._knowledge_mcp(session_key="sess-1", mode="write")
    env = cfg["knowledge"]["env"]
    assert env["ASSISTANT_SESSION_KEY"] == "sess-1" and env["ASSISTANT_MODE"] == "write"
    assert "DL_AI_BACKEND_URL" in env
    # 不传会话键时保持原样（run_stream/run_sync 等既有调用方不受影响）
    assert agent_service._knowledge_mcp()["knowledge"]["env"] == {"PYTHONPATH": str(agent_service.BACKEND_DIR)}


def test_request_confirmation_requires_live_session():
    with pytest.raises(LookupError):
        assistant_service.request_confirmation("no-such-session", "create_env", {})


def test_confirm_registry_approve_and_deny(monkeypatch):
    q = _FakeQueue()
    monkeypatch.setitem(assistant_service._SESSION_QUEUE, "s1", q)
    cid = assistant_service.request_confirmation("s1", "create_project", {"name": "demo"})
    # 事件已推给该会话（前端据此弹窗）
    assert q.items and q.items[0]["kind"] == "confirm" and q.items[0]["confirm_id"] == cid
    assert assistant_service.confirmation_status(cid)["status"] == "pending"
    assert assistant_service.decide_confirmation(cid, True)["status"] == "approved"
    assert assistant_service.confirmation_status(cid)["status"] == "approved"

    cid2 = assistant_service.request_confirmation("s1", "run_analyze", {"project_id": "p"})
    assert assistant_service.decide_confirmation(cid2, False)["status"] == "denied"


def test_confirm_expires_after_ttl(monkeypatch):
    monkeypatch.setattr(assistant_service, "CONFIRM_TTL_S", 0.0)
    q = _FakeQueue()
    monkeypatch.setitem(assistant_service._SESSION_QUEUE, "s2", q)
    cid = assistant_service.request_confirmation("s2", "run_analyze", {"project_id": "p"})
    assert assistant_service.confirmation_status(cid)["status"] == "expired"


def test_confirm_endpoints_handshake(app_client, monkeypatch):
    """HTTP 三端点：登记 → 轮询 pending → 前端确认 → 轮询 approved。"""
    q = _FakeQueue()
    monkeypatch.setitem(assistant_service._SESSION_QUEUE, "sess-http", q)

    r = app_client.post("/api/assistant/confirm",
                        json={"session_key": "sess-http", "action": "create_project",
                              "params": {"name": "demo"}})
    assert r.status_code == 200
    cid = r.json()["confirm_id"]
    assert q.items[0]["kind"] == "confirm" and q.items[0]["action"] == "create_project"

    assert app_client.get(f"/api/assistant/confirm/{cid}").json()["status"] == "pending"
    d = app_client.post(f"/api/assistant/confirm/{cid}/decide", json={"approved": True})
    assert d.status_code == 200 and d.json()["status"] == "approved"
    assert app_client.get(f"/api/assistant/confirm/{cid}").json()["status"] == "approved"


def test_confirm_endpoints_errors(app_client):
    # 无活动会话 → 409（动作不得执行）
    r = app_client.post("/api/assistant/confirm",
                        json={"session_key": "ghost", "action": "run_analyze", "params": {}})
    assert r.status_code == 409
    # 未知确认号 → 404
    assert app_client.get("/api/assistant/confirm/nope").status_code == 404
    assert app_client.post("/api/assistant/confirm/nope/decide",
                           json={"approved": True}).status_code == 404
    # 空动作名 → 400
    assert app_client.post("/api/assistant/confirm",
                           json={"session_key": "x", "action": "  "}).status_code == 400


def test_mcp_action_tools_gated_by_mode(monkeypatch):
    monkeypatch.setattr(knowledge_mcp, "_SESSION_KEY", "sess")
    monkeypatch.setattr(knowledge_mcp, "_MCP_MODE", "read")
    names = {t["name"] for t in knowledge_mcp._tools_for_session()}
    assert "platform_create_project" not in names
    # 只读模式即便被直接调用也拒绝
    with pytest.raises(ValueError):
        knowledge_mcp._dispatch_tool("platform_create_project", {"name": "x"})

    monkeypatch.setattr(knowledge_mcp, "_MCP_MODE", "write")
    names = {t["name"] for t in knowledge_mcp._tools_for_session()}
    assert {"platform_create_project", "platform_create_env", "platform_run_analyze"} <= names

    # 无会话键（未挂助手上下文）也不暴露动作工具
    monkeypatch.setattr(knowledge_mcp, "_SESSION_KEY", "")
    names = {t["name"] for t in knowledge_mcp._tools_for_session()}
    assert "platform_create_project" not in names


def test_mcp_run_action_executes_only_after_confirmation(monkeypatch):
    calls: dict = {}

    def fake_http(method, path, payload=None):
        calls.update(method=method, path=path, payload=payload)
        return {"project_id": "p1", "status": "loading"}

    monkeypatch.setattr(knowledge_mcp, "_http_json", fake_http)
    monkeypatch.setattr(knowledge_mcp, "_request_confirmation", lambda a, p: True)
    out = knowledge_mcp._run_action("create_project", {"name": "demo", "source_url": "http://x"})
    assert out["executed"] is True
    assert calls["path"] == "/api/projects" and calls["payload"]["project_type"] == "original"


def test_mcp_run_action_denied_does_not_call_backend(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("未确认不应调后端执行")

    monkeypatch.setattr(knowledge_mcp, "_http_json", boom)
    monkeypatch.setattr(knowledge_mcp, "_request_confirmation", lambda a, p: False)
    out = knowledge_mcp._run_action("run_analyze", {"project_id": "p"})
    assert out["executed"] is False and "未确认" in out["reason"]
