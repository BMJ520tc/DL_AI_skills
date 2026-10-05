"""前端 AI 助手（《新增需求补充》补充 A + 用户 2026-10-05 的增强）。

- **流式**：`agent_service.run_stream`（正文增量 + 工具调用）经 SSE 逐条推给前端；
- **会话续接**：多轮带 `session_id`（SDK `resume`），上下文连续；
- **模式**：`read`（**只读**：Read/Glob/Grep + 平台只读 MCP 工具，**不给** Write/Edit/Bash）/
  `write`（满工具）。两种模式都**不做平台不可逆动作**（训练/入库/删除/建环境）。
- 无凭证时**如实失败**（规矩 7），不伪造。

对话仍经任务队列执行（留痕），SSE 从进程内的事件流取事件。
"""
import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from app.config import BACKEND_DIR, DATA_DIR
from app.services import agent_service, project_manager, task_manager

TASK_TYPE = "assistant_chat"
REPO_ROOT = Path(BACKEND_DIR).parent

# 只读模式的工具集（写模式 = agent_service.DEFAULT_ALLOWED_TOOLS 满工具）
READ_TOOLS = ["Read", "Glob", "Grep"]
MODES = ("read", "write")

# 确认登记表：pending 超时即判 expired；已决/过期记录再多留一会供轮询取走，之后清理
CONFIRM_TTL_S = float(os.getenv("ASSISTANT_CONFIRM_TTL_S", "600"))
CONFIRM_KEEP_S = 120.0

_SYSTEM = """你是 DL-AI-skills 平台内的 AI 助手。遵守：
1) 你可以像开发者一样**用工具实地查看**——读代码与数据文件、查平台状态（list_projects / get_project /
   list_runs）与知识库（knowledge_search）。**凡涉及事实，先查再答，不要凭印象猜。**
2) **不可逆/昂贵的动作**（发起训练 / 入库 / 删除项目 / 回滚版本 / 改画布）**不要做**——
   只指引用户去对应界面操作；
3) 在「可写」模式下，你可以调用**平台动作工具**：建原始项目（platform_create_project）、
   建运行环境（platform_create_env）、结构分析（platform_run_analyze）。这些工具会**自动弹窗请用户确认**，
   用户点了「确认」才会执行——所以你**可以主动提议**，用户同意后再调用；调用前先说清打算做什么、为什么。
   用户取消时工具会返回「未确认」，不要重试、不要绕过。
4) **仓库与数据目录一律只读**（你只能读它们，不许改）；若确需写文件，**只写在你自己的工作目录里**
   （会话 scratch）。任何会改动磁盘或环境的动作，**先说明你打算做什么、为什么**，再做；
5) 不确定就如实说不确定；信息不足就说清缺什么。回答用中文、简洁、可执行。"""

# task_id → 事件队列（None 为流结束哨兵）
_STREAMS: dict[str, "asyncio.Queue[Optional[dict]]"] = {}
# 会话 key → CLI session id（常驻客户端被丢弃后靠它 resume 续接上下文）
_CONV: dict[str, str] = {}
# 会话 key → 当前活动队列：动作工具的确认事件经它推给前端（③续）
_SESSION_QUEUE: dict[str, "asyncio.Queue[Optional[dict]]"] = {}
# confirm_id → {status, action, params, session_key, ts}（进程内；助手确认握手）
_CONFIRMS: dict[str, dict] = {}


def _context_digest(context: dict) -> str:
    """上下文摘要（只读）：平台项目概览 + 当前项目一行。取不到就静默跳过。"""
    lines: list[str] = []
    try:
        projects = project_manager.list_projects()
        orig = sum(1 for p in projects if p.get("project_type") == "original")
        struct = sum(1 for p in projects if p.get("project_type") == "structured")
        lines.append(f"[平台现状] 项目共 {len(projects)} 个（original {orig} / structured {struct}）：")
        for p in projects[:20]:
            lines.append(f"- {p['project_id']} {p.get('name')} ({p.get('project_type')}/{p.get('status')})")
        if len(projects) > 20:
            lines.append("- …（仅列前 20 个，完整列表用 list_projects 工具查）")
    except Exception:
        pass
    pid = (context or {}).get("project_id")
    if pid:
        try:
            proj = project_manager.get_project(pid)
            if proj:
                lines.append(f"[当前项目] {pid} 「{proj.get('name')}」"
                             f"（{proj.get('project_type')}/{proj.get('status')}）")
        except Exception:
            pass
    return "\n".join(lines)


def _build_prompt(message: str, context: Optional[dict]) -> str:
    ctx = context or {}
    lines = ["[当前上下文]"]
    for key in ("page", "project_id", "project_name", "paper_id", "network_id"):
        if ctx.get(key):
            lines.append(f"- {key}: {ctx[key]}")
    if len(lines) == 1:
        lines.append("- （前端未提供上下文）")
    digest = _context_digest(ctx)
    body = _SYSTEM + "\n" + "\n".join(lines)
    if digest:
        body += "\n" + digest
    return body + f"\n\n[用户问题]\n{message}\n"


def _tools_for_mode(mode: Optional[str]) -> list[str]:
    # 可写模式 = 读 + 写/改（**不含 Bash**：命令里的路径静态判不准，是绕过路径闸门的口子）
    return list(READ_TOOLS) + ["Write", "Edit"] if mode == "write" else list(READ_TOOLS)


def _mcp_tools_for_mode(mode: Optional[str]) -> tuple[str, ...]:
    """MCP 白名单子集：只读恒有；动作工具（建项目/建环境/结构分析）仅在可写模式。"""
    tools = list(agent_service.KNOWLEDGE_MCP_READ_TOOLS)
    if mode == "write":
        tools += list(agent_service.KNOWLEDGE_MCP_ACTION_TOOLS)
    return tuple(tools)


def _sweep_confirms() -> None:
    """惰性清理确认登记表：pending 超时判 expired；过期后保留 CONFIRM_KEEP_S 供轮询取走。"""
    now = time.time()
    for confirm_id in list(_CONFIRMS):
        rec = _CONFIRMS[confirm_id]
        age = now - rec.get("ts", 0.0)
        if rec["status"] == "pending" and age > CONFIRM_TTL_S:
            rec["status"] = "expired"
        if age > CONFIRM_TTL_S + CONFIRM_KEEP_S:
            _CONFIRMS.pop(confirm_id, None)


def request_confirmation(session_key: str, action: str, params: dict) -> str:
    """登记一条待确认的**平台动作**并推给该会话的前端；返回 confirm_id。

    无活动会话（流已结束/会话键未知）→ LookupError（动作不得执行）。
    """
    _sweep_confirms()
    q = _SESSION_QUEUE.get(session_key or "")
    if q is None:
        raise LookupError("没有活动的助手会话可接收确认（会话可能已结束）")
    confirm_id = uuid.uuid4().hex
    _CONFIRMS[confirm_id] = {"status": "pending", "action": action, "params": params or {},
                             "session_key": session_key, "ts": time.time()}
    q.put_nowait({"kind": "confirm", "confirm_id": confirm_id,
                  "action": action, "params": params or {}})
    return confirm_id


def confirmation_status(confirm_id: str) -> Optional[dict]:
    _sweep_confirms()
    rec = _CONFIRMS.get(confirm_id)
    if rec is None:
        return None
    return {"confirm_id": confirm_id, "status": rec["status"],
            "action": rec["action"], "params": rec["params"]}


def decide_confirmation(confirm_id: str, approved: bool) -> dict:
    """用户对确认请求的裁定（前端调用）。已决的请求不改写（幂等）。"""
    rec = _CONFIRMS.get(confirm_id)
    if rec is None:
        raise LookupError("确认请求不存在或已过期")
    if rec["status"] == "pending":
        rec["status"] = "approved" if approved else "denied"
    return {"confirm_id": confirm_id, "status": rec["status"]}


# CLI 未登录/凭证失效时会把这些原文当"回复"或报错吐出来 —— 对用户没有指引性，译制掉
_AUTH_MARKERS = ("not logged in", "please run /login", "invalid api key",
                 "authentication_error", "invalid x-api-key", "unauthorized")
CREDENTIAL_HINT = ("尚未配置模型接口凭证（或凭证已失效）：请点右下「⚙ 设置 → 模型接口凭证」"
                   "填写 API Key 与接口地址后重试。")


def _looks_unauthenticated(text: Optional[str]) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _AUTH_MARKERS)


# 续接目标失效（后端重启 / 会话过期）——不该报错给用户，自动丢掉续接重开
_STALE_SESSION_MARKERS = ("no conversation found", "session not found", "no session found",
                          "conversation not found")


def _looks_stale_session(text: Optional[str]) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _STALE_SESSION_MARKERS)


def start(message: str, context: Optional[dict] = None, session_id: Optional[str] = None,
          mode: str = "read") -> str:
    """发起一次对话，返回 task_id；事件经 GET /api/assistant/chat/{task_id}/stream（SSE）取。"""
    if mode not in MODES:
        mode = "read"
    task_id = task_manager.create_task(TASK_TYPE, params={
        "message": message, "context": context or {}, "session_id": session_id, "mode": mode})
    _STREAMS[task_id] = asyncio.Queue()
    return task_id


async def stream(task_id: str):
    """异步产出事件：kind ∈ stage / delta / tool / done / error。流结束即退出。"""
    q = _STREAMS.get(task_id)
    if q is None:
        yield {"kind": "error", "message": "无此对话（或已结束）"}
        return
    try:
        while True:
            item = await q.get()
            if item is None:
                break
            yield item
    finally:
        _STREAMS.pop(task_id, None)


def register() -> None:
    task_manager.register_handler(TASK_TYPE, _run)


async def _run(params: dict, task_id: str) -> None:
    message = str(params.get("message") or "").strip()
    if not message:
        raise RuntimeError("空消息")
    q = _STREAMS.get(task_id)

    def emit(kind: str, data: dict) -> None:
        if q is not None:
            q.put_nowait({"kind": kind, **data})

    emit("stage", {"text": "思考中…"})
    conv_key = params.get("session_id")            # 前端回传的会话 key（首轮为空）
    mode = params.get("mode") or "read"
    # 稳定的会话键：首轮生成、随 done 回传前端持有 → 后续轮与 MCP 子进程都用同一个
    cid = conv_key or uuid.uuid4().hex
    resume_id = _CONV.get(cid)                     # 客户端被丢弃时用它续接

    # 隔离：agent 的工作目录 = 本会话独立 scratch —— 它的自动记忆与相对写入都落这里，
    # 不碰仓库、也不碰 ~/.claude。仓库仅经 add_dirs 供它**读**（提示词里要求仓库/数据只读）。
    scratch = DATA_DIR / "assistant_sessions" / cid
    try:
        scratch.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    # 确认通道：动作工具的确认事件经本会话当前队列推给前端
    if q is not None:
        _SESSION_QUEUE[cid] = q

    state = {"resume": resume_id}

    def options_factory():
        return agent_service.stream_options(
            tools=_tools_for_mode(mode), resume=state["resume"], max_turns=20,
            cwd=str(scratch), add_dirs=[str(REPO_ROOT)],
            # 硬闸门：读限「仓库 / 数据 / 本会话工作目录」，写只限工作目录，Bash 不给
            permission_mode="default",
            # 可写模式：MCP 白名单加动作工具，且让 MCP 子进程按 write 模式暴露它们
            mcp_tools=_mcp_tools_for_mode(mode), session_key=cid, mcp_mode=mode,
            can_use_tool=agent_service.path_gate_tool(
                read_roots=[REPO_ROOT, DATA_DIR, scratch], write_roots=[scratch], allow_bash=False))

    try:
        prompt = _build_prompt(message, params.get("context"))
        try:
            result = await agent_service.run_client_stream(
                prompt, conv_key=cid, mode_key=mode, options_factory=options_factory, on_event=emit)
        except Exception as exc:  # noqa: BLE001
            if state["resume"] and _looks_stale_session(str(exc)):
                # 续接目标已失效（后端重启 / 会话过期）→ 丢掉续接重开一次，不把错误甩给用户
                state["resume"] = None
                _CONV.pop(cid, None)
                result = await agent_service.run_client_stream(
                    prompt, conv_key=cid, mode_key=mode, options_factory=options_factory, on_event=emit)
            else:
                raise
        reply = str(result.get("result") or "").strip()
        if _looks_unauthenticated(reply):
            # CLI 未登录/凭证失效会把提示原文当回复吐出来 → 译制成可照着做的指引（规矩 7）
            raise RuntimeError(CREDENTIAL_HINT)
        if not reply:
            # 不是凭证问题（凭证问题在上面已单列）——多半是工具被拒/步数用尽/端点空回复
            raise RuntimeError("助手这一轮没有产出内容（可能是工具调用被拒、步数用尽或端点空回复）；"
                               "请重试一次或换个问法。上下文仍在，可直接接着说。")
        new_key = result.get("conv_key") or cid
        if new_key and result.get("session_id"):
            _CONV[new_key] = result["session_id"]
        task_manager.update_progress(task_id, {
            "stage": "完成", "reply": reply, "session_id": new_key})
        emit("done", {"reply": reply, "session_id": new_key})
    except Exception as exc:  # noqa: BLE001 —— 如实失败，不伪造
        msg = CREDENTIAL_HINT if _looks_unauthenticated(str(exc)) else str(exc)
        task_manager.update_progress(task_id, {"stage": "失败", "error": msg})
        emit("error", {"message": msg})
        raise RuntimeError(msg) from exc
    finally:
        _SESSION_QUEUE.pop(cid, None)
        if q is not None:
            q.put_nowait(None)
