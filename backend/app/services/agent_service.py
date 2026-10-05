"""agent 调度：Claude Agent SDK 调用 Claude Code（模块详细设计 2.5，D3，O2，架构八.1）。

任务式定义：每个 agent 任务 = 一次 Claude Code 会话（prompt + 工作目录 + 工具白名单
+ JSON Schema 结构化输出 + 轮数上限 + 每会话临时挂载知识库 MCP）。
输入输出与日志落盘到 data/agent_tasks/{task_id}/，可重放、可回溯（N3）。
"""
import asyncio
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    query,
)

from app.config import AGENT_TASKS_DIR, BACKEND_DIR, CLAUDE_CLI_PATH, DEFAULT_MODEL
from app.ids import safe_id
from app.services import task_manager

AGENT_TASK_TYPE = "agent_task"

# 工具白名单（deny 优先，白名单外不提供；架构八.1）：
# 读文件(限工作区)、写文件(限工作区)、执行命令(经独立环境)。
# 查询知识库经 mcp_servers 挂载的 knowledge MCP；注意 permission_mode="dontAsk" 下
# 「未预授权的工具直接拒绝、can_use_tool 不会被咨询」（claude_agent_sdk types.py:
# "dontAsk — Deny anything not pre-approved by allow rules"），因此 MCP 工具必须
# 显式进白名单，否则 agent 查知识库会被拒（M0 冒烟测试暴露的问题）。
DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
DEFAULT_DISALLOWED_TOOLS: list[str] = []

KNOWLEDGE_MCP_SERVER = "knowledge"
KNOWLEDGE_MCP_TOOL = "knowledge_search"
# knowledge MCP 暴露的**只读**工具（见 app/mcp/knowledge_mcp.py）：
# 知识库检索 + 平台只读查询（列项目 / 取项目 / 列运行记录）。
KNOWLEDGE_MCP_READ_TOOLS = ("knowledge_search", "list_projects", "get_project", "list_runs")
# 平台**动作**工具（③续）：建原始项目 / 建环境 / 结构分析——**写操作**，需经 UI 确认，
# 仅在助手「可写」模式暴露；默认（只读）工具集不含它们。
KNOWLEDGE_MCP_ACTION_TOOLS = ("platform_create_project", "platform_create_env", "platform_run_analyze")
# 默认（只读）工具集；`KNOWLEDGE_MCP_TOOL_RULES` 保持为只读集，供既有调用方/用例依赖
KNOWLEDGE_MCP_TOOLS = KNOWLEDGE_MCP_READ_TOOLS
# CLI 权限规则中 MCP 工具用全名 mcp__<server>__<tool>
KNOWLEDGE_MCP_TOOL_RULE = f"mcp__{KNOWLEDGE_MCP_SERVER}__{KNOWLEDGE_MCP_TOOL}"
KNOWLEDGE_MCP_TOOL_RULES = [f"mcp__{KNOWLEDGE_MCP_SERVER}__{t}" for t in KNOWLEDGE_MCP_TOOLS]


def mcp_tool_rule(name: str) -> str:
    """MCP 工具全名（CLI 权限规则口径）。"""
    return f"mcp__{KNOWLEDGE_MCP_SERVER}__{name}"

DEFAULT_TIMEOUT_S = 900
# run_sync 的临时目录前缀与保留数：这些目录没有 task 记录可查，长期不清会让 data/agent_tasks 膨胀
SYNC_DIR_PREFIX = "sync_"
SYNC_KEEP_DIRS = int(os.getenv("AGENT_SYNC_KEEP_DIRS", "200"))


def allowed_tools(attach_knowledge: bool = True, base: Optional[list[str]] = None,
                  mcp_tools: Optional[tuple[str, ...]] = None) -> list[str]:
    """本会话工具白名单：显式白名单 + 挂载知识库时的知识库 MCP 工具。

    设计依据《模块详细设计》2.5「工具白名单：…查询知识库（经每会话临时挂载的 MCP）；
    deny 优先，白名单外不提供」。dontAsk 模式下未预授权的工具会被直接拒绝，
    所以「挂载了 MCP」不等于「允许调用」，必须把工具全名加入白名单。

    `mcp_tools` 指定要进白名单的 MCP 工具名（默认＝只读集）；助手「可写」模式传
    只读 + 动作集（③续）。
    """
    tools = list(base) if base is not None else list(DEFAULT_ALLOWED_TOOLS)
    if attach_knowledge:
        for name in (mcp_tools if mcp_tools is not None else KNOWLEDGE_MCP_TOOLS):
            rule = mcp_tool_rule(name)
            if rule not in tools:
                tools.append(rule)
    return tools

_DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+[/~]",      # rm -rf / 或 ~
    r"\bmkfs\b",             # 格式化
    r"\bdd\b.*of=/dev/",     # 写块设备
    r":\(\)\s*\{",           # fork bomb
    r"\b(shutdown|reboot|halt)\b",
    r">\s*/dev/sd",          # 写磁盘设备
]


async def _can_use_tool(tool_name: str, tool_input: dict, context) -> PermissionResultAllow | PermissionResultDeny:
    """can_use_tool 回调：限制 Bash 危险命令（架构八.1 执行命令安全，deny 优先）。

    完整「经 conda run/docker exec 限独立环境」的前缀校验需在独立环境就位后收紧，
    现阶段先拦截破坏性命令。
    """
    if tool_name == "Bash":
        command = str(tool_input.get("command", ""))
        for pattern in _DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return PermissionResultDeny(message=f"拒绝危险命令: {command[:120]}")
    return PermissionResultAllow()


def path_gate_tool(read_roots: list, write_roots: list, allow_bash: bool = False):
    """构造**路径白名单**版 `can_use_tool`（助手用；配 `permission_mode="default"`）。

    读只允许 read_roots 之下；写只允许 write_roots 之下；Bash 默认不给；
    其余（含 MCP 只读工具）放行。做到「仓库/数据只读、~/.claude 与用户目录碰不到」。
    """

    def _norm(p) -> Path:
        try:
            return Path(str(p)).resolve()
        except Exception:  # noqa: BLE001
            return Path(str(p))

    r_roots = [_norm(r) for r in read_roots]
    w_roots = [_norm(r) for r in write_roots]

    def _under(path: Path, roots: list) -> bool:
        for root in roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    async def _check(tool_name: str, tool_input: dict, context):
        ti = tool_input or {}
        if tool_name in ("Read", "Glob", "Grep"):
            raw = ti.get("file_path") or ti.get("path")
            if not raw:                       # Glob/Grep 缺 path 时默认落在工作目录
                return PermissionResultAllow()
            if _under(_norm(raw), r_roots):
                return PermissionResultAllow()
            return PermissionResultDeny(
                message=f"助手只能读「仓库 / 项目数据 / 本会话工作目录」，已拒绝：{raw}")
        if tool_name in ("Write", "Edit", "NotebookEdit"):
            raw = ti.get("file_path")
            if raw and _under(_norm(raw), w_roots):
                return PermissionResultAllow()
            return PermissionResultDeny(
                message="助手只能写自己的会话工作目录；仓库与数据目录只读，请去对应界面操作")
        if tool_name == "Bash":
            if allow_bash:
                return PermissionResultAllow()
            return PermissionResultDeny(message="助手不提供命令行执行（请用只读工具，或去界面操作）")
        return PermissionResultAllow()

    return _check


# 续接目标失效（后端重启 / 会话过期）的原文特征——续接失败时据此回退到不带 resume 重跑
_STALE_SESSION_MARKERS = ("no conversation found", "session not found", "no session found",
                          "conversation not found")


def is_stale_session_error(text) -> bool:
    low = str(text or "").lower()
    return any(m in low for m in _STALE_SESSION_MARKERS)


_API_ERROR_RE = re.compile(r"API Error:\s*(.+?)(?:\s*\(request_id|$)", re.IGNORECASE | re.MULTILINE)


def _api_error_message(text) -> Optional[str]:
    """识别「CLI 把 API 层错误当回复文本返回」的情况（实测：`API Error: 402 Insufficient Balance`）。"""
    m = _API_ERROR_RE.search(str(text or ""))
    return m.group(1).strip() if m else None


def _api_error_hint(message: str) -> str:
    """把 API 层错误译成可操作的提示——**这类错误重试没有意义，必须如实抛出并中止**。"""
    low = message.lower()
    if "insufficient balance" in low or "402" in low:
        return (f"模型接口余额不足（{message}）：请充值或更换凭证后重试。"
                "**重试无意义，已中止**（此前会把这类错误当成「模型没产出内容」白试十几轮）")
    if "401" in low or "unauthorized" in low or "invalid" in low:
        return f"模型接口鉴权失败（{message}）：请到设置页检查凭证后重试（重试无意义，已中止）"
    if "429" in low or "rate limit" in low:
        return f"模型接口限流（{message}）：请稍后重试（重试无意义，已中止）"
    return f"模型接口错误：{message}"


# CLI **未登录/凭证失效**时会把提示当「回复文本」返回（实测：`Not logged in · Please run /login`）。
# 与 API 层错误同性质：**重试无意义**，必须立即中止并给可照做的指引——否则会被上层当成
# 「模型没产出内容」反复重试（2026-10-05 实测：拆解因此白试 12 轮，报成「structured_output 缺失」）。
# 判据与文案由本模块统一提供，assistant_service 复用同一份标记。
AUTH_ERROR_MARKERS = ("not logged in", "please run /login", "invalid api key",
                      "authentication_error", "invalid x-api-key", "unauthorized")
_CREDENTIAL_HINT = ("模型接口未登录或凭证已失效：请到「设置 → 模型接口凭证」填写 API Key 与接口地址，"
                    "或给后端进程设置 ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY（也可先执行 claude 登录）。"
                    "**重试无意义，已中止**")


def is_auth_error(text) -> bool:
    """CLI 的「未登录/凭证失效」提示（当作回复文本返回）识别。"""
    low = str(text or "").lower()
    return any(m in low for m in AUTH_ERROR_MARKERS)


def raise_if_agent_error(result_text) -> None:
    """把 CLI 当「回复文本」返回的**终止性错误**译制后抛出（这类重试无意义）。

    先判 API 层错误（余额/鉴权/限流），再判未登录/凭证失效；都不是则什么也不做。
    """
    api_err = _api_error_message(result_text)
    if api_err:
        raise RuntimeError(_api_error_hint(api_err))
    if is_auth_error(result_text):
        raise RuntimeError(_CREDENTIAL_HINT)


def _emit_tool_uses(msg, on_event) -> None:
    """把一次 AssistantMessage 里的工具调用回调给 on_event（供拆解等长任务报进度）。

    形如 on_event("tool", {"name": "Read"})；回调异常不影响会话。
    """
    if on_event is None or type(msg).__name__ != "AssistantMessage":
        return
    for blk in (getattr(msg, "content", None) or []):
        if type(blk).__name__ == "ToolUseBlock":
            try:
                on_event("tool", {"name": getattr(blk, "name", "")})
            except Exception:  # noqa: BLE001
                pass


def _prune_sync_dirs() -> None:
    """清理 run_sync 自建的临时目录（sync_*），按 mtime 保留最近 SYNC_KEEP_DIRS 个。

    只动 sync_* 前缀（无 task 记录、纯临时）；任务路径的 {task_id} 目录属可回溯产物，不在此列。
    """
    try:
        dirs = sorted((p for p in AGENT_TASKS_DIR.glob(f"{SYNC_DIR_PREFIX}*") if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    for old in dirs[SYNC_KEEP_DIRS:]:
        shutil.rmtree(old, ignore_errors=True)


def _sync_dir() -> Path:
    d = AGENT_TASKS_DIR / f"{SYNC_DIR_PREFIX}{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    _prune_sync_dirs()
    return d


def _task_dir(task_id: str) -> Path:
    d = AGENT_TASKS_DIR / safe_id(task_id, "task_id")  # 防路径穿越（task_id 来自 API 路径参数）
    d.mkdir(parents=True, exist_ok=True)
    return d


# 助手 MCP 子进程回连后端的地址（确认握手经 HTTP 回环；本机默认端口 8000）
BACKEND_SELF_URL = os.getenv("DL_AI_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _knowledge_mcp(session_key: Optional[str] = None, mode: Optional[str] = None) -> dict:
    """每会话临时挂载的知识库 stdio MCP server。

    传 `session_key`（助手会话键）时，把会话上下文注入 MCP 子进程环境：动作工具要靠
    `ASSISTANT_SESSION_KEY` 经 HTTP 回环向后端请求用户确认，`ASSISTANT_MODE` 决定是否
    暴露动作工具（只读模式不给）。
    """
    env = {"PYTHONPATH": str(BACKEND_DIR)}
    if session_key:
        env["ASSISTANT_SESSION_KEY"] = session_key
        env["ASSISTANT_MODE"] = mode or "read"
        env["DL_AI_BACKEND_URL"] = BACKEND_SELF_URL
    return {
        "knowledge": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "app.mcp.knowledge_mcp"],
            "env": env,
        }
    }


def _build_options(params: dict) -> ClaudeAgentOptions:
    """构造一次会话的 SDK 选项（白名单含知识库 MCP 工具、按需挂载 MCP、模型端点）。"""
    attach: bool = bool(params.get("attach_knowledge", True))
    options = ClaudeAgentOptions(
        cli_path=CLAUDE_CLI_PATH,
        cwd=params.get("cwd"),
        add_dirs=params.get("add_dirs") or [],
        allowed_tools=allowed_tools(attach, params.get("allowed_tools") or DEFAULT_ALLOWED_TOOLS),
        disallowed_tools=params.get("disallowed_tools") or DEFAULT_DISALLOWED_TOOLS,
        permission_mode=params.get("permission_mode", "dontAsk"),
        max_turns=params.get("max_turns", 30),
        output_format=params.get("output_schema"),
        setting_sources=[],
        can_use_tool=_can_use_tool,
        env={"DISABLE_AUTOUPDATER": "1", **(params.get("model_env") or {})},
    )
    if attach:
        options.mcp_servers = _knowledge_mcp()
    model = params.get("model")
    if model or DEFAULT_MODEL:
        options.model = model or DEFAULT_MODEL
    return options


def submit(
    prompt: str,
    *,
    cwd: Optional[str] = None,
    add_dirs: Optional[list] = None,
    allowed_tools: Optional[list] = None,
    disallowed_tools: Optional[list] = None,
    output_schema: Optional[dict] = None,
    max_turns: int = 30,
    permission_mode: str = "dontAsk",
    attach_knowledge: bool = True,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    model: Optional[str] = None,
    model_env: Optional[dict] = None,
) -> str:
    params = {
        "prompt": prompt,
        "cwd": cwd,
        "add_dirs": add_dirs or [],
        "allowed_tools": allowed_tools or DEFAULT_ALLOWED_TOOLS,
        "disallowed_tools": disallowed_tools or DEFAULT_DISALLOWED_TOOLS,
        "output_schema": output_schema,
        "max_turns": max_turns,
        "permission_mode": permission_mode,
        "attach_knowledge": attach_knowledge,
        "timeout_s": timeout_s,
        "model": model,
        "model_env": model_env,
    }
    return task_manager.create_task(AGENT_TASK_TYPE, params=params)


def get_result(task_id: str) -> dict:
    r"""取 agent 任务结果；**任务不存在时直接返回**（不建目录、不读文件）。

    原先先 `_task_dir`（内含 mkdir）再判存在性：未知 id 会在 agent_tasks 下留空目录，
    且 `..\` 之类 id 能越出目录建目录；结果文件非法 JSON 还会把 GET 打成 500。
    """
    task = task_manager.get_task(task_id)
    if task is None:
        return {"task": None}
    result: dict = {"task": task}
    try:
        result_path = _task_dir(task_id) / "result.json"
    except ValueError as e:  # task_id 非法（路径穿越类）
        result["error"] = str(e)
        return result
    if result_path.exists():
        try:
            result["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            result["error"] = f"结果文件不可解析: {e}"
    return result


async def _run(params: dict, task_id: str) -> None:
    d = _task_dir(task_id)
    result_path = d / "result.json"
    (d / "input.json").write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8")

    prompt = params["prompt"]
    if params.get("output_schema"):
        # DeepSeek 等端点 output_format 可能不可用（O2），让 agent 同时把结果写文件作为兜底
        prompt = (
            prompt
            + f"\n\n[系统指令] 完成分析后，除正常回复外，请同时把符合 schema 的 JSON 结果写入文件：{result_path}"
        )

    options = _build_options(params)

    structured = None
    outcome: dict = {}
    retries = 3 if params.get("output_schema") else 1
    for _ in range(retries):
        outcome = await _collect(prompt, options, params.get("timeout_s", DEFAULT_TIMEOUT_S),
                                 model_override=params.get("model"))
        # 结果：优先 structured_output，否则读 agent 写的结果文件（DeepSeek 兜底）
        structured = outcome.get("structured_output")
        if structured is None and result_path.exists():
            try:
                structured = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                structured = None
        if structured is not None or not params.get("output_schema"):
            break

    (d / "log.json").write_text(
        json.dumps(
            {
                "result": outcome.get("result"),
                "num_turns": outcome.get("num_turns"),
                "stop_reason": outcome.get("stop_reason"),
                "errors": outcome.get("errors"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if params.get("output_schema") and structured is None:
        # 结构化输出缺失，标记失败，供后续重试/换端点（2.5 异常兜底）
        errors = outcome.get("errors") or []
        denials = outcome.get("permission_denials") or []
        if errors or denials:
            raise RuntimeError(
                f"agent 工具调用失败（errors={len(errors)}, denials={len(denials)}），可能需更换端点或模型"
            )
        raise RuntimeError("agent 未产出符合 schema 的结构化输出")

    # 无 schema 的任务也留下 agent 的文本回复，便于回溯（原来恒写 {}）
    result_path.write_text(
        json.dumps(structured if structured is not None else {"result": outcome.get("result")},
                   ensure_ascii=False),
        encoding="utf-8",
    )


async def _collect(prompt: str, options: ClaudeAgentOptions, timeout_s: int, retries: int = 2,
                   on_event=None, model_override: Optional[str] = None) -> dict:
    """执行一次 agent 会话，带指数退避重试（2.5 异常兜底：429 退避、超时重试）。

    model_override：**按任务指定的模型**（如拆解单独走强模型），优先级最高——高于凭证文件里的
    模型名与全局 DEFAULT_MODEL。
    """
    # 一键封装凭证页（K2）：凭证文件在配置时覆盖进程环境（SDK 的 CLI 子进程继承读取）；
    # 文件里的模型名同时覆盖 options.model（显式保存是用户最新意图）。
    from app import settings_store

    cred = settings_store.apply_credentials_env()
    if model_override:
        options.model = model_override
    elif cred.get("model"):
        options.model = cred["model"]

    outcome: dict = {}
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            saw_result = False

            async def _iterate() -> None:
                nonlocal saw_result
                async for msg in query(prompt=prompt, options=options):
                    _emit_tool_uses(msg, on_event)
                    if isinstance(msg, ResultMessage):
                        saw_result = True
                        outcome["structured_output"] = msg.structured_output
                        outcome["result"] = msg.result
                        outcome["num_turns"] = msg.num_turns
                        outcome["stop_reason"] = getattr(msg, "stop_reason", None)
                        outcome["errors"] = msg.errors
                        outcome["permission_denials"] = msg.permission_denials
                        outcome["session_id"] = getattr(msg, "session_id", None)
                        return

            await asyncio.wait_for(_iterate(), timeout=timeout_s)
            if not saw_result:
                # 会话流结束却没有 ResultMessage：属异常终止，不能当成功返回空结果
                raise RuntimeError("agent 会话未返回 ResultMessage（会话异常终止，无结果可用）")
            # API 层错误（余额不足/鉴权/限流）与未登录/凭证失效都会被 CLI 当成「回复文本」返回：
            # 必须如实抛出中止，否则会被上层当成「模型没产出内容」反复重试（实测白试十几轮）。
            raise_if_agent_error(outcome.get("result"))
            return outcome
        except asyncio.TimeoutError as e:
            last_error = e
        except Exception as e:  # noqa: BLE001 —— 网络/端点瞬时错误可重试
            last_error = e
        if attempt < retries:
            await asyncio.sleep(2 ** attempt)  # 指数退避 1s/2s

    if last_error is not None:
        raise last_error
    return outcome


async def run_stream(
    prompt: str,
    *,
    cwd: Optional[str] = None,
    add_dirs: Optional[list] = None,
    max_turns: int = 20,
    timeout_s: int = 600,
    attach_knowledge: bool = False,
    resume: Optional[str] = None,
    tools: Optional[list[str]] = None,
    on_event=None,
) -> dict:
    """**流式**执行一次 agent 会话（不走任务队列）：on_event(kind, data) 逐条回调。

    kind：`delta`（正文增量 `text`）/ `tool`（工具调用 `name`）/ `error`。
    返回 {"result", "session_id", "errors"}。供前端对话的 SSE 流使用。
    """
    from app import settings_store

    def emit(kind: str, data: dict) -> None:
        if on_event is not None:
            try:
                on_event(kind, data)
            except Exception:  # noqa: BLE001 —— 回调失败不影响会话
                pass

    cred = settings_store.apply_credentials_env()
    options = ClaudeAgentOptions(
        cli_path=CLAUDE_CLI_PATH,
        cwd=cwd,
        add_dirs=add_dirs or [],
        allowed_tools=allowed_tools(attach_knowledge, tools or DEFAULT_ALLOWED_TOOLS),
        disallowed_tools=DEFAULT_DISALLOWED_TOOLS,
        permission_mode="dontAsk",
        max_turns=max_turns,
        setting_sources=[],
        can_use_tool=_can_use_tool,
        env={"DISABLE_AUTOUPDATER": "1"},
        include_partial_messages=True,   # 正文增量（StreamEvent）
    )
    if attach_knowledge:
        options.mcp_servers = _knowledge_mcp()
    if resume:
        options.resume = resume
    if cred.get("model"):
        options.model = cred["model"]
    elif DEFAULT_MODEL:
        options.model = DEFAULT_MODEL

    out: dict = {"result": "", "session_id": None, "errors": []}

    async def _iterate() -> None:
        async for msg in query(prompt=prompt, options=options):
            tname = type(msg).__name__
            if tname == "StreamEvent":
                ev = getattr(msg, "event", None) or {}
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        emit("delta", {"text": delta["text"]})
            elif tname == "AssistantMessage":
                for blk in (getattr(msg, "content", None) or []):
                    if type(blk).__name__ == "ToolUseBlock":
                        emit("tool", {"name": getattr(blk, "name", "")})
            elif tname == "ResultMessage":
                out["result"] = getattr(msg, "result", "") or ""
                out["session_id"] = getattr(msg, "session_id", None)
                out["errors"] = getattr(msg, "errors", None) or []
                return

    try:
        await asyncio.wait_for(_iterate(), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001
        emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        raise
    return out


# ---- 常驻会话：跨轮复用一个 ClaudeSDKClient（免每轮冷启动） ----
_CLIENTS: dict[str, ClaudeSDKClient] = {}
_CLIENT_LAST: dict[str, float] = {}
CLIENT_IDLE_S = float(os.getenv("ASSISTANT_CLIENT_IDLE_S", "900"))


async def _sweep_clients(now: float) -> None:
    for key in list(_CLIENTS):
        if now - _CLIENT_LAST.get(key, 0.0) > CLIENT_IDLE_S:
            client = _CLIENTS.pop(key, None)
            _CLIENT_LAST.pop(key, None)
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:  # noqa: BLE001
                    pass


async def run_client_stream(prompt: str, *, conv_key: Optional[str], mode_key: str,
                            options_factory, on_event=None) -> dict:
    """**常驻会话**流式：跨轮复用同一 `ClaudeSDKClient`（免冷启动）。

    conv_key 为空 = 新会话（自建 key，`done` 回传 `conv_key` 供前端持有）；
    客户端异常即丢弃（下一轮重建，靠 options 里的 `resume` 续接上下文），空闲超时自动断开。
    返回 {"result", "session_id", "conv_key"}。
    """
    now = asyncio.get_event_loop().time()
    await _sweep_clients(now)

    cid = conv_key or uuid.uuid4().hex
    key = f"{cid}::{mode_key}"
    client = _CLIENTS.get(key)
    if client is None:
        client = ClaudeSDKClient(options_factory())
        await client.connect()
        _CLIENTS[key] = client
    _CLIENT_LAST[key] = now

    out: dict = {"result": "", "session_id": None, "conv_key": cid}

    def emit(kind: str, data: dict) -> None:
        if on_event is not None:
            try:
                on_event(kind, data)
            except Exception:  # noqa: BLE001
                pass

    try:
        await client.query(prompt)
        async for msg in client.receive_response():
            tname = type(msg).__name__
            if tname == "StreamEvent":
                ev = getattr(msg, "event", None) or {}
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        emit("delta", {"text": delta["text"]})
            elif tname == "AssistantMessage":
                for blk in (getattr(msg, "content", None) or []):
                    if type(blk).__name__ == "ToolUseBlock":
                        emit("tool", {"name": getattr(blk, "name", "")})
            elif tname == "ResultMessage":
                out["result"] = getattr(msg, "result", "") or ""
                out["session_id"] = getattr(msg, "session_id", None)
                break
    except Exception as exc:  # noqa: BLE001 —— 断连即丢客户端，下一轮重建
        _CLIENTS.pop(key, None)
        _CLIENT_LAST.pop(key, None)
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001
            pass
        emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        raise
    return out


def stream_options(*, tools: Optional[list[str]] = None, attach_knowledge: bool = True,
                   resume: Optional[str] = None, max_turns: int = 20,
                   cwd: Optional[str] = None, add_dirs: Optional[list] = None,
                   permission_mode: str = "dontAsk", can_use_tool=None,
                   mcp_tools: Optional[tuple[str, ...]] = None,
                   session_key: Optional[str] = None,
                   mcp_mode: Optional[str] = None) -> ClaudeAgentOptions:
    """构造**流式/常驻会话**的 options（工具集 + 知识库 MCP + 部分消息 + 续接 + 工作目录）。

    供 `run_client_stream` 的 options_factory 使用；与 `run_stream` 的构造保持一致。
    **cwd 用于把会话隔离到独立 scratch 目录**（助手：自动记忆与相对写入都落那里，不碰仓库）。
    `mcp_tools` 指定 MCP 白名单子集（助手可写模式含动作工具）；`session_key` 为助手会话键，
    传入后 MCP 子进程获得确认握手所需的会话上下文。
    """
    from app import settings_store

    cred = settings_store.apply_credentials_env()
    options = ClaudeAgentOptions(
        cli_path=CLAUDE_CLI_PATH,
        cwd=cwd,
        add_dirs=add_dirs or [],
        allowed_tools=allowed_tools(attach_knowledge, tools or DEFAULT_ALLOWED_TOOLS, mcp_tools),
        disallowed_tools=DEFAULT_DISALLOWED_TOOLS,
        permission_mode=permission_mode,
        max_turns=max_turns,
        setting_sources=[],
        can_use_tool=can_use_tool or _can_use_tool,
        env={"DISABLE_AUTOUPDATER": "1"},
        include_partial_messages=True,
    )
    if attach_knowledge:
        options.mcp_servers = _knowledge_mcp(session_key=session_key, mode=mcp_mode)
    if resume:
        options.resume = resume
    if cred.get("model"):
        options.model = cred["model"]
    elif DEFAULT_MODEL:
        options.model = DEFAULT_MODEL
    return options


def register() -> None:
    task_manager.register_handler(AGENT_TASK_TYPE, _run)


async def run_sync(
    prompt: str,
    *,
    cwd: Optional[str] = None,
    add_dirs: Optional[list] = None,
    output_schema: Optional[dict] = None,
    max_turns: int = 20,
    timeout_s: int = 300,
    attach_knowledge: bool = False,
    resume: Optional[str] = None,
    on_event=None,
    model: Optional[str] = None,
    model_env: Optional[dict] = None,
) -> dict:
    """直接执行一次 agent 会话（不走任务队列），供其他服务的 handler 内部调用。

    返回 {"structured_output": ..., "result": ..., "session_id": ...}；结构化输出在 DeepSeek 下走文件兜底。
    attach_knowledge=True 时挂载知识库 MCP 并把其工具加入白名单（默认不挂载，保持既有调用方行为）。
    resume=会话 id 时**续接该会话**（上下文连续；前端多轮对话用）。
    on_event(kind, data) 可选：逐条回调会话事件（目前仅 `tool`＝工具调用名），供长任务报进度。
    model / model_env：**按任务覆盖模型与端点**（如拆解单独走强模型）——`model` 覆盖模型名，
    `model_env` 覆盖 CLI 子进程环境变量（`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` 等，SDK 会把
    它合并到继承环境之上）。都不给则用全局配置（行为不变）。
    """
    d = _sync_dir()
    result_path = d / "result.json"
    (d / "input.json").write_text(
        json.dumps({"prompt": prompt, "cwd": cwd, "max_turns": max_turns, "timeout_s": timeout_s},
                   ensure_ascii=False),
        encoding="utf-8",
    )

    if output_schema:
        prompt = (
            prompt
            + f"\n\n[系统指令] 完成分析后，除正常回复外，请同时把符合 schema 的 JSON 结果写入文件：{result_path}"
        )

    options = ClaudeAgentOptions(
        cli_path=CLAUDE_CLI_PATH,
        cwd=cwd,
        add_dirs=add_dirs or [],
        allowed_tools=allowed_tools(attach_knowledge, DEFAULT_ALLOWED_TOOLS),
        disallowed_tools=DEFAULT_DISALLOWED_TOOLS,
        permission_mode="dontAsk",
        max_turns=max_turns,
        output_format=output_schema,
        setting_sources=[],
        can_use_tool=_can_use_tool,
        env={"DISABLE_AUTOUPDATER": "1", **(model_env or {})},
    )
    if attach_knowledge:
        options.mcp_servers = _knowledge_mcp()
    if resume:
        options.resume = resume
    if model or DEFAULT_MODEL:
        options.model = model or DEFAULT_MODEL

    outcome = await _collect(prompt, options, timeout_s, on_event=on_event, model_override=model)
    # 留痕：agent 的文本回复/会话 id/错误（此前 run_sync 只留 result.json，排障时看不到模型说了什么）
    try:
        (d / "log.json").write_text(
            json.dumps({"result": outcome.get("result"), "session_id": outcome.get("session_id"),
                        "num_turns": outcome.get("num_turns"), "stop_reason": outcome.get("stop_reason"),
                        "errors": outcome.get("errors")}, ensure_ascii=False),
            encoding="utf-8")
    except OSError:
        pass
    structured = outcome.get("structured_output")
    if structured is None and result_path.exists():
        try:
            structured = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            structured = None
    if structured is None and outcome.get("result") is None:
        # 既无结构化输出、也无文本回复 → 不能静默返回空结果给调用方
        raise RuntimeError("agent 未返回任何结果（structured_output 与 result 均为空）")
    return {"structured_output": structured, "result": outcome.get("result"),
            "session_id": outcome.get("session_id")}
