"""agent 调度：Claude Agent SDK 调用 Claude Code（模块详细设计 2.5，D3，O2，架构八.1）。

任务式定义：每个 agent 任务 = 一次 Claude Code 会话（prompt + 工作目录 + 工具白名单
+ JSON Schema 结构化输出 + 轮数上限 + 每会话临时挂载知识库 MCP）。
输入输出与日志落盘到 data/agent_tasks/{task_id}/，可重放、可回溯（N3）。
"""
import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Optional

from claude_agent_sdk import ClaudeAgentOptions, PermissionResultAllow, PermissionResultDeny, ResultMessage, query

from app.config import AGENT_TASKS_DIR, BACKEND_DIR, CLAUDE_CLI_PATH, DEFAULT_MODEL
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
# CLI 权限规则中 MCP 工具用全名 mcp__<server>__<tool>
KNOWLEDGE_MCP_TOOL_RULE = f"mcp__{KNOWLEDGE_MCP_SERVER}__{KNOWLEDGE_MCP_TOOL}"

DEFAULT_TIMEOUT_S = 900


def allowed_tools(attach_knowledge: bool = True, base: Optional[list[str]] = None) -> list[str]:
    """本会话工具白名单：显式白名单 + 挂载知识库时的知识库 MCP 工具。

    设计依据《模块详细设计》2.5「工具白名单：…查询知识库（经每会话临时挂载的 MCP）；
    deny 优先，白名单外不提供」。dontAsk 模式下未预授权的工具会被直接拒绝，
    所以「挂载了 MCP」不等于「允许调用」，必须把工具全名加入白名单。
    """
    tools = list(base) if base is not None else list(DEFAULT_ALLOWED_TOOLS)
    if attach_knowledge and KNOWLEDGE_MCP_TOOL_RULE not in tools:
        tools.append(KNOWLEDGE_MCP_TOOL_RULE)
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


def _task_dir(task_id: str) -> Path:
    d = AGENT_TASKS_DIR / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _knowledge_mcp() -> dict:
    """每会话临时挂载的知识库 stdio MCP server。"""
    return {
        "knowledge": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "app.mcp.knowledge_mcp"],
            "env": {"PYTHONPATH": str(BACKEND_DIR)},
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
        env={"DISABLE_AUTOUPDATER": "1"},
    )
    if attach:
        options.mcp_servers = _knowledge_mcp()
    if DEFAULT_MODEL:
        options.model = DEFAULT_MODEL
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
    }
    return task_manager.create_task(AGENT_TASK_TYPE, params=params)


def get_result(task_id: str) -> dict:
    task = task_manager.get_task(task_id)
    result: dict = {"task": task}
    result_path = _task_dir(task_id) / "result.json"
    if result_path.exists():
        result["result"] = json.loads(result_path.read_text(encoding="utf-8"))
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
        outcome = await _collect(prompt, options, params.get("timeout_s", DEFAULT_TIMEOUT_S))
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

    result_path.write_text(json.dumps(structured or {}, ensure_ascii=False), encoding="utf-8")


async def _collect(prompt: str, options: ClaudeAgentOptions, timeout_s: int, retries: int = 2) -> dict:
    """执行一次 agent 会话，带指数退避重试（2.5 异常兜底：429 退避、超时重试）。"""
    outcome: dict = {}
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:

            async def _iterate() -> None:
                async for msg in query(prompt=prompt, options=options):
                    if isinstance(msg, ResultMessage):
                        outcome["structured_output"] = msg.structured_output
                        outcome["result"] = msg.result
                        outcome["num_turns"] = msg.num_turns
                        outcome["stop_reason"] = getattr(msg, "stop_reason", None)
                        outcome["errors"] = msg.errors
                        outcome["permission_denials"] = msg.permission_denials
                        return

            await asyncio.wait_for(_iterate(), timeout=timeout_s)
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
) -> dict:
    """直接执行一次 agent 会话（不走任务队列），供其他服务的 handler 内部调用。

    返回 {"structured_output": ..., "result": ...}；结构化输出在 DeepSeek 下走文件兜底。
    attach_knowledge=True 时挂载知识库 MCP 并把其工具加入白名单（默认不挂载，保持既有调用方行为）。
    """
    d = _task_dir(uuid.uuid4().hex)
    result_path = d / "result.json"

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
        env={"DISABLE_AUTOUPDATER": "1"},
    )
    if attach_knowledge:
        options.mcp_servers = _knowledge_mcp()
    if DEFAULT_MODEL:
        options.model = DEFAULT_MODEL

    outcome = await _collect(prompt, options, timeout_s)
    structured = outcome.get("structured_output")
    if structured is None and result_path.exists():
        try:
            structured = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            structured = None
    return {"structured_output": structured, "result": outcome.get("result")}
