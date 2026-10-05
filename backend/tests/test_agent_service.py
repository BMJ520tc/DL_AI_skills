"""agent 会话选项与知识库 MCP 白名单（《模块详细设计》2.5、M0 冒烟缺陷回归）。

背景（2026-10-01 M0 MCP 冒烟测试暴露）：`permission_mode="dontAsk"` 下
「未预授权的工具直接拒绝，can_use_tool 不会被咨询」，而知识库 MCP 工具
只被挂载、从未进入白名单，导致 agent 查知识库必然被拒。
修复：挂载知识库 MCP 时把 `mcp__knowledge__knowledge_search` 加入 allowed_tools。
"""
from __future__ import annotations

import pytest

from app.services import agent_service


def test_mcp_tool_rule_name_matches_cli_convention():
    assert agent_service.KNOWLEDGE_MCP_TOOL_RULE == "mcp__knowledge__knowledge_search"
    assert agent_service.KNOWLEDGE_MCP_TOOL_RULE.startswith(
        f"mcp__{agent_service.KNOWLEDGE_MCP_SERVER}__"
    )


def test_allowed_tools_includes_mcp_rule_when_attached():
    tools = agent_service.allowed_tools(attach_knowledge=True)
    assert agent_service.KNOWLEDGE_MCP_TOOL_RULE in tools
    for builtin in agent_service.DEFAULT_ALLOWED_TOOLS:
        assert builtin in tools
    # 默认白名单本身不含 MCP 规则（由 helper 统一注入，避免漏挂）
    assert agent_service.KNOWLEDGE_MCP_TOOL_RULE not in agent_service.DEFAULT_ALLOWED_TOOLS


def test_allowed_tools_excludes_mcp_rule_when_not_attached():
    assert agent_service.KNOWLEDGE_MCP_TOOL_RULE not in agent_service.allowed_tools(attach_knowledge=False)


def test_allowed_tools_respects_explicit_base_list():
    tools = agent_service.allowed_tools(True, ["Read"])
    # 挂载知识库 MCP 时，其**全部只读工具**都进白名单（knowledge_search + 平台只读查询）
    assert tools == ["Read", *agent_service.KNOWLEDGE_MCP_TOOL_RULES]
    # 显式传入的列表不被就地修改
    base = ["Read"]
    agent_service.allowed_tools(True, base)
    assert base == ["Read"]


def test_knowledge_mcp_config_is_stdio_module_entrypoint():
    cfg = agent_service._knowledge_mcp()
    assert set(cfg) == {"knowledge"}
    server = cfg["knowledge"]
    assert server["type"] == "stdio"
    assert server["args"] == ["-m", "app.mcp.knowledge_mcp"]
    assert "PYTHONPATH" in server["env"]


def test_build_options_attaches_mcp_and_whitelists_tool():
    options = agent_service._build_options({"attach_knowledge": True})
    assert options.mcp_servers == agent_service._knowledge_mcp()
    assert agent_service.KNOWLEDGE_MCP_TOOL_RULE in options.allowed_tools
    assert options.permission_mode == "dontAsk"      # 安全默认不变


def test_build_options_without_knowledge_does_not_attach():
    options = agent_service._build_options({"attach_knowledge": False})
    assert not options.mcp_servers
    assert agent_service.KNOWLEDGE_MCP_TOOL_RULE not in options.allowed_tools


def test_build_options_defaults_to_attach_knowledge():
    """默认挂载知识库（2.5 每会话临时挂载），且白名单同步包含其工具。"""
    options = agent_service._build_options({})
    assert options.mcp_servers == agent_service._knowledge_mcp()
    assert agent_service.KNOWLEDGE_MCP_TOOL_RULE in options.allowed_tools


def test_build_options_respects_caller_allowed_tools():
    options = agent_service._build_options({"attach_knowledge": True, "allowed_tools": ["Read", "Grep"]})
    assert agent_service.KNOWLEDGE_MCP_TOOL_RULE in options.allowed_tools
    assert "Read" in options.allowed_tools and "Grep" in options.allowed_tools
    assert "Bash" not in options.allowed_tools


# ---------------------------------------------------- 按任务覆盖模型 / 端点（拆解专用）

def test_build_options_model_override_and_env():
    """`model` / `model_env` 写进 options（供「拆解单独走强模型/端点」）；基线 env 不被覆盖掉。"""
    options = agent_service._build_options({
        "model": "claude-sonnet-4-6",
        "model_env": {"ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                      "ANTHROPIC_API_KEY": "sk-ant-x"},
    })
    assert options.model == "claude-sonnet-4-6"
    assert options.env["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert options.env["ANTHROPIC_API_KEY"] == "sk-ant-x"
    assert options.env["DISABLE_AUTOUPDATER"] == "1"     # 基线 env 保留


def test_build_options_without_override_leaves_env_clean():
    options = agent_service._build_options({})
    assert "ANTHROPIC_BASE_URL" not in (options.env or {})
    assert options.model == agent_service.DEFAULT_MODEL


# ---------------------------------------------------- 未登录/凭证失效：立即中止（不重试）

def test_raise_if_agent_error_classifies_not_logged_in():
    """CLI 未登录被当「回复文本」返回 → 译制后抛出（此前被当成「没产出」白试十几轮）。"""
    with pytest.raises(RuntimeError, match="未登录|凭证"):
        agent_service.raise_if_agent_error("Not logged in · Please run /login")
    with pytest.raises(RuntimeError, match="未登录|凭证"):
        agent_service.raise_if_agent_error("Please run /login")
    # API 层错误优先译制
    with pytest.raises(RuntimeError, match="余额不足"):
        agent_service.raise_if_agent_error("API Error: 402 Insufficient Balance")


def test_raise_if_agent_error_passes_through_normal_output():
    agent_service.raise_if_agent_error("已产出 IR：39 节点 / 30 边")
    agent_service.raise_if_agent_error(None)


def test_is_auth_error_markers():
    assert agent_service.is_auth_error("Not logged in · Please run /login")
    assert agent_service.is_auth_error("API Error: 401 invalid x-api-key")
    assert not agent_service.is_auth_error("已将 IR 写入 result.json")
