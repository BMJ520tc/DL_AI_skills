"""agent 会话选项与知识库 MCP 白名单（《模块详细设计》2.5、M0 冒烟缺陷回归）。

背景（2026-10-01 M0 MCP 冒烟测试暴露）：`permission_mode="dontAsk"` 下
「未预授权的工具直接拒绝，can_use_tool 不会被咨询」，而知识库 MCP 工具
只被挂载、从未进入白名单，导致 agent 查知识库必然被拒。
修复：挂载知识库 MCP 时把 `mcp__knowledge__knowledge_search` 加入 allowed_tools。
"""
from __future__ import annotations

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
    assert tools == ["Read", agent_service.KNOWLEDGE_MCP_TOOL_RULE]
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
