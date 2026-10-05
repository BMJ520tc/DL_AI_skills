"""拆解提速与进度可见的回归用例（2026-10-05「改善」：① 进度打点 / ② 重试复用会话 / ④ few-shot）。

锁三条判据：
- few-shot 示例本身必须合法（validate_ir + 再生成自检）——示例失效会把模型带偏；
- 重试第 2 次起，**能续接就只发修正指令**（resume=上一次会话），不能续接退回完整 prompt；
- 进度打点：阶段文案随尝试次数推进，工具调用作为 activity 透出且不冲掉 stage。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.services import agent_service, decompose_service, ir_codegen, project_manager, task_manager
from app.services.ir_schema import validate_ir


# ------------------------------------------------------------ ④ few-shot 示例

def test_ir_example_is_valid():
    """few-shot 示例必须能过结构校验与再生成自检，否则是把模型往坑里带。"""
    ex = decompose_service._IR_EXAMPLE
    for key in ("source_file", "entry_class", "task_type", "input_spec", "root_id", "nodes", "edges"):
        assert key in ex
    assert validate_ir(ex) == []
    ir_codegen.generate(ex)  # 不抛即通过（再生成自检）


def test_decompose_prompt_embeds_full_example():
    prompt = decompose_service._decompose_prompt([{"class": "Net", "parent": "Module"}])
    assert '"entry_class": "Net"' in prompt
    assert "完整示例" in prompt
    # 白名单外 nn.* 类的 code_hint 兜底 + 多输出汇合，都必须写进提示词（scGPT 类模型的真实拦路石）
    assert "code_hint" in prompt and "nn.TransformerEncoder" in prompt
    assert "只有一个汇点" in prompt


# ------------------------------------------------------------ ② 重试复用会话

def test_attempt_prompt_uses_resume_when_available():
    base = "BASE-PROMPT"
    ask, resume = decompose_service._attempt_prompt(base, "r", None, 1)
    assert (ask, resume) == (base, None)

    # 第 2 次且有会话 → 只发修正指令，续接复用已读上下文
    ask, resume = decompose_service._attempt_prompt(base, "缺必填参数 in_features", "sess-1", 2)
    assert resume == "sess-1"
    assert "修正" in ask and "缺必填参数 in_features" in ask
    assert base not in ask

    # 无会话可续接 → 退回完整 prompt + 修正附注
    ask, resume = decompose_service._attempt_prompt(base, "r", None, 2)
    assert resume is None and ask.startswith(base) and "修正" in ask


def test_attempt_stage_mentions_attempt_and_reason():
    assert "第 1/3" in decompose_service._attempt_stage(1, 3, "")
    s2 = decompose_service._attempt_stage(2, 3, "IR 结构校验失败: xyz")
    assert "第 2/3" in s2 and "xyz" in s2


def test_is_stale_session_error():
    assert agent_service.is_stale_session_error("No conversation found with session ID: abc")
    assert agent_service.is_stale_session_error("session not found")
    assert not agent_service.is_stale_session_error("boom")


# ------------------------------------------------------------ ① 进度打点

class ToolUseBlock:
    def __init__(self, name: str) -> None:
        self.name = name


class AssistantMessage:
    def __init__(self, content) -> None:
        self.content = content


def test_emit_tool_uses_reports_tool_names():
    got: list = []
    agent_service._emit_tool_uses(
        AssistantMessage([ToolUseBlock("Read"), ToolUseBlock("Grep")]),
        lambda k, d: got.append((k, d["name"])))
    assert got == [("tool", "Read"), ("tool", "Grep")]
    # 非 AssistantMessage 静默忽略；回调抛错也不影响会话
    agent_service._emit_tool_uses(object(), lambda k, d: got.append(("x", "")))
    assert len(got) == 2


def test_progress_reporter_keeps_stage_while_emitting_activity(app_client):
    """update_progress 是整体覆盖：滚动 activity 时不能把 stage 冲掉。"""
    tid = task_manager.create_task("decompose", params={"project_id": "progress-test"})
    set_stage, on_event = decompose_service._progress_reporter(tid)

    set_stage("拆解中（第 1/3 次尝试）：读取源码并生成 IR…")
    on_event("tool", {"name": "Read"})
    prog = json.loads(task_manager.get_task(tid)["progress"])
    assert "第 1/3" in prog["stage"] and prog["activity"] == "调用工具 Read"

    on_event("delta", {"text": "忽略我"})       # 非 tool 事件忽略
    on_event("tool", {"name": ""})              # 空名忽略
    prog = json.loads(task_manager.get_task(tid)["progress"])
    assert prog["activity"] == "调用工具 Read" and "第 1/3" in prog["stage"]


# ------------------------------------------------------------ 端到端：重试 + 续接 + 进度

@pytest.fixture()
def dec_env(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    pid = project_manager.create_project("original", source="local-test")
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    (ws / "reports").mkdir(parents=True, exist_ok=True)
    (ws / "reports" / "structure_report.json").write_text(
        json.dumps({"module_hierarchy": [{"class": "Net", "parent": "Module"}]}), encoding="utf-8")
    (ws / "source").mkdir(parents=True, exist_ok=True)
    return pid


def test_run_decompose_retries_with_resume_and_reports_progress(dec_env, monkeypatch):
    pid = dec_env
    calls: list = []

    async def fake_run_sync(prompt, **kw):
        calls.append({"prompt": prompt, "resume": kw.get("resume")})
        if len(calls) == 1:
            return {"structured_output": {"nodes": [], "edges": []}, "session_id": "sess-1"}
        return {"structured_output": dict(decompose_service._IR_EXAMPLE), "session_id": "sess-1"}

    monkeypatch.setattr(agent_service, "run_sync", fake_run_sync)
    tid = task_manager.create_task("decompose", params={"project_id": pid})

    asyncio.run(decompose_service._run_decompose({"project_id": pid}, tid))

    assert len(calls) == 2
    assert calls[0]["resume"] is None                 # 首次不带续接
    assert calls[1]["resume"] == "sess-1"             # 重试续接上一次会话
    assert "修正" in calls[1]["prompt"] and calls[0]["prompt"] not in calls[1]["prompt"]

    ws = Path(project_manager.get_project(pid)["workspace_path"])
    ir = json.loads((ws / "reports" / "ir.json").read_text(encoding="utf-8"))
    assert ir["entry_class"] == "Net" and ir["schema_version"]
    prog = json.loads(task_manager.get_task(tid)["progress"])
    assert "拆解完成" in prog["stage"]


def test_run_decompose_reruns_full_prompt_when_resume_yields_nothing(dec_env, monkeypatch):
    """续接那轮空手而归（无结果文件）→ 立刻用完整 prompt 重跑，不静默失败、不白吃一次尝试。"""
    pid = dec_env
    calls: list = []

    async def fake_run_sync(prompt, **kw):
        calls.append({"prompt": prompt, "resume": kw.get("resume")})
        if len(calls) == 1:
            return {"structured_output": {"nodes": [], "edges": []}, "session_id": "sess-1"}
        if len(calls) == 2:
            return {"structured_output": None, "session_id": "sess-1"}   # 续接轮什么都没写
        return {"structured_output": dict(decompose_service._IR_EXAMPLE), "session_id": "sess-1"}

    monkeypatch.setattr(agent_service, "run_sync", fake_run_sync)
    tid = task_manager.create_task("decompose", params={"project_id": pid})

    asyncio.run(decompose_service._run_decompose({"project_id": pid}, tid))

    assert len(calls) == 3
    assert calls[1]["resume"] == "sess-1"        # 第 2 次续接
    assert calls[2]["resume"] is None            # 空手而归 → 立刻退回完整 prompt（不带续接）
    assert "任务：阅读深度学习项目代码" in calls[2]["prompt"]
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    assert (ws / "reports" / "ir.json").exists()


def test_run_decompose_falls_back_when_resume_stale(dec_env, monkeypatch):
    """续接目标失效：退回完整 prompt 重跑一次，不把错误甩给用户。"""
    pid = dec_env
    calls: list = []

    async def fake_run_sync(prompt, **kw):
        calls.append({"prompt": prompt, "resume": kw.get("resume")})
        if len(calls) == 1:
            return {"structured_output": {"nodes": [], "edges": []}, "session_id": "sess-1"}
        if len(calls) == 2:
            raise RuntimeError("No conversation found with session ID: sess-1 (exit code: 1)")
        return {"structured_output": dict(decompose_service._IR_EXAMPLE), "session_id": "sess-2"}

    monkeypatch.setattr(agent_service, "run_sync", fake_run_sync)
    tid = task_manager.create_task("decompose", params={"project_id": pid})

    asyncio.run(decompose_service._run_decompose({"project_id": pid}, tid))

    assert len(calls) == 3
    assert calls[1]["resume"] == "sess-1"             # 先尝试续接
    assert calls[2]["resume"] is None                # 续接失效 → 退回完整 prompt（不带续接）
    assert calls[2]["prompt"].startswith(decompose_service._decompose_prompt(
        [{"class": "Net", "parent": "Module"}])) or "任务：阅读深度学习项目代码" in calls[2]["prompt"]
