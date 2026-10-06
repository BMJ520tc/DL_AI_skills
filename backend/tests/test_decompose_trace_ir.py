"""**追踪优先**（`DECOMPOSE_TRACE_IR`）接进拆解的回归用例。

背景：agent 猜不出忠实 IR（漏模块、把子模块晾成孤立死块）；`scripts/trace_ir.py` 用
`torch.export` 机械生成，实测在真实 scGPT 上通过 ⑤ 两步验证。本组用例锁住「接进去」这件事的判据：

- 开关**关**时完全惰性（不碰项目环境、不起子进程）；
- 缺前置（真实 `input_spec.shape`）时**跳过追踪**并给可操作原因（默认形状会把批量维常量折叠进 IR）；
- 开着且追踪成功时，**第 1 轮直接用追踪产物**、agent 一次都不调用，且照样过统一校验链；
- 追踪任何一步失败都**回退 agent**（行为与不开开关一致），并把结论留痕。

注：追踪脚本与保真度自检脚本**共用** `proc_util.run_command`，所以桩要**按脚本名**分派。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.services import (agent_service, analysis_service, decompose_service,
                          knowledge_service, project_manager, task_manager)


# ------------------------------------------------------------ _trace_ref_ir（纯函数）

def test_trace_ref_prefers_prev_ir():
    """有上一版 IR 就用它的 entry_class/source_file/input_spec/entry_args（重拆解即「补参后重来」）。"""
    prev = {"entry_class": "Net", "source_file": "pkg/model.py", "task_type": "classification",
            "input_spec": {"shape": [1, 8], "dtype": "float32"}, "entry_args": {"n": 3}}
    ref, why = decompose_service._trace_ref_ir(None, prev, [])
    assert why == ""
    assert ref == {"source_file": "pkg/model.py", "task_type": "classification",
                   "input_spec": {"shape": [1, 8], "dtype": "float32"},
                   "entry_args": {"n": 3}, "entry_class": "Net"}


def test_trace_ref_finds_source_file_from_structure_report():
    """没有上一版 IR 时：entry_class 由请求给出，source_file 从结构报告按类名找回。"""
    hierarchy = [{"class": "Other", "file": "a.py", "parent": "Module"},
                 {"class": "Net", "file": "b/model.py", "parent": "Module"}]
    ref, why = decompose_service._trace_ref_ir(
        "Net", {"input_spec": {"shape": [1, 4]}}, hierarchy)
    assert why == "" and ref["source_file"] == "b/model.py"
    assert ref["task_type"] == "other"          # 缺省


def test_trace_ref_requires_entry_shape_and_file():
    """三个前置各自缺失 → 各给一条**说人话**的原因（不抛异常）。"""
    ref, why = decompose_service._trace_ref_ir(None, None, [])
    assert ref is None and "entry_class" in why

    ref, why = decompose_service._trace_ref_ir("Net", None, [{"class": "Net", "file": "m.py"}])
    assert ref is None and "shape" in why and "常量折叠" in why

    ref, why = decompose_service._trace_ref_ir("Net", {"entry_class": "Net"}, [])
    assert ref is None and "source_file" in why

    ref, why = decompose_service._trace_ref_ir(
        "Net", {"input_spec": {"shape": [1, None], "dtype": "float32"}},
        [{"class": "Net", "file": "m.py"}])
    assert ref is None and "shape" in why       # [1, None] 不是「正整数具体形状」


# ------------------------------------------------------------ 桩

@pytest.fixture()
def mk_env(isolated_db, tmp_path, monkeypatch):
    """带结构报告/源码目录的临时原始项目（与 `test_decompose_retry.py::dec_env` 同构）。"""
    def _make():
        monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
        pid = project_manager.create_project("original", source="local-test")
        ws = Path(project_manager.get_project(pid)["workspace_path"])
        (ws / "reports").mkdir(parents=True, exist_ok=True)
        (ws / "reports" / "structure_report.json").write_text(
            json.dumps({"module_hierarchy": [{"class": "Net", "parent": "Module"}]}),
            encoding="utf-8")
        (ws / "source").mkdir(parents=True, exist_ok=True)
        return pid, ws
    return _make


def _preset_ir(ws: Path, spec: dict | None = None) -> None:
    """放一份既有 IR——给追踪路径提供 entry_class/source_file/input_spec.shape 这些前置。"""
    (ws / "reports").mkdir(parents=True, exist_ok=True)
    (ws / "reports" / "ir.json").write_text(json.dumps(
        {"entry_class": "Net", "source_file": "model.py", "task_type": "classification",
         "input_spec": spec or {"shape": [1, 4], "dtype": "float32"}}), encoding="utf-8")


def _stub_proc(monkeypatch, *, trace=(0, ""), trace_out=None, fidelity_ok=True):
    """按**脚本名**分派子进程桩；返回「被调用的脚本名」列表（用于断言惰性 / 是否起过追踪）。"""
    seen: list[str] = []

    async def fake_run(cmd, **kw):
        name = Path(cmd[1]).name
        seen.append(name)
        if name == decompose_service.TRACE_IR_SCRIPT.name:
            rc, log = trace
            if rc == 0 and trace_out is not None:
                Path(cmd[-1]).write_text(json.dumps(trace_out), encoding="utf-8")
            return rc, log
        # 保真度自检：只回 `{ok:true}` 让它干脆通过（否则会按「不可解析/跳过」走别的分支）
        if fidelity_ok and Path(cmd[-1]).name == "fidelity.json":
            Path(cmd[-1]).write_text(json.dumps({"ok": True}), encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(decompose_service.proc_util, "run_command", fake_run)
    return seen


def _decompose_run_metrics(pid: str) -> dict:
    runs = knowledge_service.list_runs(project_id=pid, run_type="decompose")
    assert runs, "没有 decompose 的 run_record"
    m = runs[0].get("metrics")
    return json.loads(m) if isinstance(m, str) else dict(m or {})


# ------------------------------------------------------------ _trace_ir_candidate

def test_trace_candidate_is_inert_when_flag_off(mk_env, monkeypatch):
    """开关关闭 → **在任何 IO 之前**返回：不查项目环境、不起子进程。"""
    monkeypatch.setattr(decompose_service, "DECOMPOSE_TRACE_IR", False)
    _pid, ws = mk_env()

    def _boom(*a, **k):
        raise AssertionError("开关关闭时不该碰项目环境/子进程")

    monkeypatch.setattr(analysis_service, "_project_python", _boom)
    monkeypatch.setattr(decompose_service.proc_util, "run_command", _boom)

    ir, note = asyncio.run(decompose_service._trace_ir_candidate(
        ws / "source", ws, "t1", "Net", {"source_file": "model.py",
                                         "input_spec": {"shape": [1, 4]}}, []))
    assert ir is None and "DECOMPOSE_TRACE_IR" in note


def test_trace_candidate_skips_without_project_env(mk_env, monkeypatch):
    monkeypatch.setattr(decompose_service, "DECOMPOSE_TRACE_IR", True)
    monkeypatch.setattr(analysis_service, "_project_python", lambda ws: None)
    _pid, ws = mk_env()
    ir, note = asyncio.run(decompose_service._trace_ir_candidate(
        ws / "source", ws, "t1", "Net", {"source_file": "model.py",
                                         "input_spec": {"shape": [1, 4]}}, []))
    assert ir is None and "项目环境未就绪" in note


def test_trace_candidate_returns_none_on_script_failure(mk_env, monkeypatch):
    """脚本非 0 退出 → `(None, 原因)`（绝不抛），调用方据此回退 agent。"""
    monkeypatch.setattr(decompose_service, "DECOMPOSE_TRACE_IR", True)
    monkeypatch.setattr(analysis_service, "_project_python", lambda ws: "py")
    _pid, ws = mk_env()
    _stub_proc(monkeypatch, trace=(1, "boom: torch import failed"))
    ir, note = asyncio.run(decompose_service._trace_ir_candidate(
        ws / "source", ws, "t1", "Net", {"source_file": "model.py",
                                         "input_spec": {"shape": [1, 4]}}, []))
    assert ir is None and "rc=1" in note and "boom" in note


def test_trace_candidate_rejects_structurally_invalid_output(mk_env, monkeypatch):
    monkeypatch.setattr(decompose_service, "DECOMPOSE_TRACE_IR", True)
    monkeypatch.setattr(analysis_service, "_project_python", lambda ws: "py")
    _pid, ws = mk_env()
    _stub_proc(monkeypatch, trace_out={"nodes": [], "edges": []})
    ir, note = asyncio.run(decompose_service._trace_ir_candidate(
        ws / "source", ws, "t1", "Net", {"source_file": "model.py",
                                         "input_spec": {"shape": [1, 4]}}, []))
    assert ir is None and "结构校验未通过" in note


def test_trace_candidate_ok_returns_ir_and_note(mk_env, monkeypatch):
    monkeypatch.setattr(decompose_service, "DECOMPOSE_TRACE_IR", True)
    monkeypatch.setattr(analysis_service, "_project_python", lambda ws: "py")
    _pid, ws = mk_env()
    _stub_proc(monkeypatch, trace_out=dict(decompose_service._IR_EXAMPLE))
    ir, note = asyncio.run(decompose_service._trace_ir_candidate(
        ws / "source", ws, "t1", "Net", {"source_file": "model.py",
                                         "input_spec": {"shape": [1, 4]}}, []))
    assert ir is not None and "真实 torch.export" in note


# ------------------------------------------------------------ _run_decompose 全路径

@pytest.fixture()
def dec_env(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(decompose_service, "DECOMPOSE_STEPWISE", False)
    monkeypatch.setattr(decompose_service, "DECOMPOSE_TRACE_IR", True)
    monkeypatch.setattr(analysis_service, "_project_python", lambda ws: "py")
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    pid = project_manager.create_project("original", source="local-test")
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    (ws / "reports").mkdir(parents=True, exist_ok=True)
    (ws / "reports" / "structure_report.json").write_text(
        json.dumps({"module_hierarchy": [{"class": "Net", "file": "model.py", "parent": "Module"}]}),
        encoding="utf-8")
    (ws / "source").mkdir(parents=True, exist_ok=True)
    return pid


def _agent_stub(monkeypatch, calls: list):
    async def fake_run_sync(prompt, **kw):
        calls.append(prompt)
        return {"structured_output": dict(decompose_service._IR_EXAMPLE), "session_id": "s1"}
    monkeypatch.setattr(agent_service, "run_sync", fake_run_sync)


def test_run_decompose_uses_trace_candidate_without_calling_agent(dec_env, monkeypatch):
    """追踪成功 → 第 1 轮直接用追踪产物：**agent 一次都没被调用**，IR 落盘，留痕 `via=trace`。"""
    pid = dec_env
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    _preset_ir(ws)
    _stub_proc(monkeypatch, trace_out=dict(decompose_service._IR_EXAMPLE))
    calls: list = []

    async def _no_agent(prompt, **kw):
        raise AssertionError("追踪成功时不该调用 agent")

    monkeypatch.setattr(agent_service, "run_sync", _no_agent)

    tid = task_manager.create_task("decompose", params={"project_id": pid})
    asyncio.run(decompose_service._run_decompose({"project_id": pid}, tid))

    ir = json.loads((ws / "reports" / "ir.json").read_text(encoding="utf-8"))
    assert ir["entry_class"] == "Net" and ir["schema_version"]
    prog = json.loads(task_manager.get_task(tid)["progress"])
    assert "真实追踪" in prog["stage"] and prog["via"] == "trace"
    assert _decompose_run_metrics(pid)["via"] == "trace"
    assert calls == []


def test_run_decompose_falls_back_to_agent_when_trace_fails(dec_env, monkeypatch):
    """追踪脚本失败 → 回退 agent（行为与不开开关一致），结论留痕。"""
    pid = dec_env
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    _preset_ir(ws)
    _stub_proc(monkeypatch, trace=(3, "RuntimeError: 模型起不来"))
    calls: list = []
    _agent_stub(monkeypatch, calls)

    tid = task_manager.create_task("decompose", params={"project_id": pid})
    asyncio.run(decompose_service._run_decompose({"project_id": pid}, tid))

    assert len(calls) == 1                      # 回退 agent
    prog = json.loads(task_manager.get_task(tid)["progress"])
    assert prog["via"] == "agent" and "rc=3" in prog["trace_note"]


def test_run_decompose_skips_trace_without_real_shape(dec_env, monkeypatch):
    """缺真实 `input_spec.shape` → **跳过追踪**（默认形状会把批量维常量折叠进 IR）、回退 agent。"""
    pid = dec_env
    seen = _stub_proc(monkeypatch)              # 追踪若被调用会记录脚本名
    calls: list = []
    _agent_stub(monkeypatch, calls)

    # 给足其余前置（entry_class + 结构报告里有定义文件），只缺形状 → 该由「形状」这条原因拦下
    tid = task_manager.create_task("decompose", params={"project_id": pid, "entry_class": "Net"})
    asyncio.run(decompose_service._run_decompose({"project_id": pid, "entry_class": "Net"}, tid))

    assert decompose_service.TRACE_IR_SCRIPT.name not in seen    # 没起过追踪子进程
    assert len(calls) == 1
    prog = json.loads(task_manager.get_task(tid)["progress"])
    assert prog["via"] == "agent" and "shape" in prog["trace_note"]


def test_trace_candidate_exempt_from_node_cap(dec_env, monkeypatch):
    """追踪产物**豁免节点数上限**（上限的立论是「LLM 输出越长越不稳」，对机械追踪不适用）。"""
    pid = dec_env
    monkeypatch.setattr(decompose_service, "DECOMPOSE_MAX_NODES", 5)
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    _preset_ir(ws)
    big = dict(decompose_service._IR_EXAMPLE)
    big["nodes"] = list(big["nodes"]) + [
        {"id": f"extra{i}", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net"}
        for i in range(6)]
    _stub_proc(monkeypatch, trace_out=big)

    async def _no_agent(prompt, **kw):
        raise AssertionError("追踪产物被接受时不该调用 agent")

    monkeypatch.setattr(agent_service, "run_sync", _no_agent)

    tid = task_manager.create_task("decompose", params={"project_id": pid})
    asyncio.run(decompose_service._run_decompose({"project_id": pid}, tid))

    ir = json.loads((ws / "reports" / "ir.json").read_text(encoding="utf-8"))
    assert len(ir["nodes"]) > 5                  # 超限仍被接受


def test_agent_candidate_still_subject_to_node_cap(dec_env, monkeypatch):
    """豁免只给追踪产物：回退后的 **agent 候选仍受上限约束**（不变量别被放宽）。"""
    pid = dec_env
    monkeypatch.setattr(decompose_service, "DECOMPOSE_MAX_NODES", 5)
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    _preset_ir(ws)
    _stub_proc(monkeypatch, trace=(1, "boom"))   # 追踪失败 → 回退 agent
    calls: list = []

    async def fake_run_sync(prompt, **kw):
        calls.append(prompt)
        big = dict(decompose_service._IR_EXAMPLE)
        big["nodes"] = list(big["nodes"]) + [
            {"id": f"extra{i}", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net"}
            for i in range(6)]
        return {"structured_output": big, "session_id": "s1"}

    monkeypatch.setattr(agent_service, "run_sync", fake_run_sync)

    tid = task_manager.create_task("decompose", params={"project_id": pid})
    with pytest.raises(RuntimeError, match="超过上限"):     # agent 候选被上限拒 → 拆解失败
        asyncio.run(decompose_service._run_decompose({"project_id": pid}, tid))
    assert len(calls) >= 1


def test_trace_candidate_invalid_falls_back_to_agent(dec_env, monkeypatch):
    """追踪产物过不了统一校验链（这里：inputs 不是节点 id）→ 回退 agent，不留坏 IR。"""
    pid = dec_env
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    _preset_ir(ws)
    bad = dict(decompose_service._IR_EXAMPLE)
    bad["input_spec"] = {"shape": [1, 4], "dtype": "float32", "inputs": ["not_a_node_id"]}
    _stub_proc(monkeypatch, trace_out=bad)
    calls: list = []
    _agent_stub(monkeypatch, calls)

    tid = task_manager.create_task("decompose", params={"project_id": pid})
    asyncio.run(decompose_service._run_decompose({"project_id": pid}, tid))

    assert len(calls) == 1
    prog = json.loads(task_manager.get_task(tid)["progress"])
    assert prog["via"] == "agent"
    ir = json.loads((ws / "reports" / "ir.json").read_text(encoding="utf-8"))
    # 落盘的是 agent 的（合法）产物，不是那份坏追踪产物
    assert ir["input_spec"].get("inputs") != ["not_a_node_id"]
