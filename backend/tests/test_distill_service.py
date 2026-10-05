"""任务后蒸馏与论文蒸馏（模块六 8.3、需求六.1、数据设计六.3）。

覆盖：形状容错解析、类型归一、终态钩子**只入队**（白名单/论文结论）、蒸馏任务起草与幂等、
论文蒸馏、以及走真实 task_manager worker 的钩子接线集成。
agent 调用被替换为假实现（不打真实模型端点）。
"""
from __future__ import annotations

import asyncio
import json

from app.services import distill_service, knowledge_service, task_manager


def _mk_task(task_type: str, params: dict | None = None) -> str:
    return task_manager.create_task(task_type, project_id="p-distill", params=params or {})


def _distill_tasks() -> list[dict]:
    return [t for t in task_manager.list_tasks() if t["task_type"] == "knowledge_distill"]


# ---------------------------------------------------------------- 形状容错 / 类型归一

def test_knowledge_items_accepts_all_shapes():
    """形状容错（真缺陷回归）：DeepSeek 下模型常把条目**裸数组**写进 result.json，
    还可能包在 `{"knowledge":[...]}` 或**类型名做键**的对象里。"""
    item = {"type": "param_advice", "title": "t", "content": "c"}
    assert distill_service._knowledge_items([item]) == [item]                    # 裸数组（实测形状）
    assert distill_service._knowledge_items({"knowledge": [item]}) == [item]     # 包装对象
    assert distill_service._knowledge_items({"usage_guidance": [item]}) == [item]  # 类型名做键（实测）
    assert distill_service._knowledge_items({"paper_id": "gears", "usage_guidance": [item]}) == [item]
    assert distill_service._knowledge_items(item) == [item]                      # 单条对象
    assert distill_service._knowledge_items([]) == []
    assert distill_service._knowledge_items(None) == []
    assert distill_service._knowledge_items([1, "x", item]) == [item]           # 非 dict 项剔除


def test_text_fields_tolerate_non_string_values(isolated_db, monkeypatch):
    """字段值漂移回归：模型把 content/confidence 给成**数字**时不得崩。

    实测 `(item.get("content") or "").strip()` 在 content 是 `0.9` 时抛
    `'float' object has no attribute 'strip'`，把整条蒸馏任务打挂（2026-10-05）。
    """
    run = {"run_id": "r-num", "run_type": "train", "status": "failed", "params": {}}

    async def fake(*a, **k):
        return {"structured_output": [
            {"type": "param_advice", "title": 0.9, "content": 0.9, "confidence": 0.9},  # 数字 content → 视为空、跳过
            {"type": "usage_guidance", "title": "T", "content": "有效结论", "confidence": 0.8},
        ]}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake)
    ids = asyncio.run(distill_service._draft_from_run(run, "task-num"))    # 不得抛异常

    assert [d["content"] for d in knowledge_service.list_knowledge(status="draft")] == ["有效结论"]
    assert len(ids) == 1


def test_normalize_text_helpers_with_non_string():
    assert distill_service._as_text(0.9) == "" and distill_service._as_text("x") == "x"
    assert distill_service._normalize_confidence(0.9) == "low"      # 非字符串 → 保守归 low
    assert distill_service._normalize_confidence(None) == "low"
    assert distill_service._normalize_type(0.9) == "usage_guidance"


def test_normalize_confidence_enum():
    """confidence 归一到 {high, medium, low}（实测模型写过超枚举的 medium-high）。"""
    assert distill_service._normalize_confidence("high") == "high"
    assert distill_service._normalize_confidence("medium-high") == "medium"
    assert distill_service._normalize_confidence("Low") == "low"
    assert distill_service._normalize_confidence("very high") == "high"
    assert distill_service._normalize_confidence("") == "low"
    assert distill_service._normalize_confidence(None) == "low"


def test_normalize_type_coerces_drifted_names():
    """类型名漂移归一（实测模型写过 applicability/method）→ 落到五类枚举内。"""
    for t in ("param_advice", "dependency_conflict", "reproduction_discrepancy",
              "usage_guidance", "fusion_insight"):
        assert distill_service._normalize_type(t) == t
    assert distill_service._normalize_type("applicability") == "usage_guidance"
    assert distill_service._normalize_type("method") == "usage_guidance"
    assert distill_service._normalize_type("conflict") == "dependency_conflict"
    assert distill_service._normalize_type("") == "usage_guidance"
    assert distill_service._normalize_type(None) == "usage_guidance"
    assert distill_service._normalize_type("fusion_insight",
                                           allowed=("usage_guidance", "param_advice")) == "usage_guidance"


# ---------------------------------------------------------------- 起草（_draft_from_run）

def test_draft_from_run_shapes_and_empty(isolated_db, monkeypatch):
    """从运行记录起草：形状容错 + 空结论不落 + source_task_id 溯源。"""
    run = {"run_id": "r1", "run_type": "train", "status": "failed", "error": "boom",
           "params": {"task_type": "classification"}}

    async def fake(*a, **k):
        return {"structured_output": [{"type": "param_advice", "title": "T", "content": "lr=1e-3"},
                                      {"type": "param_advice", "title": "空", "content": "   "}]}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake)
    ids = asyncio.run(distill_service._draft_from_run(run, "task-x"))

    drafts = knowledge_service.list_knowledge(status="draft")
    assert [d["content"] for d in drafts] == ["lr=1e-3"]     # 空结论被跳过
    assert len(ids) == 1
    assert '"source_task_id": "task-x"' in drafts[0]["structured"]

    async def fake_bare(*a, **k):
        return {"structured_output": [{"type": "dependency_conflict", "title": "冲突", "content": "torch 冲突"}]}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake_bare)
    asyncio.run(distill_service._draft_from_run({"run_id": "r2", "run_type": "train", "status": "failed", "params": {}}, "task-y"))
    assert sorted(d["content"] for d in knowledge_service.list_knowledge(status="draft")) == \
        sorted(["lr=1e-3", "torch 冲突"])   # 同秒 created_at 顺序不定，按集合比


def test_run_knowledge_distill_dedups_and_no_run(isolated_db, monkeypatch):
    """蒸馏任务：无运行记录→无素材（不落）；已起草→幂等跳过。"""
    called = {"n": 0}

    async def fake(*a, **k):
        called["n"] += 1
        return {"structured_output": [{"type": "usage_guidance", "title": "x", "content": "y"}]}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake)

    # 无运行记录
    asyncio.run(distill_service._run_knowledge_distill({"source_task_id": "no-run"}, "dt1"))
    assert called["n"] == 0 and knowledge_service.list_knowledge() == []

    # 有运行记录 → 起草；再次调用幂等跳过
    tid = _mk_task("network_train")
    knowledge_service.record_run({"project_id": "p-distill", "task_id": tid,
                                  "run_type": "train", "status": "failed", "error": "e"})
    asyncio.run(distill_service._run_knowledge_distill({"source_task_id": tid}, "dt2"))
    assert called["n"] == 1
    asyncio.run(distill_service._run_knowledge_distill({"source_task_id": tid}, "dt3"))
    assert called["n"] == 1


# ---------------------------------------------------------------- 终态钩子：只入队

def test_on_task_finished_enqueues_knowledge_and_paper_distill(isolated_db, monkeypatch):
    """钩子只入队：白名单任务 → knowledge_distill；论文结论任务 → paper_distill。"""
    monkeypatch.setattr(distill_service.agent_service, "run_sync",
                        lambda *a, **k: {"structured_output": []})

    tid = _mk_task("network_train")
    knowledge_service.record_run({"project_id": "p-distill", "task_id": tid,
                                  "run_type": "train", "status": "failed"})
    asyncio.run(distill_service.on_task_finished(tid, "failed"))
    kt = _distill_tasks()
    assert len(kt) == 1 and json.loads(kt[0]["params"])["source_task_id"] == tid

    # 论文结论任务 → paper_distill（带 paper_id）
    cid = task_manager.create_task("conclusion", params={"paper_id": "p-paper"})
    asyncio.run(distill_service.on_task_finished(cid, "success"))
    pd = [t for t in task_manager.list_tasks() if t["task_type"] == "paper_distill"]
    assert len(pd) == 1 and json.loads(pd[0]["params"])["paper_ids"] == ["p-paper"]


def test_on_task_finished_skips_non_whitelist_and_drafted(isolated_db, monkeypatch):
    """非白名单任务不入队；已有蒸馏知识（专用路径已起草）的不重复入队。"""
    monkeypatch.setattr(distill_service.agent_service, "run_sync",
                        lambda *a, **k: {"structured_output": []})

    # 非白名单
    t = _mk_task("pdf_parse")
    asyncio.run(distill_service.on_task_finished(t, "success"))
    assert _distill_tasks() == []

    # 白名单但专用路径已起草（structured.source_task_id 命中）→ 跳过
    t2 = _mk_task("env_create")
    knowledge_service.record_run({"project_id": "p-distill", "task_id": t2,
                                  "run_type": "env_install", "status": "failed", "error": "boom"})
    knowledge_service.record_knowledge({"type": "dependency_conflict", "title": "专用", "content": "c",
                                        "structured": {"source_task_id": t2}, "status": "draft"})
    asyncio.run(distill_service.on_task_finished(t2, "failed"))
    assert _distill_tasks() == []


# ---------------------------------------------------------------- 论文蒸馏

def test_distill_paper_from_experiment_items(isolated_db, monkeypatch):
    """论文蒸馏：从实验条目起草 draft，带 source_paper_id，幂等去重。"""
    ks = knowledge_service
    ks.record_paper({"paper_id": "p-distill", "title": "论文 X", "source": "arxiv"})
    ks.record_experiment_items("p-distill", [
        {"section_ref": "R1", "dataset_name": "D", "metric_name": "accuracy",
         "metric_value_reported": "0.9", "hyperparams": {"lr": "1e-3"}},
    ])

    async def fake_run_sync(*a, **k):
        return {"structured_output": [
            {"type": "usage_guidance", "title": "论文 X 使用建议", "content": "在 D 上用小学习率更稳"}]}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake_run_sync)

    ids = asyncio.run(distill_service.distill_paper("p-distill"))
    assert len(ids) == 1
    item = ks.get_item("knowledge", ids[0])
    assert item["type"] == "usage_guidance" and item["status"] == "draft"
    assert "source_paper_id" in item["structured"]
    assert asyncio.run(distill_service.distill_paper("p-distill")) == []   # 幂等


def test_distill_paper_skips_when_no_material(isolated_db, monkeypatch):
    knowledge_service.record_paper({"paper_id": "p-empty", "title": "空论文", "source": "local"})
    called = {"n": 0}

    async def fake_run_sync(*a, **k):
        called["n"] += 1
        return {"structured_output": []}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake_run_sync)
    assert asyncio.run(distill_service.distill_paper("p-empty")) == []
    assert called["n"] == 0


# ---------------------------------------------------------------- 集成：钩子接线

def test_register_wires_hook_and_handlers(isolated_db):
    """register() 把终态钩子与两个蒸馏任务 handler 都接上（产品启动即生效）。"""
    task_manager._finish_hooks.clear()
    task_manager._handlers.clear()
    distill_service.register()
    assert distill_service.on_task_finished in task_manager._finish_hooks
    assert "knowledge_distill" in task_manager._handlers
    assert "paper_distill" in task_manager._handlers


def test_finish_hooks_invoked_by_task_manager(isolated_db):
    """接线：走**真实 task_manager worker**，任务到达终态后注册的终态钩子**真的被调用**。

    此前该链路零覆盖（此前只有直接调起草函数的探针，没验过 task_manager 会不会调钩子）。
    确定性写法：用一个只记录调用的假钩子，避免「worker 跑 agent 起草」引入的竞态。
    真实蒸馏链路（钩子 → 入队 knowledge_distill → handler 起草）由上面的入队/起草单测分别覆盖。
    """
    seen: list[tuple[str, str]] = []

    async def fake_hook(task_id, status):
        seen.append((task_id, status))

    async def handler(params, task_id):
        return None

    async def scenario():
        task_manager._finish_hooks.clear()
        task_manager.register_on_finish(fake_hook)
        await task_manager.start()
        try:
            task_manager.register_handler("demo-hook", handler)
            tid = task_manager.create_task("demo-hook")
            for _ in range(500):
                if seen:
                    break
                await asyncio.sleep(0.01)
        finally:
            await task_manager.stop()

    asyncio.run(scenario())
    assert len(seen) == 1 and seen[0][1] == "success"
