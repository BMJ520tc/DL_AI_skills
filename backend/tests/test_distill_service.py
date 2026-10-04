"""任务后蒸馏入库（模块六 8.3、数据设计六.3）。

覆盖：白名单过滤、agent 起草落 draft、空结论不落条目、按 source_task_id 去重、非白名单不触发。
agent 调用被替换为假实现（不打真实模型端点）。
"""
from __future__ import annotations

import asyncio

from app.services import distill_service, knowledge_service, task_manager


def _mk_task(task_type: str) -> str:
    return task_manager.create_task(task_type, project_id="p-distill", params={})


def test_on_task_finished_drafts_and_dedups_and_skips_empty(isolated_db, monkeypatch):
    tid = _mk_task("network_train")
    knowledge_service.record_run({
        "project_id": "p-distill", "task_id": tid, "run_type": "train", "status": "success",
        "metrics": {"accuracy": 0.9}, "params": {"task_type": "classification"},
    })

    async def fake_run_sync(prompt, *, output_schema=None, timeout_s=300, **kw):
        return {"structured_output": {"knowledge": [
            {"type": "param_advice", "title": "T", "content": "lr=1e-3 效果好",
             "confidence": "medium", "scope": {"task_type": "classification"}},
            {"type": "param_advice", "title": "空", "content": "   "},   # 空结论 → 不落
        ]}}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake_run_sync)

    asyncio.run(distill_service.on_task_finished(tid, "success"))

    drafts = knowledge_service.list_knowledge(status="draft")
    assert [d["content"] for d in drafts] == ["lr=1e-3 效果好"]
    assert drafts[0]["status"] == "draft"
    # 素材溯源：structured 带 run_id 与 source_task_id
    assert '"source_task_id": ' in drafts[0]["structured"]

    # 去重：同一任务再次结束不再起草
    asyncio.run(distill_service.on_task_finished(tid, "success"))
    assert len(knowledge_service.list_knowledge(status="draft")) == 1


def test_knowledge_items_accepts_all_shapes():
    """形状容错（真缺陷回归）：DeepSeek 下模型常把条目**裸数组**写进 result.json，
    而非 output_format 的包装对象——不兼容会整条通道静默产不出东西。"""
    item = {"type": "param_advice", "title": "t", "content": "c"}
    assert distill_service._knowledge_items([item]) == [item]                    # 裸数组（实测形状）
    assert distill_service._knowledge_items({"knowledge": [item]}) == [item]     # 包装对象
    assert distill_service._knowledge_items({"items": [item]}) == [item]
    assert distill_service._knowledge_items(item) == [item]                      # 单条对象
    assert distill_service._knowledge_items([]) == []
    assert distill_service._knowledge_items(None) == []
    assert distill_service._knowledge_items([1, "x", item]) == [item]           # 非 dict 项剔除
    # 以类型名做键的包装（实测形状）
    assert distill_service._knowledge_items({"usage_guidance": [item]}) == [item]
    assert distill_service._knowledge_items({"paper_id": "gears", "usage_guidance": [item]}) == [item]


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
    # allowed 约束：论文蒸馏里模型误写 fusion_insight → 回落到 allowed 首项
    assert distill_service._normalize_type("fusion_insight",
                                           allowed=("usage_guidance", "param_advice")) == "usage_guidance"
    assert distill_service._normalize_type("param_advice",
                                           allowed=("usage_guidance", "param_advice")) == "param_advice"


def test_on_task_finished_handles_bare_array_output(isolated_db, monkeypatch):
    """通用通道在 agent 返回**裸数组**时也要落草稿（回归：此前会被 `or {}` 吃掉 → 0 条）。"""
    tid = _mk_task("network_train")
    knowledge_service.record_run({"project_id": "p-distill", "task_id": tid,
                                  "run_type": "train", "status": "failed", "error": "boom"})

    async def fake_run_sync(*a, **k):
        # 模型直接写出条目数组（不带 {"knowledge": ...} 包装）
        return {"structured_output": [{"type": "dependency_conflict", "title": "冲突",
                                       "content": "torch 与 numpy 冲突，改用 numpy<2"}]}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake_run_sync)
    asyncio.run(distill_service.on_task_finished(tid, "failed"))

    drafts = knowledge_service.list_knowledge(status="draft")
    assert [d["title"] for d in drafts] == ["冲突"]


def test_distill_paper_from_experiment_items(isolated_db, monkeypatch):
    """论文蒸馏（需求六.1「论文数据」素材）：从实验条目起草 draft，带 source_paper_id，幂等去重。"""
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
    # 幂等：已蒸馏过 → 跳过
    assert asyncio.run(distill_service.distill_paper("p-distill")) == []


def test_distill_paper_skips_when_no_material(isolated_db, monkeypatch):
    """无实验条目、无复现/结论的论文不蒸馏（不空跑 agent）。"""
    knowledge_service.record_paper({"paper_id": "p-empty", "title": "空论文", "source": "local"})
    called = {"n": 0}

    async def fake_run_sync(*a, **k):
        called["n"] += 1
        return {"structured_output": []}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake_run_sync)
    assert asyncio.run(distill_service.distill_paper("p-empty")) == []
    assert called["n"] == 0


def test_non_whitelist_task_not_distilled(isolated_db, monkeypatch):
    tid = _mk_task("pdf_parse")   # 纯解析类，不在白名单
    knowledge_service.record_run({"project_id": "p-distill", "task_id": tid,
                                  "run_type": "pdf_parse", "status": "success"})
    called = {"n": 0}

    async def fake_run_sync(*a, **k):
        called["n"] += 1
        return {"structured_output": {"knowledge": [{"type": "usage_guidance", "title": "x", "content": "y"}]}}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake_run_sync)

    asyncio.run(distill_service.on_task_finished(tid, "success"))

    assert called["n"] == 0
    assert knowledge_service.list_knowledge() == []


def test_dedicated_draft_blocks_general_channel(isolated_db, monkeypatch):
    """专用路径已起草（structured.source_task_id 命中）→ 通用通道跳过，不重复起草。"""
    tid = _mk_task("env_create")
    knowledge_service.record_run({"project_id": "p-distill", "task_id": tid,
                                  "run_type": "env_install", "status": "failed", "error": "boom"})
    knowledge_service.record_knowledge({
        "type": "dependency_conflict", "title": "专用草稿", "content": "c",
        "structured": {"source_task_id": tid}, "status": "draft",
    })
    assert knowledge_service.knowledge_for_task_exists(tid) is True

    called = {"n": 0}

    async def fake_run_sync(*a, **k):
        called["n"] += 1
        return {"structured_output": {"knowledge": []}}

    monkeypatch.setattr(distill_service.agent_service, "run_sync", fake_run_sync)

    asyncio.run(distill_service.on_task_finished(tid, "failed"))

    assert called["n"] == 0
    assert len(knowledge_service.list_knowledge()) == 1
