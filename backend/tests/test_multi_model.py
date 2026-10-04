"""多模型综合分析（模块六 8.4，需求六.2 扩展项）。

用合成的逐样本 predictions 产物跑核心分析：同口径样本取交集、一致/分歧识别、
融合与融合前后指标、分歧归因（agent 由假实现替代）、fusion_insight 入库。
报告目录被重定向到 tmp，不碰真实 data/。
"""
from __future__ import annotations

import asyncio
import json

from app.services import knowledge_service, multi_model_service


def _write_run(tmp_path, run_id: str, model: str, preds: list[dict], dataset_id: str | None = None) -> str:
    art = tmp_path / f"{run_id}.json"
    art.write_text(json.dumps({"metrics": {"accuracy": 0.0}, "predictions": preds}, ensure_ascii=False),
                   encoding="utf-8")
    params = {"model": model}
    if dataset_id:
        params["dataset_id"] = dataset_id
    knowledge_service.record_run({
        "run_id": run_id, "project_id": "p-mm", "run_type": "eval", "status": "success",
        "params": params, "artifact_path": str(art),
    })
    return run_id


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_model_service, "MULTI_MODEL_DIR", tmp_path / "mm")
    run1 = _write_run(tmp_path, "r1", "m1", [
        {"id": "i1", "y_true": "A", "y_pred": "A"},
        {"id": "i2", "y_true": "B", "y_pred": "B"},
        {"id": "i3", "y_true": "A", "y_pred": "B"},   # 与 m2 分歧
        {"id": "i4", "y_true": "B", "y_pred": "A"},   # 与 m2 分歧
    ])
    run2 = _write_run(tmp_path, "r2", "m2", [
        {"id": "i1", "y_true": "A", "y_pred": "A"},
        {"id": "i2", "y_true": "B", "y_pred": "B"},
        {"id": "i3", "y_true": "A", "y_pred": "A"},
        {"id": "i4", "y_true": "B", "y_pred": "B"},
    ])
    return run1, run2


def test_multi_model_analysis_and_fusion_insight(isolated_db, tmp_path, monkeypatch):
    run1, run2 = _fixture(tmp_path, monkeypatch)

    async def fake_run_sync(prompt, *, output_schema=None, timeout_s=180, **kw):
        return {"structured_output": {"summary": "分歧疑来自训练数据差异",
                                      "reasons": [{"cause": "training_data", "explanation": "i3/i4"}]}}

    monkeypatch.setattr(multi_model_service.agent_service, "run_sync", fake_run_sync)

    report = asyncio.run(multi_model_service._analyze(
        {"run_ids": [run1, run2], "task_type": "classification"}, "task-mm"))

    assert report["n_common_samples"] == 4
    assert len(report["consistent"]) == 2                 # i1, i2
    assert {d["id"] for d in report["disagreements"]} == {"i3", "i4"}
    assert report["fusion"] == "vote"                     # classification 默认投票
    assert report["metrics_before"]["m1"]["accuracy"] == 0.5
    assert report["metrics_before"]["m2"]["accuracy"] == 1.0
    assert report["attribution"]["summary"].startswith("分歧疑来自")

    # ⑥ 结论入库：fusion_insight（draft），可检索
    kid = report["knowledge_id"]
    item = knowledge_service.get_item("knowledge", kid)
    assert item["type"] == "fusion_insight" and item["status"] == "draft"
    assert kid in [k["knowledge_id"] for k in knowledge_service.list_knowledge(type_="fusion_insight")]

    # 报告落盘可经 get_report 取回
    assert multi_model_service.get_report("task-mm")["analysis_id"] == "task-mm"


def test_multi_model_requires_two_runs(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(multi_model_service, "MULTI_MODEL_DIR", tmp_path / "mm")
    run1 = _write_run(tmp_path, "solo", "m1", [{"id": "i1", "y_true": "A", "y_pred": "A"}])
    try:
        asyncio.run(multi_model_service._analyze({"run_ids": [run1]}, "t"))
    except ValueError as e:
        assert "至少需要两个" in str(e)
    else:  # pragma: no cover
        raise AssertionError("单模型应报错")


def test_multi_model_incomparable_when_ytrue_mismatch(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(multi_model_service, "MULTI_MODEL_DIR", tmp_path / "mm")
    r1 = _write_run(tmp_path, "a", "m1", [{"id": "i1", "y_true": "A", "y_pred": "A"}])
    r2 = _write_run(tmp_path, "b", "m2", [{"id": "i1", "y_true": "B", "y_pred": "A"}])

    async def fake_run_sync(*a, **k):
        return {"structured_output": {"summary": "x"}}

    monkeypatch.setattr(multi_model_service.agent_service, "run_sync", fake_run_sync)
    report = asyncio.run(multi_model_service._analyze({"run_ids": [r1, r2], "task_type": "classification"}, "t2"))

    assert report["consistent"] == [] and report["disagreements"] == []
    assert len(report["incomparable"]) == 1


def test_attribution_failure_does_not_block_report(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(multi_model_service, "MULTI_MODEL_DIR", tmp_path / "mm")
    r1 = _write_run(tmp_path, "c1", "m1", [{"id": "i1", "y_true": "A", "y_pred": "A"},
                                           {"id": "i2", "y_true": "B", "y_pred": "A"}])
    r2 = _write_run(tmp_path, "c2", "m2", [{"id": "i1", "y_true": "A", "y_pred": "A"},
                                           {"id": "i2", "y_true": "B", "y_pred": "B"}])

    async def boom(*a, **k):
        raise RuntimeError("agent 不可用")

    monkeypatch.setattr(multi_model_service.agent_service, "run_sync", boom)
    report = asyncio.run(multi_model_service._analyze({"run_ids": [r1, r2], "task_type": "classification"}, "t3"))

    assert report["attribution_error"] and "agent 不可用" in report["attribution_error"]
    assert report["knowledge_id"]                       # 报告与结论仍产出


def test_default_fusion_by_task_type():
    assert multi_model_service._default_fusion("classification") == "vote"
    assert multi_model_service._default_fusion("regression") == "weighted"
    assert multi_model_service._default_fusion(None) == "vote"


def test_alignment_label_merge_unifies_before_comparison(isolated_db, tmp_path, monkeypatch):
    """① 同口径：标签体系按数据集已确认 alignment.label_merge 归并后再比较（8.4 第 1 步）。"""
    monkeypatch.setattr(multi_model_service, "MULTI_MODEL_DIR", tmp_path / "mm")
    ks = knowledge_service
    ks.register_dataset({"dataset_id": "ds-mm", "name": "MM 数据", "task_type": "classification",
                         "alignment": {"status": "confirmed", "label_merge": {"cat": "animal", "dog": "animal"}}})
    # 两模型标签写法不同：m1 用 cat/dog，m2 用 animal——归并后应一致
    a = _write_run(tmp_path, "am", "m1", [{"id": "i1", "y_true": "animal", "y_pred": "cat"}],
                   dataset_id="ds-mm")
    b = _write_run(tmp_path, "bm", "m2", [{"id": "i1", "y_true": "animal", "y_pred": "animal"}],
                   dataset_id="ds-mm")

    async def fake_run_sync(*args, **kwargs):
        return {"structured_output": {"summary": "x"}}

    monkeypatch.setattr(multi_model_service.agent_service, "run_sync", fake_run_sync)
    report = asyncio.run(multi_model_service._analyze({"run_ids": [a, b], "task_type": "classification"}, "t-align"))

    assert report["alignment"]["used"] is True and report["alignment"]["entries"] == 2
    assert len(report["consistent"]) == 1 and report["disagreements"] == []   # 归并后一致


def test_duplicate_model_names_disambiguated(isolated_db, tmp_path, monkeypatch):
    """不同 run 同名（实测真实库 eval 全是 'fixture-rule'）→ 用 run_id 消歧，
    避免 metrics_before 按名字做键把多个模型塌成一个。"""
    monkeypatch.setattr(multi_model_service, "MULTI_MODEL_DIR", tmp_path / "mm")
    a = _write_run(tmp_path, "dupA123456", "same", [{"id": "i1", "y_true": "A", "y_pred": "A"}])
    b = _write_run(tmp_path, "dupB654321", "same", [{"id": "i1", "y_true": "A", "y_pred": "B"}])

    async def fake(*a, **k):
        return {"structured_output": []}

    monkeypatch.setattr(multi_model_service.agent_service, "run_sync", fake)
    report = asyncio.run(multi_model_service._analyze({"run_ids": [a, b], "task_type": "classification"}, "t"))

    names = [m["name"] for m in report["models"]]
    assert len(set(names)) == 2 and all("#" in n for n in names)
    assert len(report["metrics_before"]) == 2          # 两个模型各自一行，不塌成一


def test_multi_model_endpoints_validation(app_client):
    assert app_client.post("/api/multi-model", json={"run_ids": ["only-one"]}).status_code == 400
    assert app_client.post("/api/multi-model", json={"run_ids": ["a", "b"], "fusion": "bogus"}).status_code == 400
    assert app_client.get("/api/multi-model/nope").status_code == 404
