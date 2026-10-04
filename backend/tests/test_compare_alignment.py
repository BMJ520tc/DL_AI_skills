"""缺口二：跨数据集评估必须走「对齐副本」，alignment 真正生效（《模块详细设计》5.4）。

覆盖：
- 已确认对齐 → compare 先产出 `<目标数据集目录>/aligned/preprocessed.csv`，
  再把**对齐副本目录**作为 data_dir 交给 eval 入口（断言捕获到的路径）；
- 运行记录/对比表标明「本次使用对齐副本」与映射/归并条目数；
- 对齐未确认 → 沿用既有闸门（未确认不得评估），不产出副本、不调用评估。
"""
from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pytest

from app.services import alignment_apply, compare_service, knowledge_service as ks, task_manager

UNIFIED = ["id", "split", "label", "input"]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _setup(tmp_path: Path, *, status: str = "confirmed"):
    ws = tmp_path / "proj"
    self_csv = ws / "data" / "self" / "preprocessed.csv"
    _write_csv(self_csv, [{"id": "0", "split": "train", "label": "Cat", "input": "a"}], UNIFIED)
    self_id = ks.register_dataset({
        "name": "self", "source": "自带", "task_type": "classification",
        "fields": UNIFIED, "labels": ["Cat", "dog"], "local_path": str(self_csv),
    })

    target_dir = ws / "data" / "public-x"
    target_csv = target_dir / "preprocessed.csv"
    _write_csv(target_csv, [
        {"sample_id": "1", "split": "train", "label": "CAT", "sequence": "acgt", "meta_note": "n"},
        {"sample_id": "2", "split": "test", "label": "dog", "sequence": "tttt", "meta_note": "m"},
    ], ["sample_id", "split", "label", "sequence", "meta_note"])

    alignment = {
        "version": "1.0",
        "field_mapping": {"sample_id": "id", "sequence": "input"},
        "label_merge": {"CAT": "Cat"},
        "sequence": {"length_range": [1, 6], "case": "upper"},
        "status": status,
        "aligned_projects": ["p1"],
        "source_dataset_id": self_id,
    }
    target_id = ks.register_dataset({
        "name": "public-x", "source": "zenodo", "url": "https://zenodo.org/records/1",
        "task_type": "classification", "format": "csv",
        "fields": ["sample_id", "split", "label", "sequence", "meta_note"], "labels": ["cat", "dog"],
        "alignment": alignment, "local_path": str(target_csv),
    })
    return {"ws": ws, "target_dir": target_dir, "target_csv": target_csv,
            "target_id": target_id, "self_id": self_id}


def _wire(monkeypatch, info: dict, captured: dict) -> None:
    monkeypatch.setattr(compare_service.project_manager, "get_project",
                        lambda pid: {"project_id": pid, "workspace_path": str(info["ws"]),
                                     "project_type": "original", "name": "proj"})
    monkeypatch.setattr(compare_service.knowledge_service, "get_latest_run",
                        lambda pid, run_type: {"run_id": "baseline-1",
                                               "metrics": json.dumps({"accuracy": 0.79}),
                                               "params": json.dumps({"task_type": "classification"})})

    async def _run_eval(project_id, task_id, data_dir, run_type="eval", dataset_label=None,
                        *, extra_params=None, allow_fixture=False):
        captured.setdefault("eval_calls", []).append({
            "data_dir": Path(data_dir), "run_type": run_type,
            "dataset_label": dataset_label, "extra_params": extra_params or {},
        })
        return {"run_id": f"eval-{len(captured['eval_calls'])}", "metrics": {"accuracy": 0.6}}

    monkeypatch.setattr(compare_service.baseline_service, "run_eval", _run_eval)

    async def _guidance(prompt, **kwargs):
        return {"structured_output": {"summary": "s", "guidance_content": "正文",
                                      "guidance_title": "t"}, "result": "ok"}

    monkeypatch.setattr(compare_service.agent_service, "run_sync", _guidance)

    def _record_knowledge(k):
        captured["knowledge"] = k
        return "k1"

    monkeypatch.setattr(compare_service.knowledge_service, "record_knowledge", _record_knowledge)


def test_compare_evaluates_on_aligned_copy(isolated_db, tmp_path, monkeypatch):
    info = _setup(tmp_path)
    original_text = info["target_csv"].read_text(encoding="utf-8")
    task_id = task_manager.create_task("compare", project_id="p1", params={"project_id": "p1"})
    captured: dict = {}
    _wire(monkeypatch, info, captured)

    asyncio.run(compare_service._run({"project_id": "p1"}, task_id))

    assert captured["eval_calls"], "应至少发起一次跨数据集评估"
    call = captured["eval_calls"][0]
    aligned_dir = info["target_dir"] / "aligned"
    # ① data_dir 是独立子目录下的对齐副本，不是原始数据目录
    assert call["data_dir"] == aligned_dir
    assert call["data_dir"] != info["target_dir"]
    assert (aligned_dir / "preprocessed.csv").exists()
    assert alignment_apply.is_aligned_copy(aligned_dir) is True
    # ② 原始文件未被改动
    assert info["target_csv"].read_text(encoding="utf-8") == original_text
    # ③ 运行记录标明对齐已生效与条目数
    assert call["extra_params"]["alignment_used"] is True
    assert call["extra_params"]["alignment_field_mapping_applied"] == 2
    assert call["extra_params"]["alignment_label_merge_applied"] == 1
    assert call["extra_params"]["aligned_copy"].endswith(str(Path("aligned") / "preprocessed.csv"))
    # 副本自身确实被映射/归并/规范化
    with (aligned_dir / "preprocessed.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys())[:4] == UNIFIED
    assert {r["label"] for r in rows} == {"Cat", "dog"}
    assert [r["input"] for r in rows] == ["ACGT", "TTTT"]


def test_comparison_table_records_alignment_usage(isolated_db, tmp_path, monkeypatch):
    info = _setup(tmp_path)
    task_id = task_manager.create_task("compare", project_id="p1", params={"project_id": "p1"})
    captured: dict = {}
    _wire(monkeypatch, info, captured)

    asyncio.run(compare_service._run({"project_id": "p1"}, task_id))

    table = captured["knowledge"]["structured"]["comparison"]
    assert table["alignment_used"] is True
    assert "对齐副本" in table["alignment_note"]
    entry = table["alignment"]["public-x"]
    assert entry["aligned"] is True
    assert entry["aligned_copy"].endswith(str(Path("aligned") / "preprocessed.csv"))
    assert entry["source_csv"].endswith("preprocessed.csv") and "public-x" in entry["source_csv"]
    assert entry["field_mapping_applied"] == 2
    assert entry["label_merge_applied"] == 1
    assert entry["alignment_status"] == "confirmed"
    assert entry["sequence"]["case"] == "upper"
    # 进度里也能查到同一份对比表
    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert progress["comparison"]["alignment"]


def test_unconfirmed_alignment_blocks_compare_without_copy(isolated_db, tmp_path, monkeypatch):
    info = _setup(tmp_path, status="draft")
    task_id = task_manager.create_task("compare", project_id="p1", params={"project_id": "p1"})
    captured: dict = {}
    _wire(monkeypatch, info, captured)

    with pytest.raises(RuntimeError, match="尚未确认"):
        asyncio.run(compare_service._run({"project_id": "p1"}, task_id))

    assert not captured.get("eval_calls"), "对齐未确认时不得发起评估"
    assert not (info["target_dir"] / "aligned").exists(), "对齐未确认时不得产出「已对齐」副本"
    assert "knowledge" not in captured
