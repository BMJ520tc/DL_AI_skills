"""对齐应用（缺口二）：alignment 必须真正作用到评估输入上。

覆盖《模块详细设计》5.3/5.4：按已确认的对齐规则把目标数据集 `preprocessed.csv`
转成独立子目录 `aligned/` 下的副本——字段映射、标签归并、序列规范真实生效，
且**原始文件不被改动**；未确认的对齐不得应用（沿用确认闸门）。
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.services import alignment_apply

CONFIRMED = {
    "version": "1.0",
    "status": "confirmed",
    "field_mapping": {"sample_id": "id", "sequence": "input"},
    "label_merge": {"CAT": "Cat", "cat": "Cat", "Dog": "dog"},
    "sequence": {"length_range": [4, 6], "case": "upper"},
}


def _write_target(tmp_path: Path) -> tuple[Path, str]:
    d = tmp_path / "public-x"
    d.mkdir(parents=True)
    path = d / "preprocessed.csv"
    rows = [
        # 列名未统一（sample_id/sequence），标签大小写不一，序列超界/小写
        {"sample_id": "1", "split": "train", "label": "CAT", "sequence": "acgtacgt", "meta_note": "n1"},
        {"sample_id": "2", "split": "train", "label": "cat", "sequence": "acgt", "meta_note": "n2"},
        {"sample_id": "3", "split": "test", "label": "Dog", "sequence": "ACGTAC", "meta_note": "n3"},
        {"sample_id": "4", "split": "test", "label": "bird", "sequence": "acGTa", "meta_note": "n4"},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "label", "sequence", "meta_note"])
        writer.writeheader()
        writer.writerows(rows)
    return d, path.read_text(encoding="utf-8")


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_apply_alignment_writes_copy_and_leaves_original_untouched(tmp_path):
    target, original_text = _write_target(tmp_path)
    record = alignment_apply.apply_alignment(CONFIRMED, target)

    copy = target / "aligned" / "preprocessed.csv"
    assert copy.exists()
    assert Path(record["aligned_csv"]) == copy
    assert record["original_unchanged"] is True
    # 原始文件逐字节未变
    assert (target / "preprocessed.csv").read_text(encoding="utf-8") == original_text
    assert record["source_sha256"] == alignment_apply._sha256(target / "preprocessed.csv")


def test_field_mapping_renames_columns_to_unified_schema(tmp_path):
    target, _ = _write_target(tmp_path)
    alignment_apply.apply_alignment(CONFIRMED, target)
    rows = _read(target / "aligned" / "preprocessed.csv")

    assert list(rows[0].keys())[:4] == ["id", "split", "label", "input"]
    assert "sample_id" not in rows[0] and "sequence" not in rows[0]
    # 未参与映射的列原样保留
    assert rows[0]["meta_note"] == "n1"
    assert [r["id"] for r in rows] == ["1", "2", "3", "4"]


def test_label_merge_brings_target_labels_into_baseline_domain(tmp_path):
    target, _ = _write_target(tmp_path)
    record = alignment_apply.apply_alignment(CONFIRMED, target)
    rows = _read(target / "aligned" / "preprocessed.csv")

    labels = {r["label"] for r in rows}
    assert labels <= {"Cat", "dog", "bird"}          # 归并后取值域与基准写法一致
    assert "CAT" not in labels and "cat" not in labels and "Dog" not in labels
    assert record["label_merge_applied"] == 3        # CAT/cat/Dog 三条被改写
    assert record["label_merge_entries"] == 3


def test_sequence_normalization_truncates_and_cases(tmp_path):
    target, _ = _write_target(tmp_path)
    record = alignment_apply.apply_alignment(CONFIRMED, target)
    rows = _read(target / "aligned" / "preprocessed.csv")

    values = [r["input"] for r in rows]
    assert values == ["ACGTAC", "ACGT", "ACGTAC", "ACGTA"]   # 8→6 截断，统一大写
    assert all(len(v) <= 6 for v in values)
    assert record["sequence"]["truncated"] == 1
    assert record["sequence"]["case"] == "upper"
    assert record["sequence"]["length_range"] == [4, 6]


def test_apply_writes_reviewable_record(tmp_path):
    target, _ = _write_target(tmp_path)
    alignment_apply.apply_alignment(CONFIRMED, target)
    record_path = target / "aligned" / "alignment_applied.json"
    assert record_path.exists()
    import json

    saved = json.loads(record_path.read_text(encoding="utf-8"))
    assert saved["field_mapping_applied"] == 2
    assert saved["alignment_status"] == "confirmed"
    assert saved["source_csv"].endswith("preprocessed.csv")
    assert saved["columns"][:4] == ["id", "split", "label", "input"]


def test_unconfirmed_alignment_is_rejected(tmp_path):
    target, _ = _write_target(tmp_path)
    draft = {**CONFIRMED, "status": "draft"}
    with pytest.raises(RuntimeError) as exc:
        alignment_apply.apply_alignment(draft, target)
    assert "尚未确认" in str(exc.value)
    assert not (target / "aligned").exists()          # 未确认不得产出「已对齐」副本


def test_missing_source_csv_is_rejected(tmp_path):
    target = tmp_path / "empty"
    target.mkdir()
    with pytest.raises(RuntimeError) as exc:
        alignment_apply.apply_alignment(CONFIRMED, target)
    assert "缺少 preprocessed.csv" in str(exc.value)


def test_empty_rules_are_rejected(tmp_path):
    """已确认但没有任何可应用规则 → 不得复制一份原始文件「假装已对齐」。"""
    target, _ = _write_target(tmp_path)
    empty = {"version": "1.0", "status": "confirmed",
             "field_mapping": {}, "label_merge": {}, "sequence": {}}
    with pytest.raises(RuntimeError, match="对齐规则为空"):
        alignment_apply.apply_alignment(empty, target)
    assert not (target / "aligned").exists()


def test_second_apply_is_idempotent_and_still_keeps_original(tmp_path):
    target, original_text = _write_target(tmp_path)
    alignment_apply.apply_alignment(CONFIRMED, target)
    first = (target / "aligned" / "preprocessed.csv").read_text(encoding="utf-8")
    alignment_apply.apply_alignment(CONFIRMED, target)
    assert (target / "aligned" / "preprocessed.csv").read_text(encoding="utf-8") == first
    assert (target / "preprocessed.csv").read_text(encoding="utf-8") == original_text


def test_sequence_spec_accepts_alternative_shapes():
    spec = alignment_apply._sequence_spec({"min_len": "2", "max_len": 10, "case": "lowercase"})
    assert spec == {"case": "lower", "min_length": 2, "max_length": 10}
    nested = alignment_apply._sequence_spec({"input": {"length_range": [1, 3]}, "case": "upper"})
    assert nested == {"case": "upper", "min_length": 1, "max_length": 3}
    assert alignment_apply._sequence_spec({})["case"] == "keep"
