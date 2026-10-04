"""预处理清洗口径与非静默告警（label 不得被填充/截断；找不到输入列必须告警）。

背景（本次修复的缺陷）：
1. `clean()` 里 value_cols 只排除了 id 列，于是 **label 列**也会被「缺失值填众数」与
   「异常值 IQR 截断」——填众数会把缺失样本划进多数类，截断会改写类别取值，两者都伪造/污染类别。
   label 与 id 一样是标识/语义列，不应参与数值填充与截断。
2. `unify()` 定位不到输入列时静默写空字符串，用户拿到 input 全空的数据集却看不出问题。

只用 tmp_path，不触碰仓库真实 data/。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import preprocess_dataset as pp  # noqa: E402


def _read_rows(out_dir: Path) -> list[dict]:
    with (out_dir / "preprocessed.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------- 缺陷 1：label 不参与填充/截断 ----------

def test_label_missing_is_not_filled_with_mode(tmp_path):
    """label 列有缺失、且该列众数存在时，不得用众数填 label（保留为空并计数）。"""
    src = tmp_path / "with_missing_label.csv"
    src.write_text("id,input,label\n0,a,cat\n1,b,cat\n2,c,\n", encoding="utf-8")

    summary = pp.run(src, tmp_path / "out", "labels", "classification")

    assert summary["status"] == "ok"
    rows = _read_rows(tmp_path / "out")
    # 众数是 cat，但缺失行必须保持为空：被填成 cat 就是凭空造类别（伪造/污染标签）
    assert [r["label"] for r in rows] == ["cat", "cat", ""]

    cleaning = summary["cleaning"]
    assert cleaning["label_missing"] == 1                 # 如实计数：清洗后 label 仍为空的行数
    assert "label" not in cleaning["filled"]              # label 不在填充统计里
    assert "label" in cleaning["excluded_from_fill_and_clip"]


def test_numeric_label_outlier_is_not_clipped(tmp_path):
    """数值型 label 的极端值不得被 IQR 截断到分位边界（截断即改写类别取值）。"""
    src = tmp_path / "numeric_labels.csv"
    src.write_text(
        "id,input,label\n"
        "0,a,0\n1,b,1\n2,c,0\n3,d,1\n4,e,999\n5,f,-999\n",
        encoding="utf-8",
    )

    summary = pp.run(src, tmp_path / "out", "numeric-labels", "classification")

    labels = [r["label"] for r in _read_rows(tmp_path / "out")]
    # 若 label 进了 IQR，999/-999 会被压到上/下界（约 2.5 / -2.5）
    assert "999" in labels and "-999" in labels
    assert summary["cleaning"]["outliers_clipped"] == {}
    assert "label" not in summary["cleaning"]["outliers_clipped"]


def test_long_string_label_is_not_truncated(tmp_path):
    """超长字符串 label 不得被截断或改写（只做大小写归并，不改长度）。"""
    long_label = "class-" + "y" * 300
    src = tmp_path / "long_label.csv"
    src.write_text(f"id,input,label\n0,a,cat\n1,b,{long_label}\n", encoding="utf-8")

    summary = pp.run(src, tmp_path / "out", "long-labels", "classification")

    labels = [r["label"] for r in _read_rows(tmp_path / "out")]
    assert long_label in labels
    assert summary["cleaning"]["excluded_from_fill_and_clip"] == ["id", "label"]


# ---------- 防回归：普通数值列仍按既有策略填充与截断 ----------

def test_numeric_meta_column_is_still_filled_and_clipped(tmp_path):
    """普通数值列（meta_*）仍按既有策略：缺失填中位数、异常值 IQR 截断到边界。"""
    src = tmp_path / "meta_numeric.csv"
    src.write_text(
        "id,input,label,score\n"
        "0,a,cat,1\n1,b,cat,2\n2,c,dog,3\n3,d,dog,4\n4,e,cat,\n5,f,dog,1000\n",
        encoding="utf-8",
    )

    summary = pp.run(src, tmp_path / "out", "meta-numeric", "classification")

    cleaning = summary["cleaning"]
    assert cleaning["filled"]["meta_score"] == 1          # 缺失值被填充（中位数 3.0）
    assert cleaning["outliers_clipped"]["meta_score"] == 1  # 1000 被截断

    values = sorted(float(r["meta_score"]) for r in _read_rows(tmp_path / "out"))
    # 填充后为 [1,2,3,3,4,1000]：q1=2.25、q3=3.75、上界 6.0 → 1000 截断到 6.0
    assert values == [1.0, 2.0, 3.0, 3.0, 4.0, 6.0]
    # 同一份数据里 label 不受影响
    assert {r["label"] for r in _read_rows(tmp_path / "out")} == {"cat", "dog"}


# ---------- 缺陷 2：找不到输入列必须告警 ----------

def test_missing_input_column_emits_warning(tmp_path):
    """列名都不匹配输入别名时，input 会整列为空——必须留下带列名候选的告警。"""
    src = tmp_path / "no_input.csv"
    src.write_text("id,label,attr1\n0,cat,0.1\n1,dog,0.2\n2,cat,0.3\n", encoding="utf-8")

    summary = pp.run(src, tmp_path / "out", "no-input", "classification")

    assert summary["status"] == "ok"
    warnings = summary["alignment"]["warnings"]
    warn = next(w for w in warnings if w["code"] == "missing_input_column")
    # 带列名候选：原始列名 + 可接受的别名，便于人工定位该怎么改
    assert warn["columns"] == ["id", "label", "attr1"]
    assert "input" in warn["candidates"]
    assert warn["affected_rows"] == 3
    assert all(r["input"] == "" for r in _read_rows(tmp_path / "out"))


def test_unify_returns_empty_warnings_when_columns_found():
    """找得到输入列时不产生告警；unify 的第 4 项（新增）始终存在，便于调用方解包。"""
    unified, mapping, label_merge, warnings = pp.unify(
        [{"input": "a", "label": "cat"}], ["input", "label"]
    )
    assert warnings == []
    assert mapping == {"label": "label", "input": "input"}
    assert label_merge == {}
    assert unified[0]["input"] == "a"
