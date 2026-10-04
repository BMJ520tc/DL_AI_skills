"""单位统一（5.1 步骤 2）回归：单列换算、同量纲多单位合并、不换算边界、无重名列。

背景（本次修复的缺陷）：`normalize_units` 只覆盖时间/字节/比例三类量纲，且同一量纲的多个
单位列（如 latency_ms 与 latency_us）归到同一目标列时**直接跳过不换算**（只记一条 skipped），
既没有统一单位，也没有把「其实是同一个量」的信息合并起来。

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


# ---------- ① 单一单位列换算：时间 / 长度 / 数据量 / 比例各一例 ----------

def test_single_unit_columns_are_converted(tmp_path):
    """时间→s、长度→m、数据量→mb（1024 进制）、比例→小数，列名带统一基准后缀。"""
    src = tmp_path / "units.csv"
    src.write_text(
        "id,input,label,latency_ms,length_cm,size_GB,acc(%)\n"
        "0,a,cat,10,150,1,90\n"
        "1,b,dog,20,250,2,80\n"
        "2,c,cat,30,50,0.5,70\n",
        encoding="utf-8",
    )

    summary = pp.run(src, tmp_path / "out", "units", "classification")
    assert summary["status"] == "ok"

    rows = _read_rows(tmp_path / "out")
    assert [float(r["meta_latency_s"]) for r in rows] == [0.01, 0.02, 0.03]
    assert [float(r["meta_length_m"]) for r in rows] == [1.5, 2.5, 0.5]
    # 数据量按 1024 进制：1 GB = 1024 MB
    assert [float(r["meta_size_mb"]) for r in rows] == [1024.0, 2048.0, 512.0]
    assert [float(r["meta_acc_ratio"]) for r in rows] == [0.9, 0.8, 0.7]
    # 旧列名不再存在，新列名落进统一 schema
    assert "meta_latency_ms" not in summary["fields"]
    assert {"meta_latency_s", "meta_length_m", "meta_size_mb", "meta_acc_ratio"} <= set(summary["fields"])

    units = summary["alignment"]["units"]
    assert units["latency_ms"]["to"] == "s" and units["latency_ms"]["factor"] == 1e-3
    assert units["length_cm"]["to"] == "m" and units["length_cm"]["factor"] == 1e-2
    assert units["size_GB"]["to"] == "mb" and units["size_GB"]["factor"] == 1024.0
    assert units["acc(%)"]["to"] == "ratio" and units["acc(%)"]["factor"] == 0.01


# ---------- ② 同量纲多单位合并（缺陷本体） ----------

def test_same_dimension_units_are_merged_into_one_column():
    """latency_ms 与 latency_us 都换算到 s 后**合并进同一列**，并记录 merged_from 与冲突行数。"""
    rows = [
        {"id": "0", "meta_latency_ms": "10", "meta_latency_us": "5000"},
        {"id": "1", "meta_latency_ms": "", "meta_latency_us": "7000"},
        {"id": "2", "meta_latency_ms": "30", "meta_latency_us": "30000"},
    ]

    out, columns, units = pp.normalize_units(rows, ["id", "meta_latency_ms", "meta_latency_us"])

    assert columns == ["id", "meta_latency_s"]
    # 先声明者（latency_ms）为准；它非空时忽略同时非空的 latency_us
    assert [r["meta_latency_s"] for r in out] == [0.01, 0.007, 0.03]
    # 来源列已并入目标列，不再残留
    assert "meta_latency_ms" not in out[0] and "meta_latency_us" not in out[0]

    for orig in ("latency_ms", "latency_us"):
        record = units[orig]
        assert record["to"] == "s" and record["column"] == "latency_s"
        assert record["merged_from"] == ["latency_ms", "latency_us"]
        assert record["conflicts"] == 2  # 第 0、2 行两列同时非空
    assert units["latency_ms"]["factor"] == 1e-3
    assert units["latency_us"]["factor"] == 1e-6


def test_merge_is_recorded_in_schema_json(tmp_path):
    """合并结果端到端落进 dataset.schema.json 的 alignment.units（不是只存在于单元调用里）。"""
    src = tmp_path / "merge.csv"
    src.write_text(
        "id,input,label,latency_ms,latency_us\n"
        "0,a,cat,10,5000\n"
        "1,b,dog,,7000\n"
        "2,c,cat,30,30000\n",
        encoding="utf-8",
    )

    summary = pp.run(src, tmp_path / "out", "merge", "classification")

    assert "meta_latency_s" in summary["fields"]
    assert "meta_latency_ms" not in summary["fields"] and "meta_latency_us" not in summary["fields"]
    rows = _read_rows(tmp_path / "out")
    assert [float(r["meta_latency_s"]) for r in rows] == [0.01, 0.007, 0.03]
    units = summary["alignment"]["units"]
    assert units["latency_ms"]["merged_from"] == ["latency_ms", "latency_us"]
    assert units["latency_ms"]["conflicts"] == 2


# ---------- ③ 不换算的边界（防回归） ----------

def test_non_numeric_and_sparse_columns_are_not_converted():
    """非数值列、非空值不足半数的列保持原列名，并如实记 skipped（不静默）。"""
    rows = [
        {"id": "0", "meta_dur_ms": "fast", "meta_x_ms": "1"},
        {"id": "1", "meta_dur_ms": "slow", "meta_x_ms": ""},
        {"id": "2", "meta_dur_ms": "medium", "meta_x_ms": ""},
    ]

    out, columns, units = pp.normalize_units(rows, ["id", "meta_dur_ms", "meta_x_ms"])

    assert columns == ["id", "meta_dur_ms", "meta_x_ms"]
    assert out[0]["meta_dur_ms"] == "fast" and out[0]["meta_x_ms"] == "1"
    assert "skipped" in units["dur_ms"] and "skipped" in units["x_ms"]
    assert units["dur_ms"]["to"] == "s" and units["x_ms"]["to"] == "s"


def test_bits_and_bytes_are_not_conflated():
    """Mb（兆比特）与 MB（兆字节）分属不同量纲，不混同、不互相折算。"""
    rows = [{"id": "0", "meta_size_Mb": "8", "meta_size_MB": "1"}]

    out, columns, units = pp.normalize_units(rows, ["id", "meta_size_Mb", "meta_size_MB"])

    assert columns == ["id", "meta_size_mbit", "meta_size_mb"]
    assert out[0]["meta_size_mbit"] == 8.0
    assert out[0]["meta_size_mb"] == 1.0
    assert units["size_Mb"]["to"] == "mbit" and units["size_MB"]["to"] == "mb"
    assert pp._unit_lookup("Mb") == ("mbit", 1.0)
    assert pp._unit_lookup("MB") == ("mb", 1.0)


def test_sparse_and_non_numeric_columns_do_not_create_target_columns():
    """边界列不换算时不得偷偷产出目标列（否则会凭空多出空列）。"""
    rows = [
        {"id": "0", "meta_x_ms": "1"},
        {"id": "1", "meta_x_ms": ""},
        {"id": "2", "meta_x_ms": ""},
    ]

    out, columns, _ = pp.normalize_units(rows, ["id", "meta_x_ms"])

    assert columns == ["id", "meta_x_ms"]
    assert "meta_x_s" not in out[0]


# ---------- ④ 合并后无重名列 ----------

def test_merged_columns_have_no_duplicates():
    """多列合并与消歧后，列名集合里没有重复项。"""
    rows = [{"id": "0", "meta_t_ms": "10", "meta_t_us": "100", "meta_l_cm": "1", "meta_l_mm": "5"}]

    out, columns, units = pp.normalize_units(
        rows, ["id", "meta_t_ms", "meta_t_us", "meta_l_cm", "meta_l_mm"]
    )

    assert columns == ["id", "meta_t_s", "meta_l_m"]
    assert len(columns) == len(set(columns))
    assert set(out[0]) == {"id", "meta_t_s", "meta_l_m"}
    assert units["t_ms"]["merged_from"] == ["t_ms", "t_us"]
    assert units["l_cm"]["merged_from"] == ["l_cm", "l_mm"]


def test_target_name_collision_gets_stable_disambiguation():
    """目标列名与既有列重名时改用稳定消歧名，并在记录里说明，绝不产出重名列。"""
    rows = [{"id": "0", "meta_acc(%)": "90", "meta_acc_ratio": "0.5"}]

    out, columns, units = pp.normalize_units(rows, ["id", "meta_acc(%)", "meta_acc_ratio"])

    assert len(columns) == len(set(columns))
    assert "meta_acc_ratio_2" in columns          # 消歧名
    assert out[0]["meta_acc_ratio"] == "0.5"      # 既有列原样保留
    assert out[0]["meta_acc_ratio_2"] == 0.9      # 换算结果进消歧列
    assert units["acc(%)"]["disambiguation"]["used"] == "meta_acc_ratio_2"
