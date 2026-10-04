"""模块三 5.3/5.4 对齐规则的实际落地（把 alignment 真正作用于评估输入）。

背景（缺口二）：`dataset_registry.alignment` 先前只被落库与读取——`compare_service` 只把
**原始** `data_dir` 交给 eval 入口，alignment 仅当「是否 confirmed」的闸门使用，从未作用到
数据上。本模块补上消费端：按已确认对齐规则，把目标数据集目录里的 `preprocessed.csv`
转成**对齐后的副本**，落到数据集目录下的独立子目录 `aligned/`。

约束：
- **绝不覆盖原始文件**：只读原始 `preprocessed.csv`，产物写 `aligned/preprocessed.csv`，
  并在写后复核原始文件哈希未变（变了即报错，不掩盖）。
- **只应用已确认的对齐**：`status != "confirmed"` 直接失败（沿用 5.3 的确认闸门），
  不允许「假装已对齐」。
- 三类规则真实生效：
  * `field_mapping`：目标字段 → 统一字段（id/split/label/input），列改名/取列；
  * `label_merge`：目标标签 → 归并标签，改写 label 列取值；
  * `sequence`：序列长度范围（超长截断）与大小写规范（upper/lower）；
    短于下限的**不补齐**（补齐等于凭空造数据，沿用 5.1 口径），只如实计数。

产物旁边写 `aligned/alignment_applied.json`，记录映射/归并条目数、截断条数、原始文件
哈希与对齐状态，供运行记录与对比表引用（可复核）。
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ALIGNED_SUBDIR = "aligned"
PREPROCESSED_NAME = "preprocessed.csv"
RECORD_NAME = "alignment_applied.json"
UNIFIED_FIELDS = ("id", "split", "label", "input")

_CASE_ALIASES = {
    "upper": "upper",
    "uppercase": "upper",
    "upper_case": "upper",
    "lower": "lower",
    "lowercase": "lower",
    "lower_case": "lower",
    "none": "keep",
    "keep": "keep",
    "asis": "keep",
    "as-is": "keep",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        columns = [str(c) for c in (reader.fieldnames or [])]
    return rows, columns


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})


def _as_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _sequence_spec(raw: dict) -> dict:
    """把 alignment.sequence 解析成 {case, min_length, max_length}。

    兼容 agent 起草结果的多种写法：`length_range: [min, max]`、`min_length`/`max_length`、
    `min_len`/`max_len`/`truncate_at`，以及嵌在 `input` 子对象里的写法。
    """
    seq = raw if isinstance(raw, dict) else {}
    nested = seq.get("input")
    merged = {**seq, **nested} if isinstance(nested, dict) else dict(seq)

    case_raw = merged.get("case")
    case = _CASE_ALIASES.get(str(case_raw).strip().lower() if case_raw is not None else "keep", "keep")

    min_len = _as_int(merged.get("min_length") if merged.get("min_length") is not None else merged.get("min_len"))
    max_len = _as_int(merged.get("max_length") if merged.get("max_length") is not None else merged.get("max_len"))
    rng = merged.get("length_range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        min_len = min_len if min_len is not None else _as_int(rng[0])
        max_len = max_len if max_len is not None else _as_int(rng[1])
    if max_len is None:
        max_len = _as_int(merged.get("truncate_at"))
    return {"case": case, "min_length": min_len, "max_length": max_len}


def _map_fields(rows: list[dict], columns: list[str], field_mapping: dict) -> tuple[list[dict], list[str], dict, list[str]]:
    """按 field_mapping 改名/取列；返回 (行, 新列序, 实际改名, 被丢弃的冲突列)。"""
    rename: dict[str, str] = {}
    for src, dst in (field_mapping or {}).items():
        src, dst = str(src), str(dst)
        if dst in UNIFIED_FIELDS:
            rename[src] = dst

    used: dict[str, str] = {}  # 目标列名 -> 源列名
    dropped: list[str] = []
    for col in columns:
        dst = rename.get(col, col)
        if dst in used:
            # 两列映射到同一统一字段：保留先出现的，其余如实登记为丢弃，避免静默覆盖
            dropped.append(col)
            continue
        used[dst] = col

    unified = [u for u in UNIFIED_FIELDS if u in used]
    rest = [d for d in used if d not in UNIFIED_FIELDS]
    out_columns = unified + rest
    out_rows = [{dst: row.get(src) for dst, src in used.items()} for row in rows]
    applied = {src: dst for src, dst in rename.items() if src in columns and src != dst}
    return out_rows, out_columns, applied, dropped


def _merge_labels(rows: list[dict], columns: list[str], label_merge: dict) -> tuple[int, int]:
    """改写 label 列取值；返回 (被改写的行数, 命中的归并条目数)。"""
    if "label" not in columns or not label_merge:
        return 0, 0
    mapping = {str(k): str(v) for k, v in label_merge.items()}
    hit_entries: set[str] = set()
    changed_rows = 0
    for row in rows:
        value = row.get("label")
        key = "" if value is None else str(value)
        if key in mapping and mapping[key] != key:
            row["label"] = mapping[key]
            hit_entries.add(key)
            changed_rows += 1
    return changed_rows, len(hit_entries)


def _normalize_sequences(rows: list[dict], columns: list[str], spec: dict) -> dict:
    """序列规范：大小写 + 超长截断（不补齐短序列）。"""
    stats = {"case": spec.get("case"), "max_length": spec.get("max_length"),
             "min_length": spec.get("min_length"), "truncated": 0, "short": 0,
             "length_range": [0, 0]}
    if "input" not in columns:
        return stats

    case = spec.get("case")
    for row in rows:
        value = row.get("input")
        if value is None:
            continue
        text = str(value)
        if case == "upper":
            row["input"] = text.upper()
        elif case == "lower":
            row["input"] = text.lower()

    max_len, min_len = spec.get("max_length"), spec.get("min_length")
    lengths: list[int] = []
    for row in rows:
        text = "" if row.get("input") is None else str(row.get("input"))
        if max_len is not None and max_len >= 0 and len(text) > max_len:
            text = text[:max_len]
            row["input"] = text
            stats["truncated"] += 1
        if min_len is not None and text and len(text) < min_len:
            stats["short"] += 1
        if text:
            lengths.append(len(text))
    stats["length_range"] = [min(lengths), max(lengths)] if lengths else [0, 0]
    return stats


def aligned_dir_of(target_dir: Path, subdir: str = ALIGNED_SUBDIR) -> Path:
    """数据集目录下对齐副本的固定位置（`<数据集目录>/aligned`）。"""
    return Path(target_dir) / subdir


def is_aligned_copy(path: Path, subdir: str = ALIGNED_SUBDIR) -> bool:
    """判断某目录是否本模块产出的对齐副本。"""
    p = Path(path)
    return p.name == subdir and (p / RECORD_NAME).exists()


def apply_alignment(alignment: dict, target_dir: Path, *, subdir: str = ALIGNED_SUBDIR) -> dict:
    """把已确认的 alignment 应用到目标数据集目录，产出对齐副本并返回可复核摘要。

    失败（对齐未确认、原始 CSV 缺失、原始文件被改动）一律抛 RuntimeError，不静默降级。
    """
    alignment = alignment or {}
    if alignment.get("status") != "confirmed":
        raise RuntimeError(
            "对齐规则尚未确认，不能用于评估（请先调用 POST /api/datasets/{id}/alignment/confirm）"
        )
    if not (alignment.get("field_mapping") or alignment.get("label_merge") or alignment.get("sequence")):
        # 不完整对齐：没有任何可应用的规则，复制一份原始文件只会「假装已对齐」
        raise RuntimeError("对齐规则为空（field_mapping/label_merge/sequence 均为空），不能用于评估")

    target_dir = Path(target_dir)
    src_csv = target_dir / PREPROCESSED_NAME
    if not src_csv.exists():
        raise RuntimeError(f"目标数据集缺少 {PREPROCESSED_NAME}: {src_csv}")

    out_dir = aligned_dir_of(target_dir, subdir)
    out_csv = out_dir / PREPROCESSED_NAME
    if out_csv.resolve() == src_csv.resolve():
        raise RuntimeError("对齐副本路径与原始文件相同，拒绝覆盖原始数据")

    before = _sha256(src_csv)
    rows, columns = _read_csv(src_csv)
    out_rows, out_columns, applied_fields, dropped = _map_fields(
        rows, columns, alignment.get("field_mapping") or {}
    )
    label_changed, label_entries = _merge_labels(
        out_rows, out_columns, alignment.get("label_merge") or {}
    )
    seq_stats = _normalize_sequences(out_rows, out_columns, _sequence_spec(alignment.get("sequence")))

    _write_csv(out_csv, out_rows, out_columns)

    after = _sha256(src_csv)
    if after != before:
        raise RuntimeError(f"对齐副本写出过程中原始文件被改动: {src_csv}")

    record = {
        "status": "applied",
        "applied_at": _now(),
        "alignment_status": alignment.get("status"),
        "alignment_drafted_by": alignment.get("drafted_by"),
        "source_csv": str(src_csv),
        "source_sha256": before,
        "aligned_csv": str(out_csv),
        "field_mapping": applied_fields,
        "field_mapping_applied": len(applied_fields),
        "dropped_columns": dropped,
        "label_merge_applied": label_changed,
        "label_merge_entries": label_entries,
        "sequence": seq_stats,
        "original_unchanged": True,
        "columns": out_columns,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RECORD_NAME).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
