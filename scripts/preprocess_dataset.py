"""模块三 5.1 数据预处理固定脚本（模块详细设计 5.1）。

用法: python preprocess_dataset.py <input> <out_dir> [--dataset-name NAME] [--task-type T]

支持: CSV / Excel(.xlsx) / 图片目录 / 压缩包(zip、tar)；
      FASTA/PDB 与 .xls 识别后返回 unsupported（本阶段只留接口位）。

输出: <out_dir>/preprocessed.csv（统一 schema）与
      <out_dir>/dataset.schema.json（统一 schema + alignment + 清洗统计）。
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

UNIFIED_COLUMNS = ("id", "split", "label", "input")
META_PREFIX = "meta_"

INPUT_ALIASES = ("input", "image", "img", "path", "file", "filename", "filepath",
                 "text", "sentence", "sequence", "seq")
LABEL_ALIASES = ("label", "labels", "class", "category", "target", "y", "tag")
ID_ALIASES = ("id", "index", "idx", "uid", "name")
SPLIT_ALIASES = ("split", "set", "phase", "subset")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
TABLE_EXT = {".csv", ".tsv", ".xlsx", ".xls"}
UNSUPPORTED_EXT = {".fasta", ".fa", ".fna", ".pdb", ".ent"}
ARCHIVE_EXT = {".zip", ".tar", ".gz", ".tgz"}

# 图片目录中「划分」层级的目录名（不是类别名）；真实数据集的
# <root>/train/<class>/*.jpg 结构会把 train 误当标签，故需跳过并还原为 split。
SPLIT_DIR_NAMES = {
    "train": "train", "training": "train",
    "test": "test", "testing": "test",
    "val": "val", "valid": "val", "validation": "val", "dev": "val", "eval": "val",
}

MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-"}

# 单位统一（5.1 步骤 2 的量纲部分）：识别列名中的单位后缀并换算到统一量纲
# 统一目标：时间→秒，字节量→MB，比例→小数
_UNIT_TABLE = {
    "ns": ("s", 1e-9), "us": ("s", 1e-6), "µs": ("s", 1e-6), "ms": ("s", 1e-3),
    "s": ("s", 1.0), "sec": ("s", 1.0), "secs": ("s", 1.0),
    "second": ("s", 1.0), "seconds": ("s", 1.0),
    "min": ("s", 60.0), "mins": ("s", 60.0), "minute": ("s", 60.0), "minutes": ("s", 60.0),
    "h": ("s", 3600.0), "hr": ("s", 3600.0), "hour": ("s", 3600.0), "hours": ("s", 3600.0),
    "%": ("ratio", 0.01), "pct": ("ratio", 0.01), "percent": ("ratio", 0.01),
    "b": ("mb", 1 / 1048576), "kb": ("mb", 1 / 1024), "mb": ("mb", 1.0), "gb": ("mb", 1024.0),
}
# 要求单位前有分隔符（`latency_ms`、`duration(s)`），避免把 Mass/Address 这类词尾误判为单位
_UNIT_RE = re.compile(r"[_\s(（\-]+([A-Za-zµ%]{1,7})[)）]?$")


def _fail(fmt: str, message: str) -> dict:
    return {"status": "unsupported", "format": fmt, "message": message}


def detect_format(path: Path) -> str:
    if path.is_dir():
        return "image_dir"
    ext = path.suffix.lower()
    if ext in UNSUPPORTED_EXT:
        return ext.lstrip(".")
    if ext in {".xlsx", ".xls"}:
        return "excel" if ext == ".xlsx" else "xls"
    if ext in {".csv", ".tsv"}:
        return "csv"
    if ext in IMAGE_EXT:
        return "image_dir"
    if ext in ARCHIVE_EXT or path.name.lower().endswith((".tar.gz", ".tar.bz2")):
        return "archive"
    return "unknown"


# ---------- 各格式读取：统一产出 (rows: list[dict], columns: list[str]) ----------

def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    for enc in ("utf-8-sig", "gbk", "latin-1"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                sample = f.read(8192)
                f.seek(0)
                delim = "\t" if path.suffix.lower() == ".tsv" else None
                if delim is None:
                    delim = "\t" if sample.count("\t") > sample.count(",") else ","
                reader = csv.DictReader(f, delimiter=delim)
                rows = [dict(r) for r in reader]
                return rows, list(reader.fieldnames or [])
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法解码 CSV 文件: {path}")


def _read_excel(path: Path) -> tuple[list[dict], list[str]]:
    try:
        from openpyxl import load_workbook
    except ImportError as e:
        raise RuntimeError("缺少 openpyxl，无法读取 Excel") from e

    wb = load_workbook(filename=str(path), read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    it = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else f"col{i}" for i, c in enumerate(next(it))]
    rows = []
    for values in it:
        row = {header[i]: values[i] for i in range(min(len(header), len(values)))}
        if any(v is not None and str(v).strip() != "" for v in row.values()):
            rows.append(row)
    wb.close()
    return rows, header


def _image_label_and_split(p: Path, root: Path) -> tuple[str, str]:
    """从图片所在目录推导 (label, split)。

    目录层级可能是 ``<root>/<split>/<class>/``、``<root>/<class>/<split>/``、
    只有 ``<root>/<class>/`` 或完全扁平（类别写在文件名里，如 ``dog.0.jpg``）：
    取沿途第一个「非 split 名」的目录作为 label，出现过的 split 名归一到 train/test/val；
    没有类别目录时退回文件名首个分隔符前的词，仍取不到则 unknown。
    """
    label, split = "", ""
    for part in p.parent.relative_to(root).parts:
        canonical = SPLIT_DIR_NAMES.get(part.casefold())
        if canonical:
            split = split or canonical
        elif not label:
            label = part
    if not label:
        label = re.split(r"[._\-\s]", p.stem)[0].strip() or "unknown"
    return label, split


def _read_image_dir(path: Path) -> tuple[list[dict], list[str]]:
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("缺少 Pillow，无法读取图片") from e

    rows: list[dict] = []
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        label, split = _image_label_and_split(p, path)
        width = height = None
        mode = None
        try:
            with Image.open(p) as im:
                width, height, mode = im.width, im.height, im.mode
        except OSError:
            continue
        row = {"path": str(p), "label": label, "width": width, "height": height, "mode": mode}
        if split:
            row["split"] = split
        rows.append(row)
    columns = ["path", "label", "width", "height", "mode"]
    if any("split" in r for r in rows):
        columns.insert(1, "split")
    return rows, columns


def _read_one(path: Path) -> tuple[list[dict], list[str]]:
    fmt = detect_format(path)
    if fmt == "csv":
        return _read_csv(path)
    if fmt == "excel":
        return _read_excel(path)
    if fmt == "image_dir":
        return _read_image_dir(path)
    raise RuntimeError(f"不可直接读取的格式: {fmt}")


def _extract_archive(path: Path, dest: Path) -> None:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(dest)
        return
    raise RuntimeError(f"无法识别的压缩包: {path}")


# ---------- 统一化 ----------

def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[_\s(\-./]+", name.lower().strip()) if t]


def _match_alias(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    """按「整名精确 → 分词精确」两级匹配列名。

    不做朴素子串匹配：别名里的 y、id 这类短词会命中 latency、valid 等无关列名。
    """
    lowered = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    for col in columns:
        if any(token in aliases for token in _tokens(col)):
            return col
    return None


def _is_missing(v) -> bool:
    return v is None or str(v).strip().lower() in MISSING


def _to_float(v) -> float | None:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _looks_like_sequence(values: list[str]) -> bool:
    """判定是否为序列列：绝大多数取值是纯字母，且典型长度达到序列量级。

    不要求每个值都够长——真实序列长度参差，短序列是正常的。
    """
    if not values:
        return False
    alpha_ratio = sum(1 for v in values if v.isalpha()) / len(values)
    lengths = sorted(len(v) for v in values)
    median_len = lengths[len(lengths) // 2]
    return alpha_ratio >= 0.8 and median_len >= 10


def unify(rows: list[dict], columns: list[str]) -> tuple[list[dict], dict, dict]:
    """映射到统一 schema；返回 (统一行, field_mapping, labels)。"""
    col_in = _match_alias(columns, INPUT_ALIASES)
    col_label = _match_alias(columns, LABEL_ALIASES)
    col_id = _match_alias(columns, ID_ALIASES)
    col_split = _match_alias(columns, SPLIT_ALIASES)

    field_mapping: dict[str, str] = {}
    unified: list[dict] = []
    raw_labels: list[str] = []

    for idx, row in enumerate(rows):
        out: dict = {}
        out["id"] = str(row.get(col_id)) if col_id and not _is_missing(row.get(col_id)) else str(idx)
        out["split"] = str(row.get(col_split)).strip() if col_split and not _is_missing(row.get(col_split)) else "train"
        out["label"] = str(row.get(col_label)).strip() if col_label and not _is_missing(row.get(col_label)) else ""
        out["input"] = str(row.get(col_in)).strip() if col_in and not _is_missing(row.get(col_in)) else ""
        if out["label"]:
            raw_labels.append(out["label"])
        mapped = {col_id, col_split, col_label, col_in}
        for c in columns:
            if c in mapped:
                continue
            out[META_PREFIX + c] = row.get(c)
        unified.append(out)

    for src, dst in ((col_id, "id"), (col_split, "split"), (col_label, "label"), (col_in, "input")):
        if src:
            field_mapping[src] = dst

    # 标签归并：同义写法（大小写/首尾空白）归并到出现最多的原始写法
    label_merge: dict[str, str] = {}
    if raw_labels:
        groups: dict[str, Counter] = {}
        for lab in raw_labels:
            groups.setdefault(lab.casefold(), Counter())[lab] += 1
        canonical = {k: v.most_common(1)[0][0] for k, v in groups.items()}
        for lab in dict.fromkeys(raw_labels):
            canon = canonical[lab.casefold()]
            if lab != canon:
                label_merge[lab] = canon
        for out in unified:
            if out["label"]:
                out["label"] = canonical[out["label"].casefold()]
    return unified, field_mapping, label_merge


def normalize_units(unified: list[dict], columns: list[str]) -> tuple[list[dict], list[str], dict]:
    """单位统一（5.1 步骤 2 的量纲部分）：列名带单位后缀的数值列换算到统一量纲并改名。

    返回 (统一行, 新列名列表, units 记录)。未命中任何单位时 units 为空。
    """
    units: dict = {}
    rename: dict = {}
    for col in columns:
        if not col.startswith(META_PREFIX):
            continue
        orig = col[len(META_PREFIX):]
        m = _UNIT_RE.search(orig)
        if not m:
            continue
        token = m.group(1).lower()
        if token not in _UNIT_TABLE:
            continue
        base, factor = _UNIT_TABLE[token]
        values = [_to_float(r.get(col)) for r in unified]
        present = [v for v in values if v is not None]
        # 仅对数值列换算：非空值须过半且都能转成数字
        if not present or len(present) * 2 < len(unified):
            continue
        new_orig = f"{orig[:m.start()].rstrip('_ -')}_{base}"
        new_col = META_PREFIX + new_orig
        for row in unified:
            value = _to_float(row.get(col))
            row[new_col] = round(value * factor, 10) if value is not None else row.get(col)
            if new_col != col:  # 同名时不能 pop，否则刚写入的值会被删掉
                row.pop(col, None)
        rename[col] = new_col
        units[orig] = {"column": new_orig, "from": token, "to": base, "factor": factor}
    return unified, [rename.get(c, c) for c in columns], units


def normalize_sequences(unified: list[dict]) -> tuple[list[dict], dict | None, int]:
    """序列规范（5.1 步骤 4）：统一大小写，并把超长序列截断到 IQR 上界。

    不补齐短序列（补齐会凭空造数据）；截断条数记入返回的第三项。
    """
    inputs = [str(r.get("input", "")).strip() for r in unified]
    present = [v for v in inputs if v]
    if not _looks_like_sequence(present):
        return unified, None, 0

    for row in unified:
        value = str(row.get("input", "")).strip()
        if value:
            row["input"] = value.upper()

    lengths = sorted(len(str(r["input"])) for r in unified if str(r.get("input", "")).strip())
    q1, q3 = _quantile(lengths, 0.25), _quantile(lengths, 0.75)
    upper = q3 + 1.5 * (q3 - q1)
    truncated = 0
    if upper > 0:
        for row in unified:
            value = str(row.get("input", ""))
            if len(value) > upper:
                row["input"] = value[: int(upper)]
                truncated += 1

    final = [len(str(r["input"])) for r in unified if str(r.get("input", "")).strip()]
    return unified, {
        "length_range": [min(final), max(final)],
        "case": "upper",
        "truncated": truncated,
    }, truncated


def clean(unified: list[dict], columns: list[str]) -> tuple[list[dict], dict]:
    """清洗：去重、缺失值、异常值。返回 (清洗后行, 统计)。"""
    stats = {"duplicates_removed": 0, "rows_dropped_missing": 0, "filled": {}, "outliers_clipped": {}}

    # 去重：以内容字段的哈希为键（排除生成的 id，否则行号不同会导致重复行无法识别）
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in unified:
        content = {k: v for k, v in row.items() if k != "id"}
        key = hashlib.sha1(json.dumps(content, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        deduped.append(row)

    # 缺失值：整行缺失 >50% 删除；数值列填中位数、其余填众数
    value_cols = list(columns)
    kept: list[dict] = []
    for row in deduped:
        missing = sum(1 for c in value_cols if _is_missing(row.get(c)))
        if value_cols and missing / len(value_cols) > 0.5:
            stats["rows_dropped_missing"] += 1
            continue
        kept.append(row)

    numeric_cols: list[str] = []
    for c in value_cols:
        vals = [row.get(c) for row in kept if not _is_missing(row.get(c))]
        if vals and all(_to_float(v) is not None for v in vals):
            numeric_cols.append(c)

    for c in value_cols:
        present = [row.get(c) for row in kept if not _is_missing(row.get(c))]
        if not present:
            continue
        if c in numeric_cols:
            fill = statistics.median([_to_float(v) for v in present])
        else:
            fill = Counter(str(v) for v in present).most_common(1)[0][0]
        filled = 0
        for row in kept:
            if _is_missing(row.get(c)):
                row[c] = fill
                filled += 1
        if filled:
            stats["filled"][c] = filled

    # 异常值：IQR 1.5 倍，超出者截断到边界并计数
    for c in numeric_cols:
        vals = sorted(_to_float(row.get(c)) for row in kept)
        if len(vals) < 4:
            continue
        q1 = _quantile(vals, 0.25)
        q3 = _quantile(vals, 0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clipped = 0
        for row in kept:
            v = _to_float(row.get(c))
            if v is None:
                continue
            if v < lo:
                row[c] = lo
                clipped += 1
            elif v > hi:
                row[c] = hi
                clipped += 1
        if clipped:
            stats["outliers_clipped"][c] = clipped
    return kept, stats


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


# ---------- 入口 ----------

def _write_outputs(out_dir: Path, rows: list[dict], columns: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "preprocessed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})
    return csv_path


def run(input_path: Path, out_dir: Path, dataset_name: str, task_type: str) -> dict:
    fmt = detect_format(input_path)
    if fmt in {"fasta", "fa", "fna", "pdb", "ent"}:
        return _fail(fmt, f"暂不支持 {fmt.upper()} 格式（当前支持 CSV/Excel/图片/压缩包）")
    if fmt == "xls":
        return _fail(fmt, "暂不支持旧版 .xls（请另存为 .xlsx）")
    if fmt == "unknown":
        return _fail(fmt, f"无法识别格式: {input_path.name}")

    if fmt == "archive":
        # 解压到产出目录下的 _files/：图片类数据集的统一 CSV 里 input 必须指向可持续访问的路径，
        # 若解压到临时目录，本次运行结束即被清理，CSV 中的图片路径会全部失效（真实数据集暴露的问题）。
        extract_dir = out_dir / "_files"
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(input_path, extract_dir)
        inner = sorted(p for p in extract_dir.rglob("*") if p.is_file())
        if not inner:
            return _fail(fmt, "压缩包内没有文件")
        # 递归一次：优先取表格，其次取图片/子目录
        table = next((p for p in inner if p.suffix.lower() in TABLE_EXT), None)
        if table is not None:
            rows, columns = _read_one(table)
            fmt = detect_format(table)
        else:
            img_files = [p for p in inner if p.suffix.lower() in IMAGE_EXT]
            if not img_files:
                return _fail(fmt, "压缩包内没有可识别的表格或图片")
            # 仅当压缩包只有一个顶层目录（常见的「多包一层」导出）才下钻，
            # 否则以解压根目录为根：train/、test/ 并列时只取第一个会把一半数据丢掉。
            tops = {p.relative_to(extract_dir).parts[0] for p in img_files}
            single_top = next(iter(tops)) if len(tops) == 1 else None
            img_root = extract_dir
            if single_top and (extract_dir / single_top).is_dir():
                img_root = extract_dir / single_top
            rows, columns = _read_image_dir(img_root)
            fmt = "image_dir"
    else:
        rows, columns = _read_one(input_path)

    if not rows:
        return _fail(fmt, "未读取到任何数据行")

    unified, field_mapping, label_merge = unify(rows, columns)
    unified_columns = list(UNIFIED_COLUMNS) + [c for c in unified[0] if c.startswith(META_PREFIX)]
    # 顺序：单位换算 → 序列规范 → 清洗（清洗在统一量纲后进行，异常值判定才准确）
    unified, unified_columns, units = normalize_units(unified, unified_columns)
    unified, sequence, seq_truncated = normalize_sequences(unified)
    cleaned, stats = clean(unified, unified_columns)
    if seq_truncated:
        stats["sequences_truncated"] = seq_truncated

    csv_path = _write_outputs(out_dir, cleaned, unified_columns)
    labels = sorted({r["label"] for r in cleaned if r.get("label")})
    summary = {
        "status": "ok",
        "format": fmt,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "n_rows": len(cleaned),
        "n_columns": len(unified_columns),
        "fields": unified_columns,
        "labels": labels,
        "cleaning": stats,
        "output_csv": str(csv_path),
    }
    summary["alignment"] = {
        "field_mapping": field_mapping,
        "label_merge": label_merge,
        "sequence": sequence,
        "units": units,
        "version": "1.0",
    }
    (out_dir / "dataset.schema.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="模块三 5.1 数据预处理")
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--dataset-name", default="")
    ap.add_argument("--task-type", default="classification")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(json.dumps({"status": "error", "message": f"输入不存在: {input_path}"}, ensure_ascii=False))
        return 1
    try:
        summary = run(input_path, Path(args.out_dir), args.dataset_name, args.task_type)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
