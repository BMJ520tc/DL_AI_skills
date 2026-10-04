"""模块二 4.1 PDF 转 markdown 规则工具（模块详细设计 4.1，D5）。

D5 定案：规则工具为主、agent 修正为辅，不纯靠大模型保证文本保真。
规则部分用 PyMuPDF 官方 markdown 转换器 pymupdf4llm（标题层级、表格、列表、
多栏阅读顺序均在库内处理），再从其逐页 markdown 产出 section_index；表格/公式/
图注的最终修正由 paper_service 的 agent 步骤完成。

签名（由 paper_service 以固定代码调用）：

    python pdf_to_markdown.py <pdf_path> <out_markdown> <out_section_index> [out_tables]
    python pdf_to_markdown.py --index-only <markdown> <out_section_index>

out_tables 为可选的 pdfplumber 原始表格网格（tables.json），供抽取阶段优先取报告值。

--index-only 供 agent 修正 markdown 后重建 section_index。markdown 里没有分页信息，
但旧索引（就是被覆盖的那个 section_index.json）里有：重建时按文本匹配继承 page 与 n_pages
（规则见 `_inherit_pages`），匹配不到才置 null——避免「修正一次 markdown 就把前端定位能力弄丢」。

输出 section_index（数据设计四.3 section_index 字段）：

    {
      "schema_version": "1.0",
      "engine": "pymupdf4llm",
      "n_pages": 12,
      "quality": "text" | "scanned",            # 扫描版无文本层 → scanned，提示 OCR
      "sections":  [{"level":1,"title":"1 Introduction","page":1,"line":3}],
      "tables":    [{"page":2,"index":0,"caption":"Table 1 ...","line":40}],
      "equations": [{"page":2,"line":50,"text":"E = mc^2"}],
      "captions":  [{"kind":"figure","page":3,"line":60,"text":"Figure 1 ..."}]
    }

line 为 markdown 中的行号（1 起），供 4.2 抽取与前端定位；`_scan_lines` 保证 1 输入行 → 1 输出行，
因此行号与 pdf 页映射一一对应。

公式口径（`_is_equation`，本次修复）：
- 长度上限 200 字符、必须至少含一个字母（`str.isalpha`）、数学字符占比 = 命中 `_MATH_CHARS`
  的字符数 / 字符总数——这三项沿用原口径，未改集合、未改阈值基数；
- 未包裹的整行：占比 >= 0.25 且含 `=` → 公式（原口径，保持不变）；
- 被成对 `$$` 包裹的整行（行首 `$$` … 行尾 `$$`）：占比 >= 0.25，或含 `=` 且占比 >= 0.10
  → 公式。放宽只为让 `$$ x = 1 $$` 这类短表达式进索引；仍强制数学字符占比，叙述句
  （如 `$$ this sentence has no math $$`）占比为 0，不会误判；
- `$$` 块（独占一行的 `$$` 之间）内的非空行用 `_is_math_content` 判定（比严格口径更宽：块本身已是
  显式公式边界，故 `F = ma` 这类占比不足 0.25 的短式也进索引；但叙述句占比为 0，仍被挡掉）；
- 表格行（`|...|`）一律不参与公式判定：它们是表格内容，已由 tables 索引承载，误并入会污染 equations。

公式保真（确定性，不联网、不调大模型）：
- `$$` 包裹规范化：成对包裹的表达式在索引里统一记 `$$` 之间的内容；
- 常见 Unicode 数学符号按 `_MATH_SYMBOL_MAP` 转为语义等价的 ASCII 可读写法
  （如 `−`→`-`、`×`→`*`、`≤`→`<=`、`∑`→`sum`），**未列入映射表的字符原样保留**（不丢符号）；
- 独立成段的公式行在 markdown 里包成 `$$ ... $$` 公式块，与普通正文行区分，便于后续检索；
  列表项 / 引用块 / 表格行内的公式只进索引、不改写行内容，避免破坏 markdown 结构。

图注口径（`_match_caption`）：
- 标签 `Figure|Fig.|Table|Tab.|Equation|Eq.|图|表`；`kind` 沿用既有 figure/table 二分规则
  （`fig` 开头或中文「图」→ figure，其余 → table，故 Equation 仍归 table）；
- 严格正则保持原样（编号为罗马数字或阿拉伯数字），另加兜底正则覆盖原正则漏掉的形态：
  补充材料编号 `S1`/`Figure 2C`、大写标签、`Supplementary|Supplemental|Suppl.|Extended Data|补充` 前缀；
- 兜底只看「编号之后剩余文本像不像图注」：为空、或以标点/大写字母/数字/左括号/中文开头才算，
  于是正文引用 `Fig. S6 indicates that ...`（小写动词开头）不会被误收；
- 图注按 (kind, 归一化文本) 去重，PDF 分页处重复出现的同一条图注只计一次。

表格标题关联（`_attach_table_captions`）：tables[].caption 取「同页、行距最近且未被占用」的
`Table N ...` 图注文本（任一侧缺页码时退化为纯行距匹配），行距超过 30 行则留 None。
"""
import json
import re
import sys
from pathlib import Path

import pdfplumber
import pymupdf4llm

_MAX_EQ_LEN = 200
_MATH_RATIO_STRICT = 0.25
_MATH_RATIO_WRAPPED = 0.10
_TABLE_CAPTION_MAX_GAP = 30
_PREFIX_MATCH_MIN_LEN = 8

_CAPTION_RE = re.compile(r"^\s*\**(Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?|图|表)\s*[\dIVXivx]+")
# 兜底：原正则漏掉的补充材料编号（S1/S10）、大写标签、以及 Supplementary/Extended Data/补充 前缀
_CAPTION_LOOSE_RE = re.compile(
    r"^\s*[\*_>\s]*(?:(?:Extended\s+Data|Supplementary|Supplemental|Suppl\.?|补充)\s*)?"
    r"(Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?|图|表)\s*(S\d+[A-Za-z]?|\d+[A-Za-z]?|[IVXivx]+)",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MATH_FENCE_RE = re.compile(r"^\s*\$\$\s*$")
_LIST_OR_QUOTE_RE = re.compile(r"^\s*(?:[-*+>]\s|\d+[.)]\s)")
_WS_RE = re.compile(r"\s+")
_MATH_CHARS = set("=+-^_/\\{}[]()<>∫∑√≈≤≥×·±αβγθλμσΩπΔ∂∈∀∃→←↔|")
# 图注编号之后的剩余文本：以这些字符开头才算「像图注」
_CAPTION_TAIL_PUNCT = ".。:：,，;；|·-–—([])}"
# 常见 Unicode 数学符号 → 语义等价的 ASCII 可读写法；未列入的字符原样保留
_MATH_SYMBOL_MAP = {
    "\u2212": "-",   # − 减号
    "\u2010": "-",   # ‐
    "\u2011": "-",   # ‑
    "\u2012": "-",   # ‒
    "\u00d7": "*",   # × 乘号
    "\u22c5": "*",   # ⋅ 点乘
    "\u00b7": "*",   # · 间隔号（公式内按点乘处理）
    "\u00f7": "/",   # ÷
    "\u00b1": "+-",  # ±
    "\u2264": "<=",  # ≤
    "\u2265": ">=",  # ≥
    "\u2260": "!=",  # ≠
    "\u2248": "~=",  # ≈
    "\u2261": "==",  # ≡
    "\u221e": "inf",  # ∞
    "\u2192": "->",  # →
    "\u2190": "<-",  # ←
    "\u2194": "<->",  # ↔
    "\u2211": "sum",  # ∑
    "\u220f": "prod",  # ∏
    "\u222b": "int",  # ∫
    "\u221a": "sqrt",  # √
    "\u2202": "d",   # ∂
    "\u2206": "Delta",  # ∆
}


def _math_ratio(text: str) -> float:
    return sum(ch in _MATH_CHARS for ch in text) / len(text) if text else 0.0


def _unwrap_inline_math(text: str) -> tuple[str, bool]:
    """整行被成对 `$$` 包裹（行首 `$$` + 行尾 `$$`）时返回内层表达式与 True。"""
    if len(text) > 4 and text.startswith("$$") and text.endswith("$$"):
        return text[2:-2].strip(), True
    return text, False


def _map_math_symbols(text: str) -> str:
    return "".join(_MATH_SYMBOL_MAP.get(ch, ch) for ch in text)


def _normalize_math_text(text: str) -> str:
    """公式文本归一化：去 `$$` 包裹、常见 Unicode 数学符号 → ASCII 可读写法、空白折叠。"""
    core, _ = _unwrap_inline_math(text.strip())
    return _WS_RE.sub(" ", _map_math_symbols(core)).strip()


def _is_math_content(text: str) -> bool:
    """放宽口径：用于「已被 `$$` 明确界定」的内容（成对 `$$` 整行、`$$` 块内的行）。

    仍保留长度上限、字母要求与「等号 或 数学字符占比」约束，只是把严格口径的占比阈值
    从 0.25 放宽到 0.10（有等号时），或纯占比 0.25；叙述句占比为 0，不会被放进来。
    """
    stripped = text.strip()
    if not stripped or len(stripped) > _MAX_EQ_LEN:
        return False
    core, _ = _unwrap_inline_math(stripped)
    if not core or not any(ch.isalpha() for ch in core):
        return False
    ratio = _math_ratio(core)
    return ratio >= _MATH_RATIO_STRICT or ("=" in core and ratio >= _MATH_RATIO_WRAPPED)


def _is_equation(text: str) -> bool:
    """判定一行是否为公式；阈值口径见模块 docstring（未包裹走严格口径，包裹走放宽口径）。"""
    stripped = text.strip()
    if not stripped or len(stripped) > _MAX_EQ_LEN:
        return False
    _, wrapped = _unwrap_inline_math(stripped)
    if wrapped:
        return _is_math_content(stripped)
    if not any(ch.isalpha() for ch in stripped):
        return False
    return _math_ratio(stripped) >= _MATH_RATIO_STRICT and "=" in stripped


def _is_indexable_equation(line: str) -> bool:
    """可进 equations 索引的行：非表格行，且 `_is_equation` 判定为公式。"""
    if _TABLE_ROW_RE.match(line):
        return False
    return _is_equation(line)


def _is_standalone_equation(line: str) -> bool:
    """独立成段的公式行（可安全包成 `$$` 块）：非表格行 / 列表项 / 引用块 / 已有 `$$` 包裹。"""
    if _TABLE_ROW_RE.match(line) or _LIST_OR_QUOTE_RE.match(line) or _MATH_FENCE_RE.match(line):
        return False
    if line.strip().startswith("$$"):
        return False
    return _is_equation(line)


def _caption_kind(label: str) -> str:
    return "figure" if label.lower().startswith("fig") or label == "图" else "table"


def _looks_like_caption_tail(rest: str) -> bool:
    """编号之后的剩余文本是否「像图注」（用于兜底正则，挡掉正文里的小写动词引用）。"""
    rest = rest.strip()
    if not rest:
        return True
    first = rest[0]
    if first in _CAPTION_TAIL_PUNCT or first.isupper() or first.isdigit():
        return True
    return "\u4e00" <= first <= "\u9fff"


def _match_caption(line: str) -> tuple[str, str] | None:
    """匹配图注行；返回 (kind, text)，不匹配返回 None。"""
    m = _CAPTION_RE.match(line)
    if m:
        return _caption_kind(m.group(1)), line.strip("* ").strip()
    m = _CAPTION_LOOSE_RE.match(line)
    if m and _looks_like_caption_tail(line[m.end():]):
        return _caption_kind(m.group(1)), line.strip("* ").strip()
    return None


def _norm_key(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip().casefold()


def _empty_index(n_pages) -> dict:
    return {
        "schema_version": "1.0",
        "engine": "pymupdf4llm",
        "n_pages": n_pages,
        "quality": "text",
        "sections": [],
        "tables": [],
        "equations": [],
        "captions": [],
    }


def _scan_lines(lines: list[str], page_of: list, index: dict) -> list[str]:
    """逐行分类写入 index；page_of[i] 为该行所在页码（未知为 None）。返回处理后的行。

    1 输入行 → 1 输出行，行号与 page_of 严格对齐（公式行在原位改写成 `$$ ... $$`，不增删行）。
    """
    md_lines: list[str] = []
    table_seq = 0
    in_math = False
    caption_keys: set[tuple[str, str]] = set()
    for raw in lines:
        line_no = len(md_lines) + 1
        page = page_of[line_no - 1] if line_no - 1 < len(page_of) else None
        line = raw.rstrip()
        out = line
        heading = _HEADING_RE.match(line)
        caption = _match_caption(line)
        if heading:
            index["sections"].append(
                {"level": len(heading.group(1)), "title": heading.group(2), "page": page, "line": line_no}
            )
        elif _MATH_FENCE_RE.match(line):
            in_math = not in_math
        elif in_math:
            if line.strip():
                if _is_math_content(line):
                    index["equations"].append(
                        {"page": page, "line": line_no, "text": _normalize_math_text(line)}
                    )
                out = _map_math_symbols(line)
        elif caption is not None:
            kind, text = caption
            key = (kind, _norm_key(text))
            if key not in caption_keys:
                caption_keys.add(key)
                index["captions"].append({"kind": kind, "page": page, "line": line_no, "text": text})
        elif _TABLE_ROW_RE.match(line) and not (md_lines and _TABLE_ROW_RE.match(md_lines[-1])):
            index["tables"].append({"page": page, "index": table_seq, "caption": None, "line": line_no})
            table_seq += 1
        elif _is_indexable_equation(line):
            text = _normalize_math_text(line)
            index["equations"].append({"page": page, "line": line_no, "text": text})
            if _is_standalone_equation(line):
                out = f"$$ {text} $$"
        md_lines.append(out)
    return md_lines


def _attach_table_captions(index: dict, max_gap: int = _TABLE_CAPTION_MAX_GAP) -> None:
    """用同页、行距最近且未被占用的 `Table N ...` 图注填充 tables[].caption；取不到留 None。"""
    candidates = [c for c in index.get("captions", []) if c.get("kind") == "table"]
    used: set[int] = set()
    for table in index.get("tables", []):
        table["caption"] = None
        best_idx = None
        best_key = None
        for i, cap in enumerate(candidates):
            if i in used:
                continue
            if table.get("page") is not None and cap.get("page") is not None:
                if table["page"] != cap["page"]:
                    continue
            gap = abs((table.get("line") or 0) - (cap.get("line") or 0))
            if gap > max_gap:
                continue
            key = (gap, cap.get("line") or 0)
            if best_key is None or key < best_key:
                best_key, best_idx = key, i
        if best_idx is not None:
            used.add(best_idx)
            table["caption"] = candidates[best_idx]["text"]


class _PagePool:
    """旧索引的「文本 → 页码」池；take() 取一个尚未被占用的页码。"""

    def __init__(self, entries: list[dict], text_of) -> None:
        self._by_key: dict[str, list] = {}
        for e in entries:
            page = e.get("page")
            key = _norm_key(text_of(e))
            if page is None or not key:
                continue
            self._by_key.setdefault(key, []).append(page)

    def take(self, text: str):
        key = _norm_key(text)
        if not key:
            return None
        pages = self._by_key.get(key)
        if pages:
            return pages.pop(0)
        # 前缀兜底：agent 修正可能截断/补全标题，取长度最接近的唯一候选
        if len(key) < _PREFIX_MATCH_MIN_LEN:
            return None
        best = None
        for old_key, old_pages in self._by_key.items():
            if not old_pages or len(old_key) < _PREFIX_MATCH_MIN_LEN:
                continue
            if not (old_key.startswith(key) or key.startswith(old_key)):
                continue
            diff = abs(len(old_key) - len(key))
            if best is None or diff < best[0]:
                best = (diff, old_pages)
        return best[1].pop(0) if best is not None else None


def _inherit_pages(new_index: dict, old_index: dict | None) -> dict:
    """--index-only 重建时继承页码，匹配不到才留 null。

    继承规则（按此顺序，逐条取第一个命中的页码；同一旧条目只被认领一次）：
    1. n_pages / quality：新索引缺省时直接沿用旧索引（quality 无法从 markdown 重算）；
    2. sections：按标题文本（空白折叠 + 大小写折叠）精确匹配，其次「一方是另一方前缀」的前缀匹配
       （长度 >= 8，取长度最接近的候选）；
    3. tables：先按 caption 文本匹配旧索引里带 caption 的表；再在新旧 tables 数量相等时按序位兜底；
    4. captions：按图注文本匹配，数量相等时再按序位兜底；
    5. equations：按公式文本匹配。
    """
    if not isinstance(old_index, dict):
        return new_index
    if new_index.get("n_pages") is None and old_index.get("n_pages") is not None:
        new_index["n_pages"] = old_index["n_pages"]
    if old_index.get("quality"):
        new_index["quality"] = old_index["quality"]

    sections = _PagePool(old_index.get("sections") or [], lambda e: e.get("title", ""))
    for sec in new_index.get("sections", []):
        if sec.get("page") is None:
            sec["page"] = sections.take(sec.get("title", ""))

    old_tables = old_index.get("tables") or []
    table_pool = _PagePool(old_tables, lambda e: e.get("caption") or "")
    for table in new_index.get("tables", []):
        if table.get("page") is None and table.get("caption"):
            table["page"] = table_pool.take(table["caption"])
    if len(old_tables) == len(new_index.get("tables", [])):
        for table, old in zip(new_index.get("tables", []), old_tables):
            if table.get("page") is None and old.get("page") is not None:
                table["page"] = old["page"]

    old_captions = old_index.get("captions") or []
    caption_pool = _PagePool(old_captions, lambda e: e.get("text", ""))
    for cap in new_index.get("captions", []):
        if cap.get("page") is None:
            cap["page"] = caption_pool.take(cap.get("text", ""))
    if len(old_captions) == len(new_index.get("captions", [])):
        for cap, old in zip(new_index.get("captions", []), old_captions):
            if cap.get("page") is None and old.get("page") is not None:
                cap["page"] = old["page"]

    equation_pool = _PagePool(old_index.get("equations") or [], lambda e: e.get("text", ""))
    for eq in new_index.get("equations", []):
        if eq.get("page") is None:
            eq["page"] = equation_pool.take(eq.get("text", ""))
    return new_index


def _read_index(path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def extract_tables(pdf_path: str, out_tables: str) -> dict:
    """用 pdfplumber 确定性抓表格二维网格（D5 规则工具；避免 markdown 单元格合并/列错位）。

    输出 tables.json：每张表含 page / index / rows（行×列的文本网格）。
    抽取阶段优先以此为准取「论文报告值」，markdown 表格仅作可读性参考。
    """
    # 学术表多为无框线：lines 策略常抓不到，退回 text 策略（代价是相邻表可能被并到一张）
    text_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            grids = page.extract_tables() or []
            if not grids:
                grids = page.extract_tables(table_settings=text_settings) or []
            for idx, grid in enumerate(grids):
                rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in grid]
                if rows:
                    tables.append({"page": page_no, "index": idx, "rows": rows})
    doc = {"schema_version": "1.0", "engine": "pdfplumber", "tables": tables}
    Path(out_tables).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def convert(pdf_path: str, out_md: str, out_idx: str, out_tables: str | None = None) -> dict:
    image_dir = Path(out_md).parent / "images"
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, write_images=True, image_path=str(image_dir))

    all_lines: list[str] = []
    page_of: list = []
    total_chars = 0
    for page_no, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "") or ""
        total_chars += len(text.strip())
        if page_no > 1:
            all_lines.append("")
            page_of.append(page_no)
        for line in text.split("\n"):
            all_lines.append(line)
            page_of.append(page_no)

    index = _empty_index(len(chunks))
    md_lines = _scan_lines(all_lines, page_of, index)
    _attach_table_captions(index)

    # 扫描版判定：文本层过稀 → 无文本层（预留 OCR，4.1 异常与边界）
    if index["n_pages"] and total_chars / index["n_pages"] < 40:
        index["quality"] = "scanned"

    Path(out_md).write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    Path(out_idx).write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_tables:
        extract_tables(pdf_path, out_tables)
    return index


def build_index_from_markdown(md_path: str, out_idx: str) -> dict:
    """agent 修正 markdown 后重建 section_index（页码从旧索引继承，匹配不到才置 null）。"""
    lines = Path(md_path).read_text(encoding="utf-8").split("\n")
    index = _empty_index(None)
    _scan_lines(lines, [None] * len(lines), index)
    _attach_table_captions(index)  # 先按行距关联，供继承时按 caption 文本匹配旧表
    _inherit_pages(index, _read_index(out_idx))
    _attach_table_captions(index)  # 继承到页码后再按「同页」重排一次
    Path(out_idx).write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) == 3 and argv[0] == "--index-only":
        build_index_from_markdown(argv[1], argv[2])
        return
    if len(argv) == 3:
        convert(argv[0], argv[1], argv[2])
        return
    if len(argv) == 4:
        convert(argv[0], argv[1], argv[2], argv[3])
        return
    print(
        "usage: pdf_to_markdown.py <pdf> <out_md> <out_idx> [out_tables] | --index-only <md> <out_idx>",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
