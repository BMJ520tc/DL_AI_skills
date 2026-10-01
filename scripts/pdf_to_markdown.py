"""模块二 4.1 PDF 转 markdown 规则工具（模块详细设计 4.1，D5）。

D5 定案：规则工具为主、agent 修正为辅，不纯靠大模型保证文本保真。
规则部分用 PyMuPDF 官方 markdown 转换器 pymupdf4llm（标题层级、表格、列表、
多栏阅读顺序均在库内处理），再从其逐页 markdown 产出 section_index；表格/公式/
图注的最终修正由 paper_service 的 agent 步骤完成。

签名（由 paper_service 以固定代码调用）：

    python pdf_to_markdown.py <pdf_path> <out_markdown> <out_section_index> [out_tables]
    python pdf_to_markdown.py --index-only <markdown> <out_section_index>

out_tables 为可选的 pdfplumber 原始表格网格（tables.json），供抽取阶段优先取报告值；
--index-only 供 agent 修正 markdown 后重建 section_index（此时无分页信息，page 置 null）。

输出 section_index（数据设计四.3 section_index 字段）：

    {
      "schema_version": "1.0",
      "engine": "pymupdf4llm",
      "n_pages": 12,                            # --index-only 时为 null
      "quality": "text" | "scanned",            # 扫描版无文本层 → scanned，提示 OCR
      "sections":  [{"level":1,"title":"1 Introduction","page":1,"line":3}],
      "tables":    [{"page":2,"index":0,"caption":"Table 1 ...","line":40}],
      "equations": [{"page":2,"line":50,"text":"E = mc^2"}],
      "captions":  [{"kind":"figure","page":3,"line":60,"text":"Figure 1 ..."}]
    }

line 为 markdown 中的行号（1 起），供 4.2 抽取与前端定位。
"""
import json
import re
import sys
from pathlib import Path

import pdfplumber
import pymupdf4llm

_CAPTION_RE = re.compile(r"^\s*\**(Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?)\s*[\dIVXivx]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MATH_CHARS = set("=+-^_/\\{}[]()<>∫∑√≈≤≥×·±αβγθλμσΩπΔ∂∈∀∃→←↔|")


def _is_equation(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 200 or not any(ch.isalpha() for ch in stripped):
        return False
    ratio = sum(ch in _MATH_CHARS for ch in stripped) / len(stripped)
    return ratio >= 0.25 and "=" in stripped


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
    """逐行分类写入 index；page_of[i] 为该行所在页码（未知为 None）。返回剔除首尾空行的行。"""
    md_lines: list[str] = []
    table_seq = 0
    in_math = False
    for raw in lines:
        line_no = len(md_lines) + 1
        page = page_of[line_no - 1] if line_no - 1 < len(page_of) else None
        line = raw.rstrip()
        heading = _HEADING_RE.match(line)
        caption = _CAPTION_RE.match(line)
        if heading:
            index["sections"].append(
                {"level": len(heading.group(1)), "title": heading.group(2), "page": page, "line": line_no}
            )
        elif line.strip() == "$$":
            in_math = not in_math
        elif in_math and line.strip():
            if _is_equation(line):
                index["equations"].append({"page": page, "line": line_no, "text": line.strip()})
        elif caption:
            kind = "figure" if caption.group(1).lower().startswith("fig") else "table"
            index["captions"].append(
                {"kind": kind, "page": page, "line": line_no, "text": line.strip("* ").strip()}
            )
        elif _TABLE_ROW_RE.match(line) and not (md_lines and _TABLE_ROW_RE.match(md_lines[-1])):
            index["tables"].append({"page": page, "index": table_seq, "caption": None, "line": line_no})
            table_seq += 1
        md_lines.append(line)
    return md_lines


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

    # 扫描版判定：文本层过稀 → 无文本层（预留 OCR，4.1 异常与边界）
    if index["n_pages"] and total_chars / index["n_pages"] < 40:
        index["quality"] = "scanned"

    Path(out_md).write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    Path(out_idx).write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_tables:
        extract_tables(pdf_path, out_tables)
    return index


def build_index_from_markdown(md_path: str, out_idx: str) -> dict:
    """agent 修正 markdown 后重建 section_index（无分页信息，page 置 null）。"""
    lines = Path(md_path).read_text(encoding="utf-8").split("\n")
    index = _empty_index(None)
    _scan_lines(lines, [None] * len(lines), index)
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
