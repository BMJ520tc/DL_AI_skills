r"""`scripts/pdf_to_markdown.py` 索引口径修复的回归用例（公式识别、表注关联、图注兜底、页码继承）。

背景（本次修复的缺陷，均在 PDF→markdown 与索引这一层）：
1. 公式只认独占一行的 `$$`，同行 `$$ x = 1 $$` 不触发；实测 13 篇论文的 section_index.json
   里 equations 恒为 0，而需求二.1 明文要求「保留公式」。
2. tables[].caption 恒为 None；图注正则 `[\dIVXivx]+` 漏掉补充材料编号 `Figure S1`，
   esm3 正文 29 行 Figure 文本只索引到 4 条、scprint 0 条。
3. `--index-only`（agent 修正 markdown 后重建）无条件把 page/n_pages 置 null，
   「供前端定位」的能力实际失效。

本文件只用 tmp_path 造最小 markdown，不依赖真实 PDF，也不触碰仓库真实 data/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pdf_to_markdown as pp  # noqa: E402


def _build(tmp_path: Path, markdown: str, old_index: dict | None = None):
    """把 markdown 落盘并调用 `--index-only` 的重建入口；old_index 非空时先写到同一输出路径。"""
    md = tmp_path / "paper.md"
    md.write_text(markdown, encoding="utf-8")
    out = tmp_path / "section_index.json"
    if old_index is not None:
        out.write_text(json.dumps(old_index, ensure_ascii=False), encoding="utf-8")
    index = pp.build_index_from_markdown(str(md), str(out))
    return md, out, index


def _scan(lines: list[str]) -> tuple[list[str], dict]:
    """直接跑逐行分类，拿到改写后的 markdown 行与索引（页码均为 None）。"""
    index = pp._empty_index(None)
    md_lines = pp._scan_lines(list(lines), [None] * len(lines), index)
    return md_lines, index


# ---------- 1. 公式识别：同行 `$$ ... $$` 与多行 `$$` 块 ----------

def test_inline_paired_dollars_enter_equations(tmp_path):
    """同行成对 `$$ x = 1 $$` 必须进 equations（原实现只在 `in_math` 块内判定，此处会漏）。"""
    _, _, index = _build(tmp_path, "# 1 Introduction\n\nThe estimator is: \n\n$$ x = 1 $$\n")

    assert [e["text"] for e in index["equations"]] == ["x = 1"]
    assert index["equations"][0]["line"] == 5          # 行号 1 起，供前端定位
    assert index["equations"][0]["page"] is None       # 无旧索引可继承 → null，而不是编造


def test_multiline_dollar_block_enters_equations(tmp_path):
    """多行 `$$` 块内的公式行进 equations（含占比不足 0.25 的短式，块本身已是公式边界）。"""
    _, _, index = _build(tmp_path, "## 2 Method\n\n$$\nE = mc^2\nF = ma\n$$\n")

    assert [e["text"] for e in index["equations"]] == ["E = mc^2", "F = ma"]
    assert [e["line"] for e in index["equations"]] == [4, 5]


def test_narrative_text_is_not_an_equation(tmp_path):
    """叙述句里出现 `$`、短横线或成对 `$$` 但无数学内容时，不得误判成公式。"""
    markdown = (
        "# 1 Introduction\n\n"
        "The value $x$ is scaled by a 1e-4 factor and the batch size is 32.\n\n"
        "$$ this sentence has no math at all $$\n\n"
        "We report the baseline in Table 3 and the ablation in Fig. S6.\n"
    )
    _, _, index = _build(tmp_path, markdown)

    assert index["equations"] == []


# ---------- 2. 公式保真：确定性规范化 + 与正文区分 ----------

def test_equation_symbols_normalized_and_wrapped_in_markdown():
    """独立成段的公式行：Unicode 数学符号确定性转 ASCII，并包成 `$$` 块与正文区分。"""
    md_lines, index = _scan(["# 1 Method", "", "y=\u2212x\u00d72", "", "plain body text"])

    assert index["equations"] == [{"page": None, "line": 3, "text": "y=-x*2"}]
    assert md_lines[2] == "$$ y=-x*2 $$"                       # 建公式块
    assert md_lines[4] == "plain body text"                    # 正文行原样保留
    # 行数不变：1 输入行 → 1 输出行，保证行号与页码映射不错位
    assert len(md_lines) == 5


def test_list_item_math_is_indexed_but_not_restructured():
    """列表项里的公式只进索引、不改写行内容（改写会破坏 markdown 列表结构）。"""
    md_lines, index = _scan(["- 2: z = GeLU(z)"])

    assert [e["text"] for e in index["equations"]] == ["- 2: z = GeLU(z)"]
    assert md_lines[0] == "- 2: z = GeLU(z)"


def test_table_rows_are_not_treated_as_equations():
    """表格行（伪代码网格）不参与公式判定，避免污染 equations。"""
    md_lines, index = _scan(["|1: s=<br>36<br>nlayers|R|", "|2: x=x+s*Attn(x)|R|"])

    assert index["equations"] == []
    assert index["tables"] and len(index["tables"]) == 1        # 只记一次表头起始行
    assert md_lines[0].startswith("|1: s=")                     # 表格行原样保留


# ---------- 3. 表格标题关联 ----------

def test_table_caption_is_filled_from_nearby_table_text(tmp_path):
    """`Table 3 ...` 文本要填进同页相邻表格的 tables[].caption。"""
    markdown = (
        "## 3 Experiments\n\n"
        "Table 3 Hyperparameters used in training.\n\n"
        "| lr | 1e-4 |\n"
        "| bs | 32 |\n"
    )
    _, _, index = _build(tmp_path, markdown)

    assert len(index["tables"]) == 1
    assert index["tables"][0]["caption"] == "Table 3 Hyperparameters used in training."


def test_table_caption_stays_none_when_too_far(tmp_path):
    """附近没有 Table 标题的表格留 None（不硬塞一个不相干的标题）。"""
    markdown = "Table 3 Hyperparameters.\n\n" + "\n" * 60 + "| lr | 1e-4 |\n"
    _, _, index = _build(tmp_path, markdown)

    assert index["tables"][0]["caption"] is None


# ---------- 4. 图注兜底与去重 ----------

def test_caption_fallback_covers_supplementary_and_avoids_duplicates(tmp_path):
    """原正则漏掉的 `Figure S7` 也要进 captions；同一行重复只计一次；正文引用不误收。"""
    markdown = (
        "# 1 Results\n\n"
        "Figure 2 xxx\n\n"
        "Figure 2 xxx\n\n"                                  # 分页/重复行 → 去重
        "Figure S7 Supplementary detail.\n\n"
        "Fig. S6 indicates that pLDDT and pTM have good predictive power.\n\n"
        "\u8868 3 \u8bad\u7ec3\u8d85\u53c2\u6570\n"
    )
    _, _, index = _build(tmp_path, markdown)

    texts = [c["text"] for c in index["captions"]]
    assert texts.count("Figure 2 xxx") == 1                              # 不重复计数
    assert "Figure S7 Supplementary detail." in texts                    # 原正则漏检的补充材料编号
    assert [c["kind"] for c in index["captions"] if c["text"].startswith("Figure S7")] == ["figure"]
    assert all("indicates" not in t for t in texts)                      # 正文引用不是图注
    chinese = [c for c in index["captions"] if c["text"].startswith("\u8868 3")]
    assert len(chinese) == 1 and chinese[0]["kind"] == "table"           # 中文表注沿用 figure/table 规则


# ---------- 5. --index-only 页码继承 ----------

def test_index_only_inherits_page_and_n_pages_from_old_index(tmp_path):
    """重建索引时按标题/表格标题/图注文本/公式文本从旧索引继承 page 与 n_pages。"""
    markdown = (
        "# 1 Introduction\n\n"
        "Figure 1 Overview.\n\n"
        "Table 2 Data.\n\n"
        "| a | b |\n\n"
        "E = mc^2\n\n"
        "# 9 Appendix\n"
    )
    old = {
        "schema_version": "1.0",
        "engine": "pymupdf4llm",
        "n_pages": 7,
        "quality": "text",
        "sections": [
            {"level": 1, "title": "1 Introduction", "page": 1, "line": 3},
        ],
        "tables": [{"page": 2, "index": 0, "caption": "Table 2 Data.", "line": 4}],
        "equations": [{"page": 3, "line": 50, "text": "E = mc^2"}],
        "captions": [
            {"kind": "figure", "page": 1, "line": 60, "text": "Figure 1 Overview."},
            {"kind": "table", "page": 2, "line": 61, "text": "Table 2 Data."},
        ],
    }
    _, out, index = _build(tmp_path, markdown, old_index=old)

    assert index["n_pages"] == 7                                   # 分页信息不再丢
    assert [s["page"] for s in index["sections"]] == [1, None]     # 匹配不到的新章节才 null
    assert index["captions"][0]["page"] == 1
    assert index["tables"][0]["caption"] == "Table 2 Data."
    assert index["tables"][0]["page"] == 2
    assert index["equations"][0]["page"] == 3
    # 继承结果确实写回磁盘（paper_service 读的是这个文件）
    assert json.loads(out.read_text(encoding="utf-8"))["n_pages"] == 7


def test_index_only_without_old_index_keeps_null_pages(tmp_path):
    """没有旧索引可继承时，page/n_pages 仍为 null（不编造分页）。"""
    _, _, index = _build(tmp_path, "# 1 Introduction\n\nE = mc^2\n")

    assert index["n_pages"] is None
    assert index["equations"][0]["page"] is None


@pytest.mark.parametrize("broken", ["{ not json", "[]"])
def test_index_only_ignores_unreadable_old_index(tmp_path, broken):
    """旧索引损坏或不是对象时按「无旧索引」处理，不抛异常。"""
    md = tmp_path / "paper.md"
    md.write_text("# 1 Introduction\n\nE = mc^2\n", encoding="utf-8")
    out = tmp_path / "section_index.json"
    out.write_text(broken, encoding="utf-8")

    index = pp.build_index_from_markdown(str(md), str(out))

    assert index["n_pages"] is None
    assert len(index["equations"]) == 1
