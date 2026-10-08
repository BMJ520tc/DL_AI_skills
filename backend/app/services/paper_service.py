"""模块二：论文自动复现与可信度确定（模块详细设计四章 4.1~4.4，D5/D6）。

四个任务：
- pdf_parse   4.1 PDF → markdown（pymupdf4llm 规则为主 + agent 对照 PDF 修正表格/公式/图注）
                   + 固定代码保真核对（PDF ↔ markdown 逐页覆盖率，产物 fidelity_report.json）
- extract_items 4.2 实验条目抽取（agent，六要素缺失标注而非臆造；按章节/结果表定位取样）
- reproduce   4.3 自动复现（agent 生成复现脚本 + 项目独立环境逐条执行 + run_record）
- conclusion  4.4 逐条对照与可信度结论（固定代码分档 + agent 起草总结）

与模块三同口径：指标名经 `normalize_metric_name` 归一到 backend/app/contracts.py::CANONICAL_METRICS
（复用同一张别名表，使 4.4 对照与 5.x 跨数据集对比的指标键名一致）。
"""
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app import contracts
from app.config import PAPERS_DIR, resource_path, script_command
from app.ids import fs_name, safe_id
from app.services import (
    agent_service, analysis_service, env_manager, knowledge_service, proc_util,
    project_manager, task_manager,
)

TASK_PARSE = "pdf_parse"
TASK_EXTRACT = "extract_items"
TASK_REPRODUCE = "reproduce"
TASK_CONCLUSION = "conclusion"

PDF_SCRIPT = resource_path("scripts/pdf_to_markdown.py")
REPRODUCE_TEMPLATE = resource_path("scripts/reproduce_template.py")
REPRODUCE_TIMEOUT_S = 3600
REPRODUCE_MAX_PARALLEL = int(os.getenv("REPRODUCE_MAX_PARALLEL", "3"))  # 4.3 多条目在资源限额内并行
# 4.2 抽取喂给 agent 的**总预算**：不再是「只取 markdown 头部」，超预算时按
# 方法/实验章节 → 结果表格片段 → 摘要 的优先级取样（见 select_extract_context）。
MAX_EXTRACT_CHARS = 100_000
# 总预算内的取样配额：这三项之和即总预算，未用完的余量按原文顺序回填其余章节。
EXTRACT_SECTION_SHARE = float(os.getenv("EXTRACT_SECTION_SHARE", "0.55"))
EXTRACT_TABLE_SHARE = float(os.getenv("EXTRACT_TABLE_SHARE", "0.35"))

# 4.1 保真核对阈值（O3：先按设计取值，实施期用真实论文校准）
# 保真核对：从 PDF 每页取「指纹」片段（≥ 该长度的行，滤掉页边行号/页眉），
# 在归一化后的 markdown 里找命中 → 判该页内容有没有进 markdown。
_FIDELITY_MIN_PROBE_CHARS = int(os.getenv("FIDELITY_MIN_PROBE_CHARS", "40"))
FIDELITY_MIN_COVERAGE = float(os.getenv("FIDELITY_MIN_COVERAGE", "0.30"))

# D6 可信度分档阈值（O3：先按设计取值，实施期用真实复现数据校准）——按相对误差
DEVIATION_CONSISTENT = float(os.getenv("REPRO_DEVIATION_CONSISTENT", "0.05"))
DEVIATION_APPROX = float(os.getenv("REPRO_DEVIATION_APPROX", "0.20"))

# 四档结论（与《知识库与数据设计》四.3 verdict 字段一致）
VERDICT_CONSISTENT = "一致"
VERDICT_APPROX = "近似"
VERDICT_INCONSISTENT = "不一致"
VERDICT_UNREPRODUCIBLE = "无法复现"

ITEMS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_ref": {"type": "string", "description": "来源章节"},
                    "dataset_name": {"type": "string"},
                    "split_method": {"type": "string", "description": "划分方式"},
                    "metric_name": {"type": "string", "description": "评价指标名"},
                    "metric_value_reported": {"type": "string", "description": "论文报告值"},
                    "metric_unit": {"type": "string"},
                    "hyperparams": {"type": "object"},
                    "baselines": {"type": "array"},
                },
                "required": ["metric_name"],
            },
        }
    },
    "required": ["items"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_verdict": {
            "type": "string",
            "enum": [VERDICT_CONSISTENT, VERDICT_APPROX, VERDICT_INCONSISTENT, VERDICT_UNREPRODUCIBLE],
        },
        "summary": {"type": "string", "description": "总体可信度结论文本"},
    },
    "required": ["summary"],
}

_VERDICT_RANK = {VERDICT_CONSISTENT: 0, VERDICT_APPROX: 1, VERDICT_INCONSISTENT: 2, VERDICT_UNREPRODUCIBLE: 3}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paper_dir(paper_id: str) -> Path:
    """论文产物目录：`data/papers/<fs_name(paper_id)>/`。

    目录名一律经 `ids.fs_name`（与 `download_service.download_paper` 落盘时同一个函数）：
    arXiv 号 / `pubmed:456` 这类既有 id 的目录名不变，真实 DOI（`10.1101/2023.10.03.560734`，
    bioRxiv/medRxiv 检索返回的 paper_id）落成单层的 `10.1101%2F2023.10.03.560734`——
    写盘与读回同一映射，才不会出现「下载到 A 目录、解析去 B 目录找」。
    """
    d = PAPERS_DIR / fs_name(paper_id, "paper_id")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_paper(paper_id: str) -> dict:
    paper = knowledge_service.get_item("paper", paper_id)
    if paper is None:
        raise LookupError(f"paper not found: {paper_id}")
    return paper


_NUM_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
# 约数前缀：「约 0.85」「~0.85」「≈0.85」「about 0.85」——数值本身仍是确定的，
# 只是作者表明近似。剥掉前缀后**其余部分仍必须是纯数值**，不破坏下面的严格口径。
_APPROX_PREFIX_RE = re.compile(
    r"^\s*(?:约\s*为|约|大约|approximately|approx\.?|about|around|circa|~=|~|≈|∼)\s*",
    re.IGNORECASE,
)
_UNIT_SUFFIXES = ("个百分点", "%p", "pp", "%", "‰", "‱")


def _to_float(value) -> Optional[float]:
    """把报告值/实测值转成浮点。

    只接受真正的数值或「数值+单位符号」（`91.5%`、`1,234`、`9.15e-3`），以及带**约数前缀**的
    数值（`~0.85`、`≈0.85`、`约 0.85`——剥掉前缀后仍是纯数值）；
    含不等式、区间或文字的写法（`>50% improvement`、`85-90`、`未报告`）一律返回 None，
    由 `_classify` 判为「无法复现」——不从自由文本里抠数字，避免把「提升 50%」当成指标值。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace("％", "%").replace("，", ",").replace(",", "")
    if not text:
        return None
    text = _APPROX_PREFIX_RE.sub("", text)
    for suffix in _UNIT_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    if not _NUM_RE.match(text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_metric_name(name) -> Optional[str]:
    """指标名归一（4.2/4.4 与模块三同口径）：复用 contracts 的别名表与规范指标集。

    `CANONICAL_METRICS` 是「基准指标与模块二复现指标保持可比」的落地点，
    抽取到的 `metric_name`（如 `Acc`、`Macro-F1`、`ROC-AUC`）统一归一到该口径后再落库、
    再参与对照，避免同一指标在两份结果里异名导致 5.4 取交集为空而误报「不可比」。
    返回 None 表示指标名为空；未收录的指标名按 contracts 的同一规则小写规范化后原样保留。
    """
    if name is None:
        return None
    text = str(name).strip()
    return contracts._canonical(text) if text else None


def _is_canonical_metric(name: Optional[str]) -> bool:
    return bool(name) and name in contracts.CANONICAL_METRICS


def _md_hash(md_path: Path) -> str:
    """markdown 内容指纹（4.1：只有 agent 真的改了 markdown 才重建索引）。"""
    try:
        return hashlib.sha256(Path(md_path).read_bytes()).hexdigest()
    except OSError:
        return ""


_MAX_TABLES_CHARS = 40_000


def _render_tables(path: Path) -> str:
    """把 pdfplumber 抽取的表格网格渲染为文本，供抽取阶段核对数字（超限截断）。"""
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    blocks, used = [], 0
    for t in data.get("tables", []):
        block = f"[第{t['page']}页 表{t['index']}]\n" + "\n".join(" | ".join(row) for row in t.get("rows", []))
        if used + len(block) > _MAX_TABLES_CHARS:
            blocks.append("...（其余表格因体积省略）")
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _dedup_items(items: list[dict]) -> list[dict]:
    """保守去重：四要素（指标/报告值/数据集/划分）完全相同的条目视为重复。"""
    seen, out = set(), []
    for it in items:
        key = tuple(str(it.get(k) or "").strip().lower() for k in
                    ("metric_name", "metric_value_reported", "dataset_name", "split_method"))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


_ITEM_LIST_KEYS = ("items", "entries", "experiment_items", "results", "data")


def _as_items(structured) -> list[dict]:
    """规整 agent 抽取输出。

    DeepSeek 下结构化输出走文件兜底、不经 schema 校验，键名/形状可能漂移：
    兼容裸数组、{"items"/"entries"/…:[...]}、以及单条目对象。
    """
    if isinstance(structured, list):
        return [it for it in structured if isinstance(it, dict)]
    if isinstance(structured, dict):
        for key in _ITEM_LIST_KEYS:
            val = structured.get(key)
            if isinstance(val, list):
                return [it for it in val if isinstance(it, dict)]
        if any(k in structured for k in ("metric_name", "dataset_name")):
            return [structured]
        for val in structured.values():
            if isinstance(val, list):
                return [it for it in val if isinstance(it, dict)]
    return []


# ------------------- 4.1 保真核对：PDF ↔ markdown（固定代码，不依赖大模型） -------------------

_FIDELITY_TABLE_RE = re.compile(r"\bTable\s*[\dIVXivx]+")
_FIDELITY_FIGURE_RE = re.compile(r"\b(?:Figure|Fig\.?)\s*[\dIVXivx]+")


def _pdf_page_texts(pdf_path: Optional[str]) -> tuple[Optional[list[str]], Optional[str]]:
    """逐页取 PDF 文本层；失败返回 (None, 失败原因)——不伪造成通过。"""
    if not pdf_path:
        return None, "论文记录未带 pdf_path"
    path = Path(pdf_path)
    if not path.exists():
        return None, f"PDF 不存在: {pdf_path}"
    try:
        import pymupdf  # PyMuPDF 新包名
    except ImportError:
        try:
            import fitz as pymupdf  # 旧包名
        except ImportError as e:  # pragma: no cover —— 依赖缺失只在环境异常时发生
            return None, f"PyMuPDF 不可用，无法逐页比对: {e}"
    try:
        with pymupdf.open(str(path)) as doc:
            return [(page.get_text() or "") for page in doc], None
    except Exception as e:  # noqa: BLE001 —— 解析失败要如实记录原因
        return None, f"PDF 无法解析: {e}"


def _normalize_for_match(text: str) -> str:
    """归一化文本，用于判断「这段 PDF 文字是否出现在 markdown 里」。

    只留字母数字（去空白/标点/大小写/markdown 标记/行尾断词连字符），使两侧的
    排版差异不影响命中判断。
    """
    t = (text or "").lower().replace("­", "")
    t = re.sub(r"-\s*\n\s*", "", t)  # 行尾断词（"trans-\nformer" → "transformer"）
    return re.sub(r"[^0-9a-z一-鿿]+", "", t)


def _page_probes(page_text: str, limit: int = 3) -> list[str]:
    """从一页 PDF 文本里取若干「指纹」片段，用于判断该页内容是否进了 markdown。

    只取足够长的行（滤掉页边行号、页眉页脚等短行），按长度取最长的几条。
    """
    lines = [ln.strip() for ln in (page_text or "").splitlines()]
    lines = [
        ln for ln in lines
        if len(ln) >= _FIDELITY_MIN_PROBE_CHARS and not ln.replace(" ", "").isdigit()
    ]
    lines.sort(key=len, reverse=True)
    probes = (_normalize_for_match(ln) for ln in lines[:limit])
    return [p for p in probes if len(p) >= _FIDELITY_MIN_PROBE_CHARS]


def _fidelity_index_gaps(page_texts: list[str], index: dict, tables_path: Optional[Path]) -> list[str]:
    """索引与 PDF 内容明显对不上的缺口（表格/图注）。"""
    gaps: list[str] = []
    table_pages = [i for i, t in enumerate(page_texts, start=1) if _FIDELITY_TABLE_RE.search(t or "")]
    grid_tables: list = []
    if tables_path is not None and tables_path.exists():
        try:
            data = json.loads(tables_path.read_text(encoding="utf-8"))
            grid_tables = (data.get("tables") or []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, OSError):
            grid_tables = []
    if not (index.get("tables") or []):
        if table_pages:
            gaps.append(
                "表格索引为 0，但 PDF 第 "
                + "、".join(str(p) for p in table_pages[:5])
                + " 页出现 “Table” 字样——表格可能未被解析成 markdown 表格"
            )
        if grid_tables:
            gaps.append(
                f"pdfplumber 抓到 {len(grid_tables)} 张表格网格，但 markdown 表格索引为 0（表格未落到 markdown）"
            )
    # 不再用「网格数 vs 索引数」的半量比对：pdfplumber 会把图片框/分栏线当成表格网格
    # （实测 scGPT 报 60 张 vs 实际结果表 5 张），分母不可靠 → 只报「markdown 一张表都没有」这种确凿情况。
    figure_pages = [i for i, t in enumerate(page_texts, start=1) if _FIDELITY_FIGURE_RE.search(t or "")]
    has_fig_caption = any((c.get("kind") == "figure") for c in (index.get("captions") or []))
    if figure_pages and not has_fig_caption:
        gaps.append(
            "图注索引为 0，但 PDF 第 "
            + "、".join(str(p) for p in figure_pages[:5])
            + " 页出现 “Figure” 字样——图注可能未被解析"
        )
    return gaps


def _write_fidelity_report(md_path: Path, report: dict) -> dict:
    path = Path(md_path).parent / "fidelity_report.json"
    try:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(path)
    except OSError as e:
        report["report_path"] = None
        report["warnings"].append(f"保真核对报告写入失败: {e}")
    return report


def _check_fidelity(
    paper_id: str,
    pdf_path: Optional[str],
    md_path: Path,
    index: dict,
    tables_path: Optional[Path] = None,
) -> dict:
    """PDF ↔ markdown 保真核对（需求二.1「先转 markdown 再一一对照」）。

    固定代码、不依赖大模型：产出 `data/papers/<id>/fidelity_report.json`，含
    PDF 页数、markdown 章节/表格/图注/公式计数、按页文本覆盖率与明显缺口列表。
    无 PDF 或无法解析时 `ok=false` 并写明 error（**不伪造成通过**）；覆盖率过低给出明确告警。
    """
    md = Path(md_path)
    report: dict = {
        "schema_version": "1.0",
        "paper_id": paper_id,
        "created_at": _now(),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "markdown_path": str(md),
        "ok": False,
        "error": None,
        "pdf": {"pages": None, "chars": None},
        "markdown": {},
        "coverage": None,
        "gaps": [],
        "warnings": [],
    }
    try:
        md_text = md.read_text(encoding="utf-8") if md.exists() else ""
        report["markdown"] = {
            "chars": len(md_text),
            "lines": (md_text.count("\n") + 1) if md_text else 0,
            "sections": len(index.get("sections") or []),
            "tables": len(index.get("tables") or []),
            "equations": len(index.get("equations") or []),
            "captions": len(index.get("captions") or []),
            "figure_captions": len([c for c in (index.get("captions") or []) if c.get("kind") == "figure"]),
        }
        page_texts, err = _pdf_page_texts(pdf_path)
        if page_texts is None:
            report["error"] = err
            report["gaps"].append(f"PDF 侧核对未完成：{err}")
            report["warnings"].append(f"保真核对**未通过**（{err}）：markdown 内容未经原文比对")
            return _write_fidelity_report(md, report)

        page_chars = [len((t or "").strip()) for t in page_texts]
        report["pdf"] = {"pages": len(page_texts), "chars": sum(page_chars)}
        report["gaps"].extend(_fidelity_index_gaps(page_texts, index, Path(tables_path) if tables_path else None))
        if index.get("n_pages") and index["n_pages"] != len(page_texts):
            report["warnings"].append(
                f"markdown 索引记 {index['n_pages']} 页，PDF 实际 {len(page_texts)} 页（以 PDF 为准）"
            )

        # 逐页判「该页的正文片段有没有进 markdown」。
        # 不用「按锚点把字符摊回页码」——纯正文页没有标题/表格锚点，会被算成覆盖率 0%，
        # 明明内容在 markdown 里却报「内容未进入」（实测 scGPT 误报 14 页）。
        md_norm = _normalize_for_match(md_text)
        rows = []
        for i, text in enumerate(page_texts, start=1):
            probes = _page_probes(text or "")
            if not probes:
                continue  # 该页没有足量正文（封面/纯图页/扫描页），不参与判定
            hits = sum(1 for p in probes if p in md_norm)
            rows.append({
                "page": i,
                "pdf_chars": page_chars[i - 1],
                "probes": len(probes),
                "hits": hits,
                "ratio": round(hits / len(probes), 4),
            })
        if not rows:
            report["coverage"] = {
                "method": "per-page-content-presence",
                "available": False,
                "reason": "PDF 各页都没有足量正文片段（疑似纯图扫描件）",
            }
            report["warnings"].append("无法按页比对内容：PDF 各页都没有足量正文（疑似扫描件）")
        else:
            low = [r for r in rows if r["ratio"] < FIDELITY_MIN_COVERAGE]
            mean = round(sum(r["ratio"] for r in rows) / len(rows), 4)
            report["coverage"] = {
                "method": "per-page-content-presence",
                "available": True,
                "min_probe_chars": _FIDELITY_MIN_PROBE_CHARS,
                "threshold": FIDELITY_MIN_COVERAGE,
                "mean_ratio": mean,
                "pages_below_threshold": [r["page"] for r in low],
                "pages": rows,
            }
            for r in low:
                report["gaps"].append(
                    f"第 {r['page']} 页 PDF 有 {r['pdf_chars']} 字符文本，但该页正文片段在 markdown 里"
                    f"只命中 {r['hits']}/{r['probes']} ——疑似该页内容丢失（公式/表格页常见）"
                )
            if mean < FIDELITY_MIN_COVERAGE:
                report["warnings"].append(
                    f"按页内容命中率均值 {mean:.0%} 低于阈值 {FIDELITY_MIN_COVERAGE:.0%}："
                    "markdown 与 PDF 差异较大，需人工复核（不静默通过）"
                )
        report["ok"] = report["error"] is None
    except Exception as e:  # noqa: BLE001 —— 核对自身失败也必须如实落盘
        report["ok"] = False
        report["error"] = f"保真核对执行失败: {e}"
        report["gaps"].append(report["error"])
    return _write_fidelity_report(md, report)


# --------------------------- 4.1 PDF → markdown ---------------------------

def parse(paper_id: str) -> str:
    _get_paper(paper_id)
    return task_manager.create_task(TASK_PARSE, params={"paper_id": paper_id})


async def _fix_markdown(md_path: Path, index: dict, pdf_path: Optional[str] = None) -> None:
    """agent 修正表格/公式/图注（D5：规则为主、agent 为辅），原地编辑 markdown。

    需求二.1 要求「先转 markdown 再一一对照」：prompt 明确要求以 `paper.pdf` 原文为基准
    逐处对照修正，并把 PDF 所在目录加入可读目录（Claude Code 的 Read 支持 PDF）。
    无 PDF 时如实说明只能按位置索引修正，不假装做过对照。
    """
    pdf = Path(pdf_path) if pdf_path else None
    if pdf is not None and pdf.exists():
        pdf_part = (
            f"论文原文 PDF：{pdf}\n"
            "要求：先 Read 该 PDF 中表格/公式/图注所在页，与 markdown **逐处对照**，"
            "修正 markdown 里表格（行列错位、单元格合并、`<br>` 误并成一行）、公式"
            "（LaTeX 符号/上下标丢失、`$$` 块残缺）、图注（图号或说明错位）的解析错误。\n"
        )
        dirs = [str(md_path.parent), str(pdf.parent)]
    else:
        pdf_part = (
            "注意：本次未提供可读的论文 PDF（记录无 pdf_path 或文件不存在），"
            "只能依据 section_index.json 的位置索引修正明显错误；无法判断处保持原样。\n"
        )
        dirs = [str(md_path.parent)]
    prompt = (
        "任务：对照论文原文 PDF 修正一个 markdown 文件中表格/公式/图注的解析错误。\n"
        f"目标文件：{md_path}\n"
        f"{pdf_part}"
        f"同目录的 section_index.json 是位置索引（表格 {len(index.get('tables') or [])} 个、"
        f"公式 {len(index.get('equations') or [])} 个、图注 {len(index.get('captions') or [])} 个），"
        "可据此定位需要核对的表格/公式/图注。\n\n"
        "要求：直接 Read 该 markdown，用 Edit 就地修正明显错误，**保持章节标题层级不变**；"
        "只改表格/公式/图注的解析错误，不要重写正文、不要臆造原文没有的内容、不要浏览仓库、"
        "不要修改该 markdown 之外的文件。核对后确认无需改动就保持文件不变，回复“无需修正”；"
        "有改动则回复“已修正”。\n"
        "**回合预算有限：不要逐页通读整篇 PDF**——最多挑 5~8 页最可能出错的位置"
        "（多列表格页、公式密集页、图注页，可用 Read 的页码范围一次读多页）核对后立即动手，"
        "读完没发现问题就回复“无需修正”。"
    )
    # 回合/超时预算：对照 PDF 核对要多次 Read（57 页论文实测 10 回合不够、20 回合仍被读页吃光）。
    # 上调到 40 并在 prompt 里限制「不要逐页通读」。失败仍非致命：规则链路已产出 markdown，
    # 保真核对独立进行，UI 会把 agent_fix 的失败原因如实显示。
    await agent_service.run_sync(
        prompt, cwd=str(md_path.parent), add_dirs=list(dict.fromkeys(dirs)), max_turns=40, timeout_s=360
    )


async def _run_parse(params: dict, task_id: str) -> None:
    paper_id = params["paper_id"]
    paper = _get_paper(paper_id)
    pdf_path = paper.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        raise RuntimeError(f"论文 PDF 不存在: {pdf_path}")

    d = _paper_dir(paper_id)
    md_path = d / "paper.md"
    idx_path = d / "section_index.json"
    tables_path = d / "tables.json"
    log: list[str] = []

    rc, out = await proc_util.run_command(
        script_command(PDF_SCRIPT, pdf_path, md_path, idx_path, tables_path),
    )
    if rc != 0:
        raise RuntimeError(f"PDF 解析脚本失败（rc={rc}）: {out[-1500:]}")
    index = json.loads(idx_path.read_text(encoding="utf-8"))

    agent_fix = "skipped"
    markdown_changed = False
    index_rebuilt = False
    hint: Optional[str] = None
    if index.get("quality") == "scanned":
        hint = "扫描版无文本层，建议 OCR（预留）"
        log.append("扫描版 PDF：跳过 agent 修正与索引重建（无文本层可对照）；建议 OCR（预留）")
    else:
        # agent 修正为辅（D5）：失败不拖垮规则产出，保留规则 markdown 与索引
        before = _md_hash(md_path)
        try:
            await _fix_markdown(md_path, index, pdf_path)
            agent_fix = "applied"
        except Exception as e:  # noqa: BLE001
            agent_fix = f"failed: {e}"
        markdown_changed = _md_hash(md_path) != before
        log.append(f"agent 对照 PDF 修正：{agent_fix}；markdown {'有改动' if markdown_changed else '未改动'}")
        if markdown_changed:
            # 只有 agent 真的改了 markdown 才重建索引；重建时页码由脚本侧继承旧索引
            # （本服务不重复实现继承逻辑）。未改动则保留原索引，避免 page/n_pages 变空。
            rc, out = await proc_util.run_command(
                script_command(PDF_SCRIPT, "--index-only", md_path, idx_path),
            )
            if rc != 0:
                raise RuntimeError(f"索引重建失败（rc={rc}）: {out[-800:]}")
            index = json.loads(idx_path.read_text(encoding="utf-8"))
            index_rebuilt = True
            log.append("markdown 有改动 → 已重建 section_index（页码按脚本侧继承策略保留）")
        else:
            log.append("markdown 未改动 → 保留原索引（含页码），跳过 --index-only 重建")

    # 保真核对（需求二.1「一一对照」）：固定代码；失败/覆盖率过低都明确告警，不静默
    try:
        report = _check_fidelity(paper_id, pdf_path, md_path, index, tables_path)
    except Exception as e:  # noqa: BLE001 —— 核对失败不拖垮解析产出，但如实上报
        report = {"ok": False, "error": f"保真核对未执行: {e}", "gaps": [], "warnings": []}
    coverage = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    fidelity = {
        "ok": bool(report.get("ok")),
        "report_path": report.get("report_path"),
        "error": report.get("error"),
        "coverage_mean": coverage.get("mean_ratio"),
        "pages_below_threshold": coverage.get("pages_below_threshold") or [],
        "gaps": report.get("gaps") or [],
        "warnings": report.get("warnings") or [],
    }
    if not fidelity["ok"]:
        log.append(f"保真核对未通过：{fidelity['error']}")
    else:
        log.append(
            f"保真核对通过：PDF {report.get('pdf', {}).get('pages')} 页，"
            f"按页内容命中率均值 {fidelity['coverage_mean']}"
        )
    for warning in fidelity["warnings"]:
        log.append(f"告警：{warning}")
    for gap in fidelity["gaps"][:10]:
        log.append(f"缺口：{gap}")

    knowledge_service.update_paper(
        paper_id, status="parsed", markdown_path=str(md_path), section_index=index
    )
    task_manager.update_progress(task_id, {
        "markdown_path": str(md_path),
        "sections": len(index.get("sections") or []),
        "n_pages": index.get("n_pages"),
        "quality": index.get("quality"),
        "hint": hint,
        "agent_fix": agent_fix,
        "markdown_changed": markdown_changed,
        "index_rebuilt": index_rebuilt,
        "fidelity": fidelity,
        "log": log,
    })


# ----------------- 4.2 抽取取样：按章节与结果表定位（不再只取头部） -----------------

# 方法/实验/结果相关章节关键词（报告值多在下游结果表，必须保证进取样）
_EXTRACT_SECTION_KEYWORDS = (
    "method", "approach", "architecture", "experiment", "experimental", "evaluation",
    "result", "benchmark", "ablation", "implementation", "setup", "analysis",
    "方法", "实验", "结果", "评估", "消融", "实现", "设置", "分析",
)
_EXTRACT_ABSTRACT_KEYWORDS = ("abstract", "摘要", "summary")


def _split_sections(markdown: str, index: dict) -> list[dict]:
    """按 section_index 的行号把 markdown 切成分节块；索引不可用时按空行分段兜底。

    返回 `[{"title","level","start","end","text"}]`（start/end 为 0 基半开行区间）；
    兜底分段的 title 为空串（不编造标题）。
    """
    lines = markdown.split("\n")
    marks: list[tuple[int, str, Optional[int]]] = []
    for sec in index.get("sections") or []:
        line, title = sec.get("line"), str(sec.get("title") or "")
        if isinstance(line, int) and 1 <= line <= len(lines):
            marks.append((line, title, sec.get("level")))
    marks = sorted(set(marks))
    pieces: list[dict] = []
    if marks:
        if marks[0][0] > 1:
            head = "\n".join(lines[: marks[0][0] - 1]).strip()
            if head:
                pieces.append({"title": "", "level": None, "start": 0,
                               "end": marks[0][0] - 1, "text": head})
        for i, (line, title, level) in enumerate(marks):
            end = marks[i + 1][0] - 1 if i + 1 < len(marks) else len(lines)
            text = "\n".join(lines[line - 1: end]).strip()
            if text:
                pieces.append({"title": title, "level": level, "start": line - 1, "end": end, "text": text})
        return pieces
    buf: list[str] = []
    buf_chars = 0
    for line in lines:
        buf.append(line)
        buf_chars += len(line) + 1
        if not line.strip() and buf_chars >= 2000:
            text = "\n".join(buf).strip()
            if text:
                pieces.append({"title": "", "level": None, "start": 0, "end": 0, "text": text})
            buf, buf_chars = [], 0
    text = "\n".join(buf).strip()
    if text:
        pieces.append({"title": "", "level": None, "start": 0, "end": 0, "text": text})
    return pieces


def _markdown_table_fragments(markdown: str) -> list[str]:
    """markdown 里的表格块（连续以 `|` 开头的行，至少两行）——结果表片段来源之一。"""
    blocks: list[str] = []
    cur: list[str] = []
    for line in markdown.split("\n"):
        if line.lstrip().startswith("|"):
            cur.append(line)
        else:
            if len(cur) >= 2:
                blocks.append("\n".join(cur))
            cur = []
    if len(cur) >= 2:
        blocks.append("\n".join(cur))
    return blocks


def _match_keywords(title: str, keywords) -> bool:
    text = (title or "").lower()
    return bool(text) and any(k in text for k in keywords)


def _pack(blocks: list[tuple[str, str]], limit: int) -> tuple[str, int, list[str], int]:
    """把 `(小标题, 文本)` 拼进 limit 字符预算，**按剩余段数均分**后各段各自截断。

    均分（而非顺序吃满）是刻意的：一篇论文常有一个超长的方法章节，顺序分配会让它独占
    全部章节预算、把「实验/结果」整段挤掉；均分保证每个方法/实验/结果章节都有代表内容进来，
    短章节省下的预算自动留给后面的段。

    返回 `(文本, 使用字符数, 省略说明, 已纳入的段数)`；截断/未纳入的段都在说明里如实写明。
    """
    if not blocks:
        return "", 0, [], 0
    if limit <= 0:
        return "", 0, [f"预算不足，{len(blocks)} 段均未纳入"], 0
    out: list[str] = []
    used = 0
    notes: list[str] = []
    included = 0
    total = len(blocks)
    for i, (head, text) in enumerate(blocks):
        text = text or ""
        head_cost = len(head) + 5  # "### " 前缀 + 换行
        remaining = total - i
        allow = (limit - used) // remaining
        room = allow - head_cost
        if room <= 0:
            notes.append(f"“{head}”及其后 {remaining} 段因预算不足未纳入")
            break
        if len(text) <= room:
            out.append(f"### {head}\n{text}")
            used += head_cost + len(text)
        else:
            cut_note = "\n（该段按预算截断，原文更长）"
            room = max(allow - head_cost - len(cut_note), 1)
            out.append(f"### {head}\n{text[:room]}{cut_note}")
            used += head_cost + room + len(cut_note)
            notes.append(f"“{head}”因预算被截断（原 {len(text)} 字符 → 保留 {room} 字符）")
        included = i + 1
    return "\n\n".join(out), used, notes, included


def select_extract_context(
    markdown: str,
    index: dict,
    tables_text: str = "",
    budget: int = MAX_EXTRACT_CHARS,
) -> tuple[str, dict]:
    """4.2 抽取取样：按**方法/实验章节 + 结果表格片段 + 摘要**定位，不再只取 markdown 头部。

    报告值几乎都在下游结果表格里，头部截断会把它们整段切掉，于是 agent 只能看到摘要/引言。
    本函数在 `budget`（默认 `MAX_EXTRACT_CHARS`）总预算内按优先级取样：
    (a) 方法/实验/结果章节 →(b) 结果表格片段（tables.json 网格 + markdown 表格）→(c) 摘要，
    剩余预算再按原文顺序回填其余章节；被截断/未纳入的部分在正文与 meta 里如实写明。

    返回 `(喂给 agent 的正文, meta)`；meta 含本次纳入的章节/表格清单与省略说明，
    供 prompt 与任务进度如实展示「这次到底喂了什么」。
    """
    markdown = markdown or ""
    budget = max(int(budget), 1000)
    sections = _split_sections(markdown, index)
    key_sections = [s for s in sections if _match_keywords(s["title"], _EXTRACT_SECTION_KEYWORDS)]
    abstract_sections = [
        s for s in sections
        if s not in key_sections and _match_keywords(s["title"], _EXTRACT_ABSTRACT_KEYWORDS)
    ]
    chosen = {id(s) for s in key_sections} | {id(s) for s in abstract_sections}
    rest_sections = [s for s in sections if id(s) not in chosen]

    section_budget = int(budget * EXTRACT_SECTION_SHARE)
    table_budget = int(budget * EXTRACT_TABLE_SHARE)
    abstract_budget = max(budget - section_budget - table_budget, 0)

    def _label(prefix: str, sec: dict) -> str:
        return f"{prefix}：{sec['title']}" if sec.get("title") else f"{prefix}（无标题）"

    section_text, section_used, section_notes, section_included = _pack(
        [(_label("章节", s), s["text"]) for s in key_sections], section_budget
    )
    table_blocks: list[tuple[str, str]] = []
    if (tables_text or "").strip():
        table_blocks.append(("PDF 表格网格（pdfplumber 确定性抽取，仅用于核对数字）", tables_text.strip()))
    for i, fragment in enumerate(_markdown_table_fragments(markdown), start=1):
        table_blocks.append((f"markdown 表格片段 {i}", fragment))
    table_text, table_used, table_notes, table_included = _pack(table_blocks, table_budget)
    abstract_text, abstract_used, abstract_notes, abstract_included = _pack(
        [(_label("摘要", s), s["text"]) for s in abstract_sections], abstract_budget
    )
    used = section_used + table_used + abstract_used
    rest_text, rest_used, rest_notes, rest_included = _pack(
        [(_label("其余章节", s), s["text"]) for s in rest_sections], max(budget - used, 0)
    )
    used += rest_used

    parts = [p for p in (section_text, table_text, abstract_text, rest_text) if p]
    body = "\n\n".join(parts)
    if len(body) > budget:
        # 段间连接符等开销可能让拼接结果略超预算：按总预算硬截断（尾部为「其余章节」，不影响
        # 方法/实验章节与结果表格片段），并在省略说明里如实记录。
        body = body[: max(budget - len("（已按总预算截断）"), 0)] + "（已按总预算截断）"
        extra_note = [f"正文拼接超出总预算 {budget} 字符，尾部「其余章节」已硬截断"]
    else:
        extra_note = []
    omitted = (
        [s["title"] or "（无标题）" for s in key_sections[section_included:]]
        + [s["title"] or "（无标题）" for s in abstract_sections[abstract_included:]]
        + [s["title"] or "（无标题）" for s in rest_sections[rest_included:]]
    )
    meta = {
        "full_chars": len(markdown),
        "used_chars": used,
        "budget": budget,
        "sections_included": [s["title"] or "（无标题）" for s in key_sections[:section_included]],
        "abstract_included": [s["title"] or "（无标题）" for s in abstract_sections[:abstract_included]],
        "table_fragments": len(table_blocks),
        "table_fragments_included": table_included,
        "omitted_sections": omitted,
        "omitted_notes": section_notes + table_notes + abstract_notes + rest_notes + extra_note,
        "truncated": bool(section_notes or table_notes or abstract_notes or rest_notes or extra_note),
    }
    return body, meta


# --------------------------- 4.2 实验条目抽取 ---------------------------

def extract_items(paper_id: str) -> str:
    _get_paper(paper_id)
    return task_manager.create_task(TASK_EXTRACT, params={"paper_id": paper_id})


async def _run_extract(params: dict, task_id: str) -> None:
    """4.2 抽取条目。

    取样口径（不再头部截断）：`select_extract_context` 按 section_index 选出方法/实验/结果
    章节，并把 tables.json 网格与 markdown 表格片段作为**结果表片段**单独喂入；总预算
    `MAX_EXTRACT_CHARS` 内优先保方法/实验章节与结果表，被省略的部分在 prompt 与任务进度
    里如实写明。抽取到的 `metric_name` 经 `normalize_metric_name` 归一到
    `contracts.CANONICAL_METRICS` 口径后再落库。
    """
    paper_id = params["paper_id"]
    paper = _get_paper(paper_id)
    md_path = paper.get("markdown_path")
    if not md_path or not Path(md_path).exists():
        raise RuntimeError("论文尚未转 markdown，请先调用 POST /api/papers/{id}/parse")

    markdown = Path(md_path).read_text(encoding="utf-8")
    index: dict = {}
    if paper.get("section_index"):
        try:
            index = json.loads(paper["section_index"])
        except (json.JSONDecodeError, TypeError):
            index = {}

    # 报告值多在下游的结果表格中；tables.json 的网格用于核对数字（markdown 表格可能合并/错列）
    table_caps = [c.get("text") for c in index.get("captions", []) if c.get("kind") == "table"]
    tables_txt = _render_tables(Path(md_path).parent / "tables.json")
    body, sample = select_extract_context(markdown, index, tables_txt)

    def _joined(items: list) -> str:
        return "、".join(items) if items else "（无）"

    sampling_note = (
        f"\n本次取样（原文全文 {sample['full_chars']} 字符，喂入 {sample['used_chars']}/"
        f"{sample['budget']} 字符，**非头部截断**）：\n"
        f"- 纳入方法/实验/结果章节：{_joined(sample['sections_included'])}\n"
        f"- 纳入结果表格片段 {sample['table_fragments_included']}/{sample['table_fragments']} 段"
        "（含 pdfplumber 网格与 markdown 表格）\n"
        f"- 纳入摘要：{_joined(sample['abstract_included'])}\n"
        f"- 被省略：{('；'.join(sample['omitted_notes'])) if sample['omitted_notes'] else '无'}\n"
    )
    hint = f"\n提示：含论文报告值的表格标题有：{table_caps}\n" if table_caps else ""
    tables_hint = (
        "\n参考：上文「PDF 表格网格」是从 PDF 确定性抽取的表格网格（可能与相邻表合并或列错位，"
        "**仅用于核对数字**，以 markdown 正文为准）。\n"
        if tables_txt else ""
    )

    prompt = (
        "请从以下论文 markdown 的方法与实验章节中抽取实验条目，每条包含六要素："
        "数据集(dataset_name)、划分方式(split_method)、评价指标(metric_name)、"
        "论文报告值(metric_value_reported)、超参数(hyperparams)、对比基线(baselines)。\n"
        "**论文报告值必须取自结果表格中的实际数字**（如 “Table 3 …” 的表格行），"
        "不要因为数字在表格里就标为“未报告”；确实缺失的项才标注为“未报告”，**不要臆造**。\n"
        "**语言约定**：描述性文字一律**用中文**（split_method 的说明、报告值里的括注说明、"
        "缺失占位等；缺失一律写中文“未报告”，不要写 “not reported”）。"
        "但 **dataset_name（数据集名）、metric_name（指标名）、metric_unit（单位）、"
        "hyperparams 的键名、baselines 的条目、section_ref（章节位置）保留论文原文**——"
        "这些是专有名词或机器可读字段（指标名要参与归一、数据集名要用于匹配），不要翻译。\n"
        "评价指标名给出论文里的原始写法即可，系统会统一归一到规范指标名。\n"
        "**一条目只对应一个指标**：结果表同一行给出多个指标（如 Accuracy/Precision/Recall/MacroF1）时，"
        "**按指标拆成多条**（同数据集、同划分方式，只有指标不同），不要把多个指标塞进同一条，"
        "metric_name 里也不要出现逗号分隔的多指标。\n"
        "**metric_value_reported 只写该指标的值**（取结果表里的数字；约数照原样写，如 `~0.85`；"
        "确实没有数值才写「未报告」），不要写整句结论或多指标拼串——平台要拿它与实测值逐条比对。\n"
        "注意：markdown 表格单元格内的 `<br>` 表示原表该格内换行（常对应另一行数据），"
        "解析时需按列语义区分，不要把两行数字当成一个值。\n"
        '严格按照如下 JSON 结构输出（顶层键必须是 "items"）：\n'
        '{"items": [{"section_ref":"","dataset_name":"","split_method":"","metric_name":"",'
        '"metric_value_reported":"","metric_unit":"","hyperparams":{},"baselines":[]}]}\n'
        f"{sampling_note}{hint}{tables_hint}\n"
        f"论文 markdown（按章节/结果表取样）：\n{body}"
    )
    result = await agent_service.run_sync(
        prompt, cwd=str(Path(md_path).parent), output_schema=ITEMS_SCHEMA, max_turns=20
    )
    items = _as_items(result.get("structured_output"))
    renamed: list[dict] = []
    for it in items:
        raw = it.get("metric_name")
        canonical = normalize_metric_name(raw)
        if canonical and canonical != str(raw).strip():
            it["metric_name_raw"] = raw
            renamed.append({"raw": raw, "canonical": canonical})
        it["metric_name"] = canonical
    items = _dedup_items(items)
    item_ids = knowledge_service.record_experiment_items(paper_id, items)

    metrics_seen = sorted({str(it.get("metric_name")) for it in items if it.get("metric_name")})
    knowledge_service.update_paper(paper_id, status="extracted")
    task_manager.update_progress(task_id, {
        "count": len(item_ids),
        "item_ids": item_ids,
        "sampling": {
            "full_chars": sample["full_chars"],
            "used_chars": sample["used_chars"],
            "budget": sample["budget"],
            "sections": sample["sections_included"],
            "abstract": sample["abstract_included"],
            "table_fragments": sample["table_fragments_included"],
            "omitted": sample["omitted_notes"],
        },
        "metric_normalize": {
            "renamed_count": len(renamed),
            "renamed": renamed[:20],
            "canonical": [m for m in metrics_seen if _is_canonical_metric(m)],
            "unknown": [m for m in metrics_seen if not _is_canonical_metric(m)],
        },
        "log": [
            f"取样：方法/实验章节 {len(sample['sections_included'])} 个、"
            f"结果表格片段 {sample['table_fragments_included']}/{sample['table_fragments']} 段、"
            f"摘要 {len(sample['abstract_included'])} 个（{sample['used_chars']}/{sample['budget']} 字符）",
            f"指标名归一：{len(renamed)} 条被改名（复用 contracts.CANONICAL_METRICS 别名表）",
        ],
    })


# --------------------------- 4.3 自动复现执行 ---------------------------

def reproduce(paper_id: str, project_id: str) -> str:
    """发起复现。顺带记一次「论文 ↔ 项目」绑定留痕（复现板据此列出用过的项目/环境）。"""
    _get_paper(paper_id)
    task_id = task_manager.create_task(
        TASK_REPRODUCE, project_id=project_id, params={"paper_id": paper_id, "project_id": project_id}
    )
    knowledge_service.record_paper_project_binding(paper_id, project_id, task_id)
    return task_id


def _merge_progress(task_id: str, **fields) -> None:
    """合并写任务进度——`update_progress` 是**整体覆盖**，直接写会冲掉同批的键。"""
    task = task_manager.get_task(task_id) or {}
    base: dict = {}
    if task.get("progress"):
        try:
            base = json.loads(task["progress"])
        except (json.JSONDecodeError, TypeError):
            base = {}
    if not isinstance(base, dict):
        base = {}
    base.update(fields)
    task_manager.update_progress(task_id, base)


async def _run_item(
    python: str, script: Path, source: Path, item_json: Path, out_json: Path, cwd: Path,
    on_line=None,
) -> dict:
    """跑一条复现；走 proc_util（取消/超时即杀进程树）。`on_line` 用于实时上报进度。"""
    try:
        rc, log = await proc_util.run_command(
            [python, str(script), str(source), str(item_json), str(out_json)],
            cwd=str(cwd), timeout=REPRODUCE_TIMEOUT_S, on_line=on_line,
        )
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"复现运行超时（>{REPRODUCE_TIMEOUT_S}s，已终止进程树）", "log": ""}
    error = None if rc == 0 else (log[-2000:] or f"退出码 {rc}")
    return {"ok": rc == 0, "error": error, "log": log}


# 条目里的结构化字段在库里以 JSON 文本列存储，`SELECT *` 读出来是**字符串**。
# 交给复现脚本时必须是对象/数组（脚本按 `item["hyperparams"]["lr"]` 取值），否则 AttributeError。
_ITEM_JSON_FIELDS = ("hyperparams", "baselines")


def decode_item_json_fields(item: dict) -> dict:
    """把条目的 JSON 文本列解回对象/数组（解析不出则原样保留，不吞数据）。"""
    out = dict(item)
    for key in _ITEM_JSON_FIELDS:
        value = out.get(key)
        if isinstance(value, str) and value.strip():
            try:
                out[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return out


def _data_inventory(data_dir: Path | None) -> str:
    """项目数据目录清单（提示给复现 agent，让它就地取材而不是联网下大文件）。"""
    if data_dir is None or not data_dir.is_dir():
        return ""
    try:
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in data_dir.iterdir())
    except OSError:
        entries = []
    return f"（内含：{', '.join(entries)}）" if entries else "（为空）"


async def _fill_reproduce_script(
    source: Path, run_dir: Path, items: list[dict], data_dir: Path | None = None
) -> None:
    """agent 依据项目代码填写复现脚本的 reproduce()（4.3 步骤 1：固定模板 + agent 填空）。"""
    items_brief = [
        {k: it.get(k) for k in ("item_id", "dataset_name", "split_method", "metric_name",
                                "metric_value_reported", "metric_unit", "hyperparams", "baselines")}
        for it in [decode_item_json_fields(it) for it in items]
    ]
    # 论文数据与预训练权重通常不在代码仓库里（仓库只放取数脚本），而在项目工作区数据目录。
    # 不把该目录交给 agent，它就只能干猜路径或联网拉几 GB 数据 → 复现必然失败。
    data_hint = ""
    if data_dir is not None and data_dir.is_dir():
        data_hint = (
            f"\n项目工作区数据目录：{data_dir} {_data_inventory(data_dir)}\n"
            "复现所需的数据集与预训练权重**优先在该目录下就地查找**（子目录名常与条目 dataset_name 对应，"
            "如 ms/、hPancreas/），**不要联网下载大文件**；找不到再考虑仓库自带脚本取数。\n"
        )
    prompt = (
        f"复现脚本模板已放在 {run_dir / 'reproduce.py'}（函数 reproduce(source_dir, item) 待实现）。\n"
        f"论文对应的代码仓库在 {source}，实验条目如下：\n{json.dumps(items_brief, ensure_ascii=False)}\n"
        f"{data_hint}\n"
        "请阅读仓库代码，**原地编辑**该脚本，实现 reproduce()：按条目的数据集、划分方式、"
        "超参数执行评测，返回 {metric_name, metric_value_actual, n_samples, log_tail}。"
        "不要修改模板的 main() 与输出字段名；无法复现时让脚本非 0 退出即可。完成后回复“已实现”。\n"
        "**重要**：只写脚本、**不要自己运行它**（不要执行训练/评测命令，也不要跑冒烟测试）——"
        "平台随后会在项目环境里按条目执行；自行跑完整训练会占满 GPU、拖慢甚至失败。"
        "需要确认接口用法时读代码即可，不要实际跑数据。"
    )
    add_dirs = [str(run_dir), str(run_dir.parent)]
    if data_dir is not None:
        add_dirs.append(str(data_dir))
    # 默认超时 300s 对「读大仓库 + 写完整复现脚本」这类活偏紧（实测 scGPT 上被截断），
    # 放到 900s：仍受 REPRODUCE_TIMEOUT_S 总预算约束（_collect 内含退避重试）。
    await agent_service.run_sync(
        prompt, cwd=str(source), add_dirs=add_dirs, max_turns=40, timeout_s=900
    )


async def _run_reproduce(params: dict, task_id: str) -> None:
    paper_id, project_id = params["paper_id"], params["project_id"]
    project = project_manager.require_type(project_id, {"original"})
    ws = Path(project["workspace_path"])
    source = ws / "source"
    python = analysis_service._project_python(ws)
    if python is None:
        raise RuntimeError("项目环境未就绪：未找到独立环境解释器，请先完成环境创建")

    # 仅已确认条目参与复现（4.2「确认后条目生效」，对齐 5.3 对齐确认闸门先例）
    items = [it for it in knowledge_service.list_experiment_items(paper_id) if it.get("status") == "confirmed"]
    if not items:
        raise RuntimeError(
            "该论文无已确认的实验条目：请先 POST /api/papers/{id}/extract 抽取条目，"
            "再经 POST /api/papers/{id}/items/{item_id}/confirm 确认后再复现"
        )

    # 本轮复现覆盖旧对照记录，避免结论重复计入同一历史条目
    knowledge_service.clear_reproduction_results(paper_id)

    run_dir = ws / "runs" / "reproduce" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    script = run_dir / "reproduce.py"
    shutil.copyfile(REPRODUCE_TEMPLATE, script)

    # 运行记录的环境信息（数据设计五.4：类型/语言/框架/CUDA 版本）
    environment = {"type": env_manager.detect_env_type(ws),
                   **await asyncio.to_thread(env_manager.detect_versions, source, python)}

    # agent 填空失败则不修脚本，逐条运行会非 0 退出 → 记为「无法复现」（4.3 异常与边界）
    _merge_progress(task_id, stage=f"写复现脚本（agent 读仓库，共 {len(items)} 条待复现）")
    try:
        await _fill_reproduce_script(source, run_dir, items, data_dir=ws / "data")
    except Exception as e:  # noqa: BLE001
        _merge_progress(task_id, script_fill=f"failed: {e}")

    sem = asyncio.Semaphore(REPRODUCE_MAX_PARALLEL)

    async def _one(it: dict, idx: int) -> dict:
        """单条目的复现：运行→落 run_record/metrics/日志→写对照结果。失败不抛出（归无法复现）。

        复现输出契约（4.3 模板固定字段）`{metric_name, metric_value_actual, n_samples, log_tail}`
        全部接住：`metric_name` 与条目比对（不一致则记差异，不改条目口径）、`n_samples` 落
        run_record.params、`log_tail` 追加到该条日志文件。
        """
        async with sem:
            item_json = run_dir / f"item_{it['item_id']}.json"
            out_json = run_dir / f"out_{it['item_id']}.json"
            item_json.write_text(
                json.dumps(decode_item_json_fields(it), ensure_ascii=False), encoding="utf-8"
            )

            started = _now()
            label = f"{it.get('dataset_name') or '?'} · {it.get('metric_name') or '?'}"
            _merge_progress(task_id, stage=f"运行条目 {idx}/{len(items)}：{label}", live=None)

            def _on_line(line: str) -> None:
                # 只把脚本自己的进度行刷进任务进度（epoch/loss 等），别每行写一次库
                text = line.strip()
                if text.startswith("[reproduce]") or text.startswith("epoch"):
                    _merge_progress(task_id, live=text[:300])

            run = await _run_item(python, script, source, item_json, out_json, run_dir, on_line=_on_line)

            payload: dict = {}
            if run["ok"] and out_json.exists():
                try:
                    loaded = json.loads(out_json.read_text(encoding="utf-8"))
                    payload = loaded if isinstance(loaded, dict) else {}
                except (json.JSONDecodeError, OSError):
                    payload = {}
            actual = payload.get("metric_value_actual")
            item_metric = normalize_metric_name(it.get("metric_name"))
            out_metric_raw = payload.get("metric_name")
            out_metric = normalize_metric_name(out_metric_raw) if out_metric_raw else None
            metric_mismatch = bool(out_metric and item_metric and out_metric != item_metric)
            n_samples = payload.get("n_samples")
            log_tail = payload.get("log_tail")

            log_text = run.get("log") or ""
            if log_tail not in (None, ""):
                log_text += "\n[reproduce 输出 log_tail]\n" + str(log_tail)
            log_path = run_dir / f"log_{it['item_id']}.txt"
            log_path.write_text(log_text, encoding="utf-8")

            metric_notes = []
            if metric_mismatch:
                metric_notes.append(
                    f"条目 {it['item_id']}：复现脚本输出的 metric_name={out_metric_raw!r} 与条目 "
                    f"{it.get('metric_name')!r}（归一后 {item_metric}）不一致，按条目口径对照并记录差异"
                )
            elif out_metric and not item_metric:
                metric_notes.append(f"条目 {it['item_id']}：条目无指标名，采用脚本输出 {out_metric_raw!r}")

            run_id = knowledge_service.record_run({
                "project_id": project_id, "task_id": task_id, "run_type": "reproduce",
                "environment": environment,
                "command": f"{python} reproduce.py <source> item_{it['item_id']}.json out_{it['item_id']}.json",
                "params": {
                    "item_id": it["item_id"],
                    "metric_name": item_metric,
                    "metric_name_actual": out_metric,
                    "metric_name_mismatch": metric_mismatch,
                    "n_samples": n_samples,
                },
                "status": "success" if actual is not None else "failed",
                "metrics": {item_metric or out_metric or "metric": actual} if actual is not None else None,
                "error": run.get("error"),
                "artifact_path": str(out_json) if out_json.exists() else None,
                "log_path": str(log_path),
                "started_at": started, "finished_at": _now(),
            })
            knowledge_service.record_reproduction_result({
                "item_id": it["item_id"], "run_id": run_id, "metric_value_actual": actual,
                "verdict": None if actual is not None else VERDICT_UNREPRODUCIBLE,
                "evidence_path": str(out_json) if out_json.exists() else None,
            })
            return {"item_id": it["item_id"], "metric_name": item_metric,
                    "metric_name_actual": out_metric, "metric_name_mismatch": metric_mismatch,
                    "n_samples": n_samples, "metric_value_actual": actual,
                    "ok": actual is not None, "notes": metric_notes}

    gathered = await asyncio.gather(*[_one(it, i) for i, it in enumerate(items, start=1)],
                                    return_exceptions=True)
    summary = [r for r in gathered if isinstance(r, dict)]
    log = [
        note for r in summary for note in (r.get("notes") or [])
    ]
    missing = [r["item_id"] for r in summary if r.get("n_samples") in (None, "")]
    if missing:
        log.append(f"{len(missing)} 条复现输出未带 n_samples（契约字段缺失，如实记录）")

    knowledge_service.update_paper(paper_id, status="reproduced")
    _merge_progress(task_id, stage="完成", live=None, count=len(items), results=summary, log=log)


# --------------------------- 4.4 逐条对照与可信度结论 ---------------------------

def conclusion(paper_id: str) -> str:
    _get_paper(paper_id)
    return task_manager.create_task(TASK_CONCLUSION, params={"paper_id": paper_id})


_PERCENT_UNITS = {"%", "percent", "percentage", "percent improvement", "% improvement", "百分数", "百分比"}
_RATIO_UNITS = {"ratio", "proportion", "fraction", "scale", "比例", "占比", "比值"}
_PER_MILLE_UNITS = {"‰", "per mille", "per mill", "permille", "千分比", "千分率", "千分数"}
_PER_MYRIAD_UNITS = {"‱", "permyriad", "per myriad", "万分比", "万分率", "万分之"}

# 同族量纲换算（单位 → 相对基准单位的系数）；只在两值换算后落入容差内才采纳，避免乱猜
_TIME_FACTORS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0, "秒": 1.0,
    "ms": 1e-3, "msec": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3, "毫秒": 1e-3,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0, "分钟": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0, "小时": 3600.0,
}
_LENGTH_FACTORS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "米": 1.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "厘米": 0.01,
    "mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3, "毫米": 1e-3,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0, "千米": 1000.0, "公里": 1000.0,
    "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "micrometer": 1e-6, "微米": 1e-6,
    "nm": 1e-9, "nanometer": 1e-9, "纳米": 1e-9,
}
# 百分比族单位 → 「1.0 的比例」在该单位下的数值（% → 100，‰ → 1000，‱ → 10000，比例 → 1）
_PERCENT_FAMILY = {
    **{u: 100.0 for u in _PERCENT_UNITS},
    **{u: 1.0 for u in _RATIO_UNITS},
    **{u: 1000.0 for u in _PER_MILLE_UNITS},
    **{u: 10000.0 for u in _PER_MYRIAD_UNITS},
}
_FAMILY_TOLERANCE = float(os.getenv("REPRO_UNIT_FAMILY_TOLERANCE", "0.05"))


def _unit_hint(unit: Optional[str], reported=None, actual=None) -> Optional[str]:
    """确定比较用的单位提示：优先 metric_unit，其次从值里自带的 `%`/`‰` 推断。

    4.2 抽取可能把单位写进值里（`metric_value_reported="91.5%"` 而 metric_unit 为空），
    这种「report 值带 %」的情形必须能判出方向，否则会被算成 99% 偏差。
    """
    key = str(unit or "").strip().lower()
    if key:
        return key
    for raw in (reported, actual):
        text = str(raw or "")
        if "％" in text or "%" in text:
            return "%"
        if "‰" in text:
            return "‰"
        if "‱" in text:
            return "‱"
    return None


def _align_family(a: float, r: float, unit_key: str, factors: dict) -> tuple[float, float]:
    """时间/长度同族换算：报告值按单位换算到基准量纲，再在族内试解实测值的量纲。

    只有换算后偏差落在 `_FAMILY_TOLERANCE`（默认 5%）内才采纳——例如「报告 3600 s
    实测 3600000」可判定实测是毫秒（换算后完全吻合）；换成不吻合的组合就原样返回，
    由 `_classify` 如实判为「不一致」，不硬猜量纲。
    """
    base = factors.get(unit_key)
    if base is None:
        return a, r
    target = r * base
    for factor in sorted({f for f in factors.values() if f != base}):
        if target and abs(a * factor - target) / abs(target) <= _FAMILY_TOLERANCE:
            return a * factor / base, r
    return a, r


def _align_unit(a: float, r: float, unit: Optional[str], reported_raw=None, actual_raw=None) -> tuple[float, float]:
    """量纲归一：把同一指标的两种「单位/量纲写法」换算到同一口径再比。

    覆盖（4.4 D6）：
    - 百分比族：`%`/百分比、比例(ratio)、千分比(‰)、万分比(‱)——一侧是比例(0~1)、
      另一侧是该单位的数值时统一到该单位；值与单位都缺省时可由值里自带的 `%` 判定方向。
    - 同族换算：时间（秒/毫秒/分/时）、长度（m/cm/mm/km/µm/nm），仅在换算后落入 5% 容差时采纳。
    无法归一的组合原样返回（交给 `_classify` 按原始相对误差判档，或在方向不明时判「无法复现」）。
    """
    key = _unit_hint(unit, reported_raw, actual_raw)
    if key is None:
        return a, r
    hint = _PERCENT_FAMILY.get(key)
    if hint is not None:
        if hint > 1:
            # 该单位下 1.0 比例的数值 >1：另一侧若明显是 0~1 的比例就换算到该单位
            if 0 < a <= 1 < r:
                a *= hint
            elif 0 < r <= 1 < a:
                r *= hint
        elif hint == 1.0:
            # 单位是比例/分数：另一侧若明显是百分数（>1）就折成比例
            if 0 < r <= 1 < a:
                a /= 100.0
            elif 0 < a <= 1 < r:
                r /= 100.0
        return a, r
    for factors in (_TIME_FACTORS, _LENGTH_FACTORS):
        if key in factors:
            return _align_family(a, r, key, factors)
    return a, r


def _percent_scale_ambiguous(a: float, r: float) -> bool:
    """无任何单位信息、但「×100 换算」能让两值吻合——方向不明，不能猜。"""
    if 0 < a <= 1 < r <= 100:
        return abs(a * 100.0 - r) / abs(r) <= DEVIATION_APPROX
    if 0 < r <= 1 < a <= 100:
        return abs(r * 100.0 - a) / abs(a) <= DEVIATION_APPROX
    return False


def _classify(actual, reported, unit: Optional[str] = None) -> tuple[Optional[float], int, str]:
    """相对误差分档（D6）：返回 (deviation, passed_threshold, verdict)。

    报告值缺失/非数值（如「未报告」「>50% improvement」「85-90」）或为 0 → **无法复现**：
    无从比较，不能算作「不一致」（原先一律判不一致，会污染总体可信度）。
    单位/量纲不一致时先经 `_align_unit` 归一；**无任何单位线索又无法判定方向**（一侧是
    0~1 比例、另一侧只在「×100 换算」下才吻合）时同样判「无法复现」，不硬猜。
    """
    if actual is None:
        return None, 0, VERDICT_UNREPRODUCIBLE
    a, r = _to_float(actual), _to_float(reported)
    if a is None or r is None or r == 0:
        return None, 0, VERDICT_UNREPRODUCIBLE
    hint = _unit_hint(unit, reported, actual)
    aligned = hint is not None and (
        hint in _PERCENT_FAMILY or hint in _TIME_FACTORS or hint in _LENGTH_FACTORS
    )
    if aligned:
        a, r = _align_unit(a, r, unit, reported, actual)
    elif _percent_scale_ambiguous(a, r):
        return None, 0, VERDICT_UNREPRODUCIBLE
    deviation = abs(a - r) / abs(r)
    if deviation <= DEVIATION_CONSISTENT:
        return deviation, 1, VERDICT_CONSISTENT
    if deviation <= DEVIATION_APPROX:
        return deviation, 0, VERDICT_APPROX
    return deviation, 0, VERDICT_INCONSISTENT


def _rule_overall(verdicts: list[str]) -> str:
    """总体分档兜底规则：有实际值的条目里取最差档；全为无法复现则无法复现。"""
    reproduced = [v for v in verdicts if v != VERDICT_UNREPRODUCIBLE]
    if not reproduced:
        return VERDICT_UNREPRODUCIBLE
    return max(reproduced, key=lambda v: _VERDICT_RANK.get(v, 3))


async def _run_conclusion(params: dict, task_id: str) -> None:
    paper_id = params["paper_id"]
    _get_paper(paper_id)
    results = knowledge_service.list_reproduction_results(paper_id)
    if not results:
        raise RuntimeError("该论文无复现记录，请先调用 POST /api/papers/{id}/reproduce")

    item_results = []
    for r in results:
        item = knowledge_service.get_item("experiment_item", r.get("item_id") or "") or {}
        deviation, passed, verdict = _classify(
            r.get("metric_value_actual"), r.get("metric_value_reported"), item.get("metric_unit")
        )
        knowledge_service.update_reproduction_result(
            r["result_id"], deviation=deviation, passed_threshold=passed, verdict=verdict
        )
        item_results.append({
            "item_id": r["item_id"], "metric_name": r.get("metric_name"),
            "metric_value_reported": r.get("metric_value_reported"),
            "metric_value_actual": r.get("metric_value_actual"),
            "deviation": deviation, "verdict": verdict,
        })

    rule_overall = _rule_overall([it["verdict"] for it in item_results])

    prompt = (
        "以下是论文各实验条目的复现对照（相对误差 deviation 与分档 verdict）：\n"
        f"{json.dumps(item_results, ensure_ascii=False)}\n\n"
        f"请给出总体可信度分档（verdict_rule 兜底为“{rule_overall}”），并起草一段总体可信度结论，"
        "说明复现一致性、主要偏差来源与无法复现的条目。"
    )
    try:
        drafted = await agent_service.run_sync(prompt, output_schema=SUMMARY_SCHEMA, max_turns=8)
        out = drafted.get("structured_output") or {}
    except Exception:  # noqa: BLE001 —— agent 起草为辅，失败回退规则分档
        out = {}
    overall = out.get("overall_verdict")
    if overall not in _VERDICT_RANK:
        overall = rule_overall

    knowledge_service.record_credibility_conclusion(paper_id, {
        "overall_verdict": overall,
        "summary": out.get("summary") or f"总体可信度：{overall}（规则分档 {rule_overall}）",
        "item_results": item_results,
    })
    knowledge_service.update_paper(paper_id, status="concluded")
    task_manager.update_progress(task_id, {"overall_verdict": overall, "items": len(item_results)})


# --------------------------- 查询 ---------------------------

def get_conclusion(paper_id: str) -> Optional[dict]:
    return knowledge_service.get_credibility_conclusion(paper_id)


def get_detail(paper_id: str) -> dict:
    paper = _get_paper(paper_id)
    return {
        "paper": paper,
        "items": knowledge_service.list_experiment_items(paper_id),
        "reproduction_results": knowledge_service.list_reproduction_results(paper_id),
        "conclusion": knowledge_service.get_credibility_conclusion(paper_id),
        # 论文↔项目 绑定留痕（复现板：这篇论文用过哪些项目/环境复现）
        "bindings": knowledge_service.list_paper_project_bindings(paper_id),
    }


def register() -> None:
    task_manager.register_handler(TASK_PARSE, _run_parse)
    task_manager.register_handler(TASK_EXTRACT, _run_extract)
    task_manager.register_handler(TASK_REPRODUCE, _run_reproduce)
    task_manager.register_handler(TASK_CONCLUSION, _run_conclusion)
