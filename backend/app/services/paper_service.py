"""模块二：论文自动复现与可信度确定（模块详细设计四章 4.1~4.4，D5/D6）。

四个任务：
- pdf_parse   4.1 PDF → markdown（pymupdf4llm 规则为主 + agent 修正表格/公式/图注）
- extract_items 4.2 实验条目抽取（agent，六要素缺失标注而非臆造）
- reproduce   4.3 自动复现（agent 生成复现脚本 + 项目独立环境逐条执行 + run_record）
- conclusion  4.4 逐条对照与可信度结论（固定代码分档 + agent 起草总结）

与模块三同口径：指标名复用 backend/app/contracts.py::CANONICAL_METRICS。
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import PAPERS_DIR, PROJECT_ROOT
from app.ids import fs_name, safe_id
from app.services import (
    agent_service, analysis_service, env_manager, knowledge_service, proc_util,
    project_manager, task_manager,
)

TASK_PARSE = "pdf_parse"
TASK_EXTRACT = "extract_items"
TASK_REPRODUCE = "reproduce"
TASK_CONCLUSION = "conclusion"

PDF_SCRIPT = PROJECT_ROOT / "scripts" / "pdf_to_markdown.py"
REPRODUCE_TEMPLATE = PROJECT_ROOT / "scripts" / "reproduce_template.py"
REPRODUCE_TIMEOUT_S = 3600
REPRODUCE_MAX_PARALLEL = int(os.getenv("REPRODUCE_MAX_PARALLEL", "3"))  # 4.3 多条目在资源限额内并行
MAX_EXTRACT_CHARS = 100_000  # 4.2 抽取喂给 agent 的 markdown 上限（真实论文结果表在下游，过小会切掉报告值）

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
    d = PAPERS_DIR / fs_name(paper_id, "paper_id")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_paper(paper_id: str) -> dict:
    paper = knowledge_service.get_item("paper", paper_id)
    if paper is None:
        raise LookupError(f"paper not found: {paper_id}")
    return paper


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


# --------------------------- 4.1 PDF → markdown ---------------------------

def parse(paper_id: str) -> str:
    _get_paper(paper_id)
    return task_manager.create_task(TASK_PARSE, params={"paper_id": paper_id})


async def _fix_markdown(md_path: Path, index: dict) -> None:
    """agent 修正表格/公式/图注（D5：规则为主、agent 为辅），原地编辑 markdown。"""
    prompt = (
        f"任务：修正一个 markdown 文件中表格/公式/图注的解析错误。\n"
        f"目标文件：{md_path}\n"
        f"文件同目录的 section_index.json 是位置索引（表格 {len(index['tables'])} 个、"
        f"公式 {len(index['equations'])} 个、图注 {len(index['captions'])} 个）。\n\n"
        "要求：直接 Read 该 markdown，用 Edit 就地修正明显错误，保持章节标题层级不变；"
        "不要浏览仓库、不要修改其他文件、不要重写正文或臆造内容。"
        "完成后回复“已修正”。"
    )
    await agent_service.run_sync(
        prompt, cwd=str(md_path.parent), add_dirs=[str(md_path.parent)], max_turns=10, timeout_s=180
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

    rc, out = await proc_util.run_command(
        [sys.executable, str(PDF_SCRIPT), str(pdf_path), str(md_path), str(idx_path), str(tables_path)],
    )
    if rc != 0:
        raise RuntimeError(f"PDF 解析脚本失败（rc={rc}）: {out[-1500:]}")
    index = json.loads(idx_path.read_text(encoding="utf-8"))

    agent_fix = "skipped"
    if index.get("quality") == "scanned":
        task_manager.update_progress(task_id, {"quality": "scanned", "hint": "扫描版无文本层，建议 OCR（预留）"})
    else:
        # agent 修正为辅（D5）：失败不拖垮规则产出，保留规则 markdown 与索引
        try:
            await _fix_markdown(md_path, index)
            agent_fix = "applied"
        except Exception as e:  # noqa: BLE001
            agent_fix = f"failed: {e}"
        # 不论是否修正，都从当前 markdown 重建索引（无分页信息）
        rc, out = await proc_util.run_command(
            [sys.executable, str(PDF_SCRIPT), "--index-only", str(md_path), str(idx_path)],
        )
        if rc != 0:
            raise RuntimeError(f"索引重建失败（rc={rc}）: {out[-800:]}")
        index = json.loads(idx_path.read_text(encoding="utf-8"))

    knowledge_service.update_paper(
        paper_id, status="parsed", markdown_path=str(md_path), section_index=index
    )
    task_manager.update_progress(task_id, {
        "markdown_path": str(md_path),
        "sections": len(index["sections"]),
        "quality": index["quality"],
        "agent_fix": agent_fix,
    })


# --------------------------- 4.2 实验条目抽取 ---------------------------

def extract_items(paper_id: str) -> str:
    _get_paper(paper_id)
    return task_manager.create_task(TASK_EXTRACT, params={"paper_id": paper_id})


async def _run_extract(params: dict, task_id: str) -> None:
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

    # 报告值多在下游的结果表格中，提示其位置以免 agent 只读摘要/方法（4.2 六要素含论文报告值）
    table_caps = [c.get("text") for c in index.get("captions", []) if c.get("kind") == "table"]
    body = markdown if len(markdown) <= MAX_EXTRACT_CHARS else markdown[:MAX_EXTRACT_CHARS]
    truncated = "" if len(markdown) <= MAX_EXTRACT_CHARS else f"（已截断，全文 {len(markdown)} 字符）"
    hint = f"\n提示：含论文报告值的表格标题有：{table_caps}\n" if table_caps else ""
    tables_txt = _render_tables(Path(md_path).parent / "tables.json")
    tables_hint = (
        "\n参考：以下是从 PDF 确定性抽取的表格网格（可能与相邻表合并或列错位，**仅用于核对数字**，"
        "以 markdown 正文为准）：\n" + tables_txt + "\n"
        if tables_txt else ""
    )

    prompt = (
        "请从以下论文 markdown 的方法与实验章节中抽取实验条目，每条包含六要素："
        "数据集(dataset_name)、划分方式(split_method)、评价指标(metric_name)、"
        "论文报告值(metric_value_reported)、超参数(hyperparams)、对比基线(baselines)。\n"
        "**论文报告值必须取自结果表格中的实际数字**（如 “Table 3 …” 的表格行），"
        "不要因为数字在表格里就标为“未报告”；确实缺失的项才标注为“未报告”，**不要臆造**。\n"
        "注意：markdown 表格单元格内的 `<br>` 表示原表该格内换行（常对应另一行数据），"
        "解析时需按列语义区分，不要把两行数字当成一个值。\n"
        '严格按照如下 JSON 结构输出（顶层键必须是 "items"）：\n'
        '{"items": [{"section_ref":"","dataset_name":"","split_method":"","metric_name":"",'
        '"metric_value_reported":"","metric_unit":"","hyperparams":{},"baselines":[]}]}\n'
        f"{hint}{tables_hint}\n"
        f"论文 markdown{truncated}：\n{body}"
    )
    result = await agent_service.run_sync(
        prompt, cwd=str(Path(md_path).parent), output_schema=ITEMS_SCHEMA, max_turns=20
    )
    items = _dedup_items(_as_items(result.get("structured_output")))
    item_ids = knowledge_service.record_experiment_items(paper_id, items)

    knowledge_service.update_paper(paper_id, status="extracted")
    task_manager.update_progress(task_id, {"count": len(item_ids), "item_ids": item_ids})


# --------------------------- 4.3 自动复现执行 ---------------------------

def reproduce(paper_id: str, project_id: str) -> str:
    _get_paper(paper_id)
    return task_manager.create_task(
        TASK_REPRODUCE, project_id=project_id, params={"paper_id": paper_id, "project_id": project_id}
    )


async def _run_item(python: str, script: Path, source: Path, item_json: Path, out_json: Path, cwd: Path) -> dict:
    """跑一条复现；走 proc_util（取消/超时即杀进程树）。"""
    try:
        rc, log = await proc_util.run_command(
            [python, str(script), str(source), str(item_json), str(out_json)],
            cwd=str(cwd), timeout=REPRODUCE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"复现运行超时（>{REPRODUCE_TIMEOUT_S}s，已终止进程树）", "log": ""}
    error = None if rc == 0 else (log[-2000:] or f"退出码 {rc}")
    return {"ok": rc == 0, "error": error, "log": log}


async def _fill_reproduce_script(source: Path, run_dir: Path, items: list[dict]) -> None:
    """agent 依据项目代码填写复现脚本的 reproduce()（4.3 步骤 1：固定模板 + agent 填空）。"""
    items_brief = [
        {k: it.get(k) for k in ("item_id", "dataset_name", "split_method", "metric_name",
                                "metric_value_reported", "metric_unit", "hyperparams", "baselines")}
        for it in items
    ]
    prompt = (
        f"复现脚本模板已放在 {run_dir / 'reproduce.py'}（函数 reproduce(source_dir, item) 待实现）。\n"
        f"论文对应的代码仓库在 {source}，实验条目如下：\n{json.dumps(items_brief, ensure_ascii=False)}\n\n"
        "请阅读仓库代码，**原地编辑**该脚本，实现 reproduce()：按条目的数据集、划分方式、"
        "超参数执行评测，返回 {metric_name, metric_value_actual, n_samples, log_tail}。"
        "不要修改模板的 main() 与输出字段名；无法复现时让脚本非 0 退出即可。完成后回复“已实现”。"
    )
    await agent_service.run_sync(
        prompt, cwd=str(source), add_dirs=[str(run_dir), str(run_dir.parent)], max_turns=40
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
    try:
        await _fill_reproduce_script(source, run_dir, items)
    except Exception as e:  # noqa: BLE001
        task_manager.update_progress(task_id, {"script_fill": f"failed: {e}"})

    sem = asyncio.Semaphore(REPRODUCE_MAX_PARALLEL)

    async def _one(it: dict) -> dict:
        """单条目的复现：运行→落 run_record/metrics/日志→写对照结果。失败不抛出（归无法复现）。"""
        async with sem:
            item_json = run_dir / f"item_{it['item_id']}.json"
            out_json = run_dir / f"out_{it['item_id']}.json"
            item_json.write_text(json.dumps(it, ensure_ascii=False), encoding="utf-8")

            started = _now()
            run = await _run_item(python, script, source, item_json, out_json, run_dir)

            actual = None
            if run["ok"] and out_json.exists():
                try:
                    actual = json.loads(out_json.read_text(encoding="utf-8")).get("metric_value_actual")
                except (json.JSONDecodeError, OSError):
                    actual = None
            log_path = run_dir / f"log_{it['item_id']}.txt"
            log_path.write_text(run.get("log") or "", encoding="utf-8")

            run_id = knowledge_service.record_run({
                "project_id": project_id, "task_id": task_id, "run_type": "reproduce",
                "environment": environment,
                "command": f"{python} reproduce.py <source> item_{it['item_id']}.json out_{it['item_id']}.json",
                "params": {"item_id": it["item_id"], "metric_name": it.get("metric_name")},
                "status": "success" if actual is not None else "failed",
                "metrics": {it.get("metric_name") or "metric": actual} if actual is not None else None,
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
            return {"item_id": it["item_id"], "metric_name": it.get("metric_name"),
                    "metric_value_actual": actual, "ok": actual is not None}

    gathered = await asyncio.gather(*[_one(it) for it in items], return_exceptions=True)
    summary = [r for r in gathered if isinstance(r, dict)]

    knowledge_service.update_paper(paper_id, status="reproduced")
    task_manager.update_progress(task_id, {"count": len(items), "results": summary})


# --------------------------- 4.4 逐条对照与可信度结论 ---------------------------

def conclusion(paper_id: str) -> str:
    _get_paper(paper_id)
    return task_manager.create_task(TASK_CONCLUSION, params={"paper_id": paper_id})


_PERCENT_UNITS = {"%", "percent", "percentage", "percent improvement", "% improvement", "百分数", "百分比"}


def _align_unit(a: float, r: float, unit: Optional[str]) -> tuple[float, float]:
    """百分比单位下的量纲归一：一侧是小数(0~1)、另一侧是百分数(>1)时统一到百分数。

    否则「报告 91.5% vs 实测 0.915」会被算成 99% 偏差，恒判「不一致」。
    """
    u = (unit or "").strip().lower()
    if u in _PERCENT_UNITS:
        if 0 < r <= 1 < a:
            r *= 100.0
        elif 0 < a <= 1 < r:
            a *= 100.0
    return a, r


def _classify(actual, reported, unit: Optional[str] = None) -> tuple[Optional[float], int, str]:
    """相对误差分档（D6）：返回 (deviation, passed_threshold, verdict)。

    报告值缺失/非数值（如「未报告」「>50% improvement」）或为 0 → **无法复现**：
    无从比较，不能算作「不一致」（原先一律判不一致，会污染总体可信度）。
    """
    if actual is None:
        return None, 0, VERDICT_UNREPRODUCIBLE
    a, r = _to_float(actual), _to_float(reported)
    if a is None or r is None or r == 0:
        return None, 0, VERDICT_UNREPRODUCIBLE
    a, r = _align_unit(a, r, unit)
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
    }


def register() -> None:
    task_manager.register_handler(TASK_PARSE, _run_parse)
    task_manager.register_handler(TASK_EXTRACT, _run_extract)
    task_manager.register_handler(TASK_REPRODUCE, _run_reproduce)
    task_manager.register_handler(TASK_CONCLUSION, _run_conclusion)
