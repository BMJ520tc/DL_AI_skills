"""模块三 5.2 自带数据基准运行（模块详细设计 5.2）。

流程：定位权重 → 预处理自带数据（5.1）→ 在模块一环境运行 eval 入口 → 指标落库。
本服务为固定代码，不引入 agent；eval 入口契约为「项目 source/eval_entry.py」。

跨数据集评估（5.4 的输入）复用同一入口与执行逻辑，仅 run_type 取 eval。
"""
import asyncio
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.config import PROJECT_ROOT
from app.contracts import normalize_metrics
from app.services import proc_util
from app.services import analysis_service, knowledge_service, project_manager, task_manager

TASK_TYPE = "baseline"
EVAL_ENTRY_NAME = "eval_entry.py"
EVAL_TEMPLATE = PROJECT_ROOT / "scripts" / "eval_entry_template.py"
BASELINE_TIMEOUT_S = 3600
WEIGHT_EXTS = (".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".h5")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_baseline(project_id: str) -> str:
    project_manager.require_type(project_id, {"original"})
    return task_manager.create_task(TASK_TYPE, project_id=project_id, params={"project_id": project_id})


def get_latest_baseline(project_id: str) -> dict | None:
    return knowledge_service.get_latest_run(project_id, "baseline")


WEIGHT_PATH_RE = re.compile(r"[\w./\\-]+\.(?:" + "|".join(e.lstrip(".") for e in WEIGHT_EXTS) + r")\b")


def _find_weights(source: Path, ws: Path) -> list[str]:
    """定位权重（5.2 步骤 1），经结构分析报告辅助。返回命中文件的父目录去重列表。

    先扫项目文件；再用模块一的结构分析报告（reports/structure_report.json）里
    提到的权重路径做补充——权重常不在仓库内，但项目代码/配置会写明其路径。
    """
    found: list[str] = []
    for p in source.rglob("*"):
        if p.is_file() and p.suffix.lower() in WEIGHT_EXTS:
            parent = str(p.parent)
            if parent not in found:
                found.append(parent)

    report_path = ws / "reports" / "structure_report.json"
    if report_path.exists():
        try:
            mentioned = set(WEIGHT_PATH_RE.findall(report_path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            mentioned = set()
        for raw in mentioned:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = source / candidate
            candidate = candidate.resolve()  # 规范化 ../ 等相对片段，路径便于阅读与复用
            if candidate.exists():
                parent = str(candidate.parent)
                if parent not in found:
                    found.append(parent)
    return found


def _find_data_dir(ws: Path) -> Path | None:
    """预处理产物目录（5.1 输出）：优先 data/self，其次任意 data/*/。"""
    root = ws / "data"
    if (root / "self" / "preprocessed.csv").exists():
        return root / "self"
    for candidate in sorted(root.glob("*/preprocessed.csv")):
        return candidate.parent
    return None


def _record(project_id: str, task_id: str, run_type: str, run: dict) -> str:
    return knowledge_service.record_run({
        "project_id": project_id,
        "task_id": task_id,
        "run_type": run_type,
        "environment": run.get("environment"),
        "params": run.get("params"),
        "command": run.get("command"),
        "status": run.get("status", "failed"),
        "metrics": run.get("metrics"),
        "error": run.get("error"),
        "artifact_path": run.get("artifact_path"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
    })


def _fail(project_id: str, task_id: str, run_type: str, started: str, error: str) -> None:
    _record(project_id, task_id, run_type,
            {"status": "failed", "error": error, "started_at": started, "finished_at": _now()})


async def _run_entry(cmd: list[str], cwd: Path, timeout_s: int) -> dict:
    """跑 eval 入口；走 proc_util（取消/超时即杀进程树，不再用 to_thread+subprocess.run）。"""
    try:
        rc, out = await proc_util.run_command(cmd, cwd=str(cwd), timeout=timeout_s)
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"运行超时（>{timeout_s}s，已终止进程树）"}
    ok = rc == 0
    output = out[-2000:]
    return {"ok": ok, "error": None if ok else (output or f"退出码 {rc}")}


async def run_eval(
    project_id: str, task_id: str, data_dir: Path, run_type: str = "eval", dataset_label: str | None = None
) -> dict:
    """在项目独立环境跑 eval 入口并落库。失败抛 RuntimeError（同时已写入失败记录）。

    返回成功时的运行记录 dict（含 metrics 与 run_id）。
    """
    project = project_manager.get_project(project_id)
    ws = Path(project["workspace_path"])
    source = ws / "source"
    started = _now()

    python = analysis_service._project_python(ws)
    if python is None:
        err = "项目环境未就绪：未找到独立环境解释器，请先完成环境创建"
        _fail(project_id, task_id, run_type, started, err)
        raise RuntimeError(err)

    eval_entry = source / EVAL_ENTRY_NAME
    if not eval_entry.exists():
        scaffold = ws / "runs" / EVAL_ENTRY_NAME
        scaffold.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EVAL_TEMPLATE, scaffold)
        err = (f"项目缺少 eval 入口（{EVAL_ENTRY_NAME}），已生成模板到 {scaffold}，"
               "请复制到 source/ 并实现 evaluate()")
        _fail(project_id, task_id, run_type, started, err)
        raise RuntimeError(err)

    weight_dirs = _find_weights(source, ws)
    model_dir = weight_dirs[0] if weight_dirs else None
    out_json = ws / "runs" / task_id / f"{run_type}_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [python, str(eval_entry), str(data_dir), model_dir or "", str(out_json)]
    result = await _run_entry(cmd, source, BASELINE_TIMEOUT_S)

    record = {
        "command": " ".join(cmd),
        "params": {"data_dir": str(data_dir), "run_type": run_type,
                   "dataset_label": dataset_label or run_type,
                   "weights_found": bool(weight_dirs), "weight_dirs": weight_dirs},
        "started_at": started,
        "finished_at": _now(),
    }
    if not result["ok"]:
        error = result["error"]
        if not weight_dirs:
            # 5.2 异常与边界：权重缺失 → 记录并提示先跑通模块一
            error = f"{error}\n[提示] 未在项目中发现模型权重（{', '.join(WEIGHT_EXTS)}），建议先跑通模块一（结构分析）以确认权重位置。"
        record.update({"status": "failed", "error": error})
        _record(project_id, task_id, run_type, record)
        raise RuntimeError(f"评估运行失败: {error}")

    if not out_json.exists():
        err = f"eval 入口未写出指标文件: {out_json}"
        record.update({"status": "failed", "error": err})
        _record(project_id, task_id, run_type, record)
        raise RuntimeError(err)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    metrics = normalize_metrics(payload.get("metrics") or payload)
    if not metrics:
        err = "eval 入口输出中没有可识别的数值指标"
        record.update({"status": "failed", "error": err})
        _record(project_id, task_id, run_type, record)
        raise RuntimeError(err)

    record["params"].update({
        "primary_metric": payload.get("primary_metric"),
        "n_samples": payload.get("n_samples"),
        "task_type": payload.get("task_type"),
        "model": payload.get("model"),
        "dataset": payload.get("dataset"),
    })
    record.update({"status": "success", "metrics": metrics, "artifact_path": str(out_json)})
    run_id = _record(project_id, task_id, run_type, record)
    return {"run_id": run_id, **record}


async def _run(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    ws = Path(project["workspace_path"])

    data_dir = _find_data_dir(ws)
    if data_dir is None:
        err = "未找到预处理后的自带数据，请先调用 POST /api/preprocess"
        _fail(project_id, task_id, "baseline", _now(), err)
        raise RuntimeError(err)

    await run_eval(project_id, task_id, data_dir, run_type="baseline")


def register() -> None:
    task_manager.register_handler(TASK_TYPE, _run)
