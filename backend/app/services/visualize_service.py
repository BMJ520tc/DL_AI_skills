"""模块三 5.5 可视化脚本调度（模块详细设计 5.5，D4/D7）。

同步执行（不入队）：读取基准与跨数据集评估的产物 → 组装图数据 JSON
→ 调用固定脚本生成自包含 HTML → 返回文件路径。
某张图缺字段时由脚本内降级提示，不影响其他图。
"""
import asyncio
import json
import sys
from pathlib import Path

from app.config import PROJECT_ROOT
from app.services import proc_util
from app.contracts import ordered_metrics
from app.services import knowledge_service, project_manager

SCRIPTS = {
    "performance": PROJECT_ROOT / "scripts" / "visualize_performance.py",
    "error_dist": PROJECT_ROOT / "scripts" / "visualize_error_dist.py",
    "cases": PROJECT_ROOT / "scripts" / "visualize_cases.py",
}
ECHARTS_PATH = PROJECT_ROOT / "backend" / "app" / "vendor" / "echarts.min.js"

MAX_CASES = 24
MAX_SCATTER = 500

# 各图判定「降级」所依赖的必填字段
REQUIRED_FIELD = {"performance": "series", "error_dist": "values", "cases": "cases"}


def _load(raw):
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _runs(project_id: str) -> tuple[dict | None, list[dict]]:
    """取基准与跨数据集评估运行；每个数据集只保留最近一次评估（避免重复对比累积）。"""
    baseline = knowledge_service.get_latest_run(project_id, "baseline")
    latest: dict[str, dict] = {}
    for run in knowledge_service.list_runs(project_id, "eval"):  # 已按 started_at 倒序
        params = _load(run.get("params")) or {}
        label = params.get("dataset_label") or run["run_id"][:8]
        latest.setdefault(label, run)
    return baseline, list(latest.values())


def _predictions(run: dict) -> tuple[list, Path | None]:
    """从运行产物读取逐样本预测；返回 (predictions, 数据目录)。"""
    artifact = run.get("artifact_path")
    if not artifact or not Path(artifact).exists():
        return [], None
    try:
        payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], None
    params = _load(run.get("params")) or {}
    data_dir = Path(params["data_dir"]) if params.get("data_dir") else None
    return payload.get("predictions") or [], data_dir


def _performance_payload(baseline: dict, evals: list[dict]) -> dict:
    baseline_metrics = _load(baseline.get("metrics")) or {}
    series = [{"name": "baseline", "values": baseline_metrics}]
    keys = set(baseline_metrics)
    for run in evals:
        params = _load(run.get("params")) or {}
        values = _load(run.get("metrics")) or {}
        keys |= set(values)
        series.append({"name": params.get("dataset_label") or run["run_id"][:8], "values": values})
    return {"title": "性能对比图", "metrics": ordered_metrics(keys), "series": series}


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _error_dist_payload(predictions: list[dict]) -> dict:
    # 必须校验**全部**行：只抽查前 5 条时，第 6 条起的非数值会让后面的 float() 抛错打成 500
    numeric = all(_is_number(p.get("y_true")) and _is_number(p.get("y_pred")) for p in predictions) \
        and bool(predictions)
    if numeric:
        values = [abs(float(p["y_true"]) - float(p["y_pred"])) for p in predictions]
        scatter = [{"x": float(p["y_true"]), "y": float(p["y_pred"])} for p in predictions[:MAX_SCATTER]]
        unit = "绝对误差"
    else:
        # 分类场景：以「预测置信度不足」为误差代理（1 - prob）
        values = [round(1 - float(p["prob"]), 4) for p in predictions if _is_number(p.get("prob"))]
        scatter = []
        unit = "1 - 预测置信度"
    return {"title": f"误差分布图（{unit}）", "values": values, "bins": 10, "scatter": scatter}


def _cases_payload(predictions: list[dict], data_dir: Path | None) -> dict:
    def resolve(path_value):
        if not path_value or data_dir is None:
            return path_value
        candidate = Path(str(path_value))
        if candidate.is_absolute():
            return str(candidate)
        merged = data_dir / candidate
        return str(merged) if merged.exists() else str(path_value)

    wrong = [p for p in predictions if p.get("y_true") != p.get("y_pred")]
    right = [p for p in predictions if p.get("y_true") == p.get("y_pred")]
    picked = wrong[: MAX_CASES // 2] + right[: MAX_CASES // 2]
    cases = [{
        "id": p.get("id"),
        "y_true": p.get("y_true"),
        "y_pred": p.get("y_pred"),
        "prob": p.get("prob"),
        "path": resolve(p.get("path")),
        "correct": p.get("y_true") == p.get("y_pred"),
    } for p in picked]
    return {"title": "典型案例预测结果", "cases": cases}


def _payload_for(chart_type: str, project_id: str) -> dict:
    baseline, evals = _runs(project_id)
    if chart_type == "performance":
        if baseline is None:
            return {}
        return _performance_payload(baseline, evals)

    predictions, data_dir = _predictions(baseline) if baseline else ([], None)
    if chart_type == "error_dist":
        return _error_dist_payload(predictions)
    if chart_type == "cases":
        return _cases_payload(predictions, data_dir)
    raise ValueError(f"unknown chart_type: {chart_type}")


async def run(project_id: str, chart_type: str) -> dict:
    """生成一张图，返回 {chart_type, html, degraded}。"""
    project_manager.require_type(project_id, {"original"})
    if chart_type not in SCRIPTS:
        raise ValueError(f"chart_type 只能是 {sorted(SCRIPTS)}")

    project = project_manager.get_project(project_id)
    ws = Path(project["workspace_path"])
    out_dir = ws / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _payload_for(chart_type, project_id)
    input_json = out_dir / f"{chart_type}.data.json"
    input_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    html_path = out_dir / f"{chart_type}.html"
    cmd = [sys.executable, str(SCRIPTS[chart_type]), str(input_json), str(html_path), str(ECHARTS_PATH)]
    try:
        rc, out = await proc_util.run_command(cmd, timeout=300)
    except asyncio.TimeoutError:
        raise RuntimeError(f"可视化脚本超时（{chart_type}，已终止进程树）")
    if rc != 0 or not html_path.exists():
        raise RuntimeError(f"可视化脚本失败（{chart_type}）：{out[-1500:]}")

    return {
        "chart_type": chart_type,
        "html": str(html_path),
        "degraded": not payload.get(REQUIRED_FIELD[chart_type]),
    }
