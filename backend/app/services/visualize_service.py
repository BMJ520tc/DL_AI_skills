"""模块三 5.5 可视化脚本调度（模块详细设计 5.5，D4/D7）。

同步执行（不入队）：读取基准与跨数据集评估的产物 → 组装图数据 JSON
→ 调用固定脚本生成自包含 HTML → 返回文件路径。

误差分布图与典型案例图覆盖**本次对比中参与评估的全部数据集**（baseline + 各跨数据集
评估结果），每张图按数据集分组并标注来源；某数据集缺 `predictions`（老 eval 实现只给
聚合指标）时，该数据集在图上**如实标注为不可绘**并继续画其余数据集，不用代理值冒充它的
真实误差。`per_class` 有值时会参与分组（误差按类别汇总、典型案例按类别轮转挑选）并落进产物。
某张图整体无可用数据时由脚本内降级提示，不影响其他图。
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


def _artifact(run: dict) -> tuple[dict, Path | None]:
    """读一次运行的指标 JSON 产物；返回 (payload, 该数据集的数据目录)。

    数据目录取自 params.data_dir（跨数据集评估时是**对齐副本目录**），
    用于把 predictions 里的相对路径解析成可读取的缩略图路径。
    """
    artifact = run.get("artifact_path")
    if not artifact or not Path(artifact).exists():
        return {}, None
    try:
        payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, None
    params = _load(run.get("params")) or {}
    data_dir = Path(params["data_dir"]) if params.get("data_dir") else None
    if not isinstance(payload, dict):
        return {}, data_dir
    return payload, data_dir


def _dataset_runs(baseline: dict | None, evals: list[dict]) -> list[tuple[str, dict]]:
    """本次对比参与评估的数据集：(显示名, 运行记录)，baseline 在前、跨数据集评估在后。"""
    out: list[tuple[str, dict]] = []
    if baseline is not None:
        params = _load(baseline.get("params")) or {}
        out.append((str(params.get("dataset_label") or "baseline"), baseline))
    for run in evals:
        params = _load(run.get("params")) or {}
        out.append((str(params.get("dataset_label") or run["run_id"][:8]), run))
    return out


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


def _predictions_of(payload: dict) -> list[dict]:
    predictions = payload.get("predictions") or []
    return [p for p in predictions if isinstance(p, dict)]


def _per_class_of(payload: dict) -> dict:
    per_class = payload.get("per_class")
    return per_class if isinstance(per_class, dict) else {}


def _error_value(prediction: dict, numeric: bool) -> float | None:
    """单样本误差：回归取 |y_true - y_pred|；分类无逐样本真实误差，降级为 1 - 置信度代理。"""
    if numeric:
        if _is_number(prediction.get("y_true")) and _is_number(prediction.get("y_pred")):
            return abs(float(prediction["y_true"]) - float(prediction["y_pred"]))
        return None
    if _is_number(prediction.get("prob")):
        return round(1 - float(prediction["prob"]), 4)
    return None


def _class_breakdown(predictions: list[dict], per_class: dict, numeric: bool) -> list[dict]:
    """按类别汇总误差（并用上 per_class 指标）：误差分布图的数据集内分组依据。"""
    groups: dict[str, list[float]] = {}
    for prediction in predictions:
        error = _error_value(prediction, numeric)
        if error is None:
            continue
        groups.setdefault(str(prediction.get("y_true")), []).append(error)
    labels = list(groups)
    for cls in per_class:
        if str(cls) not in groups:
            labels.append(str(cls))
    rows = []
    for cls in labels:
        errors = groups.get(cls, [])
        metrics = per_class.get(cls)
        metrics = metrics if isinstance(metrics, dict) else {}
        rows.append({
            "label": cls,
            "n": len(errors),
            "mean_error": round(sum(errors) / len(errors), 6) if errors else None,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        })
    return rows


def _error_dist_datasets(baseline: dict | None, evals: list[dict]) -> list[dict]:
    """误差分布图数据：每个参与评估的数据集一项；缺 predictions 的数据集如实标为不可绘。"""
    datasets: list[dict] = []
    for name, run in _dataset_runs(baseline, evals):
        payload, _ = _artifact(run)
        predictions = _predictions_of(payload)
        per_class = _per_class_of(payload)
        if not predictions:
            datasets.append({
                "name": name, "drawable": False,
                "reason": "该数据集评估产物缺少 predictions（逐样本预测），只有聚合指标，无法绘制误差分布",
            })
            continue
        numeric = all(
            _is_number(p.get("y_true")) and _is_number(p.get("y_pred")) for p in predictions
        )
        values = [v for v in (_error_value(p, numeric) for p in predictions) if v is not None]
        if not values:
            datasets.append({
                "name": name, "drawable": False,
                "reason": "该数据集评估产物的 predictions 缺可用的 y_true/y_pred 或 prob 数值，无法绘制误差分布",
            })
            continue
        scatter = ([{"x": float(p["y_true"]), "y": float(p["y_pred"])} for p in predictions[:MAX_SCATTER]]
                   if numeric else [])
        entry = {
            "name": name, "drawable": True,
            "unit": "绝对误差" if numeric else "1 - 预测置信度",
            "values": values, "bins": 10, "scatter": scatter,
            "per_class": per_class, "proxy": not numeric,
        }
        if not numeric:
            # 代理值必须带明确标注，不得冒充真实误差
            entry["note"] = "分类任务无逐样本真实误差，以 1 - 预测置信度作为误差代理（已标注）"
        breakdown = _class_breakdown(predictions, per_class, numeric)
        if breakdown:
            entry["class_breakdown"] = breakdown
        datasets.append(entry)
    return datasets


def _resolve_path(path_value, data_dir: Path | None):
    if not path_value or data_dir is None:
        return path_value
    candidate = Path(str(path_value))
    if candidate.is_absolute():
        return str(candidate)
    merged = data_dir / candidate
    return str(merged) if merged.exists() else str(path_value)


def _pick_cases(predictions: list[dict], data_dir: Path | None, per_class: dict) -> list[dict]:
    """典型案例挑选：按类别轮转（per_class 有指标时优先这些类别），类内误判优先。"""
    grouped: dict[str, list[dict]] = {}
    for prediction in predictions:
        grouped.setdefault(str(prediction.get("y_true")), []).append(prediction)
    known_classes = {str(k) for k in per_class}
    classes = sorted(grouped, key=lambda c: (c not in known_classes, c))
    buckets: list[list[dict]] = []
    for cls in classes:
        items = grouped[cls]
        wrong = [p for p in items if p.get("y_true") != p.get("y_pred")]
        right = [p for p in items if p.get("y_true") == p.get("y_pred")]
        buckets.append(wrong + right)
    picked: list[dict] = []
    index = 0
    while len(picked) < MAX_CASES and any(index < len(bucket) for bucket in buckets):
        for bucket in buckets:
            if len(picked) >= MAX_CASES:
                break
            if index < len(bucket):
                picked.append(bucket[index])
        index += 1
    return [{
        "id": p.get("id"),
        "y_true": p.get("y_true"),
        "y_pred": p.get("y_pred"),
        "prob": p.get("prob"),
        "path": _resolve_path(p.get("path"), data_dir),
        "correct": p.get("y_true") == p.get("y_pred"),
    } for p in picked]


def _cases_datasets(baseline: dict | None, evals: list[dict]) -> list[dict]:
    """典型案例图数据：每个参与评估的数据集一项；缺 predictions 的数据集如实标为不可绘。"""
    datasets: list[dict] = []
    for name, run in _dataset_runs(baseline, evals):
        payload, data_dir = _artifact(run)
        predictions = _predictions_of(payload)
        per_class = _per_class_of(payload)
        if not predictions:
            datasets.append({
                "name": name, "drawable": False,
                "reason": "该数据集评估产物缺少 predictions（逐样本预测），只有聚合指标，无法绘制典型案例",
            })
            continue
        datasets.append({
            "name": name, "drawable": True,
            "cases": _pick_cases(predictions, data_dir, per_class),
            "per_class": per_class,
        })
    return datasets


def _payload_for(chart_type: str, project_id: str) -> dict:
    baseline, evals = _runs(project_id)
    if chart_type == "performance":
        if baseline is None:
            return {}
        return _performance_payload(baseline, evals)

    if baseline is None and not evals:
        return {}
    if chart_type == "error_dist":
        return {"title": "误差分布图（按数据集分组）",
                "datasets": _error_dist_datasets(baseline, evals)}
    if chart_type == "cases":
        return {"title": "典型案例预测结果（按数据集分组）",
                "datasets": _cases_datasets(baseline, evals)}
    raise ValueError(f"unknown chart_type: {chart_type}")


def _is_degraded(chart_type: str, payload: dict) -> bool:
    """整图降级判据：性能图缺 series；分组图里**没有任何一个数据集可绘**。"""
    if chart_type == "performance":
        return not payload.get("series")
    datasets = payload.get("datasets") or []
    return not any(d.get("drawable") for d in datasets)


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
        "degraded": _is_degraded(chart_type, payload),
    }
