"""模块三 5.4 结果对比与使用建议（模块详细设计 5.4）。

对比表为固定代码生成；差异分析与使用建议由 agent 起草，落库为 usage_guidance
蒸馏知识（status=draft），经用户确认（POST /api/knowledge/confirm/{id}）后生效。

跨数据集的评估运行（run_type=eval）在本服务内触发：5.2 的 eval 入口契约对
「自带数据」与「已对齐的公开数据」同样适用，文档未规定该运行的归属，故并入 5.4。
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from app.contracts import ordered_metrics
from app.services import agent_service, baseline_service, knowledge_service, project_manager, task_manager

TASK_TYPE = "compare"

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "一句话总体结论"},
        "differences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string", "description": "对比维度，如某数据集/某指标"},
                    "detail": {"type": "string", "description": "差异表现与可能来源"},
                },
                "required": ["aspect", "detail"],
            },
        },
        "guidance_title": {"type": "string"},
        "guidance_content": {"type": "string", "description": "使用建议正文"},
    },
    "required": ["summary", "guidance_content"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_compare(project_id: str) -> str:
    project_manager.require_type(project_id, {"original"})
    return task_manager.create_task(TASK_TYPE, project_id=project_id, params={"project_id": project_id})


def _load_json(raw):
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_alignment(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _aligned_datasets(project_id: str, self_dataset_id: str | None) -> tuple[list[dict], list[str]]:
    """已对接到本项目且**已确认**的数据集，排除自带数据本身。

    返回 (可用数据集, 已对齐但未确认的数据集名)——后者用于给出可操作的报错提示。
    """
    confirmed: list[dict] = []
    unconfirmed: list[str] = []
    for ds in knowledge_service.find_datasets(limit=200):
        if ds["dataset_id"] == self_dataset_id:
            continue
        alignment = _parse_alignment(ds.get("alignment"))
        if project_id not in (alignment.get("aligned_projects") or []):
            continue
        name = ds.get("name") or ds["dataset_id"]
        if alignment.get("status") == "confirmed":
            confirmed.append(ds)
        else:
            unconfirmed.append(name)
    return confirmed, unconfirmed


def _data_dir_of(dataset: dict) -> Path | None:
    """数据集本地数据目录（预处理产物的 local_path 指向 preprocessed.csv）。"""
    raw = dataset.get("local_path")
    if not raw:
        return None
    path = Path(raw)
    return path.parent if path.name == "preprocessed.csv" else path


def _build_table(baseline_metrics: dict, evals: list[dict]) -> dict:
    keys: set[str] = set(baseline_metrics or {})
    for item in evals:
        keys |= set(item["metrics"] or {})
    ordered = ordered_metrics(keys)
    rows = []
    for metric in ordered:
        row = {"metric": metric, "baseline": (baseline_metrics or {}).get(metric)}
        for item in evals:
            row[item["name"]] = (item["metrics"] or {}).get(metric)
        rows.append(row)
    return {
        "metrics": ordered,
        "columns": ["baseline"] + [item["name"] for item in evals],
        "rows": rows,
    }


async def _draft_guidance(table: dict, evals: list[dict], task_type: str | None) -> dict:
    prompt = (
        "一个深度学习模型在自带数据上的基准结果与在若干公开数据集上的评估结果如下，"
        "请对比并给出使用建议。\n\n"
        f"任务类型: {task_type or '未知'}\n"
        f"对比表(JSON): {json.dumps(table, ensure_ascii=False)}\n"
        f"各数据集标签体系(JSON): "
        f"{json.dumps({i['name']: {'labels': i.get('labels'), 'alignment': i.get('alignment')} for i in evals}, ensure_ascii=False)}\n\n"
        "请说明：哪类数据表现更好或更差、差异可能来自数据分布或输入适配等什么原因，"
        "以及该模型适合用在什么数据上、使用前需要注意什么。"
    )
    result = await agent_service.run_sync(prompt, output_schema=COMPARE_SCHEMA)
    return result.get("structured_output") or {}


async def _run(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    ws = Path(project["workspace_path"])

    baseline = knowledge_service.get_latest_run(project_id, "baseline")
    if baseline is None:
        raise RuntimeError("未找到基准运行记录，请先调用 POST /api/projects/{id}/baseline")

    self_ds = knowledge_service.find_datasets(local_path_prefix=str(ws / "data"), limit=1)
    self_dataset_id = self_ds[0]["dataset_id"] if self_ds else None
    aligned, unconfirmed = _aligned_datasets(project_id, self_dataset_id)
    if not aligned:
        if unconfirmed:
            raise RuntimeError(
                f"以下数据集已对齐但尚未确认，无法用于对比：{', '.join(unconfirmed)}。"
                "请先调用 POST /api/datasets/{dataset_id}/alignment/confirm 确认对齐规则"
            )
        raise RuntimeError(
            "没有已对齐的数据集，无法对比。请先调用 GET /api/datasets/search "
            "并 POST /api/projects/{id}/datasets/align"
        )

    # 跨数据集评估运行（run_type=eval），复用 5.2 的 eval 入口契约
    evals = []
    for ds in aligned:
        data_dir = _data_dir_of(ds)
        if data_dir is None or not data_dir.exists():
            continue
        run = await baseline_service.run_eval(
            project_id, task_id, data_dir, run_type="eval",
            dataset_label=ds.get("name") or ds["dataset_id"],
        )
        evals.append({
            "name": ds.get("name") or ds["dataset_id"],
            "dataset_id": ds["dataset_id"],
            "labels": ds.get("labels"),
            "alignment": _parse_alignment(ds.get("alignment")),
            "run_id": run["run_id"],
            "metrics": run["metrics"],
        })
    if not evals:
        raise RuntimeError("已对齐的数据集没有可用的本地数据目录，无法评估")

    baseline_metrics = (_load_json(baseline.get("metrics")) or {})
    baseline_params = _load_json(baseline.get("params")) or {}
    table = _build_table(baseline_metrics, evals)

    common = set(baseline_metrics) & {m for item in evals for m in (item["metrics"] or {})}
    if not common:
        raise RuntimeError(
            "基准与跨数据集评估没有共同指标，结果不可比；请先统一指标口径（eval 入口输出规范指标名）"
        )

    draft = await _draft_guidance(table, evals, baseline_params.get("task_type"))

    knowledge_id = knowledge_service.record_knowledge({
        "type": "usage_guidance",
        "title": draft.get("guidance_title") or f"使用建议（{project.get('name') or project_id}）",
        "content": draft.get("guidance_content"),
        "structured": {
            "summary": draft.get("summary"),
            "differences": draft.get("differences") or [],
            "comparison": table,
            "common_metrics": sorted(common),
        },
        "sources": [baseline["run_id"]] + [item["run_id"] for item in evals],
        "confidence": "medium",
        "scope": {"task_type": baseline_params.get("task_type")},
        "status": "draft",
    })
    task_manager.update_progress(task_id, {"knowledge_id": knowledge_id, "comparison": table})


def register() -> None:
    task_manager.register_handler(TASK_TYPE, _run)
