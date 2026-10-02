"""模块三 5.1 数据预处理与结构统一化（模块详细设计 5.1）。

固定脚本 scripts/preprocess_dataset.py 负责格式识别、统一化与清洗；
本服务负责调度、agent 补充字段映射建议、数据集登记与 alignment 落库。
"""
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import DATASETS_DIR, PROJECT_ROOT
from app.services import proc_util
from app.services import agent_service, knowledge_service, project_manager, task_manager

TASK_TYPE = "preprocess"
PREPROCESS_SCRIPT = PROJECT_ROOT / "scripts" / "preprocess_dataset.py"

MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "mappings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "unified_field": {"type": "string", "description": "input/label/id/split 之一"},
                    "reason": {"type": "string"},
                },
                "required": ["column", "unified_field"],
            },
        },
    },
    "required": ["mappings"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_preprocess(
    input_path: str,
    dataset_name: Optional[str] = None,
    task_type: str = "classification",
    project_id: Optional[str] = None,
    source: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    if project_id:
        project_manager.require_type(project_id, {"original"})
    return task_manager.create_task(
        TASK_TYPE,
        project_id=project_id,
        params={
            "input_path": input_path,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "project_id": project_id,
            # 来源溯源（5.3）：外部下载的公开数据集经预处理后仍应保留 source/url，
            # 否则 dataset_registry 会把公开数据登记成「自带」，验收时无法追溯数据来源。
            "source": source,
            "url": url,
        },
    )


def get_result(task_id: str) -> Optional[dict]:
    """预处理结果摘要（服务写入 DATASETS_DIR/<task_id>/result.json）。"""
    path = DATASETS_DIR / task_id / "result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_dirs(params: dict, task_id: str) -> tuple[Path, Path, str]:
    """返回 (输出目录, 结果目录, 数据集名)。

    无项目时输出目录按输入路径哈希固定，使同一文件重复预处理落到同一路径，
    从而让 upsert_dataset 按 name+local_path 命中并更新，而不是每次新建一条登记。
    """
    result_dir = DATASETS_DIR / task_id
    project_id = params.get("project_id")
    name = params.get("dataset_name")

    if project_id:
        project = project_manager.require_type(project_id, {"original"})
        name = name or "self"
        return Path(project["workspace_path"]) / "data" / name, result_dir, name

    input_path = Path(params["input_path"]).resolve()
    stable_key = hashlib.sha1(str(input_path).lower().encode("utf-8")).hexdigest()[:16]
    return DATASETS_DIR / stable_key, result_dir, name or input_path.stem


async def _run_script(input_path: str, out_dir: Path, name: str, task_type: str) -> dict:
    cmd = [
        sys.executable, str(PREPROCESS_SCRIPT), input_path, str(out_dir),
        "--dataset-name", name, "--task-type", task_type,
    ]
    rc, raw = await proc_util.run_command(cmd, timeout=900)
    stdout = (raw or "").strip()
    if not stdout:
        raise RuntimeError(f"预处理脚本无输出（退出码 {rc}）：{raw[-1500:]}")
    try:
        return json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        raise RuntimeError(f"预处理脚本输出无法解析：{stdout[-1500:]}")


async def _agent_field_suggestions(source: Path, summary: dict) -> list[dict]:
    """对未自动映射的 meta_* 列，交 agent 建议应归入哪个统一字段（模块详细设计 5.1 步骤 2）。"""
    unmapped = [f for f in summary.get("fields", []) if f.startswith("meta_")]
    if not unmapped:
        return []
    prompt = (
        f"这是一个已做初步预处理的数据集，统一 schema 固定为 id/split/label/input。\n"
        f"数据集名: {summary.get('dataset_name')}，任务类型: {summary.get('task_type')}，"
        f"共 {summary.get('n_rows')} 行。\n"
        f"以下列未能自动映射到统一字段: {json.dumps(unmapped, ensure_ascii=False)}\n"
        "请阅读该项目代码判断每列的语义，给出应归入的统一字段（只能取 input/label/id/split），"
        "确实属于辅助信息、不应归入统一字段的列不要出现在结果里。"
    )
    result = await agent_service.run_sync(prompt, cwd=str(source), output_schema=MAP_SCHEMA)
    mappings = (result.get("structured_output") or {}).get("mappings") or []
    return [m for m in mappings if isinstance(m, dict) and m.get("column") in unmapped]


async def _run(params: dict, task_id: str) -> None:
    out_dir, result_dir, name = _resolve_dirs(params, task_id)
    input_path = params["input_path"]
    task_type = params.get("task_type") or "classification"

    summary = await _run_script(input_path, out_dir, name, task_type)
    if summary.get("status") != "ok":
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(summary.get("message") or f"预处理失败: {summary.get('status')}")

    # agent 建议（辅助）：仅为未映射列提供归入建议，不改动已产出的数据列
    if params.get("project_id"):
        project = project_manager.get_project(params["project_id"])
        if project:
            suggestions = await _agent_field_suggestions(Path(project["workspace_path"]) / "source", summary)
            if suggestions:
                summary["alignment"]["agent_field_suggestions"] = suggestions

    dataset_id = knowledge_service.upsert_dataset({
        "name": name,
        "url": params.get("url"),
        "source": params.get("source") or "自带",
        "task_type": task_type,
        "format": summary.get("format"),
        "fields": summary.get("fields"),
        "labels": summary.get("labels"),
        "alignment": summary.get("alignment"),
        "local_path": summary.get("output_csv"),
    })
    summary["dataset_id"] = dataset_id

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 结果摘要同步落盘到数据目录，便于与产出数据同处查看
    (out_dir / "dataset.schema.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def register() -> None:
    task_manager.register_handler(TASK_TYPE, _run)
