"""画布网络：导出与训练运行（阶段4 4c，模块详细设计 7.5）。

训练链路：画布图（graph.json）→ network_export 再生成代码 → model.py + train.py
模板写入结构化项目工作区 runs/<task_id>/ → 经 proc_util 在**项目独立环境**执行 →
指标 JSON → run_record（run_type=train）落库。任务经 task_manager 排队轮询。

目标环境：默认复用父原始项目的独立环境（拆解生成的结构化项目），也可显式指定
其它 original 项目的环境（画布新建的网络没有父项目，必须指定）。环境未就绪一律
报错引导先走模块一建环境——不静默退回宿主解释器（2.3 独立环境原则）。
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.services import (
    analysis_service, knowledge_service, network_export, proc_util,
    project_manager, task_manager,
)

TASK_TYPE = "network_train"
TRAIN_TIMEOUT_S = 1800  # 训练脚本超时（任务级上限见 task_manager.TASK_TIMEOUTS）


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ws(project: dict) -> Path:
    return Path(project["workspace_path"])


def _require_network(project_id: str) -> dict:
    try:
        return project_manager.require_type(project_id, {"structured"})
    except LookupError as e:
        raise LookupError(f"network not found: {e}") from e
    except PermissionError as e:
        raise PermissionError(f"不是结构化项目（画布网络）：{e}") from e


def _graph(project: dict) -> dict:
    p = _ws(project) / "graph.json"
    if not p.exists():
        raise LookupError("graph not found（画布尚未保存过图）")
    return json.loads(p.read_text(encoding="utf-8"))


def export_network(project_id: str) -> str:
    """导出画布网络的再生成代码（与训练共用同一引擎：导出即所训）。"""
    project = _require_network(project_id)
    try:
        return network_export.generate(_graph(project))
    except network_export.ExportError as e:
        raise ValueError(str(e)) from e


def list_runs(project_id: str) -> list[dict]:
    """该网络的训练运行记录（成功记录，指标面板数据源）。"""
    _require_network(project_id)
    return knowledge_service.list_runs(project_id, "train")


def run_options(project_id: str) -> dict:
    """运行面板初始化数据：父项目、可用环境（original 项目已建好的独立环境）、
    可训练数据集（模块三预处理产物仍在的注册表条目）。"""
    project = _require_network(project_id)
    parent_id = project.get("parent_project_id")
    parent = project_manager.get_project(parent_id) if parent_id else None

    environments: list[dict] = []
    for p in project_manager.list_projects("original"):
        ws = Path(p["workspace_path"]) if p.get("workspace_path") else None
        python = analysis_service._project_python(ws) if ws else None
        if python is not None:
            environments.append({
                "project_id": p["project_id"],
                "name": p.get("name") or p["project_id"],
                "python": python,
            })

    datasets: list[dict] = []
    for ds in knowledge_service.find_datasets(limit=200):
        local = ds.get("local_path")
        if local and Path(local).exists():
            datasets.append({
                "dataset_id": ds.get("dataset_id"),
                "name": ds.get("name"),
                "task_type": ds.get("task_type"),
                "local_path": local,
            })

    return {
        "parent_project_id": parent_id,
        "parent_name": parent.get("name") if parent else None,
        "environments": environments,
        "datasets": datasets,
    }


def start_run(project_id: str, body: dict) -> str:
    """发起训练：校验数据集/环境就绪后创建任务入队。"""
    project = _require_network(project_id)

    dataset_id = body.get("dataset_id")
    dataset = knowledge_service.get_item("dataset", dataset_id) if dataset_id else None
    if dataset is None:
        raise ValueError("数据集不存在：请先在模块三下载/注册数据集")
    local = dataset.get("local_path")
    if not local or not (Path(local).parent / "preprocessed.csv").exists():
        raise ValueError(
            f"数据集 {dataset.get('name') or dataset_id} 尚无本地产物：请先在模块三完成预处理"
        )

    epochs = int(body.get("epochs", 5))
    batch_size = int(body.get("batch_size", 32))
    learning_rate = float(body.get("learning_rate", 0.001))
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs 需在 1～1000 之间")
    if not 1 <= batch_size <= 4096:
        raise ValueError("batch_size 需在 1～4096 之间")
    if not 0 < learning_rate <= 10:
        raise ValueError("learning_rate 需在 (0, 10] 之间")

    env_project_id = body.get("environment_project_id") or project.get("parent_project_id")
    if not env_project_id:
        raise ValueError(
            "该网络没有父原始项目环境可复用（画布新建）：请在运行面板指定 environment_project_id，"
            "或先从拆解生成的结构化项目出发"
        )
    env_project = project_manager.get_project(env_project_id)
    if env_project is None:
        raise ValueError(f"指定的环境项目 {env_project_id} 不存在")
    if analysis_service._project_python(_ws(env_project)) is None:
        raise ValueError(
            f"环境未就绪：项目 {env_project_id} 还没有独立运行环境——"
            f"请先走模块一建环境（POST /api/projects/{env_project_id}/env），完成后再发起训练"
        )

    params = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "environment_project_id": env_project_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    return task_manager.create_task(TASK_TYPE, project_id=project_id, params=params)


async def _run_train(params: dict, task_id: str) -> None:
    """network_train 任务：导出 → 落盘 → 项目环境执行 → 指标与运行记录落库。"""
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    if project is None:
        raise RuntimeError(f"network {project_id} not found")
    ws = _ws(project)
    started = _now()

    task_manager.update_progress(task_id, {"stage": "导出代码"})
    code = network_export.generate(_graph(project))

    dataset = knowledge_service.get_item("dataset", params["dataset_id"])
    if dataset is None or not dataset.get("local_path"):
        raise RuntimeError("数据集无本地产物，无法训练")
    data_dir = Path(dataset["local_path"]).parent
    if not (data_dir / "preprocessed.csv").exists():
        raise RuntimeError(f"数据集预处理产物缺失：{data_dir / 'preprocessed.csv'}")

    env_project_id = params["environment_project_id"]
    env_project = project_manager.get_project(env_project_id)
    if env_project is None:
        raise RuntimeError(f"环境项目 {env_project_id} 不存在")
    python = analysis_service._project_python(_ws(env_project))
    if python is None:
        raise RuntimeError(
            f"环境未就绪：项目 {env_project_id} 还没有独立运行环境——"
            f"请先走模块一建环境（POST /api/projects/{env_project_id}/env），完成后再发起训练"
        )

    # 落盘：model.py（再生成代码）+ train.py（训练入口模板）
    run_dir = ws / "runs" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.py").write_text(code, encoding="utf-8")
    template = Path(__file__).resolve().parents[2] / "templates" / "train.py"
    shutil.copyfile(template, run_dir / "train.py")

    out_json = run_dir / "train_metrics.json"
    log_path = run_dir / "train.log"
    cmd = [
        python, "train.py", str(data_dir),
        str(params["epochs"]), str(params["batch_size"]),
        f"{params['learning_rate']:g}", str(out_json),
    ]
    task_manager.update_progress(task_id, {"stage": "训练中", "command": " ".join(cmd)})
    rc, output = await proc_util.run_command(cmd, cwd=str(run_dir), timeout=TRAIN_TIMEOUT_S)
    log_path.write_text(output, encoding="utf-8", errors="replace")
    if rc != 0:
        raise RuntimeError(f"训练脚本失败（退出码 {rc}）：{output[-800:]}")

    if not out_json.exists():
        raise RuntimeError("训练脚本未产出指标文件（脚本退出码为 0 但无输出）")
    result = json.loads(out_json.read_text(encoding="utf-8"))
    metrics = result.get("metrics") if isinstance(result, dict) else None
    if not isinstance(metrics, dict):
        raise RuntimeError("训练指标 JSON 缺少 metrics 字典")

    knowledge_service.record_run({
        "project_id": project_id,
        "task_id": task_id,
        "run_type": "train",
        "environment": {"python": python, "environment_project_id": env_project_id},
        "params": {
            "dataset_id": params["dataset_id"],
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
        },
        "command": " ".join(cmd),
        "status": "success",
        "metrics": metrics,
        "artifact_path": str(out_json),
        "log_path": str(log_path),
        "started_at": started,
        "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {"stage": "完成", "metrics": metrics})


def register() -> None:
    """注册训练任务 handler（main lifespan 调用）。"""
    task_manager.register_handler(TASK_TYPE, _run_train)
