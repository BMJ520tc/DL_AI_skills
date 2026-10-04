"""画布网络：导出与训练运行（阶段4 4c，模块详细设计 7.5）。

训练链路：画布图（graph.json）→ network_export 再生成代码 → model.py + train.py
模板写入结构化项目工作区 runs/<task_id>/ → 经 proc_util 在**项目独立环境**执行 →
指标 JSON → run_record（run_type=train）落库。任务经 task_manager 排队轮询。

画布图分两类，均经 `network_export.generate` 这**一个**出口出码（导出即所存即所训）：
标准画布图走画布引擎；拆解 ir 图（节点 type="ir"）走模块四 IR 链路
（`graphir_to_ir` → `ir_codegen.generate`），故画布上的调参同样体现在训练所用代码里。
ir 图缺字段/结构不合法 → `ExportError`：导出端点 400，训练入口也在入队前拦成 400
（不让用户等到任务失败才看到原因）。

目标环境：默认复用父原始项目的独立环境（拆解生成的结构化项目），也可显式指定
其它 original 项目的环境（画布新建的网络没有父项目，必须指定）。环境未就绪一律
报错引导先走模块一建环境——不静默退回宿主解释器（2.3 独立环境原则）。

阶段4 4d-1：训练成功后「运行即提交」——指标摘要写进 network_version.json 并提交
（version_service.commit_run；失败不连坐训练结果，但必须透出：任务进度 version_error
字段 + run_record 失败留痕，并照旧记后端日志——不静默）。
阶段4 4d-2 补强（需求五.3「每次运行生成一个版本节点」）：训练**失败**同样提交一条
「训练失败 <task_id>」版本节点（run_summary.status="failed"，含失败原因摘要），
提交失败只记日志、不掩盖原始训练失败原因。
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.services import (
    analysis_service, knowledge_service, network_export, proc_util,
    project_manager, task_manager, version_service,
)

logger = logging.getLogger(__name__)

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
        # 只列「可训练」的条目：与 start_run 同口径（7.5「有预处理产物的注册条目」），
        # 否则面板会列出跑不了的数据集、点了才 400
        if local and (Path(local).parent / "preprocessed.csv").exists():
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

    # 拆解 ir 图：入队前先跑一次同源再生成（与导出端点同一个函数，导出即所训）。
    # 缺字段/结构不合法/IR 不完整 → 400 + 定位到节点/字段的原因；
    # 标准画布图保持既有行为（不在入队前介入，错误由任务失败透出）。
    try:
        graph = _graph(project)
    except LookupError:
        graph = None
    if graph is not None and network_export.is_ir_graph(graph):
        try:
            network_export.generate(graph)
        except network_export.ExportError as e:
            raise ValueError(str(e)) from e

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
    """network_train 任务入口：失败也落一条 run_record(status=failed)，且同样生成版本节点。

    《知识库与数据设计》五.2：每次实际执行都落一条运行记录，**结果与报错只写 run_record**——
    故训练失败必须可检索（与模块四验证任务同口径）。记录后原样抛出，任务状态由 task_manager 置 failed。
    需求五.3「每次运行生成一个版本节点」：失败运行在 run_record 之后提交一条
    「训练失败 <task_id>」版本（run_summary.status="failed"）；该提交失败只记日志，不掩盖原始失败原因。
    """
    ctx: dict = {}
    try:
        await _do_train(params, task_id, ctx)
    except Exception as exc:  # noqa: BLE001 —— 记失败记录后再抛出，不改变任务结果
        if ctx.get("recorded") == "success":
            # 训练已成功并落库，之后的收尾步骤再抛错不该再写一条 failed（同一任务两条互相矛盾）
            logger.exception("训练成功后收尾失败 project_id=%s task_id=%s", params.get("project_id"), task_id)
            raise
        try:
            knowledge_service.record_run({
                "project_id": params.get("project_id"),
                "task_id": task_id,
                "run_type": "train",
                "environment": ctx.get("environment"),
                "params": {
                    "dataset_id": params.get("dataset_id"),
                    "epochs": params.get("epochs"),
                    "batch_size": params.get("batch_size"),
                    "learning_rate": params.get("learning_rate"),
                },
                "command": ctx.get("command"),
                "status": "failed",
                "error": str(exc)[-2000:],
                "log_path": ctx.get("log_path"),
                "started_at": ctx.get("started_at") or _now(),
                "finished_at": _now(),
            })
        except Exception:  # noqa: BLE001 —— 失败记录的写入失败不得掩盖原异常
            logger.exception("训练失败记录落库失败 project_id=%s task_id=%s",
                             params.get("project_id"), task_id)
        # 需求五.3「每次运行生成一个版本节点」：失败也是一次运行，故在 run_record(failed) 之后
        # 再提交一条「训练失败 <task_id>」版本节点（失败原因前若干字符进 run_summary.error）。
        # 顺序要求：先留 run_record，再提交版本；提交本身失败只记日志，绝不掩盖原始训练失败原因。
        try:
            await asyncio.to_thread(
                version_service.commit_run, params.get("project_id"), task_id, None,
                status="failed", error=str(exc))
        except Exception:  # noqa: BLE001 —— 版本提交失败不得掩盖原始训练失败原因
            logger.exception("训练失败版本提交失败 project_id=%s task_id=%s",
                             params.get("project_id"), task_id)
        raise


def _version_error_message(ws: Path, task_id: str, exc: Exception) -> str:
    """运行版本提交失败的透出文案（命名与保存路径的响应字段 version_error 同口径）。

    commit_run 先写盘 network_version.json、再做 git add/commit，故失败有两种状态：
    文件已写盘（版本内容在、版本节点不在）与提交前即失败（文件未更新）——文案据实区分，
    使「训练结果已保留，但版本节点未生成」这件事对用户可读。
    """
    written = False
    try:
        payload = json.loads((ws / version_service.VERSION_FILENAME).read_text(encoding="utf-8"))
        summary = payload.get("run_summary") if isinstance(payload, dict) else None
        written = isinstance(summary, dict) and summary.get("task_id") == task_id
    except Exception:  # noqa: BLE001 —— 文案辅助读取失败按「未写盘」表述，不掩盖原错误
        written = False
    state = ("版本节点未生成（network_version.json 已写盘但未提交：版本内容在、版本节点不在）"
             if written else "版本节点未生成")
    return f"版本提交失败（训练结果已保留；{state}）: {exc}"


async def _do_train(params: dict, task_id: str, ctx: dict) -> None:
    """network_train 任务：导出 → 落盘 → 项目环境执行 → 指标与运行记录落库。"""
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    if project is None:
        raise RuntimeError(f"network {project_id} not found")
    ws = _ws(project)
    started = _now()
    ctx["started_at"] = started

    task_manager.update_progress(task_id, {"stage": "导出代码"})
    code = network_export.generate(_graph(project))

    dataset = knowledge_service.get_item("dataset", params["dataset_id"])
    if dataset is None or not dataset.get("local_path"):
        raise RuntimeError("数据集无本地产物，无法训练")
    data_dir = Path(dataset["local_path"]).parent
    if not (data_dir / "preprocessed.csv").exists():
        raise RuntimeError(f"数据集预处理产物缺失：{data_dir / 'preprocessed.csv'}")

    # 任务前知识带入（需求六.1、模块详细设计 8.2「模块五新模型运行」触发点）：
    # 按模型名/数据集/任务类型检索已确认蒸馏结论，作为「默认建议」进任务进度，供运行面板展示。
    try:
        advice = knowledge_service.bring_advice_summary(
            task_type=dataset.get("task_type"), model=project.get("name"), dataset=dataset.get("name"))
    except Exception:  # noqa: BLE001 —— 带入失败不影响训练
        advice = {}
    if advice:
        task_manager.update_progress(task_id, {"knowledge_bring": advice})

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
    ctx["command"] = " ".join(cmd)
    ctx["log_path"] = str(log_path)
    ctx["environment"] = {"python": python, "environment_project_id": env_project_id}
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

    run_params = {
        "dataset_id": params["dataset_id"],
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "learning_rate": params["learning_rate"],
    }
    knowledge_service.record_run({
        "project_id": project_id,
        "task_id": task_id,
        "run_type": "train",
        "environment": {"python": python, "environment_project_id": env_project_id},
        "params": run_params,
        "command": " ".join(cmd),
        "status": "success",
        "metrics": metrics,
        "artifact_path": str(out_json),
        "log_path": str(log_path),
        "started_at": started,
        "finished_at": _now(),
    })
    ctx["recorded"] = "success"  # 成功后若再抛错，不再写一条自相矛盾的 failed 记录
    # 4d-1 运行即提交：指标摘要入 network_version.json；失败不连坐训练结果，
    # 但必须透出（任务进度 version_error + run_record 失败留痕），不静默。
    version_error: str | None = None
    try:
        await asyncio.to_thread(version_service.commit_run, project_id, task_id, metrics)
    except Exception as exc:  # noqa: BLE001 —— 版本提交失败不连坐训练结果，但必须透出
        version_error = _version_error_message(ws, task_id, exc)
        logger.exception("运行版本提交失败 project_id=%s task_id=%s", project_id, task_id)
        try:
            knowledge_service.record_run({
                "project_id": project_id,
                "task_id": task_id,
                "run_type": "train",
                "environment": {"python": python, "environment_project_id": env_project_id},
                "params": run_params,
                "command": " ".join(cmd),
                "status": "failed",
                "error": version_error,
                "log_path": str(log_path),
                "started_at": started,
                "finished_at": _now(),
            })
        except Exception:  # noqa: BLE001 —— 留痕写入失败不得掩盖版本提交失败本身
            logger.exception("版本提交失败留痕落库失败 project_id=%s task_id=%s", project_id, task_id)
    task_manager.update_progress(
        task_id, {"stage": "完成", "metrics": metrics, "version_error": version_error})


def register() -> None:
    """注册训练任务 handler（main lifespan 调用）。"""
    task_manager.register_handler(TASK_TYPE, _run_train)
