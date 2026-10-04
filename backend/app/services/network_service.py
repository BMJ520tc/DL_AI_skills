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
from typing import Optional

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
    """该网络的运行记录（**训练 + 自动调参**，成功记录；指标/运行面板数据源）。

    自动调参此前不进列表 → 调参结果（含 winner）在界面上永远不可见；一并纳入并按 started_at 倒序。
    """
    _require_network(project_id)
    runs = (knowledge_service.list_runs(project_id, "train")
            + knowledge_service.list_runs(project_id, "autotune"))
    runs.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return runs


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
    """注册训练与自动调参任务 handler（main lifespan 调用）。"""
    task_manager.register_handler(TASK_TYPE, _run_train)
    task_manager.register_handler(TASK_AUTOTUNE, _run_autotune)


# ===========================================================================
# 自动调参闭环（B，2026-10-04 扩范围；把「知识带入 → 自动重跑 → 按指标保留 → 蒸馏回写」接成闭环）
# ===========================================================================

TASK_AUTOTUNE = "network_autotune"
_AUTOTUNE_MAX_CANDIDATES = 4
_HYPER_KEYS = ("epochs", "batch_size", "learning_rate")


def _primary_metric(metrics: dict) -> tuple[str, float | None, bool]:
    """主指标：有 accuracy 取 accuracy（越大越好），否则取 loss（越小越好）。→ (名, 值, 越大越好)"""
    acc = metrics.get("accuracy")
    if isinstance(acc, (int, float)):
        return "accuracy", float(acc), True
    loss = metrics.get("loss")
    if isinstance(loss, (int, float)):
        return "loss", float(loss), False
    return "", None, True


def _advice_candidates(advice: dict) -> list[dict]:
    """从带入的 param_advice 里抽出可用超参候选（structured.params 为约定写法）。"""
    out: list[dict] = []
    for item in advice.get("param_advice") or []:
        s = item.get("structured") if isinstance(item, dict) else None
        if isinstance(s, str):
            try:
                s = json.loads(s)
            except (json.JSONDecodeError, TypeError):
                s = None
        params = s.get("params") if isinstance(s, dict) else None
        if not isinstance(params, dict):
            continue
        cand = {k: params[k] for k in _HYPER_KEYS if isinstance(params.get(k), (int, float))}
        if cand:
            out.append(cand)
    return out


def _build_candidates(base: dict, advice: dict, explicit: list | None) -> list[dict]:
    """候选集：显式 > 建议 > 围绕 base 的小网格（lr×0.1 / lr×10）；去重 + 上限。"""
    cands: list[dict] = []
    for c in explicit or []:
        if isinstance(c, dict):
            cands.append({**base, **{k: c[k] for k in _HYPER_KEYS if k in c}})
    cands.append(dict(base))
    cands.extend({**base, **a} for a in _advice_candidates(advice))
    for factor in (0.1, 10.0):
        cands.append({**base, "learning_rate": round(float(base["learning_rate"]) * factor, 8)})
    seen: set = set()
    uniq: list[dict] = []
    for c in cands:
        try:
            key = (int(c["epochs"]), int(c["batch_size"]), float(c["learning_rate"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in seen or key[0] < 1 or key[1] < 1 or key[2] <= 0:
            continue
        seen.add(key)
        uniq.append({"epochs": key[0], "batch_size": key[1], "learning_rate": key[2]})
    return uniq[:_AUTOTUNE_MAX_CANDIDATES]


def _validate_train_inputs(project: dict, body: dict) -> tuple[dict, str]:
    """start_run / start_autotune 共用的输入校验：数据集有预处理产物、环境已就绪。"""
    dataset_id = body.get("dataset_id")
    dataset = knowledge_service.get_item("dataset", dataset_id) if dataset_id else None
    if dataset is None:
        raise ValueError("数据集不存在：请先在模块三下载/注册数据集")
    local = dataset.get("local_path")
    if not local or not (Path(local).parent / "preprocessed.csv").exists():
        raise ValueError(
            f"数据集 {dataset.get('name') or dataset_id} 尚无本地产物：请先在模块三完成预处理"
        )
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
    return dataset, env_project_id


def start_autotune(project_id: str, body: dict) -> str:
    """发起自动调参：数据集/环境校验后入队（B）。"""
    project = _require_network(project_id)
    dataset, env_project_id = _validate_train_inputs(project, body)
    base = {
        "epochs": int(body.get("epochs", 8)),
        "batch_size": int(body.get("batch_size", 32)),
        "learning_rate": float(body.get("learning_rate", 0.01)),
    }
    return task_manager.create_task(TASK_AUTOTUNE, project_id=project_id, params={
        "project_id": project_id,
        "dataset_id": dataset["dataset_id"],
        "environment_project_id": env_project_id,
        "base": base,
        "candidates": body.get("candidates"),
    })


async def _run_one_candidate(run_dir: Path, python: str, data_dir: Path, hyper: dict,
                             out_json: Path, log_path: Path) -> tuple[Optional[dict], str, str]:
    """跑一次训练脚本（同一份 model.py/train.py，只换超参）→ (metrics | None, command, output)。"""
    cmd = [python, "train.py", str(data_dir), str(hyper["epochs"]), str(hyper["batch_size"]),
           f"{hyper['learning_rate']:g}", str(out_json)]
    rc, output = await proc_util.run_command(cmd, cwd=str(run_dir), timeout=TRAIN_TIMEOUT_S)
    log_path.write_text(output, encoding="utf-8", errors="replace")
    metrics: Optional[dict] = None
    if rc == 0 and out_json.exists():
        try:
            res = json.loads(out_json.read_text(encoding="utf-8"))
            m = res.get("metrics") if isinstance(res, dict) else None
            metrics = m if isinstance(m, dict) else None
        except (json.JSONDecodeError, OSError):
            metrics = None
    return metrics, " ".join(cmd), output


async def _run_autotune(params: dict, task_id: str) -> None:
    """network_autotune：带入知识 → 生成候选超参 → 逐个训练 → 按主指标选优 → 记录 + 蒸馏回写。

    诚实边界：只搜**超参**、不搜结构；训练脚本不落权重，故「保留最优」体现为
    ——记录最优超参并蒸馏为 param_advice（供后续带入）、提交一个带最优指标摘要的版本节点。
    """
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

    env_project = project_manager.get_project(params["environment_project_id"])
    if env_project is None:
        raise RuntimeError(f"环境项目 {params['environment_project_id']} 不存在")
    python = analysis_service._project_python(_ws(env_project))
    if python is None:
        raise RuntimeError(f"环境未就绪：项目 {params['environment_project_id']} 没有独立运行环境")

    # 知识带入（需求六.1 / 8.2）：param_advice 进候选集，把「历史经验」用起来
    try:
        advice = knowledge_service.bring_advice_summary(
            task_type=dataset.get("task_type"), model=project.get("name"), dataset=dataset.get("name"))
    except Exception:  # noqa: BLE001
        advice = {}
    if advice:
        task_manager.update_progress(task_id, {"knowledge_bring": advice})

    candidates = _build_candidates(params["base"], advice, params.get("candidates"))
    if len(candidates) < 2:
        raise RuntimeError("候选超参不足两个，无法比较（请提供不同的候选或基础超参）")

    run_dir = ws / "runs" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.py").write_text(code, encoding="utf-8")
    shutil.copyfile(Path(__file__).resolve().parents[2] / "templates" / "train.py", run_dir / "train.py")

    results: list[dict] = []
    for i, cand in enumerate(candidates, 1):
        task_manager.update_progress(task_id, {"stage": f"候选 {i}/{len(candidates)}", "params": cand})
        metrics, cmd, output = await _run_one_candidate(
            run_dir, python, data_dir, cand, run_dir / f"metrics_{i}.json", run_dir / f"train_{i}.log")
        results.append({"params": cand, "metrics": metrics, "command": cmd,
                        "error": None if metrics is not None else (output or "")[-300:]})

    ok = [r for r in results if isinstance(r["metrics"], dict)]
    if not ok:
        raise RuntimeError("自动调参：所有候选训练均失败（详见各 train_<i>.log）")

    def _score(r: dict) -> float:
        _, value, bigger = _primary_metric(r["metrics"])
        if value is None:
            return float("-inf")
        return value if bigger else -value

    winner = max(ok, key=_score)
    metric_name, metric_value, bigger = _primary_metric(winner["metrics"])

    knowledge_service.record_run({
        "project_id": project_id,
        "task_id": task_id,
        "run_type": "autotune",
        "environment": {"python": python, "environment_project_id": params["environment_project_id"]},
        "params": {"dataset_id": params["dataset_id"], "base": params["base"],
                   "n_candidates": len(candidates), "winner": winner["params"],
                   "primary_metric": metric_name},
        "command": winner["command"],
        "status": "success",
        "metrics": winner["metrics"],
        "artifact_path": str(run_dir / "metrics_1.json"),
        "log_path": str(run_dir / "train_1.log"),
        "started_at": started,
        "finished_at": _now(),
    })

    # 闭环回写：把最优超参蒸馏为 param_advice（draft，供后续任务带入）
    try:
        knowledge_service.record_knowledge({
            "type": "param_advice",
            "title": f"自动调参：{project.get('name') or project_id} 在 {dataset.get('name') or ''} 上的最优超参",
            "content": (f"自动调参比较 {len(candidates)} 组超参，最优 {winner['params']}，"
                        f"主指标 {metric_name}={metric_value}。"),
            "structured": {"params": winner["params"], "primary_metric": metric_name,
                           "primary_value": metric_value, "candidates": results,
                           "source_task_id": task_id},
            "sources": [{"type": "run_record", "ref": task_id}],
            "confidence": "medium",
            "scope": {"dataset": dataset.get("name"), "model": project.get("name")},
            "status": "draft",
        })
    except Exception:  # noqa: BLE001 —— 蒸馏回写失败不连坐调参结果
        logger.exception("自动调参蒸馏回写失败 project_id=%s task_id=%s", project_id, task_id)

    version_error: Optional[str] = None
    try:
        await asyncio.to_thread(version_service.commit_run, project_id, task_id, winner["metrics"])
    except Exception as exc:  # noqa: BLE001
        version_error = _version_error_message(ws, task_id, exc)
        logger.exception("自动调参版本提交失败 project_id=%s task_id=%s", project_id, task_id)

    task_manager.update_progress(task_id, {
        "stage": "完成", "winner": winner["params"], "primary_metric": metric_name,
        "primary_value": metric_value, "results": results, "version_error": version_error})
