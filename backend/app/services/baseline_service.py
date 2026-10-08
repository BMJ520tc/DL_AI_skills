"""模块三 5.2 自带数据基准运行（模块详细设计 5.2）。

流程：定位权重 → 确保 eval 入口可用（缺失/仍是模板时由 agent 就地写出真实实现）
      → 预处理自带数据（5.1）→ 在模块一环境运行 eval 入口 → 指标落库。
本服务的**流程与执行**为固定代码（不引入 agent 编排），仅「生成项目专属 eval 入口」这一
步调用 agent：入口契约是「项目 source/eval_entry.py」，必须加载 model_dir 下的真实权重推理，
平台不提供任何规则桩兜底。

跨数据集评估（5.4 的输入）复用同一入口与执行逻辑，仅 run_type 取 eval。

边界（5.2「异常与边界」）：
- 未定位到权重 → 明确失败并提示先跑通模块一/复现，不用规则桩顶替；
- agent 无产出（端点不可用/未登录/结构化输出校验失败）→ 明确失败，不静默成功、不伪造指标；
- fixture 规则桩（`"model": "fixture-rule"`）仅测试可用，真实运行一律拒绝。
"""
import asyncio
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.config import resource_path
from app.contracts import CANONICAL_METRICS, normalize_metrics
from app.services import proc_util
from app.services import (
    agent_service, analysis_service, knowledge_service, project_manager, task_manager,
)

TASK_TYPE = "baseline"
EVAL_ENTRY_NAME = "eval_entry.py"
EVAL_TEMPLATE = resource_path("scripts/eval_entry_template.py")
BASELINE_TIMEOUT_S = 3600
WEIGHT_EXTS = (".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".h5")

# fixture 规则桩的自我标识（scripts/eval_entry_template.py 的测试替身，见 data/projects/*）。
# 真实运行禁止出现——否则「加载原始权重」这一需求会被固定规则出预测的桩悄悄顶替。
FIXTURE_MODEL_MARKERS = ("fixture-rule", "fixture")

EVAL_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "implemented": {"type": "boolean", "description": "是否已把 source/eval_entry.py 写成可运行的真实实现"},
        "script_path": {"type": "string", "description": "写出的 eval 入口脚本路径"},
        "uses_model_dir": {"type": "boolean", "description": "实现是否真的从 model_dir 加载权重后再推理"},
        "weights_source": {"type": "string", "description": "权重来源（model_dir 下的具体文件或目录）"},
        "rationale": {"type": "string", "description": "依据项目代码得出的加载/推理方式说明"},
    },
    "required": ["implemented", "uses_model_dir"],
}



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


# ---------- eval 入口：真实实现的生成与校验（缺口一） ----------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_template_entry(path: Path) -> bool:
    """入口是否仍是模板（未实现）——模板的 evaluate() 就是 `raise NotImplementedError`。"""
    if not path.exists():
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True
    return "NotImplementedError" in text


def looks_like_fixture(payload: dict) -> bool:
    """入口输出是否来自 fixture 规则桩（自我标识 `"model": "fixture-rule"`）。"""
    model = str((payload or {}).get("model") or "").strip().lower()
    return any(model == marker or model.startswith(marker + "-") or model.startswith(marker + "_")
               for marker in FIXTURE_MODEL_MARKERS)


def _eval_entry_prompt(source: Path, ws: Path, weight_dirs: list[str]) -> str:
    structure = ws / "reports" / "structure_report.json"
    return (
        "本项目缺少可用的 eval 入口实现，请你阅读项目代码后**就地写出真实实现**：\n"
        f"目标文件: {source / EVAL_ENTRY_NAME}\n"
        f"项目源码目录: {source}\n"
        f"结构分析报告: {structure}（存在则读）\n"
        f"requirements/依赖清单与 README 也在 {source} 下\n\n"
        f"平台已定位到的权重目录 model_dir 候选: {json.dumps(weight_dirs, ensure_ascii=False)}\n\n"
        "要求：\n"
        "1) 先读模型定义、训练/推理脚本、README、reports/structure_report.json、requirements，"
        "弄清权重文件格式与该项目的模型类/前向方式；\n"
        "2) 把 evaluate(data_dir, model_dir, out_json) 写成：读 data_dir/preprocessed.csv"
        "（列为统一 schema id/split/label/input，其余为 meta_*）→ **用 model_dir 下的真实权重**"
        "构造并加载模型（不得使用随机初始化、不得用固定规则出预测）→ 逐样本推理；\n"
        "3) 指标 JSON 写入 out_json，形状固定：\n"
        '   {"schema_version": "1.0", "task_type": "...", "model": "...", "dataset": "...",'
        ' "metrics": {...}, "primary_metric": "...", "n_samples": <int>,'
        ' "per_class": {...}, "predictions": [{"id","y_true","y_pred","prob","path"}]}\n'
        f"   metrics 的键名只能取: {json.dumps(list(CANONICAL_METRICS), ensure_ascii=False)}；"
        "n_samples 与 predictions 必须给出；\n"
        "4) 入口须能被 `python eval_entry.py <data_dir> <model_dir> <out_json>` 直接调用，"
        "加载失败/权重缺失时应非零退出并在 stderr 说明原因，**绝不静默返回随机或规则预测**。\n"
        "完成后按 schema 返回 implemented / uses_model_dir / weights_source / rationale。"
    )


async def ensure_eval_entry(project_id: str, task_id: str, ws: Path, source: Path,
                            weight_dirs: list[str]) -> dict | None:
    """确保 source/eval_entry.py 是真实实现；缺失或仍是模板时由 agent 就地生成。

    返回生成证据 dict（含脚本路径/哈希/权重来源）；入口已可用时返回 None。
    **任何失败都抛 RuntimeError**：平台不会回退到固定规则桩，也不会伪造指标。
    """
    entry = source / EVAL_ENTRY_NAME
    if entry.exists() and not is_template_entry(entry):
        return None

    if not entry.exists():
        # 保留模板副本到 runs/ 便于人工比对（真正的实现由 agent 写到 source/）
        scaffold = ws / "runs" / EVAL_ENTRY_NAME
        scaffold.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EVAL_TEMPLATE, scaffold)

    progress_base = {"eval_entry_path": str(entry), "weights_found": bool(weight_dirs),
                     "weight_dirs": weight_dirs}
    task_manager.update_progress(task_id, {"eval_entry_generation": {**progress_base, "status": "running"}})

    prompt = _eval_entry_prompt(source, ws, weight_dirs)
    try:
        result = await agent_service.run_sync(
            prompt, cwd=str(source), output_schema=EVAL_ENTRY_SCHEMA, max_turns=40, timeout_s=900,
        )
    except Exception as exc:
        # agent SDK/端点的异常类型开放（未安装 CLI、未登录、网络/端点错误…），
        # 这里是有意收口：一律转成「明确失败」并留进度，绝不让异常变成静默跳过。
        reason = (
            f"eval 入口生成失败：agent 调用异常（{type(exc).__name__}: {exc}）；"
            "检查模型端点/凭证配置（见 O2）；平台不会回退到固定规则桩"
        )
        task_manager.update_progress(
            task_id, {"eval_entry_generation": {**progress_base, "status": "failed", "reason": reason}}
        )
        raise RuntimeError(reason) from exc
    draft = result.get("structured_output") or {}
    if not draft:
        reason = (
            "eval 入口生成失败：agent 未返回结构化结果（检查模型端点/凭证配置，见 O2）；"
            "平台不会回退到固定规则桩，也不会伪造指标"
        )
        task_manager.update_progress(
            task_id, {"eval_entry_generation": {**progress_base, "status": "failed", "reason": reason}}
        )
        raise RuntimeError(reason)

    if not draft.get("implemented") or not draft.get("uses_model_dir"):
        reason = (
            "eval 入口生成未通过校验：agent 未确认「已实现且真的从 model_dir 加载权重」"
            f"（implemented={draft.get('implemented')}, uses_model_dir={draft.get('uses_model_dir')}）；"
            "平台不会回退到固定规则桩"
        )
        task_manager.update_progress(
            task_id, {"eval_entry_generation": {**progress_base, "status": "failed", "reason": reason}}
        )
        raise RuntimeError(reason)

    if is_template_entry(entry):
        reason = (
            f"eval 入口生成未通过校验：agent 报告已实现，但 {entry} 仍不存在或仍是模板；"
            "拒绝把模板当真实实现继续评估"
        )
        task_manager.update_progress(
            task_id, {"eval_entry_generation": {**progress_base, "status": "failed", "reason": reason}}
        )
        raise RuntimeError(reason)

    info = {
        "status": "ok",
        "project_id": project_id,
        "drafted_by": "agent",
        "script_path": str(entry),
        "sha256": _sha256(entry),
        "weights_source": draft.get("weights_source") or (weight_dirs[0] if weight_dirs else None),
        "uses_model_dir": True,
        "rationale": draft.get("rationale"),
        "generated_at": _now(),
        **progress_base,
    }
    evidence = ws / "runs" / task_id / "eval_entry_generation.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    info["evidence_path"] = str(evidence)
    task_manager.update_progress(task_id, {"eval_entry_generation": info})
    return info



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
    project_id: str, task_id: str, data_dir: Path, run_type: str = "eval",
    dataset_label: str | None = None, *, extra_params: dict | None = None,
    allow_fixture: bool = False,
) -> dict:
    """在项目独立环境跑 eval 入口并落库。失败抛 RuntimeError（同时已写入失败记录）。

    返回成功时的运行记录 dict（含 metrics 与 run_id）。

    - `extra_params`：调用方（如 5.4 跨数据集评估）附加的可复核参数（对齐副本路径/条目数）。
    - `allow_fixture`：仅测试可置 True 以跑 fixture 规则桩；真实运行一律拒绝 fixture 输出。
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

    weight_dirs = _find_weights(source, ws)
    if not weight_dirs:
        # 5.2 异常与边界：权重缺失 → 明确提示先跑通模块一/复现，不用规则桩顶替
        err = (
            f"未在项目中发现模型权重（{', '.join(WEIGHT_EXTS)}），无法加载原始模型；"
            "请先跑通模块一（结构分析）以确认权重位置，或先复现实验产出权重后再评估。"
            "平台不会用固定规则桩顶替真实权重推理。"
        )
        _fail(project_id, task_id, run_type, started, err)
        raise RuntimeError(err)

    eval_entry = source / EVAL_ENTRY_NAME
    entry_source = "existing"
    generation: dict | None = None
    try:
        generation = await ensure_eval_entry(project_id, task_id, ws, source, weight_dirs)
    except RuntimeError as exc:
        _fail(project_id, task_id, run_type, started, str(exc))
        raise
    if generation is not None:
        entry_source = "agent_generated"
    if not eval_entry.exists():
        err = f"eval 入口缺失且未能生成: {eval_entry}"
        _fail(project_id, task_id, run_type, started, err)
        raise RuntimeError(err)

    model_dir = weight_dirs[0]
    # 同一 task 下可能有多个跨数据集评估（每个数据集一次），指标文件按 data_dir 加短哈希后缀，
    # 否则后一次会覆盖前一次，多条 run_record 的 artifact_path 指向同一份被覆盖的文件。
    suffix = ""
    if run_type != "baseline":
        suffix = "_" + hashlib.sha1(str(data_dir).lower().encode("utf-8")).hexdigest()[:8]
    out_json = ws / "runs" / task_id / f"{run_type}_metrics{suffix}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [python, str(eval_entry), str(data_dir), model_dir or "", str(out_json)]
    result = await _run_entry(cmd, source, BASELINE_TIMEOUT_S)

    record = {
        "command": " ".join(cmd),
        "params": {"data_dir": str(data_dir), "run_type": run_type,
                   "dataset_label": dataset_label or run_type,
                   "weights_found": bool(weight_dirs), "weight_dirs": weight_dirs,
                   "model_dir": model_dir,
                   "eval_entry_path": str(eval_entry),
                   "eval_entry_source": entry_source,
                   "eval_entry_sha256": _sha256(eval_entry),
                   "eval_entry_generation": generation},
        "started_at": started,
        "finished_at": _now(),
    }
    if extra_params:
        record["params"].update(extra_params)

    if not result["ok"]:
        error = result["error"]
        record.update({"status": "failed", "error": error})
        _record(project_id, task_id, run_type, record)
        raise RuntimeError(f"评估运行失败: {error}")

    if not out_json.exists():
        err = f"eval 入口未写出指标文件: {out_json}"
        record.update({"status": "failed", "error": err})
        _record(project_id, task_id, run_type, record)
        raise RuntimeError(err)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    if looks_like_fixture(payload) and not allow_fixture:
        err = (
            "eval 入口输出来自 fixture 规则桩（model=fixture-rule），已拒绝："
            "真实评估必须加载 model_dir 下的原始权重；fixture 仅测试可用。"
        )
        record.update({"status": "failed", "error": err})
        _record(project_id, task_id, run_type, record)
        raise RuntimeError(err)

    metrics = normalize_metrics(payload.get("metrics") or payload)
    if not metrics:
        err = "eval 入口输出中没有可识别的数值指标"
        record.update({"status": "failed", "error": err})
        _record(project_id, task_id, run_type, record)
        raise RuntimeError(err)

    record["params"].update({
        "primary_metric": payload.get("primary_metric"),
        "n_samples": payload.get("n_samples"),
        "predictions": len(payload.get("predictions") or []) or None,
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
