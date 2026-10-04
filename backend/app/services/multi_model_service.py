"""多模型综合分析（模块六 8.4，需求六.2，扩展项 N4）。

对同一数据集上的 ≥2 个模型评估结果做同口径综合分析：
① 同口径统一（以各 run 的逐样本 predictions 与 dataset_registry.alignment 为据）
② 一致/分歧样本识别（逐样本比对 y_pred，固定代码）
③ 分歧归因（agent：差异来自模型结构 / 训练数据 / 输入适配）
④ 融合（投票 / 加权平均 / 取交集，按任务类型自动默认，可显式覆盖，K5）
⑤ 融合前后指标对比（固定代码）
⑥ 结论作为 fusion_insight 蒸馏知识入库（draft，数据设计六.1）

经任务队列异步执行（与模块三对比同口径）；报告落在 data/multi_model/{task_id}.json，
GET /api/multi-model/{task_id} 取回。命名空间与既有 `analysis.py`（/api/analysis/*）隔离（八.4 提示）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from app.config import DATA_DIR
from app.services import agent_service, knowledge_service, task_manager

logger = logging.getLogger(__name__)

TASK_TYPE = "multi_model"
MULTI_MODEL_DIR = DATA_DIR / "multi_model"

# 融合方式：vote（投票）/ weighted（加权平均，需 prob）/ intersection（取交集）
FUSION_MODES = ("vote", "weighted", "intersection")

ATTRIBUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "分歧总体说明"},
        "reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cause": {"type": "string",
                              "enum": ["model_structure", "training_data", "input_adaptation", "other"]},
                    "explanation": {"type": "string"},
                    "sample_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["cause", "explanation"],
            },
        },
    },
    "required": ["summary"],
}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _load_predictions(run: dict) -> list[dict]:
    """从评估产物的 artifact_path 读逐样本预测（baseline_service 契约 [{id,y_true,y_pred,prob,path}]）。"""
    path = run.get("artifact_path")
    if not path or not Path(path).exists():
        raise ValueError(f"运行 {run.get('run_id')} 无可用评估产物（artifact_path 缺失或文件不存在）")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    preds = payload.get("predictions") if isinstance(payload, dict) else None
    if not isinstance(preds, list) or not preds:
        raise ValueError(f"运行 {run.get('run_id')} 的产物不含逐样本 predictions，无法做综合分析")
    return preds


def _load_label_merge(dataset_id: Optional[str]) -> tuple[dict, dict]:
    """取数据集的已确认对齐里的标签归并表（数据设计八.2、8.4 第 1 步「标签体系按 alignment 统一」）。

    返回 (label_merge, alignment_info)。label_merge = {源标签: 归并标签}；未确认或无 label_merge 时为空表
    （此时不做归并，报告里 alignment_info 如实说明原因）。
    """
    if not dataset_id:
        return {}, {"dataset_id": None, "used": False, "reason": "未提供 dataset_id，未做标签归并"}
    ds = knowledge_service.get_item("dataset", dataset_id)
    if ds is None:
        return {}, {"dataset_id": dataset_id, "used": False, "reason": "数据集不存在"}
    alignment = knowledge_service._as_dict(ds.get("alignment"))
    merge = alignment.get("label_merge") if isinstance(alignment.get("label_merge"), dict) else {}
    confirmed = alignment.get("status") == "confirmed"
    if not confirmed or not merge:
        return {}, {"dataset_id": dataset_id, "used": False, "confirmed": confirmed,
                    "reason": "对齐未确认或无 label_merge"}
    return {str(k): str(v) for k, v in merge.items()}, {
        "dataset_id": dataset_id, "used": True, "confirmed": True, "entries": len(merge)}


def _resolve_dataset_id(params: dict, runs: list[dict]) -> Optional[str]:
    """确定综合分析所用的数据集：显式参数优先，否则取第一个 run 参数里的 dataset_id。"""
    if params.get("dataset_id"):
        return params["dataset_id"]
    for r in runs:
        did = knowledge_service._as_dict(r.get("params")).get("dataset_id")
        if did:
            return did
    return None


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _majority(labels: list) -> tuple[object, bool]:
    """多数表决：返回 (标签, 是否有并列)。"""
    counts: dict = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    top = max(counts.values())
    winners = [k for k, v in counts.items() if v == top]
    return winners[0], len(winners) > 1


def _weighted(preds: list[dict], labels: list) -> Optional[object]:
    """加权平均（等权）：对每条预测的 prob（dict[label]->p 或数值）求平均后取 argmax。

    prob 缺省或形状不支持时返回 None（调用方回退投票）。
    """
    acc: dict = {}
    ok = False
    for p in preds:
        prob = p.get("prob")
        if isinstance(prob, dict):
            ok = True
            for lab, val in prob.items():
                fv = _as_float(val)
                if fv is not None:
                    acc[lab] = acc.get(lab, 0.0) + fv
        elif isinstance(prob, list) and len(prob) == len(labels):
            ok = True
            for lab, val in zip(labels, prob):
                fv = _as_float(val)
                if fv is not None:
                    acc[lab] = acc.get(lab, 0.0) + fv
    if not ok or not acc:
        return None
    return max(acc.items(), key=lambda kv: kv[1])[0]


def _default_fusion(task_type: Optional[str]) -> str:
    """按任务类型自动选融合方式（K5）：分类默认投票；其余默认加权平均（不可用时回退投票）。"""
    if task_type in (None, "", "classification", "detection", "segmentation"):
        return "vote"
    return "weighted"


def _attribution_prompt(labels: list, disagreements: list[dict], model_names: dict) -> str:
    return (
        "你是多模型归纳分析助手。下面是两个及以上模型在同一数据集上对**分歧样本**的预测，"
        "请判断分歧最可能来自：模型结构(model_structure)、训练数据(training_data)、"
        "输入适配(input_adaptation) 或其它(other)，并给出简短解释。\n\n"
        f"模型名：{json.dumps(model_names, ensure_ascii=False)}\n"
        f"样本标签体系：{json.dumps(labels, ensure_ascii=False)}\n"
        f"分歧样本（最多 40 条）：\n{json.dumps(disagreements[:40], ensure_ascii=False, indent=2)}\n"
    )


async def _analyze(params: dict, task_id: str) -> dict:
    run_ids = params.get("run_ids") or []
    if len(run_ids) < 2:
        raise ValueError("多模型综合分析至少需要两个模型运行（run_ids）")
    labels_param = params.get("labels") or []
    task_type = params.get("task_type")
    fusion = params.get("fusion") or _default_fusion(task_type)
    if fusion not in FUSION_MODES:
        raise ValueError(f"未知融合方式：{fusion}（可选 {', '.join(FUSION_MODES)}）")

    raw_runs = []
    for rid in run_ids:
        run = knowledge_service.get_item("run", rid)
        if run is None:
            raise ValueError(f"运行记录不存在：{rid}")
        preds = _load_predictions(run)
        # 模型名：优先运行参数里的 model，其次运行类型，最后 run_id 前 8 位
        rparams = knowledge_service._as_dict(run.get("params"))
        name = rparams.get("model") or run.get("run_type") or rid[:8]
        raw_runs.append({"run_id": rid, "name": name, "params": rparams, "preds": preds})

    # ① 同口径统一：标签体系按数据集的已确认 alignment.label_merge 归并（数据设计八.2）
    dataset_id = _resolve_dataset_id(params, raw_runs)
    label_merge, alignment_info = _load_label_merge(dataset_id)

    def _map_label(value):
        return label_merge.get(str(value), value) if label_merge else value

    runs = []
    for r in raw_runs:
        mapped = [{**p, "y_true": _map_label(p.get("y_true")), "y_pred": _map_label(p.get("y_pred"))}
                  for p in r["preds"]]
        runs.append({"run_id": r["run_id"], "name": r["name"],
                     "preds": {p["id"]: p for p in mapped if p.get("id") is not None}})

    # ① 同口径：取所有 run 共有的样本 id
    common_ids = set(runs[0]["preds"].keys())
    for r in runs[1:]:
        common_ids &= set(r["preds"].keys())
    common_ids = sorted(common_ids, key=str)
    if not common_ids:
        raise ValueError("各模型的评估样本 id 无交集，无法同口径比较（需同一数据集同一划分）")

    # 标签体系：显式传入优先，否则取所有 y_pred 取值并集
    labels = list(labels_param)
    if not labels:
        seen: dict = {}
        for r in runs:
            for sid in common_ids:
                lab = r["preds"][sid].get("y_pred")
                seen[lab] = True
        labels = list(seen.keys())

    # ② 一致 / 分歧样本识别（y_true 不一致的样本标为不可比）
    consistent, disagreements, incomparable = [], [], []
    for sid in common_ids:
        row = [{"model": r["name"], "y_pred": r["preds"][sid].get("y_pred"), "y_true": r["preds"][sid].get("y_true")}
               for r in runs]
        y_trues = {json.dumps(item["y_true"], ensure_ascii=False) for item in row}
        if len(y_trues) > 1:
            incomparable.append({"id": sid, "reason": "各模型 y_true 不一致（标签体系未对齐）"})
            continue
        preds = [item["y_pred"] for item in row]
        if len(set(map(lambda x: json.dumps(x, ensure_ascii=False), preds))) == 1:
            consistent.append({"id": sid, "y_true": row[0]["y_true"], "y_pred": preds[0]})
        else:
            disagreements.append({"id": sid, "y_true": row[0]["y_true"], "predictions": row})

    # ④ 融合 + ⑤ 融合前后指标
    per_model_correct: dict = {r["name"]: 0 for r in runs}
    per_model_total = 0
    fused_ok = 0
    fused_total = 0
    fused_rows = []
    for sid in common_ids:
        if any(x["id"] == sid for x in incomparable):
            continue
        per_model_total += 1
        truth = runs[0]["preds"][sid].get("y_true")
        for r in runs:
            if r["preds"][sid].get("y_pred") == truth:
                per_model_correct[r["name"]] += 1
        item_preds = [r["preds"][sid] for r in runs]
        fused_label = _fuse_one(fusion, item_preds, labels)
        if fused_label is not None:
            fused_total += 1
            if fused_label == truth:
                fused_ok += 1
        fused_rows.append({"id": sid, "y_true": truth, "fused": fused_label})

    metrics_before = {
        name: {"accuracy": (correct / per_model_total) if per_model_total else None, "n": per_model_total}
        for name, correct in per_model_correct.items()
    }
    metrics_after = {"accuracy": (fused_ok / fused_total) if fused_total else None, "n": fused_total,
                     "fusion": fusion}

    # ③ 分歧归因（agent）：失败不阻断报告（如实标 attribution_error）
    attribution: dict = {}
    attribution_error: Optional[str] = None
    if disagreements:
        model_names = {r["run_id"]: r["name"] for r in runs}
        try:
            res = await agent_service.run_sync(
                _attribution_prompt(labels, disagreements, model_names),
                output_schema=ATTRIBUTION_SCHEMA, timeout_s=180)
            attribution = res.get("structured_output") or {}
            # 形状容错：DeepSeek 下结构化输出走文件兜底，模型可能把 reasons 直接写成裸数组
            if isinstance(attribution, list):
                attribution = {"reasons": [x for x in attribution if isinstance(x, dict)]}
            if not (attribution.get("summary") or attribution.get("reasons")):
                # 端点可用但未返回可用的结构化归因 → 如实记录，不阻断定量结论
                attribution_error = "agent 未返回结构化归因（检查模型端点 / 结构化输出配置，见 O2）"
        except Exception as exc:  # noqa: BLE001 —— 归因是增强项，失败不阻断定量结论
            attribution_error = str(exc)

    report = {
        "analysis_id": task_id,
        "created_at": _now(),
        "models": [{"run_id": r["run_id"], "name": r["name"]} for r in runs],
        "fusion": fusion,
        "alignment": alignment_info,
        "labels": labels,
        "n_common_samples": len(common_ids),
        "consistent": consistent,
        "disagreements": disagreements,
        "incomparable": incomparable,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "attribution": attribution,
        "attribution_error": attribution_error,
    }

    # ⑥ 结论入库（fusion_insight，draft）
    conclusion = _conclusion_text(report)
    knowledge_id = knowledge_service.record_knowledge({
        "type": "fusion_insight",
        "title": f"多模型综合分析（{'+'.join(r['name'] for r in runs)}）",
        "content": conclusion,
        "structured": {
            "models": report["models"], "fusion": fusion,
            "n_common_samples": len(common_ids),
            "n_consistent": len(consistent), "n_disagreement": len(disagreements),
            "metrics_before": metrics_before, "metrics_after": metrics_after,
            "source_task_id": task_id,
        },
        "sources": [{"type": "run_record", "ref": r["run_id"]} for r in runs],
        "confidence": "medium",
        "scope": {"task_type": task_type} if task_type else {},
        "status": "draft",
    })
    report["knowledge_id"] = knowledge_id

    MULTI_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MULTI_MODEL_DIR / f"{task_id}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _fuse_one(fusion: str, preds: list[dict], labels: list):
    """对单个样本融合：vote=多数票；weighted=平均概率 argmax（不可用回退投票）；intersection=全一致才给值。"""
    values = [p.get("y_pred") for p in preds]
    if len(set(map(lambda x: json.dumps(x, ensure_ascii=False), values))) == 1:
        return values[0]                      # 全体一致：任何融合方式结果相同
    if fusion == "intersection":
        return None                           # 有分歧 → 交集为空
    if fusion == "weighted":
        w = _weighted(preds, labels)
        if w is not None:
            return w
    return _majority(values)[0]               # vote（以及 weighted 回退）


def _conclusion_text(report: dict) -> str:
    mb = report["metrics_before"]
    ma = report["metrics_after"]
    before = "；".join(f"{name} 准确率 {v['accuracy']:.3f}" if v["accuracy"] is not None else f"{name} 无指标"
                      for name, v in mb.items())
    after = f"{ma['accuracy']:.3f}" if ma["accuracy"] is not None else "无"
    attr = report.get("attribution") or {}
    summary = attr.get("summary") or "（未产出分歧归因）"
    return (
        f"各模型同口径（{report['n_common_samples']} 个共同样本）比较：{before}。"
        f"一致样本 {len(report['consistent'])} 个、分歧样本 {len(report['disagreements'])} 个"
        f"（不可比 {len(report['incomparable'])} 个）。以「{report['fusion']}」融合后准确率 {after}。"
        f"分歧归因：{summary}"
    )


def start(run_ids: list[str], *, labels: Optional[list] = None, fusion: Optional[str] = None,
          task_type: Optional[str] = None, dataset_id: Optional[str] = None) -> str:
    """发起多模型综合分析任务，返回 task_id（报告可经 GET /api/multi-model/{task_id} 取回）。"""
    return task_manager.create_task(TASK_TYPE, params={
        "run_ids": run_ids, "labels": labels or [], "fusion": fusion,
        "task_type": task_type, "dataset_id": dataset_id,
    })


def get_report(analysis_id: str) -> Optional[dict]:
    path = MULTI_MODEL_DIR / f"{analysis_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def register() -> None:
    task_manager.register_handler(TASK_TYPE, _run)


async def _run(params: dict, task_id: str) -> None:
    task_manager.update_progress(task_id, {"stage": "综合分析"})
    await _analyze(params, task_id)
