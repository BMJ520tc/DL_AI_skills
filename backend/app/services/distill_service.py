"""任务后蒸馏入库（模块六 8.3、数据设计六.3）。

任务成功或失败结束时按**任务类型白名单**触发（K2）：以本次 run_record + 检索到的相关历史知识为素材，
由 agent 起草可复用结论（draft）→ 冲突检测（用户确认时可 supersede 旧结论）→ confirmed。
空结论不落条目（8.3 异常边界）。

与既有专用起草路径（env_manager 的 dependency_conflict、compare_service 的 usage_guidance）的关系：
两者都以 `structured.source_task_id` 标记来源；通用通道在起草前检查该任务是否已有蒸馏知识
（`knowledge_service.knowledge_for_task_exists`），有则跳过——避免同一任务被起草两遍。

本模块以 task_manager 的终态钩子方式接入：钩子**只入队**蒸馏任务（不在钩子里跑 agent），
既不阻塞任务队列、也不改变主任务结果，且起草可持久化/可重试。
"""
from __future__ import annotations

import json
import logging

from app.services import agent_service, knowledge_service, task_manager

logger = logging.getLogger(__name__)

# K2 白名单：会产出可复用结论的任务类型（纯查询/拉取类不触发，避免噪音与成本）。
DISTILL_TASK_TYPES = {
    "env_create",       # 依赖冲突（专用路径已起草时跳过）
    "network_train",    # 训练结果 → 参数建议/使用建议
    "decompose",        # 拆解结果 → 结构/形状经验
    "decompose_verify", # 两步验证结果 → 复现/数值一致性结论
    "reproduce",        # 复现执行 → 复现不一致
    "baseline",         # 自带数据基准 → 指标口径经验
    "compare",          # 跨数据集对比（专用路径已起草时跳过）
}

# 蒸馏结论的类型枚举（数据设计六.1）
_KNOWLEDGE_TYPES = [
    "param_advice", "dependency_conflict", "reproduction_discrepancy", "usage_guidance", "fusion_insight",
]

# 类型别名归一：模型在文件兜底里常写自有类型名（实测出现过 applicability / method），归到最接近的枚举值。
_TYPE_ALIASES = {
    "applicability": "usage_guidance", "method": "usage_guidance", "guidance": "usage_guidance",
    "usage": "usage_guidance", "insight": "usage_guidance", "recommendation": "usage_guidance",
    "params": "param_advice", "hyperparameters": "param_advice", "parameter_advice": "param_advice",
    "conflict": "dependency_conflict", "dependency": "dependency_conflict",
    "discrepancy": "reproduction_discrepancy", "reproduction": "reproduction_discrepancy",
    "inconsistency": "reproduction_discrepancy", "fusion": "fusion_insight",
}


_CONF_ALIASES = {
    "medium-high": "medium", "mid-high": "medium", "high-medium": "medium",
    "low-medium": "low", "medium-low": "low", "very high": "high", "very low": "low",
}


def _as_text(v) -> str:
    """把「本该是字符串」的字段安全取成 str —— 非字符串（模型偶尔把 content/confidence 给成数字或对象）
    一律视为空。

    LLM 产出的结构化字段**值**同样会漂移（先例：scope 出现列表/别名键、confidence 出现 `medium-high`）。
    此前 `(item.get("content") or "").strip()` 在 content 是数字（如 `0.9`）时直接
    `'float' object has no attribute 'strip'`，把整条蒸馏任务打挂（2026-10-05 实测）。
    """
    return v if isinstance(v, str) else ""


def _normalize_confidence(raw) -> str:
    """confidence 归一到 {high, medium, low}（数据设计六.2）。

    实测模型写过 `medium-high` 这类超出枚举的值；先查显式别名、再按词根兜底，未知/缺失归 `low`
    （蒸馏默认保守）。
    """
    c = _as_text(raw).strip().lower()
    if c in ("high", "medium", "low"):
        return c
    if c in _CONF_ALIASES:
        return _CONF_ALIASES[c]
    if "high" in c and "low" not in c:
        return "high"
    if "low" in c and "high" not in c:
        return "low"
    return "medium" if c else "low"


def _normalize_type(raw, allowed: Optional[tuple] = None) -> str:
    """把模型给的类型名归一到枚举；未知/缺失归 usage_guidance（占多数、最安全）。

    `allowed` 限定某来源允许的类型集合：越界（如论文蒸馏里模型误写 fusion_insight）时回落到
    allowed 首项，避免把「多模型综合分析结论」这类专属类型混进别的来源。
    """
    t = _as_text(raw).strip()
    if t not in _KNOWLEDGE_TYPES:
        t = _TYPE_ALIASES.get(t.lower(), "usage_guidance")
    if allowed and t not in allowed:
        return allowed[0]
    return t

DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "knowledge": {
            "type": "array",
            "description": "可复用的蒸馏结论；没有值得提炼的结论时返回空数组",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": _KNOWLEDGE_TYPES},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "structured": {"type": "object"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "scope": {"type": "object"},
                },
                "required": ["type", "title", "content"],
            },
        }
    },
    "required": ["knowledge"],
}


def _run_digest(run: dict) -> dict:
    """喂给 agent 的运行记录摘要（避免把整表字段原样灌进去）。"""
    return {
        "run_type": run.get("run_type"),
        "status": run.get("status"),
        "command": run.get("command"),
        "metrics": run.get("metrics"),
        "error": (run.get("error") or "")[:2000],
        "params": run.get("params"),
        "environment": run.get("environment"),
    }


def _looks_like_items(val) -> bool:
    return isinstance(val, list) and any(
        isinstance(x, dict) and (x.get("content") or x.get("title")) for x in val)


def _knowledge_items(structured) -> list[dict]:
    """把 agent 产出归一成知识条目列表（兼容多种漂移形状）。

    实测 DeepSeek 文件兜底会给出这些形状（全都要认，否则静默产不出东西）：
      * 裸数组 `[{...}]`；
      * `{"knowledge": [...]}` 包装；
      * **以类型名做键的包装** `{"usage_guidance": [...]}` / `{"paper_id": "...", "usage_guidance": [...]}`；
      * 单条对象（带 type/content）。
    """
    if isinstance(structured, list):
        return [x for x in structured if isinstance(x, dict)]
    if isinstance(structured, dict):
        if structured.get("content") and structured.get("type"):
            return [structured]                      # 单条
        for key in ("knowledge", "items", "results"):  # 已知包装键
            val = structured.get(key)
            if _looks_like_items(val):
                return [x for x in val if isinstance(x, dict)]
        for val in structured.values():              # 任意「值为条目数组」的键（模型按 type 名包）
            if _looks_like_items(val):
                return [x for x in val if isinstance(x, dict)]
    return []


def _prompt(run: dict, related: dict) -> str:
    return (
        "你是深度学习实验的**知识蒸馏**助手。请根据本次运行记录与相关历史已确认知识，提炼可复用的结论。\n\n"
        "## 本次运行记录（JSON）\n"
        f"{json.dumps(_run_digest(run), ensure_ascii=False, indent=2)}\n\n"
        "## 相关历史已确认知识（JSON）\n"
        f"{json.dumps(related, ensure_ascii=False, indent=2)}\n\n"
        "## 要求\n"
        "1) 只提炼**可复用**的结论——参数建议(param_advice)、依赖冲突(dependency_conflict)、"
        "复现不一致(reproduction_discrepancy)、使用建议(usage_guidance)；不要复述流水账；\n"
        "2) 每条给出 type/title/content 与 confidence，必要时给 structured；\n"
        "3) **scope 尽量填**：能从材料推断出 task_type / model / dataset 就务必写入 scope，"
        "确实推断不出时才留空（scope 决定这条知识在后续任务里能否被精准带出）；\n"
        "4) 没有值得入库的结论时返回**空数组**（不要硬凑）。"
    )


KNOWLEDGE_DISTILL_TASK_TYPE = "knowledge_distill"

# 「论文数据」来源的自动触发点：论文结论任务结束 → 触发该论文的蒸馏（需求六.1）
PAPER_DISTILL_TRIGGERS = {"conclusion"}


async def on_task_finished(task_id: str, status: str) -> None:
    """任务终态钩子：**只入队**蒸馏任务，不在钩子里跑 agent。

    为什么改成入队：钩子是 fire-and-forget（`asyncio.create_task`，不等待）——若在钩子里直接跑 agent
    起草，服务重启/关停时正在进行的起草会**整条丢失、无重试、无留痕**。改为入队一个持久化任务后，
    任务落在 task 表：崩溃丢的只是「入队」这一步的极小窗口，已入队的任务重启后按既有收敛逻辑可重试。
    钩子自身异常仍静默（不改变主任务结果）。

    触发规则：白名单任务 → 入队 `knowledge_distill`（从 run_record 蒸馏）；
    论文结论任务 → 入队 `paper_distill`（从论文实验条目/结论蒸馏）。
    """
    task = task_manager.get_task(task_id)
    if not task:
        return
    task_type = task.get("task_type")
    if task_type in DISTILL_TASK_TYPES and not knowledge_service.knowledge_for_task_exists(task_id):
        task_manager.create_task(KNOWLEDGE_DISTILL_TASK_TYPE, project_id=task.get("project_id"),
                                 params={"source_task_id": task_id})
    if task_type in PAPER_DISTILL_TRIGGERS:
        paper_id = knowledge_service._as_dict(task.get("params")).get("paper_id")
        if paper_id and not knowledge_service.knowledge_for_paper_exists(paper_id):
            task_manager.create_task(PAPER_DISTILL_TASK_TYPE, params={"paper_ids": [paper_id]})


async def _draft_from_run(run: dict, source_task_id: str) -> list[str]:
    """从一条运行记录起草蒸馏知识（draft）；返回新建的 knowledge_id 列表。"""
    params = knowledge_service._as_dict(run.get("params"))
    try:
        related = knowledge_service.bring_advice_summary(
            task_type=params.get("task_type"), model=params.get("model"), dataset=params.get("dataset"))
    except Exception:  # noqa: BLE001 —— 带入失败不影响起草
        related = {}

    result = await agent_service.run_sync(_prompt(run, related), output_schema=DISTILL_SCHEMA, timeout_s=180)
    ids: list[str] = []
    for item in _knowledge_items(result.get("structured_output")):
        if not _as_text(item.get("content")).strip():
            continue  # 空结论不落条目（8.3 异常边界）
        structured = dict(item.get("structured") or {})
        structured.setdefault("run_id", run.get("run_id"))
        structured["source_task_id"] = source_task_id
        kid = knowledge_service.record_knowledge({
            "type": _normalize_type(item.get("type")),
            "title": _as_text(item.get("title")) or f"{run.get('run_type')} 蒸馏结论",
            "content": item.get("content"),
            "structured": structured,
            "sources": [{"type": "run_record", "ref": run.get("run_id")}],
            "confidence": _normalize_confidence(item.get("confidence")),
            "scope": item.get("scope") or {},
            "status": "draft",
        })
        ids.append(kid)
        logger.info("蒸馏起草 %s（task=%s run=%s type=%s）", kid, source_task_id, run.get("run_id"), item.get("type"))
    return ids


async def _run_knowledge_distill(params: dict, task_id: str) -> None:
    """knowledge_distill 任务：从 source_task_id 的运行记录起草蒸馏知识。"""
    source_task_id = params.get("source_task_id")
    if not source_task_id:
        raise RuntimeError("knowledge_distill 缺少 source_task_id")
    if knowledge_service.knowledge_for_task_exists(source_task_id):
        return   # 专用路径或前次已起草 → 幂等跳过
    run = knowledge_service.get_run_for_task(source_task_id)
    if run is None:
        return   # 无运行记录 → 无素材
    task_manager.update_progress(task_id, {"stage": "蒸馏起草"})
    ids = await _draft_from_run(run, source_task_id)
    task_manager.update_progress(task_id, {"stage": "完成", "knowledge_ids": ids})


def register() -> None:
    """注册终态钩子 + 蒸馏任务 handler（main lifespan 调用）。"""
    task_manager.register_on_finish(on_task_finished)
    task_manager.register_handler(KNOWLEDGE_DISTILL_TASK_TYPE, _run_knowledge_distill)
    task_manager.register_handler(PAPER_DISTILL_TASK_TYPE, _run_paper_distill)


# ===========================================================================
# 论文蒸馏：从**已有论文数据**（实验条目 + 复现/可信度结论）提炼可复用结论。
# 与「任务后蒸馏」（on_task_finished）并列的另一条素材来源（需求六.1「论文数据」）。
# ===========================================================================

PAPER_DISTILL_TASK_TYPE = "paper_distill"
_PAPER_MAX_ITEMS = 30   # 每人论文最多喂给 agent 的实验条目数（控制上下文）


def _paper_digest(paper_id: str) -> dict:
    """汇总一篇论文的可蒸馏素材（结构化字段；不读整篇 markdown，避免上下文过大）。"""
    paper = knowledge_service.get_item("paper", paper_id)
    if paper is None:
        raise ValueError(f"论文不存在：{paper_id}")
    items = knowledge_service.list_experiment_items(paper_id) or []
    digest = {
        "paper_id": paper_id,
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "experiment_items": [
            {k: it.get(k) for k in ("section_ref", "dataset_name", "metric_name",
                                    "metric_value_reported", "metric_unit", "hyperparams", "baselines")}
            for it in items[:_PAPER_MAX_ITEMS]
        ],
        "n_experiment_items": len(items),
        "reproduction": [
            {"metric_value_actual": r.get("metric_value_actual"), "verdict": r.get("verdict"),
             "deviation": r.get("deviation")}
            for r in (knowledge_service.list_reproduction_results(paper_id) or [])
        ],
        "credibility": None,
    }
    conclusion = knowledge_service.get_credibility_conclusion(paper_id)
    if conclusion:
        digest["credibility"] = {"overall_verdict": conclusion.get("overall_verdict"),
                                 "summary": (conclusion.get("summary") or "")[:1500]}
    return digest


def _paper_prompt(digest: dict) -> str:
    return (
        "你是学术论文的**知识蒸馏**助手。请从下面这篇论文的结构化实验数据中，提炼**可复用**的结论，"
        "供后续类似研究参考（例如：该方法的适用场景、在哪些数据/设置下效果好、"
        "哪条实验结论在本机环境复现不一致及可能原因）。\n\n"
        "## 论文结构化数据（JSON）\n"
        f"{json.dumps(digest, ensure_ascii=False, indent=2)}\n\n"
        "## 要求\n"
        "1) 只提炼可复用结论——`usage_guidance`（方法/使用建议）、`param_advice`（参数/设置建议）、"
        "`reproduction_discrepancy`（本环境复现不一致，仅在确有 reproduction/credibility 证据时）；\n"
        "2) 每条给 type/title/content 与 confidence，必要时给 structured；\n"
        "3) **scope 尽量填**：能推断出 model（如该论文提出的方法名）/ dataset / task_type 就写入 scope；\n"
        "4) 不要复述论文流水账；材料不足或无可复用结论时返回**空数组**。"
    )


async def distill_paper(paper_id: str) -> list[str]:
    """从一篇已有论文提炼蒸馏知识（draft）；已蒸馏过则跳过；无可蒸馏素材返回空。"""
    if knowledge_service.knowledge_for_paper_exists(paper_id):
        return []
    digest = _paper_digest(paper_id)
    if not digest["experiment_items"] and not digest["credibility"] and not digest["reproduction"]:
        return []   # 无素材（未抽取、未复现）→ 不蒸馏
    result = await agent_service.run_sync(_paper_prompt(digest), output_schema=DISTILL_SCHEMA, timeout_s=180)
    ids: list[str] = []
    for item in _knowledge_items(result.get("structured_output")):
        if not _as_text(item.get("content")).strip():
            continue
        structured = dict(item.get("structured") or {})
        structured["source_paper_id"] = paper_id
        kid = knowledge_service.record_knowledge({
            # 论文蒸馏的结论只能是使用建议/参数建议/复现不一致（fusion_insight 属多模型分析专用）
            "type": _normalize_type(item.get("type"),
                                    allowed=("usage_guidance", "param_advice", "reproduction_discrepancy")),
            "title": _as_text(item.get("title")) or f"论文蒸馏：{digest['title'] or paper_id}",
            "content": item.get("content"),
            "structured": structured,
            "sources": [{"type": "paper", "ref": paper_id}],
            "confidence": _normalize_confidence(item.get("confidence")),
            "scope": item.get("scope") or {},
            "status": "draft",
        })
        ids.append(kid)
    logger.info("论文蒸馏 %s → %d 条", paper_id, len(ids))
    return ids


async def distill_papers(paper_ids: list[str]) -> dict:
    """批量论文蒸馏（顺序执行）；返回 {paper_id: [knowledge_id...], "total": n}。"""
    result: dict = {"total": 0, "papers": {}}
    for pid in paper_ids:
        try:
            ids = await distill_paper(pid)
        except Exception as exc:  # noqa: BLE001 —— 单篇失败不影响其余
            logger.exception("论文蒸馏失败 %s", pid)
            result["papers"][pid] = {"error": str(exc)}
            continue
        result["papers"][pid] = ids
        result["total"] += len(ids)
    return result


async def _run_paper_distill(params: dict, task_id: str) -> None:
    """paper_distill 任务入口：对 params.paper_ids（缺省=全部论文）逐篇蒸馏。"""
    paper_ids = params.get("paper_ids") or knowledge_service.list_paper_ids()
    task_manager.update_progress(task_id, {"stage": "论文蒸馏", "n_papers": len(paper_ids)})
    result = await distill_papers(paper_ids)
    task_manager.update_progress(task_id, {"stage": "完成", **result})


def start_paper_distill(paper_ids: Optional[list] = None) -> str:
    """发起论文蒸馏任务，返回 task_id。"""
    return task_manager.create_task(PAPER_DISTILL_TASK_TYPE, params={"paper_ids": paper_ids or []})
