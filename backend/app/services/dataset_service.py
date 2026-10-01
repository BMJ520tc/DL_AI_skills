"""模块三 5.3 公开数据检索与跨数据集对齐（模块详细设计 5.3）。

流程：数据不足判定 → 检索（dataset_registry + 外部源）→ 下载登记（复用 2.4）
      → 对齐（字段映射、序列长度、标签体系，agent 起草）→ 更新 dataset_registry.alignment。
对齐记录按项目复用：同一数据集对同一项目只对齐一次。
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from app.config import DATASETS_DIR
from app.services import agent_service, download_service, knowledge_service, project_manager, task_manager

TASK_TYPE = "align"
EXTERNAL_TIMEOUT_S = 8

ALIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "field_mapping": {"type": "object", "description": "目标数据集字段 → 统一字段 id/split/label/input"},
        "label_merge": {"type": "object", "description": "目标标签 → 归并后的标签"},
        "sequence": {"type": "object", "description": "序列长度范围与大小写规范"},
        "notes": {"type": "string", "description": "对齐时的注意事项"},
    },
    "required": ["field_mapping"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- 检索 ----------

# 自带数据是否充足的判定阈值（5.3 步骤 1「数据不足判定」）
SUFFICIENT_MIN_SAMPLES = 100


def assess_self_data(project_id: str) -> dict:
    """数据不足判定（5.3 步骤 1）：按自带数据集的样本数对比阈值。"""
    ws = Path(project_manager.get_project(project_id)["workspace_path"])
    found = knowledge_service.find_datasets(local_path_prefix=str(ws / "data"), limit=1)
    if not found:
        return {"sufficient": False, "n_samples": 0, "threshold": SUFFICIENT_MIN_SAMPLES,
                "hint": "尚未预处理自带数据，请先调用 POST /api/preprocess"}

    local_path = found[0].get("local_path")
    n_samples = 0
    if local_path and Path(local_path).exists():
        with Path(local_path).open(encoding="utf-8", newline="") as f:
            n_samples = max(0, sum(1 for _ in csv.reader(f)) - 1)  # 减去表头

    sufficient = n_samples >= SUFFICIENT_MIN_SAMPLES
    return {
        "sufficient": sufficient,
        "n_samples": n_samples,
        "threshold": SUFFICIENT_MIN_SAMPLES,
        "hint": None if sufficient else
                f"自带数据仅 {n_samples} 条（阈值 {SUFFICIENT_MIN_SAMPLES}），建议检索公开数据补充",
    }


def search_datasets(
    task_type: str | None = None,
    format: str | None = None,
    q: str | None = None,
    limit: int = 20,
    project_id: str | None = None,
) -> dict:
    """检索可用数据集：本地 registry 为主，外部源（Zenodo）尽力而为。

    传入 project_id 时先做数据不足判定，并把项目自带数据集从 local 中排除
    （5.3 要检索的是「自带之外的可用公开数据」）。
    """
    local = knowledge_service.find_datasets(task_type=task_type, format=format, limit=limit)
    if q:
        needle = q.lower()
        local = [d for d in local if needle in (d.get("name") or "").lower()
                 or needle in (d.get("task_type") or "").lower()]

    result: dict = {"local": local, "external": search_external(q or task_type or "dataset", limit=5)}
    if project_id:
        ws = Path(project_manager.get_project(project_id)["workspace_path"])
        self_found = knowledge_service.find_datasets(local_path_prefix=str(ws / "data"), limit=1)
        self_id = self_found[0]["dataset_id"] if self_found else None
        result["self_data"] = assess_self_data(project_id)
        result["local"] = [d for d in local if d["dataset_id"] != self_id]

    if not result["local"] and not result["external"]:
        # 5.3 异常与边界：无可用公开数据 → 提示并结束该步骤
        result["hint"] = "无可用公开数据，该步骤结束"
    return result


def search_external(query: str, limit: int = 5) -> list[dict]:
    """Zenodo 公开数据集检索；网络不可用时返回空列表（不阻断流程）。"""
    try:
        url = f"https://zenodo.org/api/records?q={quote(str(query))}&size={limit}&type=dataset"
        data = json.loads(download_service._http_get(url, timeout=EXTERNAL_TIMEOUT_S))
    except Exception:  # noqa: BLE001
        return []
    hits = []
    for hit in data.get("hits", {}).get("hits", [])[:limit]:
        meta = hit.get("metadata", {})
        hits.append({
            "dataset_id": None,
            "name": meta.get("title"),
            "url": hit.get("links", {}).get("self_html"),
            "source": "zenodo",
            "source_id": str(hit.get("id")),
            "task_type": None,
            "format": None,
        })
    return hits


# 外部数据源的人类可读地址（5.3 来源溯源：dataset_registry.url 需可回访）
_SOURCE_URL_TEMPLATES = {
    "zenodo": "https://zenodo.org/records/{id}",
    "figshare": "https://figshare.com/articles/{id}",
}


def source_url(source: str, source_id: str) -> str | None:
    tpl = _SOURCE_URL_TEMPLATES.get(source)
    return tpl.format(id=source_id) if tpl else None


def download_dataset(source: str, source_id: str, name: str, task_type: str | None = None) -> str:
    """下载外部数据集并登记（复用 2.4 的 download_dataset），返回 dataset_id。"""
    dest = DATASETS_DIR / f"{source}_{source_id}"
    download_service.download_dataset(source, source_id, dest)
    return knowledge_service.register_dataset({
        "name": name,
        "url": source_url(source, source_id),
        "source": source,
        "task_type": task_type,
        "format": None,
        "local_path": str(dest),
        "fields": None,
        "labels": None,
        "alignment": None,
    })


# ---------- 对齐 ----------

def create_align(project_id: str, dataset_id: str) -> str:
    project_manager.require_type(project_id, {"original"})
    return task_manager.create_task(
        TASK_TYPE, project_id=project_id,
        params={"project_id": project_id, "dataset_id": dataset_id},
    )


def confirm_alignment(dataset_id: str) -> bool:
    """对齐确认：draft → confirmed（模块详细设计 5.3「对齐规则由 agent 起草、用户确认」）。

    对齐记录会被后续评估自动复用（5.3 关键设计），故复用前必须经确认闸门。
    """
    alignment = get_alignment(dataset_id)
    if alignment is None:
        return False
    if alignment.get("status") == "confirmed":
        return True
    alignment["status"] = "confirmed"
    alignment["confirmed_at"] = _now()
    return knowledge_service.update_alignment(dataset_id, alignment)


def get_alignment(dataset_id: str) -> dict | None:
    item = knowledge_service.get_item("dataset", dataset_id)
    if item is None:
        return None
    raw = item.get("alignment")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def _as_json(value):
    """dataset_registry 的 fields/labels/alignment 以 JSON 文本存库，取值时统一解析。

    `knowledge_service.get_item` 返回的是原始行（TEXT），若直接把该字符串当列表遍历，
    只会逐字符走一遍——规则兜底会因此判定「缺少字段信息」并失败（真实数据集 E1 暴露）。
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def find_self_dataset(ws: Path) -> dict | None:
    """项目「自带数据」数据集（5.3 对齐与 5.4 对比的来源侧）。

    预处理产物统一落在 ws/data 下，外部公开数据集经 5.1 预处理后同样如此，
    因此必须排除非「自带」来源，否则会把外部数据集当成基准来源
    （E1 用真实公开数据集暴露：来源与目标退化成同一条记录）。
    """
    found = knowledge_service.find_datasets(local_path_prefix=str(ws / "data"), limit=20)
    for ds in found:
        if (ds.get("source") or "自带") == "自带":
            return ds
    return None


# 兼容既有内部调用名
_self_dataset = find_self_dataset


async def _draft_alignment(source_ds: dict, target_ds: dict, ws: Path) -> dict:
    prompt = (
        "两个数据集需要做跨数据集对齐，以保证评估口径可比（统一输出格式、标签体系、评价指标）。\n\n"
        f"【来源数据集（项目自带）】\n字段: {json.dumps(_as_json(source_ds.get('fields')), ensure_ascii=False)}\n"
        f"标签: {json.dumps(_as_json(source_ds.get('labels')), ensure_ascii=False)}\n"
        f"已有对齐: {json.dumps(_as_json(source_ds.get('alignment')), ensure_ascii=False)}\n\n"
        f"【目标数据集（待对齐）】\n名称: {target_ds.get('name')}\n"
        f"字段: {json.dumps(_as_json(target_ds.get('fields')), ensure_ascii=False)}\n"
        f"标签: {json.dumps(_as_json(target_ds.get('labels')), ensure_ascii=False)}\n\n"
        "请阅读项目代码（如有必要）后给出对齐规则：把目标数据集的字段映射到统一字段"
        "（id/split/label/input），把目标标签归并到与来源一致的标签体系，"
        "并给出序列长度与大小写规范。"
    )
    result = await agent_service.run_sync(prompt, cwd=str(ws / "source"), output_schema=ALIGN_SCHEMA)
    return result.get("structured_output") or {}


UNIFIED_FIELDS = ("id", "split", "label", "input")


def _rule_alignment(source_ds: dict, target_ds: dict) -> dict:
    """agent 未给出可用规则时的固定规则兜底（模块详细设计 5.1「固定规则 + agent 建议」）。

    统一字段同名映射 + 标签按大小写不敏感归并到来源数据集的写法。
    """
    target_fields = _as_json(target_ds.get("fields")) or []
    target_labels = _as_json(target_ds.get("labels")) or []
    source_labels = _as_json(source_ds.get("labels")) or []

    field_mapping = {f: f for f in target_fields if f in UNIFIED_FIELDS}
    canon = {str(lab).strip().casefold(): str(lab).strip() for lab in source_labels}
    label_merge = {}
    for lab in target_labels:
        text = str(lab).strip()
        if text.casefold() in canon and canon[text.casefold()] != text:
            label_merge[text] = canon[text.casefold()]
    return {"field_mapping": field_mapping, "label_merge": label_merge, "sequence": {}, "notes": None}


async def _run(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    dataset_id = params["dataset_id"]
    project = project_manager.get_project(project_id)
    ws = Path(project["workspace_path"])

    target = knowledge_service.get_item("dataset", dataset_id)
    if target is None:
        raise RuntimeError(f"数据集不存在: {dataset_id}")

    existing = get_alignment(dataset_id) or {}
    aligned_projects = existing.get("aligned_projects") or []
    if project_id in aligned_projects:
        # 5.3 关键设计：同一数据集对同一项目只对齐一次，后续评估直接复用
        return

    source_ds = _self_dataset(ws)
    if source_ds is None:
        raise RuntimeError("未找到项目自带数据集，请先调用 POST /api/preprocess")

    draft = await _draft_alignment(source_ds, target, ws)
    drafted_by = "agent"
    if not draft.get("field_mapping") and not draft.get("label_merge"):
        # agent 无有效产出时退回固定规则
        draft, drafted_by = _rule_alignment(source_ds, target), "rule_fallback"
    if not draft.get("field_mapping") and not draft.get("label_merge"):
        # 兜底规则同样无从下手（目标数据集没有字段信息，通常是尚未预处理）：
        # 此处必须失败退出——否则会写入空对齐记录并标记为「已对齐」，
        # 后续 re-align 被「只对齐一次」跳过，问题直到对比阶段才暴露。
        raise RuntimeError(
            f"数据集「{target.get('name') or dataset_id}」缺少字段信息，无法起草对齐规则。"
            "若为外部下载的数据集，请先对其调用 POST /api/preprocess 完成预处理后再对齐。"
        )

    alignment = {
        "version": "1.0",
        "field_mapping": draft.get("field_mapping") or {},
        "label_merge": draft.get("label_merge") or {},
        "sequence": draft.get("sequence") or {},
        "notes": draft.get("notes"),
        "drafted_by": drafted_by,
        "source_dataset_id": source_ds["dataset_id"],
        "status": "draft",
        "aligned_projects": aligned_projects + [project_id],
        "aligned_at": _now(),
    }
    knowledge_service.update_alignment(dataset_id, alignment)


def register() -> None:
    task_manager.register_handler(TASK_TYPE, _run)
