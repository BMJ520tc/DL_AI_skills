"""模块三 5.3 公开数据检索与跨数据集对齐（模块详细设计 5.3）。

流程：数据不足判定 → 检索（dataset_registry + 外部源）→ 下载登记（复用 2.4）
      → 对齐（字段映射、序列长度、标签体系，agent 起草）→ 更新 dataset_registry.alignment。
对齐记录按项目复用：同一数据集对同一项目只对齐一次。
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from app.config import DATASETS_DIR
from app.ids import fs_name, safe_id
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


# ---------- 数据类型（format）口径与推断 ----------

# dataset_registry.format 的取值口径与 `scripts/preprocess_dataset.py::detect_format` 对齐
# （csv / excel / xls / image_dir / archive / fasta / pdb ...），这样「按数据类型检索」在不同来源
# 之间可比；未收录的扩展名不猜，留给 infer_format 如实说明。
_EXT_FORMAT = {
    ".csv": "csv", ".tsv": "csv",
    ".xlsx": "excel", ".xls": "xls",
    ".fasta": "fasta", ".fa": "fasta", ".fna": "fasta",
    ".pdb": "pdb", ".ent": "pdb",
    ".zip": "archive", ".tar": "archive", ".gz": "archive", ".tgz": "archive",
    ".rar": "archive", ".7z": "archive", ".bz2": "archive",
    ".png": "image_dir", ".jpg": "image_dir", ".jpeg": "image_dir", ".bmp": "image_dir",
    ".tif": "image_dir", ".tiff": "image_dir", ".gif": "image_dir",
    ".json": "json", ".txt": "txt", ".xml": "xml", ".npy": "npy", ".npz": "npz",
}


def _normalize_format(value: str | None) -> str | None:
    """规范化 format 取值（去空白 + 小写）：登记与检索用同一口径，大小写差异不会漏命中。"""
    token = (value or "").strip().lower()
    return token or None


def infer_format(files: list[Path]) -> tuple[str | None, str]:
    """从下载到（或外部源返回）的文件名后缀推断数据类型；返回 `(format, 说明)`。

    只做**如实的扩展名推断**（写进 `dataset_registry.format` 的就是这里推出来的 token）；
    多种扩展名混合、有未收录扩展名、或压根没有文件名时一律返回 `(None, 原因)`——
    format 留空并把原因写进说明，绝不猜。只看后缀、不要求文件真的存在
    （外部源返回条目里只有文件名，没有本地文件）。
    """
    exts = sorted({Path(str(p)).suffix.lower() for p in files if Path(str(p)).suffix})
    if not exts:
        return None, "下载未返回带扩展名的文件，无法推断数据类型，format 留空"
    mapped = sorted({_EXT_FORMAT[ext] for ext in exts if ext in _EXT_FORMAT})
    unknown = [ext for ext in exts if ext not in _EXT_FORMAT]
    if len(mapped) == 1 and not unknown:
        return mapped[0], f"按下载文件的扩展名推断：{'/'.join(exts)} → {mapped[0]}"
    return None, (
        f"下载到多种或未收录的扩展名（{'、'.join(exts)}），无法推断单一数据类型，format 留空"
    )


def search_datasets(
    task_type: str | None = None,
    format: str | None = None,
    q: str | None = None,
    limit: int = 20,
    project_id: str | None = None,
) -> dict:
    """检索可用数据集：本地 registry 为主，外部源（Zenodo）尽力而为。

    **按任务类型 / 按数据类型检索的落地口径**（需求三.2）：

    - `task_type`、`format` 对**已登记数据集**按 `dataset_registry.task_type` /
      `dataset_registry.format` 两列精确匹配（经 `knowledge_service.find_datasets`）；
      `format` 先去空白转小写再匹配，与登记时的口径一致，避免「CSV」/「csv」互相漏命中。
    - 外部源（Zenodo）只有资源类型与文件类型分面，**没有任务类型字段**：task_type 不下推，
      format 尽力下推，做不到的部分写进每条外部结果的 `filter_note`（见 `search_external`），
      不假装过滤成功。

    传入 project_id 时先做数据不足判定，并把项目自带数据集从 local 中排除
    （5.3 要检索的是「自带之外的可用公开数据」）。
    """
    fmt = _normalize_format(format)
    local = knowledge_service.find_datasets(task_type=task_type, format=fmt, limit=limit)
    if q:
        # 关键词覆盖名称/任务类型/数据类型（登记表实际列名见 db/schema.sql dataset_registry）
        needle = q.lower()
        local = [d for d in local if any(
            needle in str(d.get(col) or "").lower() for col in ("name", "task_type", "format")
        )]

    result: dict = {
        "local": local,
        "external": search_external(
            q or task_type or "dataset", limit=5, task_type=task_type, format=fmt
        ),
    }
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


def search_external(
    query: str,
    limit: int = 5,
    *,
    task_type: str | None = None,
    format: str | None = None,
) -> list[dict]:
    """Zenodo 公开数据集检索；过滤按外部源的**真实能力**下推，做不到的写进 `filter_note`。

    - `type=dataset`：资源类型分面（既有做法），只取数据集类记录；
    - `format`：映射到 Zenodo 的文件类型分面 `file_type` 下推一次（映射表见 `_ZENODO_FILE_TYPES`），
      请求失败就退回不带该参数的检索（退回原因进 `filter_note`）；返回条目再按记录的
      `files[].key` 扩展名做**客户端复核**——这一步只作用于本次返回的这几条，不是服务端全量过滤，
      `filter_note` 里如实写明；
    - `task_type`：Zenodo 没有任务类型字段，**不下推**，只在 `filter_note` 里说明
      （任务类型过滤只对本地 dataset_registry 条目生效）。

    网络不可用/取数失败返回空列表（不阻断流程）。
    """
    fmt = _normalize_format(format)
    file_type = _zenodo_file_type(fmt)
    notes: list[str] = []
    if task_type:
        notes.append(
            f"外部源（Zenodo）没有任务类型字段：task_type={task_type!r} 未下推，"
            "本页结果未按任务类型过滤（任务类型过滤只对本地已登记数据集生效）"
        )

    data = _zenodo_search(query, limit, file_type=file_type)
    pushed, fallback = file_type, False
    if data is None and file_type:
        data = _zenodo_search(query, limit, file_type=None)
        if data is not None:
            pushed, fallback = None, True
    if data is None:
        return []

    hits_raw = ((data.get("hits") or {}).get("hits") or [])[:limit]
    entries: list[dict] = []
    missing_files = 0
    for hit in hits_raw:
        meta = hit.get("metadata") or {}
        hit_format, has_files = _hit_format(hit)
        if not has_files:
            # 没有文件清单就没法复核格式：如实保留该条，并在说明里交代
            missing_files += 1
        elif fmt and hit_format != fmt:
            continue  # 客户端复核：扩展名推不出请求的数据类型 → 排除
        entries.append({
            "dataset_id": None,
            "name": meta.get("title"),
            "url": (hit.get("links") or {}).get("self_html"),
            "source": "zenodo",
            "source_id": str(hit.get("id")),
            "task_type": None,
            "format": hit_format,
        })

    if fmt and pushed:
        notes.append(
            f"format={fmt} 已按 file_type={pushed} 下推，并按返回条目 files[].key 的扩展名做客户端复核"
            f"（只作用于本次返回的 {len(hits_raw)} 条，不是服务端全量过滤）"
        )
    elif fmt and fallback:
        notes.append(
            f"file_type={file_type} 下推请求失败（外部源不可用或参数被拒），已退回不带该参数的检索："
            "服务端未按格式过滤，仅按返回条目的扩展名做客户端复核"
        )
    elif fmt:
        notes.append(
            f"format={fmt} 未下推（Zenodo 文件类型词表里没有确定对应的取值），"
            "仅按返回条目的文件扩展名做客户端复核"
        )
    if fmt and missing_files:
        notes.append(f"{missing_files} 条外部结果没有文件清单，未做扩展名复核（如实保留，未按格式排除）")

    if notes:
        # 与 bioRxiv 检索同风格：只在**确实有做不到的过滤**时给每条结果附说明，不静默
        for entry in entries:
            entry["filter_note"] = "；".join(notes)
    return entries


_ZENODO_RECORDS_API = "https://zenodo.org/api/records"

# 可下推到 Zenodo 文件类型分面（file_type）的 format 取值：只收「文件扩展名就是该类型」的映射。
# 目录型（image_dir）、压缩包类（archive）、生物序列类（fasta/pdb）没有确定对应的分面取值，
# 一律不下推——宁可不推（并在 filter_note 里说明），也不发一个服务端不认识的参数值。
_ZENODO_FILE_TYPES = {
    "csv": "csv", "tsv": "tsv", "json": "json", "txt": "txt", "xml": "xml",
    "excel": "xlsx", "xlsx": "xlsx", "xls": "xls", "pdf": "pdf",
    "png": "png", "jpg": "jpg", "jpeg": "jpeg", "tiff": "tiff", "gif": "gif", "zip": "zip",
}


def _zenodo_file_type(fmt: str | None) -> str | None:
    return _ZENODO_FILE_TYPES.get(_normalize_format(fmt) or "")


def _zenodo_search(query: str, limit: int, file_type: str | None) -> dict | None:
    """打一次 Zenodo 记录检索；失败返回 None（调用方决定是否退回、是否放弃）。"""
    params = {"q": str(query), "size": str(limit), "type": "dataset"}
    if file_type:
        params["file_type"] = file_type
    try:
        return json.loads(
            download_service._http_get(
                f"{_ZENODO_RECORDS_API}?{urlencode(params)}", timeout=EXTERNAL_TIMEOUT_S
            )
        )
    except Exception:  # noqa: BLE001 —— 外部源不可用不阻断流程
        return None


def _hit_format(hit: dict) -> tuple[str | None, bool]:
    """从 Zenodo 记录的 `files[].key` 扩展名推断该条目的数据类型。

    返回 `(format, 是否有文件清单)`：没有文件清单时 format 为 None 且第二项为 False
    （调用方据此决定是否做格式排除，不猜）。
    """
    keys = [str((f or {}).get("key") or "") for f in (hit.get("files") or [])]
    paths = [Path(key) for key in keys if key]
    if not paths:
        return None, False
    inferred, _ = infer_format(paths)
    return inferred, True


# 外部数据源的人类可读地址（5.3 来源溯源：dataset_registry.url 需可回访）
_SOURCE_URL_TEMPLATES = {
    "zenodo": "https://zenodo.org/records/{id}",
    "figshare": "https://figshare.com/articles/{id}",
}


def source_url(source: str, source_id: str) -> str | None:
    tpl = _SOURCE_URL_TEMPLATES.get(source)
    return tpl.format(id=source_id) if tpl else None


def download_dataset(
    source: str,
    source_id: str,
    name: str,
    task_type: str | None = None,
    format: str | None = None,
) -> str:
    """下载外部数据集并登记（复用 2.4 的 download_dataset），返回 dataset_id（既有返回契约）。"""
    return download_dataset_with_info(
        source, source_id, name, task_type=task_type, format=format
    )["dataset_id"]


def download_dataset_with_info(
    source: str,
    source_id: str,
    name: str,
    *,
    task_type: str | None = None,
    format: str | None = None,
) -> dict:
    """下载外部数据集并登记，返回 `{dataset_id, local_path, n_files, format, format_source, format_note}`。

    `format`（数据类型）如实落进 `dataset_registry.format`：
    调用方显式给了就用调用方的（`format_source="caller"`），否则按**下载到的文件扩展名**推断
    （`format_source="extension"`）；推不出来（多种扩展名/未收录/无文件）就留空，
    并把原因写进 `format_note`——不编造。
    """
    safe_id(source, "source")
    safe_id(source_id, "source_id")
    dest = DATASETS_DIR / f"{fs_name(source, 'source')}_{fs_name(source_id, 'source_id')}"
    files = list(download_service.download_dataset(source, source_id, dest) or [])

    requested = _normalize_format(format)
    if requested:
        fmt: str | None = requested
        fmt_source: str | None = "caller"
        fmt_note = f"format 由调用方指定（未从下载内容推断）：{requested}"
    else:
        fmt, fmt_note = infer_format(files)
        fmt_source = "extension" if fmt else None

    dataset_id = knowledge_service.register_dataset({
        "name": name,
        "url": source_url(source, source_id),
        "source": source,
        "task_type": task_type,
        "format": fmt,
        "local_path": str(dest),
        "fields": None,
        "labels": None,
        "alignment": None,
    })
    return {
        "dataset_id": dataset_id,
        "local_path": str(dest),
        "n_files": len(files),
        "format": fmt,
        "format_source": fmt_source,
        "format_note": fmt_note,
    }


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
