"""模块二 API（模块详细设计四章 4.1~4.4）。

路径参数一律用 `{paper_id:path}`：bioRxiv/medRxiv 检索返回的 paper_id 是**含 `/` 的真实 DOI**
（`10.1101/2023.10.03.560734`），默认的单段路径参数匹配不到它，接口会在路由层就 404，
后面的解析/抽取/复现全都走不到。带 `/` 的 id 仍由 `ids.safe_id` 逐段校验、由 `ids.fs_name`
落成安全目录名（见 ids.py）；路由顺序上把带固定后缀的接口放在最前，最后的兜底 GET 才拿整个
`{paper_id:path}`，避免兜底把 `/items`、`/conclusion` 这些接口吃掉。
"""
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import knowledge_service, paper_service

router = APIRouter(prefix="/api/papers", tags=["papers"])


def _require_paper(paper_id: str) -> None:
    if knowledge_service.get_item("paper", paper_id) is None:
        raise HTTPException(status_code=404, detail=f"paper not found: {paper_id}")


@router.post("/{paper_id:path}/parse")
def parse(paper_id: str) -> dict:
    """4.1 PDF → markdown（规则 + agent 修正）。"""
    try:
        task_id = paper_service.parse(paper_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"task_id": task_id, "status": "queued"}


@router.post("/{paper_id:path}/extract")
def extract(paper_id: str) -> dict:
    """4.2 实验条目抽取（agent）。"""
    try:
        task_id = paper_service.extract_items(paper_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"task_id": task_id, "status": "queued"}


@router.get("/{paper_id:path}/items")
def list_items(paper_id: str) -> list[dict]:
    _require_paper(paper_id)
    return knowledge_service.list_experiment_items(paper_id)


class ItemEdit(BaseModel):
    section_ref: str | None = None
    dataset_name: str | None = None
    split_method: str | None = None
    metric_name: str | None = None
    metric_value_reported: str | None = None
    metric_unit: str | None = None
    hyperparams: dict | None = None
    baselines: list | None = None
    status: str | None = None


@router.put("/{paper_id:path}/items/{item_id}")
def edit_item(paper_id: str, item_id: str, body: ItemEdit) -> dict:
    """4.2 用户编辑实验条目（确认后条目生效）。"""
    item = knowledge_service.get_experiment_item(item_id)
    if item is None or item["paper_id"] != paper_id:
        raise HTTPException(status_code=404, detail=f"item not found: {item_id}")
    fields = body.model_dump(exclude_unset=True)
    if not knowledge_service.update_experiment_item(item_id, **fields):
        raise HTTPException(status_code=400, detail="no fields to update")
    return knowledge_service.get_experiment_item(item_id)


@router.post("/{paper_id:path}/items/{item_id}/confirm")
def confirm_item(paper_id: str, item_id: str) -> dict:
    """4.2 确认实验条目（extracted → confirmed）；仅已确认条目参与复现。"""
    item = knowledge_service.get_experiment_item(item_id)
    if item is None or item["paper_id"] != paper_id:
        raise HTTPException(status_code=404, detail=f"item not found: {item_id}")
    if not knowledge_service.confirm_experiment_item(item_id):
        raise HTTPException(status_code=409, detail="item 不处于可确认状态")
    return {"item_id": item_id, "status": "confirmed"}


class ReproduceBody(BaseModel):
    project_id: str


@router.post("/{paper_id:path}/reproduce")
def reproduce(paper_id: str, body: ReproduceBody) -> dict:
    """4.3 自动复现执行（project_id 提供模块一独立环境）。"""
    try:
        task_id = paper_service.reproduce(paper_id, body.project_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"task_id": task_id, "status": "queued"}


@router.get("/{paper_id:path}/reproduce")
def get_reproduce(paper_id: str) -> list[dict]:
    """4.3 复现结果查询（逐条对照）。"""
    _require_paper(paper_id)
    return knowledge_service.list_reproduction_results(paper_id)


@router.post("/{paper_id:path}/conclusion")
def create_conclusion(paper_id: str) -> dict:
    """4.4 逐条对照与可信度结论（固定代码分档 + agent 起草）。"""
    try:
        task_id = paper_service.conclusion(paper_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"task_id": task_id, "status": "queued"}


class ConclusionEdit(BaseModel):
    overall_verdict: str | None = None
    summary: str | None = None


@router.put("/{paper_id:path}/conclusion")
def confirm_conclusion(paper_id: str, body: ConclusionEdit) -> dict:
    """4.4 用户确认或修改可信度结论。"""
    _require_paper(paper_id)
    current = paper_service.get_conclusion(paper_id)
    if current is None:
        raise HTTPException(status_code=404, detail="conclusion not found")
    item_results = json.loads(current.get("item_results") or "[]")
    conclusion_id = knowledge_service.record_credibility_conclusion(paper_id, {
        "overall_verdict": body.overall_verdict or current["overall_verdict"],
        "summary": body.summary or current["summary"],
        "item_results": item_results,
    })
    return {"conclusion_id": conclusion_id, "status": "confirmed"}


@router.get("/{paper_id:path}/conclusion")
def get_conclusion(paper_id: str) -> dict:
    _require_paper(paper_id)
    conclusion = paper_service.get_conclusion(paper_id)
    if conclusion is None:
        raise HTTPException(status_code=404, detail="conclusion not found")
    return conclusion


@router.get("/{paper_id:path}")
def get_paper(paper_id: str) -> dict:
    """论文详情：记录 + 条目 + 复现对照 + 结论（4.2~4.4 汇总视图）。

    放在本模块最后：`{paper_id:path}` 会吃掉 `/api/papers/` 之后的全部内容，
    必须先让上面带固定后缀的接口（/parse、/items、/conclusion …）拿到匹配。
    """
    try:
        return paper_service.get_detail(paper_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
