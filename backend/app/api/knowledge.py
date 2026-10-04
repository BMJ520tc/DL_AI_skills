"""知识库 API（数据设计十；模块详细设计 2.6）。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import knowledge_service

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class IngestBody(BaseModel):
    data_type: str
    data: dict


@router.post("/ingest")
def ingest(body: IngestBody) -> dict:
    if body.data_type == "run":
        ref_id = knowledge_service.record_run(body.data)
    elif body.data_type == "paper":
        ref_id = knowledge_service.record_paper(body.data)
    elif body.data_type == "dataset":
        ref_id = knowledge_service.register_dataset(body.data)
    elif body.data_type == "knowledge":
        ref_id = knowledge_service.record_knowledge(body.data)
    else:
        raise HTTPException(status_code=501, detail=f"data_type={body.data_type} ingest not implemented yet")
    return {"data_type": body.data_type, "ref_id": ref_id}


@router.get("/search")
def search(
    types: str | None = None,
    task_type: str | None = None,
    model: str | None = None,
    dataset: str | None = None,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    type_list = types.split(",") if types else None
    return knowledge_service.search(type_list, task_type, model, dataset, q, limit, offset)


@router.get("/items/{data_type}/{ref_id:path}")
def get_item(data_type: str, ref_id: str) -> dict:
    """按主键取条目。`ref_id` 用 `:path` 形态：外部 id 可能是含 `/` 的 DOI
    （如 bioRxiv 的 `10.1101/2023.10.03.560734`），单段参数在路由层就匹配不到。"""
    try:
        item = knowledge_service.get_item(data_type, ref_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    return item


@router.get("/list")
def list_items(data_type: str, limit: int = 100, offset: int = 0) -> list[dict]:
    try:
        return knowledge_service.list_items(data_type, limit, offset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/confirm/{knowledge_id}")
def confirm_knowledge(knowledge_id: str) -> dict:
    if not knowledge_service.confirm_knowledge(knowledge_id):
        raise HTTPException(status_code=404, detail="knowledge not found or not in draft")
    return {"knowledge_id": knowledge_id, "status": "confirmed"}


@router.delete("/items/{data_type}/{ref_id:path}")
def delete_item(data_type: str, ref_id: str) -> dict:
    """按主键删除条目（`ref_id` 同样用 `:path` 形态以支持含 `/` 的 DOI）。"""
    try:
        deleted = knowledge_service.delete_item(data_type, ref_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ReferenceError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="item not found")
    return {"data_type": data_type, "ref_id": ref_id, "deleted": True}


@router.post("/bring")
def bring_knowledge(
    task_type: str | None = None,
    model: str | None = None,
    dataset: str | None = None,
) -> dict:
    return knowledge_service.bring_knowledge(task_type, model, dataset)
