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
    elif body.data_type == "module":
        # module 表是复合主键 (module_id, module_version)：未给版本时按下一版分配（3.5）。
        data = dict(body.data)
        if data.get("module_id") and not data.get("module_version"):
            data["module_version"] = knowledge_service.next_module_version(data["module_id"])
        res = knowledge_service.record_module(data)
        ref_id = f"{res['module_id']}:{res['module_version']}"
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
def list_items(data_type: str, status: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
    try:
        if data_type == "knowledge":
            # 蒸馏知识支持按状态过滤（草稿确认界面：status=draft）。
            return knowledge_service.list_knowledge(status, None, limit, offset)
        return knowledge_service.list_items(data_type, limit, offset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conflicts/{knowledge_id}")
def list_conflicts(knowledge_id: str) -> list[dict]:
    """该草稿的潜在冲突（同类型 + scope 相容的已确认知识，数据设计六.3）。"""
    if knowledge_service.get_item("knowledge", knowledge_id) is None:
        raise HTTPException(status_code=404, detail="item not found")
    return knowledge_service.find_conflicts(knowledge_id)


@router.post("/confirm/{knowledge_id}")
def confirm_knowledge(knowledge_id: str, supersede: bool = False) -> dict:
    """确认草稿：draft → confirmed；supersede=true 时同时把被推翻的旧 confirmed 置 superseded。"""
    if not knowledge_service.confirm_knowledge(knowledge_id, supersede_conflicts=supersede):
        raise HTTPException(status_code=404, detail="knowledge not found or not in draft")
    return {"knowledge_id": knowledge_id, "status": "confirmed"}


@router.post("/supersede/{knowledge_id}")
def supersede_knowledge(knowledge_id: str) -> dict:
    """把已确认知识置为 superseded（被后续运行推翻，数据设计六.3）。"""
    if not knowledge_service.supersede_knowledge(knowledge_id):
        raise HTTPException(status_code=404, detail="knowledge not found or not confirmed")
    return {"knowledge_id": knowledge_id, "status": "superseded"}


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


class DistillBody(BaseModel):
    paper_ids: list[str] | None = None


@router.post("/distill")
def distill_papers(body: DistillBody) -> dict:
    """论文蒸馏（模块六 8.3、需求六.1「论文数据」素材）：对指定论文（缺省=全部）起草蒸馏知识。

    经任务队列异步执行（逐篇 agent 起草）；结果以 draft 落库，在「知识库 → 草稿」确认。
    """
    from app.services import distill_service
    task_id = distill_service.start_paper_distill(body.paper_ids)
    return {"task_id": task_id, "status": "queued"}
