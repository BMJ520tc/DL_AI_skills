"""标准化模块 API（模块详细设计 6.5，数据设计七.1）。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import decompose_service, knowledge_service

router = APIRouter(prefix="/api/modules", tags=["modules"])


class IngestBody(BaseModel):
    project_id: str


@router.post("")
def ingest(body: IngestBody) -> dict:
    """拆解验证通过的模块入库（3.5 前置：IR 存在 → 验证 passed → ir_hash 未 stale）。"""
    try:
        precheck = decompose_service.ingest_precheck(body.project_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    task_id = decompose_service.ingest(body.project_id)
    return {"task_id": task_id, "status": "queued", **precheck}


@router.get("")
def list_modules() -> list[dict]:
    return knowledge_service.list_modules()


@router.get("/{module_id}")
def get_module(module_id: str, module_version: str | None = None) -> dict:
    m = knowledge_service.get_module(module_id, module_version)
    if m is None:
        raise HTTPException(status_code=404, detail="module not found")
    return m
