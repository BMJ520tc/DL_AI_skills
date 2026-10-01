"""数据预处理 API（模块详细设计 5.1）。

接口按 5.1 约定为 /api/preprocess（不带 project 前缀）。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import preprocess_service, task_manager

router = APIRouter(prefix="/api/preprocess", tags=["preprocess"])


class PreprocessBody(BaseModel):
    input_path: str
    dataset_name: str | None = None
    task_type: str = "classification"
    project_id: str | None = None


@router.post("")
def create_preprocess(body: PreprocessBody) -> dict:
    try:
        task_id = preprocess_service.create_preprocess(
            body.input_path, body.dataset_name, body.task_type, body.project_id
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"task_id": task_id, "status": "queued"}


@router.get("/{task_id}")
def get_preprocess(task_id: str) -> dict:
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    result = preprocess_service.get_result(task_id)
    return {"task_id": task_id, "status": task["status"], "error": task["error"], "result": result}
