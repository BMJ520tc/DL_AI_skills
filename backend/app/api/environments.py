"""环境管理 API（模块详细设计 2.3）。"""
from fastapi import APIRouter, HTTPException

from app.services import env_manager, project_manager

router = APIRouter(prefix="/api/projects/{project_id}/env", tags=["environments"])


@router.post("")
def create_env(project_id: str) -> dict:
    try:
        project_manager.require_type(project_id, {"original"})
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    task_id = env_manager.create_env(project_id)
    return {"task_id": task_id, "status": "queued"}


@router.get("")
def get_env_status(project_id: str) -> dict:
    return env_manager.get_env_status(project_id)
