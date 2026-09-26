"""项目管理 API（模块详细设计 2.2）。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import analysis_service, project_manager

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    project_type: str
    source: str | None = None
    name: str | None = None
    parent_project_id: str | None = None
    source_url: str | None = None  # 仓库地址或本地路径，提供则创建后加载（3.3）


@router.post("")
def create_project(body: ProjectCreate) -> dict:
    try:
        project_id = project_manager.create_project(
            body.project_type, body.source, body.name, body.parent_project_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if body.source_url:
        try:
            analysis_service.load_source(project_id, body.source_url)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"加载失败: {e}")

    return {"project_id": project_id, "status": "loading"}


@router.get("")
def list_projects(project_type: str | None = None) -> list[dict]:
    return project_manager.list_projects(project_type)


@router.get("/{project_id}")
def get_project(project_id: str) -> dict:
    project = project_manager.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project
