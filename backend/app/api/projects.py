"""项目管理 API（模块详细设计 2.2）。"""
import json
from pathlib import Path

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


def _require_structured(project_id: str) -> dict:
    try:
        return project_manager.require_type(project_id, {"structured"})
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _graph_path(project: dict) -> Path:
    return Path(project["workspace_path"]) / "graph.json"


@router.get("/{project_id}/graph")
def get_graph(project_id: str) -> dict:
    """结构化项目画布快照（GraphIR v2，模块四 6.5/7.1）。"""
    project = _require_structured(project_id)
    p = _graph_path(project)
    if not p.exists():
        raise HTTPException(status_code=404, detail="graph not found")
    return json.loads(p.read_text(encoding="utf-8"))


@router.put("/{project_id}/graph")
def put_graph(project_id: str, body: dict) -> dict:
    """画布保存（GraphIR v2 全量覆盖，模块四 7.1 最小闭环；版本树归阶段4）。"""
    project = _require_structured(project_id)
    if not isinstance(body.get("nodes"), list) or not isinstance(body.get("edges"), list):
        raise HTTPException(status_code=400, detail="非法 GraphIR：需含 nodes/edges 数组")
    _graph_path(project).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    project_manager.update_status(project_id, "ready")
    return {"status": "saved"}
