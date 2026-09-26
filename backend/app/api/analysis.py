"""模块一 API（模块详细设计 3.5/3.6/3.7）。"""
from fastapi import APIRouter, HTTPException

from app.services import analysis_service, project_manager

router = APIRouter(prefix="/api/projects/{project_id}", tags=["analysis"])


def _require_original(project_id: str) -> None:
    try:
        project_manager.require_type(project_id, {"original"})
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/verify")
def verify(project_id: str) -> dict:
    _require_original(project_id)
    task_id = analysis_service.verify(project_id)
    return {"task_id": task_id, "status": "queued"}


@router.post("/analyze")
def analyze(project_id: str) -> dict:
    _require_original(project_id)
    task_id = analysis_service.analyze(project_id)
    return {"task_id": task_id, "status": "queued"}


@router.get("/report")
def get_report(project_id: str) -> dict:
    report = analysis_service.get_report(project_id)
    if report is None:
        raise HTTPException(status_code=404, detail="report not found")
    return report
