"""模块一 API（模块详细设计 3.5/3.6/3.7）。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import (
    analysis_service, baseline_service, compare_service, dataset_service,
    project_manager, visualize_service,
)

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


@router.post("/baseline")
def create_baseline(project_id: str) -> dict:
    """模块三 5.2 自带数据基准运行。"""
    _require_original(project_id)
    return {"task_id": baseline_service.create_baseline(project_id), "status": "queued"}


class AlignBody(BaseModel):
    dataset_id: str


@router.post("/datasets/align")
def create_align(project_id: str, body: AlignBody) -> dict:
    """模块三 5.3 跨数据集对齐（同一数据集对同一项目只对齐一次）。"""
    try:
        task_id = dataset_service.create_align(project_id, body.dataset_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"task_id": task_id, "status": "queued"}


@router.post("/compare")
def create_compare(project_id: str) -> dict:
    """模块三 5.4 结果对比与使用建议（产出 usage_guidance 草稿待确认）。"""
    _require_original(project_id)
    return {"task_id": compare_service.create_compare(project_id), "status": "queued"}


@router.post("/visualize/{chart_type}")
async def visualize(project_id: str, chart_type: str) -> dict:
    """模块三 5.5 生成自包含 HTML 图表，返回文件路径。"""
    try:
        return await visualize_service.run(project_id, chart_type)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/baseline")
def get_baseline(project_id: str) -> dict:
    _require_original(project_id)
    run = baseline_service.get_latest_baseline(project_id)
    if run is None:
        raise HTTPException(status_code=404, detail="baseline run not found")
    return run
