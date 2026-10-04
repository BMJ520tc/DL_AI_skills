"""多模型综合分析 API（模块详细设计 8.4，需求六.2）。

命名空间 `/api/multi-model`，与既有模块一的 `/api/projects/{id}/analysis/*` 隔离（八.4 命名冲突提示）。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import multi_model_service

router = APIRouter(prefix="/api/multi-model", tags=["multi-model"])


class MultiModelBody(BaseModel):
    run_ids: list[str]
    labels: list | None = None
    fusion: str | None = None
    task_type: str | None = None
    dataset_id: str | None = None


@router.post("")
def start(body: MultiModelBody) -> dict:
    if len(body.run_ids) < 2:
        raise HTTPException(status_code=400, detail="多模型综合分析至少需要两个模型运行（run_ids）")
    if body.fusion and body.fusion not in multi_model_service.FUSION_MODES:
        raise HTTPException(status_code=400, detail=f"未知融合方式：{body.fusion}")
    task_id = multi_model_service.start(
        body.run_ids, labels=body.labels, fusion=body.fusion,
        task_type=body.task_type, dataset_id=body.dataset_id)
    return {"task_id": task_id, "status": "queued"}


@router.get("/{analysis_id}")
def get_report(analysis_id: str) -> dict:
    report = multi_model_service.get_report(analysis_id)
    if report is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return report
