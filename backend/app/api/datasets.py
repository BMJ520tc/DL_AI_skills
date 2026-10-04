"""公开数据检索 API（模块详细设计 5.3）。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import dataset_service

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DownloadBody(BaseModel):
    source: str
    source_id: str
    name: str
    task_type: str | None = None
    # 数据类型（csv/excel/archive/image_dir/...）。不传时按下载到的文件扩展名推断；
    # 推不出来就留空并在响应的 format_note 里说明，不编造。
    format: str | None = None


@router.get("/search")
def search_datasets(
    task_type: str | None = None,
    format: str | None = None,
    q: str | None = None,
    limit: int = 20,
    project_id: str | None = None,
) -> dict:
    """检索公开数据：task_type / format 对已登记数据集按 `dataset_registry` 对应列过滤；
    外部源（Zenodo）过滤能力的边界写进每条外部结果的 filter_note。
    传 project_id 时附带自带数据是否充足的判定（5.3 步骤 1）。"""
    try:
        return dataset_service.search_datasets(task_type, format, q, limit, project_id)
    except (LookupError, TypeError) as e:
        raise HTTPException(status_code=404, detail=f"project not found: {e}")


@router.post("/download")
def download_dataset(body: DownloadBody) -> dict:
    """下载公开数据集并登记；响应带 dataset_id 与**如实的 format 来源说明**（推不出时留空）。"""
    try:
        info = dataset_service.download_dataset_with_info(
            body.source, body.source_id, body.name, task_type=body.task_type, format=body.format
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"下载失败: {e}")
    return info


@router.get("/{dataset_id}/alignment")
def get_alignment(dataset_id: str) -> dict:
    alignment = dataset_service.get_alignment(dataset_id)
    if alignment is None:
        raise HTTPException(status_code=404, detail="dataset or alignment not found")
    return alignment


@router.post("/{dataset_id}/alignment/confirm")
def confirm_alignment(dataset_id: str) -> dict:
    """确认对齐规则（draft → confirmed）；未确认的对齐不参与结果对比。"""
    if not dataset_service.confirm_alignment(dataset_id):
        raise HTTPException(status_code=404, detail="dataset or alignment not found")
    return {"dataset_id": dataset_id, "status": "confirmed"}
