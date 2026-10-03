"""画布网络 API（阶段4 4c，模块详细设计 7.5）。

- GET  /api/networks/{id}/export        画布再生成代码（与训练共用后端引擎，导出即所训）
- GET  /api/networks/{id}/run-options   运行面板初始化数据（父项目/可用环境/可训练数据集）
- GET  /api/networks/{id}/runs          训练运行记录（成功记录，指标面板数据源）
- POST /api/networks/{id}/run           发起训练（数据集+目标环境+超参 → 后台任务）

保存收口继续走 PUT /api/projects/{id}/graph（network_id=project_id，见设计文档 7.4 偏差登记）。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import network_service

router = APIRouter(prefix="/api/networks", tags=["networks"])


class RunBody(BaseModel):
    dataset_id: str
    environment_project_id: str | None = None  # 缺省：父原始项目环境
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001


def _project_error(e: Exception) -> HTTPException:
    """项目查不到 404 / 类型不对 400，统一映射（7.7-1：网络入口权限口径与 2.2 一致 = 400）。"""
    if isinstance(e, PermissionError):
        return HTTPException(status_code=400, detail=str(e))
    return HTTPException(status_code=404, detail=str(e))


@router.get("/{project_id}/export")
def export(project_id: str) -> dict:
    """再生成代码（前端画布导出按钮；训练同源）。"""
    try:
        code = network_service.export_network(project_id)
    except (LookupError, PermissionError) as e:
        raise _project_error(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"code": code}


@router.get("/{project_id}/run-options")
def run_options(project_id: str) -> dict:
    try:
        return network_service.run_options(project_id)
    except (LookupError, PermissionError) as e:
        raise _project_error(e)


@router.get("/{project_id}/runs")
def list_runs(project_id: str) -> list[dict]:
    try:
        return network_service.list_runs(project_id)
    except (LookupError, PermissionError) as e:
        raise _project_error(e)


@router.post("/{project_id}/run")
def run(project_id: str, body: RunBody) -> dict:
    """发起训练。校验失败（数据集/环境/超参）→ 400 并说明引导路径，不静默。"""
    try:
        task_id = network_service.start_run(project_id, body.model_dump())
    except (LookupError, PermissionError) as e:
        raise _project_error(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"task_id": task_id, "status": "queued"}
