"""画布网络 API（阶段4 4c，模块详细设计 7.5）。

- GET  /api/networks/{id}/export        画布再生成代码（与训练共用后端引擎，导出即所训）
- GET  /api/networks/{id}/run-options   运行面板初始化数据（父项目/可用环境/可训练数据集）
- GET  /api/networks/{id}/runs          训练运行记录（成功记录，指标面板数据源）
- POST /api/networks/{id}/run           发起训练（数据集+目标环境+超参 → 后台任务）

保存收口继续走 PUT /api/projects/{id}/graph（network_id=project_id，见设计文档 7.4 偏差登记）。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import arch_service, network_service

router = APIRouter(prefix="/api/networks", tags=["networks"])


class RunBody(BaseModel):
    dataset_id: str
    environment_project_id: str | None = None  # 缺省：父原始项目环境
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001


class AutotuneBody(BaseModel):
    """自动调参（B，扩范围）：基础超参 + 可选显式候选；会给「带入的知识建议」留位。"""
    dataset_id: str
    environment_project_id: str | None = None
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 0.01
    candidates: list[dict] | None = None


class ArchSuggestBody(BaseModel):
    """架构级自迭代建议（需求六.1 延伸）：可选带入条件（任务类型/模型/数据集）与补充说明。"""
    task_type: str | None = None
    model: str | None = None
    dataset: str | None = None
    hint: str | None = None


class ArchApplyBody(BaseModel):
    """把某条建议作用到画布（不写盘，返回新图；前端放回画布后由用户保存 = 新版本）。"""
    task_id: str
    suggestion_id: str


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


@router.post("/{project_id}/autotune")
def autotune(project_id: str, body: AutotuneBody) -> dict:
    """自动调参（B，扩范围）：带入知识 → 候选超参逐个训练 → 按主指标选优 → 蒸馏回写。"""
    try:
        task_id = network_service.start_autotune(project_id, body.model_dump())
    except (LookupError, PermissionError) as e:
        raise _project_error(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"task_id": task_id, "status": "queued"}


@router.post("/{project_id}/arch-suggest")
def arch_suggest(project_id: str, body: ArchSuggestBody) -> dict:
    """发起架构级自迭代建议任务（agent 起草；ir 图 400，与导出/训练同一口径）。"""
    try:
        result = arch_service.start_suggest(project_id, body.model_dump())
    except (LookupError, PermissionError) as e:
        raise _project_error(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"task_id": result["task_id"], "status": "queued", "graph_hash": result["graph_hash"]}


@router.get("/{project_id}/arch-suggestions")
def arch_suggestions(project_id: str) -> dict:
    """取该项目最近一次架构建议报告（无报告 404；报告内 0 条建议仍是 200，如实说明）。"""
    try:
        arch_service._require_network(project_id)
    except (LookupError, PermissionError) as e:
        raise _project_error(e)
    report = arch_service.get_latest_report(project_id)
    if report is None:
        raise HTTPException(status_code=404, detail="尚未生成架构建议")
    return report


@router.get("/{project_id}/arch-suggestions/{task_id}")
def arch_suggestion_report(project_id: str, task_id: str) -> dict:
    try:
        arch_service._require_network(project_id)
    except (LookupError, PermissionError) as e:
        raise _project_error(e)
    report = arch_service.get_report(task_id)
    if report is None or report.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="架构建议报告不存在")
    return report


@router.post("/{project_id}/arch-apply")
def arch_apply(project_id: str, body: ArchApplyBody) -> dict:
    """把某条建议作用到当前画布图的副本上并返回（不写盘）；画布已变 409、非法建议 400。"""
    try:
        return arch_service.apply_for_report(project_id, body.task_id, body.suggestion_id)
    except arch_service.StaleGraphError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except (LookupError, PermissionError) as e:
        raise _project_error(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
