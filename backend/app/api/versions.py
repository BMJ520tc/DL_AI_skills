"""版本管理 API（阶段4 4d-2，模块详细设计 7.6）。

- GET  /api/versions/{id}/tree      版本树：git 提交历史 + 元数据摘要 + 父子关系
- GET  /api/versions/{id}/compare   两版本对比：代码差异（再生成代码 unified diff）+
                                    参数差异表（节点参数逐项比对、增删节点/连线）
- POST /api/versions/{id}/rollback  回退到目标版本（回退动作本身记为新提交）

network_id 即结构化项目 project_id；守卫与 networks 路由同口径（400/404）。
git 内部失败（RuntimeError）默认 500 并透出 detail——不静默。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import version_service

router = APIRouter(prefix="/api/versions", tags=["versions"])


class RollbackBody(BaseModel):
    target_version: str


def _error(e: Exception) -> HTTPException:
    """项目查不到/版本不存在 404、类型不对 400、参数语义问题 400，统一映射
    （7.7-1：网络入口权限口径与 2.2 一致 = 400）。"""
    if isinstance(e, PermissionError):
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, LookupError):
        return HTTPException(status_code=404, detail=str(e))
    return HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/tree")
def tree(project_id: str) -> dict:
    try:
        return version_service.version_tree(project_id)
    except (LookupError, PermissionError) as e:
        raise _error(e)


@router.get("/{project_id}/compare")
def compare(project_id: str, v1: str, v2: str) -> dict:
    try:
        return version_service.compare_versions(project_id, v1, v2)
    except (LookupError, PermissionError, ValueError) as e:
        raise _error(e)


@router.post("/{project_id}/rollback")
def rollback(project_id: str, body: RollbackBody) -> dict:
    try:
        return version_service.rollback(project_id, body.target_version)
    except (LookupError, PermissionError, ValueError) as e:
        raise _error(e)
