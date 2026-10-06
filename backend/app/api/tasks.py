"""任务管理 API（模块详细设计 2.1）。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import task_manager

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class TaskCreate(BaseModel):
    task_type: str
    project_id: str | None = None
    params: dict | None = None


@router.post("")
def create_task(body: TaskCreate) -> dict:
    task_id = task_manager.create_task(body.task_type, body.project_id, body.params)
    return {"task_id": task_id, "status": "queued"}


@router.get("")
def list_tasks(limit: int = 100, offset: int = 0, order: str = "recent") -> list[dict]:
    """任务列表。`order=board` 按看板口径排序（执行中 → 排队中 → 时间倒序）。"""
    return task_manager.list_tasks(limit, offset, order)


@router.get("/{task_id}")
def get_task(task_id: str) -> dict:
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@router.post("/{task_id}/cancel")
def cancel_task(task_id: str) -> dict:
    """仅 queued 任务可取消；任务不存在 → 404，状态不符 → 409。"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    if not task_manager.cancel_task(task_id):
        raise HTTPException(status_code=409, detail=f"only queued task can be cancelled (status={task['status']})")
    return {"status": "cancelled"}


@router.post("/{task_id}/retry")
def retry_task(task_id: str) -> dict:
    """仅 failed 任务可重试；任务不存在 → 404，状态不符 → 409。"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    if not task_manager.retry_task(task_id):
        raise HTTPException(status_code=409, detail=f"only failed task can be retried (status={task['status']})")
    return {"status": "queued"}
