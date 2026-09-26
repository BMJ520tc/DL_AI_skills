"""agent 任务 API（模块详细设计 2.5）。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import agent_service

router = APIRouter(prefix="/api/agents", tags=["agents"])


class AgentTaskCreate(BaseModel):
    prompt: str
    cwd: str | None = None
    add_dirs: list[str] | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    output_schema: dict | None = None
    max_turns: int = 30
    permission_mode: str = "dontAsk"
    attach_knowledge: bool = True
    timeout_s: int = 900


@router.post("/tasks")
def submit_agent_task(body: AgentTaskCreate) -> dict:
    task_id = agent_service.submit(
        body.prompt,
        cwd=body.cwd,
        add_dirs=body.add_dirs,
        allowed_tools=body.allowed_tools,
        disallowed_tools=body.disallowed_tools,
        output_schema=body.output_schema,
        max_turns=body.max_turns,
        permission_mode=body.permission_mode,
        attach_knowledge=body.attach_knowledge,
        timeout_s=body.timeout_s,
    )
    return {"task_id": task_id, "status": "queued"}


@router.get("/tasks/{task_id}")
def get_agent_task(task_id: str) -> dict:
    result = agent_service.get_result(task_id)
    if result.get("task") is None:
        raise HTTPException(status_code=404, detail="task not found")
    return result
