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


_SAFE_PERMISSION_MODES = {"dontAsk", "acceptEdits", "default"}
_SAFE_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep", "Bash"}


@router.post("/tasks")
def submit_agent_task(body: AgentTaskCreate) -> dict:
    """提交 agent 任务。工具白名单/权限模式**只允许收敛、不允许放宽**（架构八.1 deny-first）：
    越权请求一律 400，避免经此端点绕过工具与权限约束。
    """
    if body.permission_mode not in _SAFE_PERMISSION_MODES:
        raise HTTPException(status_code=400, detail=f"permission_mode 仅允许 {sorted(_SAFE_PERMISSION_MODES)}")
    extra = [t for t in (body.allowed_tools or []) if t not in _SAFE_TOOLS]
    if extra:
        raise HTTPException(status_code=400, detail=f"allowed_tools 含未允许项: {extra}（仅允许 {sorted(_SAFE_TOOLS)}）")
    if body.disallowed_tools:
        raise HTTPException(status_code=400, detail="disallowed_tools 由服务端固定，不接受调用方指定")
    task_id = agent_service.submit(
        body.prompt,
        cwd=body.cwd,
        add_dirs=body.add_dirs,
        allowed_tools=body.allowed_tools,
        disallowed_tools=None,
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
