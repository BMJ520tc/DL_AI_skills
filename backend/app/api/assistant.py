"""前端 AI 助手 API（《新增需求补充》补充 A + ③续平台动作确认）。

命名空间 `/api/assistant`。对话异步执行、**流式**取事件：
  POST /api/assistant/chat              → {task_id}（message + context + session_id + mode）
  GET  /api/assistant/chat/{id}/stream  → SSE 事件流（kind ∈ stage/delta/tool/confirm/done/error）

③续：动作工具的**确认握手**（MCP 子进程 ↔ 后端 ↔ 前端弹窗）：
  POST /api/assistant/confirm                 → {confirm_id}（登记并推 SSE 给该会话前端）
  GET  /api/assistant/confirm/{id}            → {status: pending|approved|denied|expired}
  POST /api/assistant/confirm/{id}/decide     → {status}（前端用户点确认/取消）
"""
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services import assistant_service

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class ChatBody(BaseModel):
    message: str
    context: dict | None = None
    session_id: str | None = None
    mode: str = "read"   # read（只读）/ write（满工具 + 平台动作）


class ConfirmBody(BaseModel):
    session_key: str
    action: str
    params: dict | None = None


class DecideBody(BaseModel):
    approved: bool


@router.post("/chat")
def chat(body: ChatBody) -> dict:
    if not (body.message or "").strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    task_id = assistant_service.start(body.message.strip(), body.context, body.session_id, body.mode)
    return {"task_id": task_id, "status": "queued"}


@router.get("/chat/{task_id}/stream")
async def stream(task_id: str) -> StreamingResponse:
    async def gen():
        async for ev in assistant_service.stream(task_id):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/confirm")
async def create_confirm(body: ConfirmBody) -> dict:
    """MCP 动作工具发起的确认请求：登记并推 SSE 给该会话前端。"""
    if not (body.action or "").strip():
        raise HTTPException(status_code=400, detail="动作名不能为空")
    try:
        confirm_id = assistant_service.request_confirmation(
            body.session_key, body.action.strip(), body.params or {})
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"confirm_id": confirm_id, "status": "pending"}


@router.get("/confirm/{confirm_id}")
async def get_confirm(confirm_id: str) -> dict:
    """MCP 动作工具轮询决定（pending/approved/denied/expired）。"""
    rec = assistant_service.confirmation_status(confirm_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="确认请求不存在或已过期")
    return rec


@router.post("/confirm/{confirm_id}/decide")
async def decide_confirm(confirm_id: str, body: DecideBody) -> dict:
    """前端用户点「确认 / 取消」。"""
    try:
        return assistant_service.decide_confirmation(confirm_id, body.approved)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
