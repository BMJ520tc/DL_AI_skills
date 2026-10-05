"""前端 AI 助手 API（《新增需求补充》补充 A）。

命名空间 `/api/assistant`。对话异步执行、**流式**取事件：
  POST /api/assistant/chat              → {task_id}（message + context + session_id + mode）
  GET  /api/assistant/chat/{id}/stream  → SSE 事件流（kind ∈ stage/delta/tool/done/error）
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
    mode: str = "read"   # read（只读）/ write（满工具）


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
