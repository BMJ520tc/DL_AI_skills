// api/assistantClient.ts — 前端 AI 助手（一期只读；《新增需求补充》补充 A）。
//
// 对接后端：POST /api/assistant/chat {message, context} → {task_id}；
// 回复经既有任务轮询（/api/tasks/{id} 的 progress.reply）取回。

import { resolveApiBaseUrl } from "./knowledgeClient";

const API_BASE: string = resolveApiBaseUrl();

export type AssistantContext = {
    page?: string;
    project_id?: string;
    paper_id?: string;
    network_id?: string;
};

export type AssistantMode = "read" | "write";

export async function startAssistantChat(
    message: string,
    context: AssistantContext,
    sessionId?: string | null,
    mode: AssistantMode = "read",
): Promise<string> {
    const res = await fetch(`${API_BASE}/api/assistant/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, context, session_id: sessionId || null, mode }),
    });
    if (!res.ok) {
        let detail: unknown = null;
        try {
            const body = await res.json();
            detail = (body as { detail?: unknown }).detail ?? body;
        } catch {
            detail = await res.text().catch(() => null);
        }
        throw new Error(typeof detail === "string" ? detail : `HTTP ${res.status}`);
    }
    const data = (await res.json()) as { task_id: string };
    return data.task_id;
}

/** SSE 事件流地址（EventSource 用；事件 kind ∈ stage/delta/tool/done/error）。 */
export function assistantStreamUrl(taskId: string): string {
    return `${API_BASE}/api/assistant/chat/${taskId}/stream`;
}
