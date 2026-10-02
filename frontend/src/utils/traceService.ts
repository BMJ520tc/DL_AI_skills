import type { TraceRequest, TraceResponse } from "../types/trace";
import { resolveApiBaseUrl } from "../api/knowledgeClient";

/**
 * 形状追踪走本机后端（与 api/client 同一套基址约定：VITE_API_BASE_URL/VITE_API_BASE，
 * 默认 http://127.0.0.1:8000）；基底原来指向远端演示站，本项目后端以 503 占位说明容器通道未启用。
 * On failure, surface the backend error so the user can fix shapes/code and retry.
 */
const BASE_URL = resolveApiBaseUrl();

/** 端点缺失 / 独立 runner 不可用的统一提示语（需求五.2 沙盒追踪，前端可见而非静默失败）。 */
export const TRACE_UNAVAILABLE_REASON = "形状追踪不可用（需独立 runner/容器通道）";

export async function runTorchLensTrace(body: TraceRequest): Promise<TraceResponse> {
    try {
        const res = await fetch(`${BASE_URL}/api/torchlens`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const text = await res.text();
            const hint = text.slice(0, 500).toLowerCase();
            // 5xx/503 或该端点的明确错误体（未挂载/未实现）→ 结构化「不可用」，供界面可见提示
            const looksUnavailable =
                res.status >= 500 ||
                res.status === 404 ||
                /torchlens|runner|container|not\s+(implemented|mounted|found)/.test(hint);
            if (looksUnavailable) {
                return {
                    entries: [],
                    unavailable: true,
                    unavailableReason: TRACE_UNAVAILABLE_REASON,
                    warnings: [
                        `${TRACE_UNAVAILABLE_REASON}：后端返回 HTTP ${res.status}${text ? ` - ${text.slice(0, 200)}` : ""}`,
                        "画布其余功能不受影响；补形状/参数校验仍在本地执行。",
                    ],
                };
            }
            throw new Error(text || `Trace request failed: ${res.status} ${res.statusText}`);
        }
        return (await res.json()) as TraceResponse;
    } catch (err) {
        console.warn("Trace request failed", err);
        const message =
            err instanceof Error && err.message
                ? err.message
                : "TorchLens trace failed: check backend logs and model shapes.";
        return {
            entries: [],
            warnings: [
                message,
                "Fix model params / shapes and retry TorchLens trace. See backend logs for the stack trace.",
            ],
        };
    }
}
