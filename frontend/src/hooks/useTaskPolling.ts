// hooks/useTaskPolling.ts — 任务轮询（模块四阶段3实施方案 3.6）。
// 默认 1s 间隔 GET /api/tasks/{id}，success/failed（含 cancelled）即停；
// 卸载或 taskId 变化时清理，避免旧任务回调污染当前视图。

import { useEffect, useRef } from "react";
import { getTask, type Task } from "../api/client";

export type UseTaskPollingOptions = {
    /** 轮询的任务 id；为 null 时不轮询。 */
    taskId: string | null;
    /** 任务成功（success）时回调。 */
    onDone: (task: Task) => void;
    /** 任务失败/取消或查询出错时回调。 */
    onError: (message: string, task: Task | null) => void;
    /** 轮询间隔，默认 1000ms。 */
    intervalMs?: number;
    /** 每次拿到任务状态时回调（可选，用于进度展示）。 */
    onProgress?: (task: Task) => void;
};

export function useTaskPolling({
    taskId,
    onDone,
    onError,
    onProgress,
    intervalMs = 1000,
}: UseTaskPollingOptions) {
    // 回调经 ref 转发，轮询 effect 只依赖 taskId/intervalMs，避免回调身份变化重启计时器。
    const handlers = useRef({ onDone, onError, onProgress });
    useEffect(() => {
        handlers.current = { onDone, onError, onProgress };
    });

    useEffect(() => {
        if (!taskId) return;
        let cancelled = false;
        let timer: ReturnType<typeof setInterval> | null = null;
        const stop = () => {
            if (timer !== null) {
                clearInterval(timer);
                timer = null;
            }
        };
        timer = setInterval(() => {
            void (async () => {
                let task: Task;
                try {
                    task = await getTask(taskId);
                } catch (e) {
                    if (cancelled) return;
                    stop();
                    handlers.current.onError(e instanceof Error ? e.message : String(e), null);
                    return;
                }
                if (cancelled) return;
                handlers.current.onProgress?.(task);
                if (task.status === "success") {
                    stop();
                    handlers.current.onDone(task);
                } else if (task.status === "failed" || task.status === "cancelled") {
                    stop();
                    handlers.current.onError(task.error || `任务失败（${task.status}）`, task);
                }
            })();
        }, intervalMs);
        return () => {
            cancelled = true;
            stop();
        };
    }, [taskId, intervalMs]);
}
