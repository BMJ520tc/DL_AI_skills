// api/client.ts — 后端 REST 客户端（模块四 B1，模块详细设计 6.x）。
// 无 axios，统一用 fetch；基址解析与知识库客户端共用一套约定
// （knowledgeClient.resolveApiBaseUrl：VITE_API_BASE_URL，默认直连 :8000，
// 设为 "/" 则走 vite dev/preview 代理，见 vite.config.ts）。

import type { GraphIR } from "../types/graph";
import { resolveApiBaseUrl } from "./knowledgeClient";

const API_BASE: string = resolveApiBaseUrl();

export class ApiError extends Error {
    readonly status: number;

    constructor(status: number, message: string) {
        super(message);
        this.status = status;
    }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...init,
    });
    if (!res.ok) {
        let detail: unknown = null;
        try {
            const body = await res.json();
            detail = (body as { detail?: unknown }).detail ?? body;
        } catch {
            detail = await res.text().catch(() => null);
        }
        const message =
            typeof detail === "string"
                ? detail
                : detail
                  ? JSON.stringify(detail)
                  : `HTTP ${res.status}`;
        throw new ApiError(res.status, message);
    }
    return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// 通用类型（与 backend 表/JSON 对应）
// ---------------------------------------------------------------------------

export interface Project {
    project_id: string;
    project_type: string;
    source: string | null;
    name: string | null;
    parent_project_id: string | null;
    status: string;
    workspace_path: string | null;
    created_at: string;
    updated_at: string;
}

export interface Task {
    task_id: string;
    task_type: string;
    project_id: string | null;
    params: string;
    status: "queued" | "running" | "success" | "failed" | "cancelled";
    progress: string | null;
    error: string | null;
    created_at: string;
    updated_at: string;
}

// IR（backend/app/services/ir_schema.py 对应）
export type IrKind = "module" | "leaf" | "container" | "op";

export interface IrNode {
    id: string;
    kind: IrKind;
    class_name?: string | null;
    module_file?: string | null;
    module_path?: string | null;
    params?: Record<string, unknown> | null;
    parent_id?: string | null;
    input_shape?: number[] | null;
    output_shape?: number[] | null;
    code_hint?: string | null;
    uncertain?: boolean | null;
}

export interface IrEdge {
    from: string;
    to: string;
    tensor_shape?: number[] | null;
}

export interface IrGraph {
    schema_version: string;
    project_id?: string;
    source_file?: string | null;
    entry_class?: string | null;
    task_type?: string | null;
    input_spec?: { shape: number[]; dtype?: string } | null;
    root_id?: string | null;
    nodes: IrNode[];
    edges: IrEdge[];
}

// 验证记录（reports/verification.json）
export interface VerificationStructure {
    passed: boolean;
    /** 带参数层数（无参层不参与判定，见实施约定 6.6-3 的结构比对口径） */
    layer_count: { original: number; regenerated: number };
    param_count: { original: number; regenerated: number };
    /** 模块总数（含无参层），参考信息，不参与判定 */
    module_count?: { original: number; regenerated: number; note?: string };
    layer_sequence_match: boolean;
    param_shapes_match: boolean;
    output_shape_match: boolean;
    diff_layers: Array<Record<string, unknown>>;
}

export interface VerificationNumeric {
    passed: boolean;
    /** 比对前已把原模型权重拷入再生成模型（实施约定 6.6-3 权重同源） */
    weight_source?: string;
    per_seed: Array<{
        seed: number;
        max_rel_err: number;
        max_abs_err: number;
        passed: boolean;
        diff_layers?: Array<Record<string, unknown>>;
    }>;
}

export interface Verification {
    schema_version?: string;
    project_id?: string;
    verified_at: string;
    ir_hash: string;
    seeds: number[];
    tolerance: { rtol: number; atol: number };
    structure: VerificationStructure;
    numeric: VerificationNumeric;
    overall: "passed" | "failed";
    failure_reason?: string | null;
}

export type VerificationStatus = "none" | "valid" | "stale";

export interface IrResponse {
    ir: IrGraph;
    verification_status: VerificationStatus;
    verification: Verification | null;
}

// 模块（module 表）
export interface ModuleItem {
    module_id: string;
    module_version: string;
    name: string | null;
    description: string | null;
    source_project_id: string | null;
    source_paper_id: string | null;
    task_type: string | null;
    input_spec: string | null;
    output_spec: string | null;
    params_schema: string | null;
    tags: string | null;
    verification: string | null;
    saved_module_compat: string | null;
    path: string | null;
    created_at: string;
    updated_at: string;
    schema_version: string;
}

// ---------------------------------------------------------------------------
// 项目（模块详细设计 2.2）
// ---------------------------------------------------------------------------

export const listProjects = (projectType?: string) =>
    request<Project[]>(`/api/projects${projectType ? `?project_type=${projectType}` : ""}`);

export const getProject = (projectId: string) => request<Project>(`/api/projects/${projectId}`);

export const createProject = (body: {
    project_type: string;
    source?: string;
    name?: string;
    parent_project_id?: string;
    source_url?: string;
}) => request<{ project_id: string; status: string }>("/api/projects", {
    method: "POST",
    body: JSON.stringify(body),
});

// 结构化项目画布快照（模块四 6.5/7.1）
export const getGraph = (projectId: string) => request<GraphIR>(`/api/projects/${projectId}/graph`);

export const putGraph = (projectId: string, graph: GraphIR) =>
    request<{ status: string }>(`/api/projects/${projectId}/graph`, {
        method: "PUT",
        body: JSON.stringify(graph),
    });

// ---------------------------------------------------------------------------
// 任务（2.1）
// ---------------------------------------------------------------------------

export const getTask = (taskId: string) => request<Task>(`/api/tasks/${taskId}`);

/** 轮询任务直至终态（success/failed/cancelled）。 */
export async function pollTask(taskId: string, intervalMs = 2000): Promise<Task> {
    for (;;) {
        const task = await getTask(taskId);
        if (["success", "failed", "cancelled"].includes(task.status)) return task;
        await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
}

// ---------------------------------------------------------------------------
// 模块一（结构分析，模块四拆解的前置）
// ---------------------------------------------------------------------------

export const postAnalyze = (projectId: string) =>
    request<{ task_id: string; status: string }>(`/api/projects/${projectId}/analyze`, { method: "POST" });

export const getReport = (projectId: string) =>
    request<Record<string, unknown>>(`/api/projects/${projectId}/report`);

// ---------------------------------------------------------------------------
// 模块四（6.1/6.2/6.3/6.4）
// ---------------------------------------------------------------------------

export const postDecompose = (projectId: string, entryClass?: string) =>
    request<{ task_id: string; status: string }>(`/api/projects/${projectId}/decompose`, {
        method: "POST",
        body: JSON.stringify(entryClass ? { entry_class: entryClass } : {}),
    });

export const postTrace = (projectId: string) =>
    request<{ task_id: string; status: string }>(`/api/projects/${projectId}/decompose/trace`, { method: "POST" });

export const postRegenerate = (projectId: string) =>
    request<{ code: string }>(`/api/projects/${projectId}/decompose/regenerate`, { method: "POST" });

export const postVerifyDecompose = (projectId: string) =>
    request<{ task_id: string; status: string }>(`/api/projects/${projectId}/decompose/verify`, { method: "POST" });

export const getIr = (projectId: string) => request<IrResponse>(`/api/projects/${projectId}/ir`);

export const putNodeParams = (projectId: string, nodeId: string, params: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/api/projects/${projectId}/ir/nodes/${nodeId}`, {
        method: "PUT",
        body: JSON.stringify({ params }),
    });

// ---------------------------------------------------------------------------
// 模块库（6.5）
// ---------------------------------------------------------------------------

export const postIngestModule = (projectId: string) =>
    request<{ task_id: string; status: string; module_id?: string; module_version?: string; existing_versions?: string[] }>("/api/modules", {
        method: "POST",
        body: JSON.stringify({ project_id: projectId }),
    });

export const listModules = () => request<ModuleItem[]>("/api/modules");
