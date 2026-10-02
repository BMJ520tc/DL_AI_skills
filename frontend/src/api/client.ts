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

// ---------------------------------------------------------------------------
// 知识库通用列表（检索界面与三并列入口共用）
// ---------------------------------------------------------------------------

export const listKnowledge = (dataType: string, limit = 200) =>
    request<Array<Record<string, unknown>>>(`/api/knowledge/list?data_type=${dataType}&limit=${limit}`);

export const confirmKnowledgeItem = (knowledgeId: string) =>
    request<{ knowledge_id: string; status: string }>(`/api/knowledge/confirm/${knowledgeId}`, { method: "POST" });

// ---------------------------------------------------------------------------
// 模块二：论文复现（三并列入口「先复现」，4.2~4.4）
// ---------------------------------------------------------------------------

export interface PaperDetail {
    paper: Record<string, unknown>;
    items: Array<Record<string, unknown>>;
    reproduction_results: Array<Record<string, unknown>>;
    conclusion: Record<string, unknown> | null;
}

export const getPaperDetail = (paperId: string) => request<PaperDetail>(`/api/papers/${paperId}`);

export const postExtractItems = (paperId: string) =>
    request<{ task_id: string; status: string }>(`/api/papers/${paperId}/extract`, { method: "POST" });

export const postReproduce = (paperId: string, projectId: string) =>
    request<{ task_id: string; status: string }>(`/api/papers/${paperId}/reproduce`, {
        method: "POST",
        body: JSON.stringify({ project_id: projectId }),
    });

export const postConclusion = (paperId: string) =>
    request<{ task_id: string; status: string }>(`/api/papers/${paperId}/conclusion`, { method: "POST" });

export const confirmPaperItem = (paperId: string, itemId: string) =>
    request<{ item_id: string; status: string }>(`/api/papers/${paperId}/items/${itemId}/confirm`, { method: "POST" });

// ---------------------------------------------------------------------------
// 模块三：原始模型使用（三并列入口「先使用」，5.2~5.5）
// ---------------------------------------------------------------------------

export const postBaseline = (projectId: string) =>
    request<{ task_id: string; status: string }>(`/api/projects/${projectId}/baseline`, { method: "POST" });

export const getBaseline = (projectId: string) => request<Record<string, unknown>>(`/api/projects/${projectId}/baseline`);

export const postAlign = (projectId: string, datasetId: string) =>
    request<{ task_id: string; status: string }>(`/api/projects/${projectId}/datasets/align`, {
        method: "POST",
        body: JSON.stringify({ dataset_id: datasetId }),
    });

export const confirmAlignment = (datasetId: string) =>
    request<{ status: string }>(`/api/datasets/${datasetId}/alignment/confirm`, { method: "POST" });

export const postCompare = (projectId: string) =>
    request<{ task_id: string; status: string }>(`/api/projects/${projectId}/compare`, { method: "POST" });

export const postVisualize = (projectId: string, chart: string) =>
    request<{ chart_type: string; html: string; degraded?: boolean }>(`/api/projects/${projectId}/visualize/${chart}`, {
        method: "POST",
    });

/** 图表文件在新标签页打开的地址（经后端图表文件服务端点转发）。 */
export const figureUrl = (projectId: string, chart: string) => `${API_BASE}/api/projects/${projectId}/figures/${chart}`;

// ---------------------------------------------------------------------------
// 画布网络（阶段4 4c，模块详细设计 7.5）
// ---------------------------------------------------------------------------

export interface NetworkRunOptions {
    parent_project_id: string | null;
    parent_name: string | null;
    environments: Array<{ project_id: string; name: string; python: string }>;
    datasets: Array<{ dataset_id: string; name: string | null; task_type: string | null; local_path: string | null }>;
}

export interface NetworkRunRecord {
    run_id: string;
    project_id: string;
    task_id: string | null;
    run_type: string;
    environment: string | null;
    params: string | null;
    command: string | null;
    status: string;
    metrics: string | null;
    error: string | null;
    artifact_path: string | null;
    log_path: string | null;
    started_at: string;
    finished_at: string | null;
    duration_s: number | null;
    schema_version: string;
}

/** 画布再生成代码（与训练共用后端引擎，导出即所训）。 */
export const exportNetwork = (projectId: string) =>
    request<{ code: string }>(`/api/networks/${projectId}/export`);

export const getNetworkRunOptions = (projectId: string) =>
    request<NetworkRunOptions>(`/api/networks/${projectId}/run-options`);

export const listNetworkRuns = (projectId: string) =>
    request<NetworkRunRecord[]>(`/api/networks/${projectId}/runs`);

export const postNetworkRun = (
    projectId: string,
    body: {
        dataset_id: string;
        environment_project_id?: string | null;
        epochs: number;
        batch_size: number;
        learning_rate: number;
    },
) =>
    request<{ task_id: string; status: string }>(`/api/networks/${projectId}/run`, {
        method: "POST",
        body: JSON.stringify(body),
    });

// ---------------------------------------------------------------------------
// 版本管理（阶段4 4d-2，模块详细设计 7.6）
// ---------------------------------------------------------------------------

export interface VersionMeta {
    saved_at: string | null;
    node_count: number;
    edge_count: number;
    input_spec: Array<{ node_id: string; type: string }>;
    output_spec: Array<{ node_id: string; type: string }>;
    run_summary: { task_id: string; metrics: Record<string, unknown>; finished_at: string } | null;
    rollback_to: string | null;
}

export interface VersionNode {
    commit: string;
    short: string;
    message: string;
    committed_at: string;
    parents: string[];
    meta: VersionMeta | null;
}

export interface VersionTree {
    current: string | null;
    versions: VersionNode[];
}

export interface ParamChange {
    key: string;
    old: unknown;
    new: unknown;
}

export interface ParamDiff {
    nodes_added: Array<{ id: string; type: string }>;
    nodes_removed: Array<{ id: string; type: string }>;
    nodes_changed: Array<{ id: string; type: string; param_changes: ParamChange[] }>;
    edges_added: Array<{ id: string; source: string; target: string }>;
    edges_removed: Array<{ id: string; source: string; target: string }>;
}

export interface VersionCompare {
    v1: string;
    v2: string;
    code_diff: string[] | null;
    code_diff_error: string | null;
    param_diff: ParamDiff;
}

export const getVersionTree = (projectId: string) =>
    request<VersionTree>(`/api/versions/${projectId}/tree`);

export const compareVersions = (projectId: string, v1: string, v2: string) =>
    request<VersionCompare>(`/api/versions/${projectId}/compare?v1=${encodeURIComponent(v1)}&v2=${encodeURIComponent(v2)}`);

export const postRollback = (projectId: string, targetVersion: string) =>
    request<{ commit: string; target: string; graph: GraphIR }>(
        `/api/versions/${projectId}/rollback`,
        { method: "POST", body: JSON.stringify({ target_version: targetVersion }) },
    );
