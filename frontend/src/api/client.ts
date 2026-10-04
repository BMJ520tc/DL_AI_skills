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

/** PUT graph 的响应（backend/app/api/projects.py:put_graph）。
 *  版本提交失败不连坐保存本身：图已落盘，失败原因在 version_error 透出——
 *  界面必须据此显示「已保存到项目，但版本节点未生成」，不能一律报成功。 */
export interface PutGraphResponse {
    status: string;
    /** version_service.commit_graph 的返回值（{commit: 短提交号}）；老后端可能只给字符串。 */
    version?: { commit?: string } | string | null;
    /** 非空即表示「图已保存但版本提交失败」，内容是失败原因。 */
    version_error?: string | null;
}

export const putGraph = (projectId: string, graph: GraphIR) =>
    request<PutGraphResponse>(`/api/projects/${projectId}/graph`, {
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

/** 最小可运行命令验证（模块一 2.3「代码能否跑起来」；异步任务，结果落 run_record）。 */
export const postProjectVerify = (projectId: string) =>
    request<{ task_id: string; status: string }>(`/api/projects/${projectId}/verify`, { method: "POST" });

// ---------------------------------------------------------------------------
// 检索与下载（模块一 需求一，模块详细设计 2.4）
// ---------------------------------------------------------------------------

export interface PaperSearchParams {
    q: string;
    source?: "arxiv" | "pubmed" | "biorxiv";
    max_results?: number;
    /** 逗号分隔的作者过滤（bioRxiv 不支持服务端过滤，结果里会带 filter_note） */
    authors?: string;
    /** YYYY[-MM[-DD]] */
    date_from?: string;
    date_to?: string;
    /** PubMed/bioRxiv 尽力抓 PMC OA 全文（取不到记 abstract_only，不报错） */
    fulltext?: boolean;
}

/** 论文检索：返回结果条目（字段随来源而异，统一以 paper_id/title/abstract/pdf_url 为主）。 */
export const searchPapers = (params: PaperSearchParams) => {
    const query = new URLSearchParams({ q: params.q });
    if (params.source) query.set("source", params.source);
    if (params.max_results !== undefined) query.set("max_results", String(params.max_results));
    if (params.authors) query.set("authors", params.authors);
    if (params.date_from) query.set("date_from", params.date_from);
    if (params.date_to) query.set("date_to", params.date_to);
    if (params.fulltext) query.set("fulltext", "true");
    return request<Array<Record<string, unknown>>>(`/api/search/papers?${query.toString()}`);
};

export interface DownloadPaperBody {
    paper_id: string;
    /** source=pubmed 时可省：后端改抓 PMC OA 全文 */
    pdf_url?: string | null;
    title?: string | null;
    abstract?: string | null;
    source?: string | null;
}

/** 单篇下载（同步接口）：落库成 paper 记录。 */
export const downloadPaper = (body: DownloadPaperBody) =>
    request<{ paper_id: string; status: string }>("/api/search/papers/download", {
        method: "POST",
        body: JSON.stringify(body),
    });

/** 批量下载（同步串行；单篇失败不整体失败，逐篇返回成功/失败原因）。 */
export const downloadPapersBatch = (papers: DownloadPaperBody[]) =>
    request<{
        total: number;
        succeeded: number;
        failed: number;
        note: string | null;
        results: Array<{ paper_id: string; status: string; reason?: string }>;
    }>("/api/search/papers/download-batch", {
        method: "POST",
        body: JSON.stringify({ papers }),
    });

export interface ExtractAddressesBody {
    /** 论文正文（必填） */
    paper_text: string;
    /** 给了 paper_id 就顺带扫 data/papers/<id>/ 下的补充材料 */
    paper_id?: string;
    /** 显式补充材料路径（PDF/压缩包/纯文本），与目录扫描合并去重 */
    supplementary_paths?: string[];
    /** 抽到仓库地址后是否立刻完整克隆（默认 true） */
    clone_repos?: boolean;
    /** 克隆口径：true=完整克隆，false=浅克隆 */
    full_clone?: boolean;
    cwd?: string;
}

/** 地址抽取任务（仓库/数据集地址 → 克隆与登记；进度里可复核每条结果）。 */
export const postExtractAddresses = (body: ExtractAddressesBody) =>
    request<{ task_id: string; status: string }>("/api/search/extract", {
        method: "POST",
        body: JSON.stringify(body),
    });

/** run_record 行（数据设计五.4；经通用知识库列表接口读出，按 project/task 过滤）。 */
export interface RunRecord {
    run_id: string;
    project_id: string | null;
    task_id: string | null;
    run_type: string;
    command: string | null;
    status: string;
    metrics: string | null;
    error: string | null;
    params: string | null;
    artifact_path: string | null;
    started_at: string | null;
    finished_at: string | null;
    duration_s: number | null;
}

/** 某项目的运行记录（既有 GET /api/knowledge/list?data_type=run，按项目过滤）。
 *  后端没有「按项目列 run_record」的专用端点，这里复用通用列表并如实按 project_id 过滤。 */
export async function listRunRecords(projectId: string, limit = 200): Promise<RunRecord[]> {
    const rows = await listKnowledge("run", limit);
    return (rows as unknown as RunRecord[]).filter(r => r.project_id === projectId);
}

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

/** 修正 IR 入口输入规格（6.1/6.2）：库型模型 agent 给不出具体维度时的补参通道；
 *  写回后 ir_hash 变化 → 旧验证变 stale（与调参同口径），入库前必须重新验证。 */
export const putIrInputSpec = (projectId: string, shape: number[], dtype?: string | null) =>
    request<{ shape: number[]; dtype?: string }>(`/api/projects/${projectId}/ir/input_spec`, {
        method: "PUT",
        body: JSON.stringify(dtype ? { shape, dtype } : { shape }),
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

/** 4.1 PDF → markdown（规则 + agent 修正）；任务进度里带保真核对 fidelity 字段。 */
export const postParsePaper = (paperId: string) =>
    request<{ task_id: string; status: string }>(`/api/papers/${paperId}/parse`, { method: "POST" });

export const postExtractItems = (paperId: string) =>
    request<{ task_id: string; status: string }>(`/api/papers/${paperId}/extract`, { method: "POST" });

/** 4.2 条目编辑字段（只传要改的键；hyperparams/baselines 传对象或数组，后端存 JSON）。 */
export interface PaperItemEdit {
    section_ref?: string | null;
    dataset_name?: string | null;
    split_method?: string | null;
    metric_name?: string | null;
    metric_value_reported?: string | null;
    metric_unit?: string | null;
    hyperparams?: Record<string, unknown> | null;
    baselines?: unknown[] | null;
    status?: string | null;
}

/** 4.2 用户编辑实验条目（确认后条目生效）；返回更新后的条目。 */
export const putPaperItem = (paperId: string, itemId: string, body: PaperItemEdit) =>
    request<Record<string, unknown>>(`/api/papers/${paperId}/items/${itemId}`, {
        method: "PUT",
        body: JSON.stringify(body),
    });

export const postReproduce = (paperId: string, projectId: string) =>
    request<{ task_id: string; status: string }>(`/api/papers/${paperId}/reproduce`, {
        method: "POST",
        body: JSON.stringify({ project_id: projectId }),
    });

export const postConclusion = (paperId: string) =>
    request<{ task_id: string; status: string }>(`/api/papers/${paperId}/conclusion`, { method: "POST" });

/** 4.4 用户确认或修改可信度结论（只传要改的键；后端在缺省时保留原值）。 */
export const putPaperConclusion = (paperId: string, body: { overall_verdict?: string; summary?: string }) =>
    request<{ conclusion_id: string; status: string }>(`/api/papers/${paperId}/conclusion`, {
        method: "PUT",
        body: JSON.stringify(body),
    });

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

// ---------------------------------------------------------------------------
// 模块三 5.1/5.3：数据预处理与公开数据检索下载
// ---------------------------------------------------------------------------

export interface PreprocessBody {
    /** 待预处理的数据文件/目录路径（必填） */
    input_path: string;
    /** 数据集名（带 project_id 时默认 "self"，产物落 <workspace>/data/<name>） */
    dataset_name?: string | null;
    task_type?: string;
    /** 传原始项目 id 时按项目自带数据处理（产物落项目工作区，便于自动定位） */
    project_id?: string | null;
    /** 来源溯源（公开数据经预处理后仍保留来源） */
    source?: string | null;
    url?: string | null;
}

/** 5.1 数据预处理（异步任务；进度经任务轮询，结果含 dataset_id/统一 schema 摘要）。 */
export const postPreprocess = (body: PreprocessBody) =>
    request<{ task_id: string; status: string }>("/api/preprocess", {
        method: "POST",
        body: JSON.stringify(body),
    });

export interface DatasetSearchParams {
    task_type?: string;
    format?: string;
    q?: string;
    limit?: number;
    /** 传 project_id 附带「自带数据是否充足」判定，并把自带数据集从 local 里排除 */
    project_id?: string;
}

export interface DatasetSearchResult {
    local: Array<Record<string, unknown>>;
    external: Array<Record<string, unknown>>;
    self_data?: {
        sufficient: boolean;
        n_samples: number;
        threshold: number;
        hint: string | null;
    };
    hint?: string;
}

/** 5.3 公开数据检索（本地 registry + Zenodo 尽力而为；不阻断流程）。 */
export const searchDatasets = (params: DatasetSearchParams) => {
    const query = new URLSearchParams();
    if (params.task_type) query.set("task_type", params.task_type);
    if (params.format) query.set("format", params.format);
    if (params.q) query.set("q", params.q);
    if (params.limit !== undefined) query.set("limit", String(params.limit));
    if (params.project_id) query.set("project_id", params.project_id);
    const suffix = query.toString();
    return request<DatasetSearchResult>(`/api/datasets/search${suffix ? `?${suffix}` : ""}`);
};

/** 5.3 下载公开数据集并登记（同步；返回新登记的 dataset_id）。 */
export const downloadDataset = (body: { source: string; source_id: string; name: string; task_type?: string | null }) =>
    request<{ dataset_id: string }>("/api/datasets/download", {
        method: "POST",
        body: JSON.stringify(body),
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

/** 原始项目的独立环境：建环境（异步任务）与状态查询（模块一 2.3；阶段4 4c-3）。 */
export const createProjectEnv = (projectId: string) =>
    request<{ task_id: string; status: string }>(`/api/projects/${projectId}/env`, { method: "POST" });

export const getProjectEnvStatus = (projectId: string) =>
    request<{ project_id: string; status: string | null }>(`/api/projects/${projectId}/env`);

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
