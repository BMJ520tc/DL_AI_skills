/**
 * 知识库检索 API 客户端（只读）。
 *
 * 对接后端 backend/app/api/knowledge.py 的三个 GET 接口：
 *   - GET /api/knowledge/search?types=<逗号分隔>&q=<关键词>&limit=&offset=
 *   - GET /api/knowledge/items/{data_type}/{ref_id}
 *   - GET /api/knowledge/list?data_type=<类型>&limit=
 *
 * 本模块只做「拼 URL + fetch + 解析」，不含任何 UI 依赖，
 * 因此既能在浏览器里被组件调用，也能被 Node 脚本直接 import 做联调校验。
 */

/** 后端默认地址；可用 Vite 环境变量 VITE_API_BASE_URL 覆盖。 */
export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

/** 后端 /api/knowledge/search 支持的 types 取值。 */
export const KNOWLEDGE_DATA_TYPES = ["knowledge", "dataset", "paper", "run", "module"] as const;

export type KnowledgeDataType = (typeof KNOWLEDGE_DATA_TYPES)[number];

export const DATA_TYPE_LABELS: Record<KnowledgeDataType, string> = {
  knowledge: "知识",
  dataset: "数据集",
  paper: "论文",
  run: "运行记录",
  module: "模块",
};

export const DATA_TYPE_COLORS: Record<KnowledgeDataType, string> = {
  knowledge: "#2563eb",
  dataset: "#0d9488",
  paper: "#7c3aed",
  run: "#ca8a04",
  module: "#db2777",
};

/** /api/knowledge/search 返回的索引条目（后端 knowledge_service.search 的统一视图）。 */
export type KnowledgeSearchHit = {
  id?: string;
  data_type?: string;
  ref_id?: string;
  title?: string | null;
  summary?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  tags?: string | null;
  keywords?: string | null;
  task_type?: string | null;
  model_name?: string | null;
  dataset_name?: string | null;
  source_project_id?: string | null;
  embedding?: string | null;
};

export type RequestFailure = {
  /** 面向用户的一句话错误描述。 */
  message: string;
  /** 排查建议。 */
  hint: string;
  /** 实际请求的 URL（便于定位地址配错）。 */
  url: string;
  /** HTTP 状态码；网络层失败时为 undefined。 */
  status?: number;
};

export type RequestOutcome<T> = { ok: true; data: T } | { ok: false; failure: RequestFailure };

/**
 * 解析后端 base URL（环境变量优先 VITE_API_BASE_URL，其次 VITE_API_BASE）：
 *   - 两者均未设置            -> http://127.0.0.1:8000（默认，直连后端）
 *   - 设为 "/" 或 ""          -> ""（同源相对路径 /api/...，走 Vite dev/preview 代理）
 *   - 设为绝对地址            -> 该地址（去掉结尾斜杠）
 * 以 "/" 开头的相对路径一律按同源处理，用于规避后端未开启 CORS 的场景。
 * 注意：此函数只在浏览器（Vite）环境调用。
 */
export function resolveApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL ?? import.meta.env.VITE_API_BASE;
  if (typeof configured === "string") {
    const trimmed = configured.trim();
    // "" 或 "/xxx" => 同源（保留非根路径前缀）
    return trimmed.replace(/\/+$/, "");
  }
  return DEFAULT_API_BASE_URL;
}

/** 供界面展示的可读后端地址。 */
export function describeApiBaseUrl(baseUrl: string): string {
  return baseUrl === "" ? "同源 /api（Vite 代理）" : baseUrl;
}

export type SearchParams = {
  q?: string;
  types?: readonly string[];
  /** 需求六.1 三维度：任务类型 / 模型 / 数据集（对应后端 task_type / model / dataset）。 */
  task_type?: string;
  model?: string;
  dataset?: string;
  limit?: number;
  offset?: number;
};

/** 构造 /api/knowledge/search 的完整 URL（URLSearchParams 负责编码，中文/空格安全）。 */
export function buildSearchUrl(baseUrl: string, params: SearchParams = {}): string {
  const base = baseUrl.replace(/\/+$/, "");
  const search = new URLSearchParams();
  if (params.types && params.types.length > 0) search.set("types", params.types.join(","));
  if (params.q) search.set("q", params.q);
  if (params.task_type) search.set("task_type", params.task_type);
  if (params.model) search.set("model", params.model);
  if (params.dataset) search.set("dataset", params.dataset);
  search.set("limit", String(params.limit ?? 20));
  search.set("offset", String(params.offset ?? 0));
  return `${base}/api/knowledge/search?${search.toString()}`;
}

/** 构造 /api/knowledge/items/{data_type}/{ref_id} 的完整 URL。 */
export function buildItemUrl(baseUrl: string, dataType: string, refId: string): string {
  const base = baseUrl.replace(/\/+$/, "");
  return `${base}/api/knowledge/items/${encodeURIComponent(dataType)}/${encodeURIComponent(refId)}`;
}

function offlineHint(baseUrl: string): string {
  const target = baseUrl === "" ? "同源 /api（Vite 开发代理）" : baseUrl;
  return `无法连接后端服务 ${target}。请确认后端已启动（在 backend 目录执行 D:\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000）；若后端在其他地址，请设置环境变量 VITE_API_BASE_URL（走代理时同时设置 VITE_DEV_API_TARGET）。`;
}

async function readJson(url: string, baseUrl: string, signal?: AbortSignal): Promise<RequestOutcome<unknown>> {
  let response: Response;
  try {
    response = await fetch(url, { signal, headers: { Accept: "application/json" } });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    return {
      ok: false,
      failure: { message: "请求后端失败：网络不可达、连接被拒绝或跨域(CORS)被浏览器拦截。", hint: offlineHint(baseUrl), url },
    };
  }

  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    let detail = "";
    if (payload && typeof payload === "object" && "detail" in payload) {
      const raw = (payload as { detail?: unknown }).detail;
      if (typeof raw === "string") detail = `：${raw}`;
    }
    return {
      ok: false,
      failure: {
        message: `接口返回 HTTP ${response.status}${detail}`,
        hint:
          response.status === 404
            ? "该条目在后端不存在，可能已被删除；请重新检索。"
            : `请检查后端日志与接口路径 ${url}。`,
        url,
        status: response.status,
      },
    };
  }

  if (payload === null) {
    return {
      ok: false,
      failure: { message: "后端返回的内容不是合法 JSON。", hint: `请检查接口 ${url} 的响应内容。`, url },
    };
  }

  return { ok: true, data: payload };
}

/** 全文检索知识库。q 为空时后端返回该类型的最近条目（相当于浏览）。 */
export async function searchKnowledge(
  baseUrl: string,
  params: SearchParams = {},
  signal?: AbortSignal,
): Promise<RequestOutcome<KnowledgeSearchHit[]>> {
  const url = buildSearchUrl(baseUrl, params);
  const outcome = await readJson(url, baseUrl, signal);
  if (!outcome.ok) return outcome;
  if (!Array.isArray(outcome.data)) {
    return {
      ok: false,
      failure: { message: "检索接口返回的不是结果数组。", hint: `请检查接口 ${url} 的返回结构。`, url },
    };
  }
  return { ok: true, data: outcome.data as KnowledgeSearchHit[] };
}

/** 取单条详情。data_type 取 knowledge/dataset/paper/run/module/experiment_item 等。 */
export async function fetchKnowledgeItem(
  baseUrl: string,
  dataType: string,
  refId: string,
  signal?: AbortSignal,
): Promise<RequestOutcome<Record<string, unknown>>> {
  const url = buildItemUrl(baseUrl, dataType, refId);
  const outcome = await readJson(url, baseUrl, signal);
  if (!outcome.ok) return outcome;
  const data = outcome.data;
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return {
      ok: false,
      failure: { message: "详情接口返回的不是对象。", hint: `请检查接口 ${url} 的返回结构。`, url },
    };
  }
  return { ok: true, data: data as Record<string, unknown> };
}

function tryParseJsonString(value: string): unknown | undefined {
  const trimmed = value.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return undefined;
  try {
    return JSON.parse(trimmed);
  } catch {
    return undefined;
  }
}

/**
 * 后端把 structured / scope / tags / section_index 等字段以「JSON 字符串」形式返回，
 * 这里递归展开，便于结构化展示。
 */
export function normalizeDetailPayload(value: unknown, depth = 0): unknown {
  if (depth > 6) return value;
  if (typeof value === "string") {
    const parsed = tryParseJsonString(value);
    return parsed === undefined ? value : normalizeDetailPayload(parsed, depth + 1);
  }
  if (Array.isArray(value)) return value.map(item => normalizeDetailPayload(item, depth + 1));
  if (value && typeof value === "object") {
    const output: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      output[key] = normalizeDetailPayload(item, depth + 1);
    }
    return output;
  }
  return value;
}

/** 取详情里可读的标题（不同 data_type 的主键/标题字段名不一致）。 */
export function pickTitle(item: Record<string, unknown> | null, fallback = ""): string {
  if (!item) return fallback;
  const candidates = ["title", "name", "dataset_name", "model_name", "paper_id", "knowledge_id", "id", "ref_id"];
  for (const key of candidates) {
    const value = item[key];
    if (typeof value === "string" && value.trim() !== "") return value;
  }
  return fallback;
}

/** 把 ISO 时间字符串格式化为本地时间；解析失败时原样返回。 */
export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())} ${pad(parsed.getHours())}:${pad(parsed.getMinutes())}`;
}

/** 单行摘要，超长截断。 */
export function truncateSummary(value: string | null | undefined, max = 160): string {
  if (!value) return "";
  const single = value.replace(/\s+/g, " ").trim();
  return single.length > max ? `${single.slice(0, max)}…` : single;
}
