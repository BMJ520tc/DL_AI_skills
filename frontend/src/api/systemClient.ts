/**
 * 系统设置与自检 API 客户端（一键封装 6.6-a）。
 *
 * 对接后端：
 *   - GET  /api/system/env-check         运行环境自检（git/python/conda/数据目录/静态服务/凭证/镜像）
 *   - GET  /api/settings/credentials     凭证配置状态（只回掩码，不回明文）
 *   - PUT  /api/settings/credentials     保存/清除凭证（api_key 为空 = 清除）
 *
 * 本模块只做「拼 URL + fetch + 解析」，不含 UI 依赖。
 */

import { resolveApiBaseUrl } from "./knowledgeClient";

const API_BASE: string = resolveApiBaseUrl();

export interface GitInfo {
    found: boolean;
    path: string | null;
    version: string | null;
}

export interface PythonInfo {
    found: boolean;
    python: string | null;
    py_launcher: string | null;
}

export interface CondaInfo {
    found: boolean;
    path: string | null;
}

export interface ClaudeCliInfo {
    found: boolean;
    path: string | null;
    version: string | null;
}

export interface EnvCheckResult {
    git: GitInfo;
    python: PythonInfo;
    conda: CondaInfo;
    claude_cli: ClaudeCliInfo;
    data_dir: { path: string; writable: boolean };
    static_served: boolean;
    credentials_configured: boolean;
    pip_index: string | null;
}

export interface CredentialsStatus {
    configured: boolean;
    source: "file" | "env" | null;
    key_mask: string | null;
    base_url: string | null;
    model: string | null;
    small_model: string | null;
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
        throw new Error(typeof detail === "string" ? detail : `HTTP ${res.status}`);
    }
    return (await res.json()) as T;
}

export function fetchEnvCheck(): Promise<EnvCheckResult> {
    return request<EnvCheckResult>("/api/system/env-check");
}

export function fetchCredentialsStatus(): Promise<CredentialsStatus> {
    return request<CredentialsStatus>("/api/settings/credentials");
}

export function saveCredentials(creds: {
    api_key: string;
    base_url: string;
    model: string;
    small_model: string;
}): Promise<CredentialsStatus> {
    return request<CredentialsStatus>("/api/settings/credentials", {
        method: "PUT",
        body: JSON.stringify(creds),
    });
}

export interface LongPathsResult {
    ok: boolean;
    enabled: boolean;
    cancelled?: boolean;
    detail: string;
}

export function fetchLongPathStatus(): Promise<{ enabled: boolean; platform: string }> {
    return request<{ enabled: boolean; platform: string }>("/api/system/long-paths");
}

/** 经 UAC 提权开启 Windows 长路径支持（装深层依赖 >260 字符会失败）。会弹一次管理员确认框。 */
export function enableLongPaths(): Promise<LongPathsResult> {
    return request<LongPathsResult>("/api/system/long-paths/enable", { method: "POST" });
}
