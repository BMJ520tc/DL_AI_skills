// views/ProjectListView.tsx — 项目列表与创建（模块四 B1）。
// 不引 react-router：顶层 App state 切换 view。原始项目 → 模型查看器；
// 结构化项目 → 画布。项目加载（env/代码挂载）是异步的，忙碌时自动轮询。

import { useCallback, useEffect, useState, type CSSProperties } from "react";
import {
    ApiError,
    createProject,
    listProjects,
    type Project,
} from "../api/client";
import SearchDownloadPanel from "./panels/SearchDownloadPanel";

const BUSY_STATUSES = new Set(["loading", "preparing", "queued", "running"]);

export type ProjectListViewProps = {
    onOpenViewer: (projectId: string) => void;
    onOpenCanvas: (projectId: string) => void;
    onOpenSandbox: () => void;
};

export default function ProjectListView({ onOpenViewer, onOpenCanvas, onOpenSandbox }: ProjectListViewProps) {
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [sourceUrl, setSourceUrl] = useState("");
    const [name, setName] = useState("");
    const [creating, setCreating] = useState(false);
    const [createError, setCreateError] = useState<string | null>(null);
    const [modelName, setModelName] = useState("");
    const [modelParentId, setModelParentId] = useState("");
    const [creatingModel, setCreatingModel] = useState(false);
    const [createModelError, setCreateModelError] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        try {
            setProjects(await listProjects());
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    // 有项目还在加载/排队时自动轮询（每 3s），全部就绪后停止
    useEffect(() => {
        const busy = projects.some(p => BUSY_STATUSES.has(p.status));
        if (!busy) return;
        const timer = setInterval(() => void refresh(), 3000);
        return () => clearInterval(timer);
    }, [projects, refresh]);

    const handleCreate = async () => {
        if (!sourceUrl.trim()) {
            setCreateError("请填写仓库地址或本地路径");
            return;
        }
        setCreating(true);
        setCreateError(null);
        try {
            await createProject({
                project_type: "original",
                source_url: sourceUrl.trim(),
                name: name.trim() || undefined,
            });
            setSourceUrl("");
            setName("");
            await refresh();
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            setCreateError(e instanceof ApiError && e.status === 400 ? `创建/加载失败：${msg}` : msg);
        } finally {
            setCreating(false);
        }
    };

    // 画布新建模型（阶段4 4a）：创建空结构化项目并直接打开画布。
    // 可选父项目：不传时画布网络没有父项目，其运行面板需要自行选择/新建环境来源。
    const handleCreateModel = async () => {
        setCreatingModel(true);
        setCreateModelError(null);
        try {
            const { project_id } = await createProject({
                project_type: "structured",
                name: modelName.trim() || undefined,
                parent_project_id: modelParentId || undefined,
            });
            setModelName("");
            setModelParentId("");
            onOpenCanvas(project_id);
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            setCreateModelError(e instanceof ApiError && e.status === 400 ? `创建失败：${msg}` : msg);
        } finally {
            setCreatingModel(false);
        }
    };

    const originals = projects.filter(p => p.project_type === "original");
    const structured = projects.filter(p => p.project_type === "structured");

    const statusBadge = (status: string) => {
        const color =
            status === "ready"
                ? "#16a34a"
                : status === "failed"
                  ? "#dc2626"
                  : "#d97706";
        return (
            <span style={{
                border: `1px solid ${color}`,
                color,
                borderRadius: 999,
                padding: "1px 8px",
                fontSize: 11,
                fontWeight: 600,
                whiteSpace: "nowrap",
            }}>
                {status}
            </span>
        );
    };

    const renderTable = (items: Project[], structuredTable: boolean) => (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
                <tr>
                    <th style={th}>名称/来源</th>
                    {structuredTable && <th style={th}>父项目</th>}
                    <th style={th}>状态</th>
                    <th style={th}>创建时间</th>
                    <th style={th}>操作</th>
                </tr>
            </thead>
            <tbody>
                {items.map(p => (
                    <tr key={p.project_id} style={{ borderBottom: "1px solid #1f2937" }}>
                        <td style={td}>
                            <div style={{ fontWeight: 600 }}>{p.name || "(未命名)"}</div>
                            <div style={{ color: "#64748b", fontSize: 11, fontFamily: "monospace" }}>{p.source || p.project_id}</div>
                        </td>
                        {structuredTable && (
                            <td style={{ ...td, fontFamily: "monospace", fontSize: 11, color: "#94a3b8" }}>
                                {p.parent_project_id || "—"}
                            </td>
                        )}
                        <td style={td}>{statusBadge(p.status)}</td>
                        <td style={{ ...td, color: "#94a3b8", fontSize: 11 }}>{new Date(p.created_at).toLocaleString()}</td>
                        <td style={td}>
                            <button style={btn} onClick={() => (structuredTable ? onOpenCanvas(p.project_id) : onOpenViewer(p.project_id))}>
                                {structuredTable ? "打开画布" : "模型查看器"}
                            </button>
                        </td>
                    </tr>
                ))}
                {!items.length && (
                    <tr>
                        <td colSpan={structuredTable ? 5 : 4} style={{ ...td, color: "#64748b", textAlign: "center", padding: 24 }}>
                            暂无{structuredTable ? "结构化" : "原始"}项目
                        </td>
                    </tr>
                )}
            </tbody>
        </table>
    );

    return (
        <div style={{ minHeight: "100vh", background: "#0b1220", color: "#e2e8f0", padding: "32px 40px" }}>
            <div style={{ maxWidth: 1100, margin: "0 auto" }}>
                <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 24 }}>
                    <h1 style={{ fontSize: 22, margin: 0 }}>项目</h1>
                    <button style={{ ...btn, background: "transparent" }} onClick={onOpenSandbox}>
                        打开本地沙盒画布（原编辑器）
                    </button>
                </div>

                {error && (
                    <div style={banner}>
                        ⚠ 后端连接失败（请确认 uvicorn :8000 已启动）：{error}
                        <button style={{ ...btn, marginLeft: 12 }} onClick={() => void refresh()}>重试</button>
                    </div>
                )}

                {/* 创建原始项目（模块详细设计 3.3：挂载仓库/本地路径） */}
                <div style={{ ...card, marginBottom: 28 }}>
                    <h2 style={{ fontSize: 15, margin: "0 0 12px" }}>创建原始项目</h2>
                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                        <input
                            style={input}
                            placeholder="仓库地址或本地路径（如 D:/repos/resnet 或 https://github.com/…）"
                            value={sourceUrl}
                            onChange={e => setSourceUrl(e.target.value)}
                        />
                        <input
                            style={{ ...input, maxWidth: 220 }}
                            placeholder="名称（可选）"
                            value={name}
                            onChange={e => setName(e.target.value)}
                        />
                        <button style={btn} onClick={() => void handleCreate()} disabled={creating}>
                            {creating ? "创建中…" : "创建并挂载"}
                        </button>
                    </div>
                    {createError && <div style={{ color: "#f87171", fontSize: 12, marginTop: 8 }}>{createError}</div>}
                </div>

                {/* 检索与下载（模块一 2.4：论文检索/下载、地址抽取） */}
                <details style={{ ...card, marginBottom: 28 }}>
                    <summary style={{ cursor: "pointer", fontSize: 15, fontWeight: 700 }}>
                        检索与下载 <span style={{ color: "#64748b", fontSize: 12, fontWeight: 400 }}>（论文检索 → 单篇/批量下载入库；地址抽取 → 仓库克隆 + 数据集登记）</span>
                    </summary>
                    <div style={{ marginTop: 14 }}>
                        <SearchDownloadPanel />
                    </div>
                </details>

                {loading ? (
                    <div style={{ color: "#64748b" }}>加载中…</div>
                ) : (
                    <>
                        <div style={card}>
                            <h2 style={{ fontSize: 15, margin: "0 0 12px" }}>
                                原始项目 <span style={{ color: "#64748b", fontSize: 12, fontWeight: 400 }}>（结构分析 → 模型拆解 → 验证 → 入库）</span>
                            </h2>
                            {renderTable(originals, false)}
                        </div>
                        <div style={card}>
                            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: 10, marginBottom: 12 }}>
                                <h2 style={{ fontSize: 15, margin: 0 }}>
                                    结构化项目 <span style={{ color: "#64748b", fontSize: 12, fontWeight: 400 }}>（模块入库后生成，也可直接新建，画布可编辑）</span>
                                </h2>
                                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                                    <input
                                        style={{ ...input, flex: "none", minWidth: 180, maxWidth: 220, padding: "5px 10px", fontSize: 12 }}
                                        placeholder="新模型名称（可选）"
                                        value={modelName}
                                        onChange={e => setModelName(e.target.value)}
                                    />
                                    <select
                                        style={{ ...input, flex: "none", minWidth: 200, maxWidth: 280, padding: "5px 10px", fontSize: 12 }}
                                        value={modelParentId}
                                        onChange={e => setModelParentId(e.target.value)}
                                        title="可选：选定父项目后，该网络运行时的默认环境就是父原始项目的独立环境"
                                    >
                                        <option value="">父项目（可选，用于默认运行环境）</option>
                                        {originals.map(p => (
                                            <option key={p.project_id} value={p.project_id}>
                                                {p.name || p.project_id}
                                            </option>
                                        ))}
                                    </select>
                                    <button style={btn} onClick={() => void handleCreateModel()} disabled={creatingModel}>
                                        {creatingModel ? "创建中…" : "＋ 新建模型"}
                                    </button>
                                </div>
                            </div>
                            {createModelError && <div style={{ color: "#f87171", fontSize: 12, marginBottom: 8 }}>{createModelError}</div>}
                            {renderTable(structured, true)}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

const th: CSSProperties = { textAlign: "left", color: "#64748b", fontWeight: 600, fontSize: 12, padding: "6px 12px" };
const td: CSSProperties = { padding: "10px 12px", verticalAlign: "top" };
const btn: CSSProperties = {
    border: "1px solid #334155",
    background: "#0f766e",
    color: "#e2e8f0",
    borderRadius: 6,
    padding: "5px 12px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
};
const input: CSSProperties = {
    flex: 1,
    minWidth: 280,
    border: "1px solid #334155",
    background: "#0f172a",
    color: "#e2e8f0",
    borderRadius: 6,
    padding: "7px 10px",
    fontSize: 13,
};
const card: CSSProperties = {
    background: "#0f172a",
    border: "1px solid #1f2937",
    borderRadius: 10,
    padding: 18,
    marginBottom: 20,
};
const banner: CSSProperties = {
    background: "#7f1d1d",
    border: "1px solid #b91c1c",
    color: "#fecaca",
    borderRadius: 8,
    padding: "10px 14px",
    fontSize: 13,
    marginBottom: 20,
};
