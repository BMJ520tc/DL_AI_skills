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
                            <h2 style={{ fontSize: 15, margin: "0 0 12px" }}>
                                结构化项目 <span style={{ color: "#64748b", fontSize: 12, fontWeight: 400 }}>（模块入库后生成，画布可编辑）</span>
                            </h2>
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
