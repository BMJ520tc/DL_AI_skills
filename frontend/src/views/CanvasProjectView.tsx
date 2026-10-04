// views/CanvasProjectView.tsx — 结构化项目画布（模块四 B3，模块详细设计 7.1）。
// 从后端读 graph.json（GraphIR v2）→ FlowEditor external 模式渲染（IrNode 通用渲染、
// data.handles 动态句柄、params 编辑）→ 「保存到项目」全量快照覆盖 PUT。
// key={projectId} 重挂载隔离不同项目；修改回流 IR / 版本树归阶段4。

import { useCallback, useEffect, useState } from "react";
import FlowEditor from "../FlowEditor";
import { ApiError, getGraph, getProject, putGraph, type Project } from "../api/client";
import type { GraphIR } from "../types/graph";
import { mergeGraphIR } from "../utils/irAdapter";

export type CanvasProjectViewProps = {
    projectId: string;
    onBack: () => void;
};

export default function CanvasProjectView({ projectId, onBack }: CanvasProjectViewProps) {
    const [graph, setGraph] = useState<GraphIR | null>(null);
    const [project, setProject] = useState<Project | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        void (async () => {
            setLoading(true);
            setError(null);
            try {
                const [g, p] = await Promise.all([getGraph(projectId), getProject(projectId)]);
                setGraph(g);
                setProject(p);
            } catch (e) {
                setError(
                    e instanceof ApiError && e.status === 404
                        ? "该结构化项目尚无画布快照（graph.json）"
                        : e instanceof Error
                          ? e.message
                          : String(e)
                );
            } finally {
                setLoading(false);
            }
        })();
    }, [projectId]);

    const handleSave = useCallback(
        async (edited: GraphIR) => {
            // 先取服务端权威 graph，再 merge（保留后端字段），避免全量重建覆盖
            const server = await getGraph(projectId);
            const saved = await putGraph(projectId, mergeGraphIR(server, edited));
            // 版本提交失败不连坐保存本身（后端把原因放在 version_error）：必须带回给画布，
            // 让界面显示「已保存到项目，但版本节点未生成：<原因>」，而不是一律「已保存 ✓」。
            const versionError =
                typeof saved.version_error === "string" && saved.version_error.trim() ? saved.version_error : null;
            return { versionError };
        },
        [projectId]
    );

    if (loading) {
        return (
            <div style={{ height: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#0b1220", color: "#64748b", fontSize: 14 }}>
                画布加载中…
            </div>
        );
    }

    if (error || !graph) {
        return (
            <div style={{ height: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14, background: "#0b1220", color: "#e2e8f0" }}>
                <div style={{ color: "#f87171", fontSize: 14 }}>{error ?? "画布加载失败"}</div>
                <button
                    style={{ border: "1px solid #334155", background: "#0f766e", color: "#e2e8f0", borderRadius: 6, padding: "6px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                    onClick={onBack}
                >
                    ← 返回项目列表
                </button>
            </div>
        );
    }

    return (
        // 纵向布局：顶部信息栏占一行、画布占其余高度 —— 原先头部绝对定位会盖住画布自身的工具栏
        <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "#0b1220" }}>
            <div
                style={{
                    flex: "0 0 auto",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    background: "#0f172a",
                    borderBottom: "1px solid #1f2937",
                    padding: "6px 12px",
                    fontSize: 12,
                    position: "relative",   // zIndex 对 static 元素无效，补上定位才生效
                    zIndex: 20,
                }}
            >
                <button
                    style={{ border: "1px solid #334155", background: "transparent", color: "#e2e8f0", borderRadius: 6, padding: "3px 10px", fontSize: 12, cursor: "pointer" }}
                    onClick={onBack}
                >
                    ← 返回
                </button>
                <span style={{ fontWeight: 700 }}>
                    {project?.name || "结构化项目"}
                    <span style={{ color: "#64748b", fontFamily: "monospace", fontWeight: 400, marginLeft: 8 }}>{projectId}</span>
                </span>
                <span style={{ color: "#94a3b8", fontSize: 11 }}>
                    节点参数可直接编辑，改动点右上「保存到项目」落盘；「导出代码 / 运行训练」用后端同一引擎（导出即所训）
                </span>
            </div>
            <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
                <FlowEditor key={projectId} initialGraph={graph} onSave={handleSave} projectId={projectId} />
            </div>
        </div>
    );
}
