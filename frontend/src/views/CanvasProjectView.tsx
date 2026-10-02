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
            await putGraph(projectId, mergeGraphIR(server, edited));
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
        <div style={{ height: "100vh", position: "relative", background: "#0b1220" }}>
            <FlowEditor key={projectId} initialGraph={graph} onSave={handleSave} />
            {/* 悬浮头部（画布编辑器自身上方） */}
            <div
                style={{
                    position: "absolute",
                    top: 10,
                    left: 10,
                    zIndex: 20,
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    background: "rgba(15, 23, 42, 0.92)",
                    border: "1px solid #1f2937",
                    borderRadius: 8,
                    padding: "6px 12px",
                    fontSize: 12,
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
                    节点参数可直接编辑，改动点右上「保存到项目」落盘（GraphIR 全量快照；版本树归阶段4）
                </span>
            </div>
        </div>
    );
}
