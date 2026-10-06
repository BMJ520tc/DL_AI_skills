// features/network/ArchSuggestPanel.tsx — 架构级自迭代建议面板（需求六.1 延伸，模块详细设计 8.5）。
// 流程：生成建议（agent 起草，任务轮询）→ 列表（换模块/加层/改连接，带理由与来源知识）
//      → 「应用到画布」经 arch-apply 得到新图 → onApply(graph) 整图替换画布（不写盘）
//      → 用户点「保存」生成新版本、再去「运行训练」手动训练。
// 边界：改动必须经用户确认；画布自生成建议后若被改动，arch-apply 返回 409，此处提示重新生成。

import { useCallback, useEffect, useState } from "react";
import type { GraphIR } from "../../types/graph";
import { ApiError, getArchSuggestions, postArchApply, postArchSuggest, type ArchSuggestReport, type ArchSuggestion } from "../../api/client";
import { useTaskPolling } from "../../hooks/useTaskPolling";

export type ArchSuggestPanelProps = {
    projectId: string;
    /** 应用成功：把返回的新图替换进画布（graphIRToFlow + setNodes/setEdges），尚未保存。 */
    onApply: (graph: GraphIR) => void;
    onClose: () => void;
};

const panelStyle: React.CSSProperties = {
    position: "absolute",
    top: 56,
    right: 12,
    width: 440,
    maxHeight: "calc(100% - 80px)",
    overflowY: "auto",
    zIndex: 20,
    background: "#0f172a",
    border: "1px solid #1f2a2f",
    borderRadius: 10,
    padding: "12px 14px",
    color: "#e2e8f0",
    fontSize: 12,
    boxShadow: "0 12px 32px rgba(0,0,0,0.45)",
};

const buttonStyle: React.CSSProperties = {
    border: "1px solid #1f2a2f",
    borderRadius: 6,
    padding: "5px 12px",
    fontWeight: 600,
    fontSize: 12,
    cursor: "pointer",
    background: "#7c3aed",
    color: "#e2e8f0",
};

const smallButtonStyle: React.CSSProperties = {
    border: "1px solid #1f2a2f",
    borderRadius: 6,
    padding: "3px 10px",
    fontWeight: 600,
    fontSize: 11,
    cursor: "pointer",
    background: "#1e293b",
    color: "#e2e8f0",
};

const OP_COLORS: Record<string, string> = {
    replace_module: "#0e7490",
    add_layer: "#15803d",
    rewire: "#b45309",
};

function payloadSummary(s: ArchSuggestion): string {
    const p = s.payload || {};
    if (s.op === "replace_module") return `→ 模块 ${p.new_module_ref ?? "(未指定)"}`;
    if (s.op === "add_layer") {
        const t = p.new_node_type === "module_ref" ? `模块 ${p.new_module_ref ?? ""}` : String(p.new_node_type ?? "");
        const anchor = p.anchor_edge_id ? `边 ${p.anchor_edge_id}` : `节点 ${p.anchor_node_id ?? "?"}`;
        return `在 ${anchor} 后插入 ${t}`;
    }
    const edits = (p.edits as unknown[] | undefined) ?? [];
    return edits
        .map((e) => {
            const ed = e as { action?: string; source?: string; target?: string };
            return `${ed.action === "disconnect" ? "断开" : "连接"} ${ed.source}→${ed.target}`;
        })
        .join("；");
}

export default function ArchSuggestPanel({ projectId, onApply, onClose }: ArchSuggestPanelProps) {
    const [report, setReport] = useState<ArchSuggestReport | null>(null);
    const [taskId, setTaskId] = useState<string | null>(null);
    const [stage, setStage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [notice, setNotice] = useState<string | null>(null);
    const [applyBusy, setApplyBusy] = useState<string | null>(null);
    const [hint, setHint] = useState("");

    const loadReport = useCallback(async () => {
        try {
            setReport(await getArchSuggestions(projectId));
        } catch (e) {
            if (e instanceof ApiError && e.status === 404) {
                setReport(null); // 尚未生成过：不是错误
            } else {
                setError(e instanceof Error ? e.message : String(e));
            }
        }
    }, [projectId]);

    useEffect(() => {
        void loadReport();
    }, [loadReport]);

    const start = useCallback(async () => {
        setError(null);
        setNotice(null);
        setStage("发起…");
        try {
            const r = await postArchSuggest(projectId, hint.trim() ? { hint: hint.trim() } : {});
            setTaskId(r.task_id);
        } catch (e) {
            setStage(null);
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [projectId, hint]);

    useTaskPolling({
        taskId,
        onProgress: (task) => {
            const p = (task.progress ?? {}) as Record<string, unknown>;
            setStage(typeof p.stage === "string" ? p.stage : "处理中…");
        },
        onDone: () => {
            setTaskId(null);
            setStage(null);
            void loadReport();
        },
        onError: (message) => {
            setTaskId(null);
            setStage(null);
            setError(message);
        },
    });

    const apply = useCallback(async (s: ArchSuggestion) => {
        if (!report) return;
        setApplyBusy(s.suggestion_id);
        setError(null);
        setNotice(null);
        try {
            const r = await postArchApply(projectId, { task_id: report.task_id, suggestion_id: s.suggestion_id });
            onApply(r.graph);
            setNotice("已应用到画布（尚未保存）。请点「保存」生成新版本，再去「运行训练」手动训练。");
        } catch (e) {
            if (e instanceof ApiError && e.status === 409) {
                setError("画布自生成建议后已变更，请重新生成建议。");
                void loadReport();
            } else {
                setError(e instanceof Error ? e.message : String(e));
            }
        } finally {
            setApplyBusy(null);
        }
    }, [projectId, report, onApply, loadReport]);

    const suggestions = report?.suggestions ?? [];

    return (
        <div style={panelStyle}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <strong style={{ fontSize: 13 }}>架构建议</strong>
                <button style={smallButtonStyle} onClick={onClose}>关闭</button>
            </div>

            <div style={{ color: "#94a3b8", marginBottom: 8, lineHeight: 1.5 }}>
                由 agent 基于「当前画布 + 已确认知识 + 入库模块」起草架构改动建议。应用后由你确认保存（= 新版本）并手动训练。
            </div>

            <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                <input
                    value={hint}
                    onChange={(e) => setHint(e.target.value)}
                    placeholder="可选：补充说明（如目标任务/想改进的方向）"
                    style={{
                        flex: 1, background: "#020617", border: "1px solid #1f2a2f", borderRadius: 6,
                        padding: "5px 8px", color: "#e2e8f0", fontSize: 12,
                    }}
                />
                <button style={buttonStyle} onClick={() => void start()} disabled={!!taskId}>
                    {taskId ? "生成中…" : "生成建议"}
                </button>
            </div>

            {stage && <div style={{ color: "#7dd3fc", marginBottom: 6 }}>进度：{stage}</div>}
            {notice && (
                <div style={{ color: "#4ade80", background: "rgba(74,222,128,0.08)", padding: "6px 8px", borderRadius: 6, marginBottom: 8 }}>
                    {notice}
                </div>
            )}
            {error && (
                <div style={{ color: "#f87171", background: "rgba(248,113,113,0.08)", padding: "6px 8px", borderRadius: 6, marginBottom: 8 }}>
                    {error}
                </div>
            )}
            {report?.agent_error && (
                <div style={{ color: "#fbbf24", marginBottom: 8 }}>agent 说明：{report.agent_error}</div>
            )}

            {report && (
                <div style={{ color: "#64748b", marginBottom: 6 }}>
                    报告 {report.task_id.slice(0, 8)} · 模块库 {report.module_catalog_size} 个 ·{" "}
                    知识带入 {Object.entries(report.knowledge_used).map(([k, v]) => `${k}:${v}`).join(" ") || "无"}
                </div>
            )}

            {!report && !taskId && (
                <div style={{ color: "#94a3b8" }}>还没有建议，点上方「生成建议」。</div>
            )}
            {report && suggestions.length === 0 && (
                <div style={{ color: "#94a3b8" }}>本次没有可用的架构改动建议（可重试或补充说明）。</div>
            )}

            {suggestions.map((s) => (
                <div key={s.suggestion_id} style={{
                    border: "1px solid #1f2a2f", borderRadius: 8, padding: "8px 10px", marginBottom: 8,
                    background: s.valid ? "#0b1220" : "#1a1013",
                }}>
                    <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 4 }}>
                        <span style={{
                            background: OP_COLORS[s.op] ?? "#334155", color: "#fff", borderRadius: 4,
                            padding: "1px 6px", fontSize: 11, fontWeight: 600,
                        }}>{s.op_label || s.op}</span>
                        <span style={{ color: "#94a3b8" }}>节点 {s.target_node_id || "?"}</span>
                        {!s.valid && <span style={{ color: "#f87171", marginLeft: "auto" }}>不可应用</span>}
                    </div>
                    <div style={{ marginBottom: 4 }}>{s.description || "（无说明）"}</div>
                    <div style={{ color: "#94a3b8", marginBottom: 4 }}>{payloadSummary(s)}</div>
                    {s.rationale && (
                        <div style={{ color: "#64748b", marginBottom: 4 }}>理由：{s.rationale}</div>
                    )}
                    {s.source_knowledge_ids.length > 0 && (
                        <div style={{ color: "#64748b", marginBottom: 4 }}>
                            来源知识：{s.source_knowledge_ids.map(id => id.slice(0, 10)).join(", ")}
                        </div>
                    )}
                    {!s.valid && s.invalid_reason && (
                        <div style={{ color: "#f87171", marginBottom: 4 }}>{s.invalid_reason}</div>
                    )}
                    <button
                        style={{ ...smallButtonStyle, opacity: s.valid ? 1 : 0.5 }}
                        disabled={!s.valid || applyBusy === s.suggestion_id}
                        onClick={() => void apply(s)}
                    >
                        {applyBusy === s.suggestion_id ? "应用中…" : "应用到画布"}
                    </button>
                </div>
            ))}
        </div>
    );
}
