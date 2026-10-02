// features/network/VersionPanel.tsx — 画布版本面板（阶段4 4d-2，模块详细设计 7.6）。
// 版本树（git 提交历史，按父子关系缩进渲染，标注当前/训练/回退版本）→ 点选两版本
// 「对比」出代码差异（再生成代码 diff，与导出/训练同源）与参数差异表 →
// 「回退」检出目标版本并替换画布（回退本身记为新版本，后端语义）。
// 回退成功回调 onRollback(恢复出的图)，画布由 FlowContent 直接替换（一次往返）。

import { useCallback, useEffect, useState } from "react";
import type { GraphIR } from "../../types/graph";
import {
    compareVersions, getVersionTree, postRollback,
    type ParamDiff, type VersionCompare, type VersionTree,
} from "../../api/client";

export type VersionPanelProps = {
    projectId: string;
    /** 回退成功：把目标版本的图替换进画布（graphIRToFlow + setNodes/setEdges）。 */
    onRollback: (graph: GraphIR) => void;
    onClose: () => void;
};

const panelStyle: React.CSSProperties = {
    position: "absolute",
    top: 56,
    right: 12,
    width: 420,
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

/** 展示值格式化：缺失（后端 None）给「无」，对象/数组走 JSON。 */
function fmt(v: unknown): string {
    if (v === null || v === undefined) return "（无）";
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
}

function commitTime(iso: string): string {
    return iso.slice(0, 16).replace("T", " ");
}

/** 版本行徽标：训练运行（run_summary）与回退（rollback_to）。 */
function badgesOf(message: string, meta: VersionTree["versions"][number]["meta"]): string {
    const parts: string[] = [];
    if (message.startsWith("训练运行")) parts.push("训练");
    if (meta?.run_summary?.task_id) parts.push(`任务 ${meta.run_summary.task_id.slice(0, 8)}`);
    if (meta?.rollback_to) parts.push("回退");
    return parts.join(" · ");
}

function diffLineStyle(line: string): React.CSSProperties {
    if (line.startsWith("+")) return { color: "#4ade80", background: "rgba(74,222,128,0.07)" };
    if (line.startsWith("-")) return { color: "#f87171", background: "rgba(248,113,113,0.07)" };
    if (line.startsWith("@@")) return { color: "#7dd3fc" };
    return { color: "#94a3b8" };
}

function paramDiffIsEmpty(d: ParamDiff): boolean {
    return (
        d.nodes_added.length === 0 &&
        d.nodes_removed.length === 0 &&
        d.nodes_changed.length === 0 &&
        d.edges_added.length === 0 &&
        d.edges_removed.length === 0
    );
}

export default function VersionPanel({ projectId, onRollback, onClose }: VersionPanelProps) {
    const [tree, setTree] = useState<VersionTree | null>(null);
    const [treeError, setTreeError] = useState<string | null>(null);

    const [selA, setSelA] = useState<string | null>(null);
    const [selB, setSelB] = useState<string | null>(null);
    const [compare, setCompare] = useState<VersionCompare | null>(null);
    const [compareBusy, setCompareBusy] = useState(false);
    const [compareError, setCompareError] = useState<string | null>(null);
    const [tab, setTab] = useState<"code" | "param">("code");

    const [rollbackTarget, setRollbackTarget] = useState<string | null>(null);
    const [rollbackBusy, setRollbackBusy] = useState(false);
    const [rollbackError, setRollbackError] = useState<string | null>(null);

    const loadTree = useCallback(async () => {
        try {
            setTree(await getVersionTree(projectId));
            setTreeError(null);
        } catch (e) {
            setTreeError(e instanceof Error ? e.message : String(e));
        }
    }, [projectId]);

    useEffect(() => {
        void loadTree();
    }, [loadTree]);

    const byCommit = new Map((tree?.versions ?? []).map(v => [v.commit, v]));

    const toggleSelect = useCallback((commit: string) => {
        if (selA === commit) {
            setSelA(null);
        } else if (selB === commit) {
            setSelB(null);
        } else if (!selA) {
            setSelA(commit);
        } else if (!selB) {
            setSelB(commit);
        } else {
            setSelA(commit);
            setSelB(null);
        }
    }, [selA, selB]);

    const handleCompare = useCallback(async () => {
        if (!selA || !selB || compareBusy) return;
        setCompareBusy(true);
        setCompareError(null);
        setCompare(null);
        try {
            setCompare(await compareVersions(projectId, selA, selB));
        } catch (e) {
            setCompareError(e instanceof Error ? e.message : String(e));
        } finally {
            setCompareBusy(false);
        }
    }, [projectId, selA, selB, compareBusy]);

    const handleRollback = useCallback(async () => {
        if (!rollbackTarget || rollbackBusy) return;
        setRollbackBusy(true);
        setRollbackError(null);
        try {
            const res = await postRollback(projectId, rollbackTarget);
            onRollback(res.graph);
            setRollbackTarget(null);
            setSelA(null);
            setSelB(null);
            setCompare(null);
            await loadTree();
        } catch (e) {
            setRollbackError(e instanceof Error ? e.message : String(e));
        } finally {
            setRollbackBusy(false);
        }
    }, [projectId, rollbackTarget, rollbackBusy, onRollback, loadTree]);

    const selShort = (commit: string | null) =>
        commit ? (byCommit.get(commit)?.short ?? commit.slice(0, 7)) : "";

    return (
        <div style={panelStyle}>
            <div style={{ display: "flex", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontWeight: 700 }}>版本管理</span>
                <button
                    onClick={() => void loadTree()}
                    style={{ ...smallButtonStyle, marginLeft: 12 }}
                >
                    刷新
                </button>
                <button
                    onClick={onClose}
                    style={{
                        marginLeft: "auto",
                        border: "none",
                        background: "transparent",
                        color: "#94a3b8",
                        cursor: "pointer",
                        fontSize: 14,
                    }}
                >
                    ✕
                </button>
            </div>

            {treeError && (
                <div style={{ color: "#f87171", marginBottom: 8 }}>
                    版本树加载失败：{treeError}
                </div>
            )}

            {!tree && !treeError && <div style={{ color: "#64748b" }}>版本树加载中…</div>}

            {tree && tree.versions.length === 0 && (
                <div style={{ color: "#64748b" }}>尚无版本（保存画布后形成第一个版本）</div>
            )}

            {tree && tree.versions.length > 0 && (
                <>
                    <div style={{ color: "#64748b", fontSize: 11, marginBottom: 4 }}>
                        共 {tree.versions.length} 个版本；点选两个版本后「对比」，可对任意历史版本「回退」
                    </div>
                    {tree.versions.map((v, i) => {
                        const selected = selA === v.commit || selB === v.commit;
                        const isCurrent = tree.current === v.short;
                        const isRoot = v.parents.length === 0;
                        const last = i === tree.versions.length - 1;
                        const parentBelow = !isRoot && !last && v.parents.includes(tree.versions[i + 1].commit);
                        return (
                            <div key={v.commit} style={{ display: "flex", alignItems: "stretch" }}>
                                {/* 演化关系左槽：竖线串起整条链，● 为提交节点；父提交不在
                                    正下方的行用 └─ 拐入（历史线性时恒为 ●─） */}
                                <div
                                    style={{
                                        width: 20,
                                        flexShrink: 0,
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        color: "#475569",
                                        fontFamily: "monospace",
                                        fontSize: 10,
                                        borderLeft: last ? "none" : "1px solid #334155",
                                    }}
                                >
                                    {isRoot ? "●" : parentBelow ? "●─" : "└─"}
                                </div>
                                <div
                                    data-version={v.short}
                                    onClick={() => toggleSelect(v.commit)}
                                    style={{
                                        flex: 1,
                                        minWidth: 0,
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 6,
                                        padding: "5px 8px",
                                        marginTop: 4,
                                        borderRadius: 6,
                                        cursor: "pointer",
                                        background: selected ? "#1e3a5f" : "#0b1220",
                                        border: selected
                                            ? "1px solid #3b82f6"
                                            : isCurrent
                                              ? "1px solid #0f766e"
                                              : "1px solid #1f2a2f",
                                        fontSize: 11,
                                    }}
                                >
                                    <span style={{ fontFamily: "monospace", color: "#93c5fd" }}>{v.short}</span>
                                    <span style={{ color: "#cbd5e1" }}>{v.message}</span>
                                    <span style={{ color: "#64748b" }}>{commitTime(v.committed_at)}</span>
                                    <span style={{ color: "#64748b" }}>
                                        {v.meta ? `${v.meta.node_count} 节点/${v.meta.edge_count} 连线` : "无元数据"}
                                    </span>
                                    {badgesOf(v.message, v.meta) && (
                                        <span style={{ color: "#fbbf24" }}>{badgesOf(v.message, v.meta)}</span>
                                    )}
                                    {isCurrent && (
                                        <span style={{ color: "#5eead4", fontWeight: 700 }}>当前</span>
                                    )}
                                    <button
                                        onClick={e => {
                                            e.stopPropagation();
                                            setRollbackTarget(v.commit);
                                            setRollbackError(null);
                                        }}
                                        disabled={isCurrent || rollbackBusy}
                                        style={{
                                            ...smallButtonStyle,
                                            marginLeft: "auto",
                                            opacity: isCurrent ? 0.4 : 1,
                                            cursor: isCurrent ? "not-allowed" : "pointer",
                                        }}
                                    >
                                        回退
                                    </button>
                                </div>
                            </div>
                        );
                    })}
                </>
            )}

            {tree && tree.versions.length > 1 && (
                <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
                    <button
                        onClick={() => void handleCompare()}
                        disabled={!selA || !selB || compareBusy}
                        style={{
                            ...smallButtonStyle,
                            opacity: !selA || !selB ? 0.5 : 1,
                            cursor: !selA || !selB ? "not-allowed" : "pointer",
                        }}
                    >
                        {compareBusy ? "对比中…" : "对比所选版本"}
                    </button>
                    <span style={{ color: "#64748b", fontSize: 11 }}>
                        {selA || selB ? `${selShort(selA)} → ${selShort(selB) || "?"}` : "未选择版本"}
                    </span>
                </div>
            )}

            {compareError && (
                <div style={{ color: "#f87171", marginTop: 8, whiteSpace: "pre-wrap" }}>{compareError}</div>
            )}

            {compare && (
                <div style={{ marginTop: 10, borderTop: "1px solid #1f2a2f", paddingTop: 8 }}>
                    <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                        {(["code", "param"] as const).map(t => (
                            <button
                                key={t}
                                onClick={() => setTab(t)}
                                style={{
                                    ...smallButtonStyle,
                                    background: tab === t ? "#0f766e" : "#1e293b",
                                }}
                            >
                                {t === "code" ? "代码差异" : "参数差异"}
                            </button>
                        ))}
                    </div>

                    {tab === "code" && (
                        compare.code_diff_error ? (
                            <div style={{ color: "#f87171", whiteSpace: "pre-wrap" }}>
                                {compare.code_diff_error}
                            </div>
                        ) : compare.code_diff && compare.code_diff.length > 0 ? (
                            <pre
                                style={{
                                    margin: 0,
                                    fontFamily: "monospace",
                                    fontSize: 10,
                                    whiteSpace: "pre-wrap",
                                    wordBreak: "break-all",
                                    background: "#0b1220",
                                    border: "1px solid #1f2a2f",
                                    borderRadius: 6,
                                    padding: 8,
                                    maxHeight: 240,
                                    overflowY: "auto",
                                }}
                            >
                                {compare.code_diff.map((line, i) => (
                                    <div key={i} style={diffLineStyle(line)}>{line || " "}</div>
                                ))}
                            </pre>
                        ) : (
                            <div style={{ color: "#64748b" }}>两版代码一致</div>
                        )
                    )}

                    {tab === "param" && (
                        paramDiffIsEmpty(compare.param_diff) ? (
                            <div style={{ color: "#64748b" }}>两版参数一致</div>
                        ) : (
                            <div style={{ fontSize: 11 }}>
                                {compare.param_diff.nodes_added.length > 0 && (
                                    <div style={{ marginBottom: 6 }}>
                                        <div style={{ color: "#4ade80", fontWeight: 700 }}>新增节点</div>
                                        {compare.param_diff.nodes_added.map(n => (
                                            <div key={n.id} style={{ marginLeft: 10 }}>
                                                {n.id}（{n.type}）
                                            </div>
                                        ))}
                                    </div>
                                )}
                                {compare.param_diff.nodes_removed.length > 0 && (
                                    <div style={{ marginBottom: 6 }}>
                                        <div style={{ color: "#f87171", fontWeight: 700 }}>移除节点</div>
                                        {compare.param_diff.nodes_removed.map(n => (
                                            <div key={n.id} style={{ marginLeft: 10 }}>
                                                {n.id}（{n.type}）
                                            </div>
                                        ))}
                                    </div>
                                )}
                                {compare.param_diff.nodes_changed.length > 0 && (
                                    <div style={{ marginBottom: 6 }}>
                                        <div style={{ color: "#fbbf24", fontWeight: 700 }}>参数变化</div>
                                        {compare.param_diff.nodes_changed.map(n => (
                                            <div key={n.id} style={{ marginLeft: 10, marginTop: 4 }}>
                                                <div>{n.id}（{n.type}）</div>
                                                <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 2 }}>
                                                    <tbody>
                                                        {n.param_changes.map(c => (
                                                            <tr key={c.key}>
                                                                <td style={{ color: "#94a3b8", padding: "1px 4px", verticalAlign: "top" }}>{c.key}</td>
                                                                <td style={{ color: "#f87171", padding: "1px 4px", fontFamily: "monospace", wordBreak: "break-all" }}>{fmt(c.old)}</td>
                                                                <td style={{ padding: "1px 4px", color: "#64748b" }}>→</td>
                                                                <td style={{ color: "#4ade80", padding: "1px 4px", fontFamily: "monospace", wordBreak: "break-all" }}>{fmt(c.new)}</td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        ))}
                                    </div>
                                )}
                                {compare.param_diff.edges_added.length > 0 && (
                                    <div style={{ marginBottom: 6 }}>
                                        <div style={{ color: "#4ade80", fontWeight: 700 }}>新增连线</div>
                                        {compare.param_diff.edges_added.map(e => (
                                            <div key={e.id} style={{ marginLeft: 10 }}>
                                                {e.source} → {e.target}（{e.id}）
                                            </div>
                                        ))}
                                    </div>
                                )}
                                {compare.param_diff.edges_removed.length > 0 && (
                                    <div style={{ marginBottom: 6 }}>
                                        <div style={{ color: "#f87171", fontWeight: 700 }}>移除连线</div>
                                        {compare.param_diff.edges_removed.map(e => (
                                            <div key={e.id} style={{ marginLeft: 10 }}>
                                                {e.source} → {e.target}（{e.id}）
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )
                    )}
                </div>
            )}

            {rollbackTarget && (
                <div
                    style={{
                        marginTop: 10,
                        borderTop: "1px solid #1f2a2f",
                        paddingTop: 8,
                    }}
                >
                    <div style={{ marginBottom: 6 }}>
                        回退到 <span style={{ fontFamily: "monospace", color: "#93c5fd" }}>{selShort(rollbackTarget)}</span>
                        （{byCommit.get(rollbackTarget)?.message ?? ""}）？
                        画布将变为该版本内容，回退动作本身记为一个新版本。
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                        <button
                            onClick={() => void handleRollback()}
                            disabled={rollbackBusy}
                            style={{
                                ...smallButtonStyle,
                                background: "#7f1d1d",
                                cursor: rollbackBusy ? "wait" : "pointer",
                            }}
                        >
                            {rollbackBusy ? "回退中…" : "确认回退"}
                        </button>
                        <button
                            onClick={() => setRollbackTarget(null)}
                            disabled={rollbackBusy}
                            style={smallButtonStyle}
                        >
                            取消
                        </button>
                    </div>
                </div>
            )}

            {rollbackError && (
                <div style={{ color: "#f87171", marginTop: 8, whiteSpace: "pre-wrap" }}>{rollbackError}</div>
            )}
        </div>
    );
}
