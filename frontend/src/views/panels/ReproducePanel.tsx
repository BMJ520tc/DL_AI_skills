// views/panels/ReproducePanel.tsx — 三并列入口「先复现」（需求四，阶段4 4a）。
// 论文选择 → 抽取实验条目 → 逐条确认（4.2 人工闸门）→ 复现执行 → 可信度结论。
// 全部驱动既有后端接口（模块二 4.2~4.4），任务轮询复用 useTaskPolling。

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import {
    confirmPaperItem,
    getPaperDetail,
    listKnowledge,
    postConclusion,
    postExtractItems,
    postReproduce,
    type PaperDetail,
} from "../../api/client";
import { useTaskPolling } from "../../hooks/useTaskPolling";

export type ReproducePanelProps = {
    projectId: string;
};

interface PanelTask {
    id: string;
    kind: string;
    after: () => Promise<void>;
}

/** 论文下拉选项的可读标题（title 缺失时回落到 id）。 */
function paperTitle(p: Record<string, unknown>): string {
    const title = typeof p.title === "string" && p.title ? p.title : "(未命名)";
    return `${title}（${String(p.paper_id).slice(0, 8)}）`;
}

export default function ReproducePanel({ projectId }: ReproducePanelProps) {
    const [papers, setPapers] = useState<Array<Record<string, unknown>>>([]);
    const [paperId, setPaperId] = useState("");
    const [detail, setDetail] = useState<PaperDetail | null>(null);
    const [task, setTask] = useState<PanelTask | null>(null);
    const [taskInfo, setTaskInfo] = useState<string | null>(null);
    const [banner, setBanner] = useState<string | null>(null);
    const [flash, setFlash] = useState<string | null>(null);

    useEffect(() => {
        void (async () => {
            try {
                setPapers(await listKnowledge("paper"));
            } catch (e) {
                setBanner(`论文列表读取失败：${e instanceof Error ? e.message : String(e)}`);
            }
        })();
    }, []);

    const refreshDetail = useCallback(async (id: string) => {
        if (!id) {
            setDetail(null);
            return;
        }
        try {
            setDetail(await getPaperDetail(id));
        } catch (e) {
            setBanner(`论文详情读取失败：${e instanceof Error ? e.message : String(e)}`);
        }
    }, []);

    useEffect(() => {
        void (async () => {
            if (!paperId) {
                setDetail(null);
                return;
            }
            try {
                setDetail(await getPaperDetail(paperId));
            } catch (e) {
                setBanner(`论文详情读取失败：${e instanceof Error ? e.message : String(e)}`);
            }
        })();
    }, [paperId]);

    // 同步重入保护：starter 返回前重复点击只发一次（与查看器操作链同口径）
    const busyRef = useRef(false);
    const [busy, setBusy] = useState(false);
    const runTask = useCallback((kind: string, starter: () => Promise<{ task_id: string }>, after: () => Promise<void>) => {
        if (busyRef.current) return;
        busyRef.current = true;
        setBusy(true);
        setBanner(null);
        setFlash(null);
        void (async () => {
            try {
                const { task_id } = await starter();
                setTaskInfo(null);
                setTask({ id: task_id, kind, after });
            } catch (e) {
                busyRef.current = false;
                setBusy(false);
                setBanner(e instanceof Error ? e.message : String(e));
            }
        })();
    }, []);

    useTaskPolling({
        taskId: task?.id ?? null,
        onProgress: t => setTaskInfo(`${t.task_type}：${t.status}${t.progress ? ` · ${t.progress}` : ""}`),
        onDone: () => {
            busyRef.current = false;
            setBusy(false);
            setTaskInfo(null);
            const after = task?.after;
            setTask(null);
            if (after) void after();
        },
        onError: message => {
            busyRef.current = false;
            setBusy(false);
            setBanner(message);
            setTask(null);
        },
    });

    const items = detail?.items ?? [];
    const confirmedCount = items.filter(i => i.status === "confirmed").length;
    const reproResults = detail?.reproduction_results ?? [];
    const conclusion = detail?.conclusion ?? null;

    const refreshAfter = async () => {
        await refreshDetail(paperId);
    };

    const confirmItem = async (itemId: string) => {
        try {
            await confirmPaperItem(paperId, itemId);
            await refreshDetail(paperId);
            setFlash("条目已确认");
        } catch (e) {
            setBanner(e instanceof Error ? e.message : String(e));
        }
    };

    return (
        <div style={{ fontSize: 12 }}>
            {/* 论文选择 */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <label style={{ color: "#94a3b8", fontSize: 11 }}>
                    论文：
                    <select
                        style={{ ...selectStyle, marginLeft: 6, maxWidth: 320 }}
                        value={paperId}
                        onChange={e => setPaperId(e.target.value)}
                        disabled={busy}
                    >
                        <option value="">（选择已入库论文）</option>
                        {papers.map(p => (
                            <option key={String(p.paper_id)} value={String(p.paper_id)}>
                                {paperTitle(p)}
                            </option>
                        ))}
                    </select>
                </label>
                {papers.length === 0 && <span style={{ color: "#64748b", fontSize: 11 }}>论文库为空（可先经检索下载论文）</span>}
            </div>

            {(banner || flash || taskInfo) && (
                <div style={{ marginTop: 8 }}>
                    {taskInfo && <div style={{ ...infoBanner, background: "#1e293b", borderColor: "#334155", color: "#cbd5e1" }}>{taskInfo}</div>}
                    {banner && (
                        <div style={{ ...infoBanner, background: "#3f1d1d", borderColor: "#b91c1c", color: "#fecaca" }}>
                            {banner}
                            <button style={{ ...btnStyle, marginLeft: 10 }} onClick={() => setBanner(null)}>关闭</button>
                        </div>
                    )}
                    {flash && (
                        <div style={infoBanner}>
                            {flash}
                            <button style={{ ...btnStyle, marginLeft: 10 }} onClick={() => setFlash(null)}>关闭</button>
                        </div>
                    )}
                </div>
            )}

            {!paperId && <div style={{ color: "#64748b", marginTop: 10 }}>选择论文后开始：抽取实验条目 → 逐条确认 → 复现 → 可信度结论。</div>}

            {paperId && (
                <>
                    {/* 操作按钮 */}
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                        <button style={btnStyle} disabled={busy} onClick={() => runTask("extract", () => postExtractItems(paperId), refreshAfter)}>
                            ① 抽取实验条目
                        </button>
                        <button
                            style={btnStyle}
                            disabled={busy || confirmedCount === 0}
                            onClick={() => runTask("reproduce", () => postReproduce(paperId, projectId), refreshAfter)}
                            title={confirmedCount === 0 ? "需先确认至少一条实验条目" : undefined}
                        >
                            ② 复现（已确认 {confirmedCount}/{items.length} 条）
                        </button>
                        <button
                            style={btnStyle}
                            disabled={busy || reproResults.length === 0}
                            onClick={() => runTask("conclusion", () => postConclusion(paperId), refreshAfter)}
                            title={reproResults.length === 0 ? "需先有复现结果" : undefined}
                        >
                            ③ 生成可信度结论
                        </button>
                    </div>

                    {/* 实验条目（4.2 确认闸门） */}
                    <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>实验条目</div>
                    {items.length === 0 ? (
                        <div style={{ color: "#64748b", fontSize: 11 }}>尚无条目（点「① 抽取实验条目」）。</div>
                    ) : (
                        <table style={tableStyle}>
                            <thead>
                                <tr>
                                    <th style={thStyle}>指标</th>
                                    <th style={thStyle}>报告值</th>
                                    <th style={thStyle}>状态</th>
                                    <th style={thStyle}>操作</th>
                                </tr>
                            </thead>
                            <tbody>
                                {items.map(item => (
                                    <tr key={String(item.item_id)}>
                                        <td style={tdStyle}>{String(item.metric_name ?? "?")}{item.metric_unit ? `（${item.metric_unit}）` : ""}</td>
                                        <td style={tdStyle}>{String(item.metric_value_reported ?? "—")}</td>
                                        <td style={tdStyle}>
                                            {item.status === "confirmed" ? <span style={{ color: "#16a34a" }}>已确认</span> : <span style={{ color: "#d97706" }}>待确认</span>}
                                        </td>
                                        <td style={tdStyle}>
                                            {item.status !== "confirmed" && (
                                                <button style={btnStyle} disabled={busy} onClick={() => void confirmItem(String(item.item_id))}>
                                                    确认
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}

                    {/* 复现对照 */}
                    <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>复现对照</div>
                    {reproResults.length === 0 ? (
                        <div style={{ color: "#64748b", fontSize: 11 }}>尚无复现结果（点「② 复现」）。</div>
                    ) : (
                        <table style={tableStyle}>
                            <thead>
                                <tr>
                                    <th style={thStyle}>指标</th>
                                    <th style={thStyle}>实测值</th>
                                    <th style={thStyle}>偏差</th>
                                    <th style={thStyle}>判定</th>
                                </tr>
                            </thead>
                            <tbody>
                                {reproResults.map(r => {
                                    const item = items.find(i => String(i.item_id) === String(r.item_id));
                                    return (
                                        <tr key={String(r.result_id)}>
                                            <td style={tdStyle}>{item ? String(item.metric_name ?? "?") : String(r.item_id).slice(0, 8)}</td>
                                            <td style={tdStyle}>{r.metric_value_actual !== null && r.metric_value_actual !== undefined ? String(r.metric_value_actual) : "—"}</td>
                                            <td style={tdStyle}>{r.deviation !== null && r.deviation !== undefined ? String(r.deviation) : "—"}</td>
                                            <td style={tdStyle}>{String(r.verdict ?? "—")}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    )}

                    {/* 可信度结论 */}
                    <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>可信度结论</div>
                    {conclusion ? (
                        <div style={{ ...infoBanner, background: "#0f2d1f", borderColor: "#16a34a" }}>
                            <div>
                                总体判定：<b>{String(conclusion.overall_verdict ?? "—")}</b>
                            </div>
                            {typeof conclusion.summary === "string" && conclusion.summary && (
                                <div style={{ marginTop: 4, color: "#94a3b8", fontSize: 11 }}>{conclusion.summary}</div>
                            )}
                        </div>
                    ) : (
                        <div style={{ color: "#64748b", fontSize: 11 }}>尚无结论（点「③ 生成可信度结论」）。</div>
                    )}
                </>
            )}
        </div>
    );
}

const btnStyle: CSSProperties = {
    border: "1px solid #334155",
    background: "#0f766e",
    color: "#e2e8f0",
    borderRadius: 6,
    padding: "4px 10px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
};
const selectStyle: CSSProperties = {
    border: "1px solid #334155",
    background: "#0f172a",
    color: "#e2e8f0",
    borderRadius: 4,
    padding: "3px 6px",
    fontSize: 12,
};
const infoBanner: CSSProperties = {
    background: "#0f2d1f",
    border: "1px solid #16a34a",
    color: "#bbf7d0",
    borderRadius: 8,
    padding: "8px 12px",
    fontSize: 12,
    marginBottom: 8,
};
const tableStyle: CSSProperties = { width: "100%", borderCollapse: "collapse", fontSize: 11 };
const thStyle: CSSProperties = { textAlign: "left", color: "#64748b", fontWeight: 600, padding: "3px 6px", borderBottom: "1px solid #1f2937" };
const tdStyle: CSSProperties = { padding: "4px 6px", borderBottom: "1px solid #1f2937", verticalAlign: "top" };
