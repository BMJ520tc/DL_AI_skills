import { useCallback, useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import {
    getMultiModelReport,
    listKnowledge,
    startMultiModel,
    type MultiModelReport,
    type RunRecord,
} from "../api/client";
import { useTaskPolling } from "../hooks/useTaskPolling";

/**
 * 多模型综合分析面板（模块详细设计 8.4，需求六.2 扩展）。
 *
 * - 勾选 ≥2 个评估运行记录（run_type=eval/baseline）→ POST /api/multi-model
 * - 任务轮询 → GET /api/multi-model/{task_id} 取回报告
 * - 展示：一致/分歧样本数、融合方式、融合前后指标、分歧归因、入库的 fusion_insight
 *
 * 独立自包含：不依赖画布状态，不改动 FlowEditor 既有逻辑。
 */

type Props = { onClose: () => void };

const PANEL_STYLE: CSSProperties = {
    position: "fixed",
    top: 0,
    right: 0,
    height: "100vh",
    width: "min(760px, 94vw)",
    background: "#0f1115",
    borderLeft: "1px solid #262b36",
    boxShadow: "-18px 0 40px rgba(0, 0, 0, 0.55)",
    display: "flex",
    flexDirection: "column",
    zIndex: 60,
    color: "#e6edf3",
    fontSize: 13,
};

const BUTTON_BASE: CSSProperties = {
    padding: "6px 14px",
    borderRadius: 6,
    border: "1px solid #3f3f46",
    cursor: "pointer",
    fontSize: 13,
};

function parseWindow(rec: RunRecord): { started: string } {
    return { started: (rec.started_at || "").slice(0, 16).replace("T", " ") };
}

type JobState =
    | { status: "idle" }
    | { status: "running"; taskId: string }
    | { status: "ready"; report: MultiModelReport }
    | { status: "error"; message: string };

export default function MultiModelPanel({ onClose }: Props) {
    const [runs, setRuns] = useState<RunRecord[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [selected, setSelected] = useState<string[]>([]);
    const [job, setJob] = useState<JobState>({ status: "idle" });

    // Esc 关闭
    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") onClose();
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [onClose]);

    useEffect(() => {
        void (async () => {
            try {
                const rows = (await listKnowledge("run", 300)) as unknown as RunRecord[];
                // 只保留可作为分析输入的评估运行（有逐样本预测产物）
                setRuns(rows.filter(r => r.run_type === "eval" || r.run_type === "baseline"));
            } catch (e) {
                setLoadError(e instanceof Error ? e.message : String(e));
            } finally {
                setLoading(false);
            }
        })();
    }, []);

    const toggle = useCallback((runId: string) => {
        setSelected(prev => (prev.includes(runId) ? prev.filter(x => x !== runId) : [...prev, runId]));
    }, []);

    useTaskPolling({
        taskId: job.status === "running" ? job.taskId : null,
        onDone: async () => {
            if (job.status !== "running") return;
            try {
                const report = await getMultiModelReport(job.taskId);
                setJob({ status: "ready", report });
            } catch (e) {
                setJob({ status: "error", message: e instanceof Error ? e.message : String(e) });
            }
        },
        onError: message => setJob({ status: "error", message }),
    });

    const handleStart = useCallback(async () => {
        if (selected.length < 2) {
            setJob({ status: "error", message: "请至少勾选两个评估运行记录" });
            return;
        }
        try {
            const res = await startMultiModel({ run_ids: selected });
            setJob({ status: "running", taskId: res.task_id });
        } catch (e) {
            setJob({ status: "error", message: e instanceof Error ? e.message : String(e) });
        }
    }, [selected]);

    const report = job.status === "ready" ? job.report : null;
    const beforeEntries = useMemo(() => (report ? Object.entries(report.metrics_before) : []), [report]);

    return (
        <>
            <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", zIndex: 59 }} aria-hidden="true" />
            <aside style={PANEL_STYLE} role="dialog" aria-label="多模型综合分析">
                <div style={{ padding: "12px 16px", borderBottom: "1px solid #262b36", display: "flex", alignItems: "center", gap: 12, background: "#12151b" }}>
                    <span style={{ fontSize: 15, fontWeight: 600 }}>🧮 多模型综合分析</span>
                    <div style={{ flex: 1 }} />
                    <button onClick={onClose} style={{ ...BUTTON_BASE, background: "#1f2937", color: "#e6edf3" }} title="关闭（Esc）">关闭</button>
                </div>

                <div style={{ padding: "12px 16px", borderBottom: "1px solid #262b36" }}>
                    <div style={{ color: "#6b7280", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 8 }}>
                        选择评估运行（≥2，需同一数据集同一划分）
                    </div>
                    {loading ? <div style={{ color: "#9ca3af" }}>加载运行记录…</div> : null}
                    {loadError ? <div style={{ color: "#f87171" }}>{loadError}</div> : null}
                    {!loading && runs.length === 0 ? (
                        <div style={{ color: "#6b7280" }}>暂无评估运行记录（run_type=eval/baseline）。</div>
                    ) : null}
                    <div style={{ maxHeight: 200, overflowY: "auto", display: "grid", gap: 4 }}>
                        {runs.map(r => (
                            <label key={r.run_id} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", padding: "3px 0" }}>
                                <input type="checkbox" checked={selected.includes(r.run_id)} onChange={() => toggle(r.run_id)} style={{ accentColor: "#1d4ed8" }} />
                                <span style={{ color: "#cbd5e1" }}>{r.run_type}</span>
                                <span style={{ color: "#6b7280", fontSize: 11 }}>{r.run_id.slice(0, 12)}</span>
                                <span style={{ color: "#4b5563", fontSize: 11, marginLeft: "auto" }}>{parseWindow(r).started}</span>
                            </label>
                        ))}
                    </div>
                    <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
                        <button onClick={() => void handleStart()} disabled={job.status === "running" || selected.length < 2}
                                style={{ ...BUTTON_BASE, background: "#1d4ed8", color: "#fff", border: "1px solid #2563eb", cursor: selected.length < 2 ? "not-allowed" : "pointer" }}>
                            {job.status === "running" ? "分析中…" : `开始分析（已选 ${selected.length}）`}
                        </button>
                        <span style={{ color: "#6b7280", fontSize: 12 }}>融合方式按任务类型自动选择（分类=投票）</span>
                    </div>
                    {job.status === "error" ? <div style={{ color: "#f87171", marginTop: 8 }}>{job.message}</div> : null}
                </div>

                <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: 16 }}>
                    {!report ? (
                        <div style={{ color: "#6b7280", lineHeight: 1.7 }}>
                            勾选两个及以上评估运行后点击「开始分析」，产出同口径综合分析报告，
                            结论以 fusion_insight 蒸馏入库（待确认）。
                        </div>
                    ) : (
                        <div style={{ display: "grid", gap: 14 }}>
                            <div>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>概览</div>
                                <div style={{ color: "#cbd5e1", lineHeight: 1.7 }}>
                                    模型：{report.models.map(m => m.name).join("、")}；
                                    共同样本 {report.n_common_samples} 个；
                                    一致 {report.consistent.length}、分歧 {report.disagreements.length}、
                                    不可比 {report.incomparable.length}；
                                    融合方式 {report.fusion}。
                                </div>
                            </div>

                            <div>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>融合前后指标（accuracy）</div>
                                <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
                                    <tbody>
                                        {beforeEntries.map(([name, v]) => (
                                            <tr key={name}>
                                                <td style={{ color: "#94a3b8", padding: "2px 0" }}>{name}</td>
                                                <td style={{ textAlign: "right", fontFamily: "monospace" }}>{v.accuracy === null ? "—" : v.accuracy.toFixed(3)}</td>
                                            </tr>
                                        ))}
                                        <tr>
                                            <td style={{ color: "#7dd3fc", padding: "2px 0" }}>融合（{report.metrics_after.fusion}）</td>
                                            <td style={{ textAlign: "right", fontFamily: "monospace", color: "#7dd3fc" }}>
                                                {report.metrics_after.accuracy === null ? "—" : report.metrics_after.accuracy.toFixed(3)}
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>

                            <div>
                                <div style={{ fontWeight: 700, marginBottom: 6 }}>分歧归因</div>
                                {report.attribution?.summary ? (
                                    <div style={{ color: "#cbd5e1", lineHeight: 1.7 }}>
                                        {report.attribution.summary}
                                        {report.attribution.reasons?.length ? (
                                            <ul style={{ margin: "6px 0 0 18px", padding: 0 }}>
                                                {report.attribution.reasons.map((r, i) => (
                                                    <li key={i} style={{ color: "#9ca3af" }}>[{r.cause}] {r.explanation}</li>
                                                ))}
                                            </ul>
                                        ) : null}
                                    </div>
                                ) : (
                                    <div style={{ color: "#6b7280" }}>
                                        {report.attribution_error ? `归因未产出：${report.attribution_error}` : "无分歧样本或未产出归因。"}
                                    </div>
                                )}
                            </div>

                            {report.knowledge_id ? (
                                <div style={{ color: "#7dd3fc", fontSize: 12 }}>
                                    已起草蒸馏知识 fusion_insight：<code>{report.knowledge_id}</code>（在「知识库 → 草稿」确认入库）
                                </div>
                            ) : null}
                        </div>
                    )}
                </div>
            </aside>
        </>
    );
}
