// views/panels/UseDatasetPanel.tsx — 三并列入口「先使用」（需求四，阶段4 4a）。
// 自带数据基准 → 跨数据集对齐（5.3 确认闸门）→ 结果对比（5.4 使用建议确认）→ 三张图（5.5）。
// 全部驱动既有后端接口（模块三 5.2~5.5），任务轮询复用 useTaskPolling。

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import {
    confirmAlignment,
    confirmKnowledgeItem,
    figureUrl,
    getBaseline,
    listKnowledge,
    postAlign,
    postBaseline,
    postCompare,
    postVisualize,
} from "../../api/client";
import { useTaskPolling } from "../../hooks/useTaskPolling";

export type UseDatasetPanelProps = {
    projectId: string;
};

interface PanelTask {
    id: string;
    kind: string;
    after: () => Promise<void>;
}

interface DatasetRow extends Record<string, unknown> {
    dataset_id: string;
    name?: string | null;
    source?: string | null;
    alignment?: string | null;
}

interface ChartResult {
    chart: string;
    html: string | null;
    degraded: boolean;
}

/** 解析数据集登记表里的对齐 JSON（可能为文本或已解析对象）。 */
function parseAlignment(raw: unknown): Record<string, unknown> | null {
    if (typeof raw === "string") {
        try {
            return JSON.parse(raw) as Record<string, unknown>;
        } catch {
            return null;
        }
    }
    return typeof raw === "object" && raw !== null ? (raw as Record<string, unknown>) : null;
}

/** 本项目待确认对齐的数据集（5.3 闸门：已对齐未确认才需要确认）。 */
function pendingAlignments(datasets: DatasetRow[], projectId: string): DatasetRow[] {
    return datasets.filter(ds => {
        const alignment = parseAlignment(ds.alignment);
        if (!alignment) return false;
        const projects = alignment.aligned_projects;
        const alignedHere = Array.isArray(projects) && projects.includes(projectId);
        return alignedHere && alignment.status !== "confirmed";
    });
}

export default function UseDatasetPanel({ projectId }: UseDatasetPanelProps) {
    const [datasets, setDatasets] = useState<DatasetRow[]>([]);
    const [selectedDatasetId, setSelectedDatasetId] = useState("");
    const [baseline, setBaseline] = useState<Record<string, unknown> | null>(null);
    const [guidance, setGuidance] = useState<Array<Record<string, unknown>>>([]);
    const [charts, setCharts] = useState<ChartResult[]>([]);
    const [task, setTask] = useState<PanelTask | null>(null);
    const [taskInfo, setTaskInfo] = useState<string | null>(null);
    const [banner, setBanner] = useState<string | null>(null);
    const [flash, setFlash] = useState<string | null>(null);

    const refreshDatasets = useCallback(async () => {
        try {
            const rows = (await listKnowledge("dataset")) as DatasetRow[];
            setDatasets(rows);
        } catch (e) {
            setBanner(`数据集列表读取失败：${e instanceof Error ? e.message : String(e)}`);
        }
    }, []);

    useEffect(() => {
        void (async () => {
            try {
                const rows = (await listKnowledge("dataset")) as DatasetRow[];
                setDatasets(rows);
            } catch (e) {
                setBanner(`数据集列表读取失败：${e instanceof Error ? e.message : String(e)}`);
            }
        })();
    }, []);

    const refreshBaseline = useCallback(async () => {
        try {
            setBaseline(await getBaseline(projectId));
        } catch {
            // 404 = 尚未跑过基准，属正常初态
            setBaseline(null);
        }
    }, [projectId]);

    useEffect(() => {
        void (async () => {
            try {
                setBaseline(await getBaseline(projectId));
            } catch {
                // 404 = 尚未跑过基准，属正常初态
                setBaseline(null);
            }
        })();
    }, [projectId]);

    const refreshGuidance = useCallback(async () => {
        try {
            const items = await listKnowledge("knowledge");
            setGuidance(items.filter(k => k.type === "usage_guidance"));
        } catch (e) {
            console.warn("使用建议列表读取失败", e);
        }
    }, []);

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

    const afterBaseline = async () => {
        await refreshBaseline();
        setFlash("基准运行完成");
    };
    const afterAlign = async () => {
        await refreshDatasets();
        setFlash("对齐已起草（确认后参与对比）");
    };
    const afterCompare = async () => {
        await refreshGuidance();
        setFlash("对比完成，使用建议已生成（待确认）");
    };

    const confirmOneAlignment = async (datasetId: string) => {
        try {
            await confirmAlignment(datasetId);
            await refreshDatasets();
            setFlash("对齐已确认");
        } catch (e) {
            setBanner(e instanceof Error ? e.message : String(e));
        }
    };

    const confirmOneKnowledge = async (knowledgeId: string) => {
        try {
            await confirmKnowledgeItem(knowledgeId);
            await refreshGuidance();
            setFlash("使用建议已确认");
        } catch (e) {
            setBanner(e instanceof Error ? e.message : String(e));
        }
    };

    const makeChart = async (chart: string) => {
        try {
            const result = await postVisualize(projectId, chart);
            setCharts(prev => [...prev.filter(c => c.chart !== chart), { chart, html: result.html, degraded: !!result.degraded }]);
        } catch (e) {
            setBanner(`图表 ${chart} 生成失败：${e instanceof Error ? e.message : String(e)}`);
        }
    };

    const pending = pendingAlignments(datasets, projectId);
    const metrics = typeof baseline?.metrics === "string" ? baseline.metrics : null;

    return (
        <div style={{ fontSize: 12 }}>
            {/* 数据集选择 */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <label style={{ color: "#94a3b8", fontSize: 11 }}>
                    目标数据集：
                    <select
                        style={{ ...selectStyle, marginLeft: 6, maxWidth: 300 }}
                        value={selectedDatasetId}
                        onChange={e => setSelectedDatasetId(e.target.value)}
                        disabled={busy}
                    >
                        <option value="">（选择已登记数据集）</option>
                        {datasets.map(ds => (
                            <option key={ds.dataset_id} value={ds.dataset_id}>
                                {ds.name ?? ds.dataset_id}（{ds.source ?? "?"}）
                            </option>
                        ))}
                    </select>
                </label>
                {datasets.length === 0 && <span style={{ color: "#64748b", fontSize: 11 }}>数据集登记表为空（可先经检索下载数据集）</span>}
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

            {/* 操作按钮 */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                <button style={btnStyle} disabled={busy} onClick={() => runTask("baseline", () => postBaseline(projectId), afterBaseline)}>
                    ① 自带数据基准运行
                </button>
                <button
                    style={btnStyle}
                    disabled={busy || !selectedDatasetId}
                    onClick={() => runTask("align", () => postAlign(projectId, selectedDatasetId), afterAlign)}
                    title={!selectedDatasetId ? "需先选择目标数据集" : undefined}
                >
                    ② 对齐所选数据集
                </button>
                <button
                    style={btnStyle}
                    disabled={busy || pending.length > 0}
                    onClick={() => runTask("compare", () => postCompare(projectId), afterCompare)}
                    title={pending.length > 0 ? "有对齐待确认（5.3 闸门）" : undefined}
                >
                    ③ 结果对比与使用建议
                </button>
                <button style={btnStyle} disabled={busy} onClick={() => void makeChart("performance")}>
                    ④ 性能对比图
                </button>
                <button style={btnStyle} disabled={busy} onClick={() => void makeChart("error_dist")}>
                    ④ 误差分布图
                </button>
                <button style={btnStyle} disabled={busy} onClick={() => void makeChart("cases")}>
                    ④ 典型案例图
                </button>
            </div>

            {/* 基准指标 */}
            <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>基准运行</div>
            {baseline ? (
                <div style={{ ...infoBanner, background: "#0f2d1f", borderColor: "#16a34a" }}>
                    <div>
                        状态：{String(baseline.status ?? "?")} · {String(baseline.started_at ?? "")}
                    </div>
                    {metrics && <pre style={{ margin: "6px 0 0", fontSize: 11, color: "#cbd5e1", whiteSpace: "pre-wrap" }}>{metrics}</pre>}
                </div>
            ) : (
                <div style={{ color: "#64748b", fontSize: 11 }}>尚未运行（点「① 自带数据基准运行」）。</div>
            )}

            {/* 对齐确认闸门 */}
            <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>对齐确认（5.3 闸门）</div>
            {pending.length === 0 ? (
                <div style={{ color: "#64748b", fontSize: 11 }}>无待确认对齐。</div>
            ) : (
                pending.map(ds => (
                    <div key={ds.dataset_id} style={{ ...infoBanner, background: "#2d1f0f", borderColor: "#d97706" }}>
                        <span>{ds.name ?? ds.dataset_id} 的对齐待确认</span>
                        <button style={{ ...btnStyle, marginLeft: 10 }} disabled={busy} onClick={() => void confirmOneAlignment(ds.dataset_id)}>
                            确认对齐
                        </button>
                    </div>
                ))
            )}

            {/* 使用建议（5.4 确认闸门） */}
            <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>使用建议（对比产出）</div>
            {guidance.length === 0 ? (
                <div style={{ color: "#64748b", fontSize: 11 }}>尚无使用建议（点「③ 结果对比与使用建议」）。</div>
            ) : (
                guidance.map(k => (
                    <div key={String(k.knowledge_id)} style={{ ...infoBanner, background: k.status === "confirmed" ? "#0f2d1f" : "#1e293b", borderColor: k.status === "confirmed" ? "#16a34a" : "#334155" }}>
                        <div style={{ fontWeight: 700 }}>{String(k.title ?? "(未命名)")}</div>
                        <div style={{ marginTop: 4, color: "#94a3b8", fontSize: 11 }}>
                            {String(k.content ?? "").slice(0, 300)}
                            {String(k.content ?? "").length > 300 ? "…" : ""}
                        </div>
                        <div style={{ marginTop: 4, color: "#64748b", fontSize: 11 }}>
                            {k.status === "confirmed" ? "已确认" : "草稿待确认"}
                            {k.status !== "confirmed" && (
                                <button style={{ ...btnStyle, marginLeft: 10 }} disabled={busy} onClick={() => void confirmOneKnowledge(String(k.knowledge_id))}>
                                    确认
                                </button>
                            )}
                        </div>
                    </div>
                ))
            )}

            {/* 三张图 */}
            <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>可视化图</div>
            {charts.length === 0 ? (
                <div style={{ color: "#64748b", fontSize: 11 }}>尚未生成（点「④」三张图按钮，生成后在此打开）。</div>
            ) : (
                charts.map(c => (
                    <div key={c.chart} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                        <span style={{ color: "#94a3b8", minWidth: 90 }}>{c.chart}</span>
                        {c.html ? (
                            <a href={figureUrl(projectId, c.chart)} target="_blank" rel="noreferrer" style={{ color: "#2dd4bf" }}>
                                在浏览器打开{c.degraded ? "（降级产出）" : ""}
                            </a>
                        ) : (
                            <span style={{ color: "#f87171" }}>生成失败</span>
                        )}
                    </div>
                ))
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
