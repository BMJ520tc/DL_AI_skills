// views/panels/UseDatasetPanel.tsx — 三并列入口「先使用」（需求四，阶段4 4a）。
// 数据预处理（5.1）→ 公开数据检索/下载（5.3，含「自带数据是否充足」判定）→ 自带数据基准运行（5.2）
// → 跨数据集对齐（5.3 确认闸门）→ 结果对比（5.4 使用建议确认，含结构化对比表）→ 三张图（5.5）。
// 全部驱动既有后端接口（模块三 5.1~5.5），任务轮询复用 useTaskPolling。

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import {
    confirmAlignment,
    confirmKnowledgeItem,
    downloadDataset,
    figureUrl,
    getBaseline,
    listKnowledge,
    postAlign,
    postBaseline,
    postCompare,
    postPreprocess,
    postVisualize,
    searchDatasets,
    type DatasetSearchResult,
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
    local_path?: string | null;
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

/** 知识条目 structured 字段（JSON 文本或对象）→ 对象。 */
function parseStructured(raw: unknown): Record<string, unknown> | null {
    if (raw && typeof raw === "object" && !Array.isArray(raw)) return raw as Record<string, unknown>;
    if (typeof raw === "string" && raw.trim()) {
        try {
            const parsed = JSON.parse(raw) as unknown;
            return parsed && typeof parsed === "object" && !Array.isArray(parsed)
                ? (parsed as Record<string, unknown>)
                : null;
        } catch {
            return null;
        }
    }
    return null;
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

/** 5.4 对比表（compare_service._build_table 的既有结构）→ 可渲染的表格行列。 */
function comparisonTable(k: Record<string, unknown>): { columns: string[]; rows: Array<Record<string, unknown>>; note: string | null } | null {
    const structured = parseStructured(k.structured);
    const comparison = structured?.comparison;
    if (!comparison || typeof comparison !== "object" || Array.isArray(comparison)) return null;
    const table = comparison as Record<string, unknown>;
    const rows = Array.isArray(table.rows) ? (table.rows.filter(r => r && typeof r === "object") as Array<Record<string, unknown>>) : [];
    const columns = Array.isArray(table.columns) ? table.columns.map(c => String(c)) : [];
    if (!rows.length) return null;
    return {
        columns: columns.length ? columns : Object.keys(rows[0]).filter(c => c !== "metric"),
        rows,
        note: typeof table.alignment_note === "string" ? table.alignment_note : null,
    };
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

    // 5.1 数据预处理
    const [preprocessPath, setPreprocessPath] = useState("");
    const [preprocessName, setPreprocessName] = useState("");
    const [preprocessTaskType, setPreprocessTaskType] = useState("classification");

    // 5.3 公开数据检索
    const [searchTaskType, setSearchTaskType] = useState("");
    const [searchFormat, setSearchFormat] = useState("");
    const [searchQ, setSearchQ] = useState("");
    const [searchResult, setSearchResult] = useState<DatasetSearchResult | null>(null);
    const [searching, setSearching] = useState(false);
    const [downloadingId, setDownloadingId] = useState<string | null>(null);

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
        onProgress: t => setTaskInfo(`${t.task_type}：${t.status}${t.progress ? ` · ${t.progress.slice(0, 120)}` : ""}`),
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

    const afterPreprocess = async () => {
        await refreshDatasets();
        await refreshBaseline();
        setFlash("数据预处理完成：数据集已登记（基准运行与画布训练的数据集下拉现在可用）");
    };
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

    // 5.1：提交预处理（产物登记进数据集表；基准运行与画布训练都依赖它）
    const runPreprocess = () => {
        if (!preprocessPath.trim()) {
            setBanner("请填写预处理输入路径，或先在下方「公开数据检索」/数据集下拉里选一个已登记数据集自动填入");
            return;
        }
        runTask(
            "preprocess",
            () => postPreprocess({
                input_path: preprocessPath.trim(),
                dataset_name: preprocessName.trim() || null,
                task_type: preprocessTaskType.trim() || "classification",
                project_id: projectId,
            }),
            afterPreprocess,
        );
    };

    // 5.3：公开数据检索（带 project_id 时附带「自带数据是否充足」判定）
    const runSearch = async () => {
        setSearching(true);
        setBanner(null);
        try {
            setSearchResult(await searchDatasets({
                task_type: searchTaskType.trim() || undefined,
                format: searchFormat.trim() || undefined,
                q: searchQ.trim() || undefined,
                project_id: projectId,
            }));
        } catch (e) {
            setBanner(`公开数据检索失败：${e instanceof Error ? e.message : String(e)}`);
        } finally {
            setSearching(false);
        }
    };

    // 5.3：下载外部数据集并登记
    const runDownload = async (entry: Record<string, unknown>) => {
        const source = String(entry.source ?? "");
        const sourceId = String(entry.source_id ?? "");
        const name = String(entry.name ?? sourceId);
        if (!source || !sourceId) {
            setBanner("该条目缺少 source/source_id，无法下载");
            return;
        }
        setDownloadingId(`${source}:${sourceId}`);
        setBanner(null);
        try {
            const { dataset_id } = await downloadDataset({
                source, source_id: sourceId, name,
                task_type: entry.task_type ? String(entry.task_type) : undefined,
            });
            await refreshDatasets();
            setFlash(`已下载并登记数据集 ${dataset_id}`);
        } catch (e) {
            setBanner(`下载失败（${name}）：${e instanceof Error ? e.message : String(e)}`);
        } finally {
            setDownloadingId(null);
        }
    };

    const pending = pendingAlignments(datasets, projectId);
    const metrics = typeof baseline?.metrics === "string" ? baseline.metrics : null;
    // 基准运行可用性：自带数据已登记为数据集（预处理产物）才真正跑得动
    const selfDatasets = datasets.filter(d => d.source === "自带" && d.local_path);

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
                {datasets.length === 0 && <span style={{ color: "#64748b", fontSize: 11 }}>数据集登记表为空（可先做预处理或检索下载）</span>}
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

            {/* 5.1 数据预处理（基准运行的第一步：自带数据必须先经此统一化） */}
            <div style={{ border: "1px solid #1f2937", borderRadius: 8, padding: "8px 10px", marginTop: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>
                    ⓪ 数据预处理（格式识别 + 统一 schema；产物登记后基准运行/画布训练才可选到数据集）
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                    <input
                        style={{ ...draftInputStyle, flex: 1, minWidth: 280 }}
                        placeholder="输入路径（原始 CSV/Excel/图片目录/FASTA/PDB…）"
                        value={preprocessPath}
                        onChange={e => setPreprocessPath(e.target.value)}
                    />
                    <select
                        style={{ ...selectStyle, maxWidth: 240 }}
                        value=""
                        onChange={e => {
                            const ds = datasets.find(d => d.dataset_id === e.target.value);
                            if (ds?.local_path) {
                                setPreprocessPath(String(ds.local_path));
                                setPreprocessName(String(ds.name ?? ""));
                            }
                        }}
                    >
                        <option value="">（从已登记数据集填入路径）</option>
                        {datasets.filter(d => d.local_path).map(ds => (
                            <option key={ds.dataset_id} value={ds.dataset_id}>
                                {ds.name ?? ds.dataset_id}
                            </option>
                        ))}
                    </select>
                    <input
                        style={{ ...draftInputStyle, width: 130 }}
                        placeholder="数据集名（可选）"
                        value={preprocessName}
                        onChange={e => setPreprocessName(e.target.value)}
                    />
                    <input
                        style={{ ...draftInputStyle, width: 120 }}
                        placeholder="任务类型"
                        value={preprocessTaskType}
                        onChange={e => setPreprocessTaskType(e.target.value)}
                    />
                    <button style={btnStyle} disabled={busy} onClick={runPreprocess}>
                        {busy ? "任务进行中…" : "运行预处理"}
                    </button>
                </div>
                <div style={{ color: "#64748b", fontSize: 11, marginTop: 4 }}>
                    带 project_id 提交：产物落本项目工作区并登记为「自带」数据集（未填数据集名时用 self）。
                </div>
            </div>

            {/* 5.3 公开数据检索与下载 */}
            <div style={{ border: "1px solid #1f2937", borderRadius: 8, padding: "8px 10px", marginTop: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>公开数据检索（5.3）</div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                    <input style={{ ...draftInputStyle, width: 120 }} placeholder="任务类型" value={searchTaskType} onChange={e => setSearchTaskType(e.target.value)} />
                    <input style={{ ...draftInputStyle, width: 100 }} placeholder="格式" value={searchFormat} onChange={e => setSearchFormat(e.target.value)} />
                    <input style={{ ...draftInputStyle, flex: 1, minWidth: 200 }} placeholder="关键词" value={searchQ} onChange={e => setSearchQ(e.target.value)} />
                    <button style={btnStyle} disabled={searching} onClick={() => void runSearch()}>
                        {searching ? "检索中…" : "检索公开数据"}
                    </button>
                </div>

                {searchResult && (
                    <div style={{ marginTop: 8 }}>
                        {searchResult.self_data && (
                            <div
                                style={{
                                    ...infoBanner,
                                    background: searchResult.self_data.sufficient ? "#0f2d1f" : "#2d1f0f",
                                    borderColor: searchResult.self_data.sufficient ? "#16a34a" : "#d97706",
                                    color: searchResult.self_data.sufficient ? "#bbf7d0" : "#fde68a",
                                }}
                            >
                                自带数据是否充足：
                                <b>{searchResult.self_data.sufficient ? "充足" : "不足"}</b>
                                （{searchResult.self_data.n_samples} 条 / 阈值 {searchResult.self_data.threshold}）
                                {searchResult.self_data.hint ? <div style={{ marginTop: 2 }}>{searchResult.self_data.hint}</div> : null}
                            </div>
                        )}
                        {searchResult.hint && <div style={{ color: "#fca5a5", marginBottom: 6 }}>{searchResult.hint}</div>}

                        <div style={{ fontWeight: 700, color: "#cbd5e1", marginBottom: 2 }}>
                            已登记数据集（{searchResult.local.length}）
                        </div>
                        {searchResult.local.length === 0 ? (
                            <div style={{ color: "#64748b", marginBottom: 6 }}>（无）</div>
                        ) : searchResult.local.map((d, i) => (
                            <div key={String(d.dataset_id ?? i)} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
                                <span style={{ color: "#e2e8f0" }}>{String(d.name ?? d.dataset_id)}</span>
                                <span style={{ color: "#64748b" }}>
                                    {String(d.source ?? "?")} · {String(d.format ?? "?")} · {String(d.task_type ?? "?")}
                                </span>
                                {d.local_path ? (
                                    <button
                                        style={{ ...btnStyle, background: "#334155" }}
                                        onClick={() => { setPreprocessPath(String(d.local_path)); setPreprocessName(String(d.name ?? "")); }}
                                    >
                                        填入预处理路径
                                    </button>
                                ) : null}
                            </div>
                        ))}

                        <div style={{ fontWeight: 700, color: "#cbd5e1", margin: "6px 0 2px" }}>
                            外部公开数据（{searchResult.external.length}）
                        </div>
                        {searchResult.external.length === 0 ? (
                            <div style={{ color: "#64748b" }}>（无：外部源不可达或无命中，检索不阻断流程）</div>
                        ) : searchResult.external.map((d, i) => {
                            const key = `${String(d.source ?? "")}:${String(d.source_id ?? i)}`;
                            return (
                                <div key={key} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
                                    <span style={{ color: "#e2e8f0" }}>{String(d.name ?? d.source_id)}</span>
                                    <span style={{ color: "#64748b" }}>{String(d.source ?? "?")}</span>
                                    <button style={btnStyle} disabled={downloadingId !== null} onClick={() => void runDownload(d)}>
                                        {downloadingId === key ? "下载中…" : "下载并登记"}
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

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
            <div style={{ fontSize: 11, color: selfDatasets.length ? "#4ade80" : "#d97706", marginBottom: 4 }}>
                自带数据登记：{selfDatasets.length
                    ? selfDatasets.map(d => String(d.name ?? d.dataset_id)).join("、")
                    : "无（先在「⓪ 数据预处理」跑一次，产物才会登记为自带数据集）"}
            </div>
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

            {/* 使用建议（5.4 确认闸门 + 结构化对比表） */}
            <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>使用建议（对比产出）</div>
            {guidance.length === 0 ? (
                <div style={{ color: "#64748b", fontSize: 11 }}>尚无使用建议（点「③ 结果对比与使用建议」）。</div>
            ) : (
                guidance.map(k => {
                    const table = comparisonTable(k);
                    return (
                        <div key={String(k.knowledge_id)} style={{ ...infoBanner, background: k.status === "confirmed" ? "#0f2d1f" : "#1e293b", borderColor: k.status === "confirmed" ? "#16a34a" : "#334155" }}>
                            <div style={{ fontWeight: 700 }}>{String(k.title ?? "(未命名)")}</div>
                            <div style={{ marginTop: 4, color: "#94a3b8", fontSize: 11 }}>
                                {String(k.content ?? "").slice(0, 300)}
                                {String(k.content ?? "").length > 300 ? "…" : ""}
                            </div>
                            {table && (
                                <div style={{ marginTop: 6, overflowX: "auto" }}>
                                    <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 3 }}>
                                        对比表（基线 vs 各数据集）{table.note ? ` · ${table.note}` : ""}
                                    </div>
                                    <table style={tableStyle}>
                                        <thead>
                                            <tr>
                                                <th style={thStyle}>metric</th>
                                                {table.columns.map(c => <th key={c} style={thStyle}>{c}</th>)}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {table.rows.map((row, i) => (
                                                <tr key={i}>
                                                    <td style={tdStyle}>{String(row.metric ?? "?")}</td>
                                                    {table.columns.map(c => (
                                                        <td key={c} style={tdStyle}>
                                                            {typeof row[c] === "number" ? (row[c] as number).toFixed(4) : row[c] === null || row[c] === undefined ? "—" : String(row[c])}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                            <div style={{ marginTop: 4, color: "#64748b", fontSize: 11 }}>
                                {k.status === "confirmed" ? "已确认" : "草稿待确认"}
                                {k.status !== "confirmed" && (
                                    <button style={{ ...btnStyle, marginLeft: 10 }} disabled={busy} onClick={() => void confirmOneKnowledge(String(k.knowledge_id))}>
                                        确认
                                    </button>
                                )}
                            </div>
                        </div>
                    );
                })
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
const draftInputStyle: CSSProperties = {
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
const tdStyle: CSSProperties = { padding: "3px 6px", borderBottom: "1px solid #1f2937", whiteSpace: "nowrap" };
