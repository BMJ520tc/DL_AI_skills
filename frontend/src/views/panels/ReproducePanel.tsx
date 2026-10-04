// views/panels/ReproducePanel.tsx — 三并列入口「先复现」（需求四，阶段4 4a）。
// 论文选择 → 解析（PDF→markdown，含保真核对）→ 抽取实验条目 → 逐条确认/编辑（4.2 人工闸门）
// → 复现执行 → 可信度结论（确认或修改）。
// 全部驱动既有后端接口（模块二 4.1~4.4），任务轮询复用 useTaskPolling。

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import {
    confirmPaperItem,
    getPaperDetail,
    listKnowledge,
    postConclusion,
    postExtractItems,
    postParsePaper,
    postReproduce,
    putPaperConclusion,
    putPaperItem,
    type PaperDetail,
    type PaperItemEdit,
    type Task,
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

/** JSON 文本/对象 → 对象；解析不出返回 null（不猜）。 */
function asObject(raw: unknown): Record<string, unknown> | null {
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

function asObjectList(raw: unknown): Array<Record<string, unknown>> {
    if (Array.isArray(raw)) return raw.filter(v => v && typeof v === "object") as Array<Record<string, unknown>>;
    if (typeof raw === "string" && raw.trim()) {
        try {
            const parsed = JSON.parse(raw) as unknown;
            return Array.isArray(parsed) ? (parsed.filter(v => v && typeof v === "object") as Array<Record<string, unknown>>) : [];
        } catch {
            return [];
        }
    }
    return [];
}

/** 数组字段（元素可能是数字/字符串/对象）→ 可读文本行。 */
function asTextLines(raw: unknown): string[] {
    if (!Array.isArray(raw)) return [];
    return raw.map(v => {
        if (v && typeof v === "object") {
            const o = v as Record<string, unknown>;
            return String(o.reason ?? o.detail ?? o.page ?? JSON.stringify(o));
        }
        return String(v);
    });
}

/** 任务进度（JSON 文本）→ 对象；拿不到返回 null。 */function parseProgress(raw: string | null): Record<string, unknown> | null {
    if (!raw) return null;
    try {
        const parsed = JSON.parse(raw) as unknown;
        return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : null;
    } catch {
        return null;
    }
}

/** 任意值 → 单行可读文本（超参数/对比基线等 JSON 字段的只读摘要）。 */
function brief(raw: unknown, limit = 80): string {
    if (raw === null || raw === undefined || raw === "") return "—";
    let text: string;
    if (typeof raw === "string") {
        const obj = asObject(raw);
        const list = asObjectList(raw);
        text = obj ? JSON.stringify(obj) : list.length ? JSON.stringify(list) : raw;
    } else {
        text = JSON.stringify(raw);
    }
    return text.length > limit ? `${text.slice(0, limit)}…` : text;
}

/** 编辑框初值：JSON 字段给可编辑的 JSON 文本，其余给字符串。 */
function draftText(raw: unknown): string {
    if (raw === null || raw === undefined) return "";
    if (typeof raw === "string") {
        const obj = asObject(raw);
        const list = asObjectList(raw);
        if (obj) return JSON.stringify(obj);
        if (list.length) return JSON.stringify(list);
        return raw;
    }
    return JSON.stringify(raw);
}

export default function ReproducePanel({ projectId }: ReproducePanelProps) {
    const [papers, setPapers] = useState<Array<Record<string, unknown>>>([]);
    const [paperId, setPaperId] = useState("");
    const [detail, setDetail] = useState<PaperDetail | null>(null);
    const [task, setTask] = useState<PanelTask | null>(null);
    const [taskInfo, setTaskInfo] = useState<string | null>(null);
    const [banner, setBanner] = useState<string | null>(null);
    const [flash, setFlash] = useState<string | null>(null);
    // 解析任务进度（4.1）：markdown 路径 / 章节数 / 保真核对 fidelity —— 保留最近一次，便于复核
    const [parseInfo, setParseInfo] = useState<Record<string, unknown> | null>(null);
    // 4.2 条目编辑
    const [editingItemId, setEditingItemId] = useState<string | null>(null);
    const [itemDraft, setItemDraft] = useState<Record<string, string>>({});
    const [itemError, setItemError] = useState<string | null>(null);
    const [savingItem, setSavingItem] = useState(false);
    // 4.4 结论确认/修改
    const [verdictDraft, setVerdictDraft] = useState("");
    const [summaryDraft, setSummaryDraft] = useState("");
    const [savingConclusion, setSavingConclusion] = useState(false);

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
        onProgress: (t: Task) => {
            setTaskInfo(`${t.task_type}：${t.status}${t.progress ? ` · ${t.progress.slice(0, 120)}` : ""}`);
            if (t.task_type === "pdf_parse") {
                const parsed = parseProgress(t.progress);
                if (parsed) setParseInfo(parsed);
            }
        },
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
    const paper = detail?.paper ?? null;
    const sectionIndex = asObject(paper?.section_index);
    const parsed = !!paper && (paper.status === "parsed" || paper.status === "extracted"
        || paper.status === "reproduced" || paper.status === "concluded");

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

    // ---------------- 4.2 条目编辑 ----------------
    const startEdit = (item: Record<string, unknown>) => {
        setEditingItemId(String(item.item_id));
        setItemError(null);
        setItemDraft({
            section_ref: draftText(item.section_ref),
            dataset_name: draftText(item.dataset_name),
            split_method: draftText(item.split_method),
            metric_name: draftText(item.metric_name),
            metric_value_reported: draftText(item.metric_value_reported),
            metric_unit: draftText(item.metric_unit),
            hyperparams: draftText(item.hyperparams),
            baselines: draftText(item.baselines),
        });
    };

    const saveEdit = async () => {
        if (!editingItemId) return;
        const body: PaperItemEdit = {
            section_ref: itemDraft.section_ref.trim() || null,
            dataset_name: itemDraft.dataset_name.trim() || null,
            split_method: itemDraft.split_method.trim() || null,
            metric_name: itemDraft.metric_name.trim() || null,
            metric_value_reported: itemDraft.metric_value_reported.trim() || null,
            metric_unit: itemDraft.metric_unit.trim() || null,
        };
        // JSON 字段就地校验：不合法的输入不发请求（后端 422 不会给出字段级定位）
        const hpRaw = itemDraft.hyperparams.trim();
        if (hpRaw) {
            const hp = asObject(hpRaw);
            if (!hp) {
                setItemError("超参数必须是 JSON 对象，如 {\"lr\": 0.001}");
                return;
            }
            body.hyperparams = hp;
        } else {
            body.hyperparams = null;
        }
        const blRaw = itemDraft.baselines.trim();
        if (blRaw) {
            let bl: unknown;
            try {
                bl = JSON.parse(blRaw);
            } catch {
                setItemError("对比基线必须是合法 JSON 数组，如 [\"ResNet-50\"]");
                return;
            }
            if (!Array.isArray(bl)) {
                setItemError("对比基线必须是 JSON 数组，如 [\"ResNet-50\"]");
                return;
            }
            body.baselines = bl;
        } else {
            body.baselines = null;
        }
        setItemError(null);
        setSavingItem(true);
        try {
            await putPaperItem(paperId, editingItemId, body);
            await refreshDetail(paperId);
            setEditingItemId(null);
            setFlash("条目已保存（编辑后仍是原确认状态；如需重新确认请用「确认」）");
        } catch (e) {
            setBanner(e instanceof Error ? e.message : String(e));
        } finally {
            setSavingItem(false);
        }
    };

    // ---------------- 4.4 结论确认/修改 ----------------
    useEffect(() => {
        if (!conclusion) return;
        setVerdictDraft(String(conclusion.overall_verdict ?? ""));
        setSummaryDraft(String(conclusion.summary ?? ""));
    }, [conclusion]);

    const saveConclusion = async () => {
        setSavingConclusion(true);
        try {
            await putPaperConclusion(paperId, {
                overall_verdict: verdictDraft.trim() || undefined,
                summary: summaryDraft.trim() || undefined,
            });
            await refreshDetail(paperId);
            setFlash("可信度结论已确认/修改");
        } catch (e) {
            setBanner(e instanceof Error ? e.message : String(e));
        } finally {
            setSavingConclusion(false);
        }
    };

    const fidelity = asObject(parseInfo?.fidelity);
    const sections = asObjectList(sectionIndex?.sections);
    const tables = asObjectList(sectionIndex?.tables);
    const equations = asObjectList(sectionIndex?.equations);
    const captions = asObjectList(sectionIndex?.captions);

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

            {!paperId && <div style={{ color: "#64748b", marginTop: 10 }}>选择论文后开始：解析 PDF → 抽取实验条目 → 逐条确认 → 复现 → 可信度结论。</div>}

            {paperId && (
                <>
                    {/* 操作按钮 */}
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                        <button
                            style={btnStyle}
                            disabled={busy}
                            onClick={() => runTask("parse", () => postParsePaper(paperId), refreshAfter)}
                            title="规则解析 PDF → markdown（+ agent 对照修正），并做固定代码保真核对"
                        >
                            ⓪ 解析论文（PDF→markdown）
                        </button>
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
                        <span style={{ alignSelf: "center", fontSize: 11, color: parsed ? "#4ade80" : "#d97706" }}>
                            {parsed ? "已解析" : "未解析（直接抽取会报「论文尚未转 markdown」）"}
                        </span>
                    </div>

                    {/* 4.1 解析与保真核对结果 */}
                    {(parseInfo || parsed) && (
                        <div style={{ ...infoBanner, background: "#0f172a", borderColor: "#334155", color: "#cbd5e1", marginTop: 10 }}>
                            <div style={{ fontWeight: 700, marginBottom: 4 }}>解析与保真核对</div>
                            {parseInfo ? (
                                <div style={{ fontFamily: "monospace", fontSize: 11, whiteSpace: "pre-wrap" }}>
                                    markdown：{String(parseInfo.markdown_path ?? "—")}
                                    {"\n"}章节 {String(parseInfo.sections ?? "?")} 个 · 页数 {String(parseInfo.n_pages ?? "?")} · 质量 {String(parseInfo.quality ?? "?")}
                                    {"\n"}agent 对照修正：{String(parseInfo.agent_fix ?? "—")} · markdown {parseInfo.markdown_changed ? "有改动" : "未改动"} · 索引 {parseInfo.index_rebuilt ? "已重建" : "保留原索引"}
                                    {parseInfo.hint ? `\n提示：${String(parseInfo.hint)}` : ""}
                                    {fidelity
                                        ? `\n保真核对：${fidelity.ok ? "通过" : "未通过"}${fidelity.coverage_mean !== undefined ? ` · 按页文本覆盖率均值 ${String(fidelity.coverage_mean)}` : ""}${fidelity.error ? ` · ${String(fidelity.error)}` : ""}`
                                        : "\n保真核对：本次任务进度未返回 fidelity 字段（旧后端或未执行核对）"}
                                </div>
                            ) : (
                                <div style={{ fontSize: 11, color: "#94a3b8" }}>
                                    论文记录已是已解析状态；解析进度详情（含保真核对）只在本次会话点过「⓪ 解析论文」后可见。
                                </div>
                            )}
                            {fidelity && asTextLines(fidelity.pages_below_threshold).length > 0 && (
                                <div style={{ color: "#fca5a5", fontSize: 11, marginTop: 4 }}>
                                    覆盖率低于阈值的页：{asTextLines(fidelity.pages_below_threshold).join("、")}
                                </div>
                            )}
                            {fidelity && asTextLines(fidelity.gaps).length > 0 && (
                                <div style={{ color: "#fca5a5", fontSize: 11, marginTop: 4 }}>
                                    缺口：{asTextLines(fidelity.gaps).slice(0, 5).join("；")}
                                </div>
                            )}
                            {parseInfo && Array.isArray(parseInfo.log) && (
                                <details style={{ marginTop: 6 }}>
                                    <summary style={{ cursor: "pointer", color: "#94a3b8" }}>解析日志（{parseInfo.log.length} 行）</summary>
                                    <pre style={{ margin: "4px 0 0", fontSize: 11, whiteSpace: "pre-wrap" }}>
                                        {(parseInfo.log as unknown[]).map(l => String(l)).join("\n")}
                                    </pre>
                                </details>
                            )}
                        </div>
                    )}

                    {/* 4.1 章节/表格/公式/图注位置索引（section_index 随论文详情返回） */}
                    <details style={{ marginTop: 10, border: "1px solid #1f2937", borderRadius: 6, padding: "4px 8px" }}>
                        <summary style={{ cursor: "pointer", fontWeight: 700 }}>
                            章节索引（章节 {sections.length} · 表格 {tables.length} · 公式 {equations.length} · 图注 {captions.length}）
                        </summary>
                        {!sectionIndex && <div style={{ color: "#64748b", fontSize: 11, marginTop: 4 }}>尚无 section_index（先点「⓪ 解析论文」）。</div>}
                        {sectionIndex && (
                            <div style={{ marginTop: 4 }}>
                                <div style={{ fontWeight: 700, color: "#cbd5e1" }}>章节</div>
                                {sections.length === 0 ? <div style={{ color: "#64748b" }}>（无）</div> : sections.map((s, i) => (
                                    <div key={i} style={{ paddingLeft: (Number(s.level ?? 1) - 1) * 12, fontFamily: "monospace" }}>
                                        {String(s.title ?? "?")}
                                        <span style={{ color: "#64748b" }}> · p{String(s.page ?? "?")} · L{String(s.line ?? "?")}</span>
                                    </div>
                                ))}
                                <div style={{ fontWeight: 700, color: "#cbd5e1", marginTop: 6 }}>表格</div>
                                {tables.length === 0 ? <div style={{ color: "#64748b" }}>（无）</div> : tables.map((t, i) => (
                                    <div key={i} style={{ fontFamily: "monospace" }}>
                                        #{String(t.index ?? i)} {String(t.caption ?? "(无标题)")}
                                        <span style={{ color: "#64748b" }}> · p{String(t.page ?? "?")} · L{String(t.line ?? "?")}</span>
                                    </div>
                                ))}
                                <div style={{ fontWeight: 700, color: "#cbd5e1", marginTop: 6 }}>公式</div>
                                {equations.length === 0 ? <div style={{ color: "#64748b" }}>（无）</div> : equations.map((e, i) => (
                                    <div key={i} style={{ fontFamily: "monospace" }}>
                                        {String(e.text ?? "").slice(0, 80)}
                                        <span style={{ color: "#64748b" }}> · p{String(e.page ?? "?")} · L{String(e.line ?? "?")}</span>
                                    </div>
                                ))}
                                <div style={{ fontWeight: 700, color: "#cbd5e1", marginTop: 6 }}>图注</div>
                                {captions.length === 0 ? <div style={{ color: "#64748b" }}>（无）</div> : captions.map((c, i) => (
                                    <div key={i} style={{ fontFamily: "monospace" }}>
                                        [{String(c.kind ?? "?")}] {String(c.text ?? "").slice(0, 80)}
                                        <span style={{ color: "#64748b" }}> · p{String(c.page ?? "?")} · L{String(c.line ?? "?")}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </details>

                    {/* 实验条目（4.2 确认闸门 + 五要素编辑） */}
                    <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>实验条目</div>
                    {items.length === 0 ? (
                        <div style={{ color: "#64748b", fontSize: 11 }}>尚无条目（点「① 抽取实验条目」）。</div>
                    ) : (
                        <table style={tableStyle}>
                            <thead>
                                <tr>
                                    <th style={thStyle}>数据集</th>
                                    <th style={thStyle}>划分方式</th>
                                    <th style={thStyle}>指标</th>
                                    <th style={thStyle}>报告值</th>
                                    <th style={thStyle}>超参数</th>
                                    <th style={thStyle}>对比基线</th>
                                    <th style={thStyle}>状态</th>
                                    <th style={thStyle}>操作</th>
                                </tr>
                            </thead>
                            <tbody>
                                {items.map(item => {
                                    const id = String(item.item_id);
                                    const editing = editingItemId === id;
                                    const set = (key: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
                                        setItemDraft(d => ({ ...d, [key]: e.target.value }));
                                    const cellInput: CSSProperties = { ...draftInputStyle, width: 110 };
                                    return (
                                        <tr key={id}>
                                            <td style={tdStyle}>
                                                {editing
                                                    ? <input style={cellInput} value={itemDraft.dataset_name ?? ""} onChange={set("dataset_name")} />
                                                    : String(item.dataset_name ?? "—")}
                                            </td>
                                            <td style={tdStyle}>
                                                {editing
                                                    ? <input style={cellInput} value={itemDraft.split_method ?? ""} onChange={set("split_method")} />
                                                    : String(item.split_method ?? "—")}
                                            </td>
                                            <td style={tdStyle}>
                                                {editing ? (
                                                    <div style={{ display: "flex", gap: 4 }}>
                                                        <input style={{ ...cellInput, width: 84 }} value={itemDraft.metric_name ?? ""} onChange={set("metric_name")} />
                                                        <input style={{ ...cellInput, width: 46 }} value={itemDraft.metric_unit ?? ""} onChange={set("metric_unit")} placeholder="单位" />
                                                    </div>
                                                ) : (
                                                    <>{String(item.metric_name ?? "?")}{item.metric_unit ? `（${String(item.metric_unit)}）` : ""}</>
                                                )}
                                            </td>
                                            <td style={tdStyle}>
                                                {editing
                                                    ? <input style={cellInput} value={itemDraft.metric_value_reported ?? ""} onChange={set("metric_value_reported")} />
                                                    : String(item.metric_value_reported ?? "—")}
                                            </td>
                                            <td style={{ ...tdStyle, maxWidth: 200 }}>
                                                {editing
                                                    ? <input style={{ ...cellInput, width: 190 }} value={itemDraft.hyperparams ?? ""} onChange={set("hyperparams")} title='JSON 对象，如 {"lr": 0.001}' />
                                                    : <span style={{ fontFamily: "monospace" }}>{brief(item.hyperparams)}</span>}
                                            </td>
                                            <td style={{ ...tdStyle, maxWidth: 200 }}>
                                                {editing
                                                    ? <input style={{ ...cellInput, width: 190 }} value={itemDraft.baselines ?? ""} onChange={set("baselines")} title='JSON 数组，如 ["ResNet-50"]' />
                                                    : <span style={{ fontFamily: "monospace" }}>{brief(item.baselines)}</span>}
                                            </td>
                                            <td style={tdStyle}>
                                                {item.status === "confirmed" ? <span style={{ color: "#16a34a" }}>已确认</span> : <span style={{ color: "#d97706" }}>待确认</span>}
                                                {editing && (
                                                    <div style={{ marginTop: 4 }}>
                                                        <div style={{ color: "#64748b", fontSize: 10 }}>原文位置</div>
                                                        <input style={{ ...cellInput, width: 70 }} value={itemDraft.section_ref ?? ""} onChange={set("section_ref")} />
                                                    </div>
                                                )}
                                            </td>
                                            <td style={tdStyle}>
                                                {editing ? (
                                                    <div style={{ display: "flex", gap: 6 }}>
                                                        <button style={btnStyle} disabled={savingItem} onClick={() => void saveEdit()}>
                                                            {savingItem ? "保存中…" : "保存"}
                                                        </button>
                                                        <button
                                                            style={{ ...btnStyle, background: "#334155" }}
                                                            onClick={() => { setEditingItemId(null); setItemError(null); }}
                                                        >
                                                            取消
                                                        </button>
                                                    </div>
                                                ) : (
                                                    <div style={{ display: "flex", gap: 6 }}>
                                                        <button style={{ ...btnStyle, background: "#334155" }} onClick={() => startEdit(item)}>编辑</button>
                                                        {item.status !== "confirmed" && (
                                                            <button style={btnStyle} disabled={busy} onClick={() => void confirmItem(id)}>
                                                                确认
                                                            </button>
                                                        )}
                                                    </div>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    )}
                    {itemError && <div style={{ color: "#fca5a5", fontSize: 11, marginTop: 4 }}>{itemError}</div>}

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

                    {/* 可信度结论（4.4 确认/修改） */}
                    <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>可信度结论</div>
                    {conclusion ? (
                        <div style={{ ...infoBanner, background: "#0f2d1f", borderColor: "#16a34a" }}>
                            <div>
                                总体判定：<b>{String(conclusion.overall_verdict ?? "—")}</b>
                            </div>
                            {typeof conclusion.summary === "string" && conclusion.summary && (
                                <div style={{ marginTop: 4, color: "#94a3b8", fontSize: 11 }}>{conclusion.summary}</div>
                            )}
                            <div style={{ marginTop: 8, borderTop: "1px solid #16a34a", paddingTop: 8 }}>
                                <div style={{ fontSize: 11, color: "#bbf7d0", marginBottom: 4 }}>
                                    确认或修改（PUT /api/papers/{paperId}/conclusion；留空表示保留原值）
                                </div>
                                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                                    <label style={{ fontSize: 11, color: "#94a3b8" }}>
                                        总体判定：
                                        <input
                                            style={{ ...draftInputStyle, width: 160, marginLeft: 4 }}
                                            value={verdictDraft}
                                            onChange={e => setVerdictDraft(e.target.value)}
                                            placeholder="如 可信 / 部分可信 / 不可信"
                                        />
                                    </label>
                                    <input
                                        style={{ ...draftInputStyle, flex: 1, minWidth: 240 }}
                                        value={summaryDraft}
                                        onChange={e => setSummaryDraft(e.target.value)}
                                        placeholder="结论文本摘要"
                                    />
                                    <button style={btnStyle} disabled={savingConclusion} onClick={() => void saveConclusion()}>
                                        {savingConclusion ? "保存中…" : "保存结论"}
                                    </button>
                                </div>
                            </div>
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
const draftInputStyle: CSSProperties = {
    border: "1px solid #334155",
    background: "#0f172a",
    color: "#e2e8f0",
    borderRadius: 4,
    padding: "2px 5px",
    fontSize: 11,
    fontFamily: "monospace",
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
