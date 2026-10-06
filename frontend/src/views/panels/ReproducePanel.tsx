// views/panels/ReproducePanel.tsx — 三并列入口「先复现」（需求四，阶段4 4a）。
// 论文选择 → 解析（PDF→markdown，含保真核对）→ 抽取实验条目 → 逐条确认/编辑（4.2 人工闸门）
// → 复现执行 → 可信度结论（确认或修改）。
// 全部驱动既有后端接口（模块二 4.1~4.4），任务轮询复用 useTaskPolling。

import { Fragment, useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import {
    confirmPaperItem,
    getPaperDetail,
    listKnowledge,
    listTasks,
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
    /** 初始选中的论文（从初始界面的「论文复现」入口带入；不传则空） */
    initialPaperId?: string;
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

/** 判定分档配色（与后端 verdict 文案对齐）。 */
function verdictColor(verdict: string | undefined): string {
    switch (verdict) {
        case "一致": return "#4ade80";
        case "近似": return "#fbbf24";
        case "不一致": return "#f87171";
        default: return "#94a3b8";
    }
}

/** ISO 时间 → 本地「月-日 时:分」。 */
function shortTime(raw: unknown): string {
    if (typeof raw !== "string" || !raw) return "—";
    const d = new Date(raw);
    if (Number.isNaN(d.getTime())) return raw.slice(0, 16).replace("T", " ");
    return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} `
        + `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** 禁用态样式：默认的 `<button disabled>` 与正常按钮外观一致，会让用户以为「按钮坏了」。 */
function dimmed(disabled: boolean): CSSProperties {
    return disabled ? { opacity: 0.45, cursor: "not-allowed" } : {};
}

/** 本面板发起的任务类型 → 面板内的操作名（用于重新接管任务）。 */
const RESUME_KINDS: Record<string, string> = {
    pdf_parse: "parse",
    extract_items: "extract",
    reproduce: "reproduce",
    conclusion: "conclusion",
};

/** 任务 params 里的 paper_id（任务 params 是 JSON 文本）。 */
function paperIdOfTask(task: Task): string | null {
    try {
        const p = JSON.parse(task.params || "{}") as Record<string, unknown>;
        return typeof p.paper_id === "string" ? p.paper_id : null;
    } catch {
        return null;
    }
}

export default function ReproducePanel({ projectId, initialPaperId }: ReproducePanelProps) {
    const [papers, setPapers] = useState<Array<Record<string, unknown>>>([]);
    const [paperId, setPaperId] = useState(initialPaperId ?? "");
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
    // 绑定记录里展开了哪个项目（看该次复现的条目）
    const [openBinding, setOpenBinding] = useState<string | null>(null);

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
        // 静默丢弃点击会被当成「按钮坏了」——明确告诉用户为什么没反应
        if (busyRef.current) {
            setFlash("已有任务在执行中，请等它结束再操作（按钮暂时不可用）");
            return;
        }
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

    // 任务状态是本组件的本地 state，离开本页即卸载 → 回来「按钮复原、进度丢失」。
    // 任务本身在库里没丢，只是没人显示它：挂载/切论文时找回本论文的活跃任务并**接管**。
    useEffect(() => {
        if (!paperId || task) return;  // 本地已有任务时不抢
        let cancelled = false;
        void (async () => {
            try {
                const rows = await listTasks("board", 100);
                const active = rows.find(
                    t => (t.status === "running" || t.status === "queued")
                        && RESUME_KINDS[t.task_type] !== undefined
                        && paperIdOfTask(t) === paperId,
                );
                if (cancelled || !active) return;
                busyRef.current = true;
                setBusy(true);
                setTask({
                    id: active.task_id,
                    kind: RESUME_KINDS[active.task_type],
                    after: () => refreshDetail(paperId),
                });
                setFlash("已接管本论文正在执行的任务（离开本页不会中断它）");
            } catch {
                /* 恢复失败不影响正常操作 */
            }
        })();
        return () => {
            cancelled = true;
        };
    }, [paperId, task, refreshDetail]);

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
    const bindings = detail?.bindings ?? [];
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
                            style={{ ...btnStyle, ...dimmed(busy) }}
                            disabled={busy}
                            onClick={() => runTask("parse", () => postParsePaper(paperId), refreshAfter)}
                            title="规则解析 PDF → markdown（+ agent 对照修正），并做固定代码保真核对"
                        >
                            ⓪ 解析论文（PDF→markdown）
                        </button>
                        <button
                            style={{ ...btnStyle, ...dimmed(busy) }}
                            disabled={busy}
                            onClick={() => runTask("extract", () => postExtractItems(paperId), refreshAfter)}
                        >
                            ① 抽取实验条目
                        </button>
                        <button
                            style={{ ...btnStyle, ...dimmed(busy || confirmedCount === 0) }}
                            disabled={busy || confirmedCount === 0}
                            onClick={() => runTask("reproduce", () => postReproduce(paperId, projectId), refreshAfter)}
                            title={busy ? "有任务在执行中" : confirmedCount === 0 ? "需先确认至少一条实验条目" : undefined}
                        >
                            ② 复现（已确认 {confirmedCount}/{items.length} 条）
                        </button>
                        <button
                            style={{ ...btnStyle, ...dimmed(busy || reproResults.length === 0) }}
                            disabled={busy || reproResults.length === 0}
                            onClick={() => runTask("conclusion", () => postConclusion(paperId), refreshAfter)}
                            title={busy ? "有任务在执行中" : reproResults.length === 0 ? "需先有复现结果" : undefined}
                        >
                            ③ 生成可信度结论
                        </button>
                        {busy ? (
                            <span style={{ alignSelf: "center", fontSize: 11, color: "#fbbf24" }}>
                                ⏳ 有任务在执行中，按钮暂不可用
                            </span>
                        ) : (
                            <span style={{ alignSelf: "center", fontSize: 11, color: parsed ? "#4ade80" : "#d97706" }}>
                                {parsed ? "已解析" : "未解析（直接抽取会报「论文尚未转 markdown」）"}
                            </span>
                        )}
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
                                        ? `\n保真核对：${fidelity.ok ? "通过" : "未通过"}${fidelity.coverage_mean !== undefined ? ` · 按页内容命中率均值 ${String(fidelity.coverage_mean)}` : ""}${fidelity.error ? ` · ${String(fidelity.error)}` : ""}`
                                        : "\n保真核对：本次任务进度未返回 fidelity 字段（旧后端或未执行核对）"}
                                </div>
                            ) : (
                                <div style={{ fontSize: 11, color: "#94a3b8" }}>
                                    论文记录已是已解析状态；解析进度详情（含保真核对）只在本次会话点过「⓪ 解析论文」后可见。
                                </div>
                            )}
                            {fidelity && asTextLines(fidelity.pages_below_threshold).length > 0 && (
                                <div style={{ color: "#fca5a5", fontSize: 11, marginTop: 4 }}>
                                    内容未命中的页：{asTextLines(fidelity.pages_below_threshold).join("、")}
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
                            {/* 定宽列：超参数/对比基线是长 JSON，不定宽会把列撑爆并与邻列重叠 */}
                            <colgroup>
                                <col style={{ width: "12%" }} />
                                <col style={{ width: "16%" }} />
                                <col style={{ width: "11%" }} />
                                <col style={{ width: "12%" }} />
                                <col style={{ width: "19%" }} />
                                <col style={{ width: "12%" }} />
                                <col style={{ width: "6%" }} />
                                <col style={{ width: "12%" }} />
                            </colgroup>
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
                                                            <button
                                                                style={{ ...btnStyle, ...dimmed(busy) }}
                                                                disabled={busy}
                                                                title={busy ? "有任务在执行中，稍后再确认" : "确认后该条目参与复现（4.2 人工闸门）"}
                                                                onClick={() => void confirmItem(id)}
                                                            >
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

                    {/* 复现板：本篇论文「报告值 vs 实测值」逐条对照 */}
                    <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>
                        复现板 <span style={{ color: "#64748b", fontSize: 11, fontWeight: 400 }}>（本论文的逐条对照）</span>
                    </div>
                    {reproResults.length === 0 ? (
                        <div style={{ color: "#64748b", fontSize: 11 }}>尚无复现结果（点「② 复现」）。</div>
                    ) : (
                        <table style={tableStyle}>
                            <colgroup>
                                <col style={{ width: "22%" }} />
                                <col style={{ width: "14%" }} />
                                <col style={{ width: "16%" }} />
                                <col style={{ width: "16%" }} />
                                <col style={{ width: "12%" }} />
                                <col style={{ width: "20%" }} />
                            </colgroup>
                            <thead>
                                <tr>
                                    <th style={thStyle}>数据集</th>
                                    <th style={thStyle}>指标</th>
                                    <th style={thStyle}>报告值</th>
                                    <th style={thStyle}>实测值</th>
                                    <th style={thStyle}>偏差</th>
                                    <th style={thStyle}>判定</th>
                                </tr>
                            </thead>
                            <tbody>
                                {reproResults.map(r => {
                                    const item = items.find(i => String(i.item_id) === String(r.item_id));
                                    const actual = r.metric_value_actual;
                                    // deviation 是**相对**误差（0.0022 = 0.22%）
                                    const dev = typeof r.deviation === "number" ? `${(r.deviation * 100).toFixed(2)}%` : "—";
                                    return (
                                        <tr key={String(r.result_id)}>
                                            <td style={tdStyle}>
                                                {String(r.dataset_name ?? item?.dataset_name ?? "—")}
                                            </td>
                                            <td style={tdStyle}>{String(r.metric_name ?? item?.metric_name ?? "?")}</td>
                                            <td style={tdStyle}>
                                                {r.metric_value_reported === null || r.metric_value_reported === undefined
                                                    ? "—" : String(r.metric_value_reported)}
                                            </td>
                                            <td style={tdStyle}>
                                                {actual === null || actual === undefined ? "—" : String(actual)}
                                            </td>
                                            <td style={tdStyle}>{dev}</td>
                                            <td style={tdStyle}>
                                                <span style={{ color: verdictColor(r.verdict as string | undefined) }}>
                                                    {String(r.verdict ?? "—")}
                                                </span>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    )}

                    {/* 绑定记录：同一篇论文可以用不同项目复现，每次绑定留痕 */}
                    <div style={{ fontWeight: 700, margin: "12px 0 4px" }}>
                        绑定记录 <span style={{ color: "#64748b", fontSize: 11, fontWeight: 400 }}>（这篇论文用过的项目/环境）</span>
                    </div>
                    {bindings.length === 0 ? (
                        <div style={{ color: "#64748b", fontSize: 11 }}>
                            尚无绑定记录——点「② 复现」后会记下「本论文 ↔ 当前项目」。
                        </div>
                    ) : (
                        <table style={tableStyle}>
                            <colgroup>
                                <col style={{ width: "26%" }} />
                                <col style={{ width: "14%" }} />
                                <col style={{ width: "16%" }} />
                                <col style={{ width: "10%" }} />
                                <col style={{ width: "22%" }} />
                                <col style={{ width: "12%" }} />
                            </colgroup>
                            <thead>
                                <tr>
                                    <th style={thStyle}>项目</th>
                                    <th style={thStyle}>首次绑定</th>
                                    <th style={thStyle}>最近使用</th>
                                    <th style={thStyle}>次数</th>
                                    <th style={thStyle}>最近一次复现的条目</th>
                                    <th style={thStyle}>操作</th>
                                </tr>
                            </thead>
                            <tbody>
                                {bindings.map(b => {
                                    const pid = String(b.project_id);
                                    const open = openBinding === pid;
                                    const taskId = b.last_task_id === null || b.last_task_id === undefined
                                        ? "" : String(b.last_task_id);
                                    // 该次复现产出的条目由后端按 task_id 反查给出（run_id 不是 task_id）
                                    const runItems = Array.isArray(b.last_run_items)
                                        ? (b.last_run_items as Array<Record<string, unknown>>)
                                        : [];
                                    return (
                                        <Fragment key={pid}>
                                            <tr>
                                                <td style={tdStyle}>
                                                    {String(b.project_name ?? pid.slice(0, 8))}
                                                    <span style={{ color: "#475569", fontFamily: "monospace", fontSize: 10, marginLeft: 6 }}>
                                                        #{pid.slice(0, 6)}
                                                    </span>
                                                </td>
                                                <td style={{ ...tdStyle, fontSize: 11, color: "#94a3b8" }}>
                                                    {shortTime(b.created_at)}
                                                </td>
                                                <td style={{ ...tdStyle, fontSize: 11, color: "#94a3b8" }}>
                                                    {shortTime(b.last_used_at)}
                                                </td>
                                                <td style={tdStyle}>{String(b.uses ?? "")}</td>
                                                <td style={{ ...tdStyle, fontSize: 11, color: "#94a3b8" }}>
                                                    {runItems.length
                                                        ? `${runItems.length} 条（${runItems.map(r => String(r.metric_name ?? "?")).join("、")}）`
                                                        : "—"}
                                                </td>
                                                <td style={tdStyle}>
                                                    <button
                                                        style={btnStyle}
                                                        disabled={!runItems.length}
                                                        title={runItems.length ? undefined : "该次没有产出对照结果"}
                                                        onClick={() => setOpenBinding(open ? null : pid)}
                                                    >
                                                        {open ? "收起" : "看条目"}
                                                    </button>
                                                </td>
                                            </tr>
                                            {open && (
                                                <tr>
                                                    <td colSpan={6} style={{ ...tdStyle, background: "#0f172a" }}>
                                                        <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>
                                                            最近一次复现任务 {taskId}
                                                        </div>
                                                        {runItems.map(r => (
                                                            <div key={String(r.result_id)} style={{ fontFamily: "monospace", fontSize: 11 }}>
                                                                {String(r.dataset_name ?? "—")} · {String(r.metric_name ?? "?")}：
                                                                报告 {String(r.metric_value_reported ?? "—")} → 实测{" "}
                                                                {r.metric_value_actual === null || r.metric_value_actual === undefined
                                                                    ? "—" : String(r.metric_value_actual)}{" "}
                                                                <span style={{ color: verdictColor(r.verdict as string | undefined) }}>
                                                                    [{String(r.verdict ?? "—")}]
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </td>
                                                </tr>
                                            )}
                                        </Fragment>
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
const tableStyle: CSSProperties = {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 11,
    tableLayout: "fixed", // 配合 colgroup 定宽
};
const thStyle: CSSProperties = { textAlign: "left", color: "#64748b", fontWeight: 600, padding: "3px 6px", borderBottom: "1px solid #1f2937" };
const tdStyle: CSSProperties = {
    padding: "4px 6px",
    borderBottom: "1px solid #1f2937",
    verticalAlign: "top",
    overflowWrap: "anywhere", // 长 JSON / 长 URL 换行而不是撑列
};
