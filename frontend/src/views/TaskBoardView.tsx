// views/TaskBoardView.tsx — 任务看板：按类型分栏，进二级页看该类型的任务 + 运行中实时进度。
//
// 为什么需要：长任务（复现 ~30 分钟、训练十几分钟）原先在界面上「看不到」——
// 任务进度只在结束时才写，用户只能干等。这里把「现在在跑什么、跑到哪了」摆到明面。
//
// 两级结构：① 类型总览（每类多少条、几条在跑、几条失败）→ ② 该类型的任务表。
// 排序用后端 board 口径（执行中 → 排队中 → 其余按创建时间倒序），
// 而不是纯时间倒序——否则一个跑半小时的长任务会被后来的一堆短任务挤下去。

import { Fragment, useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import {
    cancelTask,
    listTasks,
    retryTask,
    type Task,
} from "../api/client";

const ACTIVE = new Set(["running", "queued"]);
const POLL_MS = 3000;

/** 任务类型 → 可读名（未收录的原样显示，不隐藏信息）。 */
const TYPE_LABELS: Record<string, string> = {
    pdf_parse: "解析论文",
    extract_items: "抽取实验条目",
    reproduce: "论文复现",
    conclusion: "可信度结论",
    preprocess: "数据预处理",
    baseline: "基准运行",
    compare: "结果对比",
    network_train: "模型训练",
    network_autotune: "自动调参",
    decompose: "模型拆解",
    decompose_trace: "追踪结构",
    decompose_verify: "两步验证",
    module_ingest: "模块入库",
    multi_model: "多模型分析",
    knowledge_distill: "知识蒸馏",
    paper_distill: "论文蒸馏",
    align: "标签对齐",
    assistant_chat: "AI 助手",
    env_create: "建独立环境",
    analyze: "结构分析",
    verify: "最小可运行验证",
    extract_addresses: "地址抽取",
    agent_task: "智能体任务",
};

const STATUS_STYLE: Record<string, { label: string; color: string }> = {
    running: { label: "执行中", color: "#38bdf8" },
    queued: { label: "排队中", color: "#d97706" },
    success: { label: "成功", color: "#16a34a" },
    failed: { label: "失败", color: "#dc2626" },
    cancelled: { label: "已取消", color: "#64748b" },
};

const typeLabel = (taskType: string): string => TYPE_LABELS[taskType] ?? taskType;

/** 任务进度（JSON 文本）→ 一行可读文本；拿不到就回原文。 */
function progressText(raw: string | null): string {
    if (!raw) return "";
    try {
        const p = JSON.parse(raw) as Record<string, unknown>;
        if (p && typeof p === "object" && !Array.isArray(p)) {
            const parts = [p.stage, p.live, p.activity, p.failure_reason, p.reason]
                .filter(v => typeof v === "string" && (v as string).trim())
                .map(v => String(v));
            if (parts.length) return Array.from(new Set(parts)).join(" · ");
        }
    } catch {
        /* 非 JSON 进度：按原文显示 */
    }
    return raw;
}

/** 任务 params（JSON 文本）→ 对象；解析不出给空对象。 */
function parseParams(raw: string | null): Record<string, unknown> {
    try {
        const p = JSON.parse(raw || "{}") as unknown;
        return p && typeof p === "object" && !Array.isArray(p) ? (p as Record<string, unknown>) : {};
    } catch {
        return {};
    }
}

/**
 * 任务「作用于谁」——项目名会说谎（同名的项目不止一个），论文级任务（解析/抽取）
 * 甚至没有 project_id。所以除了项目名，再带上 id 前缀与关键 params（论文/数据集/来源任务），
 * 才能一眼看出「这条任务到底属于哪个对象」。
 */
function targetLabel(t: Task): string {
    const parts: string[] = [];
    if (t.project_name) parts.push(t.project_name);
    if (t.project_id) parts.push(`#${t.project_id.slice(0, 6)}`);
    const p = parseParams(t.params);
    if (typeof p.paper_id === "string" && p.paper_id) parts.push(`论文 ${p.paper_id}`);
    if (Array.isArray(p.paper_ids) && p.paper_ids.length) parts.push(`论文 ${p.paper_ids.join("、")}`);
    if (typeof p.dataset_id === "string" && p.dataset_id) parts.push(`数据集 ${p.dataset_id}`);
    if (typeof p.source_task_id === "string" && p.source_task_id) {
        parts.push(`源自任务 ${p.source_task_id.slice(0, 8)}`);
    }
    return parts.join(" · ") || "—";
}

/** 进度全文（展开行里给完整的、可读的文本）。 */
function progressDetail(raw: string | null): string {
    if (!raw) return "";
    try {
        return JSON.stringify(JSON.parse(raw), null, 2);
    } catch {
        return raw;
    }
}

/** 用时：跑完的用 updated-created；还在跑的用 now-created。 */
function duration(task: Task, now: number): string {
    const start = Date.parse(task.created_at);
    if (Number.isNaN(start)) return "—";
    const end = ACTIVE.has(task.status) ? now : Date.parse(task.updated_at);
    const sec = Math.max(0, Math.round(((Number.isNaN(end) ? now : end) - start) / 1000));
    if (sec < 60) return `${sec} 秒`;
    if (sec < 3600) return `${Math.floor(sec / 60)} 分 ${sec % 60} 秒`;
    return `${Math.floor(sec / 3600)} 时 ${Math.floor((sec % 3600) / 60)} 分`;
}

type TypeSummary = {
    taskType: string;
    total: number;
    running: number;
    queued: number;
    failed: number;
    latest: string;
};

/** 类型总览：有活动的置顶，其余按最近一次任务时间倒序。 */
function summarize(tasks: Task[]): TypeSummary[] {
    const byType = new Map<string, TypeSummary>();
    for (const t of tasks) {
        const s = byType.get(t.task_type)
            ?? { taskType: t.task_type, total: 0, running: 0, queued: 0, failed: 0, latest: "" };
        s.total += 1;
        if (t.status === "running") s.running += 1;
        else if (t.status === "queued") s.queued += 1;
        else if (t.status === "failed") s.failed += 1;
        if (t.created_at > s.latest) s.latest = t.created_at;
        byType.set(t.task_type, s);
    }
    return [...byType.values()].sort((a, b) => {
        const aActive = a.running + a.queued > 0 ? 0 : 1;
        const bActive = b.running + b.queued > 0 ? 0 : 1;
        return aActive - bActive || b.latest.localeCompare(a.latest);
    });
}

export type TaskBoardViewProps = {
    onBack: () => void;
};

export default function TaskBoardView({ onBack }: TaskBoardViewProps) {
    const [tasks, setTasks] = useState<Task[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [selectedType, setSelectedType] = useState<string | null>(null);
    const [expanded, setExpanded] = useState<string | null>(null);
    const [now, setNow] = useState(() => Date.now());
    const busyRef = useRef(false);

    const refresh = useCallback(async () => {
        try {
            setTasks(await listTasks("board", 300));
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    // 有活动任务时自动轮询（顺带刷新「用时」的 now）
    useEffect(() => {
        if (!tasks.some(t => ACTIVE.has(t.status))) return;
        const timer = setInterval(() => {
            setNow(Date.now());
            void refresh();
        }, POLL_MS);
        return () => clearInterval(timer);
    }, [tasks, refresh]);

    const act = async (kind: "cancel" | "retry", taskId: string) => {
        if (busyRef.current) return;
        busyRef.current = true;
        setError(null);
        try {
            if (kind === "cancel") await cancelTask(taskId);
            else await retryTask(taskId);
            await refresh();
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            busyRef.current = false;
        }
    };

    const running = tasks.filter(t => t.status === "running").length;
    const queued = tasks.filter(t => t.status === "queued").length;
    const summaries = summarize(tasks);
    const visible = selectedType ? tasks.filter(t => t.task_type === selectedType) : tasks;

    const header = (
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16, flexWrap: "wrap" }}>
            <button style={btnStyle} onClick={selectedType ? () => { setSelectedType(null); setExpanded(null); } : onBack}>
                ← 返回
            </button>
            <div>
                <div style={{ fontWeight: 700, fontSize: 16 }}>
                    任务看板{selectedType ? ` · ${typeLabel(selectedType)}` : ""}
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8" }}>
                    执行中 {running} · 排队中 {queued} · 共 {tasks.length} 条
                    {running + queued > 0 ? `（每 ${POLL_MS / 1000} 秒自动刷新）` : "（无活动任务，不轮询）"}
                </div>
            </div>
            <button style={{ ...btnStyle, marginLeft: "auto", background: "#334155" }} onClick={() => void refresh()}>
                刷新
            </button>
        </div>
    );

    const banner = error && (
        <div style={{ ...bannerStyle, background: "#3f1d1d", borderColor: "#b91c1c", color: "#fecaca" }}>
            {error}
            <button style={{ ...btnStyle, marginLeft: 10 }} onClick={() => setError(null)}>关闭</button>
        </div>
    );

    return (
        <div style={{ minHeight: "100vh", background: "#0b1220", color: "#e2e8f0", padding: "24px 32px" }}>
            <div style={{ maxWidth: 1240, margin: "0 auto" }}>
                {header}
                {banner}
                {loading ? (
                    <div style={{ color: "#64748b" }}>加载中…</div>
                ) : selectedType ? (
                    <TaskTable
                        tasks={visible}
                        now={now}
                        expanded={expanded}
                        onToggle={id => setExpanded(prev => (prev === id ? null : id))}
                        onAct={act}
                    />
                ) : (
                    <TypeOverview summaries={summaries} onOpen={setSelectedType} />
                )}
            </div>
        </div>
    );
}

function TypeOverview({ summaries, onOpen }: { summaries: TypeSummary[]; onOpen: (t: string) => void }) {
    if (!summaries.length) return <div style={{ color: "#64748b" }}>暂无任务</div>;
    return (
        <table style={tableStyle}>
            <colgroup>
                <col style={{ width: "30%" }} />
                <col style={{ width: "10%" }} />
                <col style={{ width: "22%" }} />
                <col style={{ width: "24%" }} />
                <col style={{ width: "14%" }} />
            </colgroup>
            <thead>
                <tr>
                    <th style={thStyle}>任务类型</th>
                    <th style={thStyle}>总数</th>
                    <th style={thStyle}>当前状态</th>
                    <th style={thStyle}>最近一次</th>
                    <th style={thStyle}>操作</th>
                </tr>
            </thead>
            <tbody>
                {summaries.map(s => (
                    <tr key={s.taskType} style={{ borderBottom: "1px solid #1f2937", cursor: "pointer" }}
                        onClick={() => onOpen(s.taskType)}>
                        <td style={tdStyle}>
                            <span style={{ fontWeight: 600 }}>{typeLabel(s.taskType)}</span>
                            <span style={{ color: "#475569", fontFamily: "monospace", fontSize: 11, marginLeft: 8 }}>
                                {s.taskType}
                            </span>
                        </td>
                        <td style={tdStyle}>{s.total}</td>
                        <td style={{ ...tdStyle, fontSize: 11 }}>
                            {s.running > 0 && <span style={{ color: "#38bdf8", marginRight: 8 }}>◐ 执行中 {s.running}</span>}
                            {s.queued > 0 && <span style={{ color: "#d97706", marginRight: 8 }}>排队 {s.queued}</span>}
                            {s.failed > 0 && <span style={{ color: "#dc2626", marginRight: 8 }}>失败 {s.failed}</span>}
                            {s.running + s.queued + s.failed === 0 && <span style={{ color: "#64748b" }}>—</span>}
                        </td>
                        <td style={{ ...tdStyle, color: "#94a3b8", fontSize: 11 }}>
                            {s.latest.slice(0, 19).replace("T", " ")}
                        </td>
                        <td style={tdStyle}>
                            <button style={btnSmall} onClick={e => { e.stopPropagation(); onOpen(s.taskType); }}>
                                进入 →
                            </button>
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

function TaskTable({
    tasks, now, expanded, onToggle, onAct,
}: {
    tasks: Task[];
    now: number;
    expanded: string | null;
    onToggle: (id: string) => void;
    onAct: (kind: "cancel" | "retry", id: string) => void;
}) {
    if (!tasks.length) return <div style={{ color: "#64748b" }}>该类型暂无任务</div>;
    return (
        <table style={tableStyle}>
            <colgroup>
                <col style={{ width: "8%" }} />
                <col style={{ width: "22%" }} />
                <col />
                <col style={{ width: "11%" }} />
                <col style={{ width: "7%" }} />
                <col style={{ width: "12%" }} />
            </colgroup>
            <thead>
                <tr>
                    <th style={thStyle}>状态</th>
                    <th style={thStyle}>项目 / 对象</th>
                    <th style={thStyle}>进度</th>
                    <th style={thStyle}>创建时间</th>
                    <th style={thStyle}>用时</th>
                    <th style={thStyle}>操作</th>
                </tr>
            </thead>
            <tbody>
                {tasks.map(t => {
                    const st = STATUS_STYLE[t.status] ?? { label: t.status, color: "#64748b" };
                    const prog = progressText(t.progress) || (t.error ?? "");
                    const isOpen = expanded === t.task_id;
                    return (
                        <Fragment key={t.task_id}>
                            <tr style={{ borderBottom: isOpen ? "none" : "1px solid #1f2937" }}>
                                <td style={tdStyle}>
                                    <span style={{ color: st.color, fontWeight: 600 }}>
                                        {t.status === "running" ? "◐ " : ""}{st.label}
                                    </span>
                                </td>
                                <td style={{ ...tdStyle, color: "#94a3b8" }} title={t.params || undefined}>
                                    {targetLabel(t)}
                                </td>
                                <td
                                    style={{ ...tdStyle, fontFamily: "monospace", fontSize: 11, color: prog ? "#cbd5e1" : "#475569", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                                    title={prog || undefined}
                                >
                                    {prog || "—"}
                                </td>
                                <td style={{ ...tdStyle, color: "#94a3b8", fontSize: 11 }}>
                                    {t.created_at.slice(0, 19).replace("T", " ")}
                                </td>
                                <td style={{ ...tdStyle, color: "#94a3b8", fontSize: 11 }}>{duration(t, now)}</td>
                                <td style={{ ...tdStyle, whiteSpace: "nowrap" }}>
                                    <button style={btnSmall} onClick={() => onToggle(t.task_id)}>
                                        {isOpen ? "收起" : "详情"}
                                    </button>
                                    {t.status === "queued" && (
                                        <button style={{ ...btnSmall, marginLeft: 6 }} onClick={() => onAct("cancel", t.task_id)}>取消</button>
                                    )}
                                    {t.status === "failed" && (
                                        <button style={{ ...btnSmall, marginLeft: 6 }} onClick={() => onAct("retry", t.task_id)}>重试</button>
                                    )}
                                </td>
                            </tr>
                            {isOpen && (
                                <tr style={{ borderBottom: "1px solid #1f2937" }}>
                                    <td colSpan={6} style={{ ...tdStyle, background: "#0f172a" }}>
                                        <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>
                                            任务 {t.task_id} · {typeLabel(t.task_type)}（{t.task_type}）
                                            {t.project_id ? ` · 项目 ${t.project_name || ""} #${t.project_id}` : ""}
                                        </div>
                                        <div style={{ fontSize: 11, color: "#64748b", marginBottom: 6, fontFamily: "monospace" }}>
                                            参数 {t.params || "{}"}
                                        </div>
                                        {t.error && (
                                            <div style={{ fontSize: 11, color: "#fca5a5", marginBottom: 6, whiteSpace: "pre-wrap" }}>
                                                {t.error}
                                            </div>
                                        )}
                                        <pre style={{ margin: 0, fontSize: 11, whiteSpace: "pre-wrap", maxHeight: 260, overflow: "auto", color: "#cbd5e1" }}>
                                            {progressDetail(t.progress) || "（无进度）"}
                                        </pre>
                                    </td>
                                </tr>
                            )}
                        </Fragment>
                    );
                })}
            </tbody>
        </table>
    );
}

const btnStyle: CSSProperties = {
    border: "1px solid #334155",
    background: "#0f766e",
    color: "#e2e8f0",
    borderRadius: 6,
    padding: "5px 12px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
};
const btnSmall: CSSProperties = { ...btnStyle, padding: "3px 8px", fontSize: 11 };
const bannerStyle: CSSProperties = {
    border: "1px solid",
    borderRadius: 8,
    padding: "8px 12px",
    fontSize: 12,
    marginBottom: 12,
};
const tableStyle: CSSProperties = {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 12,
    tableLayout: "fixed",
};
const thStyle: CSSProperties = { textAlign: "left", color: "#64748b", fontWeight: 600, fontSize: 11, padding: "6px 8px", borderBottom: "1px solid #1f2937" };
const tdStyle: CSSProperties = { padding: "6px 8px", verticalAlign: "top", overflowWrap: "anywhere" };
