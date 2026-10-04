// features/network/NetworkRunPanel.tsx — 画布网络运行面板（阶段4 4c，模块详细设计 7.5）。
// 数据集下拉（模块三预处理产物仍在的注册条目）→ 目标环境（默认复用父原始项目独立环境，
// 画布新建的网络可选其它 original 已就绪环境；都不就绪时给出模块一建环境引导）→
// 训练超参 → 启动 → 任务轮询（进度 stage）→ 指标与运行记录（run_type=train）。
// 启动前先把画布当前图落盘（导出即所存即所训），不静默训练旧快照。

import { useCallback, useEffect, useMemo, useState } from "react";
import {
    createProjectEnv, getNetworkRunOptions, getProjectEnvStatus, listNetworkRuns, listProjects,
    postNetworkRun, type NetworkRunOptions, type NetworkRunRecord, type Project, type Task,
} from "../../api/client";
import { useTaskPolling } from "../../hooks/useTaskPolling";

export type NetworkRunPanelProps = {
    projectId: string;
    /** 启动前把画布当前图落盘（PUT graph）。 */
    saveGraph: () => Promise<void>;
    onClose: () => void;
};

/** 任务进度（task.progress JSON）里的 stage 字段，轮询期展示。 */
function stageOf(task: Task | null): string {
    if (!task?.progress) return "";
    try {
        const parsed = JSON.parse(task.progress) as { stage?: string };
        return typeof parsed.stage === "string" ? parsed.stage : "";
    } catch {
        return "";
    }
}

function parseMetrics(rec: NetworkRunRecord): Record<string, number> {
    try {
        const m = rec.metrics ? (JSON.parse(rec.metrics) as unknown) : null;
        return m && typeof m === "object" ? (m as Record<string, number>) : {};
    } catch {
        return {};
    }
}

const inputStyle: React.CSSProperties = {
    background: "#0f172a",
    border: "1px solid #1f2a2f",
    color: "#e2e8f0",
    borderRadius: 6,
    padding: "5px 8px",
    fontSize: 12,
    width: "100%",
    boxSizing: "border-box",
};

const labelStyle: React.CSSProperties = {
    display: "block",
    fontSize: 11,
    color: "#94a3b8",
    marginBottom: 4,
};

const smallButtonStyle: React.CSSProperties = {
    border: "1px solid #1f2a2f",
    borderRadius: 6,
    padding: "2px 8px",
    fontSize: 11,
    background: "#134e4a",
    color: "#e2e8f0",
    whiteSpace: "nowrap",
};

export default function NetworkRunPanel({ projectId, saveGraph, onClose }: NetworkRunPanelProps) {
    const [options, setOptions] = useState<NetworkRunOptions | null>(null);
    const [optionsError, setOptionsError] = useState<string | null>(null);
    const [runs, setRuns] = useState<NetworkRunRecord[]>([]);

    const [datasetId, setDatasetId] = useState<string>("");
    const [environmentProjectId, setEnvironmentProjectId] = useState<string>("");
    const [epochs, setEpochs] = useState(5);
    const [batchSize, setBatchSize] = useState(32);
    const [learningRate, setLearningRate] = useState(0.001);

    const [taskId, setTaskId] = useState<string | null>(null);
    const [task, setTask] = useState<Task | null>(null);
    const [startState, setStartState] = useState<"idle" | "starting" | "error">("idle");
    const [startError, setStartError] = useState<string | null>(null);

    // 4c-3：环境（复用模块一 2.3 的既有 env 接口）。
    // 画布新建的结构化项目没有父项目，此时允许从全部 original 项目里选一个作为环境来源并就地建环境。
    const [envTaskId, setEnvTaskId] = useState<string | null>(null);
    const [envStatus, setEnvStatus] = useState<string | null>(null);
    const [envError, setEnvError] = useState<string | null>(null);
    const [originals, setOriginals] = useState<Project[]>([]);
    const [envTargetId, setEnvTargetId] = useState<string>("");

    const loadRuns = useCallback(async () => {
        try {
            setRuns(await listNetworkRuns(projectId));
        } catch {
            // 运行记录加载失败不阻塞面板（训练结束后会再次刷新）
        }
    }, [projectId]);

    const loadOriginals = useCallback(async () => {
        try {
            setOriginals(await listProjects("original"));
        } catch {
            // original 项目列表失败不阻塞面板（父项目环境仍可用）
        }
    }, []);

    /** 拉运行面板初始化数据（建环境完成后复用刷新）。 */
    const refreshOptions = useCallback(async () => {
        try {
            const opts = await getNetworkRunOptions(projectId);
            setOptions(opts);
            setOptionsError(null);
            setEnvironmentProjectId(prev => prev || opts.parent_project_id || opts.environments[0]?.project_id || "");
        } catch (e) {
            setOptionsError(e instanceof Error ? e.message : String(e));
        }
    }, [projectId]);

    useEffect(() => {
        void (async () => {
            await refreshOptions();
            await loadOriginals();
            try {
                setRuns(await listNetworkRuns(projectId));
            } catch {
                // 运行记录加载失败不阻塞面板（训练结束后会再次刷新）
            }
        })();
    }, [projectId, refreshOptions, loadOriginals]);

    // 建环境的目标：显式选择 > 父项目 > 第一个 original 项目（没有父项目也能建）
    const envTarget = envTargetId || options?.parent_project_id || originals[0]?.project_id || null;
    const envTargetIsParent = !!envTarget && envTarget === options?.parent_project_id;

    useEffect(() => {
        if (!envTarget) return;
        let cancelled = false;
        void (async () => {
            try {
                const status = (await getProjectEnvStatus(envTarget)).status;
                if (!cancelled) setEnvStatus(status);
            } catch {
                if (!cancelled) setEnvStatus(null);
            }
        })();
        return () => {
            cancelled = true;
        };
    }, [envTarget]);

    const handleStart = useCallback(async () => {
        if (startState === "starting") return;
        setStartError(null);
        if (!datasetId) {
            setStartState("error");
            setStartError("请先选择训练数据集（列表为空则需先在模块三完成预处理）");
            return;
        }
        if (!environmentProjectId) {
            setStartState("error");
            setStartError(
                "没有可用的运行环境：请先走模块一为某个原始项目建立独立环境，完成后刷新本面板",
            );
            return;
        }
        setStartState("starting");
        try {
            // 先落盘当前画布（导出即所存即所训），再发起训练
            await saveGraph();
            const res = await postNetworkRun(projectId, {
                dataset_id: datasetId,
                environment_project_id: environmentProjectId || null,
                epochs,
                batch_size: batchSize,
                learning_rate: learningRate,
            });
            setTaskId(res.task_id);
            setTask(null);
            setStartState("idle");
        } catch (e) {
            setStartState("error");
            setStartError(e instanceof Error ? e.message : String(e));
        }
    }, [projectId, datasetId, environmentProjectId, epochs, batchSize, learningRate, saveGraph, startState]);

    // 4c-3「可选新建环境」：为所选原始项目发起建环境（既有接口），任务轮询到终态后刷新面板
    const envReady = !!envTarget && !!options?.environments.some(e => e.project_id === envTarget);

    const handleCreateEnv = useCallback(async () => {
        if (!envTarget || envTaskId) return;
        setEnvError(null);
        setEnvStatus(null);
        try {
            const res = await createProjectEnv(envTarget);
            setEnvTaskId(res.task_id);
        } catch (e) {
            setEnvError(e instanceof Error ? e.message : String(e));
        }
    }, [envTarget, envTaskId]);

    useTaskPolling({
        taskId: envTaskId,
        onDone: async () => {
            setEnvTaskId(null);
            await refreshOptions();
            await loadOriginals();
        },
        onError: message => {
            setEnvTaskId(null);
            setEnvError(message);
        },
    });

    useTaskPolling({
        taskId,
        onDone: async () => {
            await loadRuns();
            setTaskId(null);
        },
        onError: message => {
            setStartState("error");
            setStartError(message);
            setTaskId(null);
        },
        onProgress: t => setTask(t),
    });

    const latestRun = runs[0] ?? null;
    const latestMetrics = useMemo(() => (latestRun ? parseMetrics(latestRun) : {}), [latestRun]);

    const running = taskId !== null;
    const stage = stageOf(task);

    return (
        <div
            style={{
                position: "absolute",
                top: 56,
                right: 12,
                width: 330,
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
            }}
        >
            <div style={{ display: "flex", alignItems: "center", marginBottom: 10 }}>
                <span style={{ fontWeight: 700 }}>训练运行</span>
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

            {optionsError && (
                <div style={{ color: "#f87171", marginBottom: 8 }}>{optionsError}</div>
            )}

            <label style={labelStyle}>数据集（模块三预处理产物）</label>
            <select
                style={inputStyle}
                value={datasetId}
                onChange={e => setDatasetId(e.target.value)}
                disabled={running}
            >
                <option value="">{options && options.datasets.length === 0 ? "（无可用数据集）" : "请选择…"}</option>
                {options?.datasets.map(d => (
                    <option key={d.dataset_id} value={d.dataset_id}>
                        {d.name ?? d.dataset_id}
                        {d.task_type ? `（${d.task_type}）` : ""}
                    </option>
                ))}
            </select>

            <label style={{ ...labelStyle, marginTop: 10 }}>目标环境（项目独立环境）</label>
            <select
                style={inputStyle}
                value={environmentProjectId}
                onChange={e => setEnvironmentProjectId(e.target.value)}
                disabled={running}
            >
                <option value="">
                    {options && options.environments.length === 0
                        ? "（无已就绪环境，请先走模块一建环境）"
                        : "请选择…"}
                </option>
                {options?.environments.map(env => (
                    <option key={env.project_id} value={env.project_id}>
                        {env.name}
                        {env.project_id === options.parent_project_id ? "（父项目）" : ""}
                    </option>
                ))}
            </select>

            {envTarget && (
                <div style={{ marginTop: 6, fontSize: 11, color: "#94a3b8",
                              display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <span>
                        {envTargetIsParent ? "父项目环境" : "所选原始项目环境"}：
                        {envTaskId ? "创建中…" : envReady ? "已就绪" : (envStatus ?? "未就绪")}
                    </span>
                    <button
                        onClick={() => void handleCreateEnv()}
                        disabled={!!envTaskId || running}
                        style={{ ...smallButtonStyle, cursor: envTaskId || running ? "wait" : "pointer" }}
                    >
                        {envTaskId ? "创建中…" : envReady ? "重建环境" : "建环境"}
                    </button>
                </div>
            )}
            {!options?.parent_project_id && (
                <div style={{ marginTop: 4, fontSize: 11, color: "#64748b" }}>
                    本项目没有父项目（画布新建）：可在下方选择任意原始项目作为环境来源，
                    并直接为它建/重建独立环境（POST /api/projects/&#123;id&#125;/env，轮询到终态后刷新）。
                </div>
            )}
            <label style={{ ...labelStyle, marginTop: 8 }}>环境来源（原始项目，可就地建环境）</label>
            <select
                style={inputStyle}
                value={envTarget ?? ""}
                onChange={e => setEnvTargetId(e.target.value)}
                disabled={running || !!envTaskId}
            >
                <option value="">{originals.length === 0 ? "（没有原始项目，请先创建）" : "请选择…"}</option>
                {originals.map(p => (
                    <option key={p.project_id} value={p.project_id}>
                        {p.name || p.project_id}
                        {p.project_id === options?.parent_project_id ? "（父项目）" : ""}
                        {options?.environments.some(e => e.project_id === p.project_id) ? " · 环境已就绪" : " · 无环境"}
                    </option>
                ))}
            </select>
            {envError && (
                <div style={{ color: "#f87171", marginTop: 4, fontSize: 11, whiteSpace: "pre-wrap" }}>{envError}</div>
            )}

            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                <div style={{ flex: 1 }}>
                    <label style={labelStyle}>epochs</label>
                    <input
                        style={inputStyle}
                        type="number"
                        min={1}
                        max={1000}
                        value={epochs}
                        onChange={e => setEpochs(Number(e.target.value))}
                        disabled={running}
                    />
                </div>
                <div style={{ flex: 1 }}>
                    <label style={labelStyle}>batch_size</label>
                    <input
                        style={inputStyle}
                        type="number"
                        min={1}
                        max={4096}
                        value={batchSize}
                        onChange={e => setBatchSize(Number(e.target.value))}
                        disabled={running}
                    />
                </div>
                <div style={{ flex: 1 }}>
                    <label style={labelStyle}>learning_rate</label>
                    <input
                        style={inputStyle}
                        type="number"
                        min={0.000001}
                        max={10}
                        step={0.0001}
                        value={learningRate}
                        onChange={e => setLearningRate(Number(e.target.value))}
                        disabled={running}
                    />
                </div>
            </div>

            <button
                onClick={() => void handleStart()}
                disabled={running || startState === "starting" || !options}
                style={{
                    marginTop: 12,
                    width: "100%",
                    border: "1px solid #1f2a2f",
                    borderRadius: 8,
                    padding: "7px 14px",
                    fontWeight: 600,
                    fontSize: 12,
                    cursor: running || startState === "starting" || !options ? "wait" : "pointer",
                    background: "#0f766e",
                    color: "#e2e8f0",
                }}
            >
                {running ? `训练中… ${stage}` : "启动训练"}
            </button>

            {startError && (
                <div style={{ color: "#f87171", marginTop: 8, whiteSpace: "pre-wrap" }}>{startError}</div>
            )}
            {!startError && !running && startState === "error" && (
                <div style={{ color: "#f87171", marginTop: 8 }}>启动失败，请检查上方配置</div>
            )}

            {latestRun && (
                <div style={{ marginTop: 14, borderTop: "1px solid #1f2a2f", paddingTop: 10 }}>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>
                        最近一次训练（{latestRun.started_at.slice(0, 16).replace("T", " ")}）
                    </div>
                    {Object.keys(latestMetrics).length > 0 ? (
                        <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
                            <tbody>
                                {Object.entries(latestMetrics).map(([k, v]) => (
                                    <tr key={k}>
                                        <td style={{ color: "#94a3b8", padding: "2px 0" }}>{k}</td>
                                        <td style={{ textAlign: "right", fontFamily: "monospace" }}>
                                            {typeof v === "number" ? v.toFixed(4) : String(v)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div style={{ color: "#64748b" }}>本次训练未产出指标</div>
                    )}
                    <div style={{ color: "#64748b", fontSize: 10, marginTop: 4 }}>
                        {runs.length} 条成功训练记录（run_type=train）
                    </div>
                </div>
            )}
        </div>
    );
}
