// App.tsx — 顶层视图切换（模块四 B1；不引 react-router，最小 state 切换）。
// projects：项目列表（创建原始项目）；viewer：模型查看器（模块四闭环操作链）；
// canvas：结构化项目画布（GraphIR 快照打开/保存）；sandbox：原沙盒画布（localStorage）。

import { useEffect, useState } from "react";
import ErrorBoundary from "./components/ErrorBoundary.tsx";
import SettingsModal from "./components/SettingsModal.tsx";
import AssistantPanel from "./components/AssistantPanel.tsx";
import { fetchCredentialsStatus } from "./api/systemClient.ts";
import FlowEditor from "./FlowEditor.tsx";
import ProjectListView from "./views/ProjectListView.tsx";
import ModelViewerView from "./views/ModelViewerView.tsx";
import CanvasProjectView from "./views/CanvasProjectView.tsx";
import ReproduceView from "./views/ReproduceView.tsx";
import TaskBoardView from "./views/TaskBoardView.tsx";

type AppView =
    | { kind: "projects" }
    | { kind: "viewer"; projectId: string }
    | { kind: "canvas"; projectId: string }
    | { kind: "reproduce"; projectId: string; paperId?: string }
    | { kind: "tasks" }
    | { kind: "sandbox" };

// 首启未配置凭证时自动弹「设置凭证」页（评审裁定第 6 项）；「稍后再说」写入跳过标记，不再每次打扰。
const FIRST_RUN_SKIP_KEY = "dlai_credentials_skip";

function App() {
    const [view, setView] = useState<AppView>({ kind: "projects" });
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [assistantOpen, setAssistantOpen] = useState(false);
    const [firstRun, setFirstRun] = useState(false);

    useEffect(() => {
        let cancelled = false;
        void (async () => {
            if (localStorage.getItem(FIRST_RUN_SKIP_KEY) === "1") return;
            try {
                const status = await fetchCredentialsStatus();
                if (!cancelled && !status.configured) setFirstRun(true);
            } catch {
                // 后端不可达时静默：自检与凭证页仍可从右下角「设置」按钮手动打开
            }
        })();
        return () => {
            cancelled = true;
        };
    }, []);

    const closeSettings = () => {
        if (firstRun) {
            localStorage.setItem(FIRST_RUN_SKIP_KEY, "1");
            setFirstRun(false);
        }
        setSettingsOpen(false);
    };

    // 每个视图各自包一层错误边界：切换视图时重置（key=视图名），
    // 某个视图渲染崩溃只影响该视图，不会整页黑屏，且能看到错误信息。
    const viewKey = view.kind === "viewer" || view.kind === "canvas" || view.kind === "reproduce"
        ? `${view.kind}:${view.projectId}`
        : view.kind;
    const renderView = () => {
        switch (view.kind) {
            case "viewer":
                return (
                    <ModelViewerView
                        projectId={view.projectId}
                        onBack={() => setView({ kind: "projects" })}
                        onOpenCanvas={projectId => setView({ kind: "canvas", projectId })}
                    />
                );
            case "canvas":
                return <CanvasProjectView projectId={view.projectId} onBack={() => setView({ kind: "projects" })} />;
            case "reproduce":
                return (
                    <ReproduceView
                        projectId={view.projectId}
                        paperId={view.paperId}
                        onBack={() => setView({ kind: "projects" })}
                    />
                );
            case "tasks":
                return <TaskBoardView onBack={() => setView({ kind: "projects" })} />;
            case "sandbox":
                return <FlowEditor />;
            default:
                return (
                    <ProjectListView
                        onOpenViewer={projectId => setView({ kind: "viewer", projectId })}
                        onOpenCanvas={projectId => setView({ kind: "canvas", projectId })}
                        onOpenReproduce={(projectId, paperId) => setView({ kind: "reproduce", projectId, paperId })}
                        onOpenTasks={() => setView({ kind: "tasks" })}
                        onOpenSandbox={() => setView({ kind: "sandbox" })}
                    />
                );
        }
    };

    const assistantContext = {
        page: view.kind,
        project_id: view.kind === "viewer" || view.kind === "canvas" || view.kind === "reproduce"
            ? view.projectId
            : undefined,
    };

    return (
        <>
            <ErrorBoundary key={viewKey}>{renderView()}</ErrorBoundary>
            <button
                onClick={() => setAssistantOpen(v => !v)}
                title="AI 助手（只读：解释现状 / 给建议，不执行动作）"
                style={{
                    position: "fixed",
                    right: 14,
                    bottom: 56,
                    zIndex: 55,
                    background: "#1e293b",
                    border: "1px solid #334155",
                    borderRadius: 20,
                    color: "#cbd5e1",
                    padding: "7px 12px",
                    fontSize: 13,
                    cursor: "pointer",
                }}
            >
                🤖 助手
            </button>
            <button
                onClick={() => setSettingsOpen(true)}
                title="设置（模型接口凭证 / 运行环境自检）"
                style={{
                    position: "fixed",
                    right: 14,
                    bottom: 14,
                    zIndex: 55,
                    background: "#1e293b",
                    border: "1px solid #334155",
                    borderRadius: 20,
                    color: "#cbd5e1",
                    padding: "7px 12px",
                    fontSize: 13,
                    cursor: "pointer",
                }}
            >
                ⚙ 设置
            </button>
            <SettingsModal open={settingsOpen || firstRun} firstRun={firstRun} onClose={closeSettings} />
            <AssistantPanel
                open={assistantOpen}
                context={assistantContext}
                onClose={() => setAssistantOpen(false)}
                onOpenSettings={() => setSettingsOpen(true)}
            />
        </>
    );
}

export default App;
