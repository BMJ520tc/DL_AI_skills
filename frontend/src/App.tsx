// App.tsx — 顶层视图切换（模块四 B1；不引 react-router，最小 state 切换）。
// projects：项目列表（创建原始项目）；viewer：模型查看器（模块四闭环操作链）；
// canvas：结构化项目画布（GraphIR 快照打开/保存）；sandbox：原沙盒画布（localStorage）。

import { useState } from "react";
import ErrorBoundary from "./components/ErrorBoundary.tsx";
import FlowEditor from "./FlowEditor.tsx";
import ProjectListView from "./views/ProjectListView.tsx";
import ModelViewerView from "./views/ModelViewerView.tsx";
import CanvasProjectView from "./views/CanvasProjectView.tsx";

type AppView =
    | { kind: "projects" }
    | { kind: "viewer"; projectId: string }
    | { kind: "canvas"; projectId: string }
    | { kind: "sandbox" };

function App() {
    const [view, setView] = useState<AppView>({ kind: "projects" });

    // 每个视图各自包一层错误边界：切换视图时重置（key=视图名），
    // 某个视图渲染崩溃只影响该视图，不会整页黑屏，且能看到错误信息。
    const viewKey = view.kind === "viewer" || view.kind === "canvas" ? `${view.kind}:${view.projectId}` : view.kind;
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
            case "sandbox":
                return <FlowEditor />;
            default:
                return (
                    <ProjectListView
                        onOpenViewer={projectId => setView({ kind: "viewer", projectId })}
                        onOpenCanvas={projectId => setView({ kind: "canvas", projectId })}
                        onOpenSandbox={() => setView({ kind: "sandbox" })}
                    />
                );
        }
    };

    return <ErrorBoundary key={viewKey}>{renderView()}</ErrorBoundary>;
}

export default App;
