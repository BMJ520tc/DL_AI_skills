import { ReactFlowProvider } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import React, { useCallback, useMemo, useRef, useState } from "react";
import type { GraphIR } from "./types/graph";
import { buildGraphIR } from "./utils/graphIR";

// Components
import DiagramView from "./components/DiagramView";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import ComputePanel from "./components/ComputePanel";
import TraceView from "./components/TraceView";
import KnowledgeSearchPanel from "./components/KnowledgeSearchPanel";
import { EditorToolbar } from "./features/editor/components/EditorToolbar";
import { EditorSidebar } from "./features/editor/components/EditorSidebar";
import { CodePanel } from "./features/editor/components/CodePanel";
import { SaveModuleModal } from "./features/editor/components/SaveModuleModal";
import { SaveCopyModal } from "./features/editor/components/SaveCopyModal";
import { ModuleEditorOverlay } from "./features/editor/components/ModuleEditorOverlay";
import { DuplicateModuleWarning } from "./features/editor/components/DuplicateModuleWarning";
import { EditorCanvas } from "./features/editor/components/EditorCanvas";

// Hooks
import { useGraphState } from "./features/editor/hooks/useGraphState";
import { useModuleSystem } from "./features/editor/hooks/useModuleSystem";
import { useGraphInteraction } from "./features/editor/hooks/useGraphInteraction";
import { useGraphLayout } from "./features/editor/hooks/useGraphLayout";
import { useTraceSystem } from "./features/editor/hooks/useTraceSystem";
import { useCodeGeneration } from "./features/editor/hooks/useCodeGeneration";
import { useExportSystem } from "./features/editor/hooks/useExportSystem";
import { LAYER_REGISTRY } from "./types/nodeTypes";
import { estimateGraphCost } from "./utils/computeEstimator";
import NetworkRunPanel from "./features/network/NetworkRunPanel";
import { exportNetwork } from "./api/client";

const TRACE_SEED_PRESETS = [42, 1337, 1234, 2020, 2021];

export type FlowEditorProps = {
    /** 外部画布模式（模块四 B3）：以 GraphIR 快照初始化且不读写 localStorage。 */
    initialGraph?: GraphIR | null;
    /** 保存回调（结构化项目画布 → PUT /api/projects/{id}/graph）。 */
    onSave?: (graph: GraphIR) => Promise<void>;
    /** 结构化项目 id：画布网络的「导出代码 / 运行训练」入口（阶段4 4c，模块详细设计 7.5）。 */
    projectId?: string;
};

function FlowContent({ initialGraph, onSave, projectId }: FlowEditorProps) {
    // 1. Core Graph State
    const {
        nodes, setNodes,
        edges, setEdges,
        onNodesChange, onEdgesChange,
        canUndo, canRedo, handleUndo, handleRedo,
        edgesWithHandlers
    } = useGraphState(initialGraph);

    // 外部画布模式（结构化项目画布）：画布代码生成/形状追踪对 ir 节点不生效（阶段3实施方案 3.8）。
    // 传空数组使两个系统空转；hook 调用顺序保持无条件，仅数据源按 props 切换。
    const isExternal = !!initialGraph;
    const codeNodes = useMemo(() => (isExternal ? [] : nodes), [isExternal, nodes]);
    const codeEdges = useMemo(() => (isExternal ? [] : edges), [isExternal, edges]);

    // 2. Code Generation
    const { generated, generatedCode, onDownloadCode } = useCodeGeneration(codeNodes, codeEdges);

    // 3. Layout & UI State
    const layout = useGraphLayout();

    // 4. Trace & Analysis
    const trace = useTraceSystem({
        nodes: codeNodes,
        edges: codeEdges,
        setNodes,
        generatedCode
    });

    // 5. Module System
    // 稳定的 schema 查询入口：保证 useModuleSystem 内部 useMemo 的依赖稳定。
    const getNodeSchema = useCallback(
        (type: string) => LAYER_REGISTRY[type]?.paramSchema,
        [],
    );
    const modSys = useModuleSystem({ nodes, edges, setNodes, getNodeSchema });

    // 6. Interaction (Drag/Drop, Selection)
    const interaction = useGraphInteraction({
        nodes,
        edges,
        setNodes,
        setEdges,
        moduleStack: modSys.moduleStack,
        setModuleStack: modSys.setModuleStack
    });

    const {
        mainFlowRef, moduleFlowRef,
        onMainDrop, onModuleDrop, onDragOver,
        highlightNodes, highlightEdges, setHighlightNodes, setHighlightEdges,
        onSelectionChange, clearSelection, selectedNodeIds,
        onConnect, onNodeDragStop, onNodeDragStart, onModuleNodeDragStart, onModuleNodeDragStop
    } = interaction;

    // Derived States for Visualization
    // 先取出回调，保证 useMemo 的依赖就是该回调本身（而不是整个 trace 对象）。
    const { getDecoratedEdges } = trace;
    const decoratedEdges = useMemo(() => {
        return getDecoratedEdges(edgesWithHandlers);
    }, [edgesWithHandlers, getDecoratedEdges]);

    const highlightedEdgesList = useMemo(() => {
        if (!highlightEdges.size) return decoratedEdges;
        let hasChanges = false;
        const newEdges = decoratedEdges.map(e => {
            const isHighlighted = highlightEdges.has(e.id);
            const data = (e.data && typeof e.data === "object") ? e.data as Record<string, unknown> : {};
            const currentlyHighlighted = !!data.highlight;
            if (isHighlighted === currentlyHighlighted) return e;
            hasChanges = true;
            return { ...e, data: { ...data, highlight: isHighlighted ? true : undefined } };
        });
        return hasChanges ? newEdges : decoratedEdges;
    }, [decoratedEdges, highlightEdges]);
    const { exportJson, exportPng, exportSvg, isExporting } = useExportSystem({ 
        nodes, 
        edges,
        modules: modSys.modules
    });
    const nodesForFlow = useMemo(() => {
        if (!highlightNodes.size) {
            // Check if any node currently has __highlight and remove it
            let hasChanges = false;
            const newNodes = nodes.map(n => {
                if (n.data && n.data.__highlight) {
                    hasChanges = true;
                    return { ...n, data: { ...n.data, __highlight: undefined } };
                }
                return n;
            });
            return hasChanges ? newNodes : nodes;
        }
        
        let hasChanges = false;
        const newNodes = nodes.map(n => {
            const isHighlighted = highlightNodes.has(n.id);
            const currentlyHighlighted = !!(n.data && n.data.__highlight);
            if (isHighlighted === currentlyHighlighted) return n;
            hasChanges = true;
            return { ...n, data: { ...(n.data || {}), __highlight: isHighlighted ? true : undefined } };
        });
        return hasChanges ? newNodes : nodes;
    }, [nodes, highlightNodes]);

    const computeSummary = useMemo(() => {
        return estimateGraphCost(nodes, edges, trace.shapeResult, LAYER_REGISTRY);
    }, [nodes, edges, trace.shapeResult]);

    // Helper for generating code toggle（外部画布模式无代码生成）
    const handleGenerateCode = () => {
        if (isExternal) return;
        layout.setShowLiveCode(v => !v);
    };

    // 知识库检索面板开关（独立于画布状态，不影响既有编辑流程）
    const [showKnowledge, setShowKnowledge] = useState(false);

    // 画布保存（模块四 B3 结构化项目最小闭环：全量 GraphIR v2 快照覆盖）
    const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
    const handleSaveGraph = async () => {
        if (!onSave || saveState === "saving") return;
        setSaveState("saving");
        try {
            await onSave(buildGraphIR(nodes, edges));
            setSaveState("saved");
            setTimeout(() => setSaveState("idle"), 2000);
        } catch (err) {
            console.error("保存画布失败", err);
            setSaveState("error");
        }
    };

    // 画布网络导出（阶段4 4c）：先落盘当前画布（导出即所存即所训），
    // 再从后端同一引擎取再生成代码下载（前端 codeCompile 对后端模块节点不可用）。
    const [exportState, setExportState] = useState<"idle" | "busy" | "error">("idle");
    const handleExportCode = async () => {
        if (!projectId || exportState === "busy") return;
        setExportState("busy");
        try {
            if (onSave) await onSave(buildGraphIR(nodes, edges));
            const { code } = await exportNetwork(projectId);
            const blob = new Blob([code], { type: "text/x-python" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "model.py";
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error("导出代码失败", err);
            setExportState("error");
            alert(err instanceof Error ? err.message : String(err));
        } finally {
            setExportState("idle");
        }
    };

    // 运行面板开关（阶段4 4c：训练任务发起、轮询与指标展示）
    const [showRunPanel, setShowRunPanel] = useState(false);

    // File Upload (ref needed)
    const uploadInputRef = useRef<HTMLInputElement>(null);
    const triggerUpload = () => uploadInputRef.current?.click();
    const onUploadGraph = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = ev => {
            try {
                const parsed = JSON.parse(String(ev.target?.result));
                if (parsed.modules && Array.isArray(parsed.modules)) {
                    modSys.mergeModules(parsed.modules);
                }
                setEdges([]);
                if (parsed.nodes && parsed.edges) {
                    
                    setNodes(parsed.nodes);
                    setEdges(parsed.edges);
                }
            } catch (err) {
                console.error("Failed to import graph", err);
                alert("Failed to import graph JSON.");
            }
        };
        reader.readAsText(file);
        event.target.value = "";
    };

    return (
        <div style={{ display: "flex", height: "100vh", width: "100%", overflow: "hidden" }}>
            <input
                ref={uploadInputRef}
                type="file"
                accept="application/json"
                style={{ display: "none" }}
                onChange={onUploadGraph}
            />
            {/* Sidebar */}
            <EditorSidebar
                sidebarCollapsed={layout.sidebarCollapsed}
                sidebarWidth={layout.sidebarWidth}
                dragSidebar={layout.dragSidebar}
                setSidebarCollapsed={layout.setSidebarCollapsed}
                setDragSidebar={layout.setDragSidebar}
                onGenerateCode={handleGenerateCode}
                showLiveCode={layout.showLiveCode}
                modules={modSys.modules}
                handleDeleteModule={modSys.handleDeleteModule}
                hideGenerateCode={isExternal}
            />

            <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", position: "relative", overflow: "hidden" }}>
                <EditorToolbar
                    canUndo={canUndo}
                    canRedo={canRedo}
                    canSaveModule={selectedNodeIds.length > 0}
                    traceLoading={trace.traceLoading}
                    traceSeedOptions={[...TRACE_SEED_PRESETS.map(String), "custom"]}
                    traceSeedPreset={trace.traceSeedPreset}
                    traceSeedCustom={trace.traceSeedCustom}
                    showCustomSeedInput={trace.traceSeedPreset === "custom"}
                    onUndo={handleUndo}
                    onRedo={handleRedo}
                    onTrace={trace.handleTrace}
                    onTraceSeedPresetChange={trace.setTraceSeedPreset}
                    onTraceSeedCustomChange={trace.setTraceSeedCustom}
                    onSaveModule={() => {
                        if (!selectedNodeIds.length) {
                            alert("Select at least one node to save as a module.");
                            return;
                        }
                        const suggestion = `Module ${modSys.modules.length + 1}`;
                        modSys.setPendingModuleName(suggestion);
                        modSys.setShowSaveModal(true);
                    }}
                    onImportJson={triggerUpload}
                    onDiagramView={() => layout.setShowDiagram(true)}
                    onExportToggle={() => layout.setExportMenuOpen(open => !open)}
                    onExportSvg={() => { exportSvg()}}
                    onExportPng={() => { exportPng()}}
                    onExportJson={() => {exportJson()}}
                    exportMenuOpen={layout.exportMenuOpen}
                    exporting={isExporting} 
                    showDiagnostics={layout.showDiagnostics}
                    showComputePanel={layout.showComputePanel}
                    failureCount={trace.shapeResult?.failures?.length ?? 0}
                    onToggleDiagnostics={() => layout.setShowDiagnostics(v => !v)}
                    onToggleComputePanel={() => layout.setShowComputePanel(v => !v)}
                    onOpenKnowledge={() => setShowKnowledge(true)}
                    hideTrace={isExternal}
                    statusSlot={
                        isExternal ? null : trace.shapeResult && trace.shapeResult.ok ? (
                            <div
                                style={{
                                    display: "inline-flex",
                                    alignItems: "center",
                                    gap: 6,
                                    padding: "4px 10px",
                                    borderRadius: 999,
                                    border: "1px solid #1f2a2f",
                                    background: "linear-gradient(90deg, #0f2d2f, #0b3b2f)",
                                    color: "#7fffd4",
                                    fontWeight: 600,
                                    fontSize: 12,
                                    letterSpacing: "0.01em",
                                    boxShadow: "0 0 0 1px rgba(100, 255, 218, 0.12)",
                                }}
                            >
                                <span aria-hidden="true">✓</span>
                                <span>All clear</span>
                                <span style={{ color: "#a7f3d0", fontWeight: 500 }}>
                                    ({Object.keys(trace.shapeResult.shapes).length} nodes)
                                </span>
                            </div>
                        ) : trace.shapeResult && !trace.shapeResult.ok ? (
                            <span style={{ color: "#f97316", fontWeight: 600 }}>{trace.shapeResult.failures.length} issue(s) detected</span>
                        ) : null
                    }
                    selectionSummary={null}
                />

                <EditorCanvas
                    nodesForFlow={nodesForFlow}
                    highlightedEdges={highlightedEdgesList}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeDragStop={onNodeDragStop}
                    onNodeDragStart={onNodeDragStart}
                    onMainDrop={onMainDrop}
                    onDragOver={onDragOver}
                    onSelectionChange={onSelectionChange}
                    clearSelection={clearSelection}
                    setMainFlowRef={(rf) => { mainFlowRef.current = rf; }}
                />

                {/* 形状追踪不可用提示（需求五.2）：端点缺失/runner 未起时可见，不静默失败 */}
                {!isExternal && trace.traceNotice && (
                    <div
                        style={{
                            position: "absolute",
                            top: 56,
                            left: 12,
                            right: 12,
                            zIndex: 9,
                            display: "flex",
                            alignItems: "center",
                            gap: 10,
                            background: "#3f2d0f",
                            border: "1px solid #d97706",
                            color: "#fde68a",
                            borderRadius: 8,
                            padding: "8px 12px",
                            fontSize: 12,
                            boxShadow: "0 8px 20px rgba(0,0,0,0.35)",
                        }}
                    >
                        <span>⚠ {trace.traceNotice}</span>
                        <button
                            onClick={() => trace.setTraceNotice(null)}
                            style={{
                                marginLeft: "auto",
                                border: "1px solid #d97706",
                                background: "transparent",
                                color: "#fde68a",
                                borderRadius: 6,
                                padding: "2px 10px",
                                fontSize: 12,
                                cursor: "pointer",
                            }}
                        >
                            关闭
                        </button>
                    </div>
                )}

                {onSave && (
                    <div
                        style={{
                            position: "absolute",
                            top: 12,
                            right: 12,
                            zIndex: 10,
                            display: "flex",
                            gap: 8,
                        }}
                    >
                        {projectId && (
                            <>
                                <button
                                    onClick={() => void handleExportCode()}
                                    disabled={exportState === "busy"}
                                    style={{
                                        border: "1px solid #1f2a2f",
                                        borderRadius: 8,
                                        padding: "6px 14px",
                                        fontWeight: 600,
                                        fontSize: 12,
                                        cursor: exportState === "busy" ? "wait" : "pointer",
                                        background: exportState === "error" ? "#7f1d1d" : "#1e293b",
                                        color: "#e2e8f0",
                                    }}
                                >
                                    {exportState === "busy"
                                        ? "导出中…"
                                        : exportState === "error"
                                          ? "导出失败，点击重试"
                                          : "导出代码"}
                                </button>
                                <button
                                    onClick={() => setShowRunPanel(v => !v)}
                                    style={{
                                        border: "1px solid #1f2a2f",
                                        borderRadius: 8,
                                        padding: "6px 14px",
                                        fontWeight: 600,
                                        fontSize: 12,
                                        cursor: "pointer",
                                        background: showRunPanel ? "#6d28d9" : "#7c3aed",
                                        color: "#e2e8f0",
                                    }}
                                >
                                    运行训练
                                </button>
                            </>
                        )}
                        <button
                            onClick={handleSaveGraph}
                            disabled={saveState === "saving"}
                            style={{
                                border: "1px solid #1f2a2f",
                                borderRadius: 8,
                                padding: "6px 14px",
                                fontWeight: 600,
                                fontSize: 12,
                                cursor: saveState === "saving" ? "wait" : "pointer",
                                background: saveState === "error" ? "#7f1d1d" : "#0f766e",
                                color: "#e2e8f0",
                            }}
                        >
                            {saveState === "saving"
                                ? "保存中…"
                                : saveState === "saved"
                                  ? "已保存 ✓"
                                  : saveState === "error"
                                    ? "保存失败，点击重试"
                                    : "保存到项目"}
                        </button>
                    </div>
                )}

                {projectId && onSave && showRunPanel && (
                    <NetworkRunPanel
                        projectId={projectId}
                        saveGraph={async () => {
                            if (onSave) await onSave(buildGraphIR(nodes, edges));
                        }}
                        onClose={() => setShowRunPanel(false)}
                    />
                )}

                {/* Panels & Overlays */}
                {layout.showDiagnostics && trace.shapeResult && !trace.shapeResult.ok && (
                    <DiagnosticsPanel
                        failures={trace.shapeResult.failures}
                        onSelect={(f) => trace.focusFailure(f, setHighlightNodes, setHighlightEdges)}
                        onClose={() => layout.setShowDiagnostics(false)}
                    />
                )}

                {modSys.moduleNameWarning && (
                    <DuplicateModuleWarning onClose={() => modSys.setModuleNameWarning(false)} />
                )}

                {layout.showComputePanel && (
                    <ComputePanel
                        summary={computeSummary}
                        onSelect={node => {
                            setHighlightNodes(new Set([node.nodeId]));
                            setHighlightEdges(new Set());
                            // fitView logic needs ref
                        }}
                        onHover={nodeId => {
                            if (!nodeId) {
                                setHighlightNodes(new Set());
                                setHighlightEdges(new Set());
                                return;
                            }
                            setHighlightNodes(new Set([nodeId]));
                            setHighlightEdges(new Set());
                        }}
                        onClose={() => layout.setShowComputePanel(false)}
                    />
                )}
            </div>

            {!isExternal && (
                <CodePanel
                    showLiveCode={layout.showLiveCode}
                    codePanelWidth={layout.codePanelWidth}
                    dragCodePanel={layout.dragCodePanel}
                    setDragCodePanel={layout.setDragCodePanel}
                    setShowLiveCode={layout.setShowLiveCode}
                    generatedCode={generatedCode}
                    onDownloadCode={onDownloadCode}
                    generated={generated}
                    handleSelectionTargets={({ nodeIds, edgeIds }) => {
                        setHighlightNodes(new Set(nodeIds));
                        setHighlightEdges(new Set(edgeIds));
                    }}
                />
            )}

            {layout.showDiagram && (
                <DiagramView
                    nodes={nodes}
                    edges={edges}
                    graph={{ nodes, edges }}
                    onClose={() => layout.setShowDiagram(false)}
                />
            )}

            {modSys.showSaveModal && (
                <SaveModuleModal
                    onClose={() => modSys.setShowSaveModal(false)}
                    onSave={modSys.handleSaveModule}
                    pendingModuleName={modSys.pendingModuleName}
                    setPendingModuleName={modSys.setPendingModuleName}
                    pendingVariables={modSys.pendingVariables}
                    // setPendingVariables={modSys.setPendingVariables}
                    paramToVariableMap={modSys.paramToVariableMap}
                    // setParamToVariableMap={modSys.setParamToVariableMap}
                    promotableParams={modSys.promotableParams}
                    onAddVariable={modSys.addVariable}
                    onDeleteVariable={modSys.deleteVariable}
                    onRenameVariable={modSys.renameVariable}
                    onUpdateMapping={modSys.updateParamMapping}
                />
            )}

            {modSys.showSaveCopyModal && (
                <SaveCopyModal
                    onClose={() => modSys.setShowSaveCopyModal(false)}
                    onSave={modSys.handleReturnCopyModule}
                    pendingName={modSys.pendingModuleCopyName}
                    setPendingName={modSys.setPendingModuleCopyName}
                />
            )}

            {modSys.openModule && (
                <ModuleEditorOverlay
                    openModule={modSys.openModule}
                    setModuleStack={modSys.setModuleStack}
                    moduleFlowRef={moduleFlowRef}
                    moduleNameInput={modSys.moduleNameInput}
                    setModuleNameInput={modSys.setModuleNameInput}
                    showModuleDiagram={modSys.showModuleDiagram}
                    setShowModuleDiagram={modSys.setShowModuleDiagram}
                    showModuleSaveMenu={modSys.showModuleSaveMenu}
                    setShowModuleSaveMenu={modSys.setShowModuleSaveMenu}
                    moduleNameWarning={modSys.moduleNameWarning}
                    setModuleNameWarning={modSys.setModuleNameWarning}
                    onModuleDrop={onModuleDrop}
                    onDragOver={onDragOver}
                    saveExistingModuleChanges={modSys.saveExistingModuleChanges}
                    saveModuleAsNew={modSys.saveModuleAsNew}
                    onNodeDragStart={onModuleNodeDragStart}
                    onNodeDragStop={onModuleNodeDragStop}
                />
            )}

            {!isExternal && trace.showTrace && (
                <TraceView
                    trace={trace.traceData}
                    loading={trace.traceLoading}
                    error={trace.traceError}
                    shapeComparisons={trace.shapeComparisons}
                    onClose={() => trace.setShowTrace(false)}
                    onSelect={ids => {
                        setHighlightNodes(new Set(ids));
                        setHighlightEdges(new Set());
                    }}
                />
            )}

            {showKnowledge && <KnowledgeSearchPanel onClose={() => setShowKnowledge(false)} />}
        </div>
    );
}

export default function Flow({ initialGraph, onSave, projectId }: FlowEditorProps) {
    return (
        <ReactFlowProvider>
            <FlowContent initialGraph={initialGraph} onSave={onSave} projectId={projectId} />
        </ReactFlowProvider>
    );
}
// function computeContract(selectedIds: Set<string>) {
//     throw new Error("Function not implemented.");
// }
