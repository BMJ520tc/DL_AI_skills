import type { ReactNode } from "react";

type Props = {
    canUndo: boolean;
    canRedo: boolean;
    canSaveModule: boolean;
    traceLoading: boolean;
    traceSeedOptions: string[];
    traceSeedPreset: string;
    traceSeedCustom: string;
    showCustomSeedInput: boolean;
    onUndo: () => void;
    onRedo: () => void;
    onTrace: () => void;
    onTraceSeedPresetChange: (value: string) => void;
    onTraceSeedCustomChange: (value: string) => void;
    onSaveModule: () => void;
    onImportJson: () => void;
    onDiagramView: () => void;
    onExportToggle: () => void;
    onExportSvg: () => void;
    onExportPng: () => void;
    onExportJson: () => void;
    exportMenuOpen: boolean;
    exporting: boolean;
    showDiagnostics: boolean;
    showComputePanel: boolean;
    failureCount: number;
    onToggleDiagnostics: () => void;
    onToggleComputePanel: () => void;
    statusSlot?: ReactNode;
    selectionSummary?: ReactNode;
};

export default function EditorHeader({
    canUndo,
    canRedo,
    canSaveModule,
    traceLoading,
    traceSeedOptions,
    traceSeedPreset,
    traceSeedCustom,
    showCustomSeedInput,
    onUndo,
    onRedo,
    onTrace,
    onTraceSeedPresetChange,
    onTraceSeedCustomChange,
    onSaveModule,
    onImportJson,
    onDiagramView,
    onExportToggle,
    onExportSvg,
    onExportPng,
    onExportJson,
    exportMenuOpen,
    exporting,
    showDiagnostics,
    showComputePanel,
    failureCount,
    onToggleDiagnostics,
    onToggleComputePanel,
    statusSlot,
    selectionSummary,
}: Props) {
    return (
        <div
            style={{
                padding: "8px 12px",
                display: "flex",
                gap: 12,
                alignItems: "center",
                minHeight: "44px",
                justifyContent: "space-between",
                position: "sticky",
                top: 0,
                zIndex: 5,
                background: "#1a1a1a",
            }}
        >
            <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "nowrap" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <span style={{ color: "#6b7280", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                        历史
                    </span>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        <button
                            className="nodrag"
                            onClick={onUndo}
                            disabled={!canUndo}
                            style={{
                                padding: "6px 10px",
                                background: canUndo ? "#333" : "#222",
                                color: canUndo ? "#fff" : "#666",
                                border: "1px solid #444",
                                borderRadius: 6,
                                cursor: canUndo ? "pointer" : "not-allowed",
                            }}
                            title="撤销"
                        >
                            ↶
                        </button>
                        <button
                            className="nodrag"
                            onClick={onRedo}
                            disabled={!canRedo}
                            style={{
                                padding: "6px 10px",
                                background: canRedo ? "#333" : "#222",
                                color: canRedo ? "#fff" : "#666",
                                border: "1px solid #444",
                                borderRadius: 6,
                                cursor: canRedo ? "pointer" : "not-allowed",
                            }}
                            title="重做"
                        >
                            ↷
                        </button>
                    </div>
                </div>
                <div style={{ height: 26, width: 1, background: "#2a2a2a" }} />
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <span style={{ color: "#6b7280", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                        分析
                    </span>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        <button
                            className="nodrag"
                            onClick={onTrace}
                            style={{
                                padding: "6px 10px",
                                background: "#333",
                                color: "#fff",
                                border: "1px solid #444",
                                borderRadius: 6,
                                cursor: "pointer",
                            }}
                            title="运行前向追踪（需要 TorchLens 后端）"
                        >
                            {traceLoading ? "追踪中…" : "张量追踪"}
                        </button>
                        <div
                            className="nodrag"
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 6,
                                padding: "4px 6px",
                                border: "1px solid #2b2b2b",
                                borderRadius: 6,
                            }}
                            title="用于生成追踪虚拟输入的种子"
                        >
                            <span style={{ color: "#cbd5e1", fontSize: 12 }}>种子</span>
                            <select
                                value={traceSeedPreset}
                                onChange={e => onTraceSeedPresetChange(e.target.value)}
                                style={{
                                    background: "#1f2937",
                                    color: "#e5e7eb",
                                    border: "1px solid #374151",
                                    borderRadius: 4,
                                    padding: "2px 6px",
                                    fontSize: 12,
                                    maxWidth: 52,
                                }}
                            >
                                {traceSeedOptions.map(seed => (
                                    <option key={seed} value={seed}>
                                        {seed === "custom" ? "自定义…" : seed}
                                    </option>
                                ))}
                            </select>
                            {showCustomSeedInput ? (
                                <input
                                    type="number"
                                    value={traceSeedCustom}
                                    onChange={e => onTraceSeedCustomChange(e.target.value)}
                                    placeholder="输入种子"
                                    style={{
                                        width: 110,
                                        background: "#111827",
                                        color: "#e5e7eb",
                                        border: "1px solid #374151",
                                        borderRadius: 4,
                                        padding: "2px 6px",
                                        fontSize: 12,
                                    }}
                                />
                            ) : null}
                        </div>
                    </div>
                </div>
                <div style={{ height: 26, width: 1, background: "#2a2a2a" }} />
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <span style={{ color: "#6b7280", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                        模块
                    </span>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        <button
                            className="nodrag"
                            onClick={onSaveModule}
                            disabled={!canSaveModule}
                            style={{
                                padding: "6px 10px",
                                background: canSaveModule ? "#335" : "#222",
                                color: canSaveModule ? "#fff" : "#666",
                                border: "1px solid #444",
                                borderRadius: 6,
                                cursor: canSaveModule ? "pointer" : "not-allowed",
                            }}
                            title="将选中节点保存为可复用模块"
                        >
                            保存模块
                        </button>
                    </div>
                </div>
                <div style={{ height: 26, width: 1, background: "#2a2a2a" }} />
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <span style={{ color: "#6b7280", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                        视图
                    </span>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        <button
                            className="nodrag"
                            onClick={onImportJson}
                            style={{
                                padding: "6px 10px",
                                background: "#333",
                                color: "#fff",
                                border: "1px solid #444",
                                borderRadius: 6,
                                cursor: "pointer",
                            }}
                            title="导入 GraphIR JSON"
                        >
                            导入 JSON
                        </button>
                        <button
                            className="nodrag"
                            onClick={onDiagramView}
                            style={{
                                padding: "6px 10px",
                                background: "#333",
                                color: "#fff",
                                border: "1px solid #444",
                                borderRadius: 6,
                                cursor: "pointer",
                            }}
                            title="打开论文风格结构图视图"
                        >
                            结构图视图
                        </button>
                        <div style={{ position: "relative" }}>
                            <button
                                className="nodrag"
                                onClick={onExportToggle}
                                style={{
                                    padding: "6px 10px",
                                    background: "#333",
                                    color: "#fff",
                                    border: "1px solid #444",
                                    borderRadius: 6,
                                    cursor: "pointer",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 6,
                                }}
                            >
                                导出 <span style={{ fontSize: 10 }}>▼</span>
                            </button>
                            {exportMenuOpen && (
                                <div
                                    style={{
                                        position: "absolute",
                                        right: 0,
                                        top: "110%",
                                        background: "#1e1e1e",
                                        border: "1px solid #333",
                                        borderRadius: 8,
                                        minWidth: 160,
                                        zIndex: 10,
                                        overflow: "hidden",
                                    }}
                                >
                                    <button
                                        onClick={onExportSvg}
                                        disabled={exporting}
                                        style={{
                                            padding: "8px 12px",
                                            width: "100%",
                                            background: "transparent",
                                            border: "none",
                                            color: exporting ? "#777" : "#e6edf3",
                                            cursor: exporting ? "not-allowed" : "pointer",
                                            textAlign: "left",
                                        }}
                                    >
                                        导出 SVG
                                    </button>
                                    <button
                                        onClick={onExportPng}
                                        disabled={exporting}
                                        style={{
                                            padding: "8px 12px",
                                            width: "100%",
                                            background: "transparent",
                                            border: "none",
                                            color: exporting ? "#777" : "#e6edf3",
                                            cursor: exporting ? "not-allowed" : "pointer",
                                            textAlign: "left",
                                            borderTop: "1px solid #333",
                                        }}
                                    >
                                        导出 PNG
                                    </button>
                                    <button
                                        onClick={onExportJson}
                                        style={{
                                            padding: "8px 12px",
                                            width: "100%",
                                            background: "transparent",
                                            border: "none",
                                            color: "#e6edf3",
                                            cursor: "pointer",
                                            textAlign: "left",
                                            borderTop: "1px solid #333",
                                        }}
                                    >
                                        导出 JSON
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4, minWidth: 260 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "nowrap", overflow: "hidden" }}>
                    {statusSlot ? (
                        <div
                            style={{
                                flex: "1 1 auto",
                                minWidth: 0,
                                whiteSpace: "nowrap",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                display: "flex",
                                alignItems: "center",
                            }}
                        >
                            {statusSlot}
                        </div>
                    ) : null}
                    <button
                        onClick={onToggleComputePanel}
                        style={{
                            padding: "4px 8px",
                            borderRadius: 6,
                            border: "1px solid #3f3f46",
                            background: showComputePanel ? "#1f1f1f" : "#111318",
                            color: "#e6edf3",
                            cursor: "pointer",
                            fontSize: 12,
                            flexShrink: 0,
                        }}
                    >
                        {showComputePanel ? "隐藏计算量" : "查看计算量"}
                    </button>
                    {failureCount > 0 && (
                        <button
                            onClick={onToggleDiagnostics}
                            style={{
                                padding: "4px 8px",
                                borderRadius: 6,
                                border: "1px solid #3f3f46",
                                background: showDiagnostics ? "#1f1f1f" : "#111318",
                                color: "#e6edf3",
                                cursor: "pointer",
                                fontSize: 12,
                                flexShrink: 0,
                            }}
                        >
                            {showDiagnostics ? "隐藏诊断" : "查看诊断"}
                        </button>
                    )}
                </div>
                {selectionSummary ? (
                    <div
                        style={{
                            maxWidth: 360,
                            maxHeight: 36,
                            overflowY: "auto",
                        }}
                    >
                        {selectionSummary}
                    </div>
                ) : null}
            </div>
        </div>
    );
}
