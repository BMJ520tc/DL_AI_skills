/* eslint-disable react-refresh/only-export-components */
// nodes/IrNode.tsx — 结构化项目画布节点（模块四 B3，GraphIR type="ir"）。
// 与 LAYER_REGISTRY 图层不同：参数 schema 不固定，来自节点 data.params（IR 调参）。
// 句柄从 data.handles 动态渲染（后端 ir_to_graphir / 前端 irAdapter.graphIRToFlow 注入）。

import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useState } from "react";
import type { GraphHandle } from "../types/graph";

const KIND_COLORS: Record<string, string> = {
    module: "#3b82f6",
    container: "#8b5cf6",
    leaf: "#10b981",
    op: "#f59e0b",
};

const KIND_LABELS: Record<string, string> = {
    module: "模块",
    container: "容器",
    leaf: "叶子层",
    op: "算子",
};

export type IrNodeData = {
    kind?: string;
    class_name?: string;
    module_path?: string | null;
    params?: Record<string, unknown>;
    ir_id?: string;
    input_shape?: number[] | null;
    output_shape?: number[] | null;
    __shape?: number[] | null;
    handles?: GraphHandle[];
    layout_hint?: { width?: number; height?: number };
    label?: string;
    [key: string]: unknown;
};

function shapeText(shape: number[] | null | undefined): string {
    return shape && shape.length ? `[${shape.join(", ")}]` : "?";
}

/** 单个参数字段：按原值类型解析（number/boolean/JSON），编辑即写回节点 data。 */
function ParamField({
    name,
    value,
    onCommit,
    onRemove,
}: {
    name: string;
    value: unknown;
    onCommit: (name: string, v: unknown) => void;
    onRemove: (name: string) => void;
}) {
    const [draft, setDraft] = useState(() =>
        value == null ? "" : typeof value === "object" ? JSON.stringify(value) : String(value)
    );

    const commit = (next: string) => {
        setDraft(next);
        if (typeof value === "number") {
            const num = Number(next);
            if (next.trim() !== "" && !Number.isNaN(num)) onCommit(name, num);
        } else if (typeof value === "boolean") {
            onCommit(name, next === "true");
        } else if (value !== null && typeof value === "object") {
            try {
                onCommit(name, JSON.parse(next));
            } catch {
                // JSON 输入中，不提交
            }
        } else {
            onCommit(name, next);
        }
    };

    return (
        <div style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11 }}>
            <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "#94a3b8" }} title={name}>
                {name}
            </span>
            <input
                type={typeof value === "boolean" ? "checkbox" : "text"}
                checked={typeof value === "boolean" ? value === true : undefined}
                value={typeof value === "boolean" ? undefined : draft}
                onChange={e =>
                    typeof value === "boolean" ? onCommit(name, e.target.checked) : commit(e.target.value)
                }
                style={{
                    width: 96,
                    border: "1px solid #334155",
                    background: "#0f172a",
                    color: "#e2e8f0",
                    borderRadius: 4,
                    padding: "1px 4px",
                    fontFamily: "monospace",
                }}
            />
            <button
                title="删除参数"
                onClick={() => onRemove(name)}
                style={{
                    border: "none",
                    background: "transparent",
                    color: "#64748b",
                    cursor: "pointer",
                    padding: 0,
                    fontSize: 11,
                }}
            >
                ×
            </button>
        </div>
    );
}

export default function IrNode({ id, data, selected }: NodeProps<Node<IrNodeData>>) {
    const { setNodes } = useReactFlow();
    const d = data as IrNodeData;

    const handles = d.handles ?? [];
    const inputs = handles.filter(h => h.kind === "input").sort((a, b) => a.order - b.order);
    const outputs = handles.filter(h => h.kind === "output").sort((a, b) => a.order - b.order);

    const hint = d.layout_hint;
    const color = KIND_COLORS[d.kind ?? "module"] ?? "#64748b";
    const params = d.params ?? {};
    const entries = Object.entries(params);

    const updateParams = (next: Record<string, unknown>) =>
        setNodes(nds =>
            nds.map(n => (n.id === id ? { ...n, data: { ...(n.data || {}), params: next } } : n))
        );

    const commitParam = (name: string, v: unknown) =>
        updateParams({ ...params, [name]: v });

    const removeParam = (name: string) =>
        updateParams(Object.fromEntries(entries.filter(([k]) => k !== name)));

    return (
        <div
            style={{
                minWidth: hint?.width ?? 220,
                minHeight: hint?.height ?? undefined,
                background: selected ? "#1e293b" : "#0f172a",
                border: `1.5px solid ${selected ? color : "#334155"}`,
                borderRadius: 8,
                padding: "8px 10px",
                fontSize: 12,
                color: "#e2e8f0",
                boxSizing: "border-box",
            }}
        >
            {inputs.map(h => (
                <Handle key={h.id} id={h.id} type="target" position={Position.Left} style={{ background: color, width: 8, height: 8 }} />
            ))}
            {outputs.map(h => (
                <Handle key={h.id} id={h.id} type="source" position={Position.Right} style={{ background: color, width: 8, height: 8 }} />
            ))}

            {/* 头部：类别徽标 + 类名 */}
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: entries.length ? 6 : 0 }}>
                <span
                    style={{
                        background: color,
                        color: "#0b1220",
                        fontWeight: 700,
                        fontSize: 10,
                        borderRadius: 999,
                        padding: "1px 7px",
                        whiteSpace: "nowrap",
                    }}
                >
                    {KIND_LABELS[d.kind ?? ""] ?? d.kind}
                </span>
                <span style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {d.class_name ?? d.label ?? id}
                </span>
            </div>

            {/* 形状：优先展示 in → out（IR 层级），否则回落到 __shape（图层/画布推断） */}
            <div style={{ color: "#64748b", fontSize: 10, fontFamily: "monospace", marginBottom: entries.length ? 6 : 0 }}>
                {d.input_shape || d.output_shape
                    ? `in ${shapeText(d.input_shape)} → out ${shapeText(d.output_shape)}`
                    : d.__shape && d.__shape.length
                      ? shapeText(d.__shape)
                      : "未知"}
                {d.module_path ? ` · ${d.module_path}` : ""}
            </div>

            {/* 参数（动态 schema，编辑即写回 data.params，保存画布时随 GraphIR 落库） */}
            {entries.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    {entries.map(([k, v]) => (
                        <ParamField key={k} name={k} value={v} onCommit={commitParam} onRemove={removeParam} />
                    ))}
                </div>
            )}
        </div>
    );
}
