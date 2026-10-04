import { type Edge, type Node, useReactFlow } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildGraphIR } from "../../../utils/graphIR";
import { runTorchLensTrace, TRACE_UNAVAILABLE_REASON } from "../../../utils/traceService";
import { buildShapeComparisons, compareTraceShapes } from "../../../utils/traceAnalysis";
import { LAYER_REGISTRY } from "../../../types/nodeTypes";
import { verifyShapes, type ShapeFailure, type ShapeResult } from "../../../utils/shape_verifier";
import { verifyIRShapes } from "../../../utils/irShapeVerifier";
import type { TraceResponse } from "../../../types/trace";
import { getModule } from "../../../utils/moduleRegistry";

// Move comparison logic here or keep in utils?
// FlowEditor line 140: buildShapeComparisons(traceData, shapeResult ...)

/**
 * 解析工具栏选择的追踪种子：预设值直接用；选「自定义」时用输入框内容。
 * 非法/空/NaN → undefined（请求体不带 seed 字段，交给后端默认值）。
 * 纯函数，供界面与自检脚本共用口径。
 */
export function resolveTraceSeed(preset: string, custom: string): number | undefined {
    const raw = preset === "custom" ? custom : preset;
    if (raw === null || raw === undefined) return undefined;
    const trimmed = String(raw).trim();
    if (!trimmed) return undefined;
    const value = Number(trimmed);
    return Number.isFinite(value) ? value : undefined;
}

type UseTraceSystemProps = {
    /** 代码生成/沙盒追踪用的节点与边（外部画布模式下为空图，避免对 ir 节点出码）。 */
    nodes: Node[];
    edges: Edge[];
    setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
    generatedCode: string;
    /** 结构化画布（IR 图）模式：形状校验改用 IR 形状字段判据（verifyIRShapes）。 */
    irShapeMode?: boolean;
    /** 形状校验/追踪真正使用的节点与边。外部画布下 nodes/edges 是空图（仅让前端出码空转），
     *  形状校验与追踪必须吃真实节点/边，否则 verifyShapes([], []) 恒通过（需求五.1）。 */
    shapeNodes?: Node[];
    shapeEdges?: Edge[];
};

export function useTraceSystem({
    nodes,
    edges,
    setNodes,
    generatedCode,
    irShapeMode = false,
    shapeNodes,
    shapeEdges,
}: UseTraceSystemProps) {
    const { fitView } = useReactFlow();
    const [showTrace, setShowTrace] = useState(false);
    const [traceData, setTraceData] = useState<TraceResponse | null>(null);
    const [traceLoading, setTraceLoading] = useState(false);
    const [traceError, setTraceError] = useState<string | null>(null);
    // 追踪端点不可用时的可见提示（需求五.2）；null 表示正常。
    const [traceNotice, setTraceNotice] = useState<string | null>(null);
    const [traceSeedPreset, setTraceSeedPreset] = useState("42");
    const [traceSeedCustom, setTraceSeedCustom] = useState("");

    // 形状校验/追踪的数据源：默认为传入的 nodes/edges（沙盒编辑器行为不变），
    // 外部画布模式由 FlowEditor 显式传入真实节点/边。
    const graphNodes = shapeNodes ?? nodes;
    const graphEdges = shapeEdges ?? edges;

    // Shape Verification Logic
    const [shapeResult, setShapeResult] = useState<ShapeResult | null>(null);
    // IR 图（结构化画布）：按 __out_shape ↔ __in_shape 判连线一致性；缺形状只计数不算失败。
    const irVerification = useMemo(
        () => (irShapeMode ? verifyIRShapes(graphNodes, graphEdges) : null),
        [irShapeMode, graphNodes, graphEdges],
    );
    const layerVerification = useMemo(
        () => (irShapeMode ? null : verifyShapes(graphNodes, graphEdges, LAYER_REGISTRY)),
        [irShapeMode, graphNodes, graphEdges],
    );
    const verificationResult: ShapeResult | null = irVerification ?? layerVerification;
    const shapeMissing = useMemo(() => irVerification?.missingShapes ?? [], [irVerification]);

    useEffect(() => {
        if (!verificationResult) return;
        setShapeResult(prev => {
            const prevStr = JSON.stringify(prev);
            const nextStr = JSON.stringify(verificationResult);
            return prevStr === nextStr ? prev : verificationResult;
        });
        if (!verificationResult.ok) {
            console.warn("Shape validation Failures:", verificationResult.failures);
        }
    }, [verificationResult]);

    const getTraceInputShapes = useCallback((): number[][] => {
        const isValidShape = (vals: number[]) => vals.length > 0 && vals.every(v => Number.isFinite(v) && v > 0);
        const readDimsFromData = (data: unknown): number[] | null => {
            const bag = (data && typeof data === "object") ? (data as Record<string, unknown>) : null;
            const liveShape = bag && Array.isArray(bag.__shape) ? (bag.__shape as number[]) : null;
            if (liveShape && isValidShape(liveShape)) return liveShape;
            const dims = bag && Array.isArray(bag.dims) ? bag.dims : [];
            if (dims.length) {
                const parsed: number[] = dims.map((d: { size?: unknown }) => Number(d?.size));
                if (isValidShape(parsed)) return parsed;
            }
            return null;
        };

        const shapes: number[][] = [];
        const inputNodes = graphNodes.filter(n => n.type === "input_layer");
        for (const node of inputNodes) {
            const data: Record<string, unknown> = (node.data && typeof node.data === "object") ? node.data : {};
            const shape = readDimsFromData(data);
            if (shape) {
                shapes.push(shape);
                continue;
            }
            const inferred = shapeResult?.shapes?.[node.id]?.defaultShape;
            if (Array.isArray(inferred) && inferred.length) shapes.push(inferred);
        }

        if (shapes.length) return shapes;

        // If no root Input nodes, infer from root-level nodes (e.g., module_ref).
        const incomingCount = new Map<string, number>();
        graphEdges.forEach(e => {
            incomingCount.set(e.target, (incomingCount.get(e.target) || 0) + 1);
        });
        const rootNodes = graphNodes.filter(n => (incomingCount.get(n.id) || 0) === 0);
        for (const node of rootNodes) {
            if (node.type === "input_layer") {
                const shape = readDimsFromData(node.data);
                if (shape) shapes.push(shape);
                continue;
            }
            if (node.type === "module_ref") {
                // moduleId 由 getInitialNodeData 写成字符串；非字符串视为未设置。
                const rawModuleId = node.data?.moduleId;
                const moduleId = typeof rawModuleId === "string" ? rawModuleId : undefined;
                const mod = moduleId ? getModule(moduleId) : undefined;
                const internal = mod?.internalNodes || [];
                const internalInputNodes = internal.filter(n => n.type === "input_layer");
                // Use internal input dims in their visual order.
                for (const inner of internalInputNodes) {
                    const shape = readDimsFromData(inner.data);
                    if (shape) shapes.push(shape);
                }
                if (internalInputNodes.length) continue;
            }
            const inferred = shapeResult?.shapes?.[node.id]?.defaultShape;
            if (Array.isArray(inferred) && inferred.length) shapes.push(inferred);
        }

        return shapes.length ? shapes : [[1, 3, 224, 224]];
    }, [graphNodes, graphEdges, shapeResult]);

    // Apply calculated shapes to nodes
    const shapeResultRef = useRef<string>("");
    useEffect(() => {
        if (!shapeResult?.shapes) return;
        // 本轮没有任何形状结果（如外部画布模式传入空 nodes/edges）时不清空已有 __shape
        if (Object.keys(shapeResult.shapes).length === 0) return;
        const currentShapesStr = JSON.stringify(shapeResult.shapes);
        if (shapeResultRef.current === currentShapesStr) return;

        setNodes(currentNodes => {
            const deepEqual = (a: unknown, b: unknown): boolean => {
                if (a === b) return true;
                if (!Array.isArray(a) || !Array.isArray(b)) return false;
                if (a.length !== b.length) return false;
                for (let i = 0; i < a.length; i++) {
                    if (Array.isArray(a[i]) && Array.isArray(b[i])) {
                        if (!deepEqual(a[i], b[i])) return false;
                    } else if (a[i] !== b[i]) {
                        return false;
                    }
                }
                return true;
            };
            let hasChanges = false;
            const nextNodes = currentNodes.map(n => {
                const shapeEntry = shapeResult.shapes[n.id];
                // 该节点本轮无形状结果：保留原有 __shape，绝不写入 undefined
                if (!shapeEntry) return n;
                const newShapeArray = shapeEntry.defaultShape;
                const currentShapeArray =
                    n.data && typeof n.data === "object" ? (n.data as { __shape?: number[] }).__shape : undefined;
                const isSame = deepEqual(currentShapeArray, newShapeArray);
                if (isSame) return n;
                hasChanges = true;
                return {
                    ...n,
                    data: { ...n.data, __shape: newShapeArray }
                };
            });
            if (hasChanges) {
                shapeResultRef.current = currentShapesStr;
                return nextNodes;
            }
            return currentNodes;
        });
    }, [shapeResult, setNodes]);

    // Trace Logic
    const handleTrace = useCallback(async () => {
        setTraceLoading(true);
        setTraceError(null);
        setTraceNotice(null);
        try {
            const graph = buildGraphIR(graphNodes, graphEdges);
            // 种子：工具栏选择必须进请求体（需求五.1 种子控件不能是死控件）；
            // 非法/未选择时 resolveTraceSeed 返回 undefined，此时不带该字段。
            const seed = resolveTraceSeed(traceSeedPreset, traceSeedCustom);
            const resp = await runTorchLensTrace({
                graph,
                inputShapes: getTraceInputShapes(),
                code: generatedCode,
                ...(seed === undefined ? {} : { seed }),
            });
            // 端点缺失 / 独立 runner 不可用：给出可见提示，但不阻断画布其它功能
            if (resp.unavailable) {
                setTraceNotice(resp.unavailableReason ?? TRACE_UNAVAILABLE_REASON);
            }
            const shapeWarnings = compareTraceShapes(resp, shapeResult, graphEdges, graphNodes, LAYER_REGISTRY);
            setTraceData({
                ...resp,
                warnings: [...(resp.warnings || []), ...shapeWarnings],
            });
            setShowTrace(true);
        } catch (err) {
            setTraceError("Trace failed. Backend unavailable or returned error.");
            setTraceNotice(TRACE_UNAVAILABLE_REASON);
            console.error("Trace failed", err);
        } finally {
            setTraceLoading(false);
        }
    }, [graphNodes, graphEdges, generatedCode, shapeResult, getTraceInputShapes, traceSeedPreset, traceSeedCustom]);

    const shapeComparisons = useMemo(
        () => (traceData ? buildShapeComparisons(traceData, shapeResult, graphEdges, graphNodes, LAYER_REGISTRY) : []),
        [traceData, shapeResult, graphEdges, graphNodes]
    );

    // Error Decoration on Edges
    const friendlyError = useCallback((failure: ShapeFailure) => {
        const label = failure.label || failure.nodeType || failure.nodeId;
        const inputs =
            failure.inputShapes && failure.inputShapes.length
                ? ` | inputs: ${failure.inputShapes.map(s => `[${s.join(",")}]`).join(", ")}`
                : "";
        const upstream = failure.upstream && failure.upstream.length ? ` | from: ${failure.upstream.join(", ")}` : "";
        const hint = ` | fix: adjust ${label} params or ensure upstream nodes output the expected shape`;
        return `${label}: ${failure.error}${inputs}${upstream}${hint}`;
    }, []);

    // Helper to decorate edges with error messages
    const getDecoratedEdges = useCallback((currentEdges: Edge[]) => {
        if (!shapeResult || shapeResult.ok) return currentEdges;

        // 1. Build a map of errors per node
        const failMap = new Map<string, ShapeFailure[]>();
        shapeResult.failures.forEach(f => {
            (f.upstream || []).forEach(src => {
                const key = `${src}->${f.nodeId}`;
                const arr = failMap.get(key) || [];
                arr.push(f);
                failMap.set(key, arr);
            });
        });

        // 2. Map existing edges to add/remove error data
        // We do NOT use edgesWithHandlers here because that is for the deletion logic. 
        // We just return the data needed. `useGraphState` handles the delete logic. 
        // We should merge them in the component? Or can we merge them here?
        // Ideally we return a decorator function or the specific error data.
        return currentEdges.map(e => {
            const key = `${e.source}->${e.target}`;
            const errs = failMap.get(key);
            if (!errs || !errs.length) return e;
            // If we return 'e', we keep old data. But if errors are GONE?
            // The logic above: if !errs return e. This presumes e doesn't have old errors?
            // If e had errors and now doesn't, we should clear them.
            // FlowEditor line 817: if (!errs) return e implies we don't clear?
            // Actually ReactFlow updates edges completely.
            // But if we return 'e' it has whatever data it had.
            // Wait, this function transforms the base edges. Base edges usually don't have transient error data unless we persisted it?
            // In FlowEditor `edges` state DOES NOT have error data. `decoratedEdges` MEMO is computed from `edgesWithHandlers`.

            const existingData = (e.data && typeof e.data === "object") ? e.data as Record<string, unknown> : {};
            return {
                ...e,
                type: "custom",
                data: {
                    ...existingData,
                    error: errs.map(friendlyError).join("\n"),
                },
            };
        });
    }, [shapeResult, friendlyError]);

    // Focus Failure
    const focusFailure = useCallback(
        (failure: ShapeFailure, setHighlightNodes: (s: Set<string>) => void, setHighlightEdges: (s: Set<string>) => void) => {
            const upstream = failure.upstream || [];
            const edgeIds = graphEdges
                .filter(e => upstream.includes(e.source) && e.target === failure.nodeId)
                .map(e => e.id);
            setHighlightNodes(new Set([failure.nodeId, ...upstream]));
            setHighlightEdges(new Set(edgeIds));
            const target = graphNodes.find(n => n.id === failure.nodeId);
            if (target) {
                void fitView({ nodes: [target], padding: 0.4 });
            }
        },
        [graphEdges, graphNodes, fitView]
    );

    return {
        showTrace, setShowTrace,
        traceData, setTraceData,
        traceLoading, setTraceLoading,
        traceError, setTraceError,
        traceNotice, setTraceNotice,
        traceSeedPreset, setTraceSeedPreset,
        traceSeedCustom, setTraceSeedCustom,
        shapeResult,
        /** 缺形状的节点 id（IR 图口径：非 op 节点 in/out 任缺其一）；沙盒模式恒为空。 */
        shapeMissing,
        handleTrace,
        shapeComparisons,
        getDecoratedEdges,
        focusFailure
    };
}
