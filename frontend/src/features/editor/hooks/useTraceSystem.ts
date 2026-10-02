import { type Edge, type Node, useReactFlow } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildGraphIR } from "../../../utils/graphIR";
import { runTorchLensTrace, TRACE_UNAVAILABLE_REASON } from "../../../utils/traceService";
import { buildShapeComparisons, compareTraceShapes } from "../../../utils/traceAnalysis";
import { LAYER_REGISTRY } from "../../../types/nodeTypes";
import { verifyShapes, type ShapeFailure, type ShapeResult } from "../../../utils/shape_verifier";
import type { TraceResponse } from "../../../types/trace";
import { getModule } from "../../../utils/moduleRegistry";

// Move comparison logic here or keep in utils?
// FlowEditor line 140: buildShapeComparisons(traceData, shapeResult ...)

type UseTraceSystemProps = {
    nodes: Node[];
    edges: Edge[];
    setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
    generatedCode: string;
};

export function useTraceSystem({ nodes, edges, setNodes, generatedCode }: UseTraceSystemProps) {
    const { fitView } = useReactFlow();
    const [showTrace, setShowTrace] = useState(false);
    const [traceData, setTraceData] = useState<TraceResponse | null>(null);
    const [traceLoading, setTraceLoading] = useState(false);
    const [traceError, setTraceError] = useState<string | null>(null);
    // 追踪端点不可用时的可见提示（需求五.2）；null 表示正常。
    const [traceNotice, setTraceNotice] = useState<string | null>(null);
    const [traceSeedPreset, setTraceSeedPreset] = useState("42");
    const [traceSeedCustom, setTraceSeedCustom] = useState("");

    // Shape Verification Logic
    const [shapeResult, setShapeResult] = useState<ShapeResult | null>(null);
    const verificationResult = useMemo(() => {
        return verifyShapes(nodes, edges, LAYER_REGISTRY);
    }, [nodes, edges]);

    useEffect(() => {
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
        const inputNodes = nodes.filter(n => n.type === "input_layer");
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
        edges.forEach(e => {
            incomingCount.set(e.target, (incomingCount.get(e.target) || 0) + 1);
        });
        const rootNodes = nodes.filter(n => (incomingCount.get(n.id) || 0) === 0);
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
    }, [nodes, edges, shapeResult]);

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
            const graph = buildGraphIR(nodes, edges);
            const resp = await runTorchLensTrace({
                graph,
                inputShapes: getTraceInputShapes(),
                code: generatedCode,
            });
            // 端点缺失 / 独立 runner 不可用：给出可见提示，但不阻断画布其它功能
            if (resp.unavailable) {
                setTraceNotice(resp.unavailableReason ?? TRACE_UNAVAILABLE_REASON);
            }
            const shapeWarnings = compareTraceShapes(resp, shapeResult, edges, nodes, LAYER_REGISTRY);
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
    }, [nodes, edges, generatedCode, shapeResult, getTraceInputShapes]);

    const shapeComparisons = useMemo(
        () => (traceData ? buildShapeComparisons(traceData, shapeResult, edges, nodes, LAYER_REGISTRY) : []),
        [traceData, shapeResult, edges, nodes]
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
            const edgeIds = edges
                .filter(e => upstream.includes(e.source) && e.target === failure.nodeId)
                .map(e => e.id);
            setHighlightNodes(new Set([failure.nodeId, ...upstream]));
            setHighlightEdges(new Set(edgeIds));
            const target = nodes.find(n => n.id === failure.nodeId);
            if (target) {
                void fitView({ nodes: [target], padding: 0.4 });
            }
        },
        [edges, nodes, fitView]
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
        handleTrace,
        shapeComparisons,
        getDecoratedEdges,
        focusFailure
    };
}
