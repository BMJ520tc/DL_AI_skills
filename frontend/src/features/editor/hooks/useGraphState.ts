import {
    applyEdgeChanges,
    applyNodeChanges,
    type Edge,
    type Node,
    type OnEdgesChange,
    type OnNodesChange,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GraphIR } from "../../../types/graph";
import { applyGraphIR, buildGraphIR } from "../../../utils/graphIR";
import { graphIRToFlow } from "../../../utils/irAdapter";
import { getId, syncIdFromNodes } from "../utils/idUtils";

// ---------------------------------------------------------------------------
// 节点复制（需求五.1「在节点上直接编辑：改参数、复制、删除、成组」）
// 纯函数放在 hook 外部：不依赖 React 状态，便于自检脚本直接核对边重映射。
// ---------------------------------------------------------------------------

/** data/边数据的深拷贝：数组与纯对象递归复制，基本类型与函数按引用保留。 */
export function deepCopyValue<T>(value: T): T {
    if (Array.isArray(value)) {
        return value.map(item => deepCopyValue(item)) as unknown as T;
    }
    if (value && typeof value === "object") {
        const out: Record<string, unknown> = {};
        for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
            out[key] = deepCopyValue(item);
        }
        return out as T;
    }
    return value;
}

export type DuplicateResult = { nodes: Node[]; edges: Edge[]; newIds: string[] };

/**
 * 复制选中节点的纯函数实现（新 id 由调用方注入，便于自检）：
 *   - 只复制 selectedIds 中的节点；空选区返回 null（调用方据此禁用/不动作）；
 *   - 新 id 一律由 newId() 生成，位置在原节点基础上偏移 offset（默认 +40/+40）；
 *   - data 深拷贝，复制节点与原节点不共享参数对象；
 *   - 连线只在两端都被复制时一并复制；跨出选区的边不复制；
 *   - 复制后选中新节点（新节点 selected=true，原节点取消选中）。
 */
export function duplicateSelection(
    nodes: Node[],
    edges: Edge[],
    selectedIds: Iterable<string>,
    newId: () => string,
    offset: { x: number; y: number } = { x: 40, y: 40 },
): DuplicateResult | null {
    const selected = new Set(selectedIds);
    const originals = nodes.filter(n => selected.has(n.id));
    if (!originals.length) return null;

    const idMap = new Map<string, string>();
    const copies: Node[] = originals.map(node => {
        const id = newId();
        idMap.set(node.id, id);
        const position = node.position ?? { x: 0, y: 0 };
        return {
            ...node,
            id,
            position: { x: position.x + offset.x, y: position.y + offset.y },
            selected: true,
            data: deepCopyValue(node.data ?? {}),
        };
    });

    const copiedEdges: Edge[] = edges
        .filter(edge => idMap.has(edge.source) && idMap.has(edge.target))
        .map(edge => ({
            ...edge,
            id: newId(),
            source: idMap.get(edge.source)!,
            target: idMap.get(edge.target)!,
            selected: false,
            data: deepCopyValue(edge.data ?? {}),
        }));

    return {
        nodes: [...nodes.map(n => (selected.has(n.id) ? { ...n, selected: false } : n)), ...copies],
        edges: [...edges, ...copiedEdges],
        newIds: copies.map(c => c.id),
    };
}

/** initialGraph 传入时进入外部画布模式（模块四 B3）：从 GraphIR 快照初始化、
 *  不再读写 localStorage（避免结构化项目画布污染沙盒编辑器的本地存档）。 */
export function useGraphState(initialGraph?: GraphIR | null) {
    // Frozen at mount (FlowEditor remounts via key={projectId}), so a plain state
    // value expresses the same lifetime as the previous ref without reading a ref
    // during render. Never updated: the canvas mode is decided once per mount.
    const [external] = useState(() => !!initialGraph);

    // -------------------------------------------------------------------------
    // 1. Storage & Initialization
    // -------------------------------------------------------------------------
    const [nodes, setNodes] = useState<Node[]>(() => {
        if (external && initialGraph) {
            const restored = graphIRToFlow(initialGraph);
            syncIdFromNodes(restored.nodes);
            return restored.nodes;
        }
        const savedGraph = localStorage.getItem("graphIR");
        if (savedGraph) {
            try {
                const parsed: GraphIR = JSON.parse(savedGraph);
                const restored = applyGraphIR(parsed);
                syncIdFromNodes(restored.nodes);
                return restored.nodes.map(n => (n.type === "input" ? { ...n, type: "input_layer" } : n));
            } catch (err) {
                console.warn("Failed to load GraphIR, falling back to nodes/edges", err);
            }
        }
        const saved = localStorage.getItem("nodes");
        if (!saved) return [];
        const parsed: Node[] = JSON.parse(saved).map((n: Node) =>
            n.type === "input" ? { ...n, type: "input_layer" } : n
        );
        syncIdFromNodes(parsed);
        return parsed;
    });

    const [edges, setEdges] = useState<Edge[]>(() => {
        if (external && initialGraph) {
            return graphIRToFlow(initialGraph).edges;
        }
        const savedGraph = localStorage.getItem("graphIR");
        if (savedGraph) {
            try {
                const parsed: GraphIR = JSON.parse(savedGraph);
                const restored = applyGraphIR(parsed);
                return restored.edges;
            } catch (err) {
                console.warn("Failed to load GraphIR edges, falling back to edges", err);
            }
        }
        const saved = localStorage.getItem("edges");
        return saved ? JSON.parse(saved) : [];
    });

    // -------------------------------------------------------------------------
    // 2. React Flow Callbacks
    // -------------------------------------------------------------------------
    const onNodesChange: OnNodesChange = useCallback(
        changes => {
            // Drop only edges attached to nodes being removed so unrelated wiring stays intact.
            const removedIds = changes.filter(c => c.type === "remove").map(c => c.id);
            if (removedIds.length) {
                setEdges(eds => eds.filter(e => !removedIds.includes(e.source) && !removedIds.includes(e.target)));
            }
            setNodes(nds => applyNodeChanges(changes, nds));
        },
        [setNodes, setEdges]
    );

    const onEdgesChange: OnEdgesChange = useCallback(
        changes => setEdges(eds => applyEdgeChanges(changes, eds)),
        [setEdges]
    );

    /** 复制当前选中的节点（工具栏「复制」按钮 / Ctrl+D 共用）；无选中返回 false。
     *  与其它编辑动作一样走 setNodes/setEdges，历史快照/撤销因此照常工作。 */
    const duplicateSelected = useCallback((): boolean => {
        const selectedIds = nodes.filter(n => n.selected).map(n => n.id);
        const result = duplicateSelection(nodes, edges, selectedIds, getId);
        if (!result) return false;
        setNodes(result.nodes);
        setEdges(result.edges);
        return true;
    }, [nodes, edges, setNodes, setEdges]);

    const deleteEdgeById = useCallback((edgeId: string) => {
        setEdges(eds => eds.filter(e => e.id !== edgeId));
    }, [setEdges]);

    // Attach helper callbacks to edges so custom edge UI can remove them cleanly.
    // NOTE: This might cause re-renders if not handled carefully, but it copies original logic.
    const edgesWithHandlers = useMemo(() => {
        return edges.map(e => ({
            ...e,
            data: {
                ...(typeof e.data === "object" && e.data !== null ? e.data : {}),
                onDelete: deleteEdgeById,
            },
        }));
    }, [edges, deleteEdgeById]);

    // -------------------------------------------------------------------------
    // 3. History (Undo/Redo)
    // -------------------------------------------------------------------------
    const historyRef = useRef<Array<{ nodes: Node[]; edges: Edge[] }>>([]);
    const historyIndexRef = useRef(0);
    const [canUndo, setCanUndo] = useState(false);
    const [canRedo, setCanRedo] = useState(false);
    const isRestoring = useRef(false);
    const skipHistory = useRef(false);

    const cloneSnapshot = useCallback((n: Node[], e: Edge[]) => {
        const copyNodes = n.map(node => ({
            ...node,
            data: node.data ? { ...node.data } : {},
            position: { ...node.position }
        }));
        const copyEdges = e.map(edge => ({ ...edge, data: edge.data ? { ...edge.data } : {} }));
        return { nodes: copyNodes, edges: copyEdges };
    }, []);

    const applySnapshot = useCallback((snapshot: { nodes: Node[]; edges: Edge[] }) => {
        isRestoring.current = true;
        setNodes(snapshot.nodes);
        setEdges(snapshot.edges);
        syncIdFromNodes(snapshot.nodes);
    }, []);

    const handleUndo = useCallback(() => {
        if (!canUndo) return;
        const targetIndex = Math.max(0, historyIndexRef.current - 1);
        historyIndexRef.current = targetIndex;
        const snapshot = historyRef.current[targetIndex];
        applySnapshot(cloneSnapshot(snapshot.nodes, snapshot.edges));
        setCanUndo(targetIndex > 0);
        setCanRedo(targetIndex < historyRef.current.length - 1);
    }, [canUndo, applySnapshot, cloneSnapshot]);

    const handleRedo = useCallback(() => {
        if (!canRedo) return;
        const targetIndex = Math.min(historyRef.current.length - 1, historyIndexRef.current + 1);
        historyIndexRef.current = targetIndex;
        const snapshot = historyRef.current[targetIndex];
        applySnapshot(cloneSnapshot(snapshot.nodes, snapshot.edges));
        setCanUndo(targetIndex > 0);
        setCanRedo(targetIndex < historyRef.current.length - 1);
    }, [canRedo, applySnapshot, cloneSnapshot]);

    useEffect(() => {
        if (!historyRef.current.length) {
            historyRef.current = [cloneSnapshot(nodes, edges)];
            historyIndexRef.current = 0;
            syncIdFromNodes(nodes);
            // canUndo/canRedo already start as false; seeding history keeps them false.
        }
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // Sync to localStorage and History on every change
    useEffect(() => {
        if (!external) {
            const graph = buildGraphIR(nodes, edges);
            localStorage.setItem("graphIR", JSON.stringify(graph));
            localStorage.setItem("nodes", JSON.stringify(nodes));
            localStorage.setItem("edges", JSON.stringify(edges));
        }

        const restoring = isRestoring.current;
        const skipping = skipHistory.current;
        isRestoring.current = false;
        skipHistory.current = false;

        if (!restoring && !skipping) {
            const trimmed = historyRef.current.slice(0, historyIndexRef.current + 1);
            trimmed.push(cloneSnapshot(nodes, edges));
            const limited = trimmed.length > 50 ? trimmed.slice(trimmed.length - 50) : trimmed;
            historyRef.current = limited;
            historyIndexRef.current = limited.length - 1;
        }
        const canUndoNow = historyIndexRef.current > 0;
        const canRedoNow = historyIndexRef.current < historyRef.current.length - 1;
        setCanUndo(canUndoNow);
        setCanRedo(canRedoNow);
        syncIdFromNodes(nodes);
    }, [nodes, edges, cloneSnapshot, external]);

    // Drop orphaned edges. Derived during render (React's "adjust state when a prop
    // changes" pattern) so the invalid wiring never reaches a commit or a paint.
    // Converges: after the corrective re-render no orphan remains.
    const liveNodeIds = new Set(nodes.map(n => n.id));
    if (edges.some(e => !liveNodeIds.has(e.source) || !liveNodeIds.has(e.target))) {
        setEdges(eds => eds.filter(e => liveNodeIds.has(e.source) && liveNodeIds.has(e.target)));
    }

    return {
        nodes,
        setNodes,
        edges,
        setEdges,
        onNodesChange,
        onEdgesChange,
        canUndo,
        canRedo,
        handleUndo,
        handleRedo,
        duplicateSelected,
        edgesWithHandlers
    };
}
