import type { Edge, Node } from "@xyflow/react";
import type { GraphDisplay, GraphEdge, GraphHandle, GraphHandleKind, GraphIR, GraphNode } from "../types/graph";
import { LEAF_W, ownContentH } from "./nodeSize";

const PAD_X = 16;
const PAD_BOTTOM = 16;

const GRAPH_VERSION = 2;

function dedupe<T>(arr: T[]): T[] {
    return Array.from(new Set(arr));
}

// Map handle usage counts to a GraphHandleKind.
function toHandleKind(sourceCount: number, targetCount: number): GraphHandleKind {
    if (sourceCount > 0 && targetCount === 0) return "output";
    if (targetCount > 0 && sourceCount === 0) return "input";
    if (sourceCount > 0 && targetCount > 0) return "other";
    return "other";
}

function formatDisplay(n: Node): GraphDisplay {
    const data = (n.data || {}) as Record<string, unknown>;
    const type = n.type || "Layer";
    const rawLabel = typeof data.label === "string" ? data.label : undefined;
    const baseTitle = rawLabel || type;
    const params = data.params ? String(data.params) : undefined;
    const shape = Array.isArray((data as Record<string, unknown>).__shape)
        ? `shape: [${((data as Record<string, unknown>).__shape as Array<unknown>).join(",")}]`
        : undefined;
    return { title: baseTitle, params, shape };
}
/**
 * Helper: Sorts nodes by hierarchy depth so Parents render before Children.
 * This prevents React Flow from detaching children on load.
 */
function sortNodesByHierarchy(nodes: Node[]): Node[] {
    const getDepth = (n: Node, allNodes: Node[]): number => {
        let depth = 0;
        let current = n;
        while (current.parentId) {
            depth++;
            const parent = allNodes.find(p => p.id === current.parentId);
            if (!parent) break; // Orphaned
            current = parent;
        }
        return depth;
    };

    // Sort: Depth 0 (Roots) -> Depth 1 (Children) -> Depth 2 (Grandchildren)
    return [...nodes].sort((a, b) => getDepth(a, nodes) - getDepth(b, nodes));
}

/**
 * Convert the current React Flow nodes/edges into a versioned GraphIR snapshot.
 * This captures stable node IDs, handle IDs, minimal data, and edge wiring.
 */
export function buildGraphIR(nodes: Node[], edges: Edge[]): GraphIR {
    const handleUsage = new Map<string, { source: string[]; target: string[] }>();

    edges.forEach(e => {
        const sourceHandle = e.sourceHandle || "out";
        const targetHandle = e.targetHandle || "in";
        const sourceEntry = handleUsage.get(e.source) || { source: [], target: [] };
        sourceEntry.source.push(sourceHandle);
        handleUsage.set(e.source, sourceEntry);
        const targetEntry = handleUsage.get(e.target) || { source: [], target: [] };
        targetEntry.target.push(targetHandle);
        handleUsage.set(e.target, targetEntry);
    });

    const graphNodes: GraphNode[] = nodes.map(n => {
        const usage = handleUsage.get(n.id) || { source: [], target: [] };
        const handleIds = dedupe([...usage.source.map(p => p), ...usage.target.map(p => p)]);
        const handles: GraphHandle[] = handleIds.sort().map((pid, idx) => ({
            id: pid,
            kind: toHandleKind(usage.source.filter(p => p === pid).length, usage.target.filter(p => p === pid).length),
            order: idx,
        }));
        const data = (n.data || {}) as Record<string, unknown>;
        const display = formatDisplay(n);
        return {
            id: n.id,
            type: n.type || "custom",
            label: typeof data.label === "string" ? data.label : n.type ?? n.id,
            display,
            handles,
            position: n.position,
            parentId: n.parentId,
            extent: n.extent === "parent" ? "parent" : undefined,
            data,
        };
    });

    const graphEdges: GraphEdge[] = edges.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle || "out",
        targetHandle: e.targetHandle || "in",
        kind: "data",
        data: (e.data || {}) as Record<string, unknown>,
    }));

    return {
        version: GRAPH_VERSION,
        createdAt: new Date().toISOString(),
        nodes: graphNodes,
        edges: graphEdges,
    };
}

/**
 * Convert a GraphIR snapshot back into React Flow nodes/edges.
 */
/** 旧 graph.json 可能没有 layout_hint（早期布局只给容器）→ 按同口径兜底补齐，
 *  否则节点会按自然尺寸渲染、相互重叠；容器自底向上由子节点推出。 */
function withFallbackHints(graph: GraphIR): void {
    const childrenOf = (id: string) => graph.nodes.filter(n => n.parentId === id);
    const hintOf = (n: (typeof graph.nodes)[number]) => (n.data as Record<string, unknown> | undefined)?.layout_hint as
        | { width?: number; height?: number }
        | undefined;

    const fill = (n: (typeof graph.nodes)[number]): { width: number; height: number } => {
        const kids = childrenOf(n.id);
        let size: { width: number; height: number };
        if (kids.length === 0) {
            const params = (n.data as Record<string, unknown> | undefined)?.params as Record<string, unknown> | undefined;
            size = { width: LEAF_W, height: ownContentH(params) };
        } else {
            const kidSizes = kids.map(fill);
            const maxW = Math.max(...kidSizes.map(k => k.width));
            const maxBottom = Math.max(...kids.map((k, i) => (k.position?.y ?? 0) + kidSizes[i].height));
            size = { width: PAD_X * 2 + maxW, height: maxBottom + PAD_BOTTOM };
        }
        const existing = hintOf(n);
        const merged = existing?.height ? { width: existing.width ?? size.width, height: existing.height } : size;
        (n.data as Record<string, unknown>).layout_hint = merged;
        return merged;
    };

    for (const n of graph.nodes) if (!n.parentId) fill(n);
    for (const n of graph.nodes) if (!hintOf(n)) fill(n);  // 游离节点兜底
}

/** 层级深度（根=0）——React Flow 平铺渲染嵌套节点，容器有实底背景，
 *  必须让子节点 z-index 高于祖先，否则子节点被父容器背景盖住（选中时才因 z-index 提升而露出）。 */
function depthOf(n: GraphNode, byId: Map<string, GraphNode>): number {
    let d = 0;
    let cur: GraphNode | undefined = n;
    while (cur?.parentId) {
        d += 1;
        cur = byId.get(cur.parentId);
    }
    return d;
}

export function applyGraphIR(graph: GraphIR): { nodes: Node[]; edges: Edge[] } {
    withFallbackHints(graph);
    const byId = new Map(graph.nodes.map(n => [n.id, n]));
    const nodes: Node[] = graph.nodes.map(n => ({
        id: n.id,
        type: n.type,
        position: n.position || { x: 0, y: 0 },
        parentId: n.parentId,
        extent: n.extent,
        ...(n.parentId ? { extent: n.extent || "parent" } : {}),
        zIndex: 10 + depthOf(n, byId) * 10,   // 子节点恒在祖先之上
        data: { ...(n.data || {}), label: n.display?.title ?? n.label },
    }));

    const edges: Edge[] = graph.edges.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
        data: e.data,
        type: "custom",
    }));
    const sortedNodes = sortNodesByHierarchy(nodes);
    return { nodes: sortedNodes, edges };
}

/**
 * Filters the graph to return only the "Root" layer (Top-level nodes and edges).
 * This is useful for compilation and verification steps that should treat
 * nested subgraphs (like Repeat Layers) as encapsulated black boxes.
 * * Usage:
 * const { rootNodes, rootEdges } = getRootGraph(nodes, edges);
 * verifyShapes(rootNodes, rootEdges, ...);
 */
export function getRootGraph(nodes: Node[], edges: Edge[]): { rootNodes: Node[]; rootEdges: Edge[] } {
    // Root nodes: only nodes that DO NOT have a parent
    const rootNodes = nodes.filter(n => !n.parentId);
    const rootNodeIds = new Set(rootNodes.map(n => n.id));
    // Root Edges: Only edges where BOTH source and target are in root
    // This excludes:
    // - Internal edges (Child -> Child)
    // - Boundary edges managed by Repeat Layers (Child-> Parent)
    // This keeps:
    // - Normal edges (Node A -> Node B)
    // - Edges connecting to Repeat Layer external handles (Node A -> Repeat Layer)

    const rootEdges = edges.filter(e => {
        return rootNodeIds.has(e.source) && rootNodeIds.has(e.target);
    });
    return { rootNodes, rootEdges };
}
