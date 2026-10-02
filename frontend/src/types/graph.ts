// Handle kinds are explicit; we no longer infer input/output from edge direction alone.
export type GraphHandleKind = "input" | "output" | "other";

export interface GraphHandle {
    id: string; // stable handle id on the node
    kind: GraphHandleKind;
    // order preserves positional meaning (e.g., arg0/arg1) after the recent convention shift.
    order: number;
}

export interface GraphDisplay {
    title: string;
    params?: string;
    shape?: string;
}

export interface GraphNode {
    id: string;
    type: string;
    label?: string;
    display?: GraphDisplay;
    handles: GraphHandle[];
    position?: { x: number; y: number };
    parentId?: string;
    extent?: "parent";
    data?: Record<string, unknown>;
}

export interface GraphEdge {
    id: string;
    source: string;
    target: string;
    // Handles must reference GraphHandle ids on the respective nodes; directional convention is now explicit.
    sourceHandle: string;
    targetHandle: string;
    kind?: "data" | "skip" | "control";
    data?: Record<string, unknown>;
}

export interface GraphIR {
    // Versioned snapshot of the diagram (IDs/handles/positions) after the convention change.
    version: number;
    createdAt: string;
    nodes: GraphNode[];
    edges: GraphEdge[];
}

// ---------------------------------------------------------------------------
// 图示投影的最小输入契约。
//
// `projectGraphToDiagram` / `DiagramView` 只读取下列字段（不读 handles/label），
// 因此完整 GraphIR 与画布上的 React Flow 快照都可以直接传入。
// ---------------------------------------------------------------------------

export interface DiagramNodeInput {
    id: string;
    type?: string;
    display?: GraphDisplay;
    data?: Record<string, unknown>;
    position?: { x: number; y: number };
}

export interface DiagramEdgeInput {
    id: string;
    source: string;
    target: string;
    sourceHandle?: string | null;
    targetHandle?: string | null;
    data?: Record<string, unknown>;
}

export interface DiagramGraphInput {
    nodes: DiagramNodeInput[];
    edges: DiagramEdgeInput[];
}
