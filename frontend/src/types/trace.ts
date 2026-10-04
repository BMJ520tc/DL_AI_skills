export interface TraceRequest {
    graph: unknown; // GraphIR snapshot sent to backend
    inputShapes: Array<number[]>; // e.g., [[1, 3, 224, 224]]
    code?: string; // optional generated PyTorch code (frontend codegen)
    /** 追踪虚拟输入使用的随机种子（工具栏「种子」下拉/自定义输入；未选择时不带该字段）。 */
    seed?: number;
}

export interface TraceEntry {
    id: string;
    scope: string;
    op: string;
    inputShape?: string;
    outputShape?: string;
    dtype?: string;
    stats?: {
        min?: number;
        max?: number;
        mean?: number;
    };
    nodeIds?: string[]; // GraphIR node ids mapped to this op
}

export interface TraceResponse {
    entries: TraceEntry[];
    warnings?: string[];
    svgBase64?: string;
    summaryText?: string;
    /** 端点/独立 runner 不可用时的结构化标记（需求五.2 沙盒追踪提示）。 */
    unavailable?: boolean;
    unavailableReason?: string;
}
