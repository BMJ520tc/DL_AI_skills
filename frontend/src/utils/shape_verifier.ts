import type { Edge, Node } from "@xyflow/react";
import type { LayerRegistry } from "../node_gen/BaseClass";
// import { LAYER_REGISTRY } from "../types/nodeTypes";

export type ShapeFailure = {
    nodeId: string;
    nodeType?: string;
    label?: string;
    error: string;
    inputShapes?: number[][];
    upstream?: string[];
};
type NodeShapes = {
    defaultShape: number[];
    byHandle?: Record<string, number[]>;
};

export type ShapeResult = {
    ok: boolean;
    shapes: Record<string, NodeShapes>;
    failures: ShapeFailure[];
};

export function verifyShapes(nodes: Node[], edges: Edge[], registry: LayerRegistry): ShapeResult {
    const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
    const sources: Record<string, string[]> = {};
    edges.forEach(e => {
        const sourceNode = byId[e.source];
        const targetNode = byId[e.target];
        if (sourceNode && targetNode) {
            if (sourceNode.parentId === targetNode.id) {
                return;
            }
        }
        if (byId[e.target]) {
            (sources[e.target] ||= []).push(e.source);
        }
    });

    const shapes: Record<string, NodeShapes> = {};
    const failures: ShapeFailure[] = [];
    const pending = new Set(
        nodes
            .filter(n => {
                if (!n.parentId) return true;
                const parentIsPresent = !!byId[n.parentId];
                return !parentIsPresent;
            })
            .map(n => n.id),
    );

    let progressed = true;
    while (pending.size && progressed) {
        progressed = false;
        for (const id of Array.from(pending)) {
            const node = byId[id];
            if (!node) {
                pending.delete(id);
                continue;
            }
            const layer = node.type ? registry[node.type] : undefined;
            if (!layer) {
                failures.push({
                    nodeId: id,
                    nodeType: node.type,
                    error: `未知节点类型: ${node.type ?? "undefined"}`,
                });
                pending.delete(id);
                progressed = true;
                continue;
            }
            const inputIds = sources[id] || [];
            if (inputIds.some(src => !byId[src])) {
                failures.push({
                    nodeId: id,
                    nodeType: node.type,
                    label: layer.label,
                    error: "缺少上游节点",
                    upstream: inputIds,
                });
                pending.delete(id);
                progressed = true;
                continue;
            }
            const ready = inputIds.every(src => shapes[src]);
            if (!ready) continue;

            const inputShapes = inputIds.map(src => shapes[src]?.defaultShape || []);
            const verdict = layer.shapeVerifier(node.data, inputShapes, registry);
            if (!verdict.ok) {
                failures.push({
                    nodeId: id,
                    nodeType: node.type,
                    label: layer.label,
                    error: verdict.error,
                    inputShapes,
                    upstream: inputIds,
                });
                pending.delete(id);
                progressed = true;
                continue;
            }
            const computed = layer.shapeCompute(node.data, inputShapes, registry);
            if (Array.isArray(computed)) {
                // 数组形态既可能是单个形状，也可能是多输出形状列表；二者历史上都直接
                // 作为 defaultShape 使用，这里保持原有行为。
                shapes[id] = { defaultShape: computed as number[] };
            } else if (computed && typeof computed === "object") {
                const byHandle = computed as Record<string, number[]>;
                const entries = Object.entries(byHandle);
                const first = entries[0]?.[1] || [];
                shapes[id] = { defaultShape: first, byHandle };
            } else {
                shapes[id] = { defaultShape: [] };
            }
            pending.delete(id);
            progressed = true;
        }
    }

    if (pending.size) {
        pending.forEach(id => {
            const node = byId[id];
            const layer = node?.type ? registry[node.type] : undefined;
            failures.push({
                nodeId: id,
                nodeType: node?.type,
                label: layer?.label,
                error: "缺少上游形状（边断开或源无效）",
                upstream: sources[id] || [],
            });
        });
    }
    console.log(failures);
    return { ok: failures.length === 0, shapes, failures };
}
