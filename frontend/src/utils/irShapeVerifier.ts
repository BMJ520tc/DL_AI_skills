// utils/irShapeVerifier.ts — 结构化画布（IR 图）的连线形状校验（需求五.1）。
//
// 背景：外部画布模式的节点是后端 IR 节点（type="ir"），没有 LAYER_REGISTRY 里的
// shapeVerifier/shapeCompute，因此 shape_verifier.verifyShapes 对它无法工作（会全部报
// 「未知节点类型」）。IR 节点的形状来自后端回填，落在 data.__in_shape / data.__out_shape
// （见 utils/irAdapter.ts，对应 ir_node.input_shape / output_shape）。
//
// 本模块只做「连线两端形状是否一致」这一件事：
//   - 任一端未知 → 放行（缺形状不算失败，避免把「还没回填形状」刷满诊断面板）；
//   - 两端都已知 → 维度数必须相同、各维必须相等；维度为 -1 / null / undefined 表示
//     该维未知，跳过该维比对（与后端「-1 表示未知维」的口径一致）。
// 缺形状的节点单独按「N 个节点缺形状」提示，与 ModelViewerView 的 missingShapes 口径一致。

import type { Edge, Node } from "@xyflow/react";
import type { ShapeFailure } from "./shape_verifier";

/** 未知维度的占位值（后端 IR 用 -1 表示未知维）。 */
export const IR_UNKNOWN_DIM = -1;

/** 形状维度：number 为已知维；-1/null/undefined 为未知维。 */
export type IrShapeDim = number | null | undefined;
export type IrShape = readonly IrShapeDim[] | null | undefined;

/** 与 ShapeResult 结构兼容（shapes/failures/ok），另带缺形状节点清单。 */
export type IrShapeVerification = {
    ok: boolean;
    failures: ShapeFailure[];
    shapes: Record<string, { defaultShape: number[] }>;
    /** 缺形状的节点 id（非 op 节点且 __in_shape / __out_shape 任缺其一）。 */
    missingShapes: string[];
};

function asBag(data: unknown): Record<string, unknown> {
    return data && typeof data === "object" ? (data as Record<string, unknown>) : {};
}

/**
 * 读取 IR 节点上的形状字段（__in_shape / __out_shape）。
 * 非数组 / 空数组 / 全非法值 → null（视为未知）；非法维度归为未知维 -1。
 */
export function readIrShape(data: unknown, key: "__in_shape" | "__out_shape"): number[] | null {
    const raw = asBag(data)[key];
    if (!Array.isArray(raw) || raw.length === 0) return null;
    const out: number[] = [];
    for (const value of raw) {
        const num = typeof value === "number" ? value : value === null ? IR_UNKNOWN_DIM : Number(value);
        out.push(Number.isFinite(num) ? num : IR_UNKNOWN_DIM);
    }
    return out;
}

/** 形状是否已知：非空数组即已知（各维仍可能是未知维）。 */
export function isIrShapeKnown(shape: IrShape): shape is readonly IrShapeDim[] {
    return Array.isArray(shape) && shape.length > 0;
}

function isUnknownDim(dim: IrShapeDim): boolean {
    return dim === null || dim === undefined || dim === IR_UNKNOWN_DIM;
}

/**
 * 形状一致性判据（结构化画布连线的唯一口径）：
 *   1) 任一端未知（null / 空数组）→ true：缺形状不算不一致，不阻拦正常连线；
 *   2) 两端都已知 → 维度数必须相同，否则 false；
 *   3) 逐维比对：任一维为未知维（-1/null/undefined）→ 跳过该维；其余维必须严格相等。
 */
export function shapesCompatible(a: IrShape, b: IrShape): boolean {
    if (!isIrShapeKnown(a) || !isIrShapeKnown(b)) return true;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (isUnknownDim(a[i]) || isUnknownDim(b[i])) continue;
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function labelOf(node: Node): string {
    const data = asBag(node.data);
    const label = data.label ?? data.class_name;
    const base = typeof label === "string" && label ? label : (node.type ?? node.id);
    // 带上节点 id：同一张图上同名层（如多个 Linear）时，诊断面板要能指认是哪一个节点。
    return base === node.id ? base : `${base}（${node.id}）`;
}

/**
 * `ancestor` 是不是 `node` 的祖先（沿 `parentId` 上溯，带上限防环）。
 *
 * 用于跳过**模块输入边**：IR 里 `模块 M → M 的子孙节点` 表示「M 的**输入**喂给该子节点」
 * （后端 `ir_codegen._module_class._ext_var` 口径），**不是**「M 的输出流过去」。拿 M 的
 * `__out_shape` 去比子节点的 `__in_shape` 必然对不上，会被误标成红线
 * （实测 scGPT 的 `encoder → encoder_embedding`）。
 */
function isAncestor(byId: Map<string, Node>, ancestor: string, node: string): boolean {
    let cur = byId.get(node);
    let hops = 0;
    while (cur && cur.parentId && hops++ < 64) {
        if (cur.parentId === ancestor) return true;
        cur = byId.get(cur.parentId);
    }
    return false;
}

/**
 * 校验 IR 图的所有连线：源节点 __out_shape ↔ 目标节点 __in_shape。
 * 同一目标节点的多条不匹配入边合并为一条 ShapeFailure（避免诊断面板被同一节点刷屏）。
 *
 * **模块输入边跳过**（源是目标的祖先）：那条边带的是模块的**输入**，不是输出（见 `isAncestor`）。
 */
export function verifyIRShapes(nodes: Node[], edges: Edge[]): IrShapeVerification {
    const byId = new Map(nodes.map(n => [n.id, n]));
    const shapes: Record<string, { defaultShape: number[] }> = {};
    const missingShapes: string[] = [];

    for (const node of nodes) {
        const data = asBag(node.data);
        const outShape = readIrShape(data, "__out_shape");
        if (outShape) shapes[node.id] = { defaultShape: outShape };
        // 缺形状口径与 ModelViewerView.missingShapes 一致：op 节点不参与统计，in/out 任缺其一即算缺。
        if (data.kind !== "op" && (!outShape || !readIrShape(data, "__in_shape"))) {
            missingShapes.push(node.id);
        }
    }

    const mismatched = new Map<string, Array<{ source: string; outShape: number[]; inShape: number[] }>>();
    for (const edge of edges) {
        const sourceNode = byId.get(edge.source);
        const targetNode = byId.get(edge.target);
        if (!sourceNode || !targetNode) continue; // 孤立边由 useGraphState 统一清理
        if (isAncestor(byId, edge.source, edge.target)) continue; // 模块输入边：带的是模块的输入
        const outShape = readIrShape(sourceNode.data, "__out_shape");
        const inShape = readIrShape(targetNode.data, "__in_shape");
        if (!isIrShapeKnown(outShape) || !isIrShapeKnown(inShape)) continue; // 缺形状不算失败
        if (shapesCompatible(outShape, inShape)) continue;
        const list = mismatched.get(targetNode.id) ?? [];
        list.push({ source: sourceNode.id, outShape: [...outShape], inShape: [...inShape] });
        mismatched.set(targetNode.id, list);
    }

    const failures: ShapeFailure[] = [];
    for (const [nodeId, items] of mismatched) {
        const node = byId.get(nodeId);
        if (!node) continue;
        const detail = items
            .map(i => `${i.source} 输出 [${i.outShape.join(", ")}] → 本节点输入 [${i.inShape.join(", ")}]`)
            .join("；");
        failures.push({
            nodeId,
            nodeType: node.type,
            label: labelOf(node),
            error: `输入形状不匹配：${detail}`,
            inputShapes: items.map(i => i.outShape),
            upstream: items.map(i => i.source),
        });
    }

    return { ok: failures.length === 0, failures, shapes, missingShapes };
}

export type IrConnection = { source?: string | null; target?: string | null };

/**
 * 连线合法性（ReactFlow 的 isValidConnection 与 onConnect 守卫共用同一口径）：
 * 两端节点都存在且形状都已知时，形状不一致 → 拒绝（false）；
 * 形状未知 / 节点缺失 / 自连 → 放行（true）——形状未知不阻拦正常连线。
 */
export function isIrConnectionCompatible(nodes: Node[], connection: IrConnection): boolean {
    const { source, target } = connection;
    if (!source || !target || source === target) return true;
    const byId = new Map(nodes.map(n => [n.id, n]));
    const sourceNode = byId.get(source);
    const targetNode = byId.get(target);
    if (!sourceNode || !targetNode) return true;
    return shapesCompatible(
        readIrShape(sourceNode.data, "__out_shape"),
        readIrShape(targetNode.data, "__in_shape"),
    );
}
