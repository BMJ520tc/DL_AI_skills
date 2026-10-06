// utils/irAdapter.ts — IR ↔ 画布 GraphIR 转换（模块四 B2/B3）。
// irToGraphIR 与后端 ir_graphir.py 同构（布局/句柄/skip 判定/display 字段对齐），
// 后端入库时同样产出一份 graph.json，前端只需对打开的画布做 graphIRToFlow 还原。

import type { Edge, Node } from "@xyflow/react";
import type { IrEdge, IrGraph, IrNode } from "../api/client";
import type { GraphEdge, GraphHandle, GraphIR, GraphNode } from "../types/graph";
import { applyGraphIR } from "./graphIR";
import { LEAF_W, ownContentH } from "./nodeSize";

const GRAPH_VERSION = 2;

// 布局参数（与后端 ir_graphir._layout 同构）：按**子树实际尺寸**自底向上排布，
// 父容器被其子节点撑开；旧实现用固定网格(240×100)+固定父宽 500，兄弟容器会互相重叠。
const LEAF_MIN_H = 32;
const PAD_X = 16;
const PAD_TOP = 34;
const PAD_BOTTOM = 16;
const GAP_X = 24;
const GAP_Y = 20;
const COLS = 2; // 仅当子节点都是叶子/操作时并排；含子容器时单列

const inEdgesOf = (ir: IrGraph, id: string): IrEdge[] => ir.edges.filter(e => e.to === id);
const outEdgesOf = (ir: IrGraph, id: string): IrEdge[] => ir.edges.filter(e => e.from === id);
const childrenOf = (ir: IrGraph, id: string): IrNode[] => ir.nodes.filter(n => n.parent_id === id);

function shapeStr(shape?: number[] | null): string | undefined {
    return shape && shape.length ? `[${shape.join(",")}]` : undefined;
}

/** 每层输入输出形状文本（模块详细设计六.2、阶段3实施方案 3.7）：缺项显示「未知」。 */
function shapeIO(inShape?: number[] | null, outShape?: number[] | null): string {
    return `${shapeStr(inShape) ?? "未知"} → ${shapeStr(outShape) ?? "未知"}`;
}

/** display 与前端 formatDisplay/后端 _display 对齐：title=类名，params=参数 JSON，shape=in → out 形状文本。 */
function displayOf(n: IrNode) {
    const params = n.params && Object.keys(n.params).length ? JSON.stringify(n.params) : undefined;
    return { title: n.class_name || n.id, params, shape: shapeIO(n.input_shape, n.output_shape) };
}

/** 单入/单出用裸 "in"/"out"，多入/多出用 "in0".."inN"（buildGraphIR 约定）。 */
function handleIds(nIns: number, nOuts: number): [string[], string[]] {
    const ins = nIns <= 1 ? ["in"] : Array.from({ length: nIns }, (_, i) => `in${i}`);
    const outs = nOuts <= 1 ? ["out"] : Array.from({ length: nOuts }, (_, i) => `out${i}`);
    return [ins, outs];
}

function handlesOf(ins: string[], outs: string[]): GraphHandle[] {
    return [
        ...ins.map((id, k) => ({ id, kind: "input" as const, order: k })),
        ...outs.map((id, k) => ({ id, kind: "output" as const, order: k })),
    ];
}

/** 子节点并排数（与后端 _cols_for 同构）：双列，单子节点单列。 */
function colsOf(kids: IrNode[]): number {
    return kids.length <= 1 ? 1 : COLS;
}

type Size = { w: number; h: number };

/** 双列网格：列宽=该列最大宽、行高=该行最大高 → 格子互不相交（兄弟容器不会重叠）。 */
function gridSlots(
    kids: IrNode[],
    sizes: Map<string, Size>,
): { coords: Map<string, { x: number; y: number }>; gridW: number; gridH: number } {
    const cols = colsOf(kids);
    const rows = Math.ceil(kids.length / cols);
    const colW = new Array(cols).fill(0);
    const rowH = new Array(rows).fill(0);
    kids.forEach((k, i) => {
        const s = sizes.get(k.id)!;
        colW[i % cols] = Math.max(colW[i % cols], s.w);
        rowH[Math.floor(i / cols)] = Math.max(rowH[Math.floor(i / cols)], s.h);
    });
    const xOf: number[] = [];
    let acc = PAD_X;
    for (let c = 0; c < cols; c++) {
        xOf.push(acc);
        acc += colW[c] + GAP_X;
    }
    const yOf: number[] = [];
    acc = PAD_TOP;
    for (let r = 0; r < rows; r++) {
        yOf.push(acc);
        acc += rowH[r] + GAP_Y;
    }
    const coords = new Map<string, { x: number; y: number }>();
    kids.forEach((k, i) => coords.set(k.id, { x: xOf[i % cols], y: yOf[Math.floor(i / cols)] }));
    return {
        coords,
        gridW: colW.reduce((a, b) => a + b, 0) + GAP_X * (cols - 1),
        gridH: rowH.reduce((a, b) => a + b, 0) + GAP_Y * (rows - 1),
    };
}

/** 自底向上算子树包围盒（父容器 = 子节点排版结果 + 内边距）。 */
function subtreeSizes(ir: IrGraph): Map<string, Size> {
    const sizes = new Map<string, Size>();
    const visit = (n: IrNode): Size => {
        const cached = sizes.get(n.id);
        if (cached) return cached;
        const kids = childrenOf(ir, n.id);
        const ownH = ownContentH(n.params);   // 参数行数必须计入高度，否则节点渲染高于预留会与兄弟重叠
        let size: Size;
        if (kids.length === 0) {
            size = { w: LEAF_W, h: Math.max(LEAF_MIN_H, ownH) };
        } else {
            kids.forEach(visit);
            const { gridW, gridH } = gridSlots(kids, sizes);
            size = { w: 2 * PAD_X + gridW, h: Math.max(ownH, PAD_TOP + gridH + PAD_BOTTOM) };
        }
        sizes.set(n.id, size);
        return size;
    };
    ir.nodes.forEach(visit);
    return sizes;
}

/** 确定性布局：子节点坐标相对父节点原点（React Flow parentId 语义），根为绝对坐标。 */
function layoutOf(ir: IrGraph): { positions: Map<string, { x: number; y: number }>; sizes: Map<string, Size> } {
    const sizes = subtreeSizes(ir);
    const positions = new Map<string, { x: number; y: number }>();
    const place = (parent: IrNode): void => {
        const kids = childrenOf(ir, parent.id);
        if (kids.length === 0) return;
        const { coords } = gridSlots(kids, sizes);
        kids.forEach(k => {
            positions.set(k.id, coords.get(k.id)!);
            place(k);
        });
    };
    const root = ir.nodes.find(n => n.id === ir.root_id) ?? ir.nodes[0];
    if (root) {
        positions.set(root.id, { x: 0, y: 0 });
        place(root);
    }
    let offsetY = 0;
    const rootW = root ? sizes.get(root.id)?.w ?? LEAF_W : 0;
    for (const n of ir.nodes) {
        if (positions.has(n.id)) continue;
        positions.set(n.id, { x: rootW + 80, y: offsetY });
        offsetY += (sizes.get(n.id)?.h ?? LEAF_MIN_H) + GAP_Y;
    }
    return { positions, sizes };
}

/** 边视觉类别（与后端 _edge_kind 一致）：多输入目标的非首条入边、或同父跳过相邻兄弟 → skip。 */
function edgeKindOf(ir: IrGraph, e: IrEdge): GraphEdge["kind"] {
    const byId = new Map(ir.nodes.map(n => [n.id, n]));
    const ins = inEdgesOf(ir, e.to);
    if (ins.length > 1 && ins[0].from !== e.from) return "skip";
    const src = byId.get(e.from);
    const tgt = byId.get(e.to);
    const parent = src?.parent_id;
    if (parent && parent === tgt?.parent_id) {
        const sibs = childrenOf(ir, parent).map(c => c.id);
        const si = sibs.indexOf(e.from);
        const ti = sibs.indexOf(e.to);
        if (si >= 0 && ti >= 0 && ti !== si + 1) return "skip";
    }
    return "data";
}

function depthOf(ir: IrGraph, id: string): number {
    const byId = new Map(ir.nodes.map(n => [n.id, n]));
    let d = 0;
    let cur = byId.get(id);
    while (cur?.parent_id) {
        d++;
        cur = byId.get(cur.parent_id);
    }
    return d;
}

/** IR → GraphIR v2 快照（父节点排在子节点前，避免画布加载时子节点脱离）。 */
export function irToGraphIR(ir: IrGraph): GraphIR {
    const { positions, sizes } = layoutOf(ir);
    const nIns = new Map<string, number>();
    const nOuts = new Map<string, number>();
    for (const n of ir.nodes) {
        nIns.set(n.id, inEdgesOf(ir, n.id).length);
        nOuts.set(n.id, outEdgesOf(ir, n.id).length);
    }

    const nodes: GraphNode[] = [...ir.nodes]
        .sort((a, b) => depthOf(ir, a.id) - depthOf(ir, b.id))
        .map(n => {
            const [ins, outs] = handleIds(nIns.get(n.id) ?? 0, nOuts.get(n.id) ?? 0);
            const data: Record<string, unknown> = {
                kind: n.kind,
                class_name: n.class_name,
                module_path: n.module_path,
                params: n.params || {},
                ir_id: n.id,
                input_shape: n.input_shape,
                output_shape: n.output_shape,
                // __shape 保留以兼容画布 IrNode；__in_shape/__out_shape 供查看器展示 in → out
                __shape: n.output_shape ?? n.input_shape,
                __in_shape: n.input_shape ?? null,
                __out_shape: n.output_shape ?? null,
            };
            // 所有节点都给尺寸（含叶子）：画布按固定高度渲染 + 参数区内部滚动，
            // 保证「渲染高度 == 布局预留高度」，父子/兄弟不会互相挤压
            const size = sizes.get(n.id)!;
            data.layout_hint = { width: size.w, height: size.h };
            const parentId = n.parent_id || undefined;
            return {
                id: n.id,
                type: "ir",
                label: n.class_name || n.id,
                display: displayOf(n),
                handles: handlesOf(ins, outs),
                position: positions.get(n.id),
                parentId,
                extent: parentId ? "parent" : undefined,
                data,
            };
        });

    const srcCount = new Map<string, number>();
    const tgtCount = new Map<string, number>();
    const edges: GraphEdge[] = ir.edges.map((e, i) => {
        const [, srcOuts] = handleIds(nIns.get(e.from) ?? 0, nOuts.get(e.from) ?? 0);
        const [tgtIns] = handleIds(nIns.get(e.to) ?? 0, nOuts.get(e.to) ?? 0);
        const si = srcCount.get(e.from) ?? 0;
        srcCount.set(e.from, si + 1);
        const ti = tgtCount.get(e.to) ?? 0;
        tgtCount.set(e.to, ti + 1);
        return {
            id: `e${i}`,
            source: e.from,
            target: e.to,
            sourceHandle: srcOuts[Math.min(si, srcOuts.length - 1)],
            targetHandle: tgtIns[Math.min(ti, tgtIns.length - 1)],
            kind: edgeKindOf(ir, e),
            data: e.tensor_shape ? { tensor_shape: e.tensor_shape } : {},
        };
    });

    return { version: GRAPH_VERSION, createdAt: new Date().toISOString(), nodes, edges };
}

/**
 * 保存画布时的 GraphIR 合并：以服务端 graph 为字段权威，避免 buildGraphIR 全量重建
 * 覆盖后端写入字段（节点 handles/display、边 kind 含 skip 跳跃边等）。
 * 规则：同 id 节点保留服务端 data（__shape/kind/class_name/handles 等）与 handles/display，
 * 采用编辑器更新后的位置；同 id 边保留服务端 kind/data；编辑器新增的节点/边照实写入；
 * 被删除的节点/边不出现在结果里。节点的 data 以服务端为基底、编辑器改动覆盖其上。
 */
export function mergeGraphIR(server: GraphIR, edited: GraphIR): GraphIR {
    const serverNodes = new Map(server.nodes.map(n => [n.id, n]));
    const serverEdges = new Map(server.edges.map(e => [e.id, e]));

    // 忽略值为 undefined 的键：否则编辑快照里的 { __shape: undefined } 会浅合并覆盖服务端字段，
    // 经 JSON.stringify 后该键直接消失（外部画布挂载时曾出现）。
    const defined = (obj?: Record<string, unknown>): Record<string, unknown> => {
        const out: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(obj || {})) {
            if (v !== undefined) out[k] = v;
        }
        return out;
    };

    const nodes: GraphNode[] = edited.nodes.map(n => {
        const base = serverNodes.get(n.id);
        if (!base) return n;
        return {
            ...base,
            position: n.position ?? base.position,
            data: {
                ...defined(base.data),
                ...defined(n.data),
                // 显式保留 in/out 形状字段，避免服务端写入的查看器字段被编辑器快照覆盖丢失
                __in_shape: n.data?.__in_shape ?? base.data?.__in_shape ?? null,
                __out_shape: n.data?.__out_shape ?? base.data?.__out_shape ?? null,
            },
        };
    });

    const edges: GraphEdge[] = edited.edges.map(e => {
        const base = serverEdges.get(e.id);
        if (!base) return e;
        return {
            ...e,
            kind: base.kind ?? e.kind,
            data: { ...(base.data || {}), ...(e.data || {}) },
        };
    });

    return { version: edited.version ?? GRAPH_VERSION, createdAt: new Date().toISOString(), nodes, edges };
}

/** GraphIR → React Flow：applyGraphIR 还原 + 把 handles 注入 data（IrNode 据此动态渲染句柄）。 */
export function graphIRToFlow(graph: GraphIR): { nodes: Node[]; edges: Edge[] } {
    const flow = applyGraphIR(graph);
    const byId = new Map(graph.nodes.map(n => [n.id, n]));
    return {
        nodes: flow.nodes.map(n => ({
            ...n,
            data: { ...(n.data || {}), handles: byId.get(n.id)?.handles ?? [] },
        })),
        edges: flow.edges,
    };
}

/** 超过这个节点数就**默认折叠容器**（只渲染容器本身、不渲染其子孙）。 */
export const COLLAPSE_HINT_THRESHOLD = 400;

/**
 * 大图默认折叠：把「位于未展开容器里」的节点剔出去。
 *
 * 为什么：真实大模型追踪出来会有几千个模块（boltz 实测 5856），一次性全渲染必卡死。
 * `expandedIds === null` 时按规模取默认——小图维持原样（全展开，行为不变），大图只留顶层。
 * 边同步过滤，避免悬挂边（`irToGraphIR` 会按 `ir.edges` 算句柄数）。
 */
export function collapseForDisplay(ir: IrGraph, expandedIds: Set<string> | null): IrGraph {
    const parentOf = new Map(ir.nodes.map(n => [n.id, n.parent_id ?? null]));
    const withKids = new Set(
        ir.nodes.filter(n => n.parent_id).map(n => n.parent_id as string),
    );
    if (withKids.size === 0) return ir;   // 没有嵌套结构，无从折叠
    const expanded = expandedIds
        ?? (ir.nodes.length > COLLAPSE_HINT_THRESHOLD ? new Set<string>() : withKids);

    const hidden = new Set<string>();
    for (const n of ir.nodes) {
        let p = parentOf.get(n.id) ?? null;
        while (p) {
            if (!expanded.has(p)) {
                hidden.add(n.id);
                break;
            }
            p = parentOf.get(p) ?? null;
        }
    }
    if (hidden.size === 0) return ir;
    return {
        ...ir,
        nodes: ir.nodes.filter(n => !hidden.has(n.id)),
        edges: ir.edges.filter(e => !hidden.has(e.from) && !hidden.has(e.to)),
    };
}

/** 所有「有子节点」的容器 id（判断能不能折叠 / 一键展开用）。 */
export function containerIds(ir: IrGraph): string[] {
    return [...new Set(ir.nodes.filter(n => n.parent_id).map(n => n.parent_id as string))];
}
