// utils/elkCompoundLayout.ts — 结构化画布（IR 图）的**左→右层叠布局**（ELK compound）。
//
// 为什么不用后端那套网格（ir_graphir._layout / irAdapter.layoutOf）：它是「2 列网格」，30 个顶层子节点
// 铺成 15 行 → 画布 1324×**2644**、数据流是"往下走"、长边来回穿、线缠在一起（用户实测反馈）。
// 试过"按数据流分层"的手写布局：**反而更差**（顶层平均网格距离 2.74 → 3.26、最长 10 → 14）——这张图
// 既有宽流水线、又有 16 层长的依赖链、还有跨链边，一维排列救不了；**必须用 ELK 的层叠 + 交叉最小化**。
//
// 口径（与 React Flow 的 parentId 嵌套语义对齐）：
//   - 叶子/op 给**固定尺寸**（沿用 `layout_hint` 或实测尺寸）；
//   - 容器（有子节点的）**不给尺寸** → 由 ELK 按子节点 + `elk.padding` 算出，尺寸写回 `layout_hint`
//     （`IrNode` 就是按 `layout_hint` 定尺寸的，所以盒子会自动包住新的子布局）；
//   - ELK 返回的子坐标**已经是相对父节点**的（本机实测确认），正是 React Flow 要的 `position`。
//
// ELK 是异步的：调用方先同步出网格、拿到结果后再覆盖（查看器/画布各自处理 loading）。

import type { Edge, Node } from "@xyflow/react";
import ELK from "elkjs/lib/elk.bundled.js";

import type { LayoutDirection } from "./layout";

const elk = new ELK();

// 与 irAdapter.ts 的 PAD_* 同值：容器内边距（ELK 的 padding 语法）
const PAD_TOP = 34;
const PAD_X = 16;
const PAD_BOTTOM = 16;
const PADDING = `[top=${PAD_TOP},left=${PAD_X},bottom=${PAD_BOTTOM},right=${PAD_X}]`;

/** 布局选项：复用 `layout.ts`（数据流图）那套层叠 + 交叉最小化，外加**嵌套**支持与左右方向。 */
function layoutOptions(direction: LayoutDirection): Record<string, string> {
    return {
        "elk.algorithm": "layered",
        "elk.direction": direction === "LR" ? "RIGHT" : "DOWN",
        // 关键：把整棵树交给同一趟层叠布局（否则每个容器各排各的，跨容器的边会乱走）
        "elk.hierarchyHandling": "INCLUDE_CHILDREN",
        "elk.separateConnectedComponents": "true",
        "elk.layered.considerModelOrder": "true",
        "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
        "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
        "elk.spacing.nodeNode": "50",
        "elk.layered.spacing.nodeNodeBetweenLayers": "120",
        "elk.spacing.edgeNode": "30",
        "elk.spacing.componentComponent": "160",
        "elk.edgeRouting": "SPLINES",
    };
}

type ElkChild = {
    id: string;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    children?: ElkChild[];
};

export type ElkLayoutResult = {
    /** 套用新 `position` 与 `layout_hint`（容器尺寸）之后的节点副本。 */
    nodes: Node[];
    /** 布局后的整体包围盒（供调用方 fitView）。 */
    width: number;
    height: number;
};

type HintLike = { width?: number; height?: number };

/** 已有尺寸：优先实测 → 节点自带 → `layout_hint`（容器旧网格尺寸也会被读，但容器不采用它）。 */
function sizeOf(n: Node): { width: number; height: number } {
    const hint = (n.data as { layout_hint?: HintLike } | undefined)?.layout_hint;
    const measured = (n as { measured?: HintLike }).measured;
    const w = measured?.width ?? n.width ?? n.style?.width ?? hint?.width ?? 220;
    const h = measured?.height ?? n.height ?? n.style?.height ?? hint?.height ?? 56;
    const num = (v: unknown, d: number) => {
        const x = typeof v === "string" ? Number(v) : v;
        return typeof x === "number" && Number.isFinite(x) && x > 0 ? x : d;
    };
    return { width: num(w, 220), height: num(h, 56) };
}

/** 按 `parentId` 把扁平节点分组：`undefined` 键 = 顶层。父不存在的按顶层处理（图形一致但不丢节点）。 */
function groupByParent(nodes: Node[]): Map<string | undefined, Node[]> {
    const ids = new Set(nodes.map(n => n.id));
    const groups = new Map<string | undefined, Node[]>();
    for (const n of nodes) {
        const key = n.parentId && ids.has(n.parentId) ? n.parentId : undefined;
        const list = groups.get(key) ?? [];
        list.push(n);
        groups.set(key, list);
    }
    return groups;
}

type ElkTreeNode = {
    id: string;
    width?: number;
    height?: number;
    layoutOptions?: Record<string, string>;
    properties?: Record<string, string>;
    children?: ElkTreeNode[];
};

/**
 * 节点 → ELK 嵌套节点。
 *
 * `priority` 用**声明序**：IR 的节点顺序即 trace 出来的构造顺序，`considerModelOrder` 会参考它，
 * 让同层节点的左右次序贴合数据流（不设的话同层顺序随机，读起来会跳）。
 */
function toElkNode(n: Node, order: number, groups: Map<string | undefined, Node[]>): ElkTreeNode {
    const base: ElkTreeNode = { id: n.id, properties: { "elk.layered.priority": String(order) } };
    const kids = groups.get(n.id) ?? [];
    if (!kids.length) {
        const { width, height } = sizeOf(n);
        return { ...base, width, height };
    }
    // 容器**不给尺寸** → 由 ELK 按子节点 + padding 算（尺寸随后写回 layout_hint）
    return {
        ...base,
        layoutOptions: { "elk.padding": PADDING },
        children: kids.map((k, i) => toElkNode(k, i, groups)),
    };
}

function buildElkTree(groups: Map<string | undefined, Node[]>): ElkTreeNode[] {
    return (groups.get(undefined) ?? []).map((n, i) => toElkNode(n, i, groups));
}

/** 收集 ELK 结果：子坐标（相对父）+ 容器尺寸。 */
function collect(res: ElkChild, positions: Map<string, { x: number; y: number }>,
                 sizes: Map<string, { width: number; height: number }>): void {
    for (const c of res.children ?? []) {
        if (typeof c.x === "number" && typeof c.y === "number") {
            positions.set(c.id, { x: c.x, y: c.y });
        }
        if (typeof c.width === "number" && typeof c.height === "number") {
            sizes.set(c.id, { width: c.width, height: c.height });
        }
        collect(c, positions, sizes);
    }
}

/**
 * 恢复**无边兄弟**的声明序（同一列内按声明序重排 y）。
 *
 * 为什么：容器（`nn.Sequential`）的子节点在 IR 里**不连边**（见 `ir_schema` 的容器规则——它们被
 * 原样内联进 `nn.Sequential`，**顺序本身就是数据流**）。ELK 对无边节点不保证顺序，会按自己的
 * 启发式排 → 画布上顺序被打乱（实测 scGPT 的 `decoder.fc` 从 0,1,2,3,4 变成 2,0,1,3,4），
 * 读不出 Sequential 的执行顺序。这里把**同一列**里、且**彼此之间没有任何边**的一组兄弟，
 * 按声明序重新占用 ELK 已经排好的那些 y 槽位（槽位不变 → 不会重叠）。
 */
function restoreSiblingOrder(
    groups: Map<string | undefined, Node[]>, edges: Edge[],
    positions: Map<string, { x: number; y: number }>,
): void {
    const hasEdge = new Set<string>();
    for (const e of edges) {
        hasEdge.add(e.source);
        hasEdge.add(e.target);
    }
    for (const [, kids] of groups) {
        if (kids.length < 2) continue;
        const cols = new Map<number, Node[]>();
        for (const k of kids) {                      // kids 已按 nodes 顺序 = 声明序
            const pos = positions.get(k.id);
            if (!pos) continue;
            const key = Math.round(pos.x);
            const list = cols.get(key) ?? [];
            list.push(k);
            cols.set(key, list);
        }
        for (const [, list] of cols) {
            if (list.length < 2) continue;
            if (list.some(k => hasEdge.has(k.id))) continue;   // 有边的组：ELK 的顺序才有意义
            const slots = list.map(k => positions.get(k.id)!.y).sort((a, b) => a - b);
            list.forEach((k, i) => { positions.get(k.id)!.y = slots[i]; });
        }
    }
}

/**
 * `ancestor` 是不是 `node` 的祖先（沿 `parentId` 上溯）。
 *
 * 用来把**模块输入边**（`模块 M → M 的子孙`，表示「M 的输入喂给该子节点」，见
 * `ir_codegen._module_class._ext_var`）排除在布局之外：它不是"M 的输出流过去"，
 * 交给 ELK 当普通边会把分层带歪——实测 `encoder → encoder_embedding` 让 `encoder` 被推到
 * `value_encoder` 右边 10 层。与 `irShapeVerifier.isAncestor` 同一口径。
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
 * 用 ELK 的 compound 层叠布局重排 IR 画布节点（默认**左→右**）。
 *
 * 失败（ELK 抛错/结构异常）时**原样返回节点**——调用方保留手上的网格布局，不炸画布。
 */
export async function relayoutWithElk(
    nodes: Node[], edges: Edge[], direction: LayoutDirection = "LR",
): Promise<ElkLayoutResult> {
    if (!nodes.length) return { nodes, width: 0, height: 0 };
    const groups = groupByParent(nodes);
    const ids = new Set(nodes.map(n => n.id));
    const byId = new Map(nodes.map(n => [n.id, n]));
    const elkEdges = edges
        .filter(e => ids.has(e.source) && ids.has(e.target))
        .filter(e => !isAncestor(byId, e.source, e.target))   // 模块输入边不是数据流（见 isAncestor）
        .map(e => ({ id: e.id, sources: [e.source], targets: [e.target] }));
    const tree = {
        id: "__root__",
        layoutOptions: layoutOptions(direction),
        children: buildElkTree(groups),
        edges: elkEdges,
    };
    let res: ElkChild;
    try {
        res = (await elk.layout(tree as unknown as Parameters<typeof elk.layout>[0])) as ElkChild;
    } catch (err) {
        // 布局只是呈现：失败就别动用户手上的坐标
        console.warn("ELK 复合布局失败，保留原有坐标：", err);
        return { nodes, width: 0, height: 0 };
    }
    const positions = new Map<string, { x: number; y: number }>();
    const sizes = new Map<string, { width: number; height: number }>();
    collect(res, positions, sizes);
    restoreSiblingOrder(groups, edges, positions);

    // 哪些节点是容器（有子节点）——只有它们采用 ELK 算出的尺寸；叶子/op 保持原尺寸
    const parents = new Set(nodes.map(n => n.parentId).filter((x): x is string => !!x));
    return {
        nodes: nodes.map(n => {
            const pos = positions.get(n.id);
            const size = parents.has(n.id) ? sizes.get(n.id) : undefined;
            if (!pos && !size) return n;
            const data = { ...(n.data || {}) } as Record<string, unknown>;
            if (size) {
                const hint = (data.layout_hint as HintLike | undefined) ?? {};
                data.layout_hint = { ...hint, width: Math.round(size.width), height: Math.round(size.height) };
            }
            return { ...n, ...(pos ? { position: { x: pos.x, y: pos.y } } : {}), data };
        }),
        width: res.width ?? 0,
        height: res.height ?? 0,
    };
}
