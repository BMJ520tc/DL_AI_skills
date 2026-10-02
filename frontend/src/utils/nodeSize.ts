// utils/nodeSize.ts — 画布节点尺寸口径（与后端 ir_graphir 同构，实测校准值）。
// 单独成模块，供 irAdapter（生成 GraphIR）与 graphIR（还原/兜底）共用，避免三处常量漂移。

export const LEAF_W = 220;

/** 无参数节点高度：内边距 + 头部 + 形状行。 */
export const NODE_BASE_H = 32;
/** 有参数节点的固定部分（内边距 + 头部 + 形状行 + 参数区上边距）。 */
export const NODE_PARAM_BASE_H = 78;
/** 每个参数行（标签 + 输入框 + 删除）实测高度。 */
export const PARAM_ROW_H = 22;
/** 参数行数上限：超过则参数区内部滚动，节点不再长高。 */
export const PARAM_MAX_ROWS = 8;

/** 节点自身内容所需高度；params 为节点参数字典。 */
export function ownContentH(params: Record<string, unknown> | null | undefined): number {
    const rows = Object.keys(params ?? {}).length;
    if (rows === 0) return NODE_BASE_H;
    return NODE_PARAM_BASE_H + Math.min(rows, PARAM_MAX_ROWS) * PARAM_ROW_H;
}
