import type { NodeTypes } from "@xyflow/react";
import { LAYER_REGISTRY, NODE_GROUPS } from "../nodes/registry";
import IrNode from "../nodes/IrNode";

export { LAYER_REGISTRY, NODE_GROUPS };

// Bridge the layer registry into React Flow's node component map.
export const nodeTypes: NodeTypes = Object.entries(LAYER_REGISTRY).reduce((acc, [key, Class]) => {
    acc[key] = Class.Component;
    return acc;
}, {} as any);

// 模块四画布节点（IR 快照通用渲染，不属图层注册表：参数 schema 来自节点 data）
nodeTypes["ir"] = IrNode as any;
