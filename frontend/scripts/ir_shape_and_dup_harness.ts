/**
 * 结构化画布修复自检（前端仓库没有测试框架，用 Node 原生 TS 直跑）。
 *
 * 覆盖两处新增的纯函数：
 *   1) src/utils/irShapeVerifier.ts — IR 连线形状判据（维度数/未知维/缺形状不算失败/
 *      连线校验 isValidConnection 口径）；
 *   2) src/features/editor/hooks/useGraphState.ts 的 duplicateSelection — 节点复制的
 *      边重映射（内部连线随选中集合一起复制、跨出选区的边不复制）与 data 深拷贝。
 *
 * 用法（在 frontend/ 下）：
 *   node --experimental-strip-types --import ./scripts/ts_harness_loader.mjs \
 *       scripts/ir_shape_and_dup_harness.ts
 */
import type { Edge, Node } from "@xyflow/react";

import { isIrConnectionCompatible, shapesCompatible, verifyIRShapes } from "../src/utils/irShapeVerifier.ts";
import { duplicateSelection } from "../src/features/editor/hooks/useGraphState.ts";

let passed = 0;
const failed: string[] = [];

function check(name: string, condition: boolean, detail?: string) {
    if (condition) {
        passed += 1;
        console.log(`  PASS  ${name}`);
    } else {
        failed.push(name);
        console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ""}`);
    }
}

const irNode = (id: string, data: Record<string, unknown>): Node =>
    ({ id, type: "ir", position: { x: 0, y: 0 }, data }) as Node;

const wire = (id: string, source: string, target: string): Edge => ({ id, source, target }) as Edge;

// ---------------------------------------------------------------------------
console.log("\n[1] shapesCompatible —— 形状一致性判据");
check("完全相同的形状一致", shapesCompatible([1, 3, 224, 224], [1, 3, 224, 224]) === true);
check("同维度不同值不一致", shapesCompatible([1, 3, 224, 224], [1, 3, 112, 112]) === false);
check("维度数不同不一致", shapesCompatible([1, 3, 224, 224], [1, 3, 224]) === false);
check("未知维（-1）跳过该维", shapesCompatible([-1, 3, 224, 224], [8, 3, 224, 224]) === true);
check("未知维（null）跳过该维", shapesCompatible([null, 3], [1, 3]) === true);
check("一端未知（null）不判失败", shapesCompatible(null, [1, 3]) === true);
check("一端未知（空数组）不判失败", shapesCompatible([], [1, 3]) === true);
check("两端未知不判失败", shapesCompatible(undefined, null) === true);

// ---------------------------------------------------------------------------
console.log("\n[2] verifyIRShapes —— 连线校验与缺形状口径");
const shapeNodes: Node[] = [
    irNode("n1", { kind: "module", label: "Up", __in_shape: [1, 3], __out_shape: [1, 512] }),
    irNode("n2", { kind: "leaf", label: "Bad", __in_shape: [1, 256], __out_shape: [1, 256] }),
    irNode("n3", { kind: "leaf", label: "Good", __in_shape: [1, 256], __out_shape: [1, 10] }),
    irNode("n4", { kind: "op", label: "Op" }), // op 节点缺形状不计入 missingShapes
    irNode("n5", { kind: "leaf", label: "HalfKnown", __in_shape: [1, 256] }), // 缺 __out_shape
    irNode("n6", { kind: "leaf", label: "Unknown" }),
];
const shapeEdges: Edge[] = [
    wire("e1", "n1", "n2"), // [1,512] -> 期望 [1,256]：不匹配
    wire("e2", "n2", "n3"), // [1,256] -> [1,256]：匹配
    wire("e3", "n4", "n3"), // 源缺形状：跳过，不算失败
    wire("e4", "n6", "n3"), // 两端都有未知：跳过
];
const verified = verifyIRShapes(shapeNodes, shapeEdges);
check("有形状不一致 → ok=false", verified.ok === false);
check("同一目标的多条不匹配入边合并计数", verified.failures.length === 1, `failures=${verified.failures.length}`);
check("失败落在目标节点 n2", verified.failures[0]?.nodeId === "n2", verified.failures[0]?.nodeId);
check("失败记录上游来源", JSON.stringify(verified.failures[0]?.upstream) === JSON.stringify(["n1"]));
check("失败记录输入形状", JSON.stringify(verified.failures[0]?.inputShapes) === JSON.stringify([[1, 512]]));
check("失败文案包含两侧形状", (verified.failures[0]?.error ?? "").includes("[1, 512]"));
check("shapes 用 __out_shape 回写（n1/n2/n3）", Object.keys(verified.shapes).sort().join(",") === "n1,n2,n3");
check("op 节点缺形状不计入 missingShapes", !verified.missingShapes.includes("n4"));
check("半缺形状计入 missingShapes", verified.missingShapes.includes("n5"));
check("完全无形状计入 missingShapes", verified.missingShapes.includes("n6"));
check("形状齐全的节点不算缺形状", !verified.missingShapes.includes("n1") && !verified.missingShapes.includes("n3"));

// 多入边聚合：两条都不匹配 → 仍是一条失败（不刷屏）
const multi = verifyIRShapes(
    [
        irNode("a", { kind: "leaf", __out_shape: [1, 8] }),
        irNode("b", { kind: "leaf", __out_shape: [1, 16] }),
        irNode("c", { kind: "leaf", __in_shape: [1, 4] }),
    ],
    [wire("m1", "a", "c"), wire("m2", "b", "c")],
);
check("多入边不匹配合并为一条", multi.failures.length === 1, `failures=${multi.failures.length}`);
check("多入边合并保留两个上游", (multi.failures[0]?.upstream ?? []).length === 2);

// ---------------------------------------------------------------------------
console.log("\n[3] isIrConnectionCompatible —— isValidConnection / onConnect 共用口径");
check("两端已知且不一致 → 拒绝", isIrConnectionCompatible(shapeNodes, { source: "n1", target: "n2" }) === false);
check("两端已知且一致 → 放行", isIrConnectionCompatible(shapeNodes, { source: "n2", target: "n3" }) === true);
check("源缺形状 → 放行", isIrConnectionCompatible(shapeNodes, { source: "n6", target: "n3" }) === true);
check("目标缺形状 → 放行", isIrConnectionCompatible(shapeNodes, { source: "n2", target: "n6" }) === true);
check("自连 → 放行", isIrConnectionCompatible(shapeNodes, { source: "n2", target: "n2" }) === true);
check("端点不在图上 → 放行", isIrConnectionCompatible(shapeNodes, { source: "zzz", target: "n2" }) === true);
check("未知维不拦（只看已知维）", isIrConnectionCompatible(
    [irNode("u", { __out_shape: [-1, 256] }), irNode("v", { __in_shape: [4, 256] })],
    { source: "u", target: "v" },
) === true);

// ---------------------------------------------------------------------------
console.log("\n[4] duplicateSelection —— 复制选中节点与边重映射");
let seq = 100;
const newId = () => `node-${seq++}`;
const dupNodes: Node[] = [
    { id: "a", type: "ir", position: { x: 10, y: 10 }, selected: true, data: { label: "A", params: { k: 1 } } } as Node,
    { id: "b", type: "ir", position: { x: 100, y: 100 }, selected: true, data: { label: "B" } } as Node,
    { id: "c", type: "ir", position: { x: 200, y: 200 }, data: { label: "C" } } as Node,
];
const dupEdges: Edge[] = [wire("e1", "a", "b"), wire("e2", "b", "c"), wire("e3", "c", "a")];

const duplicated = duplicateSelection(dupNodes, dupEdges, ["a", "b"], newId);
check("有选中 → 返回结果", duplicated !== null);
if (duplicated) {
    const copyA = duplicated.nodes.find(n => n.id === "node-100");
    const copyB = duplicated.nodes.find(n => n.id === "node-101");
    check("节点总数 = 原 3 + 新 2", duplicated.nodes.length === 5, `nodes=${duplicated.nodes.length}`);
    check("新 id 由 newId() 生成且唯一", !!copyA && !!copyB);
    check("位置偏移 +40/+40（a）", copyA?.position.x === 50 && copyA?.position.y === 50, JSON.stringify(copyA?.position));
    check("位置偏移 +40/+40（b）", copyB?.position.x === 140 && copyB?.position.y === 140, JSON.stringify(copyB?.position));
    check("复制后选中新节点", duplicated.nodes.filter(n => n.selected).map(n => n.id).sort().join(",") === "node-100,node-101");
    check("原节点取消选中", dupNodes[0].selected === true && duplicated.nodes.find(n => n.id === "a")?.selected === false);
    check("未选中节点不受影响", duplicated.nodes.find(n => n.id === "c")?.selected !== true);

    // data 深拷贝：对象与内层 params 都不是同一个引用，但值相同
    const originalA = duplicated.nodes.find(n => n.id === "a");
    check("data 深拷贝（外层对象独立）", copyA?.data !== originalA?.data);
    check("data 深拷贝（内层 params 独立）", (copyA?.data as { params?: unknown })?.params !== (originalA?.data as { params?: unknown })?.params);
    check("data 深拷贝（值保持一致）", (copyA?.data as { params?: { k?: number } })?.params?.k === 1);
    check("data.label 一并复制", (copyA?.data as { label?: string })?.label === "A");

    const copiedEdges = duplicated.edges.filter(e => !dupEdges.some(original => original.id === e.id));
    check("只复制选区内部连线（1 条）", copiedEdges.length === 1, `copied=${copiedEdges.length}`);
    check("内部连线重映射到新节点 id", copiedEdges[0]?.source === "node-100" && copiedEdges[0]?.target === "node-101");
    check("复制边使用新 id（不与原边撞 id）", copiedEdges[0]?.id !== "e1");
    check("跨出选区的边不复制（b→c / c→a 未复制）", dupEdges.length === 3 && duplicated.edges.length === 4);
}

console.log("\n[5] duplicateSelection —— 空选区");
check("无选中 → 返回 null（调用方据此不动作）", duplicateSelection(dupNodes, dupEdges, [], newId) === null);

// ---------------------------------------------------------------------------
console.log(`\n结果：${passed} passed, ${failed.length} failed`);
if (failed.length) {
    failed.forEach(name => console.error(`FAILED: ${name}`));
    process.exit(1);
}
console.log("ALL CHECKS PASSED");
