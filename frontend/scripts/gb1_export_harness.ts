/**
 * GB-1 导出链路验证：用画布同一套代码生成器把一张固定网络图导出为 PyTorch 代码。
 *
 * 覆盖 M0 验收项「画布搭网络 → 导出代码」的可复核部分：导出用的是前端真实使用的
 * `recursiveCodeGenerator`（src/utils/codeCompile.ts）+ 节点注册表（nodes/registry.ts），
 * 不是另写一份生成器；导出的模型随后在项目独立环境里跑一次训练（见 train_smoke.py）。
 *
 * 用法（在 frontend/ 下）：
 *   npx esbuild scripts/gb1_export_harness.ts --bundle --platform=node --format=cjs \
 *       --outfile=<临时目录>/harness.cjs --log-level=warning
 *   node <临时目录>/harness.cjs --out exported_model.py              # 应用等价图（边带 label + sourceHandle）
 *   node <临时目录>/harness.cjs --out no_label.py --no-labels       # 健壮性图（无 label，模拟导入/历史图）
 */
import fs from "node:fs";

import type { Edge, Node } from "@xyflow/react";
import "./gb1_stubs";
import { recursiveCodeGenerator } from "../src/utils/codeCompile";
import "../src/nodes/registry"; // 触发 registerLayer，填充 LAYER_REGISTRY

const argv = process.argv.slice(2);
const withLabels = !argv.includes("--no-labels");
const outIndex = argv.indexOf("--out");
const outPath = outIndex >= 0 ? argv[outIndex + 1] : null;

// 固定图：Input → Linear(8→16) → ReLU → Linear(16→3)
const rawNodes: Node[] = [
    { id: "in1", type: "input_layer", position: { x: 0, y: 0 }, data: {} },
    { id: "fc1", type: "linear_layer", position: { x: 200, y: 0 }, data: { in_features: 8, out_features: 16, bias: true } },
    { id: "act", type: "relu_layer", position: { x: 400, y: 0 }, data: {} },
    { id: "fc2", type: "linear_layer", position: { x: 600, y: 0 }, data: { in_features: 16, out_features: 3, bias: true } },
];

const wires = [
    { id: "e1", source: "in1", target: "fc1" },
    { id: "e2", source: "fc1", target: "act" },
    { id: "e3", source: "act", target: "fc2" },
];

// 与真实画布一致：连边带 sourceHandle，且 onConnect 会写 label = `out_<source>[_<sourceHandle>]`
const edges: Edge[] = wires.map(w => ({
    ...w,
    type: "custom",
    sourceHandle: "out-0",
    targetHandle: "in-0",
    ...(withLabels ? { data: { label: `out_${w.source}` } } : {}),
}));

const result = recursiveCodeGenerator(rawNodes, edges);
const code = result.code + "\n";
if (!code.includes("class GeneratedModel(nn.Module)")) {
    console.error("导出失败：未生成 GeneratedModel 类");
    process.exit(1);
}

if (outPath) {
    fs.writeFileSync(outPath, code, { encoding: "utf8" });
    console.error(`written: ${outPath} (${code.length} bytes, labels=${withLabels})`);
} else {
    process.stdout.write(code);
}
