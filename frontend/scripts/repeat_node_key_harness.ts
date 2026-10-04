/**
 * 张量 Repeat / 控制流重复块「类型键拆分」自检（前端没有测试框架，用 Node 原生 TS 直跑）。
 *
 * 背景（基底遗留缺陷）：`src/nodes/registry.ts` 里 `repeat_layer` 曾被注册两次——
 * torch_ops 组的 `RepeatNode`（张量 `torch.repeat` 算子）与 control 组的 `RepeatLayerNode`
 * （控制流「重复块」容器）。注册循环后写覆盖前写，于是张量 Repeat 算子在节点库里拖不出来
 * （拖出来的是控制流节点），后端导出也把它当控制流拒绝。修复后张量算子改用独立键
 * `repeat_tensor`，控制流容器继续用 `repeat_layer`。
 *
 * 断言：
 *   1) registry 里 torch_ops 组注册 `repeat_tensor → RepeatNode`、control 组注册
 *      `repeat_layer → RepeatLayerNode`；两个键互不相同，且全表键唯一（不再互相覆盖）；
 *   2) 两个类来自不同文件（pytorch_core/RepeatNode.tsx 与 control_flow/RepeatLayer.tsx），
 *      label 与 paramSchema 参数名各自独立（repeats 文本 vs repetitions 数字）；
 *   3) containerLogic.DEFAULT_CONTAINER_CONFIG 的容器类型集合只含控制流容器键
 *      （repeat_layer / module_list），**不含** repeat_tensor。
 *
 * 说明：Node 的 `--experimental-strip-types` 不处理 JSX，而 registry 依赖的节点组件都是
 * `.tsx`（import 会直接报 "Unknown file extension .tsx"），故这里做**源码结构断言**；
 * 真正的编译期正确性由 `npm run lint` / `npm run build` 与后端用例覆盖。
 *
 * 用法（在 frontend/ 下）：
 *   node --experimental-strip-types --import ./scripts/ts_harness_loader.mjs \
 *       scripts/repeat_node_key_harness.ts
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

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

function readSrc(relative: string): string {
    return readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");
}

const registrySrc = readSrc("../src/nodes/registry.ts");
const containerSrc = readSrc("../src/utils/containerLogic.ts");
const repeatTensorSrc = readSrc("../src/nodes/pytorch_core/RepeatNode.tsx");
const repeatLayerSrc = readSrc("../src/nodes/control_flow/RepeatLayer.tsx");

/** 解析 NODE_GROUPS：组名 → (类型键 → 类名)。 */
function parseGroups(src: string): Map<string, Map<string, string>> {
    const start = src.indexOf("export const NODE_GROUPS");
    const end = src.indexOf("Object.values(NODE_GROUPS)");
    const section = src.slice(start, end);
    const groupStarts = [...section.matchAll(/^ {4}(\w+): \{$/gm)];
    const groups = new Map<string, Map<string, string>>();
    groupStarts.forEach((m, i) => {
        const from = (m.index ?? 0) + m[0].length;
        const to = i + 1 < groupStarts.length ? (groupStarts[i + 1].index ?? section.length) : section.length;
        const body = section.slice(from, to);
        const entries = new Map<string, string>();
        for (const e of body.matchAll(/(?:^|[{,])\s*([a-z_][a-z0-9_]*)\s*:\s*([A-Z]\w*)/gm)) {
            entries.set(e[1], e[2]);
        }
        groups.set(m[1], entries);
    });
    return groups;
}

const groups = parseGroups(registrySrc);
const torchOps = groups.get("torch_ops");
const control = groups.get("control");

// ---------------------------------------------------------------------------
console.log("\n[1] registry.ts —— 两个节点各自注册、类型键不同且全表唯一");
check("解析到 torch_ops 组", !!torchOps);
check("解析到 control 组", !!control);
check("torch_ops 注册 repeat_tensor → RepeatNode", torchOps?.get("repeat_tensor") === "RepeatNode",
    `got=${torchOps?.get("repeat_tensor")}`);
check("control 注册 repeat_layer → RepeatLayerNode", control?.get("repeat_layer") === "RepeatLayerNode",
    `got=${control?.get("repeat_layer")}`);
check("torch_ops 不再占用 repeat_layer 键", torchOps?.has("repeat_layer") === false);
check("control 不再占用 repeat_tensor 键", control?.has("repeat_tensor") === false);
check("两个类型键不相同", "repeat_tensor" !== "repeat_layer");

const allKeys = [...groups.values()].flatMap(entries => [...entries.keys()]);
const dupes = allKeys.filter((key, idx) => allKeys.indexOf(key) !== idx);
check("全表类型键唯一（无后写覆盖）", dupes.length === 0, `dupes=${[...new Set(dupes)].join(",")}`);
check("repeat_tensor 全表只出现一次", allKeys.filter(k => k === "repeat_tensor").length === 1);
check("repeat_layer 全表只出现一次", allKeys.filter(k => k === "repeat_layer").length === 1);

// ---------------------------------------------------------------------------
console.log("\n[2] 节点实现 —— 来源文件 / label / 参数各自独立");
check("registry 从 pytorch_core/RepeatNode 导入张量算子",
    /import\s*\{\s*RepeatNode\s*\}\s*from\s*"\.\/pytorch_core\/RepeatNode"/.test(registrySrc));
check("registry 从 control_flow/RepeatLayer 导入控制流容器",
    /import\s*\{\s*RepeatLayerNode\s*\}\s*from\s*"\.\/control_flow\/RepeatLayer"/.test(registrySrc));

const labelOf = (src: string) => /static label = "([^"]*)"/.exec(src)?.[1] ?? "";
const tensorLabel = labelOf(repeatTensorSrc);
const controlLabel = labelOf(repeatLayerSrc);
check("张量算子 label 非空且为 Repeat", tensorLabel === "Repeat", `got=${tensorLabel}`);
check("控制流容器 label 非空且为 重复层", controlLabel === "重复层", `got=${controlLabel}`);
check("两个节点 label 不相同", tensorLabel !== controlLabel);

const schemaKeysOf = (src: string) => {
    const m = /static paramSchema: Record<string, FieldSpec> = \{([\s\S]*?)\n {4}\}/.exec(src);
    return [...(m?.[1] ?? "").matchAll(/^\s*([a-z_][a-z0-9_]*)\s*:/gm)].map(x => x[1]);
};
const tensorParams = schemaKeysOf(repeatTensorSrc);
const controlParams = schemaKeysOf(repeatLayerSrc);
check("张量算子参数名为 repeats（文本）", tensorParams.includes("repeats"), `got=${tensorParams.join(",")}`);
check("控制流容器参数名为 repetitions（数字）", controlParams.includes("repetitions"), `got=${controlParams.join(",")}`);
check("两者参数名不重叠", tensorParams.every(k => !controlParams.includes(k)));
check("张量算子 forward 生成 .repeat(...) 调用",
    /\.repeat\(\$\{reps\}\)/.test(repeatTensorSrc) || /\.repeat\(/.test(repeatTensorSrc));

// ---------------------------------------------------------------------------
console.log("\n[3] containerLogic —— 容器类型集合只含控制流键");
const typesMatch = /types:\s*new Set\(\[([^\]]*)\]\)/.exec(containerSrc);
const containerTypes = (typesMatch?.[1] ?? "")
    .split(",")
    .map(s => s.trim().replace(/^["']|["']$/g, ""))
    .filter(Boolean);
check("解析到 DEFAULT_CONTAINER_CONFIG.types", containerTypes.length > 0, `raw=${typesMatch?.[1]}`);
check("容器集合只含 repeat_layer 与 module_list",
    [...containerTypes].sort().join(",") === "module_list,repeat_layer", `got=${containerTypes.join(",")}`);
check("容器集合不含张量算子的 repeat_tensor", !containerTypes.includes("repeat_tensor"));
check("容器 capacities 也按控制流键登记",
    /capacities:\s*\{[\s\S]*?repeat_layer:/.test(containerSrc) && /capacities:\s*\{[\s\S]*?module_list:/.test(containerSrc));

// ---------------------------------------------------------------------------
console.log(`\n结果：${passed} passed, ${failed.length} failed`);
if (failed.length) {
    failed.forEach(name => console.error(`FAILED: ${name}`));
    process.exit(1);
}
console.log("ALL CHECKS PASSED");
