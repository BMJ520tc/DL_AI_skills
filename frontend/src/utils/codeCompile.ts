import type { Edge, Node } from "@xyflow/react";
import type { ModuleRefData } from "../nodes/ModuleRefNode";
import { LAYER_REGISTRY } from "./layerRegistry";
import { getModule, type SavedModule } from "./moduleRegistry";

export type CodeSpan = {
    line: number;
    kind: "header" | "init" | "forward" | "return";
    nodeId?: string;
    edgeIds?: string[];
};

export type CodeGenResult = {
    code: string;
    spans: CodeSpan[];
};

// Sanitize arbitrary labels/ids into valid Python identifiers.
export function sanitizeIdent(name: string): string {
    const cleaned = name.replace(/[^A-Za-z0-9_]/g, "_");
    if (!cleaned.length) return "_x";
    const safe = /^[A-Za-z_]/.test(cleaned[0]) ? cleaned : `_${cleaned}`;
    return safe;
}

/** Python 字符串字面量：JSON 转义是合法 Python 字符串字面量的子集，
 *  正确处理引号 / 反斜杠 / 换行（直接拼 `"${value}"` 会生成非法代码）。 */
export function toPythonString(value: unknown): string {
    return JSON.stringify(String(value));
}

/** 值 → Python 字面量（嵌套 list/dict 递归；阶段4 4b 的 list/dict 参数导出用）。 */
export function toPythonLiteral(value: unknown): string {
    if (value === null || value === undefined) return "None";
    if (typeof value === "boolean") return value ? "True" : "False";
    if (typeof value === "string") return toPythonString(value);
    if (typeof value === "number") return Number.isFinite(value) ? `${value}` : "None";
    if (Array.isArray(value)) return `[${value.map(toPythonLiteral).join(", ")}]`;
    if (typeof value === "object") {
        return `{${Object.entries(value as Record<string, unknown>)
            .map(([k, v]) => `${toPythonString(k)}: ${toPythonLiteral(v)}`)
            .join(", ")}}`;
    }
    return "None";
}

/** 内联模块的类名解析表：`recursiveCodeGenerator` 预扫描后填入（模块重名时加模块 id 后缀）。
 *  `ModuleRefNode.getInitCode` 用它取实际类名，保证类定义与实例化两处一致、不被同名类遮蔽。 */
const moduleClassNames = new Map<string, string>();

export function setModuleClassNames(names: Map<string, string>): void {
    moduleClassNames.clear();
    for (const [id, name] of names) moduleClassNames.set(id, name);
}

export function moduleClassNameFor(moduleId: string): string | undefined {
    return moduleClassNames.get(moduleId);
}

export function getRootGraph(nodes: Node[], edges: Edge[]) {
    // For now, we assume the provided nodes and edges ARE the root graph.
    // In a more complex setup where 'nodes' might contain everything including nested subgraphs (not how ReactFlow works usually),
    // we would filter. But here 'nodes' is the current view.
    return { rootNodes: nodes, rootEdges: edges };
}

export function createCustomComponentDAG(
    id: string,
    nodes: Node[],
    order: string[],
    color: Record<string, number>,
): boolean {
    // Color: 0 for unvisited, 1 for visiting, 2 for done visiting

    color[id] = 1; // Visiting

    let ok: boolean = true;
    nodes.forEach(child => {
        if (child.type && child.type === "module_ref") {
            const module_data = child.data as ModuleRefData;
            const module_id = module_data.moduleId as string;

            if (color[module_id] === 1) {
                return false; // Make sure to test this.
            } else if (color[module_id] !== 2) {
                const savedModule = getModule(module_id);
                const internalNodes: Node[] = savedModule?.internalNodes || [];
                ok = ok && createCustomComponentDAG(module_id, internalNodes, order, color);
            }
        }
    });

    order.push(id);
    color[id] = 2; // Visited

    return ok;
}

// This function works on the module level code generator and uses the generate main code function to generate code for individual modules.
// It first sorts the module ids based on the way they should be arranged in the code and then writes the code.
export function recursiveCodeGenerator(nodes: Node[], edges: Edge[]): CodeGenResult {
    const order: string[] = [];
    const color: Record<string, number> = {};
    createCustomComponentDAG("0", nodes, order, color);

    // *
    // For each module id, get it's code, shift it's lines by the number of previous lines
    // and add it to the main codegenresult object.
    //
    // /

    const lines: string[] = [];
    const spans: CodeSpan[] = [];

    lines.push("import torch", "import torch.nn as nn");
    spans.push({ line: 1, kind: "header" }, { line: 2, kind: "header" });

    const generatedCode: CodeGenResult = { code: lines.join("\n"), spans };

    let moduleNodes: Node[];
    let moduleEdges: Edge[];
    let moduleName: string;
    let lineOffset: number = lines.length;

    // 预扫描：给每个被引用的模块定一个不冲突的类名（模块重名 → 加模块 id 后缀）。
    // 不做这一步时，两个同名模块会内联出两个同名类，后者遮蔽前者（两个引用实例化同一模型）。
    const classNames = new Map<string, string>();
    const usedNames = new Set<string>();
    order.forEach(moduleId => {
        if (moduleId === "0") return;
        const saved = getModule(moduleId);
        let name = saved ? sanitizeIdent(saved.name) : "unknownModule";
        if (usedNames.has(name)) name = sanitizeIdent(`${name}_${sanitizeIdent(moduleId)}`);
        usedNames.add(name);
        classNames.set(moduleId, name);
    });
    setModuleClassNames(classNames);

    order.forEach(moduleId => {
        let savedModule: SavedModule | null = null;
        if (moduleId === "0") {
            // Accidental clash?
            moduleNodes = nodes;
            moduleEdges = edges;
            moduleName = "GeneratedModel";
        } else {
            savedModule = getModule(moduleId) ?? null;
            if (savedModule) {
                moduleNodes = savedModule?.internalNodes || [];
                moduleEdges = savedModule?.internalEdges || [];
                moduleName = classNames.get(moduleId) ?? sanitizeIdent(savedModule.name);
            } else {
                console.warn(`Module with ID ${moduleId} not found or contract missing.`);
                moduleNodes = [];
                moduleEdges = [];
                moduleName = classNames.get(moduleId) ?? "unknownModule";
            }
        }
        generatedCode.code += "\n\n\n";
        lineOffset += 2;

        const moduleCode = generateMainCode(moduleNodes, moduleEdges, moduleName, lineOffset, savedModule);

        generatedCode.code += moduleCode.code;
        generatedCode.spans.push(...moduleCode.spans);
        lineOffset += moduleCode.code.split("\n").length;
    });
    return generatedCode;
}

// Converts graphs into 3 components: initlines, forward lines, returnVar
export function compileGraphToScript(
    nodes: Node[],
    edges: Edge[],
    variablePrefix: string = "", // Prefix for local variables to avoid namespace collisions
    variableMap?: Record<string, Array<{ nodeId: string; paramName: string }>>, // Variable mapping for custom modules
) {
    if (nodes.length === 0) return { code: "class Model(nn.Module):\n    pass", spans: [] };
    const initLines: { text: string; span?: Omit<CodeSpan, "line"> }[] = [];
    const forwardLines: { text: string; span?: Omit<CodeSpan, "line"> }[] = [];
    const nodeOutputMap: Record<string, string[]> = {};
    // 1. Build Adjacency List
    const adj: Record<string, string[]> = {};
    const inDegree: Record<string, number> = {};
    nodes.forEach(n => {
        adj[n.id] = [];
        inDegree[n.id] = 0;
    });
    edges.forEach(e => {
        if (adj[e.source]) adj[e.source].push(e.target);
        if (inDegree[e.target] !== undefined) inDegree[e.target]++;
    });

    // 2. Topological Sort
    const queue: string[] = nodes.filter(n => inDegree[n.id] === 0).map(n => n.id);
    const sortedIds: string[] = [];

    while (queue.length > 0) {
        const u = queue.shift()!;
        sortedIds.push(u);
        if (adj[u]) {
            adj[u].forEach(v => {
                inDegree[v]--;
                if (inDegree[v] === 0) queue.push(v);
            });
        }
    }

    const finalOrderIds =
        sortedIds.length === nodes.length
            ? sortedIds
            : [...sortedIds, ...nodes.map(n => n.id).filter(id => !sortedIds.includes(id))];
    const sortedNodes = finalOrderIds.map(id => nodes.find(n => n.id === id)!);

    // This code creates a list of all the input and output edges of a node
    const incomingEdges: Record<string, Edge[]> = {};
    const outgoingEdges: Record<string, Edge[]> = {};

    nodes.forEach(n => {
        incomingEdges[n.id] = [];
        outgoingEdges[n.id] = [];
    });

    // 容器的**内部边界边**（容器与它自己的直接子节点之间那条 in-internal/out-internal 连线）
    // 不能算作容器的输入/输出边：否则容器的 forward 末行会写成内部边的名字、整图找不到输出节点
    // → 导出代码 `return x`（模型不接输入）。子节点一侧不受影响（它的子节点集合为空）。
    const childrenOf = new Map<string, Set<string>>();
    nodes.forEach(n => {
        if (!n.parentId) return;
        const set = childrenOf.get(n.parentId) ?? new Set<string>();
        set.add(n.id);
        childrenOf.set(n.parentId, set);
    });
    const isInternalBoundary = (edge: Edge, ownerId: string): boolean => {
        const kids = childrenOf.get(ownerId);
        if (!kids || kids.size === 0) return false;
        const other = edge.target === ownerId ? edge.source : edge.target;
        return kids.has(other);
    };

    edges.forEach(e => {
        if (incomingEdges[e.target] && !isInternalBoundary(e, e.target)) incomingEdges[e.target].push(e);
        if (outgoingEdges[e.source] && !isInternalBoundary(e, e.source)) outgoingEdges[e.source].push(e);
    });

    const getVarName = (base: string) => sanitizeIdent(`${variablePrefix}${base}`);
    // 边的变量名：优先用画布写入的 label（onConnect 固定写 `out_<source>[_<handle>]`）；
    // 没有 label 的边（导入的图 / 历史数据 / 结构化项目）退回按边 id 生成稳定唯一名——
    // 若像原先那样按「消费侧/生产侧各自的下标」兜底，同一根边两侧会得到不同名字，
    // 导出的 forward 会引用未定义变量、代码不可编译（GB-1 验证暴露）。
    const edgeVarName = (edge: Edge): string => {
        const label = edge.data?.label;
        // label 为动态 data 字段；非字符串时按模板插值的字符串化结果处理，保持原名。
        if (label) return getVarName(String(label));
        const stable = edge.id ? sanitizeIdent(`edge_${edge.id}`) : getVarName("edge");
        return sanitizeIdent(`${variablePrefix}${stable}`);
    };

    /** 按 targetHandle 序号排序输入边（in-0 / in-1 / …），与后端 `_ordered_in_edges` 同口径。
     *  不对称算子按位置消费输入，若沿用边数组顺序，句柄顺序与数组顺序不一致时操作数会颠倒。 */
    const orderedInEdges = (edges: Edge[]): Edge[] => {
        const seqOf = (handle: unknown): number | null => {
            if (typeof handle !== "string") return null;
            const matched = /(\d+)\s*$/.exec(handle);
            return matched ? Number(matched[1]) : null;
        };
        return edges
            .map((edge, idx) => ({ edge, idx, seq: seqOf(edge.targetHandle) }))
            .sort((a, b) => {
                if (a.seq !== null && b.seq !== null) return a.seq - b.seq || a.idx - b.idx;
                if (a.seq !== null) return -1;
                if (b.seq !== null) return 1;
                return a.idx - b.idx;
            })
            .map(entry => entry.edge);
    };

    const seedLines: { text: string; span?: Omit<CodeSpan, "line"> }[] = [];
    nodes
        .filter(n => (incomingEdges[n.id] ?? []).length === 0)
        .forEach(n => {
            const parentIsPresent = n.parentId && nodes.some(p => p.id === n.parentId);
            if (parentIsPresent) return;
            const outs = outgoingEdges[n.id] ?? [];
            outs.forEach(e => {
                const name = edgeVarName(e);
                seedLines.push({
                    text: `        ${name} = x  # input passthrough`,
                    span: { kind: "forward", nodeId: n.id, edgeIds: [e.id] },
                });
            });
        });

    sortedNodes.forEach(node => {
        const layerName = `${sanitizeIdent(node.id)}_layer`;
        const type = node.type;
        if (!type || !LAYER_REGISTRY[type]) return;

        const ClassRef = LAYER_REGISTRY[type];
        const parentNode = node.parentId ? nodes.find(n => n.id === node.parentId) : null;
        const parentClass = parentNode ? LAYER_REGISTRY[parentNode.type!] : null;
        const shouldGenerateInit = !parentNode || !parentClass || !parentClass.encapsulatesChildInit;
        if (shouldGenerateInit) {
            let line = ClassRef.getInitCode(node.data, layerName, variableMap);
            
            // Apply variable mapping if provided
            if (variableMap) {
                for (const varName in variableMap) {
                    const targets = variableMap[varName];
                    for (const target of targets) {
                        if (target.nodeId === node.id) {
                            // Replace the parameter value with the variable name.
                            // 取值可能含逗号（数组/字典/带逗号的字符串），故不能简单用 [^,)]+：
                            // 按「普通字符 | 引号串 | [...] | {...}」一轮一轮吃掉整个取值。
                            const esc = target.paramName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                            const paramRegex = new RegExp(
                                `\\b${esc}\\s*=\\s*(?:[^,()\\[\\]{}"']+|"[^"]*"|'[^']*'|\\[[^\\[\\]]*\\]|\\{[^{}]*\\})+`,
                                'g');
                            line = line.replace(paramRegex, `${target.paramName}=${varName}`);
                        }
                    }
                }
            }
            
            initLines.push({ text: `        ${line}`, span: { kind: "init", nodeId: node.id } });
        }
        const shouldGenerateForward = !parentNode;
        if (shouldGenerateForward) {
            const inEdges = orderedInEdges(incomingEdges[node.id]);
            const inputNames =
                inEdges.length === 0 ? ["x"] : inEdges.map(e => edgeVarName(e));

            const outEdges = outgoingEdges[node.id];
            const handlesSpec =
                typeof ClassRef.handles === "function" ? ClassRef.handles(node.data) : ClassRef.handles;
            const sourceHandles = handlesSpec?.sources && handlesSpec.sources.length ? handlesSpec.sources : [];
            // 有声明源句柄的节点：先按 sourceHandle 认领出边；旧图/历史图的边可能没有
            // sourceHandle，此时按顺序认领一条——否则会凭空造名，与消费侧（按 label/边 id
            // 取名）对不上，导出的 forward 引用未定义变量（与后端 _forward_line 同口径）。
            const pending = [...outEdges];
            const outputNames = sourceHandles.length
                ? sourceHandles.map((handleId, idx) => {
                      let matchIdx = pending.findIndex(e => e.sourceHandle === handleId);
                      if (matchIdx < 0 && pending.length) matchIdx = 0;
                      if (matchIdx >= 0) {
                          const [claimed] = pending.splice(matchIdx, 1);
                          return edgeVarName(claimed);
                      }
                      return sanitizeIdent(`out_${node.id}_${handleId ?? idx}`);
                  })
                : outEdges.length === 0
                  ? [sanitizeIdent(`out_${node.id}`)]
                  : outEdges.map(e => edgeVarName(e));
            nodeOutputMap[node.id] = outputNames;
            const forward_line = ClassRef.getForwardCode(node.data, layerName, inputNames, outputNames);

            forwardLines.push({
                text: `        ${forward_line}`,
                span: { kind: "forward", nodeId: node.id, edgeIds: outEdges.map(e => e.id) },
            });
        }
    });
    const terminalNodes = sortedNodes.filter(n => {
        const isTopLevel = !n.parentId || !nodes.some(p => p.id === n.parentId);
        const hasNoOutputs = (outgoingEdges[n.id] || []).length === 0;
        return isTopLevel && hasNoOutputs;
    });
    let returnVar = "x";
    const allTerminalOutputs: string[] = [];
    terminalNodes.forEach(n => {
        const outputs = nodeOutputMap[n.id] || [];
        allTerminalOutputs.push(...outputs);
    });
    if (allTerminalOutputs.length === 1) {
        returnVar = allTerminalOutputs[0];
    } else if (allTerminalOutputs.length > 1) {
        returnVar = `(${allTerminalOutputs.join(", ")})`;
    }
    return { initLines, forwardLines, returnVar };
}

// Code can be made more efficient by passing CodeGenResult by reference. Offset won't be required.
export function generateMainCode(
    nodes: Node[], 
    edges: Edge[], 
    name: string, 
    lineOffset: number,
    savedModule?: SavedModule | null // Optional: saved module with variableSchema and variableMap
): CodeGenResult {
    const lines: string[] = [];
    const spans: CodeSpan[] = [];

    lines.push(`class ${name}(nn.Module):`);
    spans.push({ line: 1 + lineOffset, kind: "header" });

    // Build __init__ signature with variable schema parameters
    const variableSchema = savedModule?.variableSchema || {};
    // spec 来自可持久化/后端注入的模块 schema，运行时 type 可能取约定外的字符串，
    // 因此按 unknown 比较，保持原有判定结果。
    const pyLiteral = (spec: { type?: unknown } | undefined, value: unknown): string => {
        if (value === undefined || value === null) return "None";
        const t = spec?.type;
        // text/select 与历史 string 同为字符串取值，一律加引号（4b-1 口径）
        if (t === "string" || t === "text" || t === "select") return toPythonString(value);
        if (t === "boolean") return value ? "True" : "False";
        if (t === "array" || t === "dict") return toPythonLiteral(value);
        return `${value}`;
    };

    const variableParams = Object.entries(variableSchema).map(([varName, spec]) => {
        const required = spec?.required ?? false;
        const defaultValue = spec?.defaultValue;
        if (required) return varName;
        if (defaultValue !== undefined) return `${varName}=${pyLiteral(spec, defaultValue)}`;
        return `${varName}=None`;
    });

    const initSignature = variableParams.length > 0 
        ? `    def __init__(self, ${variableParams.join(", ")}):` 
        : "    def __init__(self):";
    
    lines.push(initSignature);
    lines.push("        super().__init__()");

    // Compile Main Graph with variable mapping
    const { initLines, forwardLines, returnVar } = compileGraphToScript(
        nodes, 
        edges, 
        "", 
        savedModule?.variableMap
    );

    // Stitch Init
    initLines?.forEach(l => {
        lines.push(l.text);
        if (l.span) spans.push({ ...l.span, line: lines.length + lineOffset });
    });
    lines.push("");

    lines.push("    def forward(self, x):");
    forwardLines?.forEach(l => {
        lines.push(l.text);
        if (l.span) spans.push({ ...l.span, line: lines.length + lineOffset });
    });
    lines.push(`        return ${returnVar}`);
    spans.push({ line: lines.length + lineOffset, kind: "return" });

    return { code: lines.join("\n"), spans };
}
