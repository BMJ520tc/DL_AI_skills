import { type Edge, type Node } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { listModules as listBackendModules, type ModuleItem } from "../../../api/client";
import type { FieldSpec } from "../../../node_gen/BaseClass";
import type { ModuleRefData } from "../../../nodes/ModuleRefNode";
import type { GraphIR } from "../../../types/graph";
import { sanitizeIdent } from "../../../utils/codeCompile";
import { applyGraphIR, buildGraphIR } from "../../../utils/graphIR";
import {
    deleteModule,
    getModule,
    listModules,
    resolveModuleName,
    saveExistingModule,
    saveModule,
    setTransientModules,
    type SavedModule,
} from "../../../utils/moduleRegistry";
import { getActiveModule, popModule, pushModule, type OpenModule } from "../../../utils/stackNavigation";

/** 浅拷贝并去掉 saveModule 自行管理的元数据字段。
 *  id 由 saveModule 处理，createdAt/updatedAt 由它重写，因此保存入参不含时间戳。 */
function toModuleSaveInput(
    mod: SavedModule,
): Omit<SavedModule, "id" | "createdAt" | "updatedAt"> & { id?: string } {
    const copy: Record<string, unknown> = { ...mod };
    delete copy.createdAt;
    delete copy.updatedAt;
    // 运行时形状与目标类型一致：仅移除了上面两个时间戳键。
    return copy as Omit<SavedModule, "id" | "createdAt" | "updatedAt"> & { id?: string };
}

/** 后端 params_schema（{参数名: {type, default}}）→ 基底 FieldSpec 参数面板描述。
 *  可映射类型：bool→boolean、int→number(step 1)、float→number、str→text、
 *  list→array、dict→dict（JSON 控件，阶段4 4b 补齐 list/dict）；null/未知类型不进面板。 */
function paramsSchemaToVariableSchema(
    raw: string | null | undefined,
): Record<string, FieldSpec> | undefined {
    if (!raw) return undefined;
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        return undefined;
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return undefined;
    const entries = parsed as Record<string, unknown>;
    const schema: Record<string, FieldSpec> = {};
    for (const [name, entry] of Object.entries(entries)) {
        if (!entry || typeof entry !== "object") continue;
        const { type, default: defaultValue } = entry as { type?: unknown; default?: unknown };
        switch (type) {
            case "bool":
                schema[name] = {
                    type: "boolean",
                    required: false,
                    defaultValue: typeof defaultValue === "boolean" ? defaultValue : false,
                };
                break;
            case "int":
                schema[name] = {
                    type: "number",
                    required: false,
                    defaultValue: typeof defaultValue === "number" ? defaultValue : 0,
                    step: 1,
                };
                break;
            case "float":
                schema[name] = {
                    type: "number",
                    required: false,
                    defaultValue: typeof defaultValue === "number" ? defaultValue : 0,
                };
                break;
            case "str":
                schema[name] = {
                    type: "text",
                    required: false,
                    defaultValue: typeof defaultValue === "string" ? defaultValue : "",
                };
                break;
            case "list":
                schema[name] = {
                    type: "array",
                    required: false,
                    defaultValue: Array.isArray(defaultValue) ? defaultValue : [],
                };
                break;
            case "dict":
                schema[name] = {
                    type: "dict",
                    required: false,
                    defaultValue: defaultValue && typeof defaultValue === "object"
                        && !Array.isArray(defaultValue) ? defaultValue : {},
                };
                break;
            // null/未知类型：跳过（见函数说明）。
        }
    }
    return Object.keys(schema).length ? schema : undefined;
}

/** 后端 ModuleItem.saved_module_compat → 基底 SavedModule（只读注入，不落 localStorage）。 */
function compatToSavedModule(item: ModuleItem): SavedModule | null {
    if (!item.saved_module_compat) return null;
    let raw: unknown;
    try {
        raw = JSON.parse(item.saved_module_compat);
    } catch {
        return null;
    }
    if (!raw || typeof raw !== "object") return null;
    const m = raw as Record<string, unknown>;
    const graph = m.graph as GraphIR | undefined;
    if (!graph || typeof graph !== "object" || !Array.isArray(graph.nodes)) return null;
    const handlesRaw = (m.handles ?? {}) as { inputs?: unknown; outputs?: unknown };
    const toArr = (value: unknown): string[] =>
        Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
    const inputs = toArr(handlesRaw.inputs);
    const outputs = toArr(handlesRaw.outputs);
    return {
        id: typeof m.id === "string" && m.id ? m.id : `${item.module_id}@${item.module_version}`,
        name: typeof m.name === "string" && m.name ? m.name : item.name ?? item.module_id,
        version: typeof m.version === "string" && m.version ? m.version : item.module_version,
        graph,
        handles: {
            inputs: inputs.length ? inputs : ["in"],
            outputs: outputs.length ? outputs : ["out"],
        },
        description: typeof m.description === "string" ? m.description : item.description ?? undefined,
        createdAt: typeof m.createdAt === "string" ? m.createdAt : item.created_at,
        updatedAt: typeof m.updatedAt === "string" ? m.updatedAt : item.updated_at,
        variableSchema: paramsSchemaToVariableSchema(item.params_schema),
        origin: "backend",
    };
}

export type ModuleHandles = { inputs: string[]; outputs: string[] };

export type PromotableParam = {
    nodeId: string;
    nodeLabel: string;
    paramName: string;
    spec: FieldSpec;
};
type UseModuleSystemProps = {
    nodes: Node[];
    edges: Edge[];
    setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
    setModules?: React.Dispatch<React.SetStateAction<SavedModule[]>>;
    getNodeSchema?: (nodeType: string) => Record<string, FieldSpec>;
};

export function useModuleSystem({ nodes, edges, setNodes, getNodeSchema }: UseModuleSystemProps) {
    const [modules, setModules] = useState<SavedModule[]>(() => listModules());
    const [moduleStack, setModuleStack] = useState<OpenModule[]>([]);

    // 需求五.2：从后端拉取已入库标准化模块，注入只读内存模块库并与本地模块合并展示。
    // 后端未启动 / 无模块 / 解析失败一律静默降级，仅保留本地模块（不写回 localStorage）。
    useEffect(() => {
        let cancelled = false;
        void (async () => {
            try {
                const items = await listBackendModules();
                if (cancelled) return;
                const converted = items
                    .map(compatToSavedModule)
                    .filter((m): m is SavedModule => m !== null);
                setTransientModules(converted);
                setModules(listModules());
            } catch (error) {
                console.warn("后端模块库不可用，仅展示本地模块", error);
            }
        })();
        return () => {
            cancelled = true;
        };
    }, []);
    const openModule = getActiveModule(moduleStack);
    const [showModuleDiagram, setShowModuleDiagram] = useState(false);

    const [showSaveModal, setShowSaveModal] = useState(false);
    const [showSaveCopyModal, setShowSaveCopyModal] = useState(false);
    const [showModuleSaveMenu, setShowModuleSaveMenu] = useState(false);
    const [moduleNameWarning, setModuleNameWarning] = useState(false);

    const [pendingModuleName, setPendingModuleName] = useState("");
    const [pendingVariables, setPendingVariables] = useState<Record<string, FieldSpec>>({});
    const [paramToVariableMap, setParamToVariableMap] = useState<Record<string, Record<string, string>>>({});
    const [pendingModuleCopyName, setPendingModuleCopyName] = useState("");
    const [moduleNameInput, setModuleNameInput] = useState("");
    useEffect(() => {
        if (showSaveModal) {
            setPendingModuleName("");
            setPendingVariables({});
            setParamToVariableMap({});
        }
    }, [showSaveModal]);
    const addVariable = useCallback(() => {
        setPendingVariables(prev => {
            const count = Object.keys(prev).length + 1;
            let newName = `var${count}`;
            while (prev[newName]) {
                newName = `var${Math.floor(Math.random() * 1000)}`;
            }
            return { ...prev, [newName]: { type: "number", required: true } };
        });
    }, []);
    const renameVariable = useCallback((oldName: string, newName: string) => {
        if (!newName || newName == oldName) return;
        setPendingVariables(prevVars => {
            if (prevVars[newName]) {
                alert(`Variable "${newName}" already exists`);
                return prevVars;
            }
            const { [oldName]: targetValue, ...rest } = prevVars;
            return {
                ...rest,
                [newName]: targetValue,
            };
        });
        setParamToVariableMap(prevMap => {
            const newMap: Record<string, Record<string, string>> = {};
            Object.keys(prevMap).forEach(nodeId => {
                newMap[nodeId] = {};
                Object.keys(prevMap[nodeId]).forEach(paramName => {
                    const currentVar = prevMap[nodeId][paramName];
                    if (currentVar === oldName) {
                        newMap[nodeId][paramName] = newName;
                    } else {
                        newMap[nodeId][paramName] = currentVar;
                    }
                });
            });
            return newMap;
        });
    }, []);
    const deleteVariable = useCallback((varName: string) => {
        setPendingVariables(prev => {
            const rest = { ...prev };
            delete rest[varName];

            return rest;
        });
        setParamToVariableMap(prevMap => {
            const newMap: Record<string, Record<string, string>> = {};
            Object.keys(prevMap).forEach(nodeId => {
                newMap[nodeId] = {};
                Object.keys(prevMap[nodeId]).forEach(paramName => {
                    if (prevMap[nodeId][paramName] !== varName) {
                        newMap[nodeId][paramName] = prevMap[nodeId][paramName];
                    }
                });
            });
            return newMap;
        });
    }, []);
    const promotableParams = useMemo<PromotableParam[]>(() => {
        if (!showSaveModal) return [];
        const selectedNodes = nodes.filter(n => n.selected);
        const params: PromotableParam[] = [];
        selectedNodes.forEach(node => {
            const nodeLabel = (node.data?.label as string) || node.type || "Node";
            if (node.type === "module_ref") {
                const data = node.data as { moduleId?: string };
                if (data?.moduleId) {
                    const modDef = modules.find(m => m.id === data.moduleId);

                    if (modDef) {
                        const schema = modDef.variableSchema || {};
                        Object.entries(schema).forEach(([varName, spec]) => {
                            params.push({
                                nodeId: node.id,
                                nodeLabel: `${modDef.name}`,
                                paramName: varName,
                                spec: spec,
                            });
                        });
                    }
                }
            } else if (getNodeSchema) {
                const schema = getNodeSchema(node.type || "");
                if (schema) {
                    Object.entries(schema).forEach(([paramName, spec]) => {
                        params.push({
                            nodeId: node.id,
                            nodeLabel: nodeLabel,
                            paramName,
                            spec,
                        });
                    });
                }
            }
            // Logic for standard should go here
        });
        return params;
    }, [showSaveModal, nodes, modules, getNodeSchema]);
    const updateParamMapping = useCallback((nodeId: string, paramName: string, variableName: string, spec?: FieldSpec) => {
        setParamToVariableMap(prev => {
            const existingNodeParams = prev[nodeId] || {};
            return {
                ...prev,
                [nodeId]: {
                    ...existingNodeParams,
                    [paramName]: variableName,
                },
            };
        });

        // If user selects a variable which doesnt exist yet, create it
        if (variableName && spec) {
            setPendingVariables(prev => {
                if (!prev[variableName]) {
                    return { ...prev, [variableName]: spec };
                }
                return prev;
            });
        }
    }, []);
    useEffect(() => {
        setModuleNameInput(openModule?.module?.name || "");
    }, [openModule?.module?.name]);

    useEffect(() => {
        setModuleNameWarning(false);
    }, [showSaveModal, pendingModuleName, moduleNameInput, openModule]);

    useEffect(() => {
        const handler = (ev: Event) => {
            const custom = ev as CustomEvent<{ moduleId?: string; nodeId?: string; data?: ModuleRefData }>;
            const moduleId = custom.detail?.moduleId;
            if (!moduleId) return;
            const mod = getModule(moduleId);
            if (!mod) {
                alert("Module not found");
                return;
            }

            const moduleRefData = custom.detail?.data || {};
            const { nodes: rawNodes, edges: rawEdges } = applyGraphIR(mod.graph);

            const nodesWithVars = rawNodes.map(n => {
                let nodeData = n.data || {};
                if (mod.variableMap) {
                    for (const varName in mod.variableMap) {
                        const targets = mod.variableMap[varName];
                        for (const target of targets) {
                            if (target.nodeId === n.id) {
                                if (moduleRefData[varName] !== undefined) {
                                    nodeData = { ...nodeData, [target.paramName]: moduleRefData[varName] };
                                }
                            }
                        }
                    }
                }
                return { ...n, data: nodeData, selected: false };
            });

            const applied = {
                nodes: nodesWithVars.map(n => ({
                    ...n,
                    selected: false,
                    data: { ...(n.data || {}), __highlight: undefined },
                })),
                edges: rawEdges.map(e => ({ ...e, selected: false })),
            };

            if (!applied.nodes.length) {
                alert("Saved module is empty. You can now add new things and save changes.");
            }
            setModuleStack(stack =>
                pushModule(stack, {
                    module: mod,
                    nodes: applied.nodes,
                    edges: applied.edges,
                    fromNodeId: custom.detail?.nodeId,
                }),
            );
            setShowModuleDiagram(false);
        };
        window.addEventListener("module-open", handler as EventListener);
        return () => window.removeEventListener("module-open", handler as EventListener);
    }, []);

    const dedupe = <T>(arr: T[]) => Array.from(new Set(arr));

    const computeModuleHandles = useCallback(
        (selectedIds: Set<string>): ModuleHandles => {
            const incoming = edges.filter(e => !selectedIds.has(e.source) && selectedIds.has(e.target));
            const outgoing = edges.filter(e => selectedIds.has(e.source) && !selectedIds.has(e.target));
            return {
                inputs: dedupe(incoming.map(e => e.targetHandle || "in")),
                outputs: dedupe(outgoing.map(e => e.sourceHandle || "out")),
            };
        },
        [edges],
    );

    const handleSaveModule = useCallback(() => {
        const selectedNodes = nodes.filter(n => n.selected);
        const selectedIds = new Set(selectedNodes.map(n => n.id));

        if (!selectedNodes.length) {
            setShowSaveModal(false);
            return;
        }

        const internalEdges = edges.filter(e => selectedIds.has(e.source) && selectedIds.has(e.target));

        const name = pendingModuleName.trim();
        if (!name) {
            alert("Enter a module name.");
            return;
        }

        const variableMap: Record<string, Array<{ nodeId: string; paramName: string }>> = {};
        Object.keys(paramToVariableMap).forEach(nodeId => {
            const nodeParams = paramToVariableMap[nodeId];
            Object.keys(nodeParams).forEach(paramName => {
                const varName = nodeParams[paramName];
                if (varName) {
                    if (!variableMap[varName]) {
                        variableMap[varName] = [];
                    }
                    variableMap[varName].push({ nodeId, paramName });
                }
            });
        });

        const sanitizedName = sanitizeIdent(resolveModuleName(name, ""));
        const existingNames = modules.map(m => sanitizeIdent(resolveModuleName(m.name, "")));
        if (existingNames.includes(sanitizedName)) {
            setModuleNameWarning(true);
            return;
        }

        const handles = computeModuleHandles(selectedIds);
        const moduleGraph = buildGraphIR(selectedNodes, internalEdges);
        saveModule({
            name: sanitizedName,
            version: "v1",
            graph: moduleGraph,
            handles,
            internalNodes: selectedNodes,
            internalEdges,
            description: `Saved from ${selectedNodes.length} node(s)`,
            variableSchema: pendingVariables,
            variableMap,
        });
        setModules(listModules());
        setShowSaveModal(false);
    }, [nodes, edges, computeModuleHandles, pendingModuleName, modules, paramToVariableMap, pendingVariables]);

    const saveExistingModuleChanges = useCallback(() => {
        setModuleNameWarning(false);
        if (!openModule) return;
        if (openModule.module.origin === "backend") {
            // 后端模块为只读内存覆盖层：直接保存会以同 id 写进 localStorage 形成影子覆盖
            alert("后端模块库模块为只读，不可直接保存修改；如需改动请另存为新模块。");
            return;
        }

        const name = moduleNameInput.trim();
        if (!name) {
            alert("Enter a module name.");
            return;
        }

        const sanitizedName = sanitizeIdent(resolveModuleName(name, ""));
        const existingNamesExcludingCurrent = modules
            .filter(m => m.id !== openModule.module.id)
            .map(m => sanitizeIdent(resolveModuleName(m.name, "")));

        if (existingNamesExcludingCurrent.includes(sanitizedName)) {
            setModuleNameWarning(true);
            return;
        }

        saveExistingModule(openModule, sanitizedName, openModule.nodes);
        setModules(listModules());
        alert("Module saved");
        setModuleStack(popModule);
    }, [openModule, moduleNameInput, modules]);

    const saveModuleAsNew = useCallback(() => {
        if (!openModule) return;
        const baseName = resolveModuleName(moduleNameInput, openModule.module.name);
        setPendingModuleCopyName(baseName);
        setShowSaveCopyModal(true);
    }, [openModule, moduleNameInput]);

    const handleReturnCopyModule = useCallback(() => {
        if (!openModule) return;
        const updatedGraph = buildGraphIR(openModule.nodes, openModule.edges);
        const name = pendingModuleCopyName.trim();
        if (!name) {
            alert("Enter a module name.");
            return;
        }
        saveModule({
            name: name,
            version: "v1",
            graph: updatedGraph,
            handles: openModule.module.handles,
            internalNodes: openModule.nodes,
            internalEdges: openModule.edges,
            description: openModule.module.description,
        });
        setModules(listModules());
        alert("Module saved as new");
        setShowSaveCopyModal(false);
        setModuleStack(popModule);
    }, [openModule, pendingModuleCopyName]);

    const handleDeleteModule = useCallback(
        (id: string) => {
            deleteModule(id);
            const updatedModules = listModules();
            setModules(updatedModules);
            setNodes(nds => {
                const remaining = nds.filter(n => {
                    const data = (n.data || {}) as { moduleId?: string };
                    return data.moduleId !== id;
                });
                return remaining;
            });
        },
        [setNodes],
    );
    const mergeModules = useCallback((importedModules: SavedModule[]) => {
        if (!importedModules || !Array.isArray(importedModules)) return;

        let hasChanges = false;

        importedModules.forEach(importedMod => {
            if (importedMod && importedMod.name && importedMod.graph) {
                // saveModule handles ID matching, overriding, and saving to localStorage automatically
                saveModule(toModuleSaveInput(importedMod));
                hasChanges = true;
            }
        });

        // Sync the React state with the newly updated localStorage list
        if (hasChanges) {
            setModules(listModules());
        }
    }, []);
    return {
        modules,
        setModules,
        mergeModules,
        moduleStack,
        setModuleStack,
        openModule,
        showModuleDiagram,
        setShowModuleDiagram,

        // Modal States
        showSaveModal,
        setShowSaveModal,
        showSaveCopyModal,
        setShowSaveCopyModal,
        showModuleSaveMenu,
        setShowModuleSaveMenu,
        moduleNameWarning,
        setModuleNameWarning,

        // Form States
        pendingModuleName,
        setPendingModuleName,
        pendingVariables,
        // setPendingVariables,
        paramToVariableMap,
        promotableParams,
        // setParamToVariableMap,

        pendingModuleCopyName,
        setPendingModuleCopyName,
        moduleNameInput,
        setModuleNameInput,
        addVariable,
        renameVariable,
        deleteVariable,
        updateParamMapping,
        // Actions
        handleSaveModule,
        saveExistingModuleChanges,
        saveModuleAsNew,
        handleReturnCopyModule,
        handleDeleteModule,
    };
}
