 
import { useReactFlow, type NodeProps } from "@xyflow/react";
import { useMemo, useState, type ComponentType } from "react";
import {
    renderHandles,
    type FieldSpec,
    type FieldType,
    type HandleFactory,
    type HandleSpec,
    type LayerData
} from "./BaseClass";
import { ParamsList } from "./ParamsList";

// Options interface for the factory
type LayerComponentOptions<D> = {
    targetHandles?: number;
    handles?: HandleSpec | HandleFactory<D>;
    // Allow dynamic schema resolution (for 自定义模块)
    resolveSchema?: (data: D) => Record<string, FieldSpec>;
    // Allow custom buttons in the header
    renderHeaderActions?: (data: D, id: string) => React.ReactNode;
};

export function createLayerComponent<D extends LayerData = LayerData>(
    label: string,
    staticSchema: Record<string, FieldSpec>,
    options?: LayerComponentOptions<D>
): ComponentType<NodeProps> {
    const resolveSchema = options?.resolveSchema;
    const handles = options?.handles;
    const renderHeaderActions = options?.renderHeaderActions;
    const targetHandles = options?.targetHandles;
    return ({ id, data, isConnectable }: NodeProps) => {
        const { setNodes, setEdges } = useReactFlow();
        const [isExpanded, setIsExpanded] = useState(false);
        // React Flow 的 data 为动态 JSON；这里正是该图层声明的数据契约边界。
        // 记忆化以保持身份稳定，避免下游 useMemo 每帧重算。
        const safeData = useMemo(() => (data ?? {}) as D, [data]);
        const isHighlighted = !!safeData.__highlight;

        // 1. Resolve Schema 
        const paramSchema = useMemo(() => {
            if (resolveSchema) {
                return resolveSchema(safeData);
            }
            return staticSchema;
        // `resolveSchema` comes from the module-scope options object handed to the
        // factory, so it is not reactive and must not be a dependency.
        }, [safeData]);

        const { requiredParams, optionalParams } = useMemo(() => {
            const keys = Object.keys(paramSchema);
            const req = keys.filter(k => paramSchema[k].required);
            const opt = keys.filter(k => !paramSchema[k].required);
            return { requiredParams: req, optionalParams: opt };
        }, [paramSchema]);

        const onChange = (key: string, type: FieldType) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
            const raw = e.target.value;
            let newValue: string | number | boolean | undefined = raw;
            if (type === "number") {
                newValue = raw === "" ? undefined : parseFloat(raw);
            } else if (type === "boolean") {
                newValue = (e.target as HTMLInputElement).checked;
            }
            setNodes(nodes =>
                nodes.map(n => {
                    if (n.id !== id) return n;
                    // 空值表示“回到 schema 默认值”：从 data 中移除该参数键。
                    const newData: Record<string, unknown> = { ...n.data };
                    if (newValue === undefined || newValue === "") {
                        delete newData[key];
                    } else {
                        newData[key] = newValue;
                    }
                    return { ...n, data: newData };
                })
            );
        };

        const handleDelete = (e: React.MouseEvent) => {
            e.stopPropagation();
            setNodes(nodes => nodes.filter(n => n.id !== id));
            setEdges(eds => eds.filter(edge => edge.source !== id && edge.target !== id));
        };

        const paramsToShow = new Set(requiredParams);
        optionalParams.forEach(key => {
            const currentVal = safeData[key];
            const defaultVal = paramSchema[key].defaultValue;
            const isDefault = currentVal === defaultVal;
            if (isExpanded || !isDefault) paramsToShow.add(key);
        });
        const renderList = [...requiredParams, ...optionalParams.filter(k => paramsToShow.has(k))];
        const hiddenOptionCount = optionalParams.length - (renderList.length - requiredParams.length);

        const shapePreview = (() => {
            const liveShape = safeData.__shape;
            if (Array.isArray(liveShape) && liveShape.length > 0) return JSON.stringify(liveShape);
            return "";
        })();

        const resolvedHandles: HandleSpec = (() => {
            if (typeof handles === "function") return handles(safeData);
            if (handles && handles.targets && handles.sources) return handles;
            const targetCount = targetHandles ?? 1;
            return {
                targets: Array.from({ length: targetCount }).map((_, i) => `in-${i}`),
                sources: ["out-0"]
            };
        })();

        // `name ?? label`：name 可能来自动态 data，非字符串时回落到图层标签。
        const rawName = safeData.name;
        const displayName = rawName ? String(rawName) : label;

        return (
            <div
                className="layer-node"
                style={{
                    backgroundColor: isHighlighted ? "#27210d" : "#222",
                    border: isHighlighted ? "1px solid #f1c40f" : isExpanded ? "1px solid #64ffda" : "1px solid #555",
                    borderRadius: "8px",
                    minWidth: "170px",
                    transition: "all 0.2s",
                    position: "relative",
                    boxShadow: isHighlighted ? "0 0 0 2px #f1c40f, 0 0 20px #f1c40f66" : undefined,
                    transform: isHighlighted ? "translateY(-2px) scale(1.01)" : undefined,
                    color: "#e6edf3"
                }}
            >
                {renderHandles("left", resolvedHandles.targets, isConnectable)}
                <div
                    onClick={() => setIsExpanded(!isExpanded)}
                    style={{
                        fontWeight: "bold",
                        color: "#64ffda",
                        borderBottom: "1px solid #444",
                        padding: "8px",
                        cursor: "pointer",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center"
                    }}
                >
                    <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
                        <span>{displayName}</span>
                        {/* {(safeData as any).version && <span style={{ fontSize: '9px', color: '#888', fontWeight: 'normal' }}>{(safeData as any).version}</span>} */}
                    </div>
                    <div style={{ paddingLeft: 10 }}></div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        {renderHeaderActions && renderHeaderActions(safeData, id)}

                        <button
                            className="nodrag"
                            onClick={handleDelete}
                            style={{
                                cursor: "pointer",
                                border: "none",
                                background: "transparent",
                                color: "#888",
                                fontWeight: "bold",
                                fontSize: "18px",
                                lineHeight: "18px",
                                width: "24px",
                                height: "24px",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center"
                            }}
                            title="删除节点"
                        >
                            ×
                        </button>
                    </div>
                </div>

                <ParamsList
                    renderKeys={renderList}
                    optionalParams={optionalParams}
                    paramSchema={paramSchema}
                    data={safeData}
                    onChange={onChange}
                    onExpand={() => setIsExpanded(true)}
                    hiddenCount={!isExpanded ? hiddenOptionCount : 0}
                />

                <div style={{ padding: "0 10px 10px", fontSize: "10px", color: "#888" }}>
                    Shape: {shapePreview}
                </div>
                {renderHandles("right", resolvedHandles.sources, isConnectable)}
            </div>
        );
    };
}