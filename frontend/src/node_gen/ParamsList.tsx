import type { ChangeEvent } from "react";
import { type FieldSpec, type FieldType, type LayerData } from "./BaseClass";

/** 参数取值 -> 表单可渲染的原始值。React 自身也会把值强制转成字符串，
 *  这里显式转换以匹配 previous `value ?? ""` 的渲染结果。 */
function toFieldValue(value: unknown): string {
    return value == null ? "" : String(value);
}

export function InputControl({
    paramKey,
    spec,
    value,
    onChange
}: {
    paramKey: string;
    spec: FieldSpec;
    value: unknown;
    onChange: (key: string, type: FieldType) => (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void;
}) {
    const isOptional = !spec.required;
    const style = (isOptional && value !== undefined)
        ? {
            width: "60px",
            backgroundColor: "#111",
            border: "1px solid #64ffda",
            color: "white",
            borderRadius: "4px",
            padding: "2px 4px",
            fondSize: "11px"
        }
        : {
            width: "60px",
            backgroundColor: "#111",
            border: "1px solid #444",
            color: "white",
            borderRadius: "4px",
            padding: "2px 4px",
            fondSize: "11px"
        };
    const fieldValue = toFieldValue(value);
    switch (spec.type) {
        case "boolean":
            return (
                <input
                    className="nodrag"
                    type="checkbox"
                    checked={!!value}
                    onChange={onChange(paramKey, "boolean")}
                    style={{ cursor: "pointer" }}
                />
            );
        case "select":
            return (
                <select
                    className="nodrag"
                    value={fieldValue}
                    onChange={onChange(paramKey, "select")}
                    style={{ ...style, width: "80px" }}
                >
                    <option value="" disabled>...</option>
                    {spec.options?.map(opt => (
                        <option key={opt} value={opt}>
                            {opt}
                        </option>
                    ))}
                </select>
            );
        case "text":
            return (
                <input
                    className="nodrag"
                    type="text"
                    value={fieldValue}
                    onChange={onChange(paramKey, "text")}
                    style={{ ...style, width: "80px" }}
                />
            );
        case "number":
            return (
                <input
                    className="nodrag"
                    type="number"
                    step={spec.step || 1}
                    value={fieldValue}
                    onChange={onChange(paramKey, "number")}
                    placeholder={isOptional ? "" : "0"}
                    style={style}
                />
            );
    }
}

export function ParamsList({
    renderKeys,
    optionalParams,
    paramSchema,
    data,
    onChange,
    onExpand,
    hiddenCount
}: {
    renderKeys: string[];
    optionalParams: string[];
    paramSchema: Record<string, FieldSpec>;
    data: LayerData;
    onChange: (key: string, type: FieldType) => (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void;
    onExpand: () => void;
    hiddenCount: number;
}) {
    return (
        <div style={{ padding: "10px" }}>
            {renderKeys.map(key => {
                const spec = paramSchema[key];
                if (!spec) return null;
                return (
                    <div
                        key={key}
                        style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px", alignItems: "center" }}
                    >
                        <label style={{ fontSize: "11px", color: optionalParams.includes(key) ? "#aaa" : "#fff" }}>
                            {spec.label || key}
                        </label>
                        <InputControl paramKey={key} spec={spec} value={data[key]} onChange={onChange} />
                    </div>
                );
            })}
            {hiddenCount > 0 && (
                <div onClick={onExpand} style={{ fontSize: "9px", color: "#666", textAlign: "center", cursor: "pointer" }}>
                    + {hiddenCount} options
                </div>
            )}
        </div>
    );
}
