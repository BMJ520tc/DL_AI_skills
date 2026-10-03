import { useState, type ChangeEvent } from "react";
import { type FieldSpec, type FieldType, type LayerData } from "./BaseClass";

/** 参数取值 -> 表单可渲染的原始值。React 自身也会把值强制转成字符串，
 *  这里显式转换以匹配 previous `value ?? ""` 的渲染结果。 */
function toFieldValue(value: unknown): string {
    return value == null ? "" : String(value);
}

/** 数组/对象参数的 JSON 输入（阶段4 4b：list/dict 也有控件）。
 *  本地草稿保证输入过程不被打断；只有 JSON 合法时才提交（非法时红框，不写回节点数据）。 */
function JsonField({
    value,
    kind,
    onCommit,
}: {
    value: unknown;
    kind: "array" | "dict";
    onCommit: (raw: string) => void;
}) {
    const [draft, setDraft] = useState<string | null>(null);
    const text = draft ?? JSON.stringify(value ?? (kind === "array" ? [] : {}));
    let invalid = false;
    try {
        const parsed: unknown = JSON.parse(text);
        invalid = kind === "array"
            ? !Array.isArray(parsed)
            : parsed === null || typeof parsed !== "object" || Array.isArray(parsed);
    } catch {
        invalid = true;
    }
    return (
        <input
            className="nodrag"
            type="text"
            value={text}
            title='JSON 字面量，例如 [64, 128] 或 {"k": 1}'
            onChange={e => {
                setDraft(e.target.value);
                onCommit(e.target.value);
            }}
            style={{
                width: "120px",
                backgroundColor: "#111",
                border: `1px solid ${invalid ? "#ef4444" : "#444"}`,
                color: "white",
                borderRadius: "4px",
                padding: "2px 4px",
                fontFamily: "monospace",
            }}
        />
    );
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
        case "array":
        case "dict":
            return (
                <JsonField
                    value={value}
                    kind={spec.type}
                    onCommit={raw =>
                        onChange(paramKey, spec.type)(
                            { target: { value: raw } } as unknown as ChangeEvent<HTMLInputElement | HTMLSelectElement>
                        )
                    }
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
