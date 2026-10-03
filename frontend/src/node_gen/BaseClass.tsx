import { Handle, Position, type NodeProps } from "@xyflow/react";
import { type ComponentType } from "react";

export type FieldType = 'number' | 'text' | 'boolean' | 'select' | 'array' | 'dict' // This describes how user can input a param's value
export interface FieldSpec {
    // This defines all the essential requirements of a parameter that a schema should follow
    type: FieldType;
    required: boolean;
    label?: string;
    options?: string[];
    defaultValue?: unknown;
    step?: number;
}
/** 节点参数包：各图层的参数值类型不一（number/text/boolean/...），统一以 unknown
 *  存放，读取处按需收窄。 */
export type LayerData = Record<string, unknown>;

/** 模块变量映射：变量名 -> 需要被替换为该变量的 (节点, 参数) 列表。 */
export type VariableMap = Record<string, Array<{ nodeId: string; paramName: string }>>;

/** 图层注册表：图层 key -> 图层定义。各图层数据形状不同，调用处传入 React Flow 的
 *  `node.data`（同样是 Record<string, unknown>），因此这里统一按 LayerData 处理。 */
export type LayerRegistry = Record<string, LayerDefinition<LayerData>>;

/** shapeCompute 的返回形态：单个形状（number[]）、多输出形状列表（number[][]，
 *  容器类图层历史上如此返回），或按 handle 名索引的形状映射。 */
export type ShapeComputeResult = number[] | number[][] | Record<string, number[]>;

// These are static class level static function implementation
// all attributes and methods MUST be static in nature to the class
export interface LayerDefinition<D extends LayerData> {

    // Class configurations
    label: string;
    paramSchema: Record<string, FieldSpec>;
    // Optional diagram metadata to avoid ad-hoc maps in renderers.
    diagramLabel?: string;
    diagramFamily?: "input" | "output" | "merge" | "activation" | "block" | "other";
    handles?: HandleSpec | HandleFactory<D>;
    // For container type nodes where they initialize the child internally. 
    encapsulatesChildInit?: boolean | undefined;
    // Pure functions
    // shapeVerifier: checks compatibility of incoming shapes/params, must NOT modify data
    shapeVerifier(data: D, inputShapes: number[][], registry?: LayerRegistry): { ok: true } | { ok: false; error: string };
    // shapeCompute: computes output shape, assumes verifier passed
    shapeCompute(data: D, inputShapes: number[][], registry?: LayerRegistry): ShapeComputeResult;
    // estimateCost: optional params/FLOPs estimate for analysis panels. Currently WIP. 
    // 与方法型成员保持一致：参数按双变（bivariant）检查，允许各图层使用自己的数据形状。
    estimateCost?(data: D, inputShapes: number[][], outputShape: number[], context?: { registry: LayerRegistry }): { params: number; flops: number };
    getInitCode(data: D, name: string, variableMap?: VariableMap): string;
    getForwardCode(data: D, name: string, inputs: Array<string>, outputs: Array<string>): string;
    Component: ComponentType<NodeProps>;
}

export type HandleSpec = {
    targets: string[];
    sources: string[];
};
export type HandleFactory<D> = (data: D) => HandleSpec;

// Utility: get a parameter value with default fallback from schema
type HasParamSchema = { paramSchema: Record<string, FieldSpec> };

export function getParamValue(
    schemaOrLayer: Record<string, FieldSpec> | HasParamSchema,
    data: object | undefined,
    key: string
): unknown {
    const schema = (schemaOrLayer as HasParamSchema).paramSchema ?? (schemaOrLayer as Record<string, FieldSpec>);
    const spec = schema[key];
    const val = (data as Record<string, unknown> | undefined)?.[key];
    if (spec?.type === "number") {
        return typeof val === "number" && !Number.isNaN(val) ? val : spec?.defaultValue;
    }
    return val !== undefined ? val : spec?.defaultValue;
}

/** 把 unknown 参数值按 JS 关系运算的隐式转换取成数值：非数值得到 NaN
 *  （所有比较均为 false），与迁移前 `as number` 后直接比较的结果一致。 */
export function toNumberParam(value: unknown): number {
    return typeof value === "number" ? value : Number(value);
}

export function renderHandles(side: "left" | "right", ids: string[], isConnectable: boolean) {
    return ids.map((idLabel, i, arr) => {
        const topPct = `${((i + 1) / (arr.length + 1)) * 100}%`;
        const isLeft = side === "left";
        return (
            <div
                key={`${side}-${idLabel}`}
                style={{
                    position: "absolute",
                    [isLeft ? "left" : "right"]: -1,
                    top: topPct,
                    transform: "translateY(-50%)",
                    display: "flex",
                    alignItems: "center"
                }}
            >
                <Handle
                    id={idLabel}
                    type={isLeft ? "target" : "source"}
                    position={isLeft ? Position.Left : Position.Right}
                    isConnectable={isConnectable}
                    style={{
                        background: "#777",
                        border: "1px solid #222"
                    }}
                />
            </div>
        );
    });
}


// This is a base implementation of the init code that can dynamically update
//  depending on if the value has been changed to a non default param value
export function buildInitString(
    className: string,
    name: string,
    schema: Record<string, FieldSpec>,
    data: object
) {
    // 图层数据既可能是具体的数据类型别名，也可能是图层类本身（静态类实例类型没有
    // 隐式索引签名），因此这里只要求 object，按 key 读取时再局部收窄。
    const bag = data as Record<string, unknown>;
    const args: string[] = [];
    Object.keys(schema).forEach(key => {
        const spec = schema[key];
        const value = bag[key];
        const toPython = (val: unknown): string => {
            if (spec.type === 'boolean') return val ? 'True' : 'False';
            if (spec.type === 'select' || spec.type === 'text') return `${val}`
            return `${val}`
        };

        if (spec.required) {
            const valToUse = value !== undefined ? value : spec.defaultValue;
            // args.push(toPython(valToUse));
            args.push(`${key}=${toPython(valToUse)}`)
        } else {
            if (value !== undefined && value !== spec.defaultValue) {
                args.push(`${key}=${toPython(value)}`)
            }
        }
    })
    return `self.${name} = ${className}(${args.join(', ')})`
}