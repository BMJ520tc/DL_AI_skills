import { getParamValue, type FieldSpec } from "../../node_gen/BaseClass";
import { estimateElementwiseCost } from "../../utils/computeUtils";
import { createLayerComponent } from "../../node_gen/CreateNodeComponent.tsx";

type PowData = { exponent?: number };

export class PowNode {
    static label = "Pow";
    static paramSchema: Record<string, FieldSpec> = {
        exponent: { required: true, type: "number", label: "指数", defaultValue: 2, step: 1 }
    };

    static shapeVerifier(_data: PowData, inputShapes: number[][]) {
        if (inputShapes.length !== 1) return { ok: false as const, error: "Pow 期望一个输入" };
        const shape = inputShapes[0];
        if (!Array.isArray(shape) || !shape.length) return { ok: false as const, error: "输入形状必须已定义" };
        return { ok: true as const };
    }

    static shapeCompute(_data: PowData, inputShapes: number[][]) {
        return [...(inputShapes[0] || [])];
    }

    static estimateCost(_data: PowData, _inputShapes: number[][], outputShape: number[]) {
        return estimateElementwiseCost(outputShape);
    }

    static getInitCode() {
        return "# pow uses functional torch.pow";
    }

    static getForwardCode(data: PowData, _name: string, inputs: Array<string>, outputs: Array<string>) {
        const out = outputs[0] || "x";
        const inputVar = inputs[0] || "x";
        const exp = getParamValue(PowNode.paramSchema, data, "exponent");
        return `${out} = torch.pow(${inputVar}, ${exp})`;
    }

    static Component = createLayerComponent<PowData>(PowNode.label, PowNode.paramSchema);
}
