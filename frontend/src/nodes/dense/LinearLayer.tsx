import { getParamValue, type FieldSpec } from "../../node_gen/BaseClass";
import { estimateLinearCost, toNumber } from "../../utils/computeUtils";
import { createLayerComponent } from "../../node_gen/CreateNodeComponent.tsx";


type LinearData = {
    in_features: number;
    out_features: number;
    bias?: boolean
};

export class LinearLayerNode {
    static label = "线性层";
    static paramSchema: Record<string, FieldSpec> = {
        in_features: {
            required: true,
            type: 'number',
            label: '输入特征',
            defaultValue: 1,
            step: 1
        },
        out_features: {
            required: true,
            type: 'number',
            label: '输出特征',
            defaultValue: 1,
            step: 1
        },
        bias: {
            required: false,
            type: 'boolean',
            label: '偏置',
            defaultValue: true
        }
    }
    static shapeVerifier(data: LinearData, inputShapes: number[][]) {
        if (inputShapes.length !== 1) return { ok: false as const, error: "Linear 期望恰好一个输入" };
        const shape = inputShapes[0];
        if (!shape || shape.length === 0) return { ok: false as const, error: "线性层输入必须至少有一个维度。" };

        const inFeatures = getParamValue(this, data, "in_features") as number;
        const outFeatures = getParamValue(this, data, "out_features") as number;
        if (inFeatures <= 0 || outFeatures <= 0) return { ok: false as const, error: "in_features 和 out_features 必须 > 0" };

        const lastDim = shape[shape.length - 1];
        if (lastDim !== inFeatures) {
            return { ok: false as const, error: `期望 ${inFeatures} 输入特征，实际 ${lastDim}` };
        }
        return { ok: true as const };
    }
    static shapeCompute(data: LinearData, inputShapes: number[][]) {
        const shape = inputShapes[0] || [];
        const outFeatures = getParamValue(this, data, "out_features") as number;
        if (shape.length === 0) return [outFeatures];
        return [...shape.slice(0, -1), outFeatures];
    }
    static estimateCost(data: LinearData, _inputShapes: number[][], outputShape: number[]) {
        const inFeatures = toNumber(getParamValue(this, data, "in_features"), 0);
        const outFeatures = toNumber(getParamValue(this, data, "out_features"), 0);
        const bias = data.bias !== false;
        return estimateLinearCost(outputShape, inFeatures, outFeatures, bias);
    }
    static getInitCode(data: LinearData, name: string) {
        const i = data.in_features || this.paramSchema.in_features.defaultValue;
        const o = data.out_features || this.paramSchema.out_features.defaultValue;
        const biasStr = data.bias === false ? ', bias=False' : '';
        return `self.${name} = nn.Linear(in_features=${i}, out_features=${o}${biasStr})`;
    }
    static getForwardCode(_data: LinearData, name: string, inputs: Array<string>, outputs: Array<string>) {
        const inputVar = inputs.length > 0 ? inputs[0] : "x";
        const outputVar = outputs.length > 0 ? outputs[0] : "x";
        return `${outputVar} = self.${name}(${inputVar})`
    }
    static Component = createLayerComponent<LinearData>(LinearLayerNode.label, LinearLayerNode.paramSchema);
}
