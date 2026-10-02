import { buildInitString, getParamValue, type FieldSpec } from "../../../node_gen/BaseClass";
import { estimateConvCost, toNumber } from "../../../utils/computeUtils";
import { createLayerComponent } from "../../../node_gen/CreateNodeComponent.tsx";

type DepthwiseData = {
    in_channels: number;
    depth_multiplier?: number;
    kernel_size: number;
    stride?: number;
    padding?: number;
    dilation?: number;
    bias?: boolean;
};

export class DepthwiseConv2dNode {
    static label = "DepthwiseConv2d";
    static paramSchema: Record<string, FieldSpec> = {
        in_channels: { required: true, type: "number", label: "输入通道", defaultValue: 1, step: 1 },
        depth_multiplier: { required: false, type: "number", label: "深度倍数", defaultValue: 1, step: 1 },
        kernel_size: { required: true, type: "number", label: "卷积核大小", defaultValue: 3, step: 1 },
        stride: { required: false, type: "number", label: "步长", defaultValue: 1, step: 1 },
        padding: { required: false, type: "number", label: "填充", defaultValue: 0, step: 1 },
        dilation: { required: false, type: "number", label: "膨胀", defaultValue: 1, step: 1 },
        bias: { required: false, type: "boolean", label: "偏置", defaultValue: true }
    };

    static shapeVerifier(data: DepthwiseData, inputShapes: number[][]) {
        if (inputShapes.length !== 1) return { ok: false as const, error: "DepthwiseConv2d 期望一个输入" };
        const shape = inputShapes[0];
        if (shape.length !== 4) return { ok: false as const, error: "输入必须是 [N,C,H,W]" };
        const [, channels, height, width] = shape;
        const inCh = getParamValue(this, data, "in_channels") as number;
        const kernel = getParamValue(this, data, "kernel_size") as number;
        const stride = getParamValue(this, data, "stride") as number;
        const padding = getParamValue(this, data, "padding") as number;
        const dilation = getParamValue(this, data, "dilation") as number;
        if (channels !== inCh) return { ok: false as const, error: `期望 ${inCh} 通道，实际 ${channels}` };
        if (kernel <= 0 || stride <= 0 || dilation <= 0) return { ok: false as const, error: "卷积核/步长/膨胀必须 > 0" };
        if (kernel > height + 2 * padding || kernel > width + 2 * padding) return { ok: false as const, error: "卷积核大小超过填充后的维度" };
        return { ok: true as const };
    }

    static shapeCompute(data: DepthwiseData, inputShapes: number[][]) {
        const [batch, , height, width] = inputShapes[0];
        const inCh = getParamValue(this, data, "in_channels") as number;
        const mult = getParamValue(this, data, "depth_multiplier") as number;
        const kernel = getParamValue(this, data, "kernel_size") as number;
        const stride = getParamValue(this, data, "stride") as number;
        const padding = getParamValue(this, data, "padding") as number;
        const dilation = getParamValue(this, data, "dilation") as number;
        const computeDim = (dim: number) => Math.floor((dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1);
        return [batch, inCh * mult, computeDim(height), computeDim(width)];
    }

    static estimateCost(data: DepthwiseData, _inputShapes: number[][], outputShape: number[]) {
        const inCh = toNumber(getParamValue(this, data, "in_channels"), 0);
        const mult = toNumber(getParamValue(this, data, "depth_multiplier"), 1);
        const outCh = inCh * mult;
        const kernel = toNumber(getParamValue(this, data, "kernel_size"), 0);
        const bias = data.bias !== false;
        const kernelArea = kernel * kernel;
        return estimateConvCost(outputShape, inCh, outCh, kernelArea, bias, inCh || 1);
    }

    static getInitCode(data: DepthwiseData, name: string) {
        const merged = { ...data, out_channels: (getParamValue(this, data, "in_channels") as number) * (getParamValue(this, data, "depth_multiplier") as number), groups: getParamValue(this, data, "in_channels") };
        return buildInitString("nn.Conv2d", name, {
            ...DepthwiseConv2dNode.paramSchema,
            out_channels: { required: true, type: "number", label: "输出通道", defaultValue: 1 },
            groups: { required: true, type: "number", label: "分组数", defaultValue: 1 }
        }, merged);
    }

    static getForwardCode(_data: DepthwiseData, name: string, inputs: Array<string>, outputs: Array<string>) {
        const inputVar = inputs[0] || "x";
        const outputVar = outputs[0] || "x";
        return `${outputVar} = self.${name}(${inputVar})`;
    }

    static Component = createLayerComponent<DepthwiseData>(DepthwiseConv2dNode.label, DepthwiseConv2dNode.paramSchema);
}
