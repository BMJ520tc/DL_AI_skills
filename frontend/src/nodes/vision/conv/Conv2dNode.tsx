import { buildInitString, getParamValue, type FieldSpec } from "../../../node_gen/BaseClass";
import { estimateConvCost, toNumber } from "../../../utils/computeUtils";
import { createLayerComponent } from "../../../node_gen/CreateNodeComponent.tsx";

type Conv2dData = {
    in_channels: number;
    out_channels: number;
    kernel_size: number;
    stride?: number;
    padding?: number;
    bias?: boolean;
};

export class Conv2dNode {
    static label = "Conv2d";
    static paramSchema: Record<string, FieldSpec> = {
        in_channels: { required: true, type: "number", label: "输入通道", defaultValue: 1, step: 1 },
        out_channels: { required: true, type: "number", label: "输出通道", defaultValue: 1, step: 1 },
        kernel_size: { required: true, type: "number", label: "卷积核大小", defaultValue: 3, step: 1 },
        stride: { required: false, type: "number", label: "步长", defaultValue: 1, step: 1 },
        padding: { required: false, type: "number", label: "填充", defaultValue: 0, step: 1 },
        bias: { required: false, type: "boolean", label: "偏置", defaultValue: true }
    };

    static shapeVerifier(data: Conv2dData, inputShapes: number[][]) {
        if (inputShapes.length !== 1) return { ok: false as const, error: "Conv2d 期望恰好一个输入" };
        const shape = inputShapes[0];
        if (shape.length !== 4) return { ok: false as const, error: "Conv2d 输入必须是 [batch, channels, height, width]" };

        const [, channels, height, width] = shape;
        const inCh = getParamValue(this, data, "in_channels") as number;
        const outCh = getParamValue(this, data, "out_channels") as number;
        const kernel = getParamValue(this, data, "kernel_size") as number;
        const stride = getParamValue(this, data, "stride") as number;
        const padding = getParamValue(this, data, "padding") as number;

        if (inCh <= 0 || outCh <= 0) return { ok: false as const, error: "输入/输出通道数必须 > 0" };
        if (channels !== inCh) return { ok: false as const, error: `期望 ${inCh} 通道，实际 ${channels}` };
        if (kernel <= 0) return { ok: false as const, error: "卷积核大小必须 > 0" };
        if (stride <= 0) return { ok: false as const, error: "步长必须 > 0" };
        if (kernel > height + 2 * padding || kernel > width + 2 * padding) {
            return { ok: false as const, error: "卷积核大小超过填充后的空间维度" };
        }
        return { ok: true as const };
    }

    static shapeCompute(data: Conv2dData, inputShapes: number[][]) {
        const [batch, , height, width] = inputShapes[0] || [1, 1, 1, 1];
        const outCh = getParamValue(this, data, "out_channels") as number;
        const kernel = getParamValue(this, data, "kernel_size") as number;
        const stride = getParamValue(this, data, "stride") as number;
        const padding = getParamValue(this, data, "padding") as number;
        const dilation = 1;
        const computeDim = (dim: number) => Math.floor((dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1);
        return [batch, outCh, computeDim(height), computeDim(width)];
    }

    static estimateCost(data: Conv2dData, _inputShapes: number[][], outputShape: number[]) {
        const inCh = toNumber(getParamValue(this, data, "in_channels"), 0);
        const outCh = toNumber(getParamValue(this, data, "out_channels"), 0);
        const kernel = toNumber(getParamValue(this, data, "kernel_size"), 0);
        const bias = data.bias !== false;
        const kernelArea = kernel * kernel;
        return estimateConvCost(outputShape, inCh, outCh, kernelArea, bias);
    }

    static getInitCode(data: Conv2dData, name: string) {
        return buildInitString("nn.Conv2d", name, Conv2dNode.paramSchema, data);
    }

    static getForwardCode(_data: Conv2dData, name: string, inputs: Array<string>, outputs: Array<string>) {
        const inputVar = inputs[0] || "x";
        const outputVar = outputs[0] || "x";
        return `${outputVar} = self.${name}(${inputVar})`;
    }

    static Component = createLayerComponent<Conv2dData>(Conv2dNode.label, Conv2dNode.paramSchema);
}
