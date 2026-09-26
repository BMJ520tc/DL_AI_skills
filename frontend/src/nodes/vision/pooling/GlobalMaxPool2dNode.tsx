import { type FieldSpec } from "../../../node_gen/BaseClass";
import { estimatePoolCost, toNumber } from "../../../utils/computeUtils";
import { createLayerComponent } from "../../../node_gen/CreateNodeComponent.tsx";

type GlobalPoolData = Record<string, never>;

export class GlobalMaxPool2dNode {
    static label = "GlobalMaxPool2d";
    static paramSchema: Record<string, FieldSpec> = {};

    static shapeVerifier(_data: GlobalPoolData, inputShapes: number[][]) {
        if (inputShapes.length !== 1) return { ok: false as const, error: "GlobalMaxPool2d 期望一个输入" };
        const shape = inputShapes[0];
        if (shape.length !== 4) return { ok: false as const, error: "输入必须是 [batch, channels, height, width]" };
        return { ok: true as const };
    }

    static shapeCompute(_data: GlobalPoolData, inputShapes: number[][]) {
        const [n, c] = inputShapes[0];
        return [n, c, 1, 1];
    }

    static estimateCost(_data: GlobalPoolData, inputShapes: number[][], outputShape: number[]) {
        const inputShape = inputShapes[0] || [];
        const inH = toNumber(inputShape[2], 0);
        const inW = toNumber(inputShape[3], 0);
        return estimatePoolCost(outputShape, inH * inW);
    }

    static getInitCode(_data: GlobalPoolData, name: string) {
        return `self.${name} = nn.AdaptiveMaxPool2d(1)`;
    }

    static getForwardCode(_data: GlobalPoolData, name: string, inputs: Array<string>, outputs: Array<string>) {
        const inputVar = inputs[0] || "x";
        const outputVar = outputs[0] || "x";
        return `${outputVar} = self.${name}(${inputVar})`;
    }

    static Component = createLayerComponent<GlobalPoolData>(GlobalMaxPool2dNode.label, GlobalMaxPool2dNode.paramSchema);
}
