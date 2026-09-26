import { type FieldSpec } from "../../node_gen/BaseClass";
import { createLayerComponent } from "../../node_gen/CreateNodeComponent.tsx";

type MetricData = Record<string, never>;

export class AccuracyNode {
    static label = "准确率";
    static paramSchema: Record<string, FieldSpec> = {};
    static handles = { targets: ["logits", "target"], sources: ["out-0"] };

    static shapeVerifier(_data: MetricData, inputShapes: number[][]) {
        if (inputShapes.length !== 2) return { ok: false as const, error: "Accuracy 期望 logits 和目标" };
        const logits = inputShapes[0];
        const target = inputShapes[1];
        if (logits.length !== 2 && logits.length !== 3) return { ok: false as const, error: "Logits 必须是 [batch, classes] 或 [batch, seq, classes]" };
        if (logits[0] !== target[0]) return { ok: false as const, error: "Batch 大小不匹配" };
        return { ok: true as const };
    }

    static shapeCompute(_data: MetricData, _inputShapes: number[][]) {
        return []; // scalar metric
    }

    static estimateCost() {
        return { params: 0, flops: 0 };
    }

    static getInitCode() {
        return "# accuracy is computed in forward";
    }

    static getForwardCode(_data: MetricData, _name: string, inputs: Array<string>, outputs: Array<string>) {
        const logits = inputs[0] || "logits";
        const target = inputs[1] || "target";
        const out = outputs[0] || "acc";
        return `${out} = (torch.argmax(${logits}, dim=-1) == ${target}).float().mean()`;
    }

    static Component = createLayerComponent<MetricData>(AccuracyNode.label, AccuracyNode.paramSchema, {
        handles: { targets: ["logits", "target"], sources: ["out-0"] }
    });
}
