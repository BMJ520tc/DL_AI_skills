import { type FieldSpec } from "../../node_gen/BaseClass";
import { createLayerComponent } from "../../node_gen/CreateNodeComponent.tsx";


type TransposeData = {
    perm: string;
};

function parsePerm(raw: string): number[] | null {
    const parts = raw.split(",").map(p => p.trim()).filter(Boolean);
    if (!parts.length) return null;
    const perm: number[] = [];
    for (const p of parts) {
        const n = Number(p);
        if (!Number.isInteger(n)) return null;
        perm.push(n);
    }
    return perm;
}

export class TransposeNode {
    static label = "Transpose";
    static paramSchema: Record<string, FieldSpec> = {
        perm: { required: true, type: "text", label: "维度置换（逗号分隔）", defaultValue: "0,2,3,1" },
    };

    static shapeVerifier(data: TransposeData, inputShapes: number[][]) {
        if (inputShapes.length !== 1) return { ok: false as const, error: "Transpose 期望一个输入" };
        const shape = inputShapes[0];
        const perm = parsePerm(data.perm ?? "");
        if (!perm) return { ok: false as const, error: "置换必须是逗号分隔的整数" };
        if (perm.length !== shape.length) return { ok: false as const, error: "置换长度必须与秩一致" };
        const set = new Set(perm);
        if (set.size !== perm.length || Math.min(...perm) < 0 || Math.max(...perm) >= shape.length) {
            return { ok: false as const, error: "置换必须是维度的有效重排" };
        }
        return { ok: true as const };
    }

    static shapeCompute(data: TransposeData, inputShapes: number[][]) {
        const shape = inputShapes[0];
        const perm = parsePerm(data.perm ?? "") || [];
        return perm.map(i => shape[i]);
    }

    static estimateCost() {
        return { params: 0, flops: 0 };
    }

    static getInitCode() {
        return "# transpose handled in forward";
    }

    static getForwardCode(data: TransposeData, _name: string, inputs: Array<string>, outputs: Array<string>) {
        const out = outputs[0] || "x";
        const inputVar = inputs[0] || "x";
        const perm = data.perm || "";
        return `${out} = ${inputVar}.permute(${perm})`;
    }

    static computeShape(_data: TransposeData) {
        return [];
    }

    static Component = createLayerComponent<TransposeData>(TransposeNode.label, TransposeNode.paramSchema);
}
