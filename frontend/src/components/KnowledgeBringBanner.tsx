import { useEffect, useState } from "react";
import type { CSSProperties } from "react";
import { bringKnowledge, type KnowledgeBringResult } from "../api/client";

/**
 * 任务前知识带入建议横幅（模块详细设计 8.2）。
 *
 * 传入任务维度（task_type / model / dataset）后调 POST /api/knowledge/bring，
 * 把「参数建议 / 依赖冲突预警」以可读横幅展示（建议而非强制，无命中不渲染）。
 * 供原始项目查看器（模块一/三）等入口复用；画布运行面板自带默认填入逻辑，不用本组件。
 */

type Props = {
    task_type?: string;
    model?: string;
    dataset?: string;
    title?: string;
};

const BOX: CSSProperties = {
    margin: "8px 0",
    padding: "8px 10px",
    border: "1px solid #1e3a5f",
    background: "#0b1a2b",
    borderRadius: 7,
    fontSize: 12,
    color: "#cbd5e1",
    lineHeight: 1.55,
};

export default function KnowledgeBringBanner({ task_type, model, dataset, title = "知识库带入建议" }: Props) {
    const [advice, setAdvice] = useState<KnowledgeBringResult | null>(null);

    useEffect(() => {
        let cancelled = false;
        void (async () => {
            try {
                const result = await bringKnowledge({ task_type, model, dataset });
                if (!cancelled) setAdvice(result);
            } catch {
                if (!cancelled) setAdvice(null);   // 带入失败静默（8.2 异常边界）
            }
        })();
        return () => {
            cancelled = true;
        };
    }, [task_type, model, dataset]);

    if (!advice || (advice.param_advice.length === 0 && advice.dependency_conflict.length === 0)) {
        return null;
    }

    return (
        <div style={BOX}>
            <div style={{ color: "#7dd3fc", fontWeight: 600, marginBottom: 3 }}>{title}</div>
            {advice.param_advice.map((a, i) => (
                <div key={`pa-${i}`}>· 建议：{a.title || a.content}</div>
            ))}
            {advice.dependency_conflict.map((a, i) => (
                <div key={`dc-${i}`} style={{ color: "#fca5a5" }}>⚠ 依赖冲突：{a.title || a.content}</div>
            ))}
            <div style={{ color: "#64748b", marginTop: 3 }}>（来自已确认蒸馏知识，供参考，可忽略）</div>
        </div>
    );
}
