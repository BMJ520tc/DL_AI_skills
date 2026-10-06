// views/ReproduceView.tsx — 「论文复现」独立视图（模块二 4.1~4.4）。
// 入口在初始界面（项目列表）的「论文复现」卡片：选论文 + 绑定项目 → 到这里。
// 复现要跑在某个原始项目的独立环境里（模块一），故 projectId 必填。

import { useEffect, useState } from "react";
import { getProject, type Project } from "../api/client";
import ReproducePanel from "./panels/ReproducePanel";

export type ReproduceViewProps = {
    projectId: string;
    paperId?: string;
    onBack: () => void;
};

export default function ReproduceView({ projectId, paperId, onBack }: ReproduceViewProps) {
    const [project, setProject] = useState<Project | null>(null);

    useEffect(() => {
        void (async () => {
            try {
                setProject(await getProject(projectId));
            } catch {
                setProject(null);
            }
        })();
    }, [projectId]);

    return (
        <div style={{ minHeight: "100vh", background: "#0b1220", color: "#e2e8f0", padding: "24px 40px" }}>
            <div style={{ maxWidth: 1100, margin: "0 auto" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16 }}>
                    <button style={btnStyle} onClick={onBack}>← 返回</button>
                    <div>
                        <div style={{ fontWeight: 700, fontSize: 16 }}>论文复现</div>
                        <div style={{ fontSize: 11, color: "#94a3b8" }}>
                            复现绑定项目：
                            <span style={{ fontFamily: "monospace" }}>
                                {project?.name || "(未命名)"} · {projectId.slice(0, 8)}
                            </span>
                            （在它的独立环境里执行）
                        </div>
                    </div>
                </div>
                <div style={{ background: "#0f172a", border: "1px solid #1f2937", borderRadius: 10, padding: 18 }}>
                    <ReproducePanel projectId={projectId} initialPaperId={paperId} />
                </div>
            </div>
        </div>
    );
}

const btnStyle = {
    border: "1px solid #334155",
    background: "#0f766e",
    color: "#e2e8f0",
    borderRadius: 6,
    padding: "5px 12px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
} as const;
