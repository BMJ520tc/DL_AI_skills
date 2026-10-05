// components/AssistantPanel.tsx — 前端 AI 助手面板（**流式**；模式：只读 / 可写）。
//
// 事件经 SSE 逐条到达：stage（思考中）/ delta（正文增量）/ tool（工具调用）/ done / error。
// 只读模式（默认）只给读工具 + 平台只读查询；可写模式给满工具。会话续接靠 session_id。

import { useState } from "react";
import type { CSSProperties } from "react";
import {
    assistantStreamUrl,
    startAssistantChat,
    type AssistantContext,
    type AssistantMode,
} from "../api/assistantClient";

type Props = {
    open: boolean;
    context: AssistantContext;
    onClose: () => void;
    /** 报错是「未配置凭证」时，引导用户直接打开设置页 */
    onOpenSettings?: () => void;
};

type Msg = { role: "user" | "assistant"; text: string };

// 右侧整高抽屉（不再是小浮窗）
const PANEL: CSSProperties = {
    position: "fixed", top: 0, right: 0, bottom: 0,
    width: 460, maxWidth: "96vw",
    display: "flex", flexDirection: "column",
    background: "#12151b", borderLeft: "1px solid #262b36",
    boxShadow: "-10px 0 28px rgba(0, 0, 0, 0.38)",
    color: "#cbd5e1", fontSize: 13, zIndex: 60,
};
const MSG_BOX: CSSProperties = {
    flex: 1, overflowY: "auto", padding: "14px 16px",
    display: "flex", flexDirection: "column", gap: 10,
};
const INPUT: CSSProperties = {
    flex: 1, background: "#0f1722", border: "1px solid #334155", borderRadius: 6,
    color: "#e2e8f0", padding: "6px 8px", fontSize: 12, resize: "none",
};

function bubble(role: Msg["role"]): CSSProperties {
    return {
        alignSelf: role === "user" ? "flex-end" : "flex-start",
        maxWidth: "88%",
        background: role === "user" ? "#1d4ed8" : "#1e293b",
        border: "1px solid " + (role === "user" ? "#2563eb" : "#334155"),
        borderRadius: 8, padding: "6px 9px", whiteSpace: "pre-wrap", wordBreak: "break-word",
    };
}

function tab(active: boolean): CSSProperties {
    return {
        background: active ? "#1d4ed8" : "#334155", border: "none", borderRadius: 5,
        color: "#e2e8f0", padding: "2px 8px", fontSize: 11, cursor: "pointer",
    };
}

export default function AssistantPanel({ open, context, onClose, onOpenSettings }: Props) {
    const [messages, setMessages] = useState<Msg[]>([]);
    const [input, setInput] = useState("");
    const [streamText, setStreamText] = useState("");
    const [stage, setStage] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [mode, setMode] = useState<AssistantMode>("read");

    if (!open) return null;

    const send = async () => {
        const text = input.trim();
        if (!text || busy) return;
        setError(null);
        setInput("");
        setStreamText("");
        setStage("思考中…");
        setMessages(m => [...m, { role: "user", text }]);
        setBusy(true);

        let taskId: string;
        try {
            taskId = await startAssistantChat(text, context, sessionId, mode);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
            setBusy(false);
            setStage("");
            return;
        }
        const es = new EventSource(assistantStreamUrl(taskId));
        let acc = "";
        es.onmessage = e => {
            let ev: { kind?: string; text?: string; name?: string; reply?: string;
                      session_id?: string | null; message?: string };
            try {
                ev = JSON.parse(e.data) as typeof ev;
            } catch {
                return;
            }
            if (ev.kind === "delta") {
                acc += ev.text || "";
                setStreamText(acc);
            } else if (ev.kind === "tool") {
                setStage(`调用工具 ${ev.name}…`);
            } else if (ev.kind === "stage") {
                setStage(ev.text || "");
            } else if (ev.kind === "done") {
                if (ev.session_id) setSessionId(ev.session_id);
                setMessages(m => [...m, { role: "assistant", text: ev.reply || acc || "（未返回内容）" }]);
                setStreamText("");
                setStage("");
                setBusy(false);
                es.close();
            } else if (ev.kind === "error") {
                setError(ev.message || "出错了");
                setStreamText("");
                setStage("");
                setBusy(false);
                es.close();
            }
        };
        es.onerror = () => {
            setError("流中断（后端可能已重启或网络异常）");
            setBusy(false);
            setStage("");
            es.close();
        };
    };

    return (
        <div style={PANEL} role="dialog" aria-label="AI 助手">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "9px 12px", borderBottom: "1px solid #262b36" }}>
                <span style={{ fontWeight: 600 }}>🤖 AI 助手</span>
                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ color: "#64748b", fontSize: 11 }}>模式</span>
                    <button onClick={() => setMode("read")} disabled={busy} style={tab(mode === "read")} title="只读：能读代码/查库/查平台，不改任何东西">只读</button>
                    <button onClick={() => setMode("write")} disabled={busy} style={tab(mode === "write")} title="可写：满工具（能改文件/跑命令）——会先说明再动手">可写</button>
                    <button onClick={onClose} style={{ background: "#334155", border: "none", borderRadius: 5, color: "#e2e8f0", padding: "2px 9px", fontSize: 12, cursor: "pointer" }}>关闭</button>
                </span>
            </div>
            <div style={MSG_BOX}>
                {messages.length === 0 ? (
                    <div style={{ color: "#64748b" }}>
                        可以问当前项目/论文/运行记录相关的问题，助手会先查再答、给建议（能查知识库）。默认「只读」——只看和被问，不改任何东西。
                    </div>
                ) : null}
                {messages.map((m, i) => (
                    <div key={i} style={bubble(m.role)}>{m.text}</div>
                ))}
                {busy ? (
                    <div style={{ alignSelf: "flex-start", maxWidth: "88%", color: "#94a3b8", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                        {streamText || ""}
                        {stage ? <span style={{ color: "#7dd3fc" }}>{streamText ? " " : ""}{stage}</span> : null}
                    </div>
                ) : null}
                {error ? (
                    <div style={{ color: "#fca5a5" }}>
                        ⚠ {error}
                        {error.includes("凭证") && onOpenSettings ? (
                            <button
                                onClick={onOpenSettings}
                                style={{ marginLeft: 8, background: "#1d4ed8", border: "none", borderRadius: 5, color: "#e2e8f0", padding: "3px 10px", fontSize: 12, cursor: "pointer" }}
                            >
                                去填凭证
                            </button>
                        ) : null}
                    </div>
                ) : null}
            </div>
            <div style={{ display: "flex", gap: 6, padding: "8px 10px", borderTop: "1px solid #262b36" }}>
                <textarea
                    style={INPUT}
                    rows={2}
                    placeholder="问点什么…（Enter 发送 / Shift+Enter 换行）"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => {
                        if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            void send();
                        }
                    }}
                />
                <button
                    onClick={() => void send()}
                    disabled={busy}
                    style={{ background: "#1d4ed8", border: "none", borderRadius: 6, color: "#e2e8f0", padding: "6px 12px", fontSize: 12, cursor: "pointer", opacity: busy ? 0.6 : 1 }}
                >
                    {busy ? "…" : "发送"}
                </button>
            </div>
        </div>
    );
}
