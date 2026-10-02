// components/ErrorBoundary.tsx — 顶层错误边界。
// 原先任何渲染期异常都会让 React 卸载整棵树 → 页面全黑且没有任何线索；
// 这里把异常拦下并显示错误信息（含组件栈），便于定位与反馈。

import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error: Error | null; info: string };

export default class ErrorBoundary extends Component<Props, State> {
    state: State = { error: null, info: "" };

    static getDerivedStateFromError(error: Error): Partial<State> {
        return { error };
    }

    componentDidCatch(error: Error, info: ErrorInfo): void {
        console.error("界面渲染出错：", error, info.componentStack);
        this.setState({ info: info.componentStack ?? "" });
    }

    render() {
        const { error, info } = this.state;
        if (!error) return this.props.children;
        return (
            <div style={{ minHeight: "100vh", background: "#0b1220", color: "#e2e8f0", padding: 24, fontFamily: "system-ui, sans-serif" }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#f87171", marginBottom: 8 }}>界面出错了（已拦下，未黑屏）</div>
                <div style={{ fontSize: 13, marginBottom: 12, color: "#fca5a5" }}>{error.message || String(error)}</div>
                <pre style={{ fontSize: 11, color: "#94a3b8", background: "#0f172a", border: "1px solid #1f2937", borderRadius: 6, padding: 12, maxHeight: 320, overflow: "auto", whiteSpace: "pre-wrap" }}>
                    {error.stack || ""}
                    {info ? `\n组件栈:${info}` : ""}
                </pre>
                <button
                    onClick={() => window.location.reload()}
                    style={{ marginTop: 12, padding: "6px 14px", borderRadius: 8, border: "1px solid #1f2a2f", background: "#0f766e", color: "#e2e8f0", fontWeight: 600, cursor: "pointer" }}
                >
                    重新加载
                </button>
                <div style={{ marginTop: 12, fontSize: 12, color: "#64748b" }}>
                    请把上面的错误信息（含首行）反馈，便于精确定位。
                </div>
            </div>
        );
    }
}
