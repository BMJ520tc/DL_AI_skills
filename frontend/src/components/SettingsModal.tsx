import { useEffect, useState } from "react";
import type { CSSProperties } from "react";
import {
    fetchCredentialsStatus,
    fetchEnvCheck,
    saveCredentials,
    type CredentialsStatus,
    type EnvCheckResult,
} from "../api/systemClient";

/**
 * 设置弹窗（一键封装 6.6-a）：凭证设置页 + 运行环境自检。
 *
 * - 凭证（评审裁定第 6 项）：api_key / base_url / model / small_model 四字段；
 *   响应只回掩码不回明文（K2），保存后立即生效（agent 会话开始时读文件覆盖环境）。
 * - 环境自检：git / Python / conda 探测 + 数据目录可写性；缺 git/Python 给安装指引
 *   链接与「重新检测」按钮（评审 K1 口径：外置依赖让学生自装，不捆绑进包）。
 */

type Props = {
    open: boolean;
    firstRun?: boolean;          // 首启弹窗时显示「稍后再说」
    onClose: () => void;
};

const OVERLAY: CSSProperties = {
    position: "fixed",
    inset: 0,
    background: "rgba(0, 0, 0, 0.55)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 60,
};

const PANEL: CSSProperties = {
    width: 620,
    maxWidth: "94vw",
    maxHeight: "86vh",
    overflowY: "auto",
    background: "#12151b",
    border: "1px solid #262b36",
    borderRadius: 10,
    padding: "16px 20px",
    color: "#cbd5e1",
    fontSize: 13,
};

const SECTION: CSSProperties = {
    border: "1px solid #1e3a5f",
    background: "#0b1a2b",
    borderRadius: 7,
    padding: "10px 12px",
    margin: "10px 0",
};

const FIELD_ROW: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 8,
    margin: "6px 0",
};

const FIELD_LABEL: CSSProperties = { width: 110, flexShrink: 0, color: "#94a3b8" };

const INPUT: CSSProperties = {
    flex: 1,
    background: "#0f1722",
    border: "1px solid #334155",
    borderRadius: 5,
    color: "#e2e8f0",
    padding: "5px 8px",
    fontSize: 12,
    minWidth: 0,
};

const BUTTON: CSSProperties = {
    background: "#1d4ed8",
    border: "none",
    borderRadius: 5,
    color: "#e2e8f0",
    padding: "6px 14px",
    fontSize: 12,
    cursor: "pointer",
};

const BUTTON_GRAY: CSSProperties = {
    ...BUTTON,
    background: "#334155",
};

const STATUS_LINE: CSSProperties = { marginTop: 6, fontSize: 12, color: "#7dd3fc" };
const ERROR_LINE: CSSProperties = { marginTop: 6, fontSize: 12, color: "#fca5a5" };

function StatusIcon({ ok }: { ok: boolean }) {
    return <span style={{ color: ok ? "#4ade80" : "#f87171", width: 16, display: "inline-block" }}>{ok ? "✓" : "✗"}</span>;
}

function InstallLink({ href, text }: { href: string; text: string }) {
    return (
        <a href={href} target="_blank" rel="noreferrer" style={{ color: "#7dd3fc" }}>
            {text}
        </a>
    );
}

export default function SettingsModal({ open, firstRun, onClose }: Props) {
    const [credStatus, setCredStatus] = useState<CredentialsStatus | null>(null);
    const [envCheck, setEnvCheck] = useState<EnvCheckResult | null>(null);
    const [apiKey, setApiKey] = useState("");
    const [baseUrl, setBaseUrl] = useState("");
    const [model, setModel] = useState("");
    const [smallModel, setSmallModel] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!open) return;
        let cancelled = false;
        setError(null);
        void (async () => {
            try {
                const [cred, env] = await Promise.all([fetchCredentialsStatus(), fetchEnvCheck()]);
                if (cancelled) return;
                setCredStatus(cred);
                setEnvCheck(env);
                if (cred.configured) {
                    // 已配置时只展示掩码，不把任何字段回填进表单（避免泄露旧值）
                    setApiKey("");
                    setBaseUrl("");
                    setModel("");
                    setSmallModel("");
                }
            } catch (e) {
                if (!cancelled) setError(e instanceof Error ? e.message : String(e));
            }
        })();
        return () => {
            cancelled = true;
        };
    }, [open]);

    if (!open) return null;

    const doSave = async () => {
        setBusy(true);
        setError(null);
        try {
            const next = await saveCredentials({ api_key: apiKey, base_url: baseUrl, model, small_model: smallModel });
            setCredStatus(next);
            setApiKey("");
            setBaseUrl("");
            setModel("");
            setSmallModel("");
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setBusy(false);
        }
    };

    const doClear = async () => {
        setBusy(true);
        setError(null);
        try {
            const next = await saveCredentials({ api_key: "", base_url: "", model: "", small_model: "" });
            setCredStatus(next);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setBusy(false);
        }
    };

    const doRecheck = async () => {
        setError(null);
        try {
            setEnvCheck(await fetchEnvCheck());
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    };

    return (
        <div style={OVERLAY} role="dialog" aria-label="设置">
            <div style={PANEL}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 15, fontWeight: 600 }}>⚙ 设置</span>
                    <button onClick={onClose} style={{ ...BUTTON_GRAY, padding: "2px 9px" }}>关闭</button>
                </div>

                <div style={{ fontSize: 13, fontWeight: 600, marginTop: 4 }}>模型接口凭证</div>
                <div style={SECTION}>
                    {credStatus?.configured ? (
                        <div style={{ color: "#94a3b8", marginBottom: 8 }}>
                            当前已配置（来源：{credStatus.source === "file" ? "本地配置文件" : "环境变量"}）
                            {credStatus.key_mask ? `，密钥 ${credStatus.key_mask}` : ""}
                            {credStatus.model ? `，模型 ${credStatus.model}` : ""}
                            {credStatus.base_url ? `，接口 ${credStatus.base_url}` : ""}
                            。重新填写并保存可覆盖；密钥只保存在本机数据目录，接口不回明文。
                        </div>
                    ) : (
                        <div style={{ color: "#fca5a5", marginBottom: 8 }}>
                            尚未配置模型接口凭证：需要调用大模型的步骤将直接失败（这是故意设的闸门，不会伪造结论）。
                        </div>
                    )}
                    <div style={FIELD_ROW}>
                        <span style={FIELD_LABEL}>API Key</span>
                        <input style={INPUT} type="password" value={apiKey} placeholder={credStatus?.configured ? "留空则保留现有密钥" : "sk-..."} onChange={e => setApiKey(e.target.value)} />
                    </div>
                    <div style={FIELD_ROW}>
                        <span style={FIELD_LABEL}>接口地址</span>
                        <input style={INPUT} value={baseUrl} placeholder="留空使用默认（https://api.deepseek.com/anthropic）" onChange={e => setBaseUrl(e.target.value)} />
                    </div>
                    <div style={FIELD_ROW}>
                        <span style={FIELD_LABEL}>默认模型</span>
                        <input style={INPUT} value={model} placeholder="如 deepseek-chat" onChange={e => setModel(e.target.value)} />
                    </div>
                    <div style={FIELD_ROW}>
                        <span style={FIELD_LABEL}>轻量模型</span>
                        <input style={INPUT} value={smallModel} placeholder="小任务模型，可选" onChange={e => setSmallModel(e.target.value)} />
                    </div>
                    <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                        <button onClick={doSave} disabled={busy} style={{ ...BUTTON, opacity: busy ? 0.6 : 1 }}>
                            {busy ? "保存中…" : "保存"}
                        </button>
                        {credStatus?.configured ? (
                            <button onClick={doClear} disabled={busy} style={BUTTON_GRAY}>清除凭证</button>
                        ) : null}
                    </div>
                    {credStatus?.source === "env" ? (
                        <div style={STATUS_LINE}>当前凭证来自环境变量；保存后改由本机配置文件提供（优先级更高）。</div>
                    ) : null}
                </div>

                <div style={{ fontSize: 13, fontWeight: 600 }}>运行环境自检</div>
                <div style={SECTION}>
                    {envCheck ? (
                        <>
                            <div style={FIELD_ROW}>
                                <StatusIcon ok={envCheck.git.found} />
                                <span>Git</span>
                                {envCheck.git.found ? (
                                    <span style={{ color: "#64748b" }}>{envCheck.git.version}</span>
                                ) : (
                                    <span style={{ color: "#fca5a5" }}>
                                        未检测到。Git 未随本程序打包，请自行安装：
                                        <InstallLink href="https://git-scm.com/downloads/win" text="git-scm.com/downloads/win" />
                                    </span>
                                )}
                            </div>
                            <div style={FIELD_ROW}>
                                <StatusIcon ok={envCheck.python.found} />
                                <span>Python</span>
                                {envCheck.python.found ? (
                                    <span style={{ color: "#64748b" }}>{envCheck.python.python ?? envCheck.python.py_launcher}</span>
                                ) : (
                                    <span style={{ color: "#fca5a5" }}>
                                        未检测到。Python 未随本程序打包（各项目独立环境需它自建），请自行安装：
                                        <InstallLink href="https://www.python.org/downloads/" text="python.org/downloads" />
                                    </span>
                                )}
                            </div>
                            <div style={FIELD_ROW}>
                                <StatusIcon ok={envCheck.conda.found} />
                                <span>conda</span>
                                <span style={{ color: "#64748b" }}>
                                    {envCheck.conda.found
                                        ? envCheck.conda.path
                                        : "未检测到（可选：建环境时可改走 venv）。安装指引："}
                                    {!envCheck.conda.found ? (
                                        <InstallLink href="https://docs.conda.io/en/latest/miniconda.html" text="Miniconda 安装文档" />
                                    ) : null}
                                </span>
                            </div>
                            <div style={FIELD_ROW}>
                                <StatusIcon ok={envCheck.claude_cli.found} />
                                <span>Claude Code CLI</span>
                                {envCheck.claude_cli.found ? (
                                    <span style={{ color: "#64748b" }}>
                                        {envCheck.claude_cli.version ?? envCheck.claude_cli.path}
                                    </span>
                                ) : (
                                    <span style={{ color: "#fca5a5" }}>
                                        未检测到。大模型步骤（复现 / 拆解 / 蒸馏 / 助手）需要它，请自行安装：
                                        <InstallLink href="https://www.npmjs.com/package/@anthropic-ai/claude-code"
                                                     text="npm i -g @anthropic-ai/claude-code" />
                                    </span>
                                )}
                            </div>
                            <div style={FIELD_ROW}>
                                <StatusIcon ok={envCheck.data_dir.writable} />
                                <span>数据目录</span>
                                <span style={{ color: "#64748b", wordBreak: "break-all" }}>
                                    {envCheck.data_dir.path}
                                    {envCheck.data_dir.writable ? "" : "（不可写！请检查权限）"}
                                </span>
                            </div>
                            {envCheck.static_served ? (
                                <div style={FIELD_ROW}>
                                    <StatusIcon ok />
                                    <span>界面由本程序同源服务</span>
                                </div>
                            ) : null}
                            {envCheck.pip_index ? (
                                <div style={FIELD_ROW}>
                                    <StatusIcon ok />
                                    <span>pip 镜像：{envCheck.pip_index}</span>
                                </div>
                            ) : null}
                            <div style={{ marginTop: 8 }}>
                                <button onClick={doRecheck} style={BUTTON_GRAY}>重新检测</button>
                                <span style={{ color: "#64748b", marginLeft: 8 }}>
                                    装好 git / Python 后点这里刷新（装完无需重启程序）。
                                </span>
                            </div>
                        </>
                    ) : (
                        <div style={{ color: "#64748b" }}>检测中…</div>
                    )}
                </div>

                {error ? <div style={ERROR_LINE}>⚠ {error}</div> : null}

                <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 6 }}>
                    {firstRun ? (
                        <button onClick={onClose} style={BUTTON_GRAY}>稍后再说</button>
                    ) : null}
                    <button onClick={onClose} style={BUTTON}>完成</button>
                </div>
            </div>
        </div>
    );
}
