// views/panels/SearchDownloadPanel.tsx — 模块一 2.4「检索与下载」界面入口。
// 需求一：论文检索（arxiv/pubmed/biorxiv，含作者与时间范围过滤、PubMed/bioRxiv 全文）→
// 单篇/批量下载入库；地址抽取（仓库地址完整克隆 + 数据集登记，逐条结果进任务进度）。
// 全部驱动既有后端接口（/api/search/*），任务轮询复用 useTaskPolling。

import { useState, type CSSProperties } from "react";
import {
    downloadPaper,
    downloadPapersBatch,
    postExtractAddresses,
    searchPapers,
    type PaperSearchParams,
    type Task,
} from "../../api/client";
import { useTaskPolling } from "../../hooks/useTaskPolling";

/** 检索结果条目（字段随来源而异，取到的才展示）。 */
type Hit = Record<string, unknown>;

function hitId(hit: Hit): string {
    return String(hit.paper_id ?? "");
}

function hitTitle(hit: Hit): string {
    const title = hit.title;
    return typeof title === "string" && title.trim() ? title : "(无标题)";
}

/** 任务进度（JSON 文本）→ 对象；拿不到返回 null。 */
function parseProgress(raw: string | null): Record<string, unknown> | null {
    if (!raw) return null;
    try {
        const parsed = JSON.parse(raw) as unknown;
        return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : null;
    } catch {
        return null;
    }
}

const SOURCES: Array<{ value: NonNullable<PaperSearchParams["source"]>; label: string }> = [
    { value: "arxiv", label: "arXiv" },
    { value: "pubmed", label: "PubMed" },
    { value: "biorxiv", label: "bioRxiv" },
];

export default function SearchDownloadPanel() {
    // 检索
    const [q, setQ] = useState("");
    const [source, setSource] = useState<NonNullable<PaperSearchParams["source"]>>("arxiv");
    const [maxResults, setMaxResults] = useState(10);
    const [authors, setAuthors] = useState("");
    const [dateFrom, setDateFrom] = useState("");
    const [dateTo, setDateTo] = useState("");
    const [fulltext, setFulltext] = useState(false);
    const [hits, setHits] = useState<Hit[] | null>(null);
    const [selected, setSelected] = useState<Set<string>>(() => new Set());
    const [searching, setSearching] = useState(false);
    const [downloading, setDownloading] = useState(false);
    const [downloadResults, setDownloadResults] = useState<Array<Record<string, unknown>>>([]);
    const [banner, setBanner] = useState<string | null>(null);
    const [flash, setFlash] = useState<string | null>(null);

    // 地址抽取
    const [paperText, setPaperText] = useState("");
    const [extractPaperId, setExtractPaperId] = useState("");
    const [supplementary, setSupplementary] = useState("");
    const [cloneRepos, setCloneRepos] = useState(true);
    const [fullClone, setFullClone] = useState(true);
    const [extractTaskId, setExtractTaskId] = useState<string | null>(null);
    const [extractProgress, setExtractProgress] = useState<Record<string, unknown> | null>(null);

    const runSearch = async () => {
        if (!q.trim()) {
            setBanner("请填写检索关键词");
            return;
        }
        setSearching(true);
        setBanner(null);
        setFlash(null);
        setDownloadResults([]);
        try {
            const rows = await searchPapers({
                q: q.trim(), source, max_results: maxResults,
                authors: authors.trim() || undefined,
                date_from: dateFrom.trim() || undefined,
                date_to: dateTo.trim() || undefined,
                fulltext,
            });
            setHits(rows);
            setSelected(new Set());
        } catch (e) {
            setHits(null);
            setBanner(`检索失败（${source}）：${e instanceof Error ? e.message : String(e)}`);
        } finally {
            setSearching(false);
        }
    };

    const toggle = (id: string) => {
        setSelected(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    const doSingleDownload = async (hit: Hit) => {
        setDownloading(true);
        setBanner(null);
        try {
            const res = await downloadPaper({
                paper_id: hitId(hit),
                pdf_url: typeof hit.pdf_url === "string" ? hit.pdf_url : null,
                title: typeof hit.title === "string" ? hit.title : null,
                abstract: typeof hit.abstract === "string" ? hit.abstract : null,
                source: typeof hit.source === "string" ? hit.source : source,
            });
            setFlash(`已下载并入库：${res.paper_id}（${res.status}）`);
        } catch (e) {
            setBanner(`下载失败（${hitTitle(hit)}）：${e instanceof Error ? e.message : String(e)}`);
        } finally {
            setDownloading(false);
        }
    };

    const doBatchDownload = async () => {
        const chosen = (hits ?? []).filter(h => selected.has(hitId(h)));
        if (!chosen.length) {
            setBanner("请先勾选要下载的论文");
            return;
        }
        setDownloading(true);
        setBanner(null);
        try {
            const res = await downloadPapersBatch(chosen.map(h => ({
                paper_id: hitId(h),
                pdf_url: typeof h.pdf_url === "string" ? h.pdf_url : null,
                title: typeof h.title === "string" ? h.title : null,
                abstract: typeof h.abstract === "string" ? h.abstract : null,
                source: typeof h.source === "string" ? h.source : source,
            })));
            setDownloadResults(res.results as unknown as Array<Record<string, unknown>>);
            setFlash(`批量下载完成：成功 ${res.succeeded} / 失败 ${res.failed}（共 ${res.total}）${res.note ? ` · ${res.note}` : ""}`);
        } catch (e) {
            setBanner(`批量下载失败：${e instanceof Error ? e.message : String(e)}`);
        } finally {
            setDownloading(false);
        }
    };

    const runExtract = async () => {
        if (!paperText.trim()) {
            setBanner("请粘贴论文正文（地址抽取以正文为准）");
            return;
        }
        setBanner(null);
        setFlash(null);
        setExtractProgress(null);
        try {
            const paths = supplementary.split(/[;；\n]+/).map(s => s.trim()).filter(Boolean);
            const res = await postExtractAddresses({
                paper_text: paperText,
                paper_id: extractPaperId.trim() || undefined,
                supplementary_paths: paths.length ? paths : undefined,
                clone_repos: cloneRepos,
                full_clone: fullClone,
            });
            setExtractTaskId(res.task_id);
        } catch (e) {
            setBanner(`地址抽取提交失败：${e instanceof Error ? e.message : String(e)}`);
        }
    };

    useTaskPolling({
        taskId: extractTaskId,
        onProgress: (t: Task) => {
            const parsed = parseProgress(t.progress);
            if (parsed) setExtractProgress(parsed);
        },
        onDone: () => {
            setExtractTaskId(null);
            setFlash("地址抽取完成（仓库克隆与数据集登记结果见下方进度）");
        },
        onError: message => {
            setExtractTaskId(null);
            setBanner(message);
        },
    });

    return (
        <div style={{ fontSize: 12 }}>
            {(banner || flash) && (
                <div style={{ marginBottom: 8 }}>
                    {banner && (
                        <div style={{ ...infoBanner, background: "#3f1d1d", borderColor: "#b91c1c", color: "#fecaca" }}>
                            {banner}
                            <button style={{ ...btnStyle, marginLeft: 10 }} onClick={() => setBanner(null)}>关闭</button>
                        </div>
                    )}
                    {flash && (
                        <div style={infoBanner}>
                            {flash}
                            <button style={{ ...btnStyle, marginLeft: 10 }} onClick={() => setFlash(null)}>关闭</button>
                        </div>
                    )}
                </div>
            )}

            {/* 论文检索 */}
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                <input
                    style={{ ...inputStyle, flex: 1, minWidth: 220 }}
                    placeholder="检索关键词（如 resnet image classification）"
                    value={q}
                    onChange={e => setQ(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && void runSearch()}
                />
                <select style={inputStyle} value={source} onChange={e => setSource(e.target.value as NonNullable<PaperSearchParams["source"]>)}>
                    {SOURCES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                </select>
                <input
                    style={{ ...inputStyle, width: 70 }}
                    type="number" min={1} max={50}
                    value={maxResults}
                    onChange={e => setMaxResults(Number(e.target.value))}
                    title="最多返回条数"
                />
                <input style={{ ...inputStyle, width: 130 }} placeholder="作者（逗号分隔）" value={authors} onChange={e => setAuthors(e.target.value)} />
                <input style={{ ...inputStyle, width: 110 }} placeholder="起始日期" value={dateFrom} onChange={e => setDateFrom(e.target.value)} title="YYYY[-MM[-DD]]" />
                <input style={{ ...inputStyle, width: 110 }} placeholder="截止日期" value={dateTo} onChange={e => setDateTo(e.target.value)} title="YYYY[-MM[-DD]]" />
                <label style={{ color: "#94a3b8", fontSize: 11 }}>
                    <input type="checkbox" checked={fulltext} onChange={e => setFulltext(e.target.checked)} /> 抓全文
                </label>
                <button style={btnStyle} disabled={searching} onClick={() => void runSearch()}>
                    {searching ? "检索中…" : "检索论文"}
                </button>
            </div>
            <div style={{ color: "#64748b", fontSize: 11, marginTop: 4 }}>
                bioRxiv 不支持服务端作者/时间过滤，命中条目会带 filter_note 如实说明；「抓全文」取不到时记 abstract_only，不报错。
            </div>

            {hits && (
                <div style={{ marginTop: 8 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                        <span style={{ color: "#94a3b8" }}>命中 {hits.length} 条 · 已勾选 {selected.size} 条</span>
                        <button style={btnStyle} disabled={downloading || selected.size === 0} onClick={() => void doBatchDownload()}>
                            {downloading ? "下载中…" : "批量下载所选"}
                        </button>
                    </div>
                    {hits.length === 0 ? (
                        <div style={{ color: "#64748b" }}>无命中（arXiv 异常查询会返回空列表）。</div>
                    ) : (
                        <table style={tableStyle}>
                            <thead>
                                <tr>
                                    <th style={thStyle} />
                                    <th style={thStyle}>标题</th>
                                    <th style={thStyle}>来源</th>
                                    <th style={thStyle}>日期/作者</th>
                                    <th style={thStyle}>PDF</th>
                                    <th style={thStyle}>操作</th>
                                </tr>
                            </thead>
                            <tbody>
                                {hits.map(hit => {
                                    const id = hitId(hit);
                                    return (
                                        <tr key={id}>
                                            <td style={tdStyle}>
                                                <input type="checkbox" checked={selected.has(id)} onChange={() => toggle(id)} />
                                            </td>
                                            <td style={{ ...tdStyle, maxWidth: 420 }}>
                                                {hitTitle(hit)}
                                                <div style={{ color: "#64748b", fontFamily: "monospace", fontSize: 10 }}>{id}</div>
                                                {typeof hit.filter_note === "string" && hit.filter_note && (
                                                    <div style={{ color: "#f59e0b", fontSize: 10 }}>{hit.filter_note}</div>
                                                )}
                                                {typeof hit.fulltext_status === "string" && (
                                                    <div style={{ color: "#64748b", fontSize: 10 }}>全文：{hit.fulltext_status}{typeof hit.fulltext_note === "string" ? `（${hit.fulltext_note}）` : ""}</div>
                                                )}
                                            </td>
                                            <td style={tdStyle}>{String(hit.source ?? "?")}</td>
                                            <td style={tdStyle}>
                                                {String(hit.published_date ?? hit.date ?? "—")}
                                                {Array.isArray(hit.authors) && hit.authors.length ? <div style={{ color: "#64748b", fontSize: 10 }}>{(hit.authors as unknown[]).slice(0, 2).map(String).join("、")}</div> : null}
                                            </td>
                                            <td style={tdStyle}>
                                                {typeof hit.pdf_url === "string" && hit.pdf_url
                                                    ? <a href={hit.pdf_url} target="_blank" rel="noreferrer" style={{ color: "#2dd4bf" }}>链接</a>
                                                    : "—"}
                                            </td>
                                            <td style={tdStyle}>
                                                <button style={btnStyle} disabled={downloading} onClick={() => void doSingleDownload(hit)}>下载</button>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    )}
                    {downloadResults.length > 0 && (
                        <div style={{ marginTop: 6, fontSize: 11 }}>
                            <div style={{ color: "#94a3b8" }}>批量下载逐条结果：</div>
                            {downloadResults.map((r, i) => (
                                <div key={i} style={{ fontFamily: "monospace", color: r.status === "failed" ? "#fca5a5" : "#86efac" }}>
                                    [{String(r.status)}] {String(r.paper_id)}
                                    {r.reason ? ` — ${String(r.reason).slice(0, 160)}` : ""}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* 地址抽取 */}
            <div style={{ marginTop: 12, borderTop: "1px solid #1f2937", paddingTop: 8 }}>
                <div style={{ fontWeight: 700, marginBottom: 4 }}>地址抽取（仓库地址完整克隆 + 数据集登记）</div>
                <textarea
                    style={{ ...inputStyle, width: "100%", minHeight: 60, fontFamily: "monospace" }}
                    placeholder="粘贴论文正文（抽仓库地址与数据集地址）"
                    value={paperText}
                    onChange={e => setPaperText(e.target.value)}
                />
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", marginTop: 6 }}>
                    <input
                        style={{ ...inputStyle, width: 220 }}
                        placeholder="paper_id（可选，顺带扫补充材料）"
                        value={extractPaperId}
                        onChange={e => setExtractPaperId(e.target.value)}
                    />
                    <input
                        style={{ ...inputStyle, flex: 1, minWidth: 220 }}
                        placeholder="补充材料路径（可选，多个用分号分隔）"
                        value={supplementary}
                        onChange={e => setSupplementary(e.target.value)}
                    />
                    <label style={{ color: "#94a3b8", fontSize: 11 }}>
                        <input type="checkbox" checked={cloneRepos} onChange={e => setCloneRepos(e.target.checked)} /> 抽到仓库即克隆
                    </label>
                    <label style={{ color: "#94a3b8", fontSize: 11 }}>
                        <input type="checkbox" checked={fullClone} onChange={e => setFullClone(e.target.checked)} disabled={!cloneRepos} /> 完整克隆
                    </label>
                    <button style={btnStyle} disabled={!!extractTaskId} onClick={() => void runExtract()}>
                        {extractTaskId ? "抽取中…" : "提交地址抽取"}
                    </button>
                </div>
                {extractProgress && (
                    <div style={{ ...infoBanner, background: "#0f172a", borderColor: "#334155", color: "#cbd5e1", marginTop: 6 }}>
                        <pre style={{ margin: 0, fontSize: 11, whiteSpace: "pre-wrap" }}>
                            {JSON.stringify(extractProgress, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        </div>
    );
}

const btnStyle: CSSProperties = {
    border: "1px solid #334155",
    background: "#0f766e",
    color: "#e2e8f0",
    borderRadius: 6,
    padding: "4px 10px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
};
const inputStyle: CSSProperties = {
    border: "1px solid #334155",
    background: "#0f172a",
    color: "#e2e8f0",
    borderRadius: 4,
    padding: "3px 6px",
    fontSize: 12,
};
const infoBanner: CSSProperties = {
    background: "#0f2d1f",
    border: "1px solid #16a34a",
    color: "#bbf7d0",
    borderRadius: 8,
    padding: "8px 12px",
    fontSize: 12,
    marginBottom: 8,
};
const tableStyle: CSSProperties = { width: "100%", borderCollapse: "collapse", fontSize: 11 };
const thStyle: CSSProperties = { textAlign: "left", color: "#64748b", fontWeight: 600, padding: "3px 6px", borderBottom: "1px solid #1f2937" };
const tdStyle: CSSProperties = { padding: "4px 6px", borderBottom: "1px solid #1f2937", verticalAlign: "top" };
