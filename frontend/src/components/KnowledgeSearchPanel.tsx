import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  DATA_TYPE_COLORS,
  DATA_TYPE_LABELS,
  KNOWLEDGE_DATA_TYPES,
  buildItemUrl,
  buildSearchUrl,
  describeApiBaseUrl,
  fetchKnowledgeItem,
  formatTimestamp,
  normalizeDetailPayload,
  pickTitle,
  resolveApiBaseUrl,
  searchKnowledge,
  truncateSummary,
  type KnowledgeDataType,
  type KnowledgeSearchHit,
  type RequestFailure,
  type RequestOutcome,
  type SearchParams,
} from "../api/knowledgeClient";
import {
  confirmKnowledgeItem,
  getKnowledgeConflicts,
  listKnowledgeDrafts,
  supersedeKnowledgeItem,
  type KnowledgeRow,
} from "../api/client";

/**
 * 知识库检索面板（只读）。
 *
 * - 关键词 + 类型多选 -> GET /api/knowledge/search
 * - 点击结果 -> GET /api/knowledge/items/{data_type}/{ref_id}
 *
 * 独立自包含：不依赖画布状态，也不改动 FlowEditor 的任何既有逻辑。
 */

type Props = {
  onClose: () => void;
};

type DetailState =
  | { status: "idle" }
  | { status: "loading"; dataType: string; refId: string }
  | { status: "ready"; dataType: string; refId: string; payload: Record<string, unknown> }
  | { status: "error"; dataType: string; refId: string; failure: RequestFailure };

const PANEL_STYLE: CSSProperties = {
  position: "fixed",
  top: 0,
  right: 0,
  height: "100vh",
  width: "min(1080px, 94vw)",
  background: "#0f1115",
  borderLeft: "1px solid #262b36",
  boxShadow: "-18px 0 40px rgba(0, 0, 0, 0.55)",
  display: "flex",
  flexDirection: "column",
  zIndex: 60,
  color: "#e6edf3",
  fontSize: 13,
};

const SECTION_TITLE_STYLE: CSSProperties = {
  color: "#6b7280",
  fontSize: 10,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
};

const INPUT_STYLE: CSSProperties = {
  flex: 1,
  minWidth: 0,
  background: "#111827",
  color: "#e6edf3",
  border: "1px solid #374151",
  borderRadius: 6,
  padding: "7px 10px",
  fontSize: 13,
  outline: "none",
};

/** 维度筛选（task_type/model/dataset）输入框，较关键词框窄。 */
const FILTER_INPUT_STYLE: CSSProperties = {
  flex: "0 1 190px",
  minWidth: 160,
  background: "#111827",
  color: "#e6edf3",
  border: "1px solid #374151",
  borderRadius: 6,
  padding: "6px 9px",
  fontSize: 12,
  outline: "none",
};

/** 三维度筛选值。 */
type SearchFilters = { taskType: string; model: string; dataset: string };

/** task_type 静态建议（下拉可选项，实际以检索结果回填为主）。 */
const TASK_TYPE_SUGGESTIONS = [
  "classification",
  "detection",
  "segmentation",
  "generation",
  "regression",
  "other",
];

const BUTTON_BASE: CSSProperties = {
  padding: "7px 14px",
  borderRadius: 6,
  border: "1px solid #3f3f46",
  cursor: "pointer",
  fontSize: 13,
};

function typeLabel(dataType: string | undefined): string {
  if (dataType && dataType in DATA_TYPE_LABELS) return DATA_TYPE_LABELS[dataType as KnowledgeDataType];
  return dataType || "未知";
}

function typeColor(dataType: string | undefined): string {
  if (dataType && dataType in DATA_TYPE_COLORS) return DATA_TYPE_COLORS[dataType as KnowledgeDataType];
  return "#6b7280";
}

function FailureBox({ failure, onRetry }: { failure: RequestFailure; onRetry?: () => void }) {
  return (
    <div
      style={{
        border: "1px solid #7f1d1d",
        background: "#1f1113",
        borderRadius: 8,
        padding: 12,
        color: "#fecaca",
        lineHeight: 1.6,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>⚠ {failure.message}</div>
      <div style={{ color: "#fca5a5" }}>{failure.hint}</div>
      <div style={{ color: "#9ca3af", marginTop: 6, wordBreak: "break-all", fontSize: 12 }}>
        请求地址：{failure.url}
      </div>
      {onRetry ? (
        <button
          onClick={onRetry}
          style={{
            ...BUTTON_BASE,
            marginTop: 8,
            background: "#7f1d1d",
            color: "#fff",
            border: "1px solid #b91c1c",
          }}
        >
          重试
        </button>
      ) : null}
    </div>
  );
}

export default function KnowledgeSearchPanel({ onClose }: Props) {
  const baseUrl = useMemo(() => resolveApiBaseUrl(), []);

  const [query, setQuery] = useState("");
  const [selectedTypes, setSelectedTypes] = useState<KnowledgeDataType[]>([...KNOWLEDGE_DATA_TYPES]);
  const [limit, setLimit] = useState(20);
  // 需求六.1 三维度筛选
  const [taskType, setTaskType] = useState("");
  const [modelName, setModelName] = useState("");
  const [datasetName, setDatasetName] = useState("");

  // 首次浏览的检索参数（q 为空 => 后端返回最近条目）。组件挂载即发起，
  // 因此直接把初始状态置为「请求进行中」，effect 内无需同步 setState。
  const initialParams = useMemo<SearchParams>(
    () => ({
      q: "",
      types: [...KNOWLEDGE_DATA_TYPES],
      limit: 20,
      task_type: undefined,
      model: undefined,
      dataset: undefined,
    }),
    [],
  );

  const [hits, setHits] = useState<KnowledgeSearchHit[]>([]);
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState<RequestFailure | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [lastUrl, setLastUrl] = useState(() => buildSearchUrl(baseUrl, initialParams));

  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [detail, setDetail] = useState<DetailState>({ status: "idle" });
  const [showRaw, setShowRaw] = useState(false);
  // 结果按类型分栏（需求六.1 / 模块详细设计 8.1）：tab 当前选中项，"all" 表示全部
  const [activeTab, setActiveTab] = useState<string>("all");
  // 面板模式：检索 / 草稿确认（模块详细设计 8.3、K4）
  const [panelMode, setPanelMode] = useState<"search" | "drafts">("search");
  const [drafts, setDrafts] = useState<KnowledgeRow[]>([]);
  const [draftsLoading, setDraftsLoading] = useState(false);
  const [draftsError, setDraftsError] = useState<string | null>(null);
  const [conflictsByDraft, setConflictsByDraft] = useState<Record<string, KnowledgeRow[]>>({});
  const [busyDraft, setBusyDraft] = useState<string | null>(null);

  const searchAbort = useRef<AbortController | null>(null);
  const detailAbort = useRef<AbortController | null>(null);

  /** 落地一次检索结果。仅作结果归一化，不发起请求。 */
  const applyOutcome = useCallback(
    (outcome: RequestOutcome<KnowledgeSearchHit[]>, controller: AbortController) => {
      if (controller.signal.aborted) return;

      setLoading(false);
      setHasSearched(true);
      if (outcome.ok) {
        setHits(outcome.data);
      } else {
        setHits([]);
        setFailure(outcome.failure);
      }
    },
    [],
  );

  /** 发起请求并落地结果。调用方负责已经进入「请求进行中」状态。 */
  const performSearch = useCallback(
    async (params: SearchParams, controller: AbortController) => {
      const outcome = await searchKnowledge(baseUrl, params, controller.signal);
      applyOutcome(outcome, controller);
    },
    [baseUrl, applyOutcome],
  );

  const runSearch = useCallback(
    async (keyword: string, types: KnowledgeDataType[], size: number, filters: SearchFilters) => {
      searchAbort.current?.abort();
      const controller = new AbortController();
      searchAbort.current = controller;

      const params: SearchParams = {
        q: keyword,
        types,
        limit: size,
        task_type: filters.taskType.trim() || undefined,
        model: filters.model.trim() || undefined,
        dataset: filters.dataset.trim() || undefined,
      };

      setLoading(true);
      setFailure(null);
      setLastUrl(buildSearchUrl(baseUrl, params));

      await performSearch(params, controller);
    },
    [baseUrl, performSearch],
  );

  // 打开面板即浏览一次（q 为空 => 后端返回最近条目）
  // 结果在 then 回调里落地（外部系统 -> setState 的订阅式写法），
  // effect 体内不同步 setState。
  useEffect(() => {
    const controller = new AbortController();
    searchAbort.current = controller;
    searchKnowledge(baseUrl, initialParams, controller.signal).then(outcome =>
      applyOutcome(outcome, controller),
    );
    return () => {
      searchAbort.current?.abort();
      detailAbort.current?.abort();
    };
  }, [baseUrl, initialParams, applyOutcome]);

  // 以当前筛选条件发起检索
  const doSearch = useCallback(() => {
    void runSearch(query.trim(), selectedTypes, limit, { taskType, model: modelName, dataset: datasetName });
  }, [runSearch, query, selectedTypes, limit, taskType, modelName, datasetName]);

  // 维度下拉建议：静态 task_type + 已返回结果中出现过的取值
  const taskTypeOptions = useMemo(() => {
    const set = new Set<string>(TASK_TYPE_SUGGESTIONS);
    hits.forEach(hit => {
      if (hit.task_type) set.add(hit.task_type);
    });
    return Array.from(set).sort();
  }, [hits]);
  const modelOptions = useMemo(() => {
    const set = new Set<string>();
    hits.forEach(hit => {
      if (hit.model_name) set.add(hit.model_name);
    });
    return Array.from(set).sort();
  }, [hits]);
  const datasetOptions = useMemo(() => {
    const set = new Set<string>();
    hits.forEach(hit => {
      if (hit.dataset_name) set.add(hit.dataset_name);
    });
    return Array.from(set).sort();
  }, [hits]);

  // Esc 关闭
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const toggleType = (value: KnowledgeDataType) => {
    setSelectedTypes(previous =>
      previous.includes(value) ? previous.filter(item => item !== value) : [...previous, value],
    );
  };

  const openDetail = useCallback(
    async (hit: KnowledgeSearchHit) => {
      const dataType = hit.data_type || "knowledge";
      const refId = hit.ref_id || hit.id || "";
      setActiveKey(`${dataType}:${refId}`);
      setShowRaw(false);

      if (!refId) {
        setDetail({
          status: "error",
          dataType,
          refId,
          failure: {
            message: "该结果缺少 ref_id，无法取详情。",
            hint: "后端检索结果应包含 ref_id 字段。",
            url: "",
          },
        });
        return;
      }

      detailAbort.current?.abort();
      const controller = new AbortController();
      detailAbort.current = controller;

      setDetail({ status: "loading", dataType, refId });
      const outcome = await fetchKnowledgeItem(baseUrl, dataType, refId, controller.signal);
      if (controller.signal.aborted) return;

      if (outcome.ok) {
        setDetail({ status: "ready", dataType, refId, payload: outcome.data });
      } else {
        setDetail({ status: "error", dataType, refId, failure: outcome.failure });
      }
    },
    [baseUrl],
  );

  /** 加载待确认草稿 + 每个草稿的潜在冲突（8.3）。 */
  const loadDrafts = useCallback(async () => {
    setDraftsLoading(true);
    setDraftsError(null);
    try {
      const rows = await listKnowledgeDrafts();
      setDrafts(rows);
      const entries = await Promise.all(
        rows.map(async row => {
          try {
            return [row.knowledge_id, await getKnowledgeConflicts(row.knowledge_id)] as const;
          } catch {
            return [row.knowledge_id, [] as KnowledgeRow[]] as const;
          }
        }),
      );
      setConflictsByDraft(Object.fromEntries(entries));
    } catch (e) {
      setDraftsError(e instanceof Error ? e.message : String(e));
    } finally {
      setDraftsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (panelMode !== "drafts") return;
    void loadDrafts();
  }, [panelMode, loadDrafts]);

  const handleConfirmDraft = useCallback(
    async (id: string, supersede: boolean) => {
      setBusyDraft(id);
      setDraftsError(null);
      try {
        await confirmKnowledgeItem(id, supersede);
        await loadDrafts();
      } catch (e) {
        setDraftsError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyDraft(null);
      }
    },
    [loadDrafts],
  );

  const handleSupersedeKnowledge = useCallback(
    async (id: string) => {
      setBusyDraft(id);
      setDraftsError(null);
      try {
        await supersedeKnowledgeItem(id);
        await loadDrafts();
      } catch (e) {
        setDraftsError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyDraft(null);
      }
    },
    [loadDrafts],
  );

  const detailJson = useMemo(() => {
    if (detail.status !== "ready") return "";
    const normalized = normalizeDetailPayload(detail.payload);
    return JSON.stringify(showRaw ? detail.payload : normalized, null, 2);
  }, [detail, showRaw]);

  const detailContent = useMemo(() => {
    if (detail.status !== "ready") return "";
    const value = detail.payload.content;
    return typeof value === "string" && value.trim() !== "" ? value : "";
  }, [detail]);

  const handledCount = hits.length;

  // 分栏：按 data_type 统计与过滤（tab 内计数真实反映本次检索返回的条目）
  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    hits.forEach(hit => {
      const key = hit.data_type || "knowledge";
      counts[key] = (counts[key] || 0) + 1;
    });
    return counts;
  }, [hits]);
  const visibleHits = useMemo(
    () => (activeTab === "all" ? hits : hits.filter(hit => (hit.data_type || "knowledge") === activeTab)),
    [hits, activeTab],
  );

  return (
    <>
      {/* 半透明遮罩：点击关闭 */}
      <div
        onClick={onClose}
        style={{ position: "fixed", inset: 0, background: "rgba(0, 0, 0, 0.5)", zIndex: 59 }}
        aria-hidden="true"
      />
      <aside style={PANEL_STYLE} role="dialog" aria-label="知识库检索">
        {/* 头部 */}
        <div
          style={{
            padding: "12px 16px",
            borderBottom: "1px solid #262b36",
            display: "flex",
            alignItems: "center",
            gap: 12,
            background: "#12151b",
          }}
        >
          <span style={{ fontSize: 15, fontWeight: 600 }}>📚 知识库</span>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              onClick={() => setPanelMode("search")}
              style={{
                ...BUTTON_BASE, padding: "3px 12px", fontSize: 12,
                background: panelMode === "search" ? "#1d4ed8" : "#1f2937",
                color: panelMode === "search" ? "#fff" : "#cbd5e1",
                border: "1px solid #374151",
              }}
            >
              检索
            </button>
            <button
              onClick={() => setPanelMode("drafts")}
              style={{
                ...BUTTON_BASE, padding: "3px 12px", fontSize: 12,
                background: panelMode === "drafts" ? "#7c3aed" : "#1f2937",
                color: panelMode === "drafts" ? "#fff" : "#cbd5e1",
                border: "1px solid #374151",
              }}
            >
              草稿
            </button>
          </div>
          <span style={{ color: "#6b7280", fontSize: 12 }}>
            后端：<code style={{ color: "#9ca3af" }}>{describeApiBaseUrl(baseUrl)}</code>
          </span>
          <div style={{ flex: 1 }} />
          <button
            onClick={onClose}
            style={{ ...BUTTON_BASE, background: "#1f2937", color: "#e6edf3" }}
            title="关闭（Esc）"
          >
            关闭
          </button>
        </div>

        {panelMode === "search" ? (
        <>
        {/* 检索条件 */}
        <div style={{ padding: "12px 16px", borderBottom: "1px solid #262b36", display: "grid", gap: 10 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <input
              value={query}
              onChange={event => setQuery(event.target.value)}
              onKeyDown={event => {
                if (event.key === "Enter") doSearch();
              }}
              placeholder="输入关键词（留空则按类型浏览最近条目）"
              style={INPUT_STYLE}
            />
            <button
              onClick={doSearch}
              disabled={loading}
              style={{
                ...BUTTON_BASE,
                background: loading ? "#1f2937" : "#1d4ed8",
                color: "#fff",
                border: "1px solid #2563eb",
                cursor: loading ? "wait" : "pointer",
              }}
            >
              {loading ? "检索中…" : "检索"}
            </button>
          </div>

          <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
            <span style={SECTION_TITLE_STYLE}>类型</span>
            {KNOWLEDGE_DATA_TYPES.map(value => (
              <label
                key={value}
                style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer", color: "#cbd5e1" }}
              >
                <input
                  type="checkbox"
                  checked={selectedTypes.includes(value)}
                  onChange={() => toggleType(value)}
                  style={{ accentColor: typeColor(value) }}
                />
                <span style={{ color: typeColor(value), fontWeight: 600 }}>{DATA_TYPE_LABELS[value]}</span>
                <span style={{ color: "#6b7280", fontSize: 11 }}>{value}</span>
              </label>
            ))}
            <div style={{ flex: 1 }} />
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "#9ca3af" }}>
              条数
              <select
                value={limit}
                onChange={event => setLimit(Number(event.target.value))}
                style={{
                  background: "#1f2937",
                  color: "#e5e7eb",
                  border: "1px solid #374151",
                  borderRadius: 4,
                  padding: "3px 6px",
                }}
              >
                {[10, 20, 50, 100].map(size => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {/* 需求六.1 三维度：task_type / model / dataset（datalist 选项来自静态建议 + 已有结果） */}
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <span style={SECTION_TITLE_STYLE}>维度</span>
            <input
              list="ks-task-types"
              value={taskType}
              onChange={event => setTaskType(event.target.value)}
              onKeyDown={event => {
                if (event.key === "Enter") doSearch();
              }}
              placeholder="task_type（任务类型）"
              style={FILTER_INPUT_STYLE}
            />
            <datalist id="ks-task-types">
              {taskTypeOptions.map(value => (
                <option key={value} value={value} />
              ))}
            </datalist>
            <input
              list="ks-models"
              value={modelName}
              onChange={event => setModelName(event.target.value)}
              onKeyDown={event => {
                if (event.key === "Enter") doSearch();
              }}
              placeholder="model（模型名）"
              style={FILTER_INPUT_STYLE}
            />
            <datalist id="ks-models">
              {modelOptions.map(value => (
                <option key={value} value={value} />
              ))}
            </datalist>
            <input
              list="ks-datasets"
              value={datasetName}
              onChange={event => setDatasetName(event.target.value)}
              onKeyDown={event => {
                if (event.key === "Enter") doSearch();
              }}
              placeholder="dataset（数据集名）"
              style={FILTER_INPUT_STYLE}
            />
            <datalist id="ks-datasets">
              {datasetOptions.map(value => (
                <option key={value} value={value} />
              ))}
            </datalist>
            {(taskType || modelName || datasetName) && (
              <button
                onClick={() => {
                  setTaskType("");
                  setModelName("");
                  setDatasetName("");
                }}
                style={{ ...BUTTON_BASE, padding: "4px 10px", fontSize: 12, background: "#1f2937", color: "#cbd5e1" }}
              >
                清空维度
              </button>
            )}
          </div>

          {selectedTypes.length === 0 ? (
            <div style={{ color: "#fbbf24" }}>⚠ 未勾选任何类型：后端会按「不限类型」返回结果。</div>
          ) : null}
        </div>

        {/* 结果区 + 详情区 */}
        <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
          {/* 结果列表 */}
          <div style={{ width: 380, minWidth: 300, borderRight: "1px solid #262b36", overflowY: "auto" }}>
            <div
              style={{
                padding: "8px 12px",
                color: "#9ca3af",
                borderBottom: "1px solid #1b2027",
                position: "sticky",
                top: 0,
                background: "#0f1115",
                zIndex: 1,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span>结果</span>
                <span>
                  {loading ? "检索中…" : hasSearched ? `${handledCount} 条` : "—"}
                </span>
              </div>
              {/* 按类型分栏（tab）：全部 + 命中到的类型，各自带计数 */}
              {hits.length > 0 ? (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
                  <button
                    onClick={() => setActiveTab("all")}
                    style={{
                      ...BUTTON_BASE,
                      padding: "3px 9px",
                      fontSize: 11,
                      background: activeTab === "all" ? "#1d4ed8" : "#1f2937",
                      color: activeTab === "all" ? "#fff" : "#cbd5e1",
                      border: "1px solid #374151",
                    }}
                  >
                    全部 <span style={{ color: "#9ca3af" }}>{hits.length}</span>
                  </button>
                  {KNOWLEDGE_DATA_TYPES.filter(value => (typeCounts[value] || 0) > 0).map(value => (
                    <button
                      key={value}
                      onClick={() => setActiveTab(value)}
                      style={{
                        ...BUTTON_BASE,
                        padding: "3px 9px",
                        fontSize: 11,
                        background: activeTab === value ? typeColor(value) : "#1f2937",
                        color: activeTab === value ? "#fff" : typeColor(value),
                        border: "1px solid #374151",
                      }}
                    >
                      {DATA_TYPE_LABELS[value]} <span style={{ color: "#9ca3af" }}>{typeCounts[value]}</span>
                    </button>
                  ))}
                </div>
              ) : null}
            </div>

            {failure ? (
              <div style={{ padding: 12 }}>
                <FailureBox failure={failure} onRetry={doSearch} />
              </div>
            ) : null}

            {!failure && hasSearched && !loading && hits.length === 0 ? (
              <div style={{ padding: 16, color: "#9ca3af", lineHeight: 1.7 }}>
                <div style={{ fontWeight: 600, color: "#e6edf3", marginBottom: 4 }}>没有匹配的结果</div>
                <div>可以尝试：</div>
                <ul style={{ margin: "6px 0 0 18px", padding: 0 }}>
                  <li>换更短或更通用关键词（例如 csv、训练、图）</li>
                  <li>勾选更多类型，或清空关键词直接浏览</li>
                  <li>确认所选类型在库中确有数据（run/module 可能为空）</li>
                </ul>
              </div>
            ) : null}

            {!failure && !hasSearched && !loading ? (
              <div style={{ padding: 16, color: "#6b7280" }}>输入关键词后点击「检索」。</div>
            ) : null}

            {visibleHits.map((hit, index) => {
              const dataType = hit.data_type || "knowledge";
              const refId = hit.ref_id || hit.id || "";
              const key = `${dataType}:${refId}:${index}`;
              const isActive = activeKey === `${dataType}:${refId}`;
              const summary = truncateSummary(hit.summary);
              return (
                <button
                  key={key}
                  onClick={() => void openDetail(hit)}
                  style={{
                    display: "block",
                    width: "100%",
                    textAlign: "left",
                    padding: "10px 12px",
                    background: isActive ? "#161d2b" : "transparent",
                    border: "none",
                    borderBottom: "1px solid #1b2027",
                    borderLeft: isActive ? "3px solid #2563eb" : "3px solid transparent",
                    color: "inherit",
                    cursor: "pointer",
                    font: "inherit",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        color: "#fff",
                        background: typeColor(dataType),
                        borderRadius: 4,
                        padding: "1px 6px",
                        flexShrink: 0,
                      }}
                    >
                      {typeLabel(dataType)}
                    </span>
                    <span style={{ color: "#9ca3af", fontSize: 11 }}>{formatTimestamp(hit.created_at)}</span>
                  </div>
                  <div style={{ color: "#e6edf3", fontWeight: 600, lineHeight: 1.4 }}>
                    {hit.title || "(无标题)"}
                  </div>
                  <div style={{ color: "#8b949e", marginTop: 4, lineHeight: 1.5 }}>
                    {summary || "（无摘要）"}
                  </div>
                  {[hit.task_type, hit.model_name, hit.dataset_name].some(Boolean) ? (
                    <div style={{ color: "#7dd3fc", fontSize: 11, marginTop: 4 }}>
                      {[
                        hit.task_type ? `task=${hit.task_type}` : null,
                        hit.model_name ? `model=${hit.model_name}` : null,
                        hit.dataset_name ? `dataset=${hit.dataset_name}` : null,
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </div>
                  ) : null}
                  <div style={{ color: "#4b5563", fontSize: 11, marginTop: 4, wordBreak: "break-all" }}>
                    {dataType}/{refId}
                  </div>
                </button>
              );
            })}
          </div>

          {/* 详情 */}
          <div style={{ flex: 1, minWidth: 0, overflowY: "auto", padding: 16, display: "grid", gap: 12, alignContent: "start" }}>
            {detail.status === "idle" ? (
              <div style={{ color: "#6b7280", lineHeight: 1.7 }}>
                点击左侧任意结果查看详情。
                <div style={{ marginTop: 8 }}>
                  详情调用接口：
                  <code style={{ color: "#9ca3af" }}>
                    /api/knowledge/items/{"{data_type}"}/{"{ref_id}"}
                  </code>
                </div>
              </div>
            ) : null}

            {detail.status === "loading" ? <div style={{ color: "#9ca3af" }}>加载详情中…</div> : null}

            {detail.status === "error" ? <FailureBox failure={detail.failure} /> : null}

            {detail.status === "ready" ? (
              <>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, lineHeight: 1.4 }}>
                    {pickTitle(detail.payload, `${detail.dataType}/${detail.refId}`)}
                  </div>
                  <div style={{ color: "#9ca3af", marginTop: 4, wordBreak: "break-all" }}>
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        color: "#fff",
                        background: typeColor(detail.dataType),
                        borderRadius: 4,
                        padding: "1px 6px",
                        marginRight: 8,
                      }}
                    >
                      {typeLabel(detail.dataType)}
                    </span>
                    {detail.dataType}/{detail.refId}
                  </div>
                </div>

                {detailContent ? (
                  <div>
                    <div style={{ ...SECTION_TITLE_STYLE, marginBottom: 6 }}>正文</div>
                    <pre
                      style={{
                        margin: 0,
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        background: "#12151b",
                        border: "1px solid #262b36",
                        borderRadius: 8,
                        padding: 12,
                        lineHeight: 1.7,
                        fontFamily: "inherit",
                        maxHeight: 320,
                        overflowY: "auto",
                      }}
                    >
                      {detailContent}
                    </pre>
                  </div>
                ) : null}

                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                    <span style={SECTION_TITLE_STYLE}>结构化 JSON</span>
                    <button
                      onClick={() => setShowRaw(value => !value)}
                      style={{ ...BUTTON_BASE, padding: "2px 8px", fontSize: 11, background: "#1f2937", color: "#cbd5e1" }}
                    >
                      {showRaw ? "显示已展开字段" : "显示原始字段"}
                    </button>
                  </div>
                  <pre
                    style={{
                      margin: 0,
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      background: "#0b0e13",
                      border: "1px solid #262b36",
                      borderRadius: 8,
                      padding: 12,
                      lineHeight: 1.55,
                      fontSize: 12,
                      maxHeight: 520,
                      overflowY: "auto",
                    }}
                  >
                    {detailJson}
                  </pre>
                </div>
              </>
            ) : null}
          </div>
        </div>

        </>
        ) : (
        <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: 16, display: "grid", gap: 12, alignContent: "start" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontWeight: 600 }}>待确认蒸馏草稿</span>
            <span style={{ color: "#6b7280", fontSize: 12 }}>{drafts.length} 条</span>
            <div style={{ flex: 1 }} />
            <button onClick={() => void loadDrafts()} disabled={draftsLoading}
                    style={{ ...BUTTON_BASE, padding: "3px 10px", fontSize: 12, background: "#1f2937", color: "#cbd5e1" }}>
              {draftsLoading ? "刷新中…" : "刷新"}
            </button>
          </div>

          {draftsError ? <FailureBox failure={{ message: draftsError, hint: "请确认后端已启动且知识库接口可用。", url: "/api/knowledge/list" }} onRetry={() => void loadDrafts()} /> : null}

          {!draftsLoading && drafts.length === 0 && !draftsError ? (
            <div style={{ color: "#6b7280", lineHeight: 1.7 }}>
              暂无待确认草稿。任务结束后（如环境创建、训练、拆解、复现）系统会起草结论，
              在此确认后进入知识库并参与后续任务带入。
            </div>
          ) : null}

          {drafts.map(draft => {
            const conflicts = conflictsByDraft[draft.knowledge_id] || [];
            const busy = busyDraft === draft.knowledge_id;
            return (
              <div key={draft.knowledge_id}
                   style={{ border: "1px solid #262b36", background: "#12151b", borderRadius: 8, padding: 12 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: "#fff", background: "#7c3aed",
                                 borderRadius: 4, padding: "1px 6px" }}>{draft.type}</span>
                  <span style={{ fontWeight: 600 }}>{draft.title || "(无标题)"}</span>
                  {draft.confidence ? <span style={{ color: "#64748b", fontSize: 11 }}>{draft.confidence}</span> : null}
                  <span style={{ color: "#64748b", fontSize: 11, marginLeft: "auto" }}>{formatTimestamp(draft.created_at)}</span>
                </div>
                <div style={{ color: "#cbd5e1", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{draft.content}</div>
                {conflicts.length > 0 ? (
                  <div style={{ marginTop: 8, color: "#fbbf24", fontSize: 12 }}>
                    ⚠ 与 {conflicts.length} 条已确认知识冲突：{conflicts.map(c => c.title || c.knowledge_id).join("、")}
                  </div>
                ) : null}
                <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
                  <button onClick={() => void handleConfirmDraft(draft.knowledge_id, false)} disabled={busy}
                          style={{ ...BUTTON_BASE, padding: "4px 12px", fontSize: 12, background: "#1d4ed8", color: "#fff", border: "1px solid #2563eb" }}>
                    {busy ? "处理中…" : "确认入库"}
                  </button>
                  {conflicts.length > 0 ? (
                    <>
                      <button onClick={() => void handleConfirmDraft(draft.knowledge_id, true)} disabled={busy}
                              style={{ ...BUTTON_BASE, padding: "4px 12px", fontSize: 12, background: "#b45309", color: "#fff", border: "1px solid #d97706" }}>
                        确认并推翻旧结论
                      </button>
                      {conflicts.map(c => (
                        <button key={c.knowledge_id} onClick={() => void handleSupersedeKnowledge(c.knowledge_id)} disabled={busy}
                                style={{ ...BUTTON_BASE, padding: "4px 10px", fontSize: 12, background: "#1f2937", color: "#fca5a5" }}>
                          作废「{c.title || c.knowledge_id.slice(0, 8)}」
                        </button>
                      ))}
                    </>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
        )}
        {/* 底部调试信息 */}
        <div
          style={{
            padding: "6px 16px",
            borderTop: "1px solid #262b36",
            color: "#4b5563",
            fontSize: 11,
            wordBreak: "break-all",
          }}
        >
          最近一次检索：{lastUrl || "—"}
          {detail.status === "ready" ? `\u3000|\u3000详情：${buildItemUrl(baseUrl, detail.dataType, detail.refId)}` : ""}
        </div>
      </aside>
    </>
  );
}
