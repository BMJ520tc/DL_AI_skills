import type { SavedModule } from "../../../utils/moduleRegistry";
import { useSidebarSystem } from "../hooks/useSidebarSystem";
import logo2 from "../assets/sdslabs.png"; // Adjust the path to your file
import logo1 from "../assets/dsg.png";

// --- Modern Dark Theme Palette ---
const THEME = {
    bg: "#18181bff",           // Zinc-900
    bgSection: "#27272a",    // Zinc-800
    border: "#64646dff",       // Zinc-700
    textPrimary: "#fdfdfdde",  // Zinc-200
    textSecondary: "#ccccd3ff",// Zinc-400
    accent: "#0ea5e9",       // Sky-500
    accentHover: "#0284c7",  // Sky-600
    danger: "#ef4444",       // Red-500
    dangerBg: "rgba(239, 68, 68, 0.1)",
    hover: "#3f3f46",        // Zinc-700
    itemBg: "#27272a",       // Zinc-800
};

// --- Icons ---
const ChevronIcon = ({ open }: { open: boolean }) => (
    <svg
        width="16" height="16" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
        style={{
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.2s ease",
            opacity: 0.7
        }}
    >
        <polyline points="9 18 15 12 9 6"></polyline>
    </svg>
);

const SearchIcon = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.5 }}>
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
    </svg>
);

// New Grip Handle Icon (::)
const DragHandleIcon = () => (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style={{ opacity: 0.3, cursor: "grab" }}>
        <circle cx="9" cy="5" r="2" />
        <circle cx="9" cy="12" r="2" />
        <circle cx="9" cy="19" r="2" />
        <circle cx="15" cy="5" r="2" />
        <circle cx="15" cy="12" r="2" />
        <circle cx="15" cy="19" r="2" />
    </svg>
);

type EditorSidebarProps = {
    sidebarCollapsed: boolean;
    sidebarWidth: number;
    dragSidebar: boolean;
    setSidebarCollapsed: (v: boolean) => void;
    setDragSidebar: (v: boolean) => void;
    onGenerateCode: () => void;
    showLiveCode: boolean;
    modules: SavedModule[];
    handleDeleteModule: (id: string) => void;
    /** 可选：为 true 时隐藏「显示实时代码」入口（外部画布模式）。 */
    hideGenerateCode?: boolean;
};

export function EditorSidebar({
    sidebarCollapsed,
    sidebarWidth,
    dragSidebar,
    setSidebarCollapsed,
    setDragSidebar,
    onGenerateCode,
    showLiveCode,
    modules,
    handleDeleteModule,
    hideGenerateCode = false,
}: EditorSidebarProps) {
    const {
        searchQuery,
        setSearchQuery,
        openGroups,
        toggleGroup,
        filteredModules,
        filteredGroups,
        onDragStart,
        handleReset,
        openModuleEditor,
        normalizedQuery
    } = useSidebarSystem(modules);

    const renderHeader = () => (
        <div style={{ padding: "16px", borderBottom: `1px solid ${THEME.border}` }}>
            <div style={{
                display: "flex",
                alignItems: "center",
                background: THEME.bgSection,
                border: `1px solid ${THEME.border}`,
                borderRadius: "6px",
                padding: "0 12px",
                gap: "8px",
                height: "36px"
            }}>
                <SearchIcon />
                <input
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    placeholder="搜索节点..."
                    style={{
                        width: "100%",
                        background: "transparent",
                        border: "none",
                        color: THEME.textPrimary,
                        padding: "8px 0",
                        fontSize: "14px",
                        outline: "none",
                    }}
                />
            </div>
        </div>
    );

    const renderFooter = () => (
        <div style={{ padding: "16px", borderTop: `1px solid ${THEME.border}`, background: THEME.bg, display: "flex", flexDirection: "column", gap: "12px" }}>
            {!hideGenerateCode && (
                <button
                    onClick={onGenerateCode}
                    style={{
                        width: "100%",
                        padding: "12px",
                        background: showLiveCode ? "rgba(14, 165, 233, 0.15)" : THEME.itemBg,
                        color: showLiveCode ? THEME.accent : THEME.textPrimary,
                        border: `1px solid ${showLiveCode ? THEME.accent : THEME.border}`,
                        borderRadius: "6px",
                        cursor: "pointer",
                        fontSize: "13px",
                        fontWeight: 600,
                        transition: "all 0.2s"
                    }}
                >
                    {showLiveCode ? "隐藏实时代码" : "显示实时代码"}
                </button>
            )}
            <div style={{ display: "flex", gap: "8px" }}>
                <button
                    onClick={handleReset}
                    style={{
                        flex: 1,
                        padding: "8px",
                        background: "transparent",
                        color: THEME.danger,
                        border: `1px solid ${THEME.border}`,
                        borderRadius: "6px",
                        cursor: "pointer",
                        fontSize: "12px",
                        fontWeight: 500
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = THEME.dangerBg; e.currentTarget.style.borderColor = THEME.danger; }}
                    onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.borderColor = THEME.border; }}
                >
                    重置
                </button>
                <button
                    onClick={() => setSidebarCollapsed(true)}
                    style={{
                        flex: 1,
                        padding: "8px",
                        background: "transparent",
                        color: THEME.textSecondary,
                        border: `1px solid ${THEME.border}`,
                        borderRadius: "6px",
                        cursor: "pointer",
                        fontSize: "12px",
                        fontWeight: 500
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = THEME.hover; e.currentTarget.style.color = THEME.textPrimary; }}
                    onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = THEME.textSecondary; }}
                >
                    折叠
                </button>
            </div>
        </div>
    );

    return (
        <>
            <div
                style={{
                    width: sidebarCollapsed ? 48 : sidebarWidth,
                    flexShrink: 0,
                    height: "100%",
                    background: THEME.bg,
                    borderRight: `1px solid ${THEME.border}`,
                    display: "flex",
                    flexDirection: "column",
                    position: "relative",
                    transition: dragSidebar ? "none" : "width 0.2s ease",
                    overflow: "hidden",
                    userSelect: "none"
                }}
            >
                {sidebarCollapsed ? (
                    <button
                        onClick={() => setSidebarCollapsed(false)}
                        style={{
                            width: "100%",
                            height: "100%",
                            background: "transparent",
                            border: "none",
                            cursor: "pointer",
                            color: THEME.textSecondary,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                        }}
                        title="展开侧边栏"
                    >
                        <div style={{ writingMode: "vertical-rl", textOrientation: "mixed", fontSize: "12px", fontWeight: 600, letterSpacing: "1px", opacity: 0.7 }}>
                            节点
                        </div>
                    </button>
                ) : (
                    <>
                        {renderHeader()}

                        {/* --- Scrollable Content --- */}
                        <div style={{ flex: 1, overflowY: "auto", padding: "0 16px 16px" }}>

                            {/* 1. 自定义模块 Section */}
                            {modules.length > 0 && (
                                <div style={{ marginTop: "20px" }}>
                                    <div
                                        onClick={() => toggleGroup("custom_modules")}
                                        style={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: "8px",
                                            cursor: "pointer",
                                            color: THEME.textSecondary,
                                            fontSize: "12px",
                                            fontWeight: 700,
                                            textTransform: "uppercase",
                                            letterSpacing: "0.5px",
                                            marginBottom: "10px",
                                        }}
                                        onMouseEnter={e => e.currentTarget.style.color = THEME.textPrimary}
                                        onMouseLeave={e => e.currentTarget.style.color = THEME.textSecondary}
                                    >
                                        <ChevronIcon open={!!(normalizedQuery || openGroups["custom_modules"])} />
                                        自定义模块
                                        <span style={{ marginLeft: "auto", background: THEME.hover, color: THEME.textPrimary, padding: "2px 8px", borderRadius: "10px", fontSize: "11px" }}>
                                            {filteredModules.length}
                                        </span>
                                    </div>

                                    {(normalizedQuery || openGroups["custom_modules"]) && (
                                        <div style={{ display: "flex", flexDirection: "column", gap: "8px", paddingLeft: "8px" }}>
                                            {filteredModules.map(mod => (
                                                <div
                                                    key={mod.id}
                                                    draggable
                                                    onDragStart={e => onDragStart(e, "module_ref", { moduleId: mod.id })}
                                                    style={{
                                                        background: THEME.itemBg,
                                                        border: `1px solid ${THEME.border}`,
                                                        borderRadius: "6px",
                                                        padding: "12px",
                                                        cursor: "grab",
                                                        position: "relative",
                                                        transition: "border-color 0.2s, transform 0.1s"
                                                    }}
                                                    onMouseEnter={e => { e.currentTarget.style.borderColor = THEME.accent; e.currentTarget.style.transform = "translateY(-1px)"; }}
                                                    onMouseLeave={e => { e.currentTarget.style.borderColor = THEME.border; e.currentTarget.style.transform = "none"; }}
                                                >
                                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "4px" }}>
                                                        <span style={{ color: THEME.textPrimary, fontSize: "14px", fontWeight: 600 }}>
                                                            {mod.name}
                                                            {mod.origin === "backend" && (
                                                                <span
                                                                    title="来自后端标准化模块库（只读，不写入本地）"
                                                                    style={{
                                                                        marginLeft: 6,
                                                                        fontSize: 10,
                                                                        fontWeight: 600,
                                                                        color: "#7dd3fc",
                                                                        border: "1px solid #0ea5e9",
                                                                        borderRadius: 4,
                                                                        padding: "0 4px",
                                                                        verticalAlign: "middle",
                                                                    }}
                                                                >
                                                                    后端模块库
                                                                </span>
                                                            )}
                                                        </span>
                                                        <span style={{ fontSize: "11px", color: THEME.textSecondary, fontFamily: "monospace" }}>{mod.version}</span>
                                                    </div>
                                                    <div style={{ fontSize: "12px", color: THEME.textSecondary, marginBottom: "12px" }}>
                                                        {mod.handles.inputs.length} Inputs • {mod.handles.outputs.length} Outputs
                                                    </div>
                                                    <div style={{ display: "flex", gap: "8px" }}>
                                                        {mod.origin === "backend" ? (
                                                            // 后端模块只读：不可编辑（否则保存会把后端模块写进 localStorage 形成影子覆盖）
                                                            <span
                                                                style={{
                                                                    flex: 1, padding: "6px", fontSize: "11px", textAlign: "center",
                                                                    color: THEME.textSecondary, border: `1px dashed ${THEME.border}`,
                                                                    borderRadius: "4px"
                                                                }}
                                                            >
                                                                只读（后端模块库）
                                                            </span>
                                                        ) : (
                                                            <>
                                                                <button
                                                                    onClick={(e) => { e.stopPropagation(); openModuleEditor(mod.id); }}
                                                                    style={{
                                                                        flex: 1, padding: "6px", fontSize: "11px",
                                                                        background: THEME.hover, border: "none",
                                                                        color: THEME.textPrimary, borderRadius: "4px",
                                                                        cursor: "pointer", fontWeight: 500
                                                                    }}
                                                                >
                                                                    编辑
                                                                </button>
                                                                <button
                                                                    onClick={(e) => { e.stopPropagation(); handleDeleteModule(mod.id); }}
                                                                    style={{
                                                                        flex: 1, padding: "6px", fontSize: "11px",
                                                                        background: "transparent", border: `1px solid ${THEME.border}`,
                                                                        color: THEME.textSecondary, borderRadius: "4px",
                                                                        cursor: "pointer"
                                                                    }}
                                                                    onMouseEnter={e => { e.currentTarget.style.color = THEME.danger; e.currentTarget.style.borderColor = THEME.danger }}
                                                                    onMouseLeave={e => { e.currentTarget.style.color = THEME.textSecondary; e.currentTarget.style.borderColor = THEME.border }}
                                                                >
                                                                    删除
                                                                </button>
                                                            </>
                                                        )}
                                                    </div>
                                                    {/* Custom modules also get a subtle grip handle on hover if needed, or rely on the whole card */}
                                                </div>
                                            ))}
                                            {normalizedQuery && filteredModules.length === 0 && <div style={{ fontSize: "13px", color: THEME.textSecondary, fontStyle: "italic", padding: "4px" }}>没有匹配搜索的自定义模块。</div>}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* 2. Standard Node Groups */}
                            {filteredGroups.map(({ key, label, nodes }) => {
                                const isOpen = normalizedQuery ? true : !!openGroups[key];
                                return (
                                    <div key={key} style={{ marginTop: "20px" }}>
                                        <div
                                            onClick={() => toggleGroup(key)}
                                            style={{
                                                display: "flex",
                                                alignItems: "center",
                                                gap: "8px",
                                                cursor: "pointer",
                                                color: THEME.textSecondary,
                                                fontSize: "12px",
                                                fontWeight: 700,
                                                textTransform: "uppercase",
                                                letterSpacing: "0.5px",
                                                marginBottom: "8px",
                                            }}
                                            onMouseEnter={e => e.currentTarget.style.color = THEME.textPrimary}
                                            onMouseLeave={e => e.currentTarget.style.color = THEME.textSecondary}
                                        >
                                            <ChevronIcon open={isOpen} />
                                            {label}
                                        </div>

                                        {isOpen && (
                                            <div style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                gap: "6px",
                                                paddingLeft: "10px",
                                                borderLeft: `1px solid ${THEME.border}`,
                                                marginLeft: "6px"
                                            }}>
                                                {nodes.map(node => (
                                                    <div
                                                        key={node.type}
                                                        draggable
                                                        onDragStart={e => onDragStart(e, node.type)}
                                                        style={{
                                                            padding: "10px 12px",
                                                            fontSize: "14px",
                                                            color: THEME.textPrimary,
                                                            background: THEME.itemBg, // Card background
                                                            border: `1px solid ${THEME.border}`, // Distinct border
                                                            borderRadius: "6px",
                                                            cursor: "grab",
                                                            transition: "all 0.15s ease",
                                                            display: "flex",
                                                            alignItems: "center",
                                                            justifyContent: "space-between",
                                                            boxShadow: "0 1px 2px rgba(0,0,0,0.2)"
                                                        }}
                                                        onMouseEnter={e => {
                                                            e.currentTarget.style.background = THEME.hover;
                                                            e.currentTarget.style.borderColor = THEME.textSecondary;
                                                            e.currentTarget.style.transform = "translateX(2px)";
                                                        }}
                                                        onMouseLeave={e => {
                                                            e.currentTarget.style.background = THEME.itemBg;
                                                            e.currentTarget.style.borderColor = THEME.border;
                                                            e.currentTarget.style.transform = "translateX(0)";
                                                        }}
                                                    >
                                                        <span>{node.label}</span>
                                                        <DragHandleIcon />
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}

                            {normalizedQuery && filteredGroups.length === 0 && (
                                <div style={{ textAlign: "center", padding: "30px 10px", color: THEME.textSecondary, fontSize: "14px" }}>
                                    没有匹配的节点 "{searchQuery}"
                                </div>
                            )}
                        </div>

                                <div style={{ 
                                    display: "flex", 
                                    justifyContent: "center", 
                                    alignItems: "center", 
                                    gap: "20px",
                                    marginTop: "40px",
                                    paddingBottom: "10px" 
                                }}>
                                    <img src={logo1} alt="Logo 1" style={{ height: "28px", opacity: 0.5 }} />
                                    <img src={logo2} alt="Logo 2" style={{ height: "28px", opacity: 0.5 }} />
                                </div>

                        {renderFooter()}
                    </>
                )}
            </div>

            {/* Resize Handle */}
            <div
                onMouseDown={() => setDragSidebar(true)}
                style={{
                    width: "4px",
                    cursor: "col-resize",
                    background: dragSidebar ? THEME.accent : "transparent",
                    transition: "background 0.2s",
                    zIndex: 10
                }}
                onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.1)")}
                onMouseLeave={e => !dragSidebar && (e.currentTarget.style.background = "transparent")}
            />
        </>
    );
}
