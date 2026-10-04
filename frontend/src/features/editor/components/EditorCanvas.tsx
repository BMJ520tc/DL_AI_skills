import { Background, ReactFlow, type Edge, type IsValidConnection, type Node, type OnConnect, type OnEdgesChange, type OnNodeDrag, type OnNodesChange, type OnSelectionChangeFunc, type ReactFlowInstance } from "@xyflow/react";
import { edgeTypes } from "../../../types/edgeTypes";
import { nodeTypes } from "../../../types/nodeTypes";

const fitViewOptions = { padding: 0.2 };
const defaultEdgeOptions = { animated: true };

type EditorCanvasProps = {
    nodesForFlow: Node[];
    highlightedEdges: Edge[];
    onNodesChange: OnNodesChange;
    onEdgesChange: OnEdgesChange;
    onConnect: OnConnect;
    onNodeDragStop: OnNodeDrag; // Type from useContainerSystem
    onNodeDragStart: OnNodeDrag;
    onMainDrop: (e: React.DragEvent) => void;
    onDragOver: (e: React.DragEvent) => void;
    onSelectionChange: OnSelectionChangeFunc;
    clearSelection: () => void;
    setMainFlowRef: (rf: ReactFlowInstance) => void;
    /** 可选：连线合法性（结构化画布传 IR 形状判据，形状不一致时 ReactFlow 直接拒绝落线）。 */
    isValidConnection?: IsValidConnection<Edge>;
}

export function EditorCanvas({
    nodesForFlow,
    highlightedEdges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    onNodeDragStop,
    onNodeDragStart,
    onMainDrop,
    onDragOver,
    onSelectionChange,
    clearSelection,
    setMainFlowRef,
    isValidConnection
}: EditorCanvasProps) {
    // React Flow 默认「选中即抬高」（elevateNodesOnSelect）：选中时把该节点 zIndex 改成 1000、
    // 取消选中又设回 0 —— 这会把 applyGraphIR 按层级深度算好的 zIndex 冲掉，于是「移动过某个
    // 节点、再点别处」时它会掉到容器之下（看起来被置于最底层）。故关掉它即可；**不要**再自己
    // 给「选中的节点」加 zIndex 抬升：React Flow 会把内部节点对象回吐给 onNodesChange，抬升
    // 会被写回源数据、每次交互叠一次（实测 40→50→55 的慢性漂移）。
    return (
        <div style={{ flex: 1, minWidth: 0, minHeight: 0, position: "relative", overflow: "hidden" }}>
            <div style={{ position: "absolute", inset: "0 0 0 0" }}>
                <ReactFlow
                    nodes={nodesForFlow}
                    elevateNodesOnSelect={false}
                    edges={highlightedEdges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeDragStop={onNodeDragStop}
                    onNodeDragStart={onNodeDragStart}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    fitView
                    fitViewOptions={fitViewOptions}
                    onDrop={onMainDrop}
                    onDragOver={onDragOver}
                    onSelectionChange={onSelectionChange}
                    onPaneClick={clearSelection}
                    isValidConnection={isValidConnection}
                    multiSelectionKeyCode="Shift"
                    selectionOnDrag
                    defaultEdgeOptions={defaultEdgeOptions}
                    onInit={setMainFlowRef}
                >
                    <Background />
                </ReactFlow>
            </div>
        </div>
    );
}
