import { Background, ReactFlow, type Edge, type Node, type OnConnect, type OnEdgesChange, type OnNodeDrag, type OnNodesChange, type OnSelectionChangeFunc, type ReactFlowInstance } from "@xyflow/react";
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
    setMainFlowRef
}: EditorCanvasProps) {
    return (
        <div style={{ flex: 1, minWidth: 0, minHeight: 0, position: "relative", overflow: "hidden" }}>
            <div style={{ position: "absolute", inset: "0 0 0 0" }}>
                <ReactFlow
                    nodes={nodesForFlow}
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
