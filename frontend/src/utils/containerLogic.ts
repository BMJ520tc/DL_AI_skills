import { type Edge, type Node } from "@xyflow/react";
import { useCallback, useEffect, useRef, type Dispatch, type SetStateAction } from "react";

export type ContainerConfig = {
    types: Set<string>;
    capacities: Record<string, number>;
};
// Add any new control flow nodes here.
// 注意：这里只收**容器**节点。`repeat_layer` 指控制流「重复块」容器
// （nodes/control_flow/RepeatLayer.tsx）；张量 torch.repeat 算子是另一个键
// `repeat_tensor`（nodes/pytorch_core/RepeatNode.tsx），**不是容器**，不要加进来。
export const DEFAULT_CONTAINER_CONFIG: ContainerConfig = {
    types: new Set(["repeat_layer", "module_list"]),
    capacities: {
        module_list: 1,
        repeat_layer: 999,
    },
};
export const getAbsolutePosition = (node: Node, nodes: Node[]): { x: number; y: number } => {
    if (!node.parentId) return node.position;

    const parent = nodes.find(p => p.id === node.parentId);
    if (!parent) return node.position; // Orphaned case safety

    const parentAbs = getAbsolutePosition(parent, nodes);
    return {
        x: parentAbs.x + node.position.x,
        y: parentAbs.y + node.position.y,
    };
};

// Returns true if 'possibleChild' is actually a parent/grandparent of 'target'
// Used to prevent a parent being dropped into its own child.
const isAncestor = (ancestorId: string, targetNode: Node, nodes: Node[]): boolean => {
    if (!targetNode.parentId) return false;
    if (targetNode.parentId === ancestorId) return true;

    const parent = nodes.find(n => n.id === targetNode.parentId);
    if (!parent) return false;

    return isAncestor(ancestorId, parent, nodes);
};

// 3. Topology Signature (For preventing infinite loops in useEffect)
// Creates a hash of the current graph structure to check if React state needs updates.
const getTopologySig = (nodeList: Node[], edgeList: Edge[]) => {
    const nodeSig = nodeList
        .map(n => n.id)
        .sort()
        .join("|");
    const edgeSig = edgeList
        .map(e => `${e.id}:${e.source}:${e.target}`)
        .sort()
        .join("|");
    return `${nodeSig}::${edgeSig}`;
};

const getDataSig = (nodeList: Node[]) => {
    return nodeList
        .sort((a, b) => a.id.localeCompare(b.id))
        .map(n => {
            const d = n.data || {};
            // Exclude large objects or UI flags to prevent recursions
            const stableData: Record<string, unknown> = { ...d };
            delete stableData.internalNodes;
            delete stableData.internalEdges;
            delete stableData.__highlight;
            const stableString = JSON.stringify(stableData, Object.keys(stableData).sort());
            return `${n.id}.${stableString}`;
        })
        .join("||");
};
export function findBestParent(node: Node, currentNodes: Node[], config: ContainerConfig = DEFAULT_CONTAINER_CONFIG) {
    // A. Calculate the Dragged Node's Absolute Geometry
    const nodeAbsPos = getAbsolutePosition(node, currentNodes);
    const nodeWidth = node.measured?.width ?? node.width ?? 0;
    const nodeHeight = node.measured?.height ?? node.height ?? 0;

    // Use center point for hit testing (feels more intuitive than top-left)
    const nodeCenter = {
        x: nodeAbsPos.x + nodeWidth / 2,
        y: nodeAbsPos.y + nodeHeight / 2,
    };

    // B. Find All Potential Parents
    const candidates = currentNodes.filter(potentialParent => {
        // 1. Must be a Container
        if (!config.types.has(potentialParent.type || "")) return false;

        // 2. Cannot be itself
        if (potentialParent.id === node.id) return false;

        // 3. Cannot be a child of the dragged node (Cycle Prevention)
        // If I am dragging "Outer Loop", I cannot drop it into "Inner Loop".
        if (isAncestor(node.id, potentialParent, currentNodes)) return false;

        // 4. Hit Test (Absolute Coordinates)
        const pWidth = potentialParent.measured?.width ?? potentialParent.width ?? 0;
        const pHeight = potentialParent.measured?.height ?? potentialParent.height ?? 0;
        const parentAbs = getAbsolutePosition(potentialParent, currentNodes);

        return (
            nodeCenter.x >= parentAbs.x &&
            nodeCenter.x <= parentAbs.x + pWidth &&
            nodeCenter.y >= parentAbs.y &&
            nodeCenter.y <= parentAbs.y + pHeight
        );
    });
    if (candidates.length === 0) return undefined;

    return candidates.sort((a, b) => {
        const areaA = (a.measured?.width ?? 0) * (a.measured?.height ?? 0);
        const areaB = (b.measured?.width ?? 0) * (b.measured?.height ?? 0);
        return areaA - areaB; // Ascending sort: Smallest area first
    })[0];
}
/** 与 `graphIR.applyGraphIR` 同口径的层级 zIndex：根=10，每深一层 +10（子节点恒在祖先之上）。
 *  新建/拖入/改父的节点必须一并带上——否则 React Flow 按 0 算，新节点会压在所有已载入
 *  节点（z≥10）之下，看起来就是「刚拖进来就在最底层」。 */
function depthZIndex(parentId: string | undefined, all: Node[]): number {
    const byId = new Map(all.map(n => [n.id, n]));
    let depth = 0;
    let cur = parentId;
    const seen = new Set<string>();
    while (cur && !seen.has(cur)) {
        seen.add(cur);
        const parent = byId.get(cur);
        if (!parent) break;
        depth += 1;
        cur = parent.parentId;
    }
    return 10 + depth * 10;
}

export function assignParent(
    node: Node,
    currentNodes: Node[],
    config: ContainerConfig = DEFAULT_CONTAINER_CONFIG,
): Node {
    const targetParent = findBestParent(node, currentNodes);

    if (targetParent) {
        // 1. Check Capacity for new drops
        const maxCap = config.capacities[targetParent.type || ""] || 999;
        const isAlreadyChild = node.parentId === targetParent.id;
        const existingChildren = currentNodes.filter(n => n.parentId === targetParent.id);

        if (!isAlreadyChild && existingChildren.length >= maxCap) {
            // Capacity full: Drop as orphan on canvas instead
            const nodeAbs = node.parentId ? getAbsolutePosition(node, currentNodes) : node.position;
            return {
                ...node,
                parentId: undefined,
                extent: undefined,
                position: nodeAbs,
                zIndex: depthZIndex(undefined, currentNodes),
                // Position is already absolute for new drops
            };
        }

        // 2. Adopt Child
        const parentAbs = getAbsolutePosition(targetParent, currentNodes);
        // Ensure input node position is treated as absolute
        const nodeAbs = node.parentId ? getAbsolutePosition(node, currentNodes) : node.position;

        return {
            ...node,
            parentId: targetParent.id,
            extent: "parent",
            position: {
                x: nodeAbs.x - parentAbs.x,
                y: nodeAbs.y - parentAbs.y,
            },
            zIndex: depthZIndex(targetParent.id, currentNodes),
        };
    }
    const nodeAbs = node.parentId ? getAbsolutePosition(node, currentNodes) : node.position;
    // 拖出容器的节点回到根层级，zIndex 同步回到根档（否则会带着较深的值浮在其他根节点之上）
    return {
        ...node,
        parentId: undefined,
        extent: undefined,
        position: nodeAbs,
        zIndex: depthZIndex(undefined, currentNodes),
    };
}
export function syncContainerData(
    nodes: Node[],
    edges: Edge[],
    config: ContainerConfig = DEFAULT_CONTAINER_CONFIG,
): { hasChanges: boolean; nodes: Node[] } {
    const containerNodes = nodes.filter(n => config.types.has(n.type || ""));
    if (containerNodes.length === 0) return { hasChanges: false, nodes };

    let hasChanges = false;
    const nextNodes = nodes.map(node => {
        if (!config.types.has(node.type || "")) return node;

        const children = nodes.filter(child => child.parentId === node.id);
        const childIds = new Set(children.map(c => c.id));

        const internalEdges = edges.filter(e => {
            const isSourceChild = childIds.has(e.source);
            const isTargetChild = childIds.has(e.target);
            const isSourceParent = e.source === node.id;
            const isTargetParent = e.target === node.id;

            if (isSourceChild && isTargetChild) return true;
            if (isSourceParent && isTargetChild) return true;
            if (isSourceChild && isTargetParent) return true;
            return false;
        });

        const graphTopologySig = getTopologySig(children, internalEdges);
        const graphDataSig = getDataSig(children);

        const currentData = (node.data || {}) as { internalNodes?: Node[]; internalEdges?: Edge[] };
        const storedTopologySig = getTopologySig(currentData.internalNodes || [], currentData.internalEdges || []);
        const storedDataSig = getDataSig(currentData.internalNodes || []);

        if (graphTopologySig === storedTopologySig && graphDataSig === storedDataSig) {
            return node;
        }

        hasChanges = true;
        return {
            ...node,
            data: {
                ...currentData,
                internalNodes: children,
                internalEdges: internalEdges,
            },
        };
    });

    return { hasChanges, nodes: nextNodes };
}
export function useContainerSystem(
    nodes: Node[],
    edges: Edge[],
    setNodes: Dispatch<SetStateAction<Node[]>>,
    getNodes: () => Node[],
    config: ContainerConfig = DEFAULT_CONTAINER_CONFIG,
) {
    const dragStartRef = useRef<Node | null>(null);

    // 1. CAPTURE START STATE
    const onNodeDragStart = useCallback((_event: React.MouseEvent, node: Node) => {
        dragStartRef.current = JSON.parse(JSON.stringify(node));
    }, []);

    // 2. DRAG STOP
    const onNodeDragStop = useCallback(
        (_event: React.MouseEvent, node: Node) => {
            const currentNodes = getNodes(); // Helper from ReactFlow to get latest state

            // Check Capacity before assigning
            const targetParent = findBestParent(node, currentNodes);
            if (targetParent) {
                const isAlreadyChild = node.parentId === targetParent.id;
                const maxCap = config.capacities[targetParent.type || ""] || 999;
                const existingChildren = currentNodes.filter(n => n.parentId === targetParent.id && n.id !== node.id);
                if (!isAlreadyChild && existingChildren.length >= maxCap) {
                    // Revert Logic
                    if (dragStartRef.current && dragStartRef.current.id === node.id) {
                        setNodes(nds => nds.map(n => (n.id === node.id ? dragStartRef.current! : n)));
                        dragStartRef.current = null;
                        return;
                    }
                }
            }

            // Apply Parenting
            const processedNode = assignParent(node, currentNodes, config);

            setNodes(nds => nds.map(n => (n.id === node.id ? processedNode : n)));
            dragStartRef.current = null;
        },
        [getNodes, setNodes, config],
    );

    // 3. SYNC DATA
    useEffect(() => {
        const result = syncContainerData(nodes, edges);
        if (result.hasChanges) {
            setNodes(result.nodes);
        }
    }, [nodes, edges, setNodes]);

    return { onNodeDragStop, onNodeDragStart, assignParent };
}
