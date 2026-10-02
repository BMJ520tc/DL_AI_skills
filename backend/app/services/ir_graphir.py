"""IR → 画布 GraphIR v2（结构化项目 graph.json；模块详细设计 6.2/6.4 与 7.1）。

与前端 utils/graphIR.ts::buildGraphIR 同构（version/句柄/display/data 字段对齐），
前端 applyGraphIR 可直接还原为 React Flow nodes/edges。所有节点 type="ir"（画布
IrNode 通用渲染，前端注册），层级经 parentId 表达（applyGraphIR 对 parentId 节点
强制 extent="parent"）。

布局为确定性分层算法：顶层节点按最长路径分层排 x、层内声明序排 y；子节点相对父
节点两列网格排布，父节点 data.layout_hint 给出建议尺寸（画布渲染器据此定框）。
"""
import json
from datetime import datetime, timezone

from app.services.ir_schema import children_of, in_edges, nodes_by_id, out_edges

GRAPH_VERSION = 2

# 布局参数：按**子树实际尺寸**自底向上排布（父容器被其子节点撑开），与前端 irAdapter 同构。
# 旧实现给所有父节点的子节点用固定网格（240×100）+ 固定父宽 500，兄弟容器互相重叠、
# 子节点溢出父框，故改为「格子尺寸 = 该层子节点里最大的子树尺寸」。
_LEAF_W = 220.0
_LEAF_H = 64.0
_PAD_X = 16.0
_PAD_TOP = 34.0  # 留出父节点标题行
_PAD_BOTTOM = 16.0
_GAP_X = 24.0
_GAP_Y = 20.0
_COLS = 2  # 仅当子节点都是叶子/操作时并排；含子容器时单列（避免相互遮挡）


def _shape_str(shape) -> str:
    """单个形状的文本（与前端 shapeStr 同构：`[1,3,32,32]`；缺项为「未知」）。"""
    return f"[{','.join(str(v) for v in shape)}]" if shape else "未知"


def _shape_io(node: dict) -> str:
    """每层输入输出形状文本（与前端 shapeIO 同构）：`[1,3,32,32] → [1,8,32,32]`。"""
    return f"{_shape_str(node.get('input_shape'))} → {_shape_str(node.get('output_shape'))}"


def _display(node: dict) -> dict:
    """display 与前端 formatDisplay 对齐：title=类名，params=参数 JSON 文本，shape=in → out 形状文本。"""
    params = node.get("params")
    return {
        "title": node.get("class_name") or node["id"],
        "params": json.dumps(params, ensure_ascii=False, separators=(",", ":")) if params else None,
        "shape": _shape_io(node),
    }


def _handle_ids(n_ins: int, n_outs: int) -> tuple[list[str], list[str]]:
    """单入/单出用裸 "in"/"out"，多入/多出用 "in0".."inN"（buildGraphIR 约定）。"""
    ins = ["in"] if n_ins <= 1 else [f"in{i}" for i in range(n_ins)]
    outs = ["out"] if n_outs <= 1 else [f"out{i}" for i in range(n_outs)]
    return ins, outs


def _handles(ins: list[str], outs: list[str]) -> list[dict]:
    hs = [{"id": i, "kind": "input", "order": k} for k, i in enumerate(ins)]
    hs += [{"id": o, "kind": "output", "order": k} for k, o in enumerate(outs)]
    return hs


def _cols_for(ir: dict, kids: list[dict]) -> int:
    """子节点并排数（与前端同构）：双列排布，单子节点单列。"""
    return 1 if len(kids) <= 1 else _COLS


def _grid_slots(ir: dict, kids: list[dict], sizes: dict[str, tuple[float, float]]):
    """把子节点排进双列网格：返回 (每个子节点的相对坐标, 网格宽, 网格高)。

    列宽 = 该列子节点的最大宽，行高 = 该行子节点的最大高 —— 因此格子互不相交，
    兄弟（含嵌套容器）不会叠在一起；父容器包围盒即网格 + 内边距。
    """
    cols = _cols_for(ir, kids)
    rows = (len(kids) + cols - 1) // cols
    col_w = [0.0] * cols
    row_h = [0.0] * rows
    for i, k in enumerate(kids):
        w, h = sizes[k["id"]]
        col_w[i % cols] = max(col_w[i % cols], w)
        row_h[i // cols] = max(row_h[i // cols], h)
    x_of = []
    acc = _PAD_X
    for c in range(cols):
        x_of.append(acc)
        acc += col_w[c] + _GAP_X
    y_of = []
    acc = _PAD_TOP
    for r in range(rows):
        y_of.append(acc)
        acc += row_h[r] + _GAP_Y
    coords = {k["id"]: {"x": x_of[i % cols], "y": y_of[i // cols]} for i, k in enumerate(kids)}
    grid_w = sum(col_w) + _GAP_X * (cols - 1)
    grid_h = sum(row_h) + _GAP_Y * (rows - 1)
    return coords, grid_w, grid_h


def _subtree_sizes(ir: dict) -> dict[str, tuple[float, float]]:
    """自底向上算每棵子树的包围盒：父容器 = 子节点排版结果 + 内边距（保证包含全部子节点）。"""
    sizes: dict[str, tuple[float, float]] = {}

    def visit(n: dict) -> tuple[float, float]:
        nid = n["id"]
        kids = children_of(ir, nid)
        if not kids:
            sizes[nid] = (_LEAF_W, _LEAF_H)
            return sizes[nid]
        kid_sizes = {k["id"]: visit(k) for k in kids}
        _coords, grid_w, grid_h = _grid_slots(ir, kids, kid_sizes)
        sizes[nid] = (2 * _PAD_X + grid_w, _PAD_TOP + grid_h + _PAD_BOTTOM)
        return sizes[nid]

    for node in ir["nodes"]:
        visit(node)
    return sizes


def _layout(ir: dict) -> tuple[dict[str, dict], dict[str, tuple[float, float]]]:
    """确定性布局：{node_id: {x, y}}（子节点坐标相对父节点原点）+ 各节点尺寸。

    同一 IR 恒产同一布局；父容器尺寸取自 _subtree_sizes，父子坐标自洽（子节点恒在父框内）。
    """
    sizes = _subtree_sizes(ir)
    positions: dict[str, dict] = {}

    def place(parent: dict) -> None:
        kids = children_of(ir, parent["id"])
        if not kids:
            return
        coords, _w, _h = _grid_slots(ir, kids, sizes)
        for k in kids:
            positions[k["id"]] = coords[k["id"]]
            place(k)

    node_map = nodes_by_id(ir)
    root = node_map.get(ir.get("root_id") or "")
    if root is None and ir["nodes"]:
        root = ir["nodes"][0]
    if root is not None:
        positions[root["id"]] = {"x": 0.0, "y": 0.0}
        place(root)

    # 兜底：与根无层级关系的游离节点，排在根右侧（多根/异常 IR 也不重叠）
    offset_y = 0.0
    root_w = sizes.get(root["id"], (_LEAF_W, _LEAF_H))[0] if root is not None else 0.0
    for n in ir["nodes"]:
        if n["id"] in positions:
            continue
        positions[n["id"]] = {"x": root_w + 80.0, "y": offset_y}
        offset_y += sizes.get(n["id"], (_LEAF_W, _LEAF_H))[1] + _GAP_Y
    return positions, sizes


def _edge_kind(ir: dict, e: dict) -> str:
    """边视觉类别：多输入目标的非首条入边、或同父跳过相邻兄弟的边 → skip（残差类）。"""
    node_map = nodes_by_id(ir)
    src, tgt = e["from"], e["to"]
    ins = in_edges(ir, tgt)
    if len(ins) > 1 and ins[0]["from"] != src:
        return "skip"
    parent = node_map[src].get("parent_id")
    if parent and parent == node_map[tgt].get("parent_id"):
        sibs = [c["id"] for c in children_of(ir, parent)]
        if src in sibs and tgt in sibs and sibs.index(tgt) != sibs.index(src) + 1:
            return "skip"
    return "data"


def _depth_of(ir: dict, node_id: str) -> int:
    d, cur = 0, nodes_by_id(ir).get(node_id)
    while cur and cur.get("parent_id"):
        d += 1
        cur = nodes_by_id(ir).get(cur["parent_id"])
    return d


def ir_to_graphir(ir: dict) -> dict:
    """IR → GraphIR v2 快照（父节点排在子节点前，避免画布加载时子节点脱离）。"""
    node_map = nodes_by_id(ir)
    positions, sizes = _layout(ir)
    n_ins = {nid: len(in_edges(ir, nid)) for nid in node_map}
    n_outs = {nid: len(out_edges(ir, nid)) for nid in node_map}

    graph_nodes = []
    for n in sorted(ir["nodes"], key=lambda n: _depth_of(ir, n["id"])):
        nid = n["id"]
        ins, outs = _handle_ids(n_ins[nid], n_outs[nid])
        data = {
            "kind": n.get("kind"),
            "class_name": n.get("class_name"),
            "module_path": n.get("module_path"),
            "params": n.get("params") or {},
            "ir_id": nid,
            "input_shape": n.get("input_shape"),
            "output_shape": n.get("output_shape"),
            # __shape 兼容既有画布节点；__in_shape/__out_shape 供查看器与数据流图展示 in → out
            "__shape": n.get("output_shape") or n.get("input_shape"),
            "__in_shape": n.get("input_shape"),
            "__out_shape": n.get("output_shape"),
        }
        if children_of(ir, nid):
            # 父容器尺寸 = 子树包围盒（子节点恒在框内），画布按此撑开容器
            w, h = sizes[nid]
            data["layout_hint"] = {"width": w, "height": h}
        graph_nodes.append({
            "id": nid,
            "type": "ir",
            "label": n.get("class_name") or nid,
            "display": _display(n),
            "handles": _handles(ins, outs),
            "position": positions[nid],
            "parentId": n.get("parent_id"),
            "extent": "parent" if n.get("parent_id") else None,
            "data": data,
        })

    src_count: dict[str, int] = {}
    tgt_count: dict[str, int] = {}
    graph_edges = []
    for i, e in enumerate(ir["edges"]):
        src, tgt = e["from"], e["to"]
        _, src_outs = _handle_ids(n_ins[src], n_outs[src])
        tgt_ins, _ = _handle_ids(n_ins[tgt], n_outs[tgt])
        si = src_count.get(src, 0)
        src_count[src] = si + 1
        ti = tgt_count.get(tgt, 0)
        tgt_count[tgt] = ti + 1
        data = {"tensor_shape": e["tensor_shape"]} if e.get("tensor_shape") else {}
        graph_edges.append({
            "id": f"e{i}",
            "source": src,
            "target": tgt,
            "sourceHandle": src_outs[min(si, len(src_outs) - 1)],
            "targetHandle": tgt_ins[min(ti, len(tgt_ins) - 1)],
            "kind": _edge_kind(ir, e),
            "data": data,
        })

    return {
        "version": GRAPH_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "nodes": graph_nodes,
        "edges": graph_edges,
    }
