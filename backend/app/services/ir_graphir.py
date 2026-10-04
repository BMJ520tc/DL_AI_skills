"""IR ↔ 画布 GraphIR v2（结构化项目 graph.json；模块详细设计 6.2/6.4 与 7.1）。

正向 `ir_to_graphir`：与前端 utils/graphIR.ts::buildGraphIR 同构（version/句柄/display/
data 字段对齐），前端 applyGraphIR 可直接还原为 React Flow nodes/edges。所有节点
type="ir"（画布 IrNode 通用渲染，前端注册），层级经 parentId 表达（applyGraphIR 对
parentId 节点强制 extent="parent"）。

布局为确定性分层算法：顶层节点按最长路径分层排 x、层内声明序排 y；子节点相对父
节点两列网格排布，父节点 data.layout_hint 给出建议尺寸（画布渲染器据此定框）。

反向 `graphir_to_ir`（模块详细设计 6.3/7.5：画布上的 ir 图导出/训练）：把画布快照
还原成模块四 IR，再交 `ir_codegen.generate` 出码——**画布上的调参、层级与连线都体现
在结果里**。两向映射同文件便于对照；还原后一律过 `ir_schema.validate_ir`，不合法即
抛 `GraphIRMappingError`（消息定位到具体节点/字段），绝不静默产出坏 IR。
"""
import json
import re
from datetime import datetime, timezone
from typing import Optional

from app.services.ir_schema import (
    KINDS, SCHEMA_VERSION, TASK_TYPES, children_of, in_edges, nodes_by_id, out_edges, validate_ir,
)

GRAPH_VERSION = 2

# 布局参数：按**子树实际尺寸**自底向上排布（父容器被其子节点撑开），与前端 irAdapter 同构。
# 旧实现给所有父节点的子节点用固定网格（240×100）+ 固定父宽 500，兄弟容器互相重叠、
# 子节点溢出父框，故改为「格子尺寸 = 该层子节点里最大的子树尺寸」。
_LEAF_W = 220.0
_LEAF_MIN_H = 56.0
_PAD_X = 16.0
_PAD_TOP = 34.0  # 留出父节点标题行
_PAD_BOTTOM = 16.0
_GAP_X = 24.0
_GAP_Y = 20.0
_COLS = 2  # 仅当子节点都是叶子/操作时并排；含子容器时单列（避免相互遮挡）

# 节点自身内容高度（画布 IrNode 渲染：头部 + 形状 + 参数单列）——
# 参数行数必须计入预留高度，否则节点实际渲染高于预留值，会与下方兄弟节点重叠（界面表现为「内容挤在一起」）。
# 数值由无头浏览器实测校准（与前端 irAdapter 同构）：
#   无参数节点 ≈ 28px（内边距 + 头部 + 形状行）；带参数节点固定部分 ≈ 78px，每行参数 ≈ 21.5px（留余量取 22）。
_NODE_BASE_H = 32.0        # 无参数时所需高度
_NODE_PARAM_BASE_H = 78.0  # 有参数时的固定部分（内边距 + 头部 + 形状行 + 参数区上边距）
_PARAM_ROW_H = 22.0        # 每个参数行（标签 + 输入框 + 删除）
_PARAM_MAX_ROWS = 8        # 超过该行数时参数区内部滚动（高度不再增长）


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


def _own_content_h(node: dict) -> float:
    """节点**自身内容**所需高度（内边距 + 头部 + 形状 + 参数单列），与前端 IrNode 实测口径一致。"""
    rows = len(node.get("params") or {})
    if rows == 0:
        return _NODE_BASE_H
    return _NODE_PARAM_BASE_H + min(rows, _PARAM_MAX_ROWS) * _PARAM_ROW_H


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
        own_h = _own_content_h(n)
        if not kids:
            sizes[nid] = (_LEAF_W, max(_LEAF_MIN_H, own_h))
            return sizes[nid]
        kid_sizes = {k["id"]: visit(k) for k in kids}
        _coords, grid_w, grid_h = _grid_slots(ir, kids, kid_sizes)
        sizes[nid] = (2 * _PAD_X + grid_w, max(own_h, _PAD_TOP + grid_h + _PAD_BOTTOM))
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
    root_id = _root_id_of(ir)
    root = node_map.get(root_id) if root_id else None
    if root is not None:
        positions[root["id"]] = {"x": 0.0, "y": 0.0}
        place(root)

    # 兜底：与根无层级关系的游离节点，排在根右侧（多根/异常 IR 也不重叠）
    offset_y = 0.0
    root_w = sizes.get(root["id"], (_LEAF_W, _LEAF_MIN_H))[0] if root is not None else 0.0
    for n in ir["nodes"]:
        if n["id"] in positions:
            continue
        positions[n["id"]] = {"x": root_w + 80.0, "y": offset_y}
        offset_y += sizes.get(n["id"], (_LEAF_W, _LEAF_MIN_H))[1] + _GAP_Y
    return positions, sizes


def _root_id_of(ir: dict) -> Optional[str]:
    """IR 的入口节点 id：root_id 合法则用它，否则退回首个节点（与 _layout 老兜底同口径）。"""
    node_map = nodes_by_id(ir)
    rid = ir.get("root_id")
    if isinstance(rid, str) and rid in node_map:
        return rid
    nodes = ir.get("nodes") or []
    return nodes[0]["id"] if nodes else None


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


def _ir_meta(ir: dict, root_id: Optional[str]) -> dict:
    """供反向映射复原的 IR 级元数据（画布本身没有这些字段，只能随图携带）。

    只挂在根节点 data 上；前端 mergeGraphIR/参数面板都以服务端 data 为基底，
    多余键随 data 原样往返（IrNode 只读已知字段），故不会因保存画布而丢失。
    """
    return {
        "schema_version": ir.get("schema_version") or SCHEMA_VERSION,
        "source_file": ir.get("source_file") or "",
        "entry_class": ir.get("entry_class") or "",
        "task_type": ir.get("task_type"),
        "input_spec": ir.get("input_spec") or {},
        "root_id": root_id,
    }


def ir_to_graphir(ir: dict) -> dict:
    """IR → GraphIR v2 快照（父节点排在子节点前，避免画布加载时子节点脱离）。"""
    node_map = nodes_by_id(ir)
    root_id = _root_id_of(ir)
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
            # 反向映射（graphir_to_ir）需要、但画布 UI 不展示的字段：code_hint 是
            # 「白名单外叶子/算子的构造表达式」，丢了就无法再生成，故随 data 携带。
            "code_hint": n.get("code_hint"),
            "uncertain": n.get("uncertain"),
        }
        if nid == root_id:
            # 根节点标记 + IR 级元数据：反向映射据此还原 root_id/entry_class/… ；
            # 缺失时（老 graph.json / 前端全量重建）退回「唯一顶层节点」等兜底推导。
            data["ir_root"] = True
            data["ir_meta"] = _ir_meta(ir, root_id)
        # 所有节点都给尺寸（含叶子）：画布按固定高度渲染、参数区内部滚动，
        # 保证「渲染高度 == 布局预留高度」，父子与兄弟都不会互相挤压
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


# ---------------------------------------------------------------------------
# 反向：GraphIR v2（画布 ir 图）→ IR
# ---------------------------------------------------------------------------

IR_NODE_TYPE = "ir"


class GraphIRMappingError(ValueError):
    """画布 ir 图 → IR 反向映射失败：message 定位到具体节点/字段（不静默产出坏 IR）。"""


def _handle_index(handle) -> Optional[int]:
    """句柄末尾序号（in0/in-0/out1 → 0/0/1）；裸句柄（in/out）→ None。"""
    if not isinstance(handle, str):
        return None
    matched = re.search(r"(\d+)\s*$", handle)
    return int(matched.group(1)) if matched else None


def _ordered_graph_edges(graph_edges: list[dict]) -> list[dict]:
    """边的还原顺序：画布声明序；**多输入目标**的入边按 `targetHandle` 序号重排。

    与 `network_export._ordered_in_edges` 同口径：IR 里多输入算子的操作数顺序由 edges
    声明序决定（`ir_codegen._render_op` 按 `in_edges` 顺序取输入），而画布 JSON 的边数组
    顺序可能被保存/合并打乱，句柄序号才是「哪个输入」的权威。单入边目标不受影响；
    缺句柄序号的边排在带序号的之后，彼此保持画布相对顺序。
    """
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(graph_edges):
        groups.setdefault(str(e.get("target")), []).append(i)

    def rank(j: int) -> tuple[int, int, int]:
        num = _handle_index(graph_edges[j].get("targetHandle"))
        return (0, num, j) if num is not None else (1, j, 0)

    out: list[dict] = []
    placed: set[int] = set()
    for i, e in enumerate(graph_edges):
        if i in placed:
            continue
        group = groups[str(e.get("target"))]
        if len(group) > 1:
            for j in sorted(group, key=rank):
                out.append(graph_edges[j])
                placed.add(j)
        else:
            out.append(e)
            placed.add(i)
    return out


def _shape_of(data: dict, key: str) -> Optional[list]:
    """节点形状（保真用，不参与再生成语义）：缺省/非法一律按 null 处理。

    `__in_shape`/`__out_shape` 是画布查看器字段（ir_to_graphir 与前端 irAdapter 都写），
    老图若只写了它们，同样能还原。
    """
    raw = data.get(key)
    if raw is None:
        raw = data.get("__in_shape" if key == "input_shape" else "__out_shape")
    if raw is None:
        return None
    if isinstance(raw, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in raw):
        return raw
    return None


def _parent_of(node: dict, data: dict) -> Optional[str]:
    """子节点归属：优先节点级 `parentId`（React Flow / ir_to_graphir 口径），
    兼容 data 里同名或 `parent_id` 的历史写法。"""
    for key in ("parentId", "parent_id"):
        val = node.get(key)
        if isinstance(val, str) and val:
            return val
    for key in ("parentId", "parent_id"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _resolve_root_id(nodes: list[dict], edges: list[dict], meta: dict,
                     marked: Optional[str]) -> str:
    """入口节点 id：根标记 > 元数据 root_id > 唯一顶层节点 > 顶层里唯一无入边者。"""
    ids = {n["id"] for n in nodes}
    for cand in (marked, meta.get("root_id") if isinstance(meta, dict) else None):
        if isinstance(cand, str) and cand in ids:
            return cand
    top = [n["id"] for n in nodes if not n.get("parent_id")]
    if len(top) == 1:
        return top[0]
    consumed = {e["to"] for e in edges}
    if len(top) > 1:
        free = [nid for nid in top if nid not in consumed]
        if len(free) == 1:
            return free[0]
    raise GraphIRMappingError(
        f"无法确定拆解 ir 图的根节点：顶层（parentId 为空）节点有 {len(top)} 个"
        f"（{top}），无入边的有 {[n for n in top if n not in consumed]} 个——"
        "根节点须唯一（入口模块），请把其余节点挂到父节点下或删除"
    )


def graphir_to_ir(graph: dict) -> dict:
    """画布 GraphIR v2（ir 图）→ IR（`ir_schema.validate_ir` 通过，可直接交 `ir_codegen.generate`）。

    映射规则（与 `ir_to_graphir` 逐项对照）：

    | IR 字段            | graph.json 来源                                             |
    |--------------------|-------------------------------------------------------------|
    | `id`               | `data.ir_id`（缺省回退 `node.id`）                          |
    | `kind`/`class_name`| `data.kind` / `data.class_name`                              |
    | `module_path`      | `data.module_path`                                          |
    | `params`           | `data.params`（**画布参数面板写回的调参结果**）              |
    | `code_hint`/`uncertain` | `data.code_hint` / `data.uncertain`                    |
    | `parent_id`        | `node.parentId`（层级树）                                    |
    | `input_shape`/`output_shape` | `data.input_shape`/`data.output_shape`（缺失回退 `__in_shape`/`__out_shape`） |
    | `edges[].from/to`  | 边的 `source`/`target`（经 node.id→ir_id 换算）              |
    | `edges[].tensor_shape` | `edge.data.tensor_shape`                                |
    | `root_id`          | `data.ir_root` 标记 > `data.ir_meta.root_id` > 唯一顶层节点   |
    | `entry_class` 等元数据 | `data.ir_meta`（老图无此字段时按根节点类名/缺省兜底）      |

    不可逆字段：边的 `kind`（data/skip）是 `_edge_kind` 由 IR 结构推导的**视觉类别**，
    IR 里并无对应字段，还原时不参与映射（往返后再用 `_edge_kind` 重算即可复现）；
    `position`/`handles`/`display`/`layout_hint` 为画布呈现字段，同样不进 IR。
    """
    if not isinstance(graph, dict):
        raise GraphIRMappingError("非法 GraphIR：需为对象")
    graph_nodes = graph.get("nodes")
    graph_edges = graph.get("edges")
    if not isinstance(graph_nodes, list) or not isinstance(graph_edges, list):
        raise GraphIRMappingError("非法 GraphIR：需含 nodes/edges 数组")
    if not graph_nodes:
        raise GraphIRMappingError("画布图为空：没有可还原为 IR 的 ir 节点")

    id_of: dict[str, str] = {}      # 画布节点 id → IR 节点 id
    meta: dict = {}
    marked: Optional[str] = None
    nodes_out: list[dict] = []
    for gn in graph_nodes:
        if not isinstance(gn, dict):
            raise GraphIRMappingError(f"非法 GraphIR：节点需为对象，实际 {gn!r}")
        if gn.get("type") != IR_NODE_TYPE:
            raise GraphIRMappingError(
                f"节点 {gn.get('id')!r} 的类型是 {gn.get('type')!r}，不是拆解 ir 节点："
                f"整张图须全部为 type=\"{IR_NODE_TYPE}\" 的节点"
            )
        data = gn.get("data")
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise GraphIRMappingError(
                f"节点 {gn.get('id')!r} 的 data 不是对象：ir 节点属性（kind/params/…）须放在 data 里"
            )
        nid = data.get("ir_id")
        if not isinstance(nid, str) or not nid:
            nid = gn.get("id")
        if not isinstance(nid, str) or not nid:
            raise GraphIRMappingError(
                f"ir 节点缺少 id（node.id 与 data.ir_id 均为空）：{gn.get('type')!r}"
            )
        node_id = gn.get("id")
        if isinstance(node_id, str) and node_id:
            id_of[node_id] = nid

        kind = data.get("kind")
        if kind not in KINDS:
            raise GraphIRMappingError(
                f"节点 {nid} 的 kind 缺失/非法：{kind!r}（应为 {list(KINDS)} 之一）"
            )
        class_name = data.get("class_name")
        if not isinstance(class_name, str):
            raise GraphIRMappingError(
                f"节点 {nid} 的 class_name 缺失/非字符串：{class_name!r}"
            )
        params = data.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise GraphIRMappingError(
                f"节点 {nid} 的 data.params 不是对象：{params!r}（画布参数须为键值对）"
            )

        node_out = {
            "id": nid,
            "kind": kind,
            "class_name": class_name,
            "params": params,
            "parent_id": _parent_of(gn, data),
            "input_shape": _shape_of(data, "input_shape"),
            "output_shape": _shape_of(data, "output_shape"),
        }
        module_path = data.get("module_path")
        if isinstance(module_path, str) and module_path:
            node_out["module_path"] = module_path
        code_hint = data.get("code_hint")
        if isinstance(code_hint, str) and code_hint:
            node_out["code_hint"] = code_hint
        if data.get("uncertain") is True:
            node_out["uncertain"] = True
        if data.get("ir_root") is True:
            marked = nid
        node_meta = data.get("ir_meta")
        if isinstance(node_meta, dict) and not meta:
            meta = node_meta
        nodes_out.append(node_out)

    ids = [n["id"] for n in nodes_out]
    if len(set(ids)) != len(ids):
        dupes = sorted({nid for nid in ids if ids.count(nid) > 1})
        raise GraphIRMappingError(f"ir 节点 id 重复：{dupes}（node.id 与 data.ir_id 冲突）")

    for ge in graph_edges:
        if not isinstance(ge, dict):
            raise GraphIRMappingError(f"非法 GraphIR：边需为对象，实际 {ge!r}")

    edges_out: list[dict] = []
    for ge in _ordered_graph_edges(graph_edges):
        src, tgt = ge.get("source"), ge.get("target")
        if not isinstance(src, str) or not isinstance(tgt, str):
            raise GraphIRMappingError(
                f"边 {ge.get('id')!r} 缺少 source/target 字符串：{ge!r}"
            )
        edge_out = {"from": id_of.get(src, src), "to": id_of.get(tgt, tgt)}
        e_data = ge.get("data")
        shape = e_data.get("tensor_shape") if isinstance(e_data, dict) else None
        if isinstance(shape, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in shape):
            edge_out["tensor_shape"] = shape
        edges_out.append(edge_out)

    # 层级归属经 node.id → ir_id 换算（画布节点 id 与 ir_id 不一致时仍能对上）
    for n in nodes_out:
        parent = n.get("parent_id")
        if parent is not None:
            n["parent_id"] = id_of.get(parent, parent)

    root_id = _resolve_root_id(nodes_out, edges_out, meta, marked)
    root = next((n for n in nodes_out if n["id"] == root_id), None)
    if root is None:
        raise GraphIRMappingError(f"根节点 {root_id} 不在图中")
    if root.get("parent_id"):
        raise GraphIRMappingError(
            f"拆解图根节点 {root_id} 有父节点 {root['parent_id']}：根节点须位于顶层（parentId 为空）"
        )

    task_type = meta.get("task_type")
    if task_type not in TASK_TYPES:
        task_type = "other"
    input_spec = meta.get("input_spec")
    if not isinstance(input_spec, dict) or not input_spec:
        input_spec = {"shape": root["input_shape"]} if root.get("input_shape") else {}
    entry_class = meta.get("entry_class")
    if not (isinstance(entry_class, str) and entry_class):
        entry_class = root["class_name"] or root_id
    schema_version = meta.get("schema_version")
    if not (isinstance(schema_version, str) and schema_version):
        schema_version = SCHEMA_VERSION
    source_file = meta.get("source_file")
    if not isinstance(source_file, str):
        source_file = ""
    ir = {
        "schema_version": schema_version,
        "source_file": source_file,
        "entry_class": entry_class,
        "task_type": task_type,
        "input_spec": input_spec,
        "root_id": root_id,
        "nodes": nodes_out,
        "edges": edges_out,
    }

    errors = validate_ir(ir)
    if errors:
        raise GraphIRMappingError(
            "画布 ir 图还原为 IR 后结构校验不通过：" + "；".join(errors)
        )
    return ir
