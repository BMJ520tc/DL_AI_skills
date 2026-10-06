"""架构级自迭代建议（需求六.1 延伸；模块详细设计 8.5、7.5 边界修订）。

自迭代闭环原只覆盖**超参**（`network_autotune`）；本服务补上**架构级**，且**人在环**：

    建议（agent 起草）→ 用户确认 → 改动落到画布（保存 = 新版本）→ 用户手动训练
    → 现有任务后蒸馏钩子回写知识 → 下一轮再带入

分工：
- **agent 起草**建议：读当前画布图 + 已确认知识 + 入库标准化模块库，产出结构化改动建议
  （`replace_module` 换模块 / `add_layer` 加层 / `rewire` 改连接），每条带理由与来源知识 id。
- **确定性 apply**：`apply_suggestion` 是**纯函数**（`copy.deepcopy` 入参），把某条建议作用到 GraphIR
  上并返回新图；非法即 `ValueError`。**不写盘**——由前端把新图放回画布、用户保存时才落 `graph.json`
  （`PUT /api/projects/{id}/graph` → `version_service.commit_graph`，即「每次改动 = 一个新版本」）。

边界（与导出/训练同一口径）：只支持「**标准节点 + module_ref**」图；拆解出的 **ir 图**在导出/训练
侧就禁止与标准/模块节点混拼（`network_export.generate`），故此处**一律拒绝**（建议入口 400、校验标无效、
apply 守卫三层拦截）。「换模块」只认**入库模块**（`module_id:version`），本地内存模块不参与。

报告落在 `data/arch_suggest/{task_id}.json`，`GET /api/networks/{id}/arch-suggestions` 取最新。
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import DATA_DIR
from app.services import (
    agent_service, ir_graphir, knowledge_service, network_export, project_manager, task_manager,
)

logger = logging.getLogger(__name__)

TASK_TYPE = "arch_suggest"
ARCH_DIR = DATA_DIR / "arch_suggest"

OPS = ("replace_module", "add_layer", "rewire")
_OP_LABELS = {"replace_module": "换模块", "add_layer": "加层", "rewire": "改连接"}
# 别名归一：模型在文件兜底里常写自有 op 名，归到三枚举值（参照 distill_service._TYPE_ALIASES 的做法）。
_OP_ALIASES = {
    "replace": "replace_module", "swap": "replace_module", "replace_node": "replace_module",
    "change_module": "replace_module", "module": "replace_module",
    "insert": "add_layer", "add": "add_layer", "add_node": "add_layer", "add_layer_node": "add_layer",
    "connect": "rewire", "rewire_edges": "rewire", "edge": "rewire", "connection": "rewire",
    "reconnect": "rewire",
}

ARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "description": "架构改动建议；没有可靠依据时返回空数组",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": list(OPS)},
                    "target_node_id": {"type": "string", "description": "被改动的既有节点 id（改连接时填源节点）"},
                    "description": {"type": "string", "description": "一句话说明改什么"},
                    "rationale": {"type": "string", "description": "为什么这么改（依据）"},
                    "source_knowledge_ids": {"type": "array", "items": {"type": "string"}},
                    "new_module_ref": {"type": "string", "description": "换模块用：入库模块 ref，形如 <module_id>:<version>"},
                    "anchor_edge_id": {"type": "string", "description": "加层用：在这条边上插入（优先）"},
                    "anchor_node_id": {"type": "string", "description": "加层用：在该节点之后插入（其唯一出边）"},
                    "new_node_type": {"type": "string", "description": "加层用：标准节点类型或 module_ref"},
                    "params": {"type": "object"},
                    "edits": {
                        "type": "array",
                        "description": "改连接用：逐条增删边",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["connect", "disconnect"]},
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "sourceHandle": {"type": "string"},
                                "targetHandle": {"type": "string"},
                            },
                            "required": ["action", "source", "target"],
                        },
                    },
                },
                "required": ["op", "target_node_id", "description"],
            },
        }
    },
    "required": ["suggestions"],
}

_MAX_SUGGESTIONS = 3
_SUMMARY_MAX_NODES = 120
_SUMMARY_MAX_EDGES = 300


class StaleGraphError(ValueError):
    """画布自生成建议后已变更（graph_hash 不符）——API 映射 409。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# 项目 / 图 读取
# ===========================================================================

def _ws(project: dict) -> Path:
    return Path(project["workspace_path"])


def _require_network(project_id: str) -> dict:
    try:
        return project_manager.require_type(project_id, {"structured"})
    except LookupError as e:
        raise LookupError(f"network not found: {e}") from e
    except PermissionError as e:
        raise PermissionError(f"不是结构化项目（画布网络）：{e}") from e


def _read_graph(project: dict) -> dict:
    p = _ws(project) / "graph.json"
    if not p.exists():
        raise ValueError("画布尚未保存过图（graph.json 不存在）")
    return json.loads(p.read_text(encoding="utf-8"))


def _assert_standard_graph(graph: dict) -> None:
    """守卫：只接受「标准节点 + module_ref」图；ir 图（拆解产物）一律拒绝。"""
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("画布图缺少 nodes 数组")
    if network_export.is_ir_graph(graph):
        raise ValueError("拆解 ir 图不支持架构级改动（ir 与标准/模块节点不能混拼，与导出/训练同一口径）")
    for n in nodes:
        if not isinstance(n, dict):
            raise ValueError("画布图节点形态非法")
        if n.get("type") == ir_graphir.IR_NODE_TYPE:
            raise ValueError("画布含拆解 ir 节点，不支持架构级改动（与导出/训练同一口径）")


def _graph_hash(graph: dict) -> str:
    """图的内容指纹（**位置/显示无关**）——用于「建议后画布被改」的漂移检测。

    只取结构性字段（节点 id/type/parentId/非下划线 data、边的拓扑与句柄），排序后在 JSON 上取 sha256；
    拖动节点、改 shape 元数据不改变指纹。
    """
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    proj_nodes = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        data = n.get("data") or {}
        proj_nodes.append({
            "id": n.get("id"),
            "type": n.get("type"),
            "parentId": n.get("parentId"),
            "data": {k: v for k, v in data.items() if not str(k).startswith("__")},
        })
    proj_nodes.sort(key=lambda x: str(x.get("id")))
    proj_edges = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        proj_edges.append({
            "source": e.get("source"), "sourceHandle": e.get("sourceHandle"),
            "target": e.get("target"), "targetHandle": e.get("targetHandle"),
        })
    proj_edges.sort(key=lambda x: (str(x["source"]), str(x["sourceHandle"]),
                                   str(x["target"]), str(x["targetHandle"])))
    blob = json.dumps({"nodes": proj_nodes, "edges": proj_edges}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ===========================================================================
# 形状容错（抄 distill_service：DeepSeek 文件兜底的容器/字段都会漂移）
# ===========================================================================

def _as_text(v) -> str:
    return v if isinstance(v, str) else ""


def _as_text_list(v) -> list[str]:
    if not isinstance(v, list):
        return []
    return [x for x in v if isinstance(x, str)]


def _normalize_op(raw) -> Optional[str]:
    t = _as_text(raw).strip()
    if t in OPS:
        return t
    return _OP_ALIASES.get(t.lower())


def _looks_like_suggestions(val) -> bool:
    return isinstance(val, list) and any(isinstance(x, dict) and x.get("op") for x in val)


def _suggestion_items(structured) -> list[dict]:
    """把 agent 产出归一成建议列表（兼容裸数组 / {"suggestions":[...]} / 任意数组值包装键 / 单对象）。"""
    if isinstance(structured, list):
        return [x for x in structured if isinstance(x, dict)]
    if isinstance(structured, dict):
        if structured.get("op"):
            return [structured]
        for key in ("suggestions", "items", "results"):
            if _looks_like_suggestions(structured.get(key)):
                return [x for x in structured[key] if isinstance(x, dict)]
        for val in structured.values():
            if _looks_like_suggestions(val):
                return [x for x in val if isinstance(x, dict)]
    return []


def _normalize_edits(raw) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for ed in raw:
        if not isinstance(ed, dict):
            continue
        action = _as_text(ed.get("action")).strip().lower()
        if action in ("remove", "delete", "cut"):
            action = "disconnect"
        elif action in ("add", "link"):
            action = "connect"
        out.append({
            "action": action,
            "source": _as_text(ed.get("source")),
            "target": _as_text(ed.get("target")),
            "sourceHandle": _as_text(ed.get("sourceHandle")),
            "targetHandle": _as_text(ed.get("targetHandle")),
        })
    return out


def _normalize_suggestion(raw: dict) -> Optional[dict]:
    op = _normalize_op(raw.get("op"))
    if op is None:
        return None
    s = {
        "suggestion_id": uuid.uuid4().hex[:12],
        "op": op,
        "op_label": _OP_LABELS[op],
        "target_node_id": (_as_text(raw.get("target_node_id")) or _as_text(raw.get("target"))
                           or _as_text(raw.get("node_id")) or _as_text(raw.get("source"))),
        "description": _as_text(raw.get("description")) or _as_text(raw.get("title")),
        "rationale": _as_text(raw.get("rationale")) or _as_text(raw.get("reason")),
        "source_knowledge_ids": _as_text_list(raw.get("source_knowledge_ids") or raw.get("sources")),
    }
    payload: dict = {}
    if op == "replace_module":
        payload = {"new_module_ref": (_as_text(raw.get("new_module_ref"))
                                      or _as_text(raw.get("module_ref"))
                                      or _as_text(raw.get("module_id")))}
    elif op == "add_layer":
        payload = {
            "anchor_edge_id": _as_text(raw.get("anchor_edge_id")),
            "anchor_node_id": _as_text(raw.get("anchor_node_id")),
            "new_node_type": (_as_text(raw.get("new_node_type")) or _as_text(raw.get("node_type"))
                              or _as_text(raw.get("type"))),
            "new_module_ref": _as_text(raw.get("new_module_ref")),
            "params": raw.get("params") if isinstance(raw.get("params"), dict) else {},
        }
    elif op == "rewire":
        payload = {"edits": _normalize_edits(raw.get("edits") or raw.get("edge_edits") or raw.get("changes"))}
    s["payload"] = payload
    return s


# ===========================================================================
# 模块库目录
# ===========================================================================

def _as_compat(value) -> dict:
    """解析 saved_module_compat 文本为 dict，容忍**多重 JSON 编码**。

    `record_module` 会对入参再 `json.dumps` 一次；调用方若已自行 dumps（历史上出现过），落库即双重编码，
    `_as_dict` 只解一层会得到 str → `{}`（handles 退回默认、误判句柄数）。此处循环解到 dict 为止。
    """
    for _ in range(3):
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return {}
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _module_handles(ref: Optional[str]) -> Optional[dict]:
    """按 `module_id:version` 取入库模块的句柄与展示信息；不存在返回 None（不猜）。"""
    if not ref:
        return None
    m = knowledge_service.get_module_by_ref(ref)
    if m is None:
        return None
    compat = _as_compat(m.get("saved_module_compat"))
    h = compat.get("handles") if isinstance(compat.get("handles"), dict) else {}
    inputs = h.get("inputs") or ["in"]
    outputs = h.get("outputs") or ["out"]
    return {
        "inputs": [str(x) for x in inputs if isinstance(x, (str, int))],
        "outputs": [str(x) for x in outputs if isinstance(x, (str, int))],
        "name": m.get("name"),
        "version": m.get("module_version"),
        "description": m.get("description"),
    }


def _module_catalog(limit: int = 200) -> list[dict]:
    out: list[dict] = []
    for m in knowledge_service.list_modules(limit=limit):
        ref = f"{m.get('module_id')}:{m.get('module_version')}"
        mh = _module_handles(ref)
        if mh is None:
            continue
        out.append({
            "ref": ref, "name": mh["name"], "version": mh["version"],
            "description": (m.get("description") or "")[:300],
            "task_type": m.get("task_type"),
            "handles": {"inputs": mh["inputs"], "outputs": mh["outputs"]},
        })
    return out


# ===========================================================================
# 图结构助手
# ===========================================================================

def _nodes(graph: dict) -> list[dict]:
    return [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]


def _edges(graph: dict) -> list[dict]:
    return [e for e in (graph.get("edges") or []) if isinstance(e, dict)]


def _node_by_id(graph: dict, node_id: str) -> Optional[dict]:
    for n in _nodes(graph):
        if n.get("id") == node_id:
            return n
    return None


def _handles_of_node(node: dict) -> dict:
    return network_export._handles_of(node.get("type"), node.get("data") or {})


def _effective_handles(spec: dict) -> dict:
    """把 NODE_TABLE 的句柄规格补成「有效句柄」：sources 为空 = 未声明输出（默认单输出 out-0）。

    注意 **targets 为空不做默认** —— `_HANDLES_INPUT`/`_HANDLES_CONST` 的 0 输入是语义（源节点），
    而 `_HANDLES_UNSPEC` 的 targets 本就是 ["in-0"]（显式单入）。
    """
    targets = list(spec.get("targets") or [])
    sources = list(spec.get("sources") or [])
    if not sources:
        sources = ["out-0"]
    return {"targets": targets, "sources": sources}


def _all_ids(graph: dict) -> set:
    ids = {n.get("id") for n in _nodes(graph)}
    ids |= {e.get("id") for e in _edges(graph)}
    return ids


def _new_node_id(graph: dict) -> str:
    existing = _all_ids(graph)
    i = 1
    while f"arch-node-{i}" in existing:
        i += 1
    return f"arch-node-{i}"


def _new_edge_id(graph: dict) -> str:
    existing = _all_ids(graph)
    i = 1
    while f"earch-{i}" in existing:
        i += 1
    return f"earch-{i}"


def _position_of(graph: dict, node_id: str) -> Optional[dict]:
    n = _node_by_id(graph, node_id)
    if not n:
        return None
    pos = n.get("position")
    if isinstance(pos, dict) and isinstance(pos.get("x"), (int, float)) and isinstance(pos.get("y"), (int, float)):
        return {"x": float(pos["x"]), "y": float(pos["y"])}
    return None


def _resolve_anchor(graph: dict, payload: dict) -> dict:
    """定位加层锚点边：优先 anchor_edge_id，其次 anchor_node_id 的唯一出边。"""
    edges = _edges(graph)
    edge_id = payload.get("anchor_edge_id")
    if edge_id:
        for e in edges:
            if e.get("id") == edge_id:
                return e
        raise ValueError(f"锚点边不存在：{edge_id}")
    node_id = payload.get("anchor_node_id")
    if not node_id:
        raise ValueError("加层未指定锚点：需要 anchor_edge_id 或 anchor_node_id")
    out = [e for e in edges if e.get("source") == node_id]
    if len(out) == 1:
        return out[0]
    if not out:
        raise ValueError(f"节点 {node_id} 没有出边，无法在其后加层")
    raise ValueError("节点 {} 有多条出边，请用 anchor_edge_id 指定其一：{}".format(
        node_id, ", ".join(str(e.get("id")) for e in out)))


def _validate_edit(graph: dict, ed: dict) -> tuple[bool, Optional[str]]:
    action = ed.get("action")
    if action not in ("connect", "disconnect"):
        return False, f"未知动作：{action or '(空)'}"
    ids = {n.get("id") for n in _nodes(graph)}
    src, tgt = ed.get("source"), ed.get("target")
    if src not in ids:
        return False, f"源节点不存在：{src}"
    if tgt not in ids:
        return False, f"目标节点不存在：{tgt}"
    if src == tgt:
        return False, "不允许自环"
    return True, None


# ===========================================================================
# 建议校验（确定性，展示给用户前先筛）
# ===========================================================================

def validate_suggestion(graph: dict, s: dict) -> tuple[bool, Optional[str]]:
    try:
        _assert_standard_graph(graph)
    except ValueError as e:
        return False, str(e)
    op = s.get("op")
    payload = s.get("payload") or {}
    tid = s.get("target_node_id")
    nodes = {n.get("id") for n in _nodes(graph)}

    if op == "replace_module":
        if tid not in nodes:
            return False, f"目标节点不存在：{tid}"
        mh = _module_handles(payload.get("new_module_ref"))
        if mh is None:
            return False, f"模块不在库中：{payload.get('new_module_ref') or '(未指定)'}"
        in_deg = sum(1 for e in _edges(graph) if e.get("target") == tid)
        out_deg = sum(1 for e in _edges(graph) if e.get("source") == tid)
        if len(mh["inputs"]) != in_deg or len(mh["outputs"]) != out_deg:
            return False, (f"模块句柄数（{len(mh['inputs'])} 入/{len(mh['outputs'])} 出）"
                           f"与被替换节点的边数（{in_deg} 入/{out_deg} 出）不匹配")
        return True, None

    if op == "add_layer":
        nt = payload.get("new_node_type")
        if not nt:
            return False, "未指定新节点类型（new_node_type）"
        if nt == "module_ref":
            mh = _module_handles(payload.get("new_module_ref"))
            if mh is None:
                return False, f"模块不在库中：{payload.get('new_module_ref') or '(未指定)'}"
            if len(mh["inputs"]) != 1 or len(mh["outputs"]) != 1:
                return False, "加层模块需单入单出"
        else:
            spec = network_export.NODE_TABLE.get(nt)
            if spec is None:
                return False, f"未知标准节点类型：{nt}"
            eff = _effective_handles(spec.get("handles") or {})
            if len(eff["targets"]) != 1 or len(eff["sources"]) != 1:
                return False, f"加层节点需单入单出（{nt} 为 {len(eff['targets'])} 入/{len(eff['sources'])} 出）"
        try:
            _resolve_anchor(graph, payload)
        except ValueError as e:
            return False, str(e)
        return True, None

    if op == "rewire":
        edits = payload.get("edits") or []
        if not edits:
            return False, "未提供任何连接改动"
        for i, ed in enumerate(edits, 1):
            ok, reason = _validate_edit(graph, ed)
            if not ok:
                return False, f"第 {i} 条改动无效：{reason}"
        return True, None

    return False, f"未知改动类型：{op}"


# ===========================================================================
# 确定性 apply（纯函数；非法一律 ValueError）
# ===========================================================================

def _remap_handles(edges: list[dict], old_handles: list[str], new_handles: list[str],
                   *, is_target: bool) -> None:
    """按旧句柄顺序把一组边的句柄重映射为新句柄（顺序即位置）。"""
    key = "targetHandle" if is_target else "sourceHandle"

    def order(e: dict) -> int:
        h = e.get(key)
        return old_handles.index(h) if h in old_handles else 999

    for i, e in enumerate(sorted(edges, key=order)):
        if i < len(new_handles):
            e[key] = new_handles[i]


def _apply_replace_module(graph: dict, s: dict) -> dict:
    out = copy.deepcopy(graph)
    tid = s.get("target_node_id")
    ref = (s.get("payload") or {}).get("new_module_ref")
    mh = _module_handles(ref)
    if mh is None:
        raise ValueError(f"模块不在库中：{ref}")
    node = _node_by_id(out, tid)
    if node is None:
        raise ValueError(f"目标节点不存在：{tid}")
    in_edges = [e for e in _edges(out) if e.get("target") == tid]
    out_edges = [e for e in _edges(out) if e.get("source") == tid]
    if len(mh["inputs"]) != len(in_edges) or len(mh["outputs"]) != len(out_edges):
        raise ValueError("模块句柄数与被替换节点的边数不匹配")
    old = _handles_of_node(node)
    _remap_handles(in_edges, list(old.get("targets") or []), mh["inputs"], is_target=True)
    _remap_handles(out_edges, list(old.get("sources") or []), mh["outputs"], is_target=False)
    data: dict = {"moduleId": ref, "handles": {"inputs": mh["inputs"], "outputs": mh["outputs"]}}
    if mh.get("name"):
        data["name"] = mh["name"]
        data["label"] = mh["name"]
    if mh.get("version"):
        data["version"] = mh["version"]
    if mh.get("description"):
        data["description"] = mh["description"]
    node["type"] = "module_ref"
    node["data"] = data
    return out


def _apply_add_layer(graph: dict, s: dict) -> dict:
    out = copy.deepcopy(graph)
    payload = s.get("payload") or {}
    nt = payload.get("new_node_type")
    if nt == "module_ref":
        mh = _module_handles(payload.get("new_module_ref"))
        if mh is None:
            raise ValueError(f"模块不在库中：{payload.get('new_module_ref')}")
        if len(mh["inputs"]) != 1 or len(mh["outputs"]) != 1:
            raise ValueError("加层模块需单入单出")
        new_type = "module_ref"
        new_data: dict = {"moduleId": payload.get("new_module_ref"),
                          "handles": {"inputs": mh["inputs"], "outputs": mh["outputs"]}}
        if mh.get("name"):
            new_data["name"] = mh["name"]
            new_data["label"] = mh["name"]
        new_in, new_out = mh["inputs"][0], mh["outputs"][0]
    else:
        spec = network_export.NODE_TABLE.get(nt)
        if spec is None:
            raise ValueError(f"未知标准节点类型：{nt}")
        eff = _effective_handles(spec.get("handles") or {})
        if len(eff["targets"]) != 1 or len(eff["sources"]) != 1:
            raise ValueError(f"加层节点需单入单出（{nt} 为 {len(eff['targets'])} 入/{len(eff['sources'])} 出）")
        new_type = nt
        new_data = dict(payload.get("params") or {})
        new_in, new_out = eff["targets"][0], eff["sources"][0]

    anchor = _resolve_anchor(out, payload)
    a_id, b_id = anchor.get("source"), anchor.get("target")
    node_a, node_b = _node_by_id(out, a_id), _node_by_id(out, b_id)
    if node_a is None or node_b is None:
        raise ValueError("锚点边的端点不存在")
    src_h = anchor.get("sourceHandle") or (_handles_of_node(node_a).get("sources") or ["out-0"])[0]
    tgt_h = anchor.get("targetHandle") or (_handles_of_node(node_b).get("targets") or [""])[0]

    pos_a, pos_b = _position_of(out, a_id), _position_of(out, b_id)
    if pos_a and pos_b:
        pos = {"x": (pos_a["x"] + pos_b["x"]) / 2.0, "y": (pos_a["y"] + pos_b["y"]) / 2.0}
    elif pos_a:
        pos = {"x": pos_a["x"], "y": pos_a["y"] + 120.0}
    else:
        pos = {"x": 0.0, "y": 0.0}

    node_id = _new_node_id(out)
    new_node = {"id": node_id, "type": new_type, "position": pos, "data": new_data}
    out["edges"] = [e for e in _edges(out) if e is not anchor]
    out["nodes"] = [n for n in out.get("nodes") or [] if isinstance(n, dict)] + [new_node]
    e1 = {"id": _new_edge_id(out), "source": a_id, "sourceHandle": src_h,
          "target": node_id, "targetHandle": new_in, "data": {}}
    out["edges"].append(e1)
    e2 = {"id": _new_edge_id(out), "source": node_id, "sourceHandle": new_out,
          "target": b_id, "targetHandle": tgt_h, "data": {}}
    out["edges"].append(e2)
    return out


def _edge_match(e: dict, src: str, tgt: str, sh: str, th: str) -> bool:
    if e.get("source") != src or e.get("target") != tgt:
        return False
    if sh and e.get("sourceHandle") != sh:
        return False
    if th and e.get("targetHandle") != th:
        return False
    return True


def _apply_rewire(graph: dict, s: dict) -> dict:
    out = copy.deepcopy(graph)
    edits = (s.get("payload") or {}).get("edits") or []
    if not edits:
        raise ValueError("未提供任何连接改动")
    for i, ed in enumerate(edits, 1):
        ok, reason = _validate_edit(out, ed)
        if not ok:
            raise ValueError(f"第 {i} 条改动无效：{reason}")
        action, src, tgt = ed.get("action"), ed.get("source"), ed.get("target")
        sh, th = ed.get("sourceHandle") or "", ed.get("targetHandle") or ""
        if action == "disconnect":
            before = len(out.get("edges") or [])
            out["edges"] = [e for e in _edges(out) if not _edge_match(e, src, tgt, sh, th)]
            if len(out["edges"]) == before:
                raise ValueError(f"第 {i} 条：未找到要断开的边 {src}→{tgt}")
        else:
            node_src, node_tgt = _node_by_id(out, src), _node_by_id(out, tgt)
            src_handles = _handles_of_node(node_src).get("sources") or ["out-0"]
            tgt_handles = _handles_of_node(node_tgt).get("targets") or []
            sh2 = sh or src_handles[0]
            th2 = th or (tgt_handles[0] if tgt_handles else "")
            if tgt_handles and th2 not in tgt_handles:
                raise ValueError(f"第 {i} 条：目标句柄 {th2} 不在 {tgt} 的入句柄 {tgt_handles} 中")
            # 该入句柄已被占用 → 替换（单入句柄不叠加）
            out["edges"] = [e for e in _edges(out)
                            if not (e.get("target") == tgt and e.get("targetHandle") == th2)]
            out["edges"].append({"id": _new_edge_id(out), "source": src, "sourceHandle": sh2,
                                 "target": tgt, "targetHandle": th2, "data": {}})
    return out


_APPLY = {"replace_module": _apply_replace_module, "add_layer": _apply_add_layer, "rewire": _apply_rewire}


def apply_suggestion(graph: dict, s: dict) -> dict:
    """把一条建议作用到图的**副本**上并返回新图（不写盘）。非法 → ValueError。"""
    _assert_standard_graph(graph)
    op = s.get("op")
    fn = _APPLY.get(op)
    if fn is None:
        raise ValueError(f"未知改动类型：{op}")
    return fn(graph, s)


# ===========================================================================
# prompt
# ===========================================================================

def _graph_summary(graph: dict) -> dict:
    nodes, edges = _nodes(graph), _edges(graph)
    out_nodes = []
    for n in nodes[:_SUMMARY_MAX_NODES]:
        d = n.get("data") or {}
        item = {"id": n.get("id"), "type": n.get("type")}
        if n.get("type") == "module_ref":
            item["moduleId"] = d.get("moduleId")
        params = {k: v for k, v in d.items()
                  if not str(k).startswith("__")
                  and k not in ("handles", "moduleId", "label", "description", "name", "version")}
        if params:
            item["params"] = params
        out_nodes.append(item)
    out_edges = [{"id": e.get("id"), "source": e.get("source"), "sourceHandle": e.get("sourceHandle"),
                  "target": e.get("target"), "targetHandle": e.get("targetHandle")}
                 for e in edges[:_SUMMARY_MAX_EDGES]]
    return {"nodes": out_nodes, "edges": out_edges, "n_nodes": len(nodes), "n_edges": len(edges),
            "truncated": len(nodes) > _SUMMARY_MAX_NODES or len(edges) > _SUMMARY_MAX_EDGES}


def _addable_node_types() -> list[str]:
    """可作为加层节点（单入单出）的标准节点类型清单——喂给 agent，避免它编造类型名。"""
    out = []
    for name, spec in network_export.NODE_TABLE.items():
        eff = _effective_handles((spec or {}).get("handles") or {})
        if len(eff["targets"]) == 1 and len(eff["sources"]) == 1:
            out.append(name)
    return sorted(out)


def _prompt(graph: dict, knowledge: dict, catalog: list[dict], project: dict, hint: Optional[str]) -> str:
    summary = _graph_summary(graph)
    kn = {
        "param_advice": knowledge.get("param_advice") or [],
        "usage_guidance": knowledge.get("others") or [],
        "dependency_conflict": knowledge.get("dependency_conflict") or [],
    }
    return (
        "你是深度学习**模型架构改进**助手。目标：基于「当前画布网络 + 已确认的蒸馏知识 + 可用的入库标准化模块」，"
        "给出**架构级**改动建议（不是调超参）。\n\n"
        "## 可用的三种改动（op）\n"
        "- `replace_module`（换模块）：把一个既有节点替换为某个**入库模块**；需要 `target_node_id` 与 `new_module_ref`；"
        "模块的输入/输出句柄数必须与被替换节点的入边/出边数一致。\n"
        "- `add_layer`（加层）：在某条边上插入一个**单入单出**的节点（标准节点用 `new_node_type`，模块用 `new_node_type=\"module_ref\"` + `new_module_ref`）；"
        "优先用 `anchor_edge_id` 指定要插入的边（更精确），也可给 `anchor_node_id`（其唯一出边，出边多于一条时必须用 anchor_edge_id）。\n"
        "- `rewire`（改连接）：用 `edits` 逐条增删边，每条 `{action: connect|disconnect, source, target, sourceHandle?, targetHandle?}`。\n\n"
        "## 硬约束\n"
        f"1. `op` 只能是 {list(OPS)} 之一；最多产出 {_MAX_SUGGESTIONS} 条建议；无可靠依据就返回空数组。\n"
        "2. 每条建议必须给 `target_node_id`（必须是下方图里存在的节点 id）、`description`、`rationale`，"
        "并在 `source_knowledge_ids` 里列出用到的知识 id（若用了知识）。\n"
        "3. `new_module_ref` 必须是下方「模块目录」里的 `ref`（形如 module_id:version）；不得编造。\n"
        "4. 只做「标准节点 + 模块节点」的改动；**绝不要**产出 type=\"ir\" 的节点。\n"
        "5. 建议要**保守且可执行**：宁可少给，也不要给出无法落地（句柄数不符/节点不存在）的改动。\n\n"
        f"## 可用于加层的标准节点类型（单入单出，new_node_type 必须取自此表）\n"
        f"{json.dumps(_addable_node_types(), ensure_ascii=False)}\n\n"
        f"## 当前画布图（JSON）\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"## 入库标准化模块目录（JSON）\n{json.dumps(catalog, ensure_ascii=False, indent=2)}\n\n"
        f"## 相关已确认知识（JSON）\n{json.dumps(kn, ensure_ascii=False, indent=2)}\n\n"
        + (f"## 用户补充说明\n{hint}\n\n" if hint else "")
        + "请只输出 JSON（suggestions 数组），简洁、可执行。"
    )


# ===========================================================================
# 报告落盘
# ===========================================================================

def _write_report(report: dict) -> None:
    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    (ARCH_DIR / f"{report['task_id']}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def get_report(task_id: str) -> Optional[dict]:
    path = ARCH_DIR / f"{task_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_latest_report(project_id: str) -> Optional[dict]:
    if not ARCH_DIR.exists():
        return None
    latest: Optional[dict] = None
    latest_ts = ""
    for p in ARCH_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("project_id") != project_id:
            continue
        ts = d.get("created_at") or ""
        if latest is None or ts > latest_ts:
            latest, latest_ts = d, ts
    return latest


# ===========================================================================
# 任务
# ===========================================================================

def start_suggest(project_id: str, body: dict) -> dict:
    """发起架构建议任务：校验结构化项目 + 非 ir 图后入队。返回 {task_id, graph_hash}。"""
    project = _require_network(project_id)
    graph = _read_graph(project)
    _assert_standard_graph(graph)
    graph_hash = _graph_hash(graph)
    task_id = task_manager.create_task(TASK_TYPE, project_id=project_id, params={
        "project_id": project_id,
        "graph_hash": graph_hash,
        "task_type": body.get("task_type"),
        "model": body.get("model") or project.get("name"),
        "dataset": body.get("dataset"),
        "hint": body.get("hint"),
    })
    return {"task_id": task_id, "graph_hash": graph_hash}


async def _run(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    if project is None:
        raise RuntimeError(f"project not found: {project_id}")

    task_manager.update_progress(task_id, {"stage": "载入画布与知识"})
    graph = _read_graph(project)
    _assert_standard_graph(graph)

    try:
        knowledge = knowledge_service.bring_knowledge(
            task_type=params.get("task_type"), model=params.get("model"), dataset=params.get("dataset"))
    except Exception:  # noqa: BLE001 —— 带入失败不阻断建议生成
        logger.exception("架构建议：知识带入失败 project_id=%s", project_id)
        knowledge = {}
    catalog = _module_catalog()

    report = {
        "task_id": task_id,
        "project_id": project_id,
        "created_at": _now(),
        "graph_hash": params.get("graph_hash"),
        "task_type": params.get("task_type"),
        "model": params.get("model"),
        "dataset": params.get("dataset"),
        "knowledge_used": {k: len(v) for k, v in knowledge.items() if isinstance(v, list)},
        "module_catalog_size": len(catalog),
        "suggestions": [],
        "agent_error": None,
    }

    task_manager.update_progress(task_id, {"stage": "agent 起草架构建议"})
    structured = None
    try:
        res = await agent_service.run_sync(
            _prompt(graph, knowledge, catalog, project, params.get("hint")),
            output_schema=ARCH_SCHEMA, timeout_s=180)
        structured = res.get("structured_output")
    except Exception as exc:  # noqa: BLE001 —— agent 失败不伪造结论：如实记 agent_error
        logger.exception("架构建议 agent 失败 project_id=%s", project_id)
        report["agent_error"] = str(exc)

    task_manager.update_progress(task_id, {"stage": "校验建议"})
    suggestions = []
    for raw in _suggestion_items(structured):
        s = _normalize_suggestion(raw)
        if s is None:
            continue
        ok, reason = validate_suggestion(graph, s)
        s["valid"] = bool(ok)
        s["invalid_reason"] = None if ok else reason
        s["source"] = "agent"
        suggestions.append(s)
    if report["agent_error"] is None and not suggestions:
        report["agent_error"] = "agent 未产出可用的架构建议（形状或类型不符）"
    report["suggestions"] = suggestions

    _write_report(report)
    task_manager.update_progress(task_id, {"stage": "完成", "n_suggestions": len(suggestions)})


def apply_for_report(project_id: str, task_id: str, suggestion_id: str) -> dict:
    """按报告里的某条建议把改动作用到**当前** graph.json 的副本上并返回（不写盘）。"""
    project = _require_network(project_id)
    report = get_report(task_id)
    if report is None or report.get("project_id") != project_id:
        raise LookupError("架构建议报告不存在")
    s = next((x for x in report.get("suggestions") or [] if x.get("suggestion_id") == suggestion_id), None)
    if s is None:
        raise LookupError(f"建议不存在：{suggestion_id}")
    graph = _read_graph(project)
    if report.get("graph_hash") and report["graph_hash"] != _graph_hash(graph):
        raise StaleGraphError("画布自生成建议后已变更，请重新生成建议")
    new_graph = apply_suggestion(graph, s)
    return {
        "graph": new_graph,
        "suggestion_id": suggestion_id,
        "changes": {"op": s.get("op"), "op_label": s.get("op_label"),
                    "target_node_id": s.get("target_node_id")},
    }


def register() -> None:
    task_manager.register_handler(TASK_TYPE, _run)
