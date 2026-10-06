"""架构级自迭代建议（arch_service）：归一容错 / 确定性校验 / 三种 apply / 图指纹 / ir 守卫。

纯函数为主，不启 worker、不跑 agent；模块查库走临时知识库（isolated_db）。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.services import arch_service


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _std_graph() -> dict:
    """标准链 Linear→ReLU→Linear（带句柄，可加层/换模块/改连接）。"""
    return {
        "nodes": [
            {"id": "n1", "type": "linear_layer", "position": {"x": 0, "y": 0},
             "data": {"in_features": 4, "out_features": 8, "bias": True}},
            {"id": "n2", "type": "relu_layer", "position": {"x": 200, "y": 0}, "data": {}},
            {"id": "n3", "type": "linear_layer", "position": {"x": 400, "y": 0},
             "data": {"in_features": 8, "out_features": 2, "bias": True}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "out-0", "target": "n2", "targetHandle": "in-0"},
            {"id": "e2", "source": "n2", "sourceHandle": "out-0", "target": "n3", "targetHandle": "in-0"},
        ],
    }


def _ir_graph() -> dict:
    return {"nodes": [{"id": "m", "type": "ir", "data": {"kind": "module"}}], "edges": []}


def _record_module(tmp_path, mid: str = "mod_arch", ver: str = "v1",
                   inputs=("in",), outputs=("out",)) -> str:
    from app.services import knowledge_service as ks

    pkg = tmp_path / f"{mid}_{ver}"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "module.py").write_text(
        "import torch.nn as nn\nclass M(nn.Module):\n    def forward(self, x):\n        return x\n",
        encoding="utf-8")
    ks.record_module({
        "module_id": mid, "module_version": ver, "name": "M", "description": None,
        "source_project_id": None, "source_paper_id": None, "task_type": None,
        "input_spec": None, "output_spec": None, "params_schema": None, "tags": None,
        "verification": None,
        "saved_module_compat": {
            "id": f"{mid}:{ver}", "name": "M", "version": ver,
            "handles": {"inputs": list(inputs), "outputs": list(outputs)},
            "graph": {"nodes": [], "edges": []},
        },
        "path": str(pkg),
    })
    return f"{mid}:{ver}"


def _sug(op: str, **kw) -> dict:
    s = {"suggestion_id": "s1", "op": op, "target_node_id": kw.pop("target_node_id", "n2"),
         "description": "d", "rationale": "r", "source_knowledge_ids": [], "payload": {}}
    s["payload"].update(kw)
    return s


# ---------------------------------------------------------------------------
# 归一
# ---------------------------------------------------------------------------

def test_suggestion_items_container_shapes():
    bare = [{"op": "add_layer"}, {"op": "rewire"}]
    assert len(arch_service._suggestion_items(bare)) == 2
    assert len(arch_service._suggestion_items({"suggestions": bare})) == 2
    # 以 op 名做键的包装（模型按类型名包一层）
    assert len(arch_service._suggestion_items({"add_layer": bare})) == 2
    # 单条对象
    assert len(arch_service._suggestion_items({"op": "add_layer"})) == 1
    # 垃圾
    assert arch_service._suggestion_items(None) == []
    assert arch_service._suggestion_items({"foo": "bar"}) == []


def test_normalize_op_aliases_and_drop_unknown():
    assert arch_service._normalize_op("insert") == "add_layer"
    assert arch_service._normalize_op("swap") == "replace_module"
    assert arch_service._normalize_op("connect") == "rewire"
    assert arch_service._normalize_op("teleport") is None
    # 未知 op → 整条丢弃
    assert arch_service._normalize_suggestion({"op": "teleport"}) is None


def test_normalize_field_coercion():
    s = arch_service._normalize_suggestion({
        "op": "add_layer", "target_node_id": 123, "description": None,
        "source_knowledge_ids": ["k1", 5, None],
        "new_node_type": "relu_layer", "params": "not-a-dict",
    })
    assert s["target_node_id"] == ""           # 非字符串一律空
    assert s["description"] == ""
    assert s["source_knowledge_ids"] == ["k1"]  # 只留字符串
    assert s["payload"]["params"] == {}         # 非 dict → {}


def test_normalize_edits_action_aliases():
    edits = arch_service._normalize_edits([
        {"action": "remove", "source": "a", "target": "b"},
        {"action": "add", "source": "b", "target": "c"},
    ])
    assert [e["action"] for e in edits] == ["disconnect", "connect"]


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def test_validate_rejects_ir_graph():
    ok, reason = arch_service.validate_suggestion(_ir_graph(), _sug("add_layer", new_node_type="relu_layer"))
    assert ok is False and "ir" in reason


def test_validate_add_layer_standard_entry():
    g = _std_graph()
    ok, _ = arch_service.validate_suggestion(
        g, _sug("add_layer", anchor_edge_id="e2", new_node_type="relu_layer"))
    assert ok is True
    # 多入多出节点不允许加层
    ok2, reason2 = arch_service.validate_suggestion(
        g, _sug("add_layer", anchor_edge_id="e2", new_node_type="add_layer"))
    assert ok2 is False and "单入单出" in reason2
    # 未知类型
    ok3, _ = arch_service.validate_suggestion(
        g, _sug("add_layer", anchor_edge_id="e2", new_node_type="nope_layer"))
    assert ok3 is False
    # 锚点边不存在
    ok4, _ = arch_service.validate_suggestion(
        g, _sug("add_layer", anchor_edge_id="missing", new_node_type="relu_layer"))
    assert ok4 is False


def test_validate_add_layer_multi_out_needs_edge_id():
    g = _std_graph()
    # 让 n2 有两条出边（同时连 n3 与 n1 造环无妨，仅测锚点解析）
    g["edges"].append({"id": "e3", "source": "n2", "target": "n1", "targetHandle": "in-0"})
    ok, reason = arch_service.validate_suggestion(
        g, _sug("add_layer", anchor_node_id="n2", new_node_type="relu_layer"))
    assert ok is False and "e2" in reason and "e3" in reason   # 列出候选边
    ok2, _ = arch_service.validate_suggestion(
        g, _sug("add_layer", anchor_edge_id="e2", new_node_type="relu_layer"))
    assert ok2 is True


def test_validate_replace_module_handle_mismatch(isolated_db, tmp_path):
    ref1 = _record_module(tmp_path, mid="mod1", inputs=("in",), outputs=("out",))
    ref2 = _record_module(tmp_path, mid="mod2", inputs=("a", "b"), outputs=("out",))
    g = _std_graph()  # n2 relu: 1 入 1 出
    ok, _ = arch_service.validate_suggestion(g, _sug("replace_module", target_node_id="n2", new_module_ref=ref1))
    assert ok is True
    ok2, reason2 = arch_service.validate_suggestion(
        g, _sug("replace_module", target_node_id="n2", new_module_ref=ref2))
    assert ok2 is False and "匹配" in reason2
    ok3, _ = arch_service.validate_suggestion(
        g, _sug("replace_module", target_node_id="n2", new_module_ref="ghost:v9"))
    assert ok3 is False


def test_validate_rewire_endpoints():
    g = _std_graph()
    ok, _ = arch_service.validate_suggestion(g, _sug("rewire", edits=[
        {"action": "disconnect", "source": "n2", "target": "n3"}]))
    assert ok is True
    ok2, _ = arch_service.validate_suggestion(g, _sug("rewire", edits=[
        {"action": "connect", "source": "n2", "target": "ghost"}]))
    assert ok2 is False
    ok3, _ = arch_service.validate_suggestion(g, _sug("rewire", edits=[
        {"action": "connect", "source": "n2", "target": "n2"}]))
    assert ok3 is False  # 自环


# ---------------------------------------------------------------------------
# apply（纯函数）
# ---------------------------------------------------------------------------

def test_apply_ir_graph_raises():
    with pytest.raises(ValueError):
        arch_service.apply_suggestion(_ir_graph(), _sug("add_layer", new_node_type="relu_layer"))


def test_apply_add_layer_splits_edge():
    g = _std_graph()
    out = arch_service.apply_suggestion(
        g, _sug("add_layer", anchor_edge_id="e2", new_node_type="relu_layer"))
    assert len(out["nodes"]) == 4
    assert len(out["edges"]) == 3
    new = [n for n in out["nodes"] if n["id"].startswith("arch-node-")][0]
    assert new["type"] == "relu_layer"
    # n2 → new → n3
    assert any(e["source"] == "n2" and e["target"] == new["id"] for e in out["edges"])
    assert any(e["source"] == new["id"] and e["target"] == "n3" for e in out["edges"])
    # 原图未被改动（纯函数）
    assert len(g["nodes"]) == 3 and len(g["edges"]) == 2
    # 不引入 ir 节点
    assert all(n["type"] != "ir" for n in out["nodes"])


def test_apply_replace_module_remaps_edges(isolated_db, tmp_path):
    ref = _record_module(tmp_path, mid="mod1", inputs=("in",), outputs=("out",))
    out = arch_service.apply_suggestion(_std_graph(), _sug("replace_module", target_node_id="n2", new_module_ref=ref))
    n2 = [n for n in out["nodes"] if n["id"] == "n2"][0]
    assert n2["type"] == "module_ref" and n2["data"]["moduleId"] == ref
    assert n2["data"]["handles"] == {"inputs": ["in"], "outputs": ["out"]}
    # 边句柄按位置重映射到模块句柄
    assert [e for e in out["edges"] if e["target"] == "n2"][0]["targetHandle"] == "in"
    assert [e for e in out["edges"] if e["source"] == "n2"][0]["sourceHandle"] == "out"


def test_apply_rewire_connect_replaces_single_target():
    g = _std_graph()
    out = arch_service.apply_suggestion(g, _sug("rewire", edits=[
        {"action": "connect", "source": "n1", "target": "n3", "targetHandle": "in-0"}]))
    # n3 的 in-0 原被 e2 占用 → 替换，不叠加
    into_n3 = [e for e in out["edges"] if e["target"] == "n3"]
    assert len(into_n3) == 1 and into_n3[0]["source"] == "n1"


def test_apply_rewire_disconnect_missing_raises():
    with pytest.raises(ValueError):
        arch_service.apply_suggestion(_std_graph(), _sug("rewire", edits=[
            {"action": "disconnect", "source": "n1", "target": "n3"}]))


def test_new_ids_avoid_collision():
    g = _std_graph()
    g["nodes"].append({"id": "arch-node-1", "type": "relu_layer", "data": {}})
    g["edges"].append({"id": "earch-1", "source": "n3", "target": "arch-node-1", "targetHandle": "in-0"})
    out = arch_service.apply_suggestion(g, _sug("add_layer", anchor_edge_id="e1", new_node_type="relu_layer"))
    ids = {n["id"] for n in out["nodes"]}
    new_id = [i for i in ids if i.startswith("arch-node-") and i != "arch-node-1"]
    assert new_id and new_id[0] != "arch-node-1"
    edge_ids = [e["id"] for e in out["edges"]]
    assert len(edge_ids) == len(set(edge_ids))   # 边 id 不重复


# ---------------------------------------------------------------------------
# 图指纹
# ---------------------------------------------------------------------------

def test_graph_hash_position_insensitive_topology_sensitive():
    g1 = _std_graph()
    g2 = _std_graph()
    for n in g2["nodes"]:
        n["position"] = {"x": n["position"]["x"] + 999, "y": n["position"]["y"] - 5}
    assert arch_service._graph_hash(g1) == arch_service._graph_hash(g2)
    g3 = _std_graph()
    g3["edges"] = g3["edges"][:1]   # 拓扑变了
    assert arch_service._graph_hash(g1) != arch_service._graph_hash(g3)


# ---------------------------------------------------------------------------
# _run：归一 + 校验接线（agent 用桩）
# ---------------------------------------------------------------------------

def test_run_writes_report_with_validated_suggestions(isolated_db, tmp_path, monkeypatch):
    from app.services import agent_service, project_manager

    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    project_id = project_manager.create_project("structured", None, "arch-case")
    ws = Path(project_manager.get_project(project_id)["workspace_path"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "graph.json").write_text(json.dumps(_std_graph()), encoding="utf-8")

    async def fake_run_sync(prompt, output_schema=None, timeout_s=None, **kw):
        # 模拟漂移：裸数组 + 一条非法（目标不存在）
        return {"structured_output": [
            {"op": "add_layer", "target_node_id": "n2", "description": "加激活",
             "anchor_edge_id": "e2", "new_node_type": "relu_layer"},
            {"op": "replace_module", "target_node_id": "ghost", "description": "坏",
             "new_module_ref": "x:v1"},
        ]}

    monkeypatch.setattr(agent_service, "run_sync", fake_run_sync)
    asyncio.run(arch_service._run({"project_id": project_id, "graph_hash": "h",
                                   "task_type": None, "model": None, "dataset": None}, "t1"))

    report = arch_service.get_report("t1")
    assert report is not None and len(report["suggestions"]) == 2
    assert report["suggestions"][0]["valid"] is True
    assert report["suggestions"][1]["valid"] is False
    assert arch_service.get_latest_report(project_id)["task_id"] == "t1"
