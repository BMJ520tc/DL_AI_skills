"""模块四「IR 结构编辑（边/节点增删改）」的回归用例（模块详细设计 6.2 延伸）。

背景：agent 产出的 IR 常「结构合法但不忠实」（漏子树、空 module、把标准层当 module、
inputs 填参数名…），此前只能人工改 `reports/ir.json`。本轮把它产品化为一小组原子操作。

口径：
- 结构改动落盘后回传 `validate_ir + incomplete_ir` 的当前结果（供界面提示），
  但**不因软错误拒绝写入**（「新增 op 尚无入边」是走向完整图的合法中间态）；
- 硬错误（id 非法/重复、kind 非法、parent 不存在或成环、边两端不存在/自环/重边/成环、
  删根、删有子节点的节点）一律 400；
- 任何节点/边改动都让 `ir_hash` 变化 → 旧验证转 stale（与调参同口径）。

只使用临时库（conftest.isolated_db）与临时工作区/模块目录，不碰真实 data/。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import decompose_service, project_manager
from app.services.ir_schema import ir_hash


@pytest.fixture()
def ir_env(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    monkeypatch.setattr(decompose_service, "MODULES_DIR", modules_dir)
    project_id = project_manager.create_project("original", source="local-test")
    return {"project_id": project_id}


def _ir(project_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "project_id": project_id,
        "source_file": "model.py",
        "entry_class": "Net",
        "task_type": "classification",
        "input_spec": {"shape": [1, 8], "dtype": "float32"},
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None, "module_path": ""},
            {"id": "fc1", "kind": "leaf", "class_name": "nn.Linear", "module_path": "fc1",
             "parent_id": "net", "params": {"in_features": 8, "out_features": 8}},
            {"id": "relu", "kind": "leaf", "class_name": "nn.ReLU", "module_path": "relu",
             "parent_id": "net"},
            {"id": "fc2", "kind": "leaf", "class_name": "nn.Linear", "module_path": "fc2",
             "parent_id": "net", "params": {"in_features": 8, "out_features": 4}},
            {"id": "addn", "kind": "op", "class_name": "add", "parent_id": "net"},
        ],
        "edges": [
            {"from": "fc1", "to": "relu"},
            {"from": "relu", "to": "addn"},
            {"from": "fc2", "to": "addn"},
        ],
    }


def _reports(project_id: str) -> Path:
    ws = Path(project_manager.get_project(project_id)["workspace_path"])
    d = ws / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_ir(project_id: str, ir: dict) -> None:
    (_reports(project_id) / "ir.json").write_text(
        json.dumps(ir, ensure_ascii=False), encoding="utf-8")


def _write_verification(project_id: str, ir: dict) -> None:
    (_reports(project_id) / "verification.json").write_text(json.dumps({
        "schema_version": "1.0", "overall": "passed", "ir_hash": ir_hash(ir),
        "structure": {"passed": True}, "numeric": {"passed": True, "per_seed": []},
    }, ensure_ascii=False), encoding="utf-8")


@pytest.fixture()
def project_id(ir_env):
    pid = ir_env["project_id"]
    _write_ir(pid, _ir(pid))
    return pid


def _node_ids(pid: str) -> set[str]:
    return {n["id"] for n in decompose_service.read_ir(pid)["nodes"]}


def _edges(pid: str) -> list[tuple[str, str]]:
    return [(e["from"], e["to"]) for e in decompose_service.read_ir(pid)["edges"]]


# ---------------------------------------------------------------- 新增节点

def test_add_node_appends_and_returns_validation(project_id):
    res = decompose_service.add_node(project_id, {
        "id": "drop", "kind": "leaf", "class_name": "nn.Dropout", "parent_id": "net"})
    assert res["node"]["id"] == "drop"
    assert res["errors"] == []          # 白名单叶子、合法结构 → 无校验错误
    assert "drop" in _node_ids(project_id)


def test_add_node_marks_verification_stale(project_id):
    _write_verification(project_id, decompose_service.read_ir(project_id))
    assert decompose_service.verification_status(project_id)[0] == "valid"

    decompose_service.add_node(project_id, {
        "id": "drop", "kind": "leaf", "class_name": "nn.Dropout", "parent_id": "net"})

    assert decompose_service.verification_status(project_id)[0] == "stale"


@pytest.mark.parametrize("bad", [
    {"id": "fc1", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net"},   # id 重复
    {"id": "1bad", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net"},  # 非法标识符
    {"id": "ok", "kind": "nope", "class_name": "nn.ReLU", "parent_id": "net"},    # kind 非法
    {"id": "ok", "kind": "leaf", "class_name": "", "parent_id": "net"},           # 空类名
    {"id": "ok", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "ghost"},  # 父不存在
])
def test_add_node_rejects_hard_errors(project_id, bad):
    with pytest.raises(ValueError):
        decompose_service.add_node(project_id, bad)
    assert "ok" not in _node_ids(project_id)


# ---------------------------------------------------------------- 改节点

def test_update_node_changes_kind_and_class(project_id):
    """把叶子改成自描述算子（人工配方的典型一步：leaf → op + code_hint）。"""
    res = decompose_service.update_node(project_id, "relu", {
        "kind": "op", "class_name": "cosine_similarity",
        "code_hint": "F.cosine_similarity({inputs}, {inputs}, dim=-1)"})
    node = res["node"]
    assert node["kind"] == "op" and node["class_name"] == "cosine_similarity"
    assert node["code_hint"].startswith("F.cosine_similarity")


def test_update_node_reparents(project_id):
    decompose_service.update_node(project_id, "fc1", {"parent_id": None})
    by_id = {n["id"]: n for n in decompose_service.read_ir(project_id)["nodes"]}
    assert by_id["fc1"]["parent_id"] is None


def test_update_node_rejects_unknown_field_and_id(project_id):
    with pytest.raises(ValueError):
        decompose_service.update_node(project_id, "fc1", {"id": "renamed"})
    with pytest.raises(ValueError):
        decompose_service.update_node(project_id, "fc1", {"bogus": 1})


def test_update_node_rejects_parent_cycle(project_id):
    # net 是 fc1 的祖先；把 net 的 parent 指到 fc1 会成环
    with pytest.raises(ValueError):
        decompose_service.update_node(project_id, "net", {"parent_id": "fc1"})


def test_update_node_rejects_unknown_node(project_id):
    with pytest.raises(LookupError):
        decompose_service.update_node(project_id, "ghost", {"params": {}})


def test_update_node_requires_ir(ir_env):
    with pytest.raises(LookupError):
        decompose_service.update_node(ir_env["project_id"], "fc1", {"params": {}})


# ---------------------------------------------------------------- 删节点

def test_delete_node_removes_node_and_incident_edges(project_id):
    res = decompose_service.delete_node(project_id, "fc2")
    assert res["deleted"] == ["fc2"]
    assert "fc2" not in _node_ids(project_id)
    assert ("fc2", "addn") not in _edges(project_id)      # 关联边一并删除


def test_delete_node_rejects_root(project_id):
    with pytest.raises(ValueError):
        decompose_service.delete_node(project_id, "net")


def test_delete_node_rejects_children_unless_recursive(project_id):
    decompose_service.add_node(project_id, {
        "id": "sub", "kind": "module", "class_name": "Sub", "parent_id": "net"})
    decompose_service.add_node(project_id, {
        "id": "s1", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "sub"})

    with pytest.raises(ValueError):
        decompose_service.delete_node(project_id, "sub")
    assert {"sub", "s1"} <= _node_ids(project_id)

    res = decompose_service.delete_node(project_id, "sub", recursive=True)
    assert set(res["deleted"]) == {"sub", "s1"}
    assert "s1" not in _node_ids(project_id)


def test_delete_node_rejects_unknown(project_id):
    with pytest.raises(LookupError):
        decompose_service.delete_node(project_id, "ghost")


# ---------------------------------------------------------------- 增删边

def test_add_edge_and_delete_edge(project_id):
    res = decompose_service.add_edge(project_id, "fc2", "relu")
    assert res["edge"] == {"from": "fc2", "to": "relu"}
    assert ("fc2", "relu") in _edges(project_id)

    res = decompose_service.delete_edge(project_id, "fc2", "relu")
    assert res["deleted"] == 1
    assert ("fc2", "relu") not in _edges(project_id)


def test_add_edge_with_tensor_shape(project_id):
    decompose_service.add_edge(project_id, "fc2", "relu", [1, 4])
    edge = [e for e in decompose_service.read_ir(project_id)["edges"]
            if e["from"] == "fc2" and e["to"] == "relu"][0]
    assert edge["tensor_shape"] == [1, 4]


@pytest.mark.parametrize("a,b", [
    ("fc1", "fc1"),      # 自环
    ("fc1", "relu"),     # 重边
    ("ghost", "relu"),   # from 不存在
    ("fc1", "ghost"),    # to 不存在
    ("addn", "fc1"),     # 成环（fc1 → relu → addn 已存在）
])
def test_add_edge_rejects_hard_errors(project_id, a, b):
    before = _edges(project_id)
    with pytest.raises(ValueError):
        decompose_service.add_edge(project_id, a, b)
    assert _edges(project_id) == before


def test_delete_edge_rejects_missing(project_id):
    with pytest.raises(LookupError):
        decompose_service.delete_edge(project_id, "fc2", "relu")


# ---------------------------------------------------------------- 软错误回传 + 闸门

def test_structural_edit_reports_soft_errors(project_id):
    """删掉 addn 的一条入边 → op 只剩一条入边：写入成功，但校验结果带出来（供界面提示）。"""
    res = decompose_service.delete_edge(project_id, "relu", "addn")
    assert res["errors"], "应回传校验错误"
    assert any("addn" in e for e in res["errors"])


def test_structural_edit_rejected_for_non_original(ir_env):
    structured = project_manager.create_project("structured", source="canvas")
    with pytest.raises(PermissionError):
        decompose_service.add_node(structured, {
            "id": "x", "kind": "leaf", "class_name": "nn.ReLU"})


# ---------------------------------------------------------------- API 契约

@pytest.fixture()
def api_env(app_client, tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    monkeypatch.setattr(decompose_service, "MODULES_DIR", modules_dir)
    pid = project_manager.create_project("original", source="local-test")
    _write_ir(pid, _ir(pid))
    return {"client": app_client, "project_id": pid}


def test_api_add_update_delete_node(api_env):
    client, pid = api_env["client"], api_env["project_id"]

    r = client.post(f"/api/projects/{pid}/ir/nodes", json={
        "id": "drop", "kind": "leaf", "class_name": "nn.Dropout", "parent_id": "net"})
    assert r.status_code == 200, r.text
    assert r.json()["node"]["id"] == "drop"
    assert "errors" in r.json()

    r = client.put(f"/api/projects/{pid}/ir/nodes/fc1", json={"out_features": 16})
    assert r.status_code == 400, "未知字段应 400"
    r = client.put(f"/api/projects/{pid}/ir/nodes/fc1",
                   json={"params": {"in_features": 8, "out_features": 16}})
    assert r.status_code == 200, r.text
    assert r.json()["node"]["params"]["out_features"] == 16

    r = client.put(f"/api/projects/{pid}/ir/nodes/net", json={})
    assert r.status_code == 400, "空 patch 应 400"

    r = client.delete(f"/api/projects/{pid}/ir/nodes/net")
    assert r.status_code == 400, "删根应 400"

    r = client.delete(f"/api/projects/{pid}/ir/nodes/drop")
    assert r.status_code == 200, r.text


def test_api_edge_endpoints_use_from_alias(api_env):
    client, pid = api_env["client"], api_env["project_id"]

    r = client.post(f"/api/projects/{pid}/ir/edges", json={"from": "fc2", "to": "relu"})
    assert r.status_code == 200, r.text
    assert r.json()["edge"]["from"] == "fc2"

    r = client.post(f"/api/projects/{pid}/ir/edges", json={"from": "fc2", "to": "relu"})
    assert r.status_code == 400, "重边应 400"

    r = client.delete(f"/api/projects/{pid}/ir/edges/fc2/relu")
    assert r.status_code == 200, r.text

    r = client.delete(f"/api/projects/{pid}/ir/edges/fc2/relu")
    assert r.status_code == 404, "删不存在的边应 404"


def test_api_get_ir_includes_validation(api_env):
    client, pid = api_env["client"], api_env["project_id"]
    body = client.get(f"/api/projects/{pid}/ir").json()
    assert body["ir_errors"] == []
