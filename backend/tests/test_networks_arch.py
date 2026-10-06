"""架构级自迭代建议 API（networks.arch-*）：守卫 400/404、漂移 409、apply 返回新图。

工作区与报告目录经 monkeypatch 指向临时路径；happy 路径直接写报告走纯函数 apply，不启 worker / 不跑 agent。
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture()
def tmp_arch(app_client, tmp_path, monkeypatch):
    from app.services import arch_service, project_manager

    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(arch_service, "ARCH_DIR", tmp_path / "arch_suggest")
    return app_client, tmp_path


def _create_structured(client, name: str = "画布网络") -> str:
    r = client.post("/api/projects", json={"project_type": "structured", "name": name})
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def _create_original(client) -> str:
    r = client.post("/api/projects", json={"project_type": "original", "source": "loc", "name": "原始"})
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def _std_graph() -> dict:
    return {
        "nodes": [
            {"id": "n1", "type": "linear_layer", "data": {"in_features": 4, "out_features": 8, "bias": True}},
            {"id": "n2", "type": "relu_layer", "data": {}},
            {"id": "n3", "type": "linear_layer", "data": {"in_features": 8, "out_features": 2, "bias": True}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "out-0", "target": "n2", "targetHandle": "in-0"},
            {"id": "e2", "source": "n2", "sourceHandle": "out-0", "target": "n3", "targetHandle": "in-0"},
        ],
    }


def _put_graph(client, project_id: str, graph: dict) -> None:
    r = client.put(f"/api/projects/{project_id}/graph", json=graph)
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# 入口守卫
# ---------------------------------------------------------------------------

def test_arch_suggest_unknown_project_404(tmp_arch):
    client, _ = tmp_arch
    assert client.post("/api/networks/nope/arch-suggest", json={}).status_code == 404


def test_arch_suggest_non_structured_400(tmp_arch):
    client, _ = tmp_arch
    pid = _create_original(client)
    assert client.post(f"/api/networks/{pid}/arch-suggest", json={}).status_code == 400


def test_arch_suggest_ir_graph_400(tmp_arch):
    client, _ = tmp_arch
    pid = _create_structured(client)
    _put_graph(client, pid, {"nodes": [{"id": "m", "type": "ir", "data": {"kind": "module"}}], "edges": []})
    r = client.post(f"/api/networks/{pid}/arch-suggest", json={})
    assert r.status_code == 400 and "ir" in r.json()["detail"]


def test_arch_suggest_queues_and_returns_hash(tmp_arch, monkeypatch):
    from app.services import arch_service, task_manager

    client, _ = tmp_arch
    pid = _create_structured(client)
    _put_graph(client, pid, _std_graph())
    monkeypatch.setattr(task_manager, "create_task", lambda *a, **k: "task_fake")
    r = client.post(f"/api/networks/{pid}/arch-suggest", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task_id"] == "task_fake" and body["graph_hash"] == arch_service._graph_hash(_std_graph())


# ---------------------------------------------------------------------------
# 报告读取
# ---------------------------------------------------------------------------

def test_arch_suggestions_no_report_404(tmp_arch):
    client, _ = tmp_arch
    pid = _create_structured(client)
    assert client.get(f"/api/networks/{pid}/arch-suggestions").status_code == 404


def test_arch_apply_missing_report_404(tmp_arch):
    client, _ = tmp_arch
    pid = _create_structured(client)
    _put_graph(client, pid, _std_graph())
    r = client.post(f"/api/networks/{pid}/arch-apply", json={"task_id": "ghost", "suggestion_id": "s"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# apply：happy + 漂移 409 + 非法 400
# ---------------------------------------------------------------------------

def _write_report(client, project_id: str, task_id: str, suggestion: dict, *, stale: bool = False):
    from app.services import arch_service

    cur = client.get(f"/api/projects/{project_id}/graph").json()
    gh = arch_service._graph_hash(cur)
    arch_service._write_report({
        "task_id": task_id, "project_id": project_id, "created_at": "2026-01-01T00:00:00+00:00",
        "graph_hash": ("deadbeef" if stale else gh), "task_type": None, "model": None, "dataset": None,
        "knowledge_used": {}, "module_catalog_size": 0, "suggestions": [suggestion], "agent_error": None,
    })


def _sgo(op, **kw):
    s = {"suggestion_id": "sug1", "op": op, "op_label": op, "target_node_id": kw.pop("target_node_id", "n2"),
         "description": "d", "rationale": "r", "source_knowledge_ids": [], "payload": {}, "valid": True,
         "invalid_reason": None, "source": "agent"}
    s["payload"].update(kw)
    return s


def test_arch_apply_happy_returns_new_graph(tmp_arch):
    client, _ = tmp_arch
    pid = _create_structured(client)
    _put_graph(client, pid, _std_graph())
    _write_report(client, pid, "t1", _sgo("add_layer", anchor_edge_id="e2", new_node_type="relu_layer"))
    r = client.post(f"/api/networks/{pid}/arch-apply", json={"task_id": "t1", "suggestion_id": "sug1"})
    assert r.status_code == 200, r.text
    graph = r.json()["graph"]
    assert len(graph["nodes"]) == 4 and len(graph["edges"]) == 3
    assert all(n["type"] != "ir" for n in graph["nodes"])
    # 不写盘：服务端 graph.json 未变
    assert len(client.get(f"/api/projects/{pid}/graph").json()["nodes"]) == 3


def test_arch_apply_stale_graph_409(tmp_arch):
    client, _ = tmp_arch
    pid = _create_structured(client)
    _put_graph(client, pid, _std_graph())
    _write_report(client, pid, "t2", _sgo("add_layer", anchor_edge_id="e2", new_node_type="relu_layer"),
                  stale=True)
    r = client.post(f"/api/networks/{pid}/arch-apply", json={"task_id": "t2", "suggestion_id": "sug1"})
    assert r.status_code == 409


def test_arch_apply_invalid_suggestion_400(tmp_arch):
    client, _ = tmp_arch
    pid = _create_structured(client)
    _put_graph(client, pid, _std_graph())
    # 锚点边不存在 → apply 抛 ValueError → 400
    _write_report(client, pid, "t3", _sgo("add_layer", anchor_edge_id="ghost", new_node_type="relu_layer"))
    r = client.post(f"/api/networks/{pid}/arch-apply", json={"task_id": "t3", "suggestion_id": "sug1"})
    assert r.status_code == 400


def test_arch_suggestion_report_by_task_id(tmp_arch):
    client, _ = tmp_arch
    pid = _create_structured(client)
    _put_graph(client, pid, _std_graph())
    _write_report(client, pid, "t4", _sgo("add_layer", anchor_edge_id="e1", new_node_type="relu_layer"))
    r = client.get(f"/api/networks/{pid}/arch-suggestions/t4")
    assert r.status_code == 200 and r.json()["task_id"] == "t4"
    assert client.get(f"/api/networks/{pid}/arch-suggestions").json()["task_id"] == "t4"
    assert client.get(f"/api/networks/{pid}/arch-suggestions/ghost").status_code == 404
