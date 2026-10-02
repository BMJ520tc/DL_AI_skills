"""项目管理 API（阶段4 4a）：画布新建结构化项目、画布快照类型守卫、图表文件服务。

工作区目录经 monkeypatch 指向临时路径，测试不触碰真实 data/projects（AGENTS.md 规矩 5）。
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def tmp_projects(app_client, tmp_path, monkeypatch):
    """项目工作区落到临时目录（应用已随 app_client 启动，路径在请求期读取故即时生效）。"""
    from app.services import project_manager

    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    return app_client


def _create_original(client, source: str = "local-test") -> str:
    r = client.post("/api/projects", json={"project_type": "original", "source": source})
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def test_create_structured_initializes_empty_graph(tmp_projects):
    """画布新建模型：结构化项目创建即带空画布快照、状态 ready（阶段4 4a）。"""
    r = tmp_projects.post("/api/projects", json={"project_type": "structured", "name": "新模型"})
    assert r.status_code == 200, r.text
    project_id = r.json()["project_id"]

    p = tmp_projects.get(f"/api/projects/{project_id}").json()
    assert p["status"] == "ready"

    g = tmp_projects.get(f"/api/projects/{project_id}/graph").json()
    assert g == {"nodes": [], "edges": []}

    # 保存一次再读：空项目画布保存链路可用
    body = {"nodes": [{"id": "n1", "type": "linear", "data": {}}], "edges": []}
    assert tmp_projects.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    assert tmp_projects.get(f"/api/projects/{project_id}/graph").json()["nodes"][0]["id"] == "n1"


def test_graph_endpoints_reject_original(tmp_projects):
    """D12 守卫回归：原始项目不可在画布打开/修改（阶段4 4a）。"""
    project_id = _create_original(tmp_projects)

    r = tmp_projects.get(f"/api/projects/{project_id}/graph")
    assert r.status_code == 400
    assert "not allowed" in r.json()["detail"]

    r = tmp_projects.put(f"/api/projects/{project_id}/graph", json={"nodes": [], "edges": []})
    assert r.status_code == 400

    assert tmp_projects.get("/api/projects/not-exist/graph").status_code == 404


def test_create_project_invalid_type(tmp_projects):
    r = tmp_projects.post("/api/projects", json={"project_type": "bogus"})
    assert r.status_code == 400


def test_figures_endpoint_serves_whitelisted_chart(tmp_projects, tmp_path):
    """图表文件服务（模块三 5.5）：白名单图型可打开，其余拒绝。"""
    from app.services import project_manager

    project_id = _create_original(tmp_projects)
    ws = tmp_path / "projects" / project_id
    (ws / "reports" / "figures").mkdir(parents=True)
    (ws / "reports" / "figures" / "performance.html").write_text("<html>性能对比</html>", encoding="utf-8")

    r = tmp_projects.get(f"/api/projects/{project_id}/figures/performance")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "性能对比" in r.text

    # 图型白名单：未知图型与含路径字符的图型一律 404
    assert tmp_projects.get(f"/api/projects/{project_id}/figures/bogus").status_code == 404
    assert tmp_projects.get(f"/api/projects/{project_id}/figures/..%2Fgraph").status_code == 404
    # 图型合法但文件未生成（未跑可视化）→ 404 且文案可读
    r = tmp_projects.get(f"/api/projects/{project_id}/figures/error_dist")
    assert r.status_code == 404
    assert "figure not found" in r.json()["detail"]

    assert tmp_projects.get("/api/projects/not-exist/figures/performance").status_code == 404

    # 工作区目录确实在临时路径，未触碰真实数据目录
    project = project_manager.get_project(project_id)
    assert str(ws) == project["workspace_path"]
