"""API 冒烟：路由挂载 + 关键读接口（阶段1/2/2' 的数据面）。

用 TestClient 走真实 lifespan（含启动收敛），数据库为临时库。
"""
from __future__ import annotations


def _seed():
    from app.services import knowledge_service as ks

    paper_id = ks.record_paper({
        "paper_id": "api-p1",
        "title": "API Paper on Widgets",
        "abstract": "api abstract",
        "source": "local",
    })
    ks.record_experiment_items(paper_id, [{"metric_name": "accuracy", "metric_value_reported": 0.91}])
    knowledge_id = ks.record_knowledge({
        "type": "usage_guidance",
        "title": "API knowledge",
        "content": "api content about widgets",
    })
    ks.register_dataset({"name": "api-dataset", "source": "zenodo", "url": "https://zenodo.org/records/2"})
    return paper_id, knowledge_id


def test_health(app_client):
    assert app_client.get("/api/health").json() == {"status": "ok"}


def test_route_surface_is_mounted(app_client):
    paths = set(app_client.get("/openapi.json").json()["paths"])
    for expected in (
        "/api/health",
        "/api/tasks",
        "/api/projects",
        "/api/knowledge/search",
        "/api/knowledge/items/{data_type}/{ref_id}",
        "/api/datasets/{dataset_id}/alignment/confirm",
        "/api/papers/{paper_id}/reproduce",
        "/api/projects/{project_id}/visualize/{chart_type}",
        "/api/projects/{project_id}/figures/{chart_type}",
    ):
        assert expected in paths, f"路由缺失: {expected}"
    assert len(paths) >= 40


def test_knowledge_read_endpoints(app_client):
    paper_id, knowledge_id = _seed()

    r = app_client.get("/api/knowledge/search", params={"types": "paper", "q": "widgets"})
    assert r.status_code == 200
    assert [x["ref_id"] for x in r.json()] == [paper_id]

    r = app_client.get("/api/knowledge/search", params={"types": "paper,knowledge", "q": "widgets"})
    assert sorted(x["data_type"] for x in r.json()) == ["knowledge", "paper"]

    r = app_client.get("/api/knowledge/list", params={"data_type": "knowledge"})
    assert r.status_code == 200
    assert [x["knowledge_id"] for x in r.json()] == [knowledge_id]

    r = app_client.get(f"/api/knowledge/items/paper/{paper_id}")
    assert r.status_code == 200
    assert r.json()["title"] == "API Paper on Widgets"

    r = app_client.get("/api/knowledge/items/experiment_item/not-exist")
    assert r.status_code == 404

    r = app_client.get("/api/knowledge/items/bad_type/x")
    assert r.status_code == 400


def test_task_endpoints(app_client):
    r = app_client.get("/api/tasks", params={"limit": 10})
    assert r.status_code == 200 and r.json() == []

    r = app_client.post("/api/tasks", json={"task_type": "no-such-handler"})
    assert r.status_code == 200
    task_id = r.json()["task_id"]

    r = app_client.get(f"/api/tasks/{task_id}")
    assert r.status_code == 200

    r = app_client.get("/api/tasks/not-a-task")
    assert r.status_code == 404


def test_cors_allows_local_dev_origin(app_client):
    """前端开发态跨域（GB-5）：Vite dev server 直连后端必须拿到 CORS 头。"""
    r = app_client.get("/api/health", headers={"Origin": "http://127.0.0.1:5199"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:5199"


def test_cors_preflight_allows_local_origin(app_client):
    r = app_client.options("/api/knowledge/search", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
    })
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_does_not_echo_foreign_origin(app_client):
    r = app_client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_lifespan_reconciles_stale_running(isolated_db):
    """启动收敛集成验证（GB-7）：上次进程遗留的 running 在服务启动时被置 failed。"""
    from fastapi.testclient import TestClient

    from app.db.connection import get_connection
    from app.main import app

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO task(task_id, task_type, project_id, params, status, created_at, updated_at) "
            "VALUES ('stale-boot', 'pdf_parse', NULL, '{}', 'running', "
            "'2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00')"
        )
        conn.commit()
    finally:
        conn.close()

    with TestClient(app) as client:
        task = client.get("/api/tasks/stale-boot").json()
        assert task["status"] == "failed"
        assert "重启" in task["error"]
