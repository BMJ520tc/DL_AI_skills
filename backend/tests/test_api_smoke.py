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


def test_knowledge_ingest_module_compound_ref(app_client):
    """ingest 分发 module（模块详细设计 2.6、数据设计十）：复合主键条目经统一入口入库，
    未给版本时按下一版分配，返回复合 ref_id 且该 ref_id 可取回详情。"""
    r = app_client.post("/api/knowledge/ingest", json={
        "data_type": "module",
        "data": {"module_id": "mod-ingest-1", "name": "IngestModule", "task_type": "classification"},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["data_type"] == "module"
    assert body["ref_id"] == "mod-ingest-1:v1"

    # 统一索引里可检索到（types=module）
    hits = app_client.get("/api/knowledge/search", params={"types": "module", "q": "IngestModule"}).json()
    assert [h["ref_id"] for h in hits] == ["mod-ingest-1:v1"]

    # 复合 ref 经 get_item 特判（3.5）可读
    detail = app_client.get("/api/knowledge/items/module/mod-ingest-1:v1")
    assert detail.status_code == 200
    assert detail.json()["name"] == "IngestModule"

    # 第二个同名模块自动进 v2（不与 v1 冲突）
    r2 = app_client.post("/api/knowledge/ingest", json={
        "data_type": "module",
        "data": {"module_id": "mod-ingest-1", "name": "IngestModule2"},
    })
    assert r2.json()["ref_id"] == "mod-ingest-1:v2"


def test_knowledge_draft_confirm_supersede_endpoints(app_client):
    """草稿确认 + 冲突可见 + supersede 端点（模块详细设计 8.3、数据设计六.3）。"""
    from app.services import knowledge_service as ks

    old = ks.record_knowledge({"type": "param_advice", "title": "old", "content": "lr=1e-2",
                               "scope": {"task_type": "classification"}, "status": "confirmed"})
    new = ks.record_knowledge({"type": "param_advice", "title": "new", "content": "lr=1e-3",
                               "scope": {"task_type": "classification"}})

    # 草稿列表按 status 过滤
    drafts = app_client.get("/api/knowledge/list", params={"data_type": "knowledge", "status": "draft"}).json()
    assert [d["knowledge_id"] for d in drafts] == [new]

    # 冲突可见
    conflicts = app_client.get(f"/api/knowledge/conflicts/{new}").json()
    assert [c["knowledge_id"] for c in conflicts] == [old]

    # 确认并推翻旧结论
    r = app_client.post(f"/api/knowledge/confirm/{new}", params={"supersede": True})
    assert r.status_code == 200
    assert ks.get_item("knowledge", old)["status"] == "superseded"

    # 已确认条目可显式 supersede；重复操作 404
    r = app_client.post(f"/api/knowledge/supersede/{new}")
    assert r.status_code == 200
    assert ks.get_item("knowledge", new)["status"] == "superseded"
    assert app_client.post(f"/api/knowledge/supersede/{new}").status_code == 404

    # 未知草稿：conflicts 404
    assert app_client.get("/api/knowledge/conflicts/nope").status_code == 404


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
