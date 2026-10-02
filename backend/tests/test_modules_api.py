"""模块列表接口契约（阶段4 4b）：前端模块库注入依赖的字段口径。

`GET /api/modules` 必须原样返回 module 表全列——尤其是 params_schema 与
saved_module_compat 两个 JSON 文本列：前端 `compatToSavedModule` 用前者生成
参数面板（variableSchema），用后者构造只读模块注入（id 含版本号，画布旧引用
不随新版本漂移）。
"""
from __future__ import annotations

import json


def _record_sample_module():
    from app.services import knowledge_service as ks

    ks.record_module({
        "module_id": "mod_api_0001",
        "module_version": "v1",
        "name": "SampleModule",
        "description": "contract module",
        "source_project_id": None,
        "source_paper_id": None,
        "task_type": "vision",
        "input_spec": {"shape": [3, 224, 224]},
        "output_spec": {"shape": [1000]},
        "params_schema": {
            "hidden_size": {"type": "int", "default": 128},
            "dropout": {"type": "float", "default": 0.5},
            "use_bias": {"type": "bool", "default": True},
            "act": {"type": "str", "default": "relu"},
            "norm": {"type": "list", "default": [3, 0.5, 1, 2]},
        },
        "tags": ["vision", "decompose"],
        "verification": {"overall": "passed"},
        "saved_module_compat": {
            "id": "mod_api_0001:v1",
            "name": "SampleModule",
            "version": "v1",
            "description": "contract module",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
            "graph": {"nodes": [], "edges": []},
            "createdAt": "2026-10-03T00:00:00",
            "updatedAt": "2026-10-03T00:00:00",
        },
        "path": "/tmp/not-used",
    })


def test_list_modules_exposes_params_schema_and_compat(app_client):
    _record_sample_module()
    rows = app_client.get("/api/modules").json()
    assert len(rows) == 1
    row = rows[0]

    assert row["module_id"] == "mod_api_0001"
    assert row["module_version"] == "v1"

    # params_schema：列值是 JSON 文本，字段口径 {type, default}
    schema = json.loads(row["params_schema"])
    assert schema["hidden_size"] == {"type": "int", "default": 128}
    assert schema["dropout"] == {"type": "float", "default": 0.5}
    assert schema["use_bias"] == {"type": "bool", "default": True}
    assert schema["act"] == {"type": "str", "default": "relu"}
    assert schema["norm"] == {"type": "list", "default": [3, 0.5, 1, 2]}

    # saved_module_compat：id 含版本号（画布旧引用版本边界），graph/handles 齐备
    compat = json.loads(row["saved_module_compat"])
    assert compat["id"] == "mod_api_0001:v1"
    assert compat["handles"] == {"inputs": ["in"], "outputs": ["out"]}
    assert compat["graph"] == {"nodes": [], "edges": []}


def test_list_modules_excludes_unindexed_leftovers(app_client):
    """入库失败回滚后（delete_module 清理），列表不应再有该模块行。"""
    from app.services import knowledge_service as ks

    ks.record_module({
        "module_id": "mod_api_0002",
        "module_version": "v1",
        "name": "GoneModule",
        "description": None,
        "source_project_id": None,
        "source_paper_id": None,
        "task_type": None,
        "input_spec": None,
        "output_spec": None,
        "params_schema": None,
        "tags": None,
        "verification": None,
        "saved_module_compat": json.dumps({
            "id": "mod_api_0002:v1", "name": "GoneModule", "version": "v1",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
            "graph": {"nodes": [], "edges": []},
        }),
        "path": None,
    })
    ks.delete_module("mod_api_0002", "v1")
    rows = app_client.get("/api/modules").json()
    assert all(r["module_id"] != "mod_api_0002" for r in rows)
