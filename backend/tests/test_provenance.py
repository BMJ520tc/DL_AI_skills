"""数据来源溯源（《模块详细设计》5.3、GB-3）。

背景（2026-10-01 审核发现）：外部下载的公开数据集经预处理后来源被丢弃
（`preprocess_service` 固定写 `url=None, source="自带"`），验收时无法追溯数据来自公网，
`public-pub` 这类本地样例与真实公开数据在 registry 里长得一样。
"""
from __future__ import annotations

import json

from app.services import dataset_service, preprocess_service, task_manager


def test_source_url_templates():
    assert dataset_service.source_url("zenodo", "23080173") == "https://zenodo.org/records/23080173"
    assert dataset_service.source_url("figshare", "12345") == "https://figshare.com/articles/12345"
    assert dataset_service.source_url("kaggle", "x") is None
    assert dataset_service.source_url("local", "x") is None


def test_create_preprocess_carries_provenance_into_task_params(isolated_db):
    task_id = preprocess_service.create_preprocess(
        "/tmp/public.csv", "public-real", "classification",
        source="zenodo", url="https://zenodo.org/records/23080173",
    )
    params = json.loads(task_manager.get_task(task_id)["params"])
    assert params["source"] == "zenodo"
    assert params["url"] == "https://zenodo.org/records/23080173"


def test_create_preprocess_defaults_to_self_data(isolated_db):
    """未提供来源时保持既有行为（自带数据，无 url）。"""
    task_id = preprocess_service.create_preprocess("/tmp/self.csv", "self", "classification")
    params = json.loads(task_manager.get_task(task_id)["params"])
    assert params["source"] is None and params["url"] is None


def test_preprocess_api_accepts_provenance(app_client, monkeypatch):
    captured: dict = {}

    def fake_create(input_path, dataset_name=None, task_type="classification", project_id=None,
                    source=None, url=None):
        captured.update({"input_path": input_path, "source": source, "url": url})
        return "task-fake"

    monkeypatch.setattr(preprocess_service, "create_preprocess", fake_create)
    r = app_client.post("/api/preprocess", json={
        "input_path": "/tmp/public.csv", "dataset_name": "public-real",
        "source": "zenodo", "url": "https://zenodo.org/records/23080173",
    })
    assert r.status_code == 200
    assert captured["source"] == "zenodo"
    assert captured["url"] == "https://zenodo.org/records/23080173"
