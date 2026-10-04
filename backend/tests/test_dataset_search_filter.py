"""公开数据检索的「按任务类型 / 按数据类型」过滤（需求三.2 缺口）。

覆盖：
1. 已登记数据集按 `dataset_registry.task_type` / `dataset_registry.format` 两列过滤能命中/排除；
2. 下载登记时 format 如实写入（能从扩展名推断就写；推不出/多扩展名就留空并给说明，不编造）；
3. 外部源（Zenodo）按其真实能力下推过滤，做不到的部分在返回条目里带 `filter_note`（mock HTTP，不触网）。
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

from app.services import dataset_service, download_service, knowledge_service as ks


def _no_external(monkeypatch) -> None:
    monkeypatch.setattr(dataset_service, "search_external", lambda *args, **kwargs: [])


def _names(result: dict) -> list[str]:
    return sorted(item["name"] for item in result["local"])


# ---------- 1. 已登记数据集按列过滤 ----------

def test_local_filter_by_task_type_and_format(isolated_db, monkeypatch):
    _no_external(monkeypatch)
    ks.register_dataset({"name": "csv-cls", "source": "zenodo",
                         "task_type": "classification", "format": "csv"})
    ks.register_dataset({"name": "img-cls", "source": "zenodo",
                         "task_type": "classification", "format": "image_dir"})
    ks.register_dataset({"name": "csv-seg", "source": "zenodo",
                         "task_type": "segmentation", "format": "csv"})

    assert _names(dataset_service.search_datasets(format="csv")) == ["csv-cls", "csv-seg"]
    assert _names(dataset_service.search_datasets(task_type="classification")) == ["csv-cls", "img-cls"]
    assert _names(dataset_service.search_datasets(task_type="classification", format="csv")) == ["csv-cls"]
    assert _names(dataset_service.search_datasets(task_type="segmentation", format="image_dir")) == []
    # 大小写不敏感（登记与检索同一口径），且过滤为空时不放宽成「返回全部」
    assert _names(dataset_service.search_datasets(format="CSV")) == ["csv-cls", "csv-seg"]
    assert dataset_service.search_datasets(format="parquet")["local"] == []
    # 关键词也覆盖数据类型列
    assert _names(dataset_service.search_datasets(q="csv")) == ["csv-cls", "csv-seg"]


# ---------- 2. 下载登记时 format 如实写入 ----------

def test_download_registers_inferred_format(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path / "datasets")

    def fake_download(source, source_id, dest):
        dest.mkdir(parents=True, exist_ok=True)
        f = dest / "data.csv"
        f.write_text("a,b\n1,2\n", encoding="utf-8")
        return [f]

    monkeypatch.setattr(download_service, "download_dataset", fake_download)
    info = dataset_service.download_dataset_with_info(
        "zenodo", "23080173", "ds-csv", task_type="classification"
    )
    assert info["format"] == "csv" and info["format_source"] == "extension"
    assert "扩展名" in info["format_note"]

    row = ks.get_item("dataset", info["dataset_id"])
    assert row["format"] == "csv" and row["task_type"] == "classification"
    # 写入的 format 真的能被「按数据类型」检索命中
    assert [d["dataset_id"] for d in ks.find_datasets(format="csv")] == [info["dataset_id"]]
    # 兼容既有调用：download_dataset 仍返回可用的 dataset_id（str 契约不变）
    again = dataset_service.download_dataset("zenodo", "23080173", "ds-csv")
    assert isinstance(again, str) and ks.get_item("dataset", again) is not None


def test_download_format_stays_empty_and_explained_when_not_inferable(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path / "datasets")

    def fake_mixed(source, source_id, dest):
        dest.mkdir(parents=True, exist_ok=True)
        files = []
        for name in ("data.csv", "README.txt"):
            f = dest / name
            f.write_text("x", encoding="utf-8")
            files.append(f)
        return files

    monkeypatch.setattr(download_service, "download_dataset", fake_mixed)
    mixed = dataset_service.download_dataset_with_info("zenodo", "1", "ds-mixed")
    assert mixed["format"] is None and mixed["format_source"] is None
    assert "无法推断" in mixed["format_note"]
    assert ks.get_item("dataset", mixed["dataset_id"])["format"] is None
    assert ks.find_datasets(format="csv") == []  # 没有编造出一个格式

    # 完全没有文件时同样留空并说明
    monkeypatch.setattr(download_service, "download_dataset", lambda s, sid, dest: [])
    empty = dataset_service.download_dataset_with_info("zenodo", "2", "ds-empty")
    assert empty["format"] is None and "未返回" in empty["format_note"]
    assert ks.get_item("dataset", empty["dataset_id"])["format"] is None

    # 调用方显式指定 format：以调用方为准，说明里如实标注来源
    monkeypatch.setattr(download_service, "download_dataset", fake_mixed)
    given = dataset_service.download_dataset_with_info("zenodo", "3", "ds-caller", format="CSV")
    assert given["format"] == "csv" and given["format_source"] == "caller"
    assert ks.get_item("dataset", given["dataset_id"])["format"] == "csv"


# ---------- 3. 外部源：能下推的下推，不能的如实标注 ----------

_ZENODO_PAYLOAD = {
    "hits": {
        "hits": [
            {"id": 1, "metadata": {"title": "CSV dataset"}, "files": [{"key": "a.csv"}],
             "links": {"self_html": "https://zenodo.org/records/1"}},
            {"id": 2, "metadata": {"title": "Image dataset"}, "files": [{"key": "img.png"}],
             "links": {"self_html": "https://zenodo.org/records/2"}},
            {"id": 3, "metadata": {"title": "No file list"},
             "links": {"self_html": "https://zenodo.org/records/3"}},
        ]
    }
}


def test_external_pushes_file_type_and_discloses_task_type_unsupported(monkeypatch):
    captured: dict = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        return json.dumps(_ZENODO_PAYLOAD).encode()

    monkeypatch.setattr(download_service, "_http_get", fake_get)
    hits = dataset_service.search_external("cancer", limit=5, task_type="classification", format="csv")

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["type"] == ["dataset"] and query["file_type"] == ["csv"]  # 真实能力范围内下推
    # 客户端复核：扩展名推不出 csv 的条目被排除；无文件清单的条目如实保留
    assert [h["source_id"] for h in hits] == ["1", "3"]
    assert hits[0]["format"] == "csv"
    note = hits[0]["filter_note"]
    assert "task_type" in note and "未下推" in note          # 外部源不支持：如实标注
    assert "file_type=csv" in note and "客户端复核" in note   # 下推了什么、复核范围多大：写明
    assert "没有文件清单" in hits[1]["filter_note"]


def test_external_without_filters_has_no_note(monkeypatch):
    monkeypatch.setattr(
        download_service, "_http_get",
        lambda url, **kwargs: json.dumps(_ZENODO_PAYLOAD).encode(),
    )
    hits = dataset_service.search_external("cancer")
    # 未请求过滤：不做任何排除，也不附说明（与 bioRxiv 检索同风格）
    assert [h["source_id"] for h in hits] == ["1", "2", "3"]
    assert all("filter_note" not in h for h in hits)
    assert [h["format"] for h in hits] == ["csv", "image_dir", None]


def test_external_format_without_zenodo_mapping_not_pushed_and_disclosed(monkeypatch):
    captured: dict = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        return json.dumps({
            "hits": {"hits": [
                {"id": 9, "metadata": {"title": "FASTA"},
                 "files": [{"key": "seq.fasta"}], "links": {"self_html": "https://zenodo.org/records/9"}},
            ]}
        }).encode()

    monkeypatch.setattr(download_service, "_http_get", fake_get)
    hits = dataset_service.search_external("q", format="fasta")
    assert "file_type" not in parse_qs(urlparse(captured["url"]).query)  # 词表里没有 → 不下推
    assert hits and hits[0]["format"] == "fasta"  # 客户端按扩展名复核后保留
    assert "未下推" in hits[0]["filter_note"]


def test_external_falls_back_when_file_type_pushdown_fails(monkeypatch):
    urls: list[str] = []

    def fake_get(url, **kwargs):
        urls.append(url)
        if "file_type=" in url:
            raise RuntimeError("外部源拒绝了 file_type 参数")
        return json.dumps({
            "hits": {"hits": [
                {"id": 5, "metadata": {"title": "T"},
                 "files": [{"key": "b.csv"}], "links": {"self_html": "https://zenodo.org/records/5"}},
            ]}
        }).encode()

    monkeypatch.setattr(download_service, "_http_get", fake_get)
    hits = dataset_service.search_external("q", format="csv")
    assert len(urls) == 2 and "file_type=" in urls[0] and "file_type=" not in urls[1]
    assert len(hits) == 1
    note = hits[0]["filter_note"]
    assert "已退回不带该参数的检索" in note and "服务端未按格式过滤" in note


def test_external_network_failure_returns_empty(monkeypatch):
    def boom(url, **kwargs):
        raise RuntimeError("网络不可用")

    monkeypatch.setattr(download_service, "_http_get", boom)
    assert dataset_service.search_external("q", format="csv", task_type="classification") == []


# ---------- API 层：过滤参数与 format 来源说明真的透出 ----------

def test_search_api_passes_filters_and_download_api_reports_format(app_client, monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(dataset_service, "search_external", lambda *args, **kwargs: [])
    ks.register_dataset({"name": "ds-csv", "source": "zenodo",
                         "task_type": "classification", "format": "csv"})
    ks.register_dataset({"name": "ds-img", "source": "zenodo",
                         "task_type": "classification", "format": "image_dir"})

    resp = app_client.get("/api/datasets/search", params={"task_type": "classification", "format": "csv"})
    assert resp.status_code == 200
    assert [d["name"] for d in resp.json()["local"]] == ["ds-csv"]

    def fake_download(source, source_id, dest):
        dest.mkdir(parents=True, exist_ok=True)
        f = dest / "t.csv"
        f.write_text("x", encoding="utf-8")
        return [f]

    monkeypatch.setattr(download_service, "download_dataset", fake_download)
    resp = app_client.post("/api/datasets/download", json={
        "source": "zenodo", "source_id": "42", "name": "ds-api", "task_type": "classification",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["format"] == "csv" and body["format_source"] == "extension"
    assert ks.get_item("dataset", body["dataset_id"])["format"] == "csv"

    # 非法 source_id（含穿越写法）仍按 400 拒绝
    resp = app_client.post("/api/datasets/download", json={
        "source": "zenodo", "source_id": "../../etc/passwd", "name": "bad",
    })
    assert resp.status_code == 400
