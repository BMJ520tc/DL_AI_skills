"""检索过滤、批量下载、PubMed 全文与三维索引（P1~P5 回归）。

网络请求一律 monkeypatch `download_service._http_get`/`urllib`，不触真实外部源。
"""
from __future__ import annotations

import json
import urllib.parse

import pytest

from app.services import download_service, knowledge_service


# ---------- P1：三维索引写入 ----------

def test_run_dimensions_indexed_and_filterable(isolated_db):
    knowledge_service.record_run({
        "run_id": "r-dim",
        "project_id": "proj-1",
        "run_type": "eval",
        "params": {"task_type": "classification", "model": "resnet50", "dataset": "cifar10"},
    })
    hits = knowledge_service.search(
        types=["run"], task_type="classification", model="resnet50", dataset="cifar10"
    )
    assert [h["ref_id"] for h in hits] == ["r-dim"]
    # 维度不匹配不返回
    assert knowledge_service.search(model="vgg16") == []


def test_dimensions_not_invented_when_absent(isolated_db):
    knowledge_service.record_run({"run_id": "r-plain", "project_id": "p", "run_type": "env_install"})
    row = knowledge_service.search(types=["run"])[0]
    assert row["task_type"] is None and row["model_name"] is None and row["dataset_name"] is None
    # 未填维度的条目不会被「按模型检索」命中（不放宽成「为空也返回」）
    assert knowledge_service.search(model="resnet50") == []


def test_knowledge_scope_dimensions_indexed(isolated_db):
    kid = knowledge_service.record_knowledge({
        "type": "usage_guidance",
        "title": "Guidance",
        "content": "content",
        "scope": {"task_type": "segmentation", "model": "unet", "dataset": "cityscapes"},
    })
    hits = knowledge_service.search(task_type="segmentation", model="unet", dataset="cityscapes")
    assert [h["ref_id"] for h in hits] == [kid]


def test_dataset_entry_keeps_task_type_and_name(isolated_db):
    ds_id = knowledge_service.register_dataset({
        "name": "ds-x", "source": "zenodo", "task_type": "detection",
    })
    hits = knowledge_service.search(types=["dataset"], task_type="detection", dataset="ds-x")
    assert [h["ref_id"] for h in hits] == [ds_id]


# ---------- P2：检索过滤参数 ----------

def test_arxiv_query_includes_author_and_date(monkeypatch):
    captured: dict = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        return b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    monkeypatch.setattr(download_service, "_http_get", fake_get)
    papers = download_service.search_arxiv(
        "graphene", authors=["Jane Doe"], date_from="2020-01-01", date_to="2021-12-31"
    )
    assert papers == []
    # 查询串经 urlencode 编码（空格→+、引号→%22）：用 unquote_plus 还原后再断言
    query = urllib.parse.unquote_plus(captured["url"])
    assert 'au:"Jane Doe"' in query
    assert "submittedDate:[202001010000 TO 202112312359]" in query
    assert "graphene" in query


def test_pubmed_term_and_abstract(monkeypatch):
    captured: dict = {}

    def fake_get(url, **kwargs):
        captured.setdefault("urls", []).append(url)
        if "esearch.fcgi" in url:
            return b'{"esearchresult": {"idlist": ["111"]}}'
        return (
            b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>111</PMID>'
            b'<Article><ArticleTitle>A title</ArticleTitle>'
            b'<Abstract><AbstractText>Some abstract.</AbstractText></Abstract>'
            b'</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'
        )

    monkeypatch.setattr(download_service, "_http_get", fake_get)
    papers = download_service.search_pubmed("cancer", authors=["Smith J"], date_from="2020", date_to="2023")
    assert papers[0]["title"] == "A title"
    assert papers[0]["abstract"] == "Some abstract."
    esearch = urllib.parse.unquote(next(u for u in captured["urls"] if "esearch.fcgi" in u))
    assert "Smith J[Author]" in esearch and "2020:2023[PDAT]" in esearch


def test_biorxiv_filter_is_client_side_and_disclosed(monkeypatch):
    payload = json.dumps({
        "collection": [{
            "doi": "10.1/x", "title": "T", "abstract": "A",
            "authors": "Jane Doe; John Roe", "date": "2026-09-01",
        }]
    }).encode()
    monkeypatch.setattr(download_service, "_http_get", lambda url, **kw: payload)

    filtered = download_service.search_biorxiv("t", authors=["jane doe"])
    assert len(filtered) == 1 and "filter_note" in filtered[0]
    # 过滤确实生效（不静默忽略）
    assert download_service.search_biorxiv("t", authors=["nobody"]) == []
    # 未传过滤时不需要说明
    assert "filter_note" not in download_service.search_biorxiv("t")[0]


# ---------- P3：批量下载 ----------

def test_download_batch_partial_failure(app_client, monkeypatch):
    def fake_download(paper_id, pdf_url=None, *, title=None, abstract=None, source=None):
        if paper_id == "bad":
            raise RuntimeError("模拟下载失败")
        return knowledge_service.record_paper({
            "paper_id": paper_id, "title": title, "source": source, "status": "downloaded",
        })

    monkeypatch.setattr(download_service, "download_paper", fake_download)
    resp = app_client.post("/api/search/papers/download-batch", json={"papers": [
        {"paper_id": "ok1", "pdf_url": "http://x/1.pdf"},
        {"paper_id": "bad", "pdf_url": "http://x/2.pdf"},
    ]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2 and body["succeeded"] == 1 and body["failed"] == 1
    by_id = {r["paper_id"]: r for r in body["results"]}
    assert by_id["ok1"]["status"] == "downloaded"
    assert by_id["bad"]["status"] == "failed" and "模拟下载失败" in by_id["bad"]["reason"]


# ---------- P4：PubMed 全文（PMC OA） ----------

def test_pubmed_fulltext_oa(monkeypatch):
    def fake_get(url, **kwargs):
        if "esearch.fcgi" in url:
            return b'{"esearchresult": {"idlist": ["123"]}}'
        return b"<pmc-articleset><article><body><p>Full text here.</p></body></article></pmc-articleset>"

    monkeypatch.setattr(download_service, "_http_get", fake_get)
    info = download_service.fetch_pubmed_fulltext("999")
    assert info["fulltext_status"] == "oa_fulltext"
    assert info["pmcid"] == "PMC123"
    assert "Full text here." in info["fulltext"]


def test_pubmed_fulltext_non_oa_records_abstract_only(monkeypatch):
    monkeypatch.setattr(
        download_service, "_http_get",
        lambda url, **kw: b'{"esearchresult": {"idlist": []}}',
    )
    info = download_service.fetch_pubmed_fulltext("999")
    assert info["fulltext_status"] == "abstract_only"
    assert info["fulltext"] is None and "仅摘要" in info["note"]


def test_pubmed_fulltext_network_error_does_not_raise(monkeypatch):
    def boom(url, **kwargs):
        raise RuntimeError("网络不可用")

    monkeypatch.setattr(download_service, "_http_get", boom)
    info = download_service.fetch_pubmed_fulltext("999")
    assert info["fulltext_status"] == "abstract_only" and info["note"]


# ---------- P5：Kaggle 凭证 ----------

def test_kaggle_credentials_from_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert download_service._kaggle_credentials() == ("u", "k")


def test_kaggle_without_credentials_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(download_service, "_kaggle_credentials", lambda: None)
    with pytest.raises(RuntimeError) as excinfo:
        download_service.download_dataset("kaggle", "owner/dataset", tmp_path / "d")
    message = str(excinfo.value)
    assert "KAGGLE_USERNAME" in message and "KAGGLE_KEY" in message
