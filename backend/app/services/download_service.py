"""检索与下载（模块详细设计 2.4）。

论文检索（arXiv）、仓库与数据集地址抽取（agent）、克隆与下载。
PubMed/bioRxiv 检索为后续接入点，阶段1 仅实现 arXiv。
"""
import http.client
import json
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from app.config import PAPERS_DIR
from app.services import agent_service, knowledge_service

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_USER_AGENT = "DL-AI-skills/0.1 (research tool)"


def _http_get(url: str, timeout: int = 30, accept: str = "*/*", retries: int = 3) -> bytes:
    """GET 请求，带指数退避重试。

    arXiv 会限流：表现为 HTTP 406/429 或 HTTP 200 + 纯文本 "Rate exceeded."。
    请求间退避 3s/6s/12s（arXiv 建议请求间隔 ≥3 秒）。
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path + ("?" + parsed.query if parsed.query else "")
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        conn = http.client.HTTPSConnection(parsed.netloc, timeout=timeout)
        try:
            conn.request("GET", path, headers={"User-Agent": _USER_AGENT, "Accept": accept})
            resp = conn.getresponse()
            body = resp.read()
            if resp.status == 200:
                if body.strip().startswith(b"Rate exceeded"):
                    raise RuntimeError("arxiv rate exceeded")
                return body
            if resp.status in (406, 429, 503):
                raise RuntimeError(f"HTTP {resp.status} {resp.reason} (可能限流)")
            raise RuntimeError(f"HTTP {resp.status} {resp.reason} for {url}")
        except (RuntimeError, OSError) as e:
            last_error = e
        finally:
            conn.close()
        if attempt < retries - 1:
            time.sleep(3 * (2 ** attempt))

    raise last_error if last_error else RuntimeError(f"GET failed: {url}")

ADDRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "repositories": {"type": "array", "items": {"type": "string"}, "description": "代码仓库 URL 列表"},
        "datasets": {"type": "array", "items": {"type": "string"}, "description": "数据集 URL 列表"},
    },
    "required": ["repositories", "datasets"],
}


def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    url = ARXIV_API + "?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": max_results}
    )
    root = ET.fromstring(_http_get(url, accept="application/atom+xml"))

    entries = root.findall("atom:entry", _ATOM_NS)
    # arXiv 陷阱：malformed 参数返回 HTTP 200 + 单条 title="Error" 的 entry
    if len(entries) == 1 and (entries[0].findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip() == "Error":
        return []

    papers = []
    for entry in entries:
        title = " ".join((entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").split())
        raw_id = (entry.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
        pdf_url = None
        for link in entry.findall("atom:link", _ATOM_NS):
            if link.get("rel") == "related" and link.get("type") == "application/pdf":
                pdf_url = link.get("href")
        papers.append(
            {
                "paper_id": raw_id.rstrip("/").rsplit("/", 1)[-1],
                "title": title,
                "abstract": summary,
                "pdf_url": pdf_url,
                "source": "arxiv",
            }
        )
    return papers


def verify_repo(repo_url: str) -> bool:
    try:
        proc = subprocess.run(["git", "ls-remote", repo_url], capture_output=True, text=True, timeout=30)
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def clone_repo(repo_url: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True, capture_output=True)


def extract_addresses(paper_text: str, cwd: Optional[str] = None) -> str:
    prompt = (
        "请从以下论文内容中抽取代码仓库地址（GitHub/GitLab）和数据集地址（Zenodo/Figshare/Kaggle）。\n\n"
        f"{paper_text[:8000]}\n\n"
        "只输出真实出现的 URL，不要臆造不存在的地址。"
    )
    return agent_service.submit(prompt, cwd=cwd, output_schema=ADDRESS_SCHEMA, max_turns=15)


def search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    data = _http_get(
        f"{base}/esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmax={max_results}&retmode=json"
    )
    ids = json.loads(data)["esearchresult"].get("idlist", [])
    if not ids:
        return []
    result = json.loads(
        _http_get(f"{base}/esummary.fcgi?db=pubmed&id={','.join(ids)}&retmode=json")
    ).get("result", {})
    papers = []
    for pmid in ids:
        r = result.get(pmid, {})
        papers.append({"paper_id": pmid, "title": r.get("title", ""), "abstract": "", "source": "pubmed"})
    return papers


def search_biorxiv(query: str, max_results: int = 10) -> list[dict]:
    # bioRxiv API 无关键词搜索，仅支持日期范围/最近 N 天/DOI。拉最近 30 天，客户端按关键词过滤。
    # 日期范围必须窄（1-3 天），否则超时返回空 body。
    url = "https://api.biorxiv.org/details/biorxiv/30d/0"
    try:
        body = _http_get(url)
        data = json.loads(body) if body.strip() else {}
    except (RuntimeError, ValueError):
        # bioRxiv API 对部分网络/地区返回空 body，视为无结果，不阻断
        return []
    papers = []
    q = query.lower()
    for item in data.get("collection", []):
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        if q in (title + " " + abstract).lower():
            papers.append(
                {"paper_id": item.get("doi", ""), "title": title, "abstract": abstract, "source": "biorxiv"}
            )
        if len(papers) >= max_results:
            break
    return papers


def download_pdf(pdf_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_http_get(pdf_url))


def download_paper(
    paper_id: str, pdf_url: str, *, title: Optional[str] = None, abstract: Optional[str] = None, source: Optional[str] = None
) -> str:
    """下载论文 PDF 并创建 paper 记录 + 统一索引（3.1）。"""
    dest_dir = PAPERS_DIR / paper_id
    pdf_path = dest_dir / "paper.pdf"
    download_pdf(pdf_url, pdf_path)
    return knowledge_service.record_paper(
        {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "source": source,
            "url": pdf_url,
            "pdf_path": str(pdf_path),
            "status": "downloaded",
        }
    )


def download_zenodo_dataset(record_id: str, dest: Path) -> list[Path]:
    data = json.loads(_http_get(f"https://zenodo.org/api/records/{record_id}"))
    downloaded = []
    for f in data.get("files", []):
        link = (f.get("links") or {}).get("self")
        if not link:
            continue
        fpath = dest / f.get("key", "file")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(_http_get(link))
        downloaded.append(fpath)
    return downloaded


def download_figshare_article(article_id: str, dest: Path) -> list[Path]:
    data = json.loads(_http_get(f"https://api.figshare.com/v2/articles/{article_id}"))
    downloaded = []
    for f in data.get("files", []):
        link = f.get("download_url")
        if not link:
            continue
        fpath = dest / f.get("name", "file")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(_http_get(link))
        downloaded.append(fpath)
    return downloaded


def download_dataset(source: str, source_id: str, dest: Path) -> list[Path]:
    """按来源下载数据集（需求一.1、2.4）。Kaggle 需 API key，暂不支持。"""
    if source == "zenodo":
        return download_zenodo_dataset(source_id, dest)
    if source == "figshare":
        return download_figshare_article(source_id, dest)
    if source == "kaggle":
        raise RuntimeError("Kaggle 数据集下载需配置 API key，暂未实现")
    raise ValueError(f"不支持的数据集来源: {source}（zenodo/figshare/kaggle）")
