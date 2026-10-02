"""检索与下载（模块详细设计 2.4）。

论文检索（arXiv/PubMed/bioRxiv，支持作者与时间范围过滤）、仓库与数据集地址抽取（agent）、
克隆与下载、PubMed PMC OA 全文尽力抓取、Kaggle 凭证校验。
"""
import asyncio
import base64
import http.client
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Optional

from app.config import DATASETS_DIR, PAPERS_DIR
from app.ids import fs_name, safe_id
from app.services import agent_service, knowledge_service, task_manager

ARXIV_API = "https://export.arxiv.org/api/query"
EUTILS_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
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
        except (RuntimeError, OSError, http.client.HTTPException) as e:
            # HTTPException（BadStatusLine/IncompleteRead/UnknownProtocol）不是 OSError，
            # 不捕会让一次畸形响应直接冒泡成 500
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


def _compact_date(value: Optional[str], end: bool) -> Optional[str]:
    """把 YYYY / YYYY-MM / YYYY-MM-DD 规整成 8 位 YYYYMMDD（end 时补齐到期末）。"""
    digits = re.sub(r"\D", "", str(value or ""))
    if not digits:
        return None
    if len(digits) == 4:
        digits += "1231" if end else "0101"
    elif len(digits) == 6:
        digits += "30" if end else "01"
    return digits[:8]


def _arxiv_date(value: Optional[str], end: bool) -> Optional[str]:
    """arXiv submittedDate 需要的 12 位 YYYYMMDDHHMM。"""
    day = _compact_date(value, end)
    return day + ("2359" if end else "0000") if day else None


def _build_arxiv_query(
    query: str, authors: Optional[list[str]], date_from: Optional[str], date_to: Optional[str]
) -> str:
    """拼 arXiv 查询语法：关键词 + ``au:"..."`` + ``submittedDate:[a TO b]``。"""
    terms = [query.strip()] if query and query.strip() else []
    for author in authors or []:
        terms.append(f'au:"{author}"')
    if date_from or date_to:
        start = _arxiv_date(date_from, end=False) or "199107010000"  # arXiv 起始于 1991-07
        finish = _arxiv_date(date_to, end=True) or "999912312359"
        terms.append(f"submittedDate:[{start} TO {finish}]")
    return " AND ".join(terms)


def search_arxiv(
    query: str,
    max_results: int = 10,
    *,
    authors: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict]:
    url = ARXIV_API + "?" + urllib.parse.urlencode(
        {
            "search_query": _build_arxiv_query(query, authors, date_from, date_to),
            "start": 0,
            "max_results": max_results,
        }
    )
    try:
        root = ET.fromstring(_http_get(url, accept="application/atom+xml"))
    except ET.ParseError as e:  # 200 + HTML/错误页也很常见，不能让它冒泡成 500
        raise RuntimeError(f"arXiv 返回内容无法解析: {e}") from e

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
    # core.longpaths=true：Windows 默认 260 字符路径上限，真实仓库常有超长文件名
    # （如 results/per_seed/...--seed42.txt）；不开启会 clone 失败（exit 128）。
    proc = subprocess.run(
        ["git", "-c", "core.longpaths=true", "clone", "--depth", "1", repo_url, str(target_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"git clone 失败（exit {proc.returncode}）: {detail}")


EXTRACT_TASK_TYPE = "extract_addresses"

# 数据集地址 → (来源, 源内 id)，用于下载与登记
_DATASET_URL_PATTERNS = (
    (re.compile(r"zenodo\.org/(?:records?|record)/(\d+)"), "zenodo"),
    (re.compile(r"figshare\.com/articles/(?:[^/]+/)*(\d+)"), "figshare"),
    (re.compile(r"kaggle\.com/datasets/([\w.-]+/[\w.-]+)"), "kaggle"),
)


def extract_addresses(paper_text: str, cwd: Optional[str] = None) -> str:
    """触发地址抽取任务：抽出仓库与数据集地址，并对数据集下载登记（3.2 步骤 3）。"""
    return task_manager.create_task(
        EXTRACT_TASK_TYPE, params={"paper_text": paper_text, "cwd": cwd}
    )


def _as_url(value) -> str:
    """把 agent 输出的一项规整成 URL 字符串。

    结构化输出经 result.json 兜底、不经 schema 校验，条目可能是
    `{"url": ...}` 这类对象而非纯字符串，这里做容错取值。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("url", "link", "href", "address", "dataset"):
            inner = value.get(key)
            if isinstance(inner, str) and inner:
                return inner
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _parse_dataset_url(raw) -> Optional[tuple[str, str]]:
    url = _as_url(raw)
    for pattern, source in _DATASET_URL_PATTERNS:
        m = pattern.search(url)
        if m:
            return source, m.group(1)
    return None


def _register_extracted(datasets: list) -> tuple[list[dict], list[dict]]:
    """下载并登记抽出的数据集；失效地址只记录、不阻断（3.2 异常与边界）。

    只有下载成功的数据集才登记——登记表里放不可用的条目会污染模块三的检索与对齐。
    """
    registered: list[dict] = []
    failed: list[dict] = []
    for raw in datasets:
        url = _as_url(raw)
        parsed = _parse_dataset_url(url)
        if parsed is None:
            failed.append({"url": url, "reason": "无法识别的数据集地址（支持 zenodo/figshare/kaggle）"})
            continue
        source, source_id = parsed
        safe_id(source, "source")
        safe_id(source_id, "source_id")
        dest = DATASETS_DIR / f"{fs_name(source, 'source')}_{fs_name(source_id, 'source_id')}"
        try:
            files = download_dataset(source, source_id, dest)
        except Exception as e:  # noqa: BLE001 —— 单个地址失败不影响其余
            failed.append({"url": url, "reason": str(e)})
            continue
        dataset_id = knowledge_service.register_dataset({
            "name": f"{source}:{source_id}",
            "url": url,
            "source": source,
            "task_type": None,
            "format": None,
            "fields": None,
            "labels": None,
            "alignment": None,
            "local_path": str(dest),
        })
        registered.append({"url": url, "dataset_id": dataset_id, "n_files": len(files)})
    return registered, failed


async def _run_extract(params: dict, task_id: str) -> None:
    prompt = (
        "请从以下论文内容中抽取代码仓库地址（GitHub/GitLab）和数据集地址（Zenodo/Figshare/Kaggle）。\n\n"
        f"{params.get('paper_text', '')[:8000]}\n\n"
        "只输出真实出现的 URL，不要臆造不存在的地址。"
    )
    result = await agent_service.run_sync(
        prompt, cwd=params.get("cwd"), output_schema=ADDRESS_SCHEMA, max_turns=15
    )
    data = result.get("structured_output") or {}
    repositories = [_as_url(item) for item in (data.get("repositories") or [])]
    registered, failed = await asyncio.to_thread(_register_extracted, data.get("datasets") or [])

    # 仓库地址只列出（多仓库论文由用户选择主仓库，3.2 异常与边界）；
    # 克隆由 3.3 任意项目加载在用户指定 source 后执行。
    task_manager.update_progress(task_id, {
        "repositories": repositories,
        "datasets_registered": registered,
        "datasets_failed": failed,
    })


def _pubmed_date(value: Optional[str], end: bool) -> Optional[str]:
    """PubMed [PDAT] 可接受 YYYY 或 YYYY/MM/DD。"""
    digits = re.sub(r"\D", "", str(value or ""))
    if not digits:
        return None
    if len(digits) >= 8:
        return f"{digits[:4]}/{digits[4:6]}/{digits[6:8]}"
    if len(digits) == 6:
        return f"{digits[:4]}/{digits[4:6]}"
    return digits[:4]


def _build_pubmed_term(
    query: str, authors: Optional[list[str]], date_from: Optional[str], date_to: Optional[str]
) -> str:
    """拼 PubMed 检索式：关键词 + ``作者[Author]`` + ``起:止[PDAT]``。"""
    terms = [f"({query})"] if query and query.strip() else []
    for author in authors or []:
        terms.append(f"{author}[Author]")
    start, finish = _pubmed_date(date_from, False), _pubmed_date(date_to, True)
    if start or finish:
        terms.append(f"({start or '1900'}:{finish or '3000'}[PDAT])")
    return " AND ".join(terms) or (query or "")


def _parse_pubmed_articles(body: bytes, ids: list[str]) -> list[dict]:
    """从 efetch XML 解析标题与摘要（esummary 不含摘要，故改用 efetch）。"""
    root = ET.fromstring(body)
    by_id: dict[str, dict] = {}
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
        if not pmid:
            continue
        title_node = art.find(".//Article/ArticleTitle")
        title = " ".join("".join(title_node.itertext()).split()) if title_node is not None else ""
        abstract = " ".join(
            " ".join("".join(node.itertext()).split())
            for node in art.findall(".//Article/Abstract/AbstractText")
        )
        by_id[pmid] = {"paper_id": pmid, "title": title, "abstract": abstract, "source": "pubmed"}
    return [
        by_id.get(pmid, {"paper_id": pmid, "title": "", "abstract": "", "source": "pubmed"})
        for pmid in ids
    ]


def _esearch_pmc_oa(paper_id: str) -> Optional[str]:
    """PubMed ID → PMC 开放获取子集的 PMCID；非 OA 收录返回 None。"""
    term = urllib.parse.quote(f"{paper_id}[pmid] AND open access[filter]")
    body = _http_get(f"{EUTILS_API}/esearch.fcgi?db=pmc&term={term}&retmode=json")
    ids = json.loads(body).get("esearchresult", {}).get("idlist", [])
    return str(ids[0]) if ids else None


def _pmc_fulltext(xml_body: bytes) -> Optional[str]:
    """从 PMC efetch XML 取正文文本；非 OA（error）或无 body 时返回 None。"""
    root = ET.fromstring(xml_body)
    if root.tag.lower() == "error" or root.find(".//error") is not None:
        return None
    body = root.find(".//body")
    if body is None:
        return None
    return " ".join(" ".join(body.itertext()).split()) or None


def fetch_pubmed_fulltext(paper_id: str) -> dict:
    """尽力抓 PubMed 全文（E-utilities + PMC 开放获取子集）。

    非 OA、无全文或网络异常都记「仅摘要/无全文」并正常返回，不抛异常——
    批量下载时单篇取不到全文不应中断其余篇目。
    """
    result = {
        "paper_id": paper_id, "pmcid": None, "fulltext": None,
        "fulltext_status": "abstract_only", "note": None, "fulltext_url": None,
    }
    try:
        pmcid = _esearch_pmc_oa(paper_id)
        if not pmcid:
            result["note"] = "该文献不在 PMC 开放获取（OA）子集，仅摘要/无全文"
            return result
        result["pmcid"] = f"PMC{pmcid}"
        result["fulltext_url"] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
        text = _pmc_fulltext(_http_get(f"{EUTILS_API}/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml"))
        if text:
            result["fulltext"] = text
            result["fulltext_status"] = "oa_fulltext"
        else:
            result["note"] = "PMC 有记录但未取到开放全文，仅摘要/无全文"
    except Exception as e:  # noqa: BLE001 —— 全文抓取失败不阻断调用方
        result["note"] = f"全文抓取失败：{e}（仅摘要/无全文）"
    return result


def search_pubmed(
    query: str,
    max_results: int = 10,
    *,
    authors: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    fulltext: bool = False,
) -> list[dict]:
    """PubMed 检索：esearch 取 ID、efetch 取标题与摘要；fulltext=True 时逐篇尽力抓 PMC OA 全文。"""
    term = _build_pubmed_term(query, authors, date_from, date_to)
    data = _http_get(
        f"{EUTILS_API}/esearch.fcgi?db=pubmed&term={urllib.parse.quote(term)}"
        f"&retmax={max_results}&retmode=json"
    )
    try:
        ids = json.loads(data).get("esearchresult", {}).get("idlist", [])
    except (json.JSONDecodeError, AttributeError) as e:  # 200 + 非 JSON（限流页等）
        raise RuntimeError(f"PubMed 返回内容无法解析: {e}") from e
    if not ids:
        return []
    papers = _parse_pubmed_articles(
        _http_get(f"{EUTILS_API}/efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"), ids
    )
    if fulltext:
        for paper in papers:
            info = fetch_pubmed_fulltext(paper["paper_id"])
            paper["pmcid"] = info.get("pmcid")
            paper["fulltext_status"] = info.get("fulltext_status")
            paper["fulltext_url"] = info.get("fulltext_url")
            if info.get("note"):
                paper["note"] = info["note"]
    return papers


def search_biorxiv(
    query: str,
    max_results: int = 10,
    *,
    authors: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict]:
    """bioRxiv 检索：API 仅支持最近 N 天窗口，关键词/作者/时间都在客户端尽力过滤。

    bioRxiv API 不支持服务端作者/时间过滤，故传入这些过滤时在每条结果里附 filter_note 如实说明。
    """
    url = "https://api.biorxiv.org/details/biorxiv/30d/0"
    try:
        body = _http_get(url)
        data = json.loads(body) if body.strip() else {}
    except (RuntimeError, ValueError):
        # bioRxiv API 对部分网络/地区返回空 body，视为无结果，不阻断
        return []
    needle = (query or "").lower()
    author_filters = [a.lower() for a in (authors or []) if a.strip()]
    d_from = _compact_date(date_from, end=False)
    d_to = _compact_date(date_to, end=True)
    filters_requested = bool(author_filters or d_from or d_to)
    papers = []
    for item in data.get("collection", []):
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        if needle and needle not in (title + " " + abstract).lower():
            continue
        if author_filters and not any(a in (item.get("authors") or "").lower() for a in author_filters):
            continue
        day = re.sub(r"\D", "", str(item.get("date") or ""))[:8]
        if d_from and (not day or day < d_from):
            continue
        if d_to and (not day or day > d_to):
            continue
        paper = {
            "paper_id": item.get("doi", ""), "title": title, "abstract": abstract,
            "source": "biorxiv", "published_date": item.get("date"),
        }
        if filters_requested:
            paper["filter_note"] = (
                "bioRxiv API 不支持服务端作者/时间过滤：这里仅在最近 30 天窗口内做客户端尽力过滤，"
                "结果可能不完整，更早文献请改用 DOI 直接查询。"
            )
        papers.append(paper)
        if len(papers) >= max_results:
            break
    return papers


def download_pdf(pdf_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_http_get(pdf_url))


def download_paper(
    paper_id: str,
    pdf_url: Optional[str] = None,
    *,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    source: Optional[str] = None,
) -> str:
    """下载论文 PDF 并创建 paper 记录 + 统一索引（3.1）。

    - 给了 pdf_url：照旧下载 PDF。
    - 无 pdf_url 且 source=pubmed：改走 E-utilities 尽力抓 PMC OA 全文；
      抓不到全文则记「仅摘要/无全文」（status=abstract_only）后照常落库，不抛异常。
    - 其余情况缺 pdf_url：报错（保持既有语义）。
    """
    # 防路径穿越（id 直接拼进数据目录）；带冒号的 id（zenodo:123）另按 fs_name 改写为目录安全名
    dest_dir = PAPERS_DIR / fs_name(paper_id, "paper_id")
    if pdf_url:
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

    if (source or "").lower() == "pubmed":
        info = fetch_pubmed_fulltext(paper_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "source": source,
            "url": info.get("fulltext_url") or f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
        }
        if info.get("fulltext"):
            md_path = dest_dir / "fulltext.txt"
            md_path.write_text(info["fulltext"], encoding="utf-8")
            record["markdown_path"] = str(md_path)
            record["status"] = "downloaded"
        else:
            record["status"] = "abstract_only"
        return knowledge_service.record_paper(record)

    raise ValueError("缺少 pdf_url：除 PubMed 全文抓取外，下载必须提供 pdf_url")


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


def _kaggle_credentials() -> Optional[tuple[str, str]]:
    """Kaggle 凭证：优先环境变量 KAGGLE_USERNAME/KAGGLE_KEY，其次 ~/.kaggle/kaggle.json。"""
    user, key = os.getenv("KAGGLE_USERNAME"), os.getenv("KAGGLE_KEY")
    if user and key:
        return user, key
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("username") and data.get("key"):
            return str(data["username"]), str(data["key"])
    return None


def download_kaggle_dataset(dataset_ref: str, dest: Path) -> list[Path]:
    """经 Kaggle REST API 下载数据集压缩包并解压（架构九.1 凭证走配置注入）。

    凭证缺失时给出明确文案（需 KAGGLE_USERNAME/KAGGLE_KEY 或 ~/.kaggle/kaggle.json），
    而不是裸 raise 一个无指引的异常。
    """
    creds = _kaggle_credentials()
    if creds is None:
        raise RuntimeError(
            "Kaggle 数据集下载需凭证：请设置环境变量 KAGGLE_USERNAME / KAGGLE_KEY，"
            "或提供 ~/.kaggle/kaggle.json（Kaggle 账户 Settings > API > Create New Token 获取）。"
        )
    ref = dataset_ref.strip().strip("/")
    if ref.count("/") != 1:
        raise ValueError(f"Kaggle 数据集应形如 owner/dataset，实际为: {dataset_ref}")

    token = base64.b64encode(f"{creds[0]}:{creds[1]}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/datasets/download/{ref}",
        headers={"Authorization": f"Basic {token}", "User-Agent": _USER_AGENT},
    )
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
    except Exception as e:  # noqa: BLE001 —— 网络/鉴权失败统一为可读错误
        raise RuntimeError(f"Kaggle 下载失败：{e}（请确认凭证有效且该数据集可访问）") from e

    zip_path = dest / f"{ref.split('/')[1]}.zip"
    zip_path.write_bytes(payload)
    downloaded = [zip_path]
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
            downloaded.extend(p for p in (dest / name for name in z.namelist()) if p.is_file())
    except zipfile.BadZipFile:
        pass  # 非压缩包则保留原始文件
    return downloaded


def download_dataset(source: str, source_id: str, dest: Path) -> list[Path]:
    """按来源下载数据集（需求一.1、2.4）。"""
    if source == "zenodo":
        return download_zenodo_dataset(source_id, dest)
    if source == "figshare":
        return download_figshare_article(source_id, dest)
    if source == "kaggle":
        return download_kaggle_dataset(source_id, dest)
    raise ValueError(f"不支持的数据集来源: {source}（zenodo/figshare/kaggle）")


def register() -> None:
    task_manager.register_handler(EXTRACT_TASK_TYPE, _run_extract)
