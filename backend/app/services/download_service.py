"""检索与下载（模块详细设计 2.4）。

论文检索（arXiv/PubMed/bioRxiv，支持作者与时间范围过滤）、仓库与数据集地址抽取（agent，
正文 + 补充材料一起喂）、**完整克隆**（全历史 + 子模块 + Git LFS 尽力拉取，需求一.1）、
数据集下载、PubMed PMC OA 与 bioRxiv（经 Europe PMC / PMC OA）全文尽力抓取、Kaggle 凭证校验。
"""
import asyncio
import base64
import http.client
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import DATA_DIR, DATASETS_DIR, PAPERS_DIR
from app.ids import fs_name, safe_id
from app.services import agent_service, knowledge_service, prompts, task_manager

ARXIV_API = "https://export.arxiv.org/api/query"
EUTILS_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Europe PMC REST：bioRxiv/medRxiv 预印本的开放全文通路（官方 API 只给元数据，见 fetch_biorxiv_fulltext）
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_USER_AGENT = "DL-AI-skills/0.1 (research tool)"

# 抽址后自动克隆的仓库落地目录（需求一.1）。
# 每个仓库一个子目录；用户可把这些目录当作 3.3「任意项目加载」的本地 source 直接挂载。
REPOS_DIR = DATA_DIR / "repos"

# 补充材料：可当纯文本直读的后缀，以及单份材料喂给 agent 的字符上限
_SUPPLEMENTARY_TEXT_SUFFIXES = (
    ".md", ".markdown", ".txt", ".rst", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
)
_SUPPLEMENTARY_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
SUPPLEMENTARY_MAX_CHARS = 8000

# 压缩包解包安全约束（与 scripts/preprocess_dataset.py 的既有消毒口径一致：条目数 + 解压后总字节）
_ARCHIVE_MAX_MEMBERS = 200_000
_ARCHIVE_MAX_BYTES = 5 * 1024 ** 3  # 5 GiB，防解包炸弹


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _git_lfs_available() -> bool:
    """本机 Git LFS 是否可用：`shutil.which("git-lfs")` 命中，或 `git lfs version` 能跑通。"""
    if shutil.which("git-lfs"):
        return True
    try:
        proc = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def clone_repo(
    repo_url: str, target_dir: Path, *, full: bool = True, with_lfs: bool = True
) -> dict:
    """克隆代码仓库到 target_dir（需求一.1「把代码库完整克隆到本地」）。

    **默认口径 `full=True` 是完整克隆**：
    - 不加 `--depth 1`，取全部提交历史（可切分支、看历史版本、按论文时代复现）；
    - 加 `--recurse-submodules`，克隆后再用 `git submodule update --init --recursive` 兜底
      初始化子模块（按父仓库钉住的 commit 检出，与仓库作者当时的版本一致）；
    - Git LFS：本机 `git-lfs` 可用时执行 `git lfs pull` 拉取 LFS 大文件（权重/数据）；
      不可用时**如实告警**（进返回值 warnings，并由调用方写进任务进度），绝不静默当作完整。
    `full=False` 为浅克隆（`--depth 1`、不取子模块、不拉 LFS），仅供测试与「只想快速看一眼」的
    场景；正式复现一律用默认的完整克隆。

    说明：不使用 `--remote-submodules`——它会让子模块取各自默认分支的最新提交，与父仓库钉住的
    commit 不一致，反而破坏复现要求的版本一致性。

    `-c core.longpaths=true` 保留：Windows 默认 260 字符路径上限，真实仓库常有超长文件名
    （如 results/per_seed/...--seed42.txt），不开启会 clone 失败（exit 128）。

    返回值：{"command", "full", "submodules", "lfs", "warnings"}；
    `lfs` 取值 pulled / failed / unavailable / skipped，`submodules` 表示子模块是否初始化成功。
    克隆本身失败仍抛 RuntimeError（保持既有调用方语义，如 analysis_service.load_source）。
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "-c", "core.longpaths=true", "clone"]
    if full:
        cmd.append("--recurse-submodules")
    else:
        cmd += ["--depth", "1"]
    cmd += [repo_url, str(target_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"git clone 失败（exit {proc.returncode}）: {detail}")

    warnings: list[str] = []
    submodules = False
    if full:
        # --recurse-submodules 在部分 git 版本 / 无凭证场景会静默跳过，这里再显式兜底一次
        sub = subprocess.run(
            ["git", "-c", "core.longpaths=true", "-C", str(target_dir),
             "submodule", "update", "--init", "--recursive"],
            capture_output=True, text=True,
        )
        if sub.returncode == 0:
            submodules = True
        else:
            detail = (sub.stderr or sub.stdout or "").strip()[-400:]
            warnings.append(f"子模块初始化失败（exit {sub.returncode}）：{detail}；仓库内容可能不完整")

    lfs = "skipped"
    if with_lfs and full:
        # 浅克隆（full=False）口径就是「快速看一眼」，不拉 LFS；要 LFS 完整性必须用完整克隆
        if _git_lfs_available():
            pull = subprocess.run(
                ["git", "-C", str(target_dir), "lfs", "pull"], capture_output=True, text=True
            )
            if pull.returncode == 0:
                lfs = "pulled"
            else:
                lfs = "failed"
                detail = (pull.stderr or pull.stdout or "").strip()[-400:]
                warnings.append(f"Git LFS 拉取失败（exit {pull.returncode}）：{detail}；LFS 大文件未取到")
        else:
            lfs = "unavailable"
            warnings.append(
                "Git LFS 不可用（未找到 git-lfs，`git lfs version` 也跑不通）："
                "仓库中经 LFS 管理的大文件（权重/数据）**未拉取**，克隆结果不完整；"
                "请安装 Git LFS 后重新克隆。"
            )

    return {
        "command": " ".join(cmd),
        "full": full,
        "submodules": submodules,
        "lfs": lfs,
        "warnings": warnings,
    }


EXTRACT_TASK_TYPE = "extract_addresses"

# 数据集地址 → (来源, 源内 id)，用于下载与登记
_DATASET_URL_PATTERNS = (
    (re.compile(r"zenodo\.org/(?:records?|record)/(\d+)"), "zenodo"),
    (re.compile(r"figshare\.com/articles/(?:[^/]+/)*(\d+)"), "figshare"),
    (re.compile(r"kaggle\.com/datasets/([\w.-]+/[\w.-]+)"), "kaggle"),
)


def extract_addresses(
    paper_text: str,
    cwd: Optional[str] = None,
    *,
    paper_id: Optional[str] = None,
    supplementary_paths: Optional[list[str]] = None,
    clone_repos: bool = True,
    full_clone: bool = True,
) -> str:
    """触发地址抽取任务（3.2 步骤 3）：从正文 + 补充材料抽仓库/数据集地址，
    数据集下载登记，仓库地址**逐个自动克隆到本地**（需求一.1）。

    参数：
    - `paper_text`：论文正文（必填，保持既有契约）；
    - `paper_id`：论文 id，给了就顺带扫 `data/papers/<id>/` 下的补充材料（supplementary*/PDF/压缩包）；
    - `supplementary_paths`：调用方显式指定的补充材料文件路径，与目录扫描结果合并去重；
    - `clone_repos`：是否在抽到仓库地址后自动完整克隆（默认 True；False 时只列出地址，
      供只想先看地址、不下载大仓库的场景）；
    - `full_clone`：传给 `clone_repo(full=...)`，默认 True 完整克隆（子模块 + LFS）；
      False 为浅克隆，仅供快速场景。
    """
    return task_manager.create_task(
        EXTRACT_TASK_TYPE,
        params={
            "paper_text": paper_text,
            "cwd": cwd,
            "paper_id": paper_id,
            "supplementary_paths": supplementary_paths or [],
            "clone_repos": clone_repos,
            "full_clone": full_clone,
        },
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
        # 登记格式：按下载文件扩展名**如实推断**（推不出留空并写明原因）。此前恒写 None，
        # 导致抽址登记的公开数据集「按数据类型检索」必漏（2026-10-04 完备性核查登记）。
        # 延迟导入：dataset_service 顶层 import 了本模块，函数内导入避免循环。
        from app.services.dataset_service import infer_format

        fmt, fmt_note = infer_format(list(files))
        dataset_id = knowledge_service.register_dataset({
            "name": f"{source}:{source_id}",
            "url": url,
            "source": source,
            "task_type": None,  # 抽址阶段拿不到任务类型，不编造
            "format": fmt,
            "fields": None,
            "labels": None,
            "alignment": None,
            "local_path": str(dest),
        })
        registered.append({"url": url, "dataset_id": dataset_id, "n_files": len(files),
                           "format": fmt, "format_note": fmt_note})
    return registered, failed


# ---------------- 抽出的仓库地址：自动完整克隆（需求一.1） ----------------

def _repo_dir_name(url: str) -> str:
    """从仓库地址推出本地目录名：`owner__repo`（不同 owner 的同名仓库不互相覆盖）。

    只保留文件系统安全字符；非法字符统一改写为 `_`（外部 URL 不可信，不拼原始串进路径）。
    """
    cleaned = (url or "").strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    parts = [p for p in re.split(r"[/:\\]+", cleaned) if p]
    tail = parts[-1] if parts else "repo"
    owner = parts[-2] if len(parts) >= 2 else ""
    name = f"{owner}__{tail}" if owner and owner not in ("https", "http", "ssh", "git") else tail
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "repo"


def _record_clone_run(
    url: str,
    dest: Path,
    task_id: Optional[str],
    status: str,
    *,
    info: Optional[dict] = None,
    error: Optional[str] = None,
    started_at: Optional[str] = None,
) -> None:
    """每次克隆写一条 run_record（可检索记录，数据设计五.4）。

    成功记 artifact_path=本地目录、params 带 URL 与完整/浅克隆口径、指标带告警；
    失败记 error=失败原因。写记录失败不吞异常（数据库有问题要如实失败，不伪造成成功）。
    """
    info = info or {}
    knowledge_service.record_run({
        "run_id": uuid.uuid4().hex,
        "task_id": task_id,
        "run_type": "clone_repo",
        "params": {"url": url, "full": info.get("full", True), "lfs": info.get("lfs")},
        "command": info.get("command"),
        "status": status,
        "error": error,
        "artifact_path": str(dest),
        "metrics": {"warnings": info.get("warnings") or []} if info else None,
        "started_at": started_at,
        "finished_at": _now_iso(),
    })


def _clone_extracted_repos(
    repositories: list[str], *, full: bool = True, task_id: Optional[str] = None
) -> tuple[list[dict], list[dict]]:
    """把抽出的仓库地址逐个克隆到 `data/repos/<owner__repo>`（需求一.1）。

    **多仓库策略：全部克隆**。依据：需求一.1 原文是「抽出地址 →（复数）把代码库完整克隆到本地」，
    设计文档「多仓库论文由用户选择主仓库」约束的是后续复现选哪一个作主仓库，并不排斥其余仓库落地本地；
    全部落地后用户既能直接看到论文涉及的代码全貌，也能任选一个 `data/repos/<dir>` 作为 3.3 的本地 source
    （克隆结果按地址顺序返回，第一个成功克隆的即默认主仓库候选）。

    单个仓库失败**不中断**其余：失败原因进 failed 列表并各记一条 run_record；
    成功/失败都写进任务进度，调用方仍可继续手工指定 source（既有路径不变）。
    """
    cloned: list[dict] = []
    failed: list[dict] = []
    for url in repositories:
        dest = REPOS_DIR / _repo_dir_name(url)
        started_at = _now_iso()
        try:
            REPOS_DIR.mkdir(parents=True, exist_ok=True)
            info = clone_repo(url, dest, full=full)
        except Exception as e:  # noqa: BLE001 —— 单个仓库失败不影响其余地址
            failed.append({"url": url, "reason": str(e)})
            _record_clone_run(url, dest, task_id, "failed", error=str(e), started_at=started_at)
            continue
        cloned.append({
            "url": url,
            "local_path": str(dest),
            "full": info.get("full"),
            "submodules": info.get("submodules"),
            "lfs": info.get("lfs"),
            "warnings": info.get("warnings") or [],
        })
        _record_clone_run(url, dest, task_id, "success", info=info, started_at=started_at)
    return cloned, failed


# ---------------- 补充材料：获取与解析（需求一.1「正文和补充材料里自动抽出地址」） ----------------

def _check_archive_limits(members: list[tuple[str, int]]) -> None:
    """解包前估算规模：条目数与解压后总字节超限即拒绝（与 preprocess_dataset 同口径）。"""
    total = sum(int(size or 0) for _name, size in members)
    if len(members) > _ARCHIVE_MAX_MEMBERS or total > _ARCHIVE_MAX_BYTES:
        raise RuntimeError(
            f"压缩包规模超限（条目 {len(members)}、解压后约 {total / 1024 ** 3:.1f} GiB），已拒绝解包"
        )


def _extract_archive_safely(path: Path, dest: Path) -> None:
    """解包补充材料压缩包，沿用仓库既有解包安全约束。

    - 解包前按条目数/解压后总字节限流（防解包炸弹）；
    - tar 走 stdlib `filter="data"` 消毒：拒绝绝对路径、`..` 逃逸、符号链接与设备文件；
    - zip 逐条校验成员名（绝对路径 / 盘符 / `..` 一律拒绝）后再解包。
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            _check_archive_limits([(i.filename, i.file_size) for i in infos])
            for info in infos:
                name = info.filename
                if name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
                    raise RuntimeError(f"压缩包含绝对路径成员，已拒绝解包: {name}")
                if ".." in re.split(r"[\\/]+", name):
                    raise RuntimeError(f"压缩包含路径逃逸成员（..），已拒绝解包: {name}")
            z.extractall(dest)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            _check_archive_limits([(m.name, m.size) for m in t.getmembers()])
            try:
                # filter="data" 由 stdlib 消毒：拒绝绝对路径、`..` 逃逸、符号链接与设备文件
                t.extractall(dest, filter="data")
            except TypeError:  # pragma: no cover —— 极老 Python 无 filter 参数（当前环境 3.13，有）
                t.extractall(dest)
        return
    raise RuntimeError(f"无法识别的压缩包: {path}")


def _archive_text(path: Path, dest: Path) -> tuple[Optional[str], Optional[str]]:
    """解包压缩包并拼接其中纯文本文件的文本，返回 (文本, 失败原因)。"""
    try:
        _extract_archive_safely(path, dest)
    except Exception as e:  # noqa: BLE001 —— 解包失败如实记录原因，不阻断其余材料
        return None, f"补充材料压缩包解包失败: {e}"
    chunks: list[str] = []
    total = 0
    for item in sorted(dest.rglob("*")):
        if not item.is_file() or item.suffix.lower() not in _SUPPLEMENTARY_TEXT_SUFFIXES:
            continue
        try:
            body = item.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            continue
        chunks.append(f"--- {item.relative_to(dest)} ---\n{body}")
        total += len(body)
        if total >= SUPPLEMENTARY_MAX_CHARS:
            break
    if not chunks:
        return None, "压缩包内没有可读的文本文件（仅二进制/图片等），未取到补充材料文本"
    return "\n".join(chunks)[:SUPPLEMENTARY_MAX_CHARS], None


def _read_supplementary_file(path: Path) -> tuple[Optional[str], Optional[str]]:
    """读一份补充材料文件，返回 (文本, 失败原因)。

    PDF 用仓库既有 PyMuPDF（与模块二 4.1 同一依赖，pymupdf / 旧包名 fitz 兜底）；
    压缩包走 `_archive_text` 的安全解包；其余纯文本后缀直接读。
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pymupdf  # PyMuPDF 新包名
        except ImportError:
            try:
                import fitz as pymupdf  # 旧包名
            except ImportError as e:
                return None, f"PyMuPDF 不可用，无法解析 PDF 补充材料: {e}"
        try:
            with pymupdf.open(str(path)) as doc:
                text = "\n".join((page.get_text() or "") for page in doc).strip()
        except Exception as e:  # noqa: BLE001 —— 解析失败要如实记录原因
            return None, f"PDF 补充材料无法解析: {e}"
        if not text:
            return None, "PDF 补充材料无文本层（扫描版），未取到文本"
        return text[:SUPPLEMENTARY_MAX_CHARS], None

    if suffix in _SUPPLEMENTARY_ARCHIVE_SUFFIXES or suffix == ".gz":
        tmp_dir = Path(tempfile.mkdtemp(prefix="supp_extract_"))
        try:
            return _archive_text(path, tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if suffix in _SUPPLEMENTARY_TEXT_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as e:
            return None, f"补充材料读取失败: {e}"
        if not text:
            return None, "补充材料文件为空"
        return text[:SUPPLEMENTARY_MAX_CHARS], None

    return None, f"不支持的补充材料格式: {suffix or path.name}（支持 PDF/压缩包/纯文本）"


def _supplementary_candidates(
    paper_id: Optional[str], supplementary_paths: Optional[list[str]]
) -> tuple[list[Path], list[dict]]:
    """收集补充材料来源，返回 (存在的文件, 缺失/非法项)。

    数据来源优先级（需求一.1）：
    1. 论文目录 `data/papers/<paper_id>/` 下已有的补充材料：`supplementary*` 以及同目录的
       PDF / 压缩包（**排除正文本体** paper.pdf / fulltext.txt，它们已由正文通路负责）；
    2. 调用方显式给的 `supplementary_paths`（POST /api/search/extract 的可选参数）。
    本仓库没有「期刊页补充材料链接」的稳定公开通路，故不做网络抓取，只按上述本地来源处理
    （宁缺毋滥：抓不到的补充材料不会被伪造成已解析）。
    """
    found: list[Path] = []
    missing: list[dict] = []

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if all(p.resolve() != resolved for p in found):
            found.append(path)

    if paper_id:
        try:
            paper_dir = PAPERS_DIR / fs_name(paper_id, "paper_id")
        except ValueError as e:
            paper_dir = None
            missing.append({"path": str(paper_id), "reason": f"非法 paper_id，未扫描论文目录: {e}"})
        if paper_dir is not None and paper_dir.is_dir():
            for item in sorted(paper_dir.iterdir()):
                if not item.is_file():
                    continue
                low = item.name.lower()
                if low in ("paper.pdf", "fulltext.txt") or low.startswith("paper."):
                    continue  # 正文本体不算补充材料
                if low.startswith("supplementary") or item.suffix.lower() in (
                    ".pdf", ".zip", ".tar", ".gz",
                ):
                    _add(item)

    for raw in supplementary_paths or []:
        path = Path(str(raw))
        if path.is_file():
            _add(path)
        else:
            missing.append({"path": str(raw), "reason": "补充材料路径不存在或不是文件"})

    return found, missing


def _collect_supplementary(
    paper_id: Optional[str], supplementary_paths: Optional[list[str]]
) -> tuple[list[dict], list[dict]]:
    """读取全部补充材料：返回 (取到文本的条目, 失败条目)，失败只记录、不中断。

    条目含 path/chars/text；进度里只放 path 与 chars（正文不进进度 JSON，避免任务表膨胀）。
    """
    used: list[dict] = []
    failed: list[dict] = []
    candidates, missing = _supplementary_candidates(paper_id, supplementary_paths)
    failed.extend(missing)
    for path in candidates:
        text, reason = _read_supplementary_file(path)
        if text is None:
            failed.append({"path": str(path), "reason": reason or "未取到补充材料文本"})
            continue
        used.append({"path": str(path), "chars": len(text), "text": text})
    return used, failed


def _build_extract_prompt(paper_text: str, supplementary: list[dict]) -> str:
    """拼地址抽取 prompt：模板（`agents/prompts/address_extract.md`）+ 本次运行期上下文。

    模板是提示词的**单一事实来源**（角色/任务/约束/字段说明），本函数只负责把本次的
    正文与补充材料分段标注后追加为「运行期上下文」，结构化输出 schema 由代码一并追加
    （避免模板与 schema 漂移）；补充材料与正文同等对待（说明补充材料同样是地址来源）。
    """
    parts = [
        "### 论文正文",
        (paper_text or "（未提供正文）")[:8000],
    ]
    if supplementary:
        parts += [
            "",
            "### 补充材料（以下内容来自论文的补充材料文件，可能包含代码/数据可用性声明；"
            "请与正文同等对待，一并抽取其中真实出现的地址）",
        ]
        for item in supplementary:
            parts.append(f"--- 补充材料: {item['path']} ---")
            parts.append(item["text"])
    return prompts.render_prompt("address_extract", context="\n".join(parts), schema=ADDRESS_SCHEMA)


async def _run_extract(params: dict, task_id: str) -> None:
    """地址抽取任务：正文 + 补充材料 → 抽地址 → 数据集下载登记、仓库逐个完整克隆。"""
    supplementary, supp_failed = await asyncio.to_thread(
        _collect_supplementary,
        params.get("paper_id"),
        params.get("supplementary_paths") or [],
    )
    prompt = _build_extract_prompt(params.get("paper_text") or "", supplementary)
    result = await agent_service.run_sync(
        prompt, cwd=params.get("cwd"), output_schema=ADDRESS_SCHEMA, max_turns=15
    )
    data = result.get("structured_output") or {}
    repositories = [_as_url(item) for item in (data.get("repositories") or [])]
    registered, failed = await asyncio.to_thread(_register_extracted, data.get("datasets") or [])

    clone_repos = bool(params.get("clone_repos", True))
    cloned: list[dict] = []
    clone_failed: list[dict] = []
    if repositories and clone_repos:
        cloned, clone_failed = await asyncio.to_thread(
            _clone_extracted_repos,
            repositories,
            full=bool(params.get("full_clone", True)),
            task_id=task_id,
        )

    warnings: list[str] = []
    if repositories and not clone_repos:
        warnings.append(
            "本次未自动克隆（clone_repos=false）：仓库地址只列出；"
            "可改用默认自动克隆，或在 3.3 任意项目加载时手工指定 source（本地路径或仓库地址）。"
        )
    for item in cloned:
        for warn in item.get("warnings") or []:
            warnings.append(f"仓库 {item['url']}：{warn}")

    # 任务进度是本次抽址的完整台账：抽到的地址、克隆成功/失败、数据集登记/失败、补充材料使用与失败
    task_manager.update_progress(task_id, {
        "repositories": repositories,
        "repositories_cloned": [
            {k: v for k, v in item.items() if k != "warnings"} for item in cloned
        ],
        "repositories_failed": clone_failed,
        "datasets_registered": registered,
        "datasets_failed": failed,
        "supplementary_used": [
            {"path": item["path"], "chars": item["chars"]} for item in supplementary
        ],
        "supplementary_failed": supp_failed,
        "warnings": warnings,
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


def _pmc_search_ids(term: str) -> Optional[str]:
    """PMC esearch：检索式 → 第一个 PMCID（无结果返回 None）。"""
    body = _http_get(f"{EUTILS_API}/esearch.fcgi?db=pmc&term={urllib.parse.quote(term)}&retmode=json")
    ids = json.loads(body).get("esearchresult", {}).get("idlist", [])
    return str(ids[0]) if ids else None


def _esearch_pmc_oa(paper_id: str) -> Optional[str]:
    """PubMed ID → PMC 开放获取子集的 PMCID；非 OA 收录返回 None。"""
    return _pmc_search_ids(f"{paper_id}[pmid] AND open access[filter]")


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


def _apply_fulltext(paper: dict, info: dict) -> None:
    """把全文抓取结果挂到检索条目上（PubMed / bioRxiv 共用口径，字段名一致）。"""
    paper["pmcid"] = info.get("pmcid")
    paper["fulltext_status"] = info.get("fulltext_status")
    paper["fulltext_url"] = info.get("fulltext_url")
    if info.get("note"):
        paper["note"] = info["note"]
    if info.get("fulltext_candidates"):
        paper["fulltext_candidates"] = info["fulltext_candidates"]


def _biorxiv_page_url(doi: str, version: Optional[str] = None) -> str:
    """bioRxiv/medRxiv 预印本在站点上的正文页（人工浏览器可开；脚本访问常被反爬挡住）。"""
    suffix = f"v{version}" if str(version or "").strip() else ""
    return f"https://www.biorxiv.org/content/{doi}{suffix}"


def _europepmc_pmcid(doi: str) -> tuple[Optional[str], Optional[str]]:
    """Europe PMC 按 DOI 查记录，返回 (PMCID, 说明)。

    Europe PMC 收录 PMC 全文与 bioRxiv/medRxiv 预印本；有 PMCID 才说明可能取到开放全文。
    错误形态是 HTTP 200 + `errCode`（无 resultList），这里按结构判断、不按状态码。
    """
    query = urllib.parse.quote(f'DOI:"{doi}"')
    body = _http_get(f"{EUROPEPMC_API}/search?query={query}&format=json&resultType=core&pageSize=1")
    payload = json.loads(body) if body.strip() else {}
    if payload.get("errCode"):
        return None, f"Europe PMC 返回错误（errCode={payload.get('errCode')}: {payload.get('errMsg')}）"
    results = ((payload.get("resultList") or {}).get("result")) or []
    if not results:
        return None, "Europe PMC 未收录该 DOI"
    pmcid = results[0].get("pmcid")
    if not pmcid:
        return None, "Europe PMC 有该预印本记录但未给出 PMCID（未进入 PMC 开放获取子集）"
    return str(pmcid), None


def fetch_biorxiv_fulltext(doi: str, version: Optional[str] = None) -> dict:
    """尽力抓 bioRxiv/medRxiv 预印本全文（需求一.1，容错哲学同 fetch_pubmed_fulltext）。

    现实约束（如实记录，绝不伪造成全文）：
    - bioRxiv/medRxiv 官方 API（api.biorxiv.org/details）**只提供元数据，没有全文接口**；
    - 站点正文/PDF（`content/<doi>vN.full.pdf`）受 Cloudflare 保护，脚本访问常被 403/挑战页挡住，
      因此**不作为可靠通路**，只作为「可尝试的 URL」记进 fulltext_candidates 供人工打开；
    - 可靠通路是 Europe PMC / PMC 开放获取子集：部分预印本已被收录且开放全文，
      其 JATS XML（`{EUROPEPMC_API}/<PMCID>/fullTextXML`）可用；PMC OA 侧再用 E-utilities 兜底一次。
    抓取顺序：Europe PMC（按 DOI）→ PMC OA（按 DOI）→ 如实降级为 abstract_only 并写明原因与可尝试 URL。
    未收录/非 OA/网络异常都**不抛异常**，批量检索与批量下载不被单篇中断。
    """
    result = {
        "paper_id": doi, "pmcid": None, "fulltext": None,
        "fulltext_status": "abstract_only", "note": None, "fulltext_url": None,
        "fulltext_candidates": [], "fulltext_via": None,
    }
    page_url = _biorxiv_page_url(doi, version)
    result["fulltext_candidates"].append(page_url)
    result["fulltext_candidates"].append(f"{page_url}.full.pdf")
    notes: list[str] = []

    # 1) Europe PMC：按 DOI 找 PMCID → 取 JATS 全文 XML
    pmcid_seen: Optional[str] = None
    try:
        pmcid, why = _europepmc_pmcid(doi)
        if pmcid:
            pmcid_seen = pmcid
            xml_url = f"{EUROPEPMC_API}/{pmcid}/fullTextXML"
            result["fulltext_candidates"].append(xml_url)
            text = _pmc_fulltext(_http_get(xml_url))  # JATS/PMC XML 正文都是 .//body，共用解析
            if text:
                result.update({
                    "pmcid": pmcid, "fulltext": text, "fulltext_status": "oa_fulltext",
                    "fulltext_via": "europepmc",
                    "fulltext_url": f"https://europepmc.org/article/PMC/{pmcid}",
                })
                return result
            notes.append(f"Europe PMC 有记录（{pmcid}）但 fullTextXML 未取到开放全文")
        else:
            notes.append(why or "Europe PMC 无可用记录")
    except Exception as e:  # noqa: BLE001 —— 全文抓取失败不阻断调用方
        notes.append(f"Europe PMC 全文抓取失败：{e}")

    # 2) PMC OA 兜底：按 DOI 检索 PMC 开放获取子集（覆盖 Europe PMC 未索引到的 PMC 收录）
    try:
        pmcid = _pmc_search_ids(f'"{doi}"[doi] AND open access[filter]')
        if pmcid and pmcid != pmcid_seen:
            result["pmcid"] = f"PMC{pmcid}"
            xml_url = f"{EUTILS_API}/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml"
            result["fulltext_candidates"].append(xml_url)
            text = _pmc_fulltext(_http_get(xml_url))
            if text:
                result.update({
                    "fulltext": text, "fulltext_status": "oa_fulltext", "fulltext_via": "pmc_oa",
                    "fulltext_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                })
                return result
            notes.append(f"PMC 开放获取子集有记录（PMC{pmcid}）但未取到全文")
        elif not pmcid:
            notes.append("PMC 开放获取（OA）子集未收录该 DOI")
    except Exception as e:  # noqa: BLE001 —— 同上，失败只记录
        notes.append(f"PMC OA 全文抓取失败：{e}")

    result["note"] = (
        "bioRxiv/medRxiv 无开放全文通路：官方 API 只给元数据，站点全文/PDF 受反爬保护、"
        "脚本访问不可靠；本次亦未在 Europe PMC / PMC OA 取到开放全文。"
        f"原因：{'；'.join(notes)}；仅摘要/无全文。可人工尝试的 URL 见 fulltext_candidates。"
    )
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
            _apply_fulltext(paper, fetch_pubmed_fulltext(paper["paper_id"]))
    return papers


def search_biorxiv(
    query: str,
    max_results: int = 10,
    *,
    authors: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    fulltext: bool = False,
) -> list[dict]:
    """bioRxiv 检索：API 仅支持最近 N 天窗口，关键词/作者/时间都在客户端尽力过滤。

    bioRxiv API 不支持服务端作者/时间过滤，故传入这些过滤时在每条结果里附 filter_note 如实说明。
    fulltext=True 时逐篇尽力抓全文（fetch_biorxiv_fulltext，走 Europe PMC / PMC OA）；
    取不到只记 fulltext_status=abstract_only 与原因，不抛错、不中断批量。
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
            "version": item.get("version"),
        }
        if filters_requested:
            paper["filter_note"] = (
                "bioRxiv API 不支持服务端作者/时间过滤：这里仅在最近 30 天窗口内做客户端尽力过滤，"
                "结果可能不完整，更早文献请改用 DOI 直接查询。"
            )
        if fulltext:
            # 逐篇尽力抓全文；取不到只记 abstract_only 与原因（与 PubMed 同一容错口径）
            _apply_fulltext(paper, fetch_biorxiv_fulltext(paper["paper_id"], paper.get("version")))
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
    - 无 pdf_url 且 source=biorxiv：改走 Europe PMC / PMC OA 尽力抓全文（paper_id 传 DOI），
      同样抓不到就记 abstract_only（并在 note 里说明 bioRxiv 无开放全文通路与可尝试 URL）。
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

    if (source or "").lower() == "biorxiv":
        info = fetch_biorxiv_fulltext(paper_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "source": source,
            # 无全文时指向预印本站点正文页（人工可开），不把「抓失败的 XML 地址」当全文地址
            "url": info.get("fulltext_url") or _biorxiv_page_url(paper_id),
        }
        if info.get("fulltext"):
            md_path = dest_dir / "fulltext.txt"
            md_path.write_text(info["fulltext"], encoding="utf-8")
            record["markdown_path"] = str(md_path)
            record["status"] = "downloaded"
        else:
            record["status"] = "abstract_only"
        return knowledge_service.record_paper(record)

    raise ValueError("缺少 pdf_url：除 PubMed / bioRxiv 全文抓取外，下载必须提供 pdf_url")


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
