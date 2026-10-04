"""需求一.1 补齐项的回归：完整克隆、抽址后自动克隆、bioRxiv 全文、补充材料抽取。

网络一律 monkeypatch（`download_service._http_get` / `subprocess.run` / agent `run_sync`），
不触真实外部源；`data/` 相关目录全部指向 tmp_path，不碰真实运行数据。
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import zipfile
from pathlib import Path

from app.db.connection import get_connection
from app.services import download_service, knowledge_service, task_manager

REPO_GOOD = "https://github.com/owner/good"
REPO_BAD = "https://github.com/owner/bad"


# ---------- 一.1-1 完整克隆：默认口径与 Git LFS 告警 ----------

def _fake_git(monkeypatch, *, clone_rc: int = 0, lfs_rc: int = 0, submodule_rc: int = 0) -> list[list[str]]:
    """替换 subprocess.run 记录命令行；按子命令返回不同退出码。"""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "clone" in cmd:
            rc, err = clone_rc, ("" if clone_rc == 0 else "fatal: 模拟 clone 失败")
        elif "submodule" in cmd:
            rc, err = submodule_rc, ("" if submodule_rc == 0 else "fatal: 模拟子模块失败")
        elif "lfs" in cmd:
            rc, err = lfs_rc, ("" if lfs_rc == 0 else "fatal: 模拟 lfs 失败")
        else:
            rc, err = 0, ""
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)

    monkeypatch.setattr(download_service.subprocess, "run", fake_run)
    return calls


def test_clone_repo_full_default_uses_submodules_and_no_depth(monkeypatch, tmp_path):
    """默认完整克隆：去掉 --depth 1、带 --recurse-submodules、初始化子模块、拉 LFS。"""
    calls = _fake_git(monkeypatch)
    info = download_service.clone_repo(REPO_GOOD, tmp_path / "repo")

    clone_cmd = calls[0]
    assert clone_cmd[:4] == ["git", "-c", "core.longpaths=true", "clone"]  # 长路径修复保留
    assert "--recurse-submodules" in clone_cmd
    assert "--depth" not in clone_cmd
    assert clone_cmd[-2:] == [REPO_GOOD, str(tmp_path / "repo")]

    assert any("submodule" in c and "--recursive" in c for c in calls)  # 克隆后兜底初始化子模块
    assert any("lfs" in c and "pull" in c for c in calls)
    assert info["full"] is True
    assert info["submodules"] is True
    assert info["lfs"] == "pulled"
    assert info["warnings"] == []


def test_clone_repo_shallow_switch_keeps_depth_one(monkeypatch, tmp_path):
    """浅克隆开关（快速场景/测试）：--depth 1、不取子模块、不拉 LFS。"""
    calls = _fake_git(monkeypatch)
    info = download_service.clone_repo(REPO_GOOD, tmp_path / "repo", full=False)

    clone_cmd = calls[0]
    assert "--depth" in clone_cmd and clone_cmd[clone_cmd.index("--depth") + 1] == "1"
    assert "--recurse-submodules" not in clone_cmd
    assert not any("submodule" in c for c in calls)
    assert not any("lfs" in c for c in calls)
    assert info["full"] is False and info["lfs"] == "skipped" and info["warnings"] == []


def test_clone_repo_warns_when_git_lfs_unavailable(monkeypatch, tmp_path):
    """git-lfs 不可用时必须**如实告警**（返回 warnings），不静默当作完整克隆。"""
    def fake_run(cmd, **kwargs):
        rc = 1 if "lfs" in cmd else 0
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="git: 'lfs' is not a git command")

    monkeypatch.setattr(download_service.subprocess, "run", fake_run)
    monkeypatch.setattr(download_service.shutil, "which", lambda name: None)

    info = download_service.clone_repo(REPO_GOOD, tmp_path / "repo")
    assert info["lfs"] == "unavailable"
    assert len(info["warnings"]) == 1 and "Git LFS 不可用" in info["warnings"][0]
    assert info["full"] is True  # 克隆本身成功，但结果如实标注为不完整


# ---------- 一.1-2 抽址后自动克隆：多仓库全克隆 + 单仓库失败隔离 ----------

def _extract_task() -> str:
    """建一条 extract_addresses 任务（测试里 _queue 为 None，不会真的入队执行）。"""
    return task_manager.create_task(download_service.EXTRACT_TASK_TYPE, params={"paper_text": "正文"})


def _patch_dirs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(download_service, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(download_service, "REPOS_DIR", tmp_path / "repos")
    monkeypatch.setattr(download_service, "DATASETS_DIR", tmp_path / "datasets")


def _fake_download_dataset(source, source_id, dest):
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / "data.csv"
    f.write_text("x", encoding="utf-8")
    return [f]


def test_extract_clones_all_repos_and_isolates_single_failure(isolated_db, monkeypatch, tmp_path):
    """抽到 2 个仓库 + 1 个数据集：两个仓库都尝试克隆，单个失败不中断其余，失败原因进进度与 run_record。"""
    dataset_url = "https://zenodo.org/records/42"

    async def fake_agent(prompt, **kwargs):
        return {"structured_output": {"repositories": [REPO_GOOD, REPO_BAD], "datasets": [dataset_url]}}

    monkeypatch.setattr(download_service.agent_service, "run_sync", fake_agent)
    _patch_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(download_service, "download_dataset", _fake_download_dataset)

    cloned: list[str] = []

    def fake_clone(url, dest, **kwargs):
        cloned.append(url)
        if url == REPO_BAD:
            raise RuntimeError("模拟克隆失败：仓库不可达")
        dest.mkdir(parents=True, exist_ok=True)
        return {"command": f"git clone {url}", "full": True, "submodules": True,
                "lfs": "unavailable", "warnings": ["Git LFS 不可用：LFS 大文件未拉取"]}

    monkeypatch.setattr(download_service, "clone_repo", fake_clone)

    task_id = _extract_task()
    asyncio.run(download_service._run_extract({"paper_text": "正文"}, task_id))

    assert cloned == [REPO_GOOD, REPO_BAD]  # 全部克隆：失败的那个没有中断后一个
    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert progress["repositories"] == [REPO_GOOD, REPO_BAD]
    assert [c["url"] for c in progress["repositories_cloned"]] == [REPO_GOOD]
    assert progress["repositories_cloned"][0]["local_path"] == str(tmp_path / "repos" / "owner__good")
    assert [f["url"] for f in progress["repositories_failed"]] == [REPO_BAD]
    assert "模拟克隆失败" in progress["repositories_failed"][0]["reason"]
    # 克隆期的告警（如 LFS 不可用）也要进进度，不能静默
    assert any("Git LFS 不可用" in w for w in progress["warnings"])

    # 数据集仍按既有口径下载并登记
    assert progress["datasets_registered"][0]["url"] == dataset_url
    assert knowledge_service.get_item("dataset", progress["datasets_registered"][0]["dataset_id"])

    # 可检索记录：成功/失败各一条 run_record（run_type=clone_repo）
    conn = get_connection()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM run_record WHERE run_type = 'clone_repo'"
        ).fetchall()]
    finally:
        conn.close()
    assert len(rows) == 2
    by_status = {r["status"]: r for r in rows}
    assert "模拟克隆失败" in by_status["failed"]["error"]
    assert by_status["success"]["artifact_path"].endswith("owner__good")


def test_extract_clone_switch_off_only_lists_repos(isolated_db, monkeypatch, tmp_path):
    """clone_repos=False：只列出地址不克隆，且如实说明（不伪装成已克隆）。"""
    async def fake_agent(prompt, **kwargs):
        return {"structured_output": {"repositories": [REPO_GOOD], "datasets": []}}

    monkeypatch.setattr(download_service.agent_service, "run_sync", fake_agent)
    _patch_dirs(monkeypatch, tmp_path)
    called: list = []
    monkeypatch.setattr(download_service, "clone_repo", lambda *a, **k: called.append(a))

    task_id = _extract_task()
    asyncio.run(download_service._run_extract({"paper_text": "正文", "clone_repos": False}, task_id))

    assert called == []
    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert progress["repositories"] == [REPO_GOOD]
    assert progress["repositories_cloned"] == [] and progress["repositories_failed"] == []
    assert any("未自动克隆" in w for w in progress["warnings"])


# ---------- 一.1-4 补充材料：进抽取输入 + 地址走同一登记路径 ----------

def test_extract_feeds_supplementary_into_prompt_and_registers_addresses(isolated_db, monkeypatch, tmp_path):
    """论文目录下的 supplementary.md 与 supplementary_data.zip 都要解析进 prompt；
    paper.pdf（正文本体）不算补充材料；抽出的地址仍走原来的克隆/登记路径。"""
    _patch_dirs(monkeypatch, tmp_path)
    paper_dir = tmp_path / "papers" / "p1"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper.pdf").write_bytes("%PDF-1.4 正文本体里的地址".encode("utf-8"))
    (paper_dir / "supplementary.md").write_text(
        "代码可用性：https://github.com/supp/code", encoding="utf-8"
    )
    with zipfile.ZipFile(paper_dir / "supplementary_data.zip", "w") as z:
        z.writestr("tables/notes.txt", "数据可用性：https://zenodo.org/records/99")

    captured: dict = {}

    async def fake_agent(prompt, **kwargs):
        captured["prompt"] = prompt
        return {"structured_output": {
            "repositories": ["https://github.com/supp/code"],
            "datasets": ["https://zenodo.org/records/99"],
        }}

    monkeypatch.setattr(download_service.agent_service, "run_sync", fake_agent)
    monkeypatch.setattr(download_service, "download_dataset", _fake_download_dataset)
    cloned: list[str] = []

    def fake_clone(url, dest, **kwargs):
        cloned.append(url)
        dest.mkdir(parents=True, exist_ok=True)
        return {"command": "git clone", "full": True, "submodules": True, "lfs": "pulled", "warnings": []}

    monkeypatch.setattr(download_service, "clone_repo", fake_clone)

    task_id = _extract_task()
    asyncio.run(download_service._run_extract({"paper_text": "正文", "paper_id": "p1"}, task_id))

    prompt = captured["prompt"]
    assert "--- 补充材料: " in prompt  # 补充材料按来源分段标注（prompt 模板 + 运行期上下文）
    assert "supplementary.md" in prompt and "supplementary_data.zip" in prompt
    assert "https://github.com/supp/code" in prompt  # 文本补充材料进了抽取输入
    assert "https://zenodo.org/records/99" in prompt  # 压缩包内文本也进了抽取输入
    assert "正文本体里的地址" not in prompt  # paper.pdf 不作补充材料重复喂

    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert {Path(i["path"]).name for i in progress["supplementary_used"]} == {
        "supplementary.md", "supplementary_data.zip",
    }
    assert progress["supplementary_failed"] == []
    assert cloned == ["https://github.com/supp/code"]  # 抽出的仓库地址照旧自动克隆
    assert progress["datasets_registered"][0]["url"] == "https://zenodo.org/records/99"


def test_extract_without_supplementary_keeps_previous_behavior(isolated_db, monkeypatch, tmp_path):
    """无补充材料（未给 paper_id/路径）时，行为与改动前一致：只有正文、地址照旧登记。"""
    _patch_dirs(monkeypatch, tmp_path)
    captured: dict = {}

    async def fake_agent(prompt, **kwargs):
        captured["prompt"] = prompt
        return {"structured_output": {
            "repositories": [], "datasets": ["https://zenodo.org/records/5"],
        }}

    monkeypatch.setattr(download_service.agent_service, "run_sync", fake_agent)
    monkeypatch.setattr(download_service, "download_dataset", _fake_download_dataset)

    task_id = _extract_task()
    asyncio.run(download_service._run_extract({"paper_text": "只有正文"}, task_id))

    assert "只有正文" in captured["prompt"] and "--- 补充材料: " not in captured["prompt"]
    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert progress["supplementary_used"] == [] and progress["supplementary_failed"] == []
    assert progress["repositories_cloned"] == [] and progress["repositories_failed"] == []
    assert progress["datasets_registered"][0]["url"] == "https://zenodo.org/records/5"


def test_supplementary_missing_path_and_unsafe_archive_are_recorded(monkeypatch, tmp_path):
    """缺路径如实记为失败；带 `..` 逃逸成员的压缩包拒绝解包（沿用既有解包安全约束）。"""
    used, failed = download_service._collect_supplementary(None, [str(tmp_path / "nope.md")])
    assert used == []
    assert len(failed) == 1 and "不存在" in failed[0]["reason"]

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../evil.txt", "逃逸内容")
    text, reason = download_service._read_supplementary_file(evil)
    assert text is None and reason and "逃逸" in reason


# ---------- 一.1-3 bioRxiv 全文：拿到全文 / 无通路如实降级 ----------

def test_biorxiv_fulltext_via_europepmc(monkeypatch):
    """Europe PMC 收录且开放全文：经 fullTextXML 拿到全文（status=oa_fulltext）。"""
    jats = b"<article><body><sec><p>Preprint full text here.</p></sec></body></article>"

    def fake_get(url, **kwargs):
        if "/search?" in url:
            return json.dumps({"resultList": {"result": [
                {"pmcid": "PMC1234567", "doi": "10.1101/x", "isOpenAccess": "Y"},
            ]}}).encode()
        assert "/PMC1234567/fullTextXML" in url
        return jats

    monkeypatch.setattr(download_service, "_http_get", fake_get)
    info = download_service.fetch_biorxiv_fulltext("10.1101/x", "2")

    assert info["fulltext_status"] == "oa_fulltext"
    assert info["fulltext_via"] == "europepmc"
    assert info["pmcid"] == "PMC1234567"
    assert "Preprint full text here." in info["fulltext"]
    assert info["note"] is None


def test_biorxiv_fulltext_without_oa_path_degrades_honestly(monkeypatch):
    """未收录/非 OA：记 abstract_only + 明确「无开放全文通路」+ 可尝试 URL，不抛错。"""
    def fake_get(url, **kwargs):
        if "/search?" in url:
            return b'{"resultList": {"result": []}}'
        if "esearch.fcgi" in url:
            return b'{"esearchresult": {"idlist": []}}'
        raise AssertionError(f"不应请求: {url}")

    monkeypatch.setattr(download_service, "_http_get", fake_get)
    info = download_service.fetch_biorxiv_fulltext("10.1101/x", "1")

    assert info["fulltext_status"] == "abstract_only" and info["fulltext"] is None
    assert "无开放全文通路" in info["note"]
    assert "Europe PMC 未收录该 DOI" in info["note"]
    assert any(u.endswith("10.1101/xv1.full.pdf") for u in info["fulltext_candidates"])


def test_biorxiv_fulltext_network_error_does_not_raise(monkeypatch):
    """网络异常同样只记原因（不抛错），保证批量检索/下载不被单篇中断。"""
    def boom(url, **kwargs):
        raise RuntimeError("网络不可用")

    monkeypatch.setattr(download_service, "_http_get", boom)
    info = download_service.fetch_biorxiv_fulltext("10.1101/x")

    assert info["fulltext_status"] == "abstract_only"
    assert "网络不可用" in info["note"] and "无开放全文通路" in info["note"]


def test_search_biorxiv_fulltext_flag_attaches_status(monkeypatch):
    """fulltext=True 时逐篇挂全文状态；不传时行为与原来一致（不多发请求）。"""
    payload = json.dumps({"collection": [{
        "doi": "10.1101/x", "title": "T", "abstract": "A",
        "authors": "Jane Doe", "date": "2026-09-01", "version": "2",
    }]}).encode()
    requested: list[str] = []

    def fake_get(url, **kwargs):
        requested.append(url)
        if "api.biorxiv.org" in url:
            return payload
        if "/search?" in url:
            return b'{"resultList": {"result": []}}'
        if "esearch.fcgi" in url:
            return b'{"esearchresult": {"idlist": []}}'
        raise AssertionError(f"不应请求: {url}")

    monkeypatch.setattr(download_service, "_http_get", fake_get)
    papers = download_service.search_biorxiv("t", fulltext=True)
    assert papers[0]["fulltext_status"] == "abstract_only" and papers[0]["note"]
    assert any("europepmc" in u for u in requested)

    requested.clear()
    plain = download_service.search_biorxiv("t")
    assert "fulltext_status" not in plain[0]
    assert not any("europepmc" in u for u in requested)


def test_download_biorxiv_without_pdf_url_records_abstract_only(isolated_db, monkeypatch, tmp_path):
    """source=biorxiv 且无 pdf_url：走全文抓取；抓不到记 abstract_only 并指向预印本站点页。"""
    monkeypatch.setattr(download_service, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(download_service, "fetch_biorxiv_fulltext", lambda doi, version=None: {
        "paper_id": doi, "pmcid": None, "fulltext": None, "fulltext_status": "abstract_only",
        "note": "bioRxiv/medRxiv 无开放全文通路", "fulltext_url": None,
    })

    paper_id = download_service.download_paper("biorxiv-sample-1", source="biorxiv", title="T", abstract="A")
    record = knowledge_service.get_item("paper", paper_id)
    assert record["status"] == "abstract_only"
    assert record["url"] == "https://www.biorxiv.org/content/biorxiv-sample-1"


# ---------- API 契约：向后兼容 + 新可选参数透传 ----------

def test_extract_addresses_task_params_carry_new_options(isolated_db):
    """服务层入口：新参数必须落进任务 params（否则 handler 收不到补充材料/克隆开关）。"""
    task_id = download_service.extract_addresses(
        "正文", "cwd", paper_id="p1", supplementary_paths=["/tmp/s.md"],
        clone_repos=False, full_clone=False,
    )
    row = task_manager.get_task(task_id)
    params = json.loads(row["params"])
    assert row["task_type"] == download_service.EXTRACT_TASK_TYPE
    assert params["paper_text"] == "正文" and params["cwd"] == "cwd"
    assert params["paper_id"] == "p1" and params["supplementary_paths"] == ["/tmp/s.md"]
    assert params["clone_repos"] is False and params["full_clone"] is False


def test_repo_dir_name_keeps_same_name_repos_apart():
    """不同 owner 的同名仓库不能落到同一目录（否则后克隆的会覆盖先克隆的）。"""
    assert download_service._repo_dir_name("https://github.com/a/tool.git") == "a__tool"
    assert download_service._repo_dir_name("https://gitlab.com/b/tool") == "b__tool"
    assert download_service._repo_dir_name("git@github.com:c/tool.git") == "c__tool"


def test_extract_api_is_backward_compatible_and_passes_new_options(app_client, monkeypatch):
    captured: dict = {}

    def fake_extract(paper_text, cwd=None, **kwargs):
        captured.update({"paper_text": paper_text, "cwd": cwd, **kwargs})
        return "task-1"

    monkeypatch.setattr(download_service, "extract_addresses", fake_extract)

    resp = app_client.post("/api/search/extract", json={"paper_text": "正文"})
    assert resp.status_code == 200 and resp.json() == {"task_id": "task-1", "status": "queued"}
    assert captured["paper_id"] is None and captured["supplementary_paths"] is None
    assert captured["clone_repos"] is True and captured["full_clone"] is True

    resp = app_client.post("/api/search/extract", json={
        "paper_text": "正文", "paper_id": "p1", "supplementary_paths": ["/tmp/s.md"],
        "clone_repos": False, "full_clone": False,
    })
    assert resp.status_code == 200
    assert captured["paper_id"] == "p1"
    assert captured["supplementary_paths"] == ["/tmp/s.md"]
    assert captured["clone_repos"] is False and captured["full_clone"] is False
