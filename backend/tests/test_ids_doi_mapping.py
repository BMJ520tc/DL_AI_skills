"""DOI 类真实外部 id 的校验与落盘映射（真实 bioRxiv/medRxiv 论文落库缺口）。

覆盖四件事：
1. 含 `/` 的真实 DOI 通过 id 校验，并落成**安全的单层目录名**；
2. 写盘与读回用同一映射（写→读断言路径一致），且 `download_service` 与 `paper_service` 收敛到同一目录；
3. 旧的安全 id 映射结果逐字不变（兼容已在 `data/papers/` 落库的目录）；
4. 恶意 id（穿越/绝对路径/盘符/控制字符/保留设备名）仍被拒绝或安全化，落盘路径不逃出数据根目录。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import ids
from app.services import download_service, knowledge_service as ks, paper_service

DOI = "10.1101/2023.10.03.560734"

# data/papers/ 下真实出现过的目录名即这些 id 的映射结果（旧映射 = 安全 id 原样、`:` 改写成 `_`）
LEGACY_IDS = [
    "2301.12345",
    "p-parse",
    "e2e-paper",
    "scan-paper",
    "test-paper-1",
    "biorxiv-sample-1",
    "zenodo:23080173",
    "pubmed:12345",
    "02175c903ff046e0af17704f06885b4e",
    "10.1101",
    "x",
]

# 必须被拒或被安全化的逃逸写法
HOSTILE_IDS = [
    "../../etc/passwd",
    "..\\x",
    "/etc/passwd",
    "C:/Windows/System32",
    "C:foo",
    "C:",
    "a/../b",
    "10.1101/../../x",
    "10.1101/..",
    "a//b",
    "a/",
    "/a",
    "a/./b",
    "a/\x00b",
    "\x00",
    "a\\b",
]

# 一定被**拒绝**（不做安全化）的那几条
REJECTED_IDS = [
    "../../etc/passwd",
    "..\\x",
    "/etc/passwd",
    "C:/Windows/System32",
    "C:foo",
    "C:",
    "a/../b",
    "10.1101/../../x",
    "a//b",
    "a/",
    "/a",
    "a/\x00b",
    "\x00",
    "a\\b",
]


# ---------- 1. DOI 通过校验并落成安全目录名 ----------

def test_doi_passes_check_and_becomes_single_safe_component():
    assert ids.safe_id(DOI, "paper_id") == DOI  # 业务层放行（原样入库/检索）
    name = ids.fs_name(DOI, "paper_id")
    assert name == "10.1101%2F2023.10.03.560734"
    assert "/" not in name and "\\" not in name and ":" not in name
    assert Path(name).name == name  # 单层目录名：不再含任何路径分隔符
    # DOI 版本后缀等同类形态同样放行
    assert ids.fs_name("10.1101/2023.10.03.560734v2", "paper_id") == "10.1101%2F2023.10.03.560734v2"
    # Kaggle 数据集 id（owner/dataset）也属多段 id
    assert ids.fs_name("owner/dataset", "source_id") == "owner%2Fdataset"


def test_fs_name_is_stable_and_unique_over_sample():
    """同一 id 永远同一映射（稳定），不同 id 不撞名（唯一，`%` 不是合法 id 字符）。"""
    sample = [
        DOI,
        "10.1101/2023.10.03.560734v2",
        "10.1101_2023.10.03.560734",
        "2301.12345",
        "zenodo:23080173",
        "a/b",
        "a_b",
        "a-b",
    ]
    mapped = [ids.fs_name(value, "paper_id") for value in sample]
    assert mapped == [ids.fs_name(value, "paper_id") for value in sample]  # 稳定
    assert len(set(mapped)) == len(sample)  # 唯一


# ---------- 3. 旧 id 映射结果不变（兼容既有数据） ----------

def test_legacy_ids_map_exactly_as_before():
    """旧映射规则 = 安全 id 原样、`:` → `_`；这些 id 的映射结果必须逐字不变。"""

    def legacy(value: str) -> str:
        return value.replace(":", "_")

    for value in LEGACY_IDS:
        assert ids.safe_id(value, "paper_id") == value
        assert ids.fs_name(value, "paper_id") == legacy(value)


# ---------- 4. 逃逸写法仍被拒绝/安全化，且不逃出数据目录 ----------

def test_hostile_ids_are_rejected():
    for raw in REJECTED_IDS:
        with pytest.raises(ValueError):
            ids.safe_id(raw, "paper_id")
        with pytest.raises(ValueError):
            ids.fs_name(raw, "paper_id")


def test_hostile_ids_never_escape_data_root(tmp_path):
    """全量逃逸写法：要么被拒绝，要么安全化后仍落在数据根目录内（单层名字）。"""
    root = (tmp_path / "papers").resolve()
    root.mkdir()
    accepted = 0
    for raw in HOSTILE_IDS + ["CON", "nul.txt", "COM1", "zenodo:23080173", DOI, "owner/dataset"]:
        try:
            name = ids.fs_name(raw, "paper_id")
        except ValueError:
            continue  # 拒绝也是一种合格结果
        accepted += 1
        assert Path(name).name == name  # 安全化后是单层名
        target = (root / name).resolve()
        assert target != root and root in target.parents  # 落在根目录内，且不是根目录本身
    assert accepted >= 6  # 安全化分支确实被执行到（不是「全靠拒绝」蒙混过去）


def test_windows_reserved_names_are_sanitized():
    # 这类 id 的旧映射在 Windows 上根本建不出目录，安全化不损失已落库数据
    assert ids.fs_name("CON", "paper_id") == "_CON"
    assert ids.fs_name("nul.txt", "paper_id") == "_nul.txt"
    assert ids.fs_name("COM1", "paper_id") == "_COM1"
    # 形近的普通 id 不受影响
    assert ids.fs_name("console", "paper_id") == "console"
    assert ids.fs_name("lpt10", "paper_id") == "lpt10"


# ---------- 2. 写盘→读回同一映射（含跨模块收敛） ----------

def test_doi_paper_dir_write_read_same_mapping(isolated_db, monkeypatch, tmp_path):
    root = tmp_path / "papers"
    monkeypatch.setattr(paper_service, "PAPERS_DIR", root)
    monkeypatch.setattr(download_service, "PAPERS_DIR", root)
    ks.record_paper({"paper_id": DOI, "title": "T"})

    written = paper_service._paper_dir(DOI)  # 写：建目录
    (written / "paper.md").write_text("markdown", encoding="utf-8")
    read_back = paper_service._paper_dir(DOI)  # 读：同一映射 → 同一目录

    assert read_back == written
    assert (read_back / "paper.md").read_text(encoding="utf-8") == "markdown"
    assert written == root / "10.1101%2F2023.10.03.560734"
    assert written.parent == root  # 单层：不会多出 10.1101/ 子目录
    assert not (root / "10.1101").exists()


def test_biorxiv_doi_download_and_parse_converge_on_same_dir(isolated_db, monkeypatch, tmp_path):
    """真实 DOI 走 bioRxiv 落库：download_service 落盘目录 == paper_service 读回目录。"""
    root = tmp_path / "papers"
    monkeypatch.setattr(download_service, "PAPERS_DIR", root)
    monkeypatch.setattr(paper_service, "PAPERS_DIR", root)
    monkeypatch.setattr(
        download_service, "fetch_biorxiv_fulltext",
        lambda doi, version=None: {
            "fulltext": "# T\n正文", "fulltext_status": "oa_fulltext", "fulltext_url": None,
        },
    )

    paper_id = download_service.download_paper(DOI, source="biorxiv", title="T", abstract="A")
    assert paper_id == DOI  # 原样落库（不改写 id，只改写目录名）
    record = ks.get_item("paper", DOI)
    assert record["status"] == "downloaded"

    md_path = Path(record["markdown_path"])
    assert md_path.read_text(encoding="utf-8") == "# T\n正文"
    assert md_path.parent == root / "10.1101%2F2023.10.03.560734"
    assert paper_service._paper_dir(DOI) == md_path.parent  # 读回同一映射
    assert root.resolve() in md_path.resolve().parents  # 没逃出数据目录
