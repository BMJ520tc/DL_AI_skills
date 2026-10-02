"""版本管理底座（阶段4 4d-1）：git 初始化、保存即提交、运行即提交、失败透出。

工作区目录经 monkeypatch 指向临时路径，测试不触碰真实 data/projects（AGENTS.md 规矩 5）。
git 命令为真实子进程（短平快），提交身份用仓库级配置、不依赖全局 git。
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_versions(app_client, tmp_path, monkeypatch):
    """项目工作区落到临时目录（应用已随 app_client 启动，路径在请求期读取故即时生效）。"""
    from app.services import project_manager

    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    return app_client, tmp_path


def _create_original(client, name: str = "原始项目") -> str:
    r = client.post("/api/projects", json={"project_type": "original", "source": "local-test", "name": name})
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def _create_structured(client, parent_project_id: str | None = None) -> tuple[str, dict]:
    body: dict = {"project_type": "structured", "name": "画布网络"}
    if parent_project_id:
        body["parent_project_id"] = parent_project_id
    r = client.post("/api/projects", json=body)
    assert r.status_code == 200, r.text
    return r.json()["project_id"], r.json()


def _git(ws: Path, *args: str) -> str:
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(ws), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _rmtree_win(path: Path) -> None:
    """Windows 下 .git 对象文件是只读的：先清只读位再删（模拟老项目用）。"""
    import os
    import stat

    def _onerror(func, p, _exc_info):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onerror=_onerror)


def _head_file(ws: Path, filename: str) -> str:
    return _git(ws, "show", f"HEAD:{filename}")


def _log_count(ws: Path) -> int:
    return int(_git(ws, "rev-list", "--count", "HEAD"))


def _chain_graph() -> dict:
    """Linear→ReLU→Linear 级连：入度 0 的 n1 是输入、出度 0 的 n3 是输出。"""
    return {
        "nodes": [
            {"id": "n1", "type": "linear_layer", "data": {"in_features": 4, "out_features": 8}},
            {"id": "n2", "type": "relu_layer", "data": {}},
            {"id": "n3", "type": "linear_layer", "data": {"in_features": 8, "out_features": 2}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2", "targetHandle": "in-0",
             "data": {"label": "out_n1"}},
            {"id": "e2", "source": "n2", "sourceHandle": "out-0", "target": "n3",
             "targetHandle": "in-0", "data": {"label": "out_n2_out-0"}},
        ],
    }


# ---------------------------------------------------------------------------
# 创建即初始化：工作区 .git + 仓库级提交身份 + 空画布初始提交
# ---------------------------------------------------------------------------

def test_structured_creation_inits_git_and_initial_commit(tmp_versions):
    client, tmp_path = tmp_versions
    project_id, resp = _create_structured(client)
    ws = tmp_path / "projects" / project_id

    assert (ws / ".git").exists()
    # 提交身份走仓库级配置（不依赖全局 git）
    assert _git(ws, "config", "user.name") == "DL-AI-skills"
    assert _git(ws, "config", "user.email") == "dl-ai-skills@local"
    # 空画布已形成初始提交，network_version.json 随提交入库
    assert _log_count(ws) == 1
    version = json.loads(_head_file(ws, "network_version.json"))
    assert version["schema_version"] == 1
    assert version["input_spec"] == [] and version["output_spec"] == []
    assert resp["version"]["commit"] is not None
    assert "version_error" not in resp or resp["version_error"] is None


def test_original_projects_get_no_git_repo(tmp_versions):
    """版本仓库只归结构化项目（画布网络）；original 项目工作区不初始化 git。"""
    client, tmp_path = tmp_versions
    project_id = _create_original(client)
    assert not (tmp_path / "projects" / project_id / ".git").exists()


# ---------------------------------------------------------------------------
# 保存即提交：每次 PUT graph 一个新提交，network_version.json 随提交变化
# ---------------------------------------------------------------------------

def test_each_save_creates_commit_with_version_metadata(tmp_versions):
    client, tmp_path = tmp_versions
    project_id, _ = _create_structured(client)
    ws = tmp_path / "projects" / project_id

    r1 = client.put(f"/api/projects/{project_id}/graph", json=_chain_graph())
    assert r1.status_code == 200, r1.text
    assert r1.json()["version"]["commit"] is not None
    assert _log_count(ws) == 2

    # 输入/输出规格按 DAG 推导：无入边为输入、无出边为输出
    version = json.loads(_head_file(ws, "network_version.json"))
    assert [n["node_id"] for n in version["input_spec"]] == ["n1"]
    assert [n["node_id"] for n in version["output_spec"]] == ["n3"]

    # 保存即提交（4d-1 口径）：同图再存也是一次版本动作，network_version.json 随提交变化
    r2 = client.put(f"/api/projects/{project_id}/graph", json=_chain_graph())
    assert r2.json()["version"]["commit"] is not None
    assert _log_count(ws) == 3

    graph = _chain_graph()
    graph["nodes"][2]["data"]["out_features"] = 3
    r3 = client.put(f"/api/projects/{project_id}/graph", json=graph)
    assert r3.json()["version"]["commit"] is not None
    assert _log_count(ws) == 4
    v1 = json.loads(_git(ws, "show", "HEAD~1:network_version.json"))
    v2 = json.loads(_head_file(ws, "network_version.json"))
    assert v2["saved_at"] > v1["saved_at"]
    assert v2["output_spec"][0]["type"] == "linear_layer"


# ---------------------------------------------------------------------------
# 失败透出与懒初始化
# ---------------------------------------------------------------------------

def test_put_graph_version_failure_is_transparent(tmp_versions, monkeypatch):
    """git 失败不连坐保存：图已落盘，version_error 透出在响应里（不静默）。"""
    client, tmp_path = tmp_versions
    project_id, _ = _create_structured(client)

    from app.services import version_service

    def broken_git(ws, *args, **kwargs):
        raise RuntimeError("git 不在 PATH（模拟）")

    monkeypatch.setattr(version_service, "_run_git", broken_git)
    r = client.put(f"/api/projects/{project_id}/graph", json=_chain_graph())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "saved"
    assert body["version"] is None
    assert "git 不在 PATH" in body["version_error"]
    # 图本身已保存成功
    assert client.get(f"/api/projects/{project_id}/graph").json()["nodes"][0]["id"] == "n1"


def test_legacy_project_without_git_lazy_inits_on_save(tmp_versions):
    """4d-1 之前创建的老结构化项目（无 .git）：保存时懒初始化并形成提交。"""
    client, tmp_path = tmp_versions
    project_id, _ = _create_structured(client)
    ws = tmp_path / "projects" / project_id
    _rmtree_win(ws / ".git")  # 模拟老项目
    assert not (ws / ".git").exists()

    r = client.put(f"/api/projects/{project_id}/graph", json=_chain_graph())
    assert r.status_code == 200
    assert r.json()["version"]["commit"] is not None
    assert (ws / ".git").exists()
    assert _log_count(ws) == 1  # 懒初始化 + 首次提交


# ---------------------------------------------------------------------------
# 运行即提交：训练成功后指标摘要入 network_version.json 并提交
# ---------------------------------------------------------------------------

def test_train_run_commits_metrics_summary(tmp_versions, monkeypatch):
    client, tmp_path = tmp_versions
    original_id = _create_original(client)
    project_id, _ = _create_structured(client, original_id)

    # 假环境解释器（_project_python 就绪探测）
    exe = tmp_path / "projects" / original_id / "env" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")

    # 数据集（真实 preprocessed.csv）
    from app.services import knowledge_service as ks

    data_dir = tmp_path / "ds_smoke"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "preprocessed.csv").write_text(
        "id,split,label,input\n1,train,0,0.5\n2,test,1,0.9\n", encoding="utf-8"
    )
    dataset_id = ks.register_dataset({
        "dataset_id": "ds_smoke", "name": "smoke", "task_type": "tabular",
        "local_path": str(data_dir / "preprocessed.csv"),
    })

    assert client.put(f"/api/projects/{project_id}/graph", json=_chain_graph()).status_code == 200

    from app.services import network_service, proc_util

    async def fake_run(cmd, *, cwd, timeout):
        Path(cmd[6]).write_text(
            json.dumps({"metrics": {"loss": 0.1, "accuracy": 0.95}}), encoding="utf-8"
        )
        return 0, "[epoch 1/1] loss=0.100000"

    monkeypatch.setattr(proc_util, "run_command", fake_run)

    params = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "environment_project_id": original_id,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.01,
    }
    asyncio.run(network_service._run_train(params, "task-net-ver"))

    ws = tmp_path / "projects" / project_id
    # 运行即提交：最新提交为「训练运行 …」，run_summary 含指标摘要
    assert "训练运行 task-net-ver" in _git(ws, "log", "-1", "--format=%s")
    version = json.loads(_head_file(ws, "network_version.json"))
    summary = version["run_summary"]
    assert summary["task_id"] == "task-net-ver"
    assert summary["metrics"] == {"loss": 0.1, "accuracy": 0.95}
