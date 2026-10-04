"""项目管理 API（阶段4 4a）：画布新建结构化项目、画布快照类型守卫、图表文件服务。

另覆盖阶段5 前置修复：任意项目加载成功落态 loaded、加载失败回滚半成品项目，
以及回滚时对「工作区 source 指向用户本机目录的符号链接/联接」的安全删除。

工作区目录经 monkeypatch 指向临时路径，测试不触碰真实 data/projects（AGENTS.md 规矩 5）。
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_projects(app_client, tmp_path, monkeypatch):
    """项目工作区落到临时目录（应用已随 app_client 启动，路径在请求期读取故即时生效）。"""
    from app.services import project_manager

    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    return app_client


def _create_original(client, source: str = "local-test") -> str:
    r = client.post("/api/projects", json={"project_type": "original", "source": source})
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def test_create_structured_initializes_empty_graph(tmp_projects):
    """画布新建模型：结构化项目创建即带空画布快照、状态 ready（阶段4 4a）。"""
    r = tmp_projects.post("/api/projects", json={"project_type": "structured", "name": "新模型"})
    assert r.status_code == 200, r.text
    project_id = r.json()["project_id"]

    p = tmp_projects.get(f"/api/projects/{project_id}").json()
    assert p["status"] == "ready"

    g = tmp_projects.get(f"/api/projects/{project_id}/graph").json()
    assert g == {"nodes": [], "edges": []}

    # 保存一次再读：空项目画布保存链路可用
    body = {"nodes": [{"id": "n1", "type": "linear", "data": {}}], "edges": []}
    assert tmp_projects.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    assert tmp_projects.get(f"/api/projects/{project_id}/graph").json()["nodes"][0]["id"] == "n1"


def test_graph_endpoints_reject_original(tmp_projects):
    """D12 守卫回归：原始项目不可在画布打开/修改（阶段4 4a）。"""
    project_id = _create_original(tmp_projects)

    r = tmp_projects.get(f"/api/projects/{project_id}/graph")
    assert r.status_code == 400
    assert "not allowed" in r.json()["detail"]

    r = tmp_projects.put(f"/api/projects/{project_id}/graph", json={"nodes": [], "edges": []})
    assert r.status_code == 400

    assert tmp_projects.get("/api/projects/not-exist/graph").status_code == 404


def test_create_project_invalid_type(tmp_projects):
    r = tmp_projects.post("/api/projects", json={"project_type": "bogus"})
    assert r.status_code == 400


def test_figures_endpoint_serves_whitelisted_chart(tmp_projects, tmp_path):
    """图表文件服务（模块三 5.5）：白名单图型可打开，其余拒绝。"""
    from app.services import project_manager

    project_id = _create_original(tmp_projects)
    ws = tmp_path / "projects" / project_id
    (ws / "reports" / "figures").mkdir(parents=True)
    (ws / "reports" / "figures" / "performance.html").write_text("<html>性能对比</html>", encoding="utf-8")

    r = tmp_projects.get(f"/api/projects/{project_id}/figures/performance")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "性能对比" in r.text

    # 图型白名单：未知图型与含路径字符的图型一律 404
    assert tmp_projects.get(f"/api/projects/{project_id}/figures/bogus").status_code == 404
    assert tmp_projects.get(f"/api/projects/{project_id}/figures/..%2Fgraph").status_code == 404
    # 图型合法但文件未生成（未跑可视化）→ 404 且文案可读
    r = tmp_projects.get(f"/api/projects/{project_id}/figures/error_dist")
    assert r.status_code == 404
    assert "figure not found" in r.json()["detail"]

    assert tmp_projects.get("/api/projects/not-exist/figures/performance").status_code == 404

    # 工作区目录确实在临时路径，未触碰真实数据目录
    project = project_manager.get_project(project_id)
    assert str(ws) == project["workspace_path"]


# ---------------- 任意项目加载（3.3）：落态与失败回滚 ----------------


def _create_original_with_url(client, source_url: str):
    return client.post(
        "/api/projects",
        json={"project_type": "original", "source": "local-test", "source_url": source_url},
    )


def _project_id_from_detail(detail: str) -> str:
    m = re.search(r"project_id=([0-9a-f]{32})", detail)
    assert m, f"detail 未带 project_id: {detail}"
    return m.group(1)


def _try_dir_link(target: Path, link: Path) -> str | None:
    """在 link 处建指向 target 的目录链接：优先符号链接，无权限时退到 Windows 目录联接。

    两者都不行（非 Windows 且无 symlink 权限）返回 None，由调用方决定 skip。
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return "symlink"
    except OSError:
        pass
    if os.name == "nt":
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if proc.returncode == 0:
            return "junction"
    return None


def _drop_link(link: Path) -> None:
    for fn in (link.rmdir, link.unlink):
        try:
            fn()
            return
        except OSError:
            continue


def test_create_original_with_local_source_marks_loaded(tmp_projects, tmp_path):
    """加载成功：响应与库里的项目状态都推进为 loaded（前端 BUSY_STATUSES 不再命中）。"""
    from app.services import project_manager

    src = tmp_path / "repo"
    src.mkdir()
    (src / "train.py").write_text("print('hi')", encoding="utf-8")

    r = _create_original_with_url(tmp_projects, str(src))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "loaded"

    project = project_manager.get_project(body["project_id"])
    assert project["status"] == "loaded"
    assert (Path(project["workspace_path"]) / "source" / "train.py").exists()


def test_create_original_without_source_url_stays_loading(tmp_projects):
    """不传 source_url：保持 loading，等用户显式触发分析（本次不改这条语义）。"""
    from app.services import project_manager

    r = tmp_projects.post("/api/projects", json={"project_type": "original", "source": "local-test"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "loading"
    assert project_manager.get_project(body["project_id"])["status"] == "loading"


def test_load_failure_rolls_back_project(tmp_projects, tmp_path):
    """加载失败：400、detail 带 project_id 与原因，且半成品项目记录与工作区都被清掉。"""
    from app.services import project_manager

    r = _create_original_with_url(tmp_projects, str(tmp_path / "not-exist-dir"))
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "加载失败" in detail
    assert "本地路径不存在" in detail

    project_id = _project_id_from_detail(detail)
    assert project_manager.get_project(project_id) is None
    assert not (tmp_path / "projects" / project_id).exists()


def test_rollback_does_not_follow_source_link(tmp_projects, tmp_path, monkeypatch):
    """回滚只删工作区里的 source 链接本身，绝不跟随链接删用户原目录（本地挂载安全硬要求）。

    构造「挂载成功之后再失败」的真实场景：把 _mount_local 换成建链接后抛错。
    """
    from app.services import analysis_service, project_manager

    target = tmp_path / "user_dir"
    target.mkdir()
    (target / "important.txt").write_text("用户数据", encoding="utf-8")

    probe = tmp_path / "probe_link"
    kind = _try_dir_link(target, probe)
    if kind is None:
        pytest.skip("本机无创建目录符号链接/目录联接的权限：跳过「回滚不跟随链接」断言")
    _drop_link(probe)

    created: list[str] = []

    def _mount_then_fail(local: Path, source_dir: Path) -> None:
        if source_dir.is_dir() and not source_dir.is_symlink():
            source_dir.rmdir()  # create_project 建的空 source 目录，先让位给链接
        kind = _try_dir_link(local, source_dir)
        assert kind is not None, "探针可建链接，实际挂载却失败"
        created.append(kind)
        raise RuntimeError("模拟挂载后加载失败")

    monkeypatch.setattr(analysis_service, "_mount_local", _mount_then_fail)
    r = _create_original_with_url(tmp_projects, str(target))
    assert r.status_code == 400, r.text
    assert created, "未走到挂载步骤"
    project_id = _project_id_from_detail(r.json()["detail"])

    # 用户原目录与其内容完好（回滚没有递归进链接目标）
    assert target.is_dir()
    assert (target / "important.txt").read_text(encoding="utf-8") == "用户数据"

    # 链接所在的工作区已清理
    assert project_manager.get_project(project_id) is None
    assert not (tmp_path / "projects" / project_id).exists()


def test_delete_project_removes_readonly_git_workspace(tmp_projects, tmp_path):
    """工作区含 git 仓库（对象文件只读）时也必须整棵删掉。

    2026-10-04 实测缺陷：`_remove_dir_safely` 遇到只读文件 `PermissionError` 就跳过，
    而 Windows 上 `.git/objects/**` 是只读的 → 目录非空 → `rmdir` 失败，
    「删项目」只删了数据库记录、工作区（含版本历史）留在盘上。
    """
    from app.services import project_manager

    project_id = _create_original(tmp_projects)
    ws = tmp_path / "projects" / project_id

    git_obj = ws / ".git" / "objects" / "ab" / "deadbeef"
    git_obj.parent.mkdir(parents=True, exist_ok=True)
    git_obj.write_bytes(b"x")
    os.chmod(git_obj, 0o444)  # 只读，模拟 git 对象文件
    readonly_dir = ws / "runs"
    readonly_dir.mkdir(exist_ok=True)
    (readonly_dir / "log.txt").write_text("x", encoding="utf-8")

    assert project_manager.delete_project(project_id) is True

    assert project_manager.get_project(project_id) is None
    assert not ws.exists(), "工作区（含只读 git 对象）应被完整删除"

