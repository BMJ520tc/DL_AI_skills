"""Windows 长路径上限（>260）的检测、提权开启与安装早停（2026-10-05）。

背景：pip 装深层包（orbax 测试夹具在 site-packages 内相对路径就有 228 字符）时完整路径 >260 →
`WinError 3 / Errno 2`，安装中断（scGPT 建环境卡死）。修法：注册表 `LongPathsEnabled=1`（经 UAC 提权），
并在检测到该错误时**立刻中止**（同一路径重试只会白等几分钟）。

用例不触发真实 UAC（`enable()` 在已开启/被 monkeypatch 时短路）。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.services import env_manager, knowledge_service, long_paths, project_manager

PIP_HINT = ("ERROR: Could not install packages due to an OSError: [WinError 3] 系统找不到指定的路径。: "
            "'D:\\DLenv\\0500b33a\\Lib\\site-packages\\orbax\\...'\n"
            "HINT: This error might have occurred since this system does not have Windows Long Path support "
            "enabled. You can find information on how to enable this at "
            "https://pip.pypa.io/warnings/enable-long-paths")


def test_is_long_path_error_detects_pip_hint():
    assert long_paths.is_long_path_error(PIP_HINT)
    assert long_paths.is_long_path_error("[WinError 3] 系统找不到指定的路径")
    assert not long_paths.is_long_path_error("ModuleNotFoundError: No module named 'torch'")
    assert not long_paths.is_long_path_error("")


def test_is_enabled_returns_bool():
    assert isinstance(long_paths.is_enabled(), bool)


def test_enable_short_circuits_when_already_enabled(monkeypatch):
    monkeypatch.setattr(long_paths, "is_enabled", lambda: True)
    r = long_paths.enable()          # 已开启 → 直接返回，不弹 UAC
    assert r["ok"] is True and r["enabled"] is True


def test_long_paths_endpoints(app_client, monkeypatch):
    assert app_client.get("/api/system/long-paths").json()["enabled"] in (True, False)
    # 打桩 enable，避免用例触发 UAC 弹窗
    monkeypatch.setattr(long_paths, "enable",
                        lambda: {"ok": True, "enabled": True, "detail": "stub"})
    r = app_client.post("/api/system/long-paths/enable")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_install_with_fix_breaks_early_on_long_path_error(tmp_path, monkeypatch, isolated_db):
    """长路径错误只试一次就中止：不再重试、不再找 agent 出建议（每次重试要几分钟）。"""
    source = tmp_path / "src"
    source.mkdir()
    (source / "requirements.txt").write_text("orbax\n", encoding="utf-8")
    monkeypatch.setattr(project_manager, "get_project",
                        lambda pid: {"project_id": pid, "workspace_path": str(tmp_path)})
    monkeypatch.setattr(long_paths, "is_enabled", lambda: False)
    calls: list = []

    async def fake_try(pip, req_file, index_url=None, extra_index_url=None):
        calls.append(1)
        return {"ok": False, "error": PIP_HINT, "command": "pip install", "index_url": index_url,
                "extra_index_url": extra_index_url,
                "started_at": env_manager._now(), "finished_at": env_manager._now()}

    monkeypatch.setattr(env_manager, "_try_install", fake_try)

    ok = asyncio.run(env_manager._install_with_fix(source, tmp_path / "envdir", "p", "t", "venv"))

    assert ok is False and len(calls) == 1


def test_long_path_hint_mentions_enabling(monkeypatch, isolated_db, tmp_path):
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(long_paths, "is_enabled", lambda: False)
    pid = project_manager.create_project("original", source="t")
    now = env_manager._now()
    knowledge_service.record_run({
        "project_id": pid, "task_id": "t1", "run_type": "env_install", "status": "failed",
        "error": PIP_HINT, "started_at": now, "finished_at": now,
    })
    hint = env_manager._long_path_hint(pid)
    assert "长路径" in hint and "开启长路径支持" in hint

    # 已开启长路径时不再提示
    monkeypatch.setattr(long_paths, "is_enabled", lambda: True)
    assert env_manager._long_path_hint(pid) == ""
