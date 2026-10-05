"""独立环境改短路径根（规避 Windows 260 上限，2026-10-05）。

背景：环境原落 `data/projects/<32位项目id>/env`（前缀 ~99 字符），装 orbax 这类深层测试夹具的包
时完整路径 >260 → pip `WinError 3` 中断（scGPT 建环境卡死）。改为 `<ENV_ROOT>/<项目id前8位>`。
本文件锁三条：路径形状要短；解释器查找**短根优先、旧 ws/env 兼容**；状态接口透出 env_dir/env_python。
"""
from __future__ import annotations

import json

from app import config
from app.services import analysis_service, env_manager, project_manager, task_manager


def test_project_env_dir_uses_8char_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_ROOT", tmp_path / "DLenv")
    d = config.project_env_dir("0500b33a57714798bd9766611d4f7ab7")
    assert d == tmp_path / "DLenv" / "0500b33a"      # 取前 8 位
    assert d.name == "0500b33a" and len(d.name) == 8


def test_real_env_root_is_short():
    """本机默认根必须短：旧写法前缀 `data/projects/<32位id>/env` 就已 ~99 字符，是 260 上限的根因。"""
    assert config.ENV_ROOT.name == "DLenv"
    real = str(config.project_env_dir("0" * 32))
    assert len(real) <= 24, real                     # 短根 + 8 位 id（远短于旧 ~99 的前缀）


def test_project_python_prefers_short_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_ROOT", tmp_path / "DLenv")
    pid = "abcdef1234567890"
    ws = tmp_path / "projects" / pid
    scripts = config.project_env_dir(pid) / "Scripts"
    scripts.mkdir(parents=True)
    py = scripts / "python.exe"
    py.write_text("", encoding="utf-8")

    assert analysis_service._project_python(ws) == str(py)   # ws 末段即项目 id


def test_project_python_falls_back_to_legacy_ws_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_ROOT", tmp_path / "DLenv")   # 短根下什么都没有
    pid = "legacy0001"
    ws = tmp_path / "projects" / pid
    scripts = ws / "env" / "Scripts"
    scripts.mkdir(parents=True)
    py = scripts / "python.exe"
    py.write_text("", encoding="utf-8")

    assert analysis_service._project_python(ws, pid) == str(py)   # 旧位置仍兼容


def test_get_env_status_exposes_dir_and_python(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(config, "ENV_ROOT", tmp_path / "DLenv")
    pid = project_manager.create_project("original", source="t")

    st = env_manager.get_env_status(pid)
    assert st["env_dir"] == str(config.project_env_dir(pid))
    assert st["env_python"] is None                  # 未建环境

    scripts = config.project_env_dir(pid) / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_text("", encoding="utf-8")
    assert env_manager.get_env_status(pid)["env_python"] == str(scripts / "python.exe")


def test_env_progress_reporter_merges_keys(app_client):
    """进度打点：`update_progress` 整体覆盖，滚动 stage 时既有键（env_python/env_precheck…）不能被冲掉。"""
    tid = task_manager.create_task("env_create", params={"project_id": "env-prog-test"})
    report = env_manager._progress_reporter(tid)

    report("创建 venv 环境（解释器 python.exe）…", env_python={"required": ">=3.10"})
    prog = json.loads(task_manager.get_task(tid)["progress"])
    assert prog["stage"].startswith("创建 venv") and prog["env_python"] == {"required": ">=3.10"}

    report("安装依赖（第 1/3 次尝试）…（下载/安装可能数分钟）")
    prog = json.loads(task_manager.get_task(tid)["progress"])
    assert "第 1/3 次尝试" in prog["stage"]
    assert prog["env_python"] == {"required": ">=3.10"}      # 既有键保留
