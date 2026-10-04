"""环境管理缺陷修复用例（《模块详细设计》2.3 步骤 1/2 与「异常与边界」）。

覆盖两处缺陷：
① 容器通道：带 Dockerfile 的项目先探测本机 docker，探测不到给「未检测到 docker」的明确提示；
   探测到也不再抛「未启用」死错误，而是明确报「本仓库尚未实现容器环境创建」（既不假装成功也不静默改道）。
② 版本判断生效：清单声明的 Python 要求（pyproject/setup.py/environment.yml）被识别并驱动 venv
   解释器选择（选不到则告警回退）；驱动 CUDA 与依赖清单里的 torch/CUDA 要求不匹配 → 选 CPU 版
   wheel，且结论进 run_record（environment.cuda_plan）与任务进度。

全部用临时目录 + 临时库（isolated_db），不碰真实 data/。
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from app.services import env_manager, knowledge_service, project_manager


def _fake_project(ws: Path):
    return {"project_id": "pid", "workspace_path": str(ws)}


def _make_ws(tmp_path: Path, monkeypatch, files: dict) -> Path:
    """临时工作区 source/ + 伪项目行（改写 project_manager.get_project，不建真实项目）。"""
    ws = tmp_path / "ws"
    source = ws / "source"
    source.mkdir(parents=True)
    for name, text in files.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    monkeypatch.setattr(project_manager, "get_project", lambda pid: _fake_project(ws))
    return ws


def _src(tmp_path: Path, name: str, files: dict) -> Path:
    source = tmp_path / name
    source.mkdir(parents=True)
    for fname, text in files.items():
        (source / fname).write_text(text, encoding="utf-8")
    return source


# ---------------------------------------------------------------- ① 容器通道


def test_detect_env_type_prefers_dockerfile(tmp_path):
    ws = tmp_path / "ws"
    (ws / "source").mkdir(parents=True)
    (ws / "source" / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
    assert env_manager.detect_env_type(ws) == "container"


def test_container_without_docker_reports_missing_docker(tmp_path, monkeypatch, isolated_db):
    """本机无 docker → 明确「未检测到 docker」提示，并指向 conda/venv。"""
    ws = _make_ws(tmp_path, monkeypatch, {"Dockerfile": "FROM python:3.11\n"})
    monkeypatch.delenv("DOCKER_PATH", raising=False)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)

    with pytest.raises(RuntimeError) as ei:
        asyncio.run(env_manager._run_env_create({"project_id": "pid"}, "task-nodocker"))

    msg = str(ei.value)
    assert "未检测到 docker" in msg
    assert "conda/venv" in msg
    # 不假装成功：既没建环境目录，也没留成功的运行记录
    assert not (ws / "env").exists()
    assert knowledge_service.get_latest_run("pid", "env_install") is None


def test_container_with_docker_reports_unimplemented_not_disabled(tmp_path, monkeypatch, isolated_db):
    """本机有 docker → 不再抛「未启用」，而是明确「尚未实现容器环境创建」（如实报错）。"""
    ws = _make_ws(tmp_path, monkeypatch, {"Dockerfile": "FROM python:3.11\n"})
    monkeypatch.delenv("DOCKER_PATH", raising=False)
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: r"C:\fake\docker.EXE" if name == "docker" else None)

    with pytest.raises(RuntimeError) as ei:
        asyncio.run(env_manager._run_env_create({"project_id": "pid"}, "task-docker"))

    msg = str(ei.value)
    assert "未启用" not in msg
    assert "尚未实现容器环境创建" in msg
    assert "conda/venv" in msg
    assert not (ws / "env").exists()
    assert knowledge_service.get_latest_run("pid", "env_install") is None


def test_docker_path_env_override_detected(tmp_path, monkeypatch):
    """DOCKER_PATH 可覆盖探测口径（与 config 的 CONDA_EXE 同口径）。"""
    monkeypatch.setenv("DOCKER_PATH", r"C:\custom\docker.exe")
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: name if name == r"C:\custom\docker.exe" else None)
    assert env_manager._detect_docker() == r"C:\custom\docker.exe"
    assert "已检测到 docker" in env_manager._container_unavailable_reason()


# ---------------------------------------------------------------- ② 版本判断


def test_python_requirement_and_frameworks_from_environment_yml(tmp_path):
    source = _src(tmp_path, "conda_src", {"environment.yml": (
        "name: demo\n"
        "dependencies:\n"
        "  - python=3.9\n"
        "  - pytorch=2.0.1\n"
        "  - pip\n"
        "  - pip:\n"
        "      - torch==2.0.1\n"
    )})

    info = env_manager.detect_versions(source)

    assert info["python_required"] == "==3.9.*"
    assert info["python_requirement_source"] == "environment.yml"
    assert any(f.lower().startswith(("torch", "pytorch")) for f in info["frameworks"])


def test_python_requirement_and_frameworks_from_pyproject(tmp_path):
    source = _src(tmp_path, "pyp_src", {"pyproject.toml": (
        "[project]\n"
        'name = "demo"\n'
        'requires-python = ">=3.9,<3.12"\n'
        'dependencies = ["torch==2.1.2", "numpy"]\n'
    )})

    info = env_manager.detect_versions(source)

    assert info["python_required"] == ">=3.9,<3.12"
    assert info["python_requirement_source"] == "pyproject.toml"
    assert "torch==2.1.2" in info["frameworks"]


def test_python_requirement_and_frameworks_from_setup_py(tmp_path):
    source = _src(tmp_path, "setup_src", {"setup.py": (
        "from setuptools import setup\n"
        'setup(name="demo", python_requires=">=3.8", install_requires=["torch>=2.0", "numpy"])\n'
    )})

    info = env_manager.detect_versions(source)

    assert info["python_required"] == ">=3.8"
    assert info["python_requirement_source"] == "setup.py"
    assert "torch>=2.0" in info["frameworks"]


def test_resolve_env_python_falls_back_with_warning(tmp_path, monkeypatch):
    """声明的 Python 要求本机无匹配解释器 → 告警回退默认解释器（不得静默）。"""
    source = _src(tmp_path, "req_src", {"pyproject.toml":
                                        '[project]\nrequires-python = "==3.7.*"\n'})
    monkeypatch.setattr(env_manager, "ENV_VENV_PYTHON", None)
    monkeypatch.setattr(env_manager, "_iter_python_interpreter_cmds", lambda spec: [["python3.7"]])
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)

    cmd, note = env_manager.resolve_env_python(source)

    assert cmd == [env_manager.sys.executable]
    assert note["required"] == "==3.7.*"
    assert note["matched"] is False
    assert note["origin"] == "fallback"
    assert "未找到满足要求的解释器" in note["warning"]


def test_resolve_env_python_picks_matching_interpreter(tmp_path, monkeypatch):
    """有匹配解释器 → 作为 venv 解释器（语言版本真的驱动了选择）。"""
    source = _src(tmp_path, "match_src", {"pyproject.toml":
                                          '[project]\nrequires-python = "==3.11.*"\n'})
    monkeypatch.setattr(env_manager, "ENV_VENV_PYTHON", None)
    monkeypatch.setattr(env_manager, "_iter_python_interpreter_cmds", lambda spec: [["python3.11"]])
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: "C:/fake/python3.11.exe" if name == "python3.11" else None)
    monkeypatch.setattr(env_manager, "_cmd_python_version", lambda cmd: "3.11.9")

    cmd, note = env_manager.resolve_env_python(source)

    assert cmd == ["C:/fake/python3.11.exe"]
    assert note["matched"] is True
    assert note["origin"] == "detected"
    assert note["warning"] is None


def test_resolve_env_python_keeps_backend_when_satisfied(tmp_path, monkeypatch):
    """后端解释器本身就满足要求 → 行为与旧版一致（不额外探测）。"""
    source = _src(tmp_path, "host_src", {"pyproject.toml":
                                         '[project]\nrequires-python = ">=3.9"\n'})
    monkeypatch.setattr(env_manager, "ENV_VENV_PYTHON", None)

    def _boom(spec):
        raise AssertionError("后端解释器已满足要求，不应再探测候选解释器")

    monkeypatch.setattr(env_manager, "_iter_python_interpreter_cmds", _boom)

    cmd, note = env_manager.resolve_env_python(source)

    assert cmd == [env_manager.sys.executable]
    assert note["matched"] is True and note["origin"] == "backend"


def test_cuda_mismatch_selects_cpu_wheel_and_records_it(tmp_path, monkeypatch, isolated_db):
    """驱动 CUDA(11.8) < 依赖要求(cu121) → CPU 版 wheel，结论进 run_record 与安装命令。"""
    source = _src(tmp_path, "cuda_src", {"requirements.txt": "torch==2.1.2+cu121\nnumpy\n"})
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: "11.8")

    versions = env_manager.detect_versions(source)
    plan = versions["cuda_plan"]

    assert plan["driver"] == "11.8"
    assert plan["required"] == "12.1"
    assert plan["match"] is False
    assert plan["action"] == "cpu_wheel"
    assert plan["index"] == env_manager.CPU_TORCH_INDEX

    # 生效①：安装命令带上 CPU wheel 额外索引
    cmd = env_manager._install_cmd("pip", source / "requirements.txt", None, plan["index"])
    assert "--extra-index-url" in cmd and plan["index"] in cmd

    # 生效②：结论入 run_record（environment.cuda_plan + params.extra_index_url）
    env_manager._record_install(
        "pid-cuda", "task-cuda", 1,
        {"ok": False, "error": "boom", "command": " ".join(cmd), "index_url": None,
         "extra_index_url": plan["index"],
         "started_at": env_manager._now(), "finished_at": env_manager._now()},
        versions, "venv",
    )
    latest = knowledge_service.get_latest_run("pid-cuda", "env_install", "failed")
    assert latest is not None
    assert json.loads(latest["environment"])["cuda_plan"]["action"] == "cpu_wheel"
    assert json.loads(latest["params"])["extra_index_url"] == plan["index"]


def test_cuda_without_driver_selects_cpu_wheel(tmp_path, monkeypatch):
    """无 NVIDIA 驱动（nvidia-smi 不可用）而依赖要求 cu118 → 同样降级 CPU。"""
    source = _src(tmp_path, "nodrv_src", {"requirements.txt": "torch==2.1.2+cu118\n"})
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: None)

    plan = env_manager.detect_versions(source)["cuda_plan"]

    assert plan["match"] is False and plan["action"] == "cpu_wheel"
    assert "未检测到 NVIDIA 驱动" in plan["reason"]


def test_cuda_match_keeps_default_wheel(tmp_path, monkeypatch):
    """驱动满足要求 → 不降级（action=None）。"""
    source = _src(tmp_path, "ok_src", {"requirements.txt": "torch==2.1.2+cu118\n"})
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: "12.1")

    plan = env_manager.detect_versions(source)["cuda_plan"]

    assert plan["match"] is True and plan["action"] is None


def test_cuda_plan_none_without_declaration(tmp_path, monkeypatch):
    """依赖清单未声明 CUDA → 不臆断降级（match/action 均为 None）。"""
    source = _src(tmp_path, "nocuda_src", {"requirements.txt": "numpy\npandas\n"})
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: "12.1")

    plan = env_manager.detect_versions(source)["cuda_plan"]

    assert plan["match"] is None and plan["action"] is None


def test_install_with_fix_applies_cpu_index_and_progress(tmp_path, monkeypatch, isolated_db):
    """修正循环不改：CUDA 降级时 pip 调用带额外索引，进度里留下结论。"""
    source = _src(tmp_path, "install_src", {"requirements.txt": "torch==2.1.2+cu121\n"})
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: None)
    captured: dict = {}

    async def fake_try_install(pip, req_file, index_url=None, extra_index_url=None):
        captured["extra_index_url"] = extra_index_url
        return {"ok": True, "error": None, "command": "pip install", "index_url": index_url,
                "extra_index_url": extra_index_url,
                "started_at": env_manager._now(), "finished_at": env_manager._now()}

    monkeypatch.setattr(env_manager, "_try_install", fake_try_install)
    monkeypatch.setattr(env_manager.task_manager, "update_progress",
                        lambda task_id, progress: captured.setdefault("progress", []).append(progress))

    ok = asyncio.run(env_manager._install_with_fix(source, tmp_path / "envdir", "pid-i", "task-i", "venv"))

    assert ok is True
    assert captured["extra_index_url"] == env_manager.CPU_TORCH_INDEX
    assert captured["progress"][0]["env_cuda"]["action"] == "cpu_wheel"
