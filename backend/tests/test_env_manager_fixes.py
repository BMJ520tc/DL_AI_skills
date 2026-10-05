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


# --------------------------------------- ④ 额外依赖持久化（清单漏声明；重建环境不丢）

def test_record_extra_dep_persists_and_dedupes(tmp_path):
    """补装过的「清单漏声明」包记进工作区清单；大小写/版本约束不同视为同一个包。"""
    ws = tmp_path / "ws"
    env_manager.record_extra_dep(ws, "IPython")
    env_manager.record_extra_dep(ws, "ipython")        # 大小写不同 → 不重复记
    env_manager.record_extra_dep(ws, "ipython>=8")     # 带版本约束 → 同一个包
    env_manager.record_extra_dep(ws, "somepkg")

    assert env_manager.load_extra_deps(ws) == ["IPython", "somepkg"]
    assert env_manager.extra_deps_path(ws).name == "deps_extra.txt"


def test_install_with_fix_installs_recorded_extra_deps(tmp_path, monkeypatch, isolated_db):
    """重建环境必须把「此前自动补装过的额外依赖」一并装回（2026-10-06 scGPT/IPython 实测）。"""
    source = _src(tmp_path, "extra_src", {"requirements.txt": "torch\n"})
    ws = tmp_path / "ws_extra"
    env_manager.record_extra_dep(ws, "ipython")
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: None)
    calls: list[str] = []

    async def fake_try_install(pip, req_file, index_url=None, extra_index_url=None):
        calls.append(str(req_file))
        return {"ok": True, "error": None, "command": "pip install", "index_url": index_url,
                "extra_index_url": extra_index_url,
                "started_at": env_manager._now(), "finished_at": env_manager._now()}

    monkeypatch.setattr(env_manager, "_try_install", fake_try_install)
    monkeypatch.setattr(env_manager.task_manager, "update_progress", lambda *a, **k: None)

    ok = asyncio.run(env_manager._install_with_fix(
        source, tmp_path / "envdir", "pid-x", "task-x", "venv", ws=ws))

    assert ok is True
    assert any(p.endswith("deps_extra.txt") for p in calls), calls


def test_install_extras_failure_fails_env_creation(tmp_path, monkeypatch, isolated_db):
    """额外依赖装不上 → 建环境判失败（不留「建成功但 import 起不来」的环境）。"""
    source = _src(tmp_path, "extra_fail_src", {"requirements.txt": "torch\n"})
    ws = tmp_path / "ws_extra_fail"
    env_manager.record_extra_dep(ws, "nonexistent-pkg-xyz")
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: None)

    async def fake_try_install(pip, req_file, index_url=None, extra_index_url=None):
        ok = not str(req_file).endswith("deps_extra.txt")
        return {"ok": ok, "error": None if ok else "no such package", "command": "pip install",
                "index_url": index_url, "extra_index_url": extra_index_url,
                "started_at": env_manager._now(), "finished_at": env_manager._now()}

    monkeypatch.setattr(env_manager, "_try_install", fake_try_install)
    monkeypatch.setattr(env_manager.task_manager, "update_progress", lambda *a, **k: None)

    ok = asyncio.run(env_manager._install_with_fix(
        source, tmp_path / "envdir", "pid-y", "task-y", "venv", ws=ws))
    assert ok is False


# --------------------------------------- ③ 任务前带入：依赖冲突预检（模块详细设计 8.2）


def test_pins_from_conflicts_parses_both_shapes():
    """可操作形状（设计 schema {pkg, resolution}/{pkg, version_b}）产出钉；非可操作形状不产出。"""
    conflicts = [
        {"structured": {"pkg": "torch", "resolution": "torch==2.1.2"}},
        {"structured": '{"pkg": "numpy", "resolution": "==1.26.4"}'},   # JSON 文本 + 纯版本式
        {"structured": {"pkg": "scanpy", "version_b": "1.9.0"}},
        {"structured": {"project_id": "p1", "stage": "创建(venv)", "error": "boom"}},  # 无 pkg → 跳过
        {"structured": None},
    ]
    pins = env_manager._pins_from_conflicts(conflicts)
    assert pins == {"torch": "torch==2.1.2", "numpy": "numpy==1.26.4", "scanpy": "scanpy==1.9.0"}


def test_apply_known_pins_rewrites_only_matching_lines(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("torch==1.0.0\nnumpy>=1.0\n# torch comment\npandas\n", encoding="utf-8")
    out = env_manager._apply_known_pins(req, {"torch": "torch==2.1.2", "absent": "absent==9"})

    assert out is not None
    assert out["applied"] == [{"pkg": "torch", "line": "torch==2.1.2"}]
    text = Path(out["path"]).read_text(encoding="utf-8")
    assert "torch==2.1.2" in text
    assert "numpy>=1.0" in text and "# torch comment" in text   # 非目标行与注释不动
    assert out["path"] != str(req)                               # 原清单不被改写


def test_apply_known_pins_noop_when_nothing_matches(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("pandas\n", encoding="utf-8")
    assert env_manager._apply_known_pins(req, {"torch": "torch==2.1.2"}) is None


def test_dependency_precheck_surfaces_relevant_and_filters_foreign(isolated_db):
    """预检把「通用 / 本项目」的已确认冲突预警出来，其它项目的专属冲突不外泄；未确认的不算。"""
    ks = knowledge_service
    ks.record_knowledge({"type": "dependency_conflict", "title": "通用冲突",
                         "content": "torch 与 numpy 版本互斥", "structured": {"pkg": "torch", "resolution": "torch==2.1.2"},
                         "status": "confirmed"})
    ks.record_knowledge({"type": "dependency_conflict", "title": "别家冲突",
                         "content": "x", "structured": {"project_id": "other"}, "status": "confirmed"})
    ks.record_knowledge({"type": "dependency_conflict", "title": "本项目草稿", "content": "y",
                         "structured": {"project_id": "pid"}, "status": "draft"})  # 未确认 → 不算

    hits = env_manager._dependency_precheck("pid")

    titles = [h["title"] for h in hits]
    assert "通用冲突" in titles
    assert "别家冲突" not in titles          # 其它项目专属不外泄
    assert "本项目草稿" not in titles        # 未确认不算


def test_install_with_fix_preapplies_known_pins(tmp_path, monkeypatch, isolated_db):
    """预应用：已确认冲突的版本钉在任何安装尝试前改写依赖清单，预警与应用合并进任务进度。"""
    source = _src(tmp_path, "pre_src", {"requirements.txt": "torch==1.0.0\n"})
    knowledge_service.record_knowledge({
        "type": "dependency_conflict", "title": "torch 冲突", "content": "改用 2.1.2",
        "structured": {"pkg": "torch", "resolution": "torch==2.1.2"}, "status": "confirmed",
    })
    used: dict = {}

    async def fake_try_install(pip, req_file, index_url=None, extra_index_url=None):
        used["req_file"] = req_file
        return {"ok": True, "error": None, "command": "pip install", "index_url": index_url,
                "extra_index_url": extra_index_url,
                "started_at": env_manager._now(), "finished_at": env_manager._now()}

    monkeypatch.setattr(env_manager, "_try_install", fake_try_install)
    captured: list = []
    monkeypatch.setattr(env_manager.task_manager, "update_progress",
                        lambda task_id, progress: captured.append(progress))

    ok = asyncio.run(env_manager._install_with_fix(
        source, tmp_path / "envdir", "pid-pre", "task-pre", "venv"))

    assert ok is True
    # 实际安装用的是预应用后的清单副本（原 requirements.txt 未被改写）
    assert Path(used["req_file"]).read_text(encoding="utf-8").strip() == "torch==2.1.2"
    assert (source / "requirements.txt").read_text(encoding="utf-8").strip() == "torch==1.0.0"
    # 预警与「已应用」在同一次进度写入里（避免 update_progress 覆盖丢键）
    assert any(p.get("env_precheck") and p.get("env_precheck_applied") for p in captured)
