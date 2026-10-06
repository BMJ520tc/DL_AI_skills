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


def test_cuda_match_uses_cuda_wheel(tmp_path, monkeypatch):
    """驱动满足要求 → **装 CUDA 版 wheel**。

    原来这里是 `action=None`（「按默认 wheel 安装」）——但**默认索引（Windows 上的 PyPI/镜像）
    给的 torch 就是 CPU 版**，于是环境永远拿不到 GPU 轮子（实测 scGPT 装出 `torch 2.14.1+cpu`）。
    """
    source = _src(tmp_path, "ok_src", {"requirements.txt": "torch==2.1.2+cu118\n"})
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: "12.1")
    monkeypatch.setattr(env_manager, "_index_reachable", lambda *a, **k: True)

    plan = env_manager.detect_versions(source)["cuda_plan"]

    assert plan["match"] is True and plan["action"] == "cuda_wheel"
    assert plan["index"].endswith("/cu121")                 # 驱动 12.1 → 最高可用档
    assert plan["torch_spec"] == "torch==2.1.2+cu118"       # 尊重清单的钉


def test_cuda_plan_uses_cuda_wheel_without_declaration(tmp_path, monkeypatch):
    """清单**未声明 CUDA** 但本机有 GPU → 照样装 CUDA 版（有驱动却装 CPU 轮子才是缺陷）。"""
    source = _src(tmp_path, "nocuda_src", {"requirements.txt": "numpy\npandas\n"})
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: "13.1")
    monkeypatch.setattr(env_manager, "_index_reachable", lambda *a, **k: True)

    plan = env_manager.detect_versions(source)["cuda_plan"]

    assert plan["required"] is None and plan["match"] is True
    assert plan["action"] == "cuda_wheel" and plan["index"].endswith("/cu130")
    assert plan["torch_spec"] == "torch"                    # 清单没有 torch → 不钉版本


def test_cuda_plan_disabled_by_flag(tmp_path, monkeypatch):
    """`ENV_USE_GPU_TORCH=0` → 按默认 wheel 安装（不装 CUDA 版），原因写明。"""
    source = _src(tmp_path, "off_src", {"requirements.txt": "numpy\n"})
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: "13.1")
    monkeypatch.setattr(env_manager, "GPU_TORCH_ENABLED", False)

    plan = env_manager.detect_versions(source)["cuda_plan"]

    assert plan["action"] is None and "ENV_USE_GPU_TORCH=0" in plan["reason"]


def test_cuda_plan_index_unreachable_falls_back(tmp_path, monkeypatch):
    """候选索引全不可达 → 回落默认 wheel，并把「都不可达」写进原因（不静默）。"""
    source = _src(tmp_path, "down_src", {"requirements.txt": "numpy\n"})
    monkeypatch.setattr(env_manager, "_detect_cuda", lambda: "13.1")
    monkeypatch.setattr(env_manager, "_index_reachable", lambda *a, **k: False)

    plan = env_manager.detect_versions(source)["cuda_plan"]

    assert plan["action"] is None and "不可达" in plan["reason"]


def test_preinstall_torch_command_shape(tmp_path, monkeypatch, isolated_db):
    """预装命令：有镜像就 `--find-links <wheel 页>` + **钉住版本**（否则 CPU 版会赢）；否则 `--index-url <cu>`。"""
    seen: list[list[str]] = []

    async def fake_run(cmd, **kw):
        seen.append(list(cmd))
        return 0, ""

    monkeypatch.setattr(env_manager.proc_util, "run_command", fake_run)
    # 解析要联网，这里给确定结果：钉 + 「真正列出 wheel 的那一层」
    monkeypatch.setattr(env_manager, "resolve_mirror_torch_pin",
                        lambda py, links: ("torch==2.14.1+cu126", str(links[0]).rstrip("/") + "/torch/"))
    plan = {"index": "https://download.pytorch.org/whl/cu126", "torch_spec": "torch==2.14.1+cu126",
            "find_links": ["https://mirror.sjtu.edu.cn/pytorch-wheels/cu126/"]}
    assert asyncio.run(env_manager._preinstall_torch(
        "pip", plan, {}, "https://mirrors.cloud.tencent.com/pypi/simple",
        "pid-pp", "task-pp")) is True
    cmd = seen[0]
    assert "--find-links" in cmd and any("pytorch-wheels/cu126/torch/" in a for a in cmd)
    assert "torch==2.14.1+cu126" in cmd
    assert "--index-url" in cmd and any("pypi/simple" in a for a in cmd)   # 其余依赖走主索引

    seen.clear()
    plan2 = {"index": "https://download.pytorch.org/whl/cu126", "torch_spec": "torch", "find_links": []}
    asyncio.run(env_manager._preinstall_torch("pip", plan2, {}, None, "pid-pp", "task-pp"))
    cmd2 = seen[0]
    assert "--index-url" in cmd2 and plan2["index"] in cmd2 and "--find-links" not in cmd2


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


# --------------------------- 建环境顺带取权重：扫描与筛选 ---------------------------


def test_plan_weight_downloads_dedupes_and_mirrors(tmp_path, monkeypatch):
    """按文件名去重；HuggingFace 官方域名换 hf-mirror（本机 huggingface.co 不可达）。"""
    from app.services import env_manager

    src = tmp_path / "source"
    src.mkdir()
    (src / "main.py").write_text(
        'A = "https://huggingface.co/org/repo/resolve/main/model.ckpt"\n'
        'B = "https://gateway.example.com/model.ckpt"\n'      # 同名的另一份 → 只留一条
        'C = "https://hf-mirror.com/org/repo/resolve/main/other.safetensors"\n'
        'D = "https://example.com/not_a_weight.txt"\n',
        encoding="utf-8",
    )

    plans = env_manager.plan_weight_downloads(src)
    names = sorted(p["filename"] for p in plans)
    assert names == ["model.ckpt", "other.safetensors"]        # 同名去重、非权重后缀忽略
    by_name = {p["filename"]: p["url"] for p in plans}
    assert by_name["model.ckpt"].startswith("https://hf-mirror.com/")
    assert by_name["other.safetensors"].startswith("https://hf-mirror.com/")


def test_plan_weight_downloads_filters_by_entry_class(tmp_path, monkeypatch):
    """默认只下与入口类同名的权重（entry）；全下用 ENV_WEIGHT_SCOPE=all。"""
    from app.services import env_manager

    src = tmp_path / "source"
    src.mkdir()
    (src / "main.py").write_text(
        'A = "https://hf-mirror.com/x/boltz1_conf.ckpt"\n'
        'B = "https://hf-mirror.com/x/boltz2_conf.ckpt"\n'
        'C = "https://hf-mirror.com/x/boltz2_aff.ckpt"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(env_manager, "ENV_WEIGHT_SCOPE", "entry")
    names = [p["filename"] for p in env_manager.plan_weight_downloads(src, "Boltz2")]
    assert names == ["boltz2_aff.ckpt", "boltz2_conf.ckpt"]     # boltz1 被滤掉

    monkeypatch.setattr(env_manager, "ENV_WEIGHT_SCOPE", "all")
    assert len(env_manager.plan_weight_downloads(src, "Boltz2")) == 3


def test_plan_weight_downloads_falls_back_when_name_matches_nothing(tmp_path, monkeypatch):
    """入口类名对不上任何权重时退回全部——不能「因为名字对不上就一个都不下」。"""
    from app.services import env_manager

    src = tmp_path / "source"
    src.mkdir()
    (src / "m.py").write_text('A = "https://hf-mirror.com/x/weights.ckpt"\n', encoding="utf-8")
    monkeypatch.setattr(env_manager, "ENV_WEIGHT_SCOPE", "entry")
    assert [p["filename"] for p in env_manager.plan_weight_downloads(src, "TransformerModel")] \
        == ["weights.ckpt"]


# --------------------------- 镜像 torch 版本钉（CPU 版抢镜的修复） ---------------------------


def test_resolve_mirror_torch_pin_picks_newest_matching_cu_wheel(monkeypatch):
    """从镜像列表里挑**匹配本解释器/平台**的最新 cu 轮子 → `torch==<ver>+cuXXX`。

    为什么必须钉：pip 在「镜像 + 主索引」间只按版本号取高，PyPI 的 CPU 版 torch 版本号常更高
    （实测 2.14.1 > 2.9.1+cu126）→ 镜像配了也白配，环境装成 CPU 版。
    """
    from app.services import env_manager

    html = "\n".join([
        "torch-2.9.1%2Bcu126-cp312-cp312-win_amd64.whl",
        "torch-2.8.0%2Bcu126-cp312-cp312-win_amd64.whl",
        "torch-9.9.9%2Bcu126-cp311-cp311-win_amd64.whl",              # 别的解释器 → 不选
        "torch-9.9.9%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl",  # 别的平台 → 不选
    ])
    monkeypatch.setattr(env_manager, "_fetch_text", lambda url: html)
    monkeypatch.setattr(env_manager, "_python_tag", lambda py: "cp312")
    monkeypatch.setattr(env_manager, "_platform_tag", lambda: "win_amd64")

    assert env_manager.resolve_mirror_torch_pin("python", ["https://m/x"]) == (
        "torch==2.9.1+cu126", "https://m/x/torch/"      # 链接要指到真正列出 wheel 的那一层
    )


def test_resolve_mirror_torch_pin_none_when_no_match(monkeypatch):
    """镜像里没有匹配的轮子 / 列表取不到 → None（调用方回退旧行为，不阻断建环境）。"""
    from app.services import env_manager

    monkeypatch.setattr(env_manager, "_python_tag", lambda py: "cp312")
    monkeypatch.setattr(env_manager, "_platform_tag", lambda: "win_amd64")
    monkeypatch.setattr(env_manager, "_fetch_text", lambda url: "torch-2.9.1%2Bcu126-cp311-cp311-win_amd64.whl")
    assert env_manager.resolve_mirror_torch_pin("python", ["https://m/x"]) is None

    monkeypatch.setattr(env_manager, "_fetch_text", lambda url: None)
    assert env_manager.resolve_mirror_torch_pin("python", ["https://m/x"]) is None


def test_preinstall_torch_cmd_pins_torch_when_mirror_configured(monkeypatch):
    """配了镜像且钉到版本 → 命令里用 `torch==<ver>+cuXXX`（而不是会被 CPU 版压过的 `torch>=2.2`）。"""
    from app.services import env_manager

    plan = {"torch_spec": "torch>=2.2", "index": "https://download.pytorch.org/whl/cu130",
            "find_links": ["https://m/cu126/"]}
    cmd = env_manager.preinstall_torch_cmd("pip", plan, "https://pypi/simple",
                                           torch_pin="torch==2.9.1+cu126",
                                           torch_link="https://m/cu126/torch/")
    assert "torch==2.9.1+cu126" in cmd
    assert "--find-links" in cmd and "https://m/cu126/torch/" in cmd
    # 没钉到 → 回退原来的宽松 spec
    cmd2 = env_manager.preinstall_torch_cmd("pip", plan, "https://pypi/simple")
    assert "torch>=2.2" in cmd2


# --------------------------- 仓库自身装 editable（防缝合怪） ---------------------------


def test_repo_package_name_reads_pyproject_and_setup(tmp_path):
    """从 pyproject.toml / setup.cfg / setup.py 读出「仓库自己发布的包名」。"""
    from app.services import env_manager

    src = tmp_path / "with_pyproject"
    src.mkdir()
    (src / "pyproject.toml").write_text(
        '[project]\nname = "boltz"\nversion = "2.2.1"\n', encoding="utf-8")
    assert env_manager._repo_package_name(src) == "boltz"

    src2 = tmp_path / "with_setup"
    src2.mkdir()
    (src2 / "setup.py").write_text('setup(name="MyPkg", version="1.0")\n', encoding="utf-8")
    assert env_manager._repo_package_name(src2) == "MyPkg"

    src3 = tmp_path / "nothing"
    src3.mkdir()
    assert env_manager._repo_package_name(src3) is None


def test_install_repo_editable_skipped_without_packaging(tmp_path, monkeypatch):
    """没有打包文件（pyproject/setup）就不做——不能凭空 pip install -e。"""
    from app.services import env_manager

    src = tmp_path / "src"
    src.mkdir()
    called: list = []
    monkeypatch.setattr(env_manager.proc_util, "run_command",
                        lambda *a, **k: called.append(a) or (0, ""))
    asyncio.run(env_manager._install_repo_editable(src, tmp_path / "env", "t", lambda *a, **k: None))
    assert called == []


def test_install_repo_editable_runs_with_no_deps(tmp_path, monkeypatch):
    """有 pyproject 就 `pip install -e <src> --no-deps`（只换来源、不动依赖图）。"""
    from app.services import env_manager

    src = tmp_path / "src"
    src.mkdir()
    (src / "pyproject.toml").write_text('[project]\nname = "boltz"\n', encoding="utf-8")
    seen: list = []

    async def fake_run(cmd, **kw):
        seen.append(list(cmd))
        return 0, ""

    monkeypatch.setattr(env_manager.proc_util, "run_command", fake_run)
    monkeypatch.setattr(env_manager, "_env_pip", lambda env_dir: "pip")
    msgs: list = []
    asyncio.run(env_manager._install_repo_editable(src, tmp_path / "env", "t",
                                                   lambda s, **k: msgs.append(s)))
    assert seen and "-e" in seen[0] and "--no-deps" in seen[0] and str(src) in seen[0]
    assert any("editable" in m for m in msgs)
