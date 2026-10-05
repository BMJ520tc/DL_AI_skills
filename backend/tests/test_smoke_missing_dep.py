"""最小命令验证的两处修复（2026-10-05，scGPT 实测暴露）：

① **装完但 import 缺包**（依赖清单漏声明，如 scGPT 用了 IPython 却没写进 pyproject）——
   最小命令验证失败时应按缺包名**自动补装一次并重跑**，让这类项目自愈；
② 库型项目兜底候选**不该挑 `tests/`、`docs/` 这类无信息量目录**
   （`python -c "import tests"` 只是导入空壳包、恒成功 → 假阳性，稀释掉真实的 `import <pkg>` 失败）。
"""
from __future__ import annotations

import asyncio

from app.services import analysis_service, env_manager, project_manager


# ---------------------------------------------------------------- ② 候选选择

def test_package_candidates_skip_non_entry_dirs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("scgpt", "tests", "docs", "examples", "scripts"):
        d = src / name
        d.mkdir()
        (d / "__init__.py").write_text("", encoding="utf-8")

    cands = analysis_service._package_candidates(src)

    assert ["-c", "import scgpt"] in cands
    assert all("import scgpt" == c[-1] for c in cands)   # tests/docs/… 全部被跳过


# ---------------------------------------------------------------- ① 缺包补装

def test_missing_module_and_package_mapping():
    err = "ModuleNotFoundError: No module named 'IPython'"
    assert analysis_service._missing_module(err) == "IPython"
    assert analysis_service._missing_module("ModuleNotFoundError: No module named 'a.b'") == "a"
    assert analysis_service._missing_module("SyntaxError: bad") is None
    assert analysis_service._module_to_package("IPython") == "ipython"
    assert analysis_service._module_to_package("sklearn") == "scikit-learn"
    assert analysis_service._module_to_package("some_pkg") == "some-pkg"


def test_missing_dep_to_fix_dedups_and_caps():
    err = "ModuleNotFoundError: No module named 'IPython'"
    assert analysis_service._missing_dep_to_fix(err, []) == "ipython"
    assert analysis_service._missing_dep_to_fix(err, ["ipython"]) is None      # 已补过不重复
    assert analysis_service._missing_dep_to_fix(err, ["a", "b", "c"]) is None  # 上限 3
    assert analysis_service._missing_dep_to_fix("RuntimeError: boom", []) is None


def test_install_missing_package_uses_env_pip(tmp_path, monkeypatch):
    captured: dict = {}

    async def fake_run(cmd, env=None, timeout=None):
        captured["cmd"] = cmd
        return 0, "Successfully installed"

    monkeypatch.setattr(env_manager.proc_util, "run_command", fake_run)
    res = asyncio.run(env_manager.install_missing_package(tmp_path / "env", "ipython"))

    assert res["ok"] is True
    assert captured["cmd"][1:3] == ["install", "ipython"]      # <env pip> install ipython …


def test_install_missing_package_falls_back_to_secondary_index(tmp_path, monkeypatch):
    """索引瞬时抽风（`from versions: none`）→ 切备源重试一次（与主安装同口径）。"""
    calls: list = []

    async def fake_run(cmd, env=None, timeout=None):
        calls.append(cmd)
        if len(calls) == 1:
            return 1, "ERROR: Could not find a version that satisfies the requirement ipython (from versions: none)"
        return 0, "Successfully installed"

    monkeypatch.setattr(env_manager.proc_util, "run_command", fake_run)
    monkeypatch.setattr(env_manager, "PIP_INDEX_URL", "https://mirror.example/simple")
    monkeypatch.setattr(env_manager, "PIP_FALLBACK_INDEX", "https://pypi.org/simple")

    res = asyncio.run(env_manager.install_missing_package(tmp_path / "env", "ipython"))

    assert res["ok"] is True and len(calls) == 2
    assert "--index-url" in calls[1] and "pypi.org" in " ".join(calls[1])


# ---------------------------------------------------------------- 裸命令（pytest 等）

def test_try_command_dispatch(tmp_path, monkeypatch):
    """`pytest x.py` 必须跑环境里的 pytest.exe，**不能**前缀成 `python pytest …`。"""
    captured: dict = {}

    def fake_run(full_cmd, cwd, grace_s):
        captured["cmd"] = full_cmd
        return {"ok": True, "error": None, "mode": "exited"}

    monkeypatch.setattr(analysis_service, "_run_with_grace", fake_run)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("", encoding="utf-8")
    scripts = tmp_path / "env" / "Scripts"
    scripts.mkdir(parents=True)
    py = str(scripts / "python.exe")
    (scripts / "python.exe").write_text("", encoding="utf-8")
    (scripts / "pytest.exe").write_text("", encoding="utf-8")

    async def run(cmd):
        return await analysis_service._try_command(py, cmd, src)

    asyncio.run(run(["pytest", "tests/test_x.py"]))
    assert captured["cmd"] == [str(scripts / "pytest.exe"), "tests/test_x.py"]

    asyncio.run(run(["main.py"]))
    assert captured["cmd"] == [py, "main.py"]                 # 本地 .py → 环境解释器

    asyncio.run(run(["python", "main.py"]))
    assert captured["cmd"] == [py, "main.py"]                 # README 的 python 前缀 → 换成环境解释器

    asyncio.run(run(["-m", "pytest"]))
    assert captured["cmd"] == [py, "-m", "pytest"]

    asyncio.run(run(["-c", "import scgpt"]))                  # 库型候选：python -c …
    assert captured["cmd"] == [py, "-c", "import scgpt"]


def test_run_verify_installs_missing_dep_and_retries(monkeypatch, isolated_db, tmp_path):
    """缺包 → 自动补装 → 重跑同一命令 → 通过（不再直接判失败）。"""
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    pid = project_manager.create_project("original", source="t")
    calls = {"try": 0, "installed": []}

    async def fake_candidates(source, python=None):
        return [{"command": ["-c", "import scgpt"], "source": "package"}]

    def _res(ok, error):
        now = analysis_service._now()
        return {"ok": ok, "error": error, "mode": "exit", "command": "python -c import scgpt",
                "started_at": now, "finished_at": now}

    async def fake_try(python, cmd, source, grace_s=None):
        calls["try"] += 1
        if calls["try"] == 1:
            return _res(False, "ModuleNotFoundError: No module named 'IPython'")
        return _res(True, None)

    async def fake_install(project_id, task_id, env_dir, package):
        calls["installed"].append(package)
        return True

    monkeypatch.setattr(analysis_service, "_project_python", lambda ws, pid=None: "py")
    monkeypatch.setattr(analysis_service, "_candidate_commands", fake_candidates)
    monkeypatch.setattr(analysis_service, "_try_command", fake_try)
    monkeypatch.setattr(analysis_service, "_install_missing_dep", fake_install)

    asyncio.run(analysis_service._run_verify({"project_id": pid}, "task-miss"))

    assert calls["installed"] == ["ipython"]     # 按缺包名补装
    assert calls["try"] == 2                     # 补装后重跑一次并通过
