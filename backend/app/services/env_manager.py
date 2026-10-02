"""环境管理（模块详细设计 2.3）。

决策树: Dockerfile → 容器（可选扩展，需本机 docker）; environment.yml → conda; 否则 venv。
依赖修正循环: 安装失败 → agent 判断 → 降级/替换/移除 → 重试（上限 3 次）。
环境创建步骤与每次安装尝试写 run_record(env_install)，报错入 error 字段（供蒸馏知识提炼，需求六.1）。
"""
import asyncio
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import CONDA_PATH, ENV_VENV_PYTHON, PIP_FALLBACK_INDEX, PIP_INDEX_URL
from app.services import agent_service, knowledge_service, project_manager, task_manager

ENV_TASK_TYPE = "env_create"

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "修正后的完整依赖清单，每行一条（pip requirements 语法），供直接替换安装",
        },
        "reason": {"type": "string", "description": "修正理由：改了哪些包、为什么"},
    },
    "required": ["requirements", "reason"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_env_type(workspace: Path) -> str:
    source = workspace / "source"
    if (source / "Dockerfile").exists():
        return "container"
    if (source / "environment.yml").exists() or (source / "environment.yaml").exists():
        return "conda"
    return "venv"


def create_env(project_id: str) -> str:
    project_manager.require_type(project_id, {"original"})
    return task_manager.create_task(ENV_TASK_TYPE, project_id=project_id, params={"project_id": project_id})


def get_env_status(project_id: str) -> dict:
    project = project_manager.get_project(project_id)
    return {"project_id": project_id, "status": project["status"] if project else None}


async def _run_env_create(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    if project is None:
        raise RuntimeError("project not found")
    ws = Path(project["workspace_path"])
    source = ws / "source"
    env_type = detect_env_type(ws)

    if env_type == "container":
        raise RuntimeError("项目带 Dockerfile，需容器环境；容器通道为可选扩展，当前未启用（默认走 conda/venv）")

    env_dir = ws / "env"
    created_at = _now()
    if env_type == "conda":
        cmd = _conda_create_cmd(source, env_dir)
    else:
        # 默认用后端解释器；ENV_VENV_PYTHON 指定其他版本（项目依赖钉旧 Python 时）
        cmd = [ENV_VENV_PYTHON or sys.executable, "-m", "venv", str(env_dir)]
    err = await _create_env_dir(env_type, source, env_dir, cmd)
    if err is not None:
        knowledge_service.record_run(
            {"project_id": project_id, "task_id": task_id, "run_type": "env_install",
             "environment": {"type": env_type}, "params": {"step": "env_create"},
             "command": " ".join(cmd), "status": "failed", "error": err,
             "started_at": created_at, "finished_at": _now()}
        )
        project_manager.update_status(project_id, "env_failed")
        raise RuntimeError(f"环境创建失败({env_type}): {err}")

    knowledge_service.record_run(
        {"project_id": project_id, "task_id": task_id, "run_type": "env_install",
         "environment": {"type": env_type}, "params": {"step": "env_create"},
         "command": " ".join(cmd), "status": "success",
         "started_at": created_at, "finished_at": _now()}
    )

    ok = await _install_with_fix(source, env_dir, project_id, task_id, env_type)
    if ok:
        project_manager.update_status(project_id, "env_ready")
    else:
        project_manager.update_status(project_id, "env_failed")
        raise RuntimeError("环境安装失败（依赖修正循环耗尽）")


def _conda_create_cmd(source: Path, env_dir: Path) -> list[str]:
    """conda 环境创建命令：environment.yml/yaml 优先，否则建基础环境 + 后续 pip 装依赖。"""
    if CONDA_PATH is None:
        raise RuntimeError("未找到 conda 可执行文件")
    for name in ("environment.yml", "environment.yaml"):
        env_yml = source / name
        if env_yml.exists():
            return [CONDA_PATH, "env", "create", "-f", str(env_yml), "-p", str(env_dir), "-y"]
    return [CONDA_PATH, "create", "-p", str(env_dir), "python=3.11", "-y"]


def _conda_update_cmd(source: Path, env_dir: Path) -> list[str]:
    """conda 环境更新命令（environment.yml 的 pip: 段重跑；prefix 已存在时 update 可用）。

    注意：conda env update 不支持 -y（env create 才支持），故不加该参数。
    """
    if CONDA_PATH is None:
        raise RuntimeError("未找到 conda 可执行文件")
    for name in ("environment.yml", "environment.yaml"):
        env_yml = source / name
        if env_yml.exists():
            return [CONDA_PATH, "env", "update", "-f", str(env_yml), "-p", str(env_dir)]
    return [CONDA_PATH, "env", "update", "-p", str(env_dir)]


def _decode_err(e: subprocess.CalledProcessError) -> str:
    err = e.stderr or b""
    return err.decode(errors="ignore")[-2000:] if isinstance(err, bytes) else str(err)[-2000:]


def _run_create(cmd: list[str], index_url: Optional[str]) -> None:
    env = {**os.environ, "PIP_INDEX_URL": index_url} if index_url else None
    subprocess.run(cmd, check=True, capture_output=True, env=env)


async def _create_env_dir(env_type: str, source: Path, env_dir: Path, cmd: list[str]) -> Optional[str]:
    """执行环境创建命令，成功返回 None、失败返回错误文本。

    conda 的 environment.yml 常带 `pip:` 段（torch 等），其 pip 安装发生在 conda 内部、
    不经过 _install_with_fix，故索引不可达时同样切备源重试一次（用 env update 覆盖 pip 段）。
    """
    try:
        await asyncio.to_thread(_run_create, cmd, None)
        return None
    except subprocess.CalledProcessError as e:
        err = _decode_err(e)
    if env_type == "conda" and PIP_FALLBACK_INDEX and _is_index_error(err):
        try:
            await asyncio.to_thread(_run_create, _conda_update_cmd(source, env_dir), PIP_FALLBACK_INDEX)
            return None
        except subprocess.CalledProcessError as e2:
            return _decode_err(e2)
    return err


async def _install_with_fix(
    source: Path, env_dir: Path, project_id: str, task_id: str, env_type: str
) -> bool:
    pip = _env_pip(env_dir)
    req_file = _find_requirements(source)
    versions = detect_versions(source, _env_python(env_dir))
    index_url = PIP_INDEX_URL  # None → 用 pip 自身配置（用户 pip.ini）

    for attempt in range(1, 4):
        result = await asyncio.to_thread(_try_install, pip, req_file, index_url)
        _record_install(project_id, task_id, attempt, result, versions, env_type)
        if result["ok"]:
            return True
        # 索引不可达（非依赖冲突）：切备源重试一次，不消耗依赖修正循环
        if index_url != PIP_FALLBACK_INDEX and _is_index_error(result["error"]):
            index_url = PIP_FALLBACK_INDEX
            result = await asyncio.to_thread(_try_install, pip, req_file, index_url)
            _record_install(project_id, task_id, attempt, result, versions, env_type, step="pip_install_fallback")
            if result["ok"]:
                return True
        if attempt >= 3:
            break
        advice = await _agent_fix_advice(source, result["error"])
        new_req = _apply_advice(req_file, advice)
        # 记录建议与是否真的改动了依赖（旧实现静默无输出，问题难定位）
        task_manager.update_progress(task_id, {"env_fix": {
            "attempt": attempt,
            "reason": advice.get("reason") if isinstance(advice, dict) else None,
            "requirements_updated": new_req is not req_file,
        }})
        req_file = new_req
    return False


def _env_python(env_dir: Path) -> str:
    for c in (env_dir / "Scripts" / "python.exe", env_dir / "python.exe", env_dir / "bin" / "python"):
        if c.exists():
            return str(c)
    return sys.executable


def _env_pip(env_dir: Path) -> str:
    for c in (env_dir / "Scripts" / "pip.exe", env_dir / "bin" / "pip"):
        if c.exists():
            return str(c)
    return str(env_dir / "Scripts" / "pip.exe")


def _find_requirements(source: Path) -> Optional[Path]:
    for name in ("requirements.txt", "pyproject.toml", "setup.py"):
        p = source / name
        if p.exists():
            return p
    return None


# 项目清单（非 requirements 风格）：安装用 `pip install <项目目录>` 而非 `-r`
_PROJECT_MANIFESTS = ("pyproject.toml", "setup.py")


def _install_cmd(pip: str, req_file: Path, index_url: Optional[str] = None) -> list[str]:
    if req_file.name.lower() in _PROJECT_MANIFESTS:
        cmd = [pip, "install", str(req_file.parent)]  # 安装项目及其声明依赖
    else:
        cmd = [pip, "install", "-r", str(req_file)]
    if index_url:
        cmd += ["--index-url", index_url]
    return cmd


def _detect_cuda() -> Optional[str]:
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return None
        m = re.search(r"CUDA Version:\s*([\d.]+)", proc.stdout)
        return m.group(1) if m else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _python_version(python_exe: Optional[str]) -> str:
    """取目标环境解释器版本（非宿主），失败则回退宿主版本。"""
    if python_exe:
        try:
            proc = subprocess.run(
                [python_exe, "-c", "import sys;print(sys.version.split()[0])"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return sys.version.split()[0]


def detect_versions(source: Path, python_exe: Optional[str] = None) -> dict:
    """版本判断（需求一.2、2.3）：解析依赖清单与代码 import 确定语言/框架/CUDA 版本。"""
    info: dict = {"python": _python_version(python_exe), "frameworks": [], "cuda": _detect_cuda()}

    req = source / "requirements.txt"
    if req.exists():
        text = req.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            low = line.strip().lower()
            for fw in ("torch", "tensorflow", "keras", "jax", "paddlepaddle"):
                if low.startswith(fw) and not any(f.startswith(fw) for f in info["frameworks"]):
                    info["frameworks"].append(line.strip())

    for py in source.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for fw in ("torch", "tensorflow", "keras", "jax"):
            if f"import {fw}" in text or f"from {fw}" in text:
                if not any(f.startswith(fw) for f in info["frameworks"]):
                    info["frameworks"].append(fw)

    return info


def _try_install(pip: str, req_file: Optional[Path], index_url: Optional[str] = None) -> dict:
    if req_file is None or not req_file.exists():
        # 无依赖清单（如 conda 项目仅 environment.yml），跳过 pip 安装
        return {"ok": True, "error": None, "command": None, "index_url": index_url,
                "started_at": _now(), "finished_at": _now()}
    cmd = _install_cmd(pip, req_file, index_url)
    started = _now()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        ok = proc.returncode == 0
        error = None if ok else (proc.stderr or proc.stdout)[-2000:]
    except subprocess.TimeoutExpired:
        ok, error = False, "pip install 超时"
    except Exception as e:  # noqa: BLE001
        ok, error = False, str(e)
    return {"ok": ok, "error": error, "command": " ".join(cmd), "index_url": index_url,
            "started_at": started, "finished_at": _now()}


# pip 取不到版本（索引不可达/被拒）的特征；与「依赖冲突」区分：前者换源重试，后者走修正循环
_INDEX_ERROR_MARKERS = (
    "from versions: none",
    "connectionerror", "connection aborted", "connection reset", "newconnectionerror",
    "read timed out", "readtimeouterror", "temporary failure", "max retries exceeded",
    "sslerror", "certificate verify failed", "proxy",
)


def _is_index_error(error: Optional[str]) -> bool:
    if not error:
        return False
    low = error.lower()
    return any(marker in low for marker in _INDEX_ERROR_MARKERS)


def _record_install(
    project_id: str, task_id: str, attempt: int, result: dict, versions: dict, env_type: str,
    step: str = "pip_install",
) -> None:
    knowledge_service.record_run(
        {
            "project_id": project_id,
            "task_id": task_id,
            "run_type": "env_install",
            "environment": {"type": env_type, **versions},
            "params": {"attempt": attempt, "step": step, "index_url": result.get("index_url")},
            "command": result.get("command"),
            "status": "success" if result["ok"] else "failed",
            "error": result.get("error"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        }
    )


async def _agent_fix_advice(source: Path, error: str) -> dict:
    prompt = (
        f"一个项目在安装依赖时失败，错误信息如下：\n\n{error}\n\n"
        f"请阅读项目代码（目录 {source}）与依赖清单，判断代码实际 import 了哪些库，"
        "给出**修正后的完整依赖清单**（每行一条、pip requirements 语法）——按需降级/替换版本、"
        "移除未实际使用的项；仅在某版本在当前 Python 下确无可安装 wheel 时才放宽或去掉它的版本钉，"
        "不要无谓放宽。\n"
        "严格按如下 JSON 输出，顶层键必须同时为 requirements 与 reason：\n"
        '{"requirements": ["numpy", "pandas==2.2.2"], "reason": "改了哪些包、为什么"}'
    )
    result = await agent_service.run_sync(prompt, cwd=str(source), output_schema=FIX_SCHEMA)
    return result.get("structured_output") or {}


def _advice_requirements(advice) -> Optional[list[str]]:
    """从 agent 建议里取「修正后的完整依赖清单」，兼容多种形状。

    DeepSeek 下结构化输出走文件兜底、不经 schema 校验，键名/形状可能漂移
    （实测曾返回 task/root_cause/fix_recommendations 这类自有结构，导致旧逻辑静默不动作）。
    """
    if isinstance(advice, list):
        lines = [str(x).strip() for x in advice]
        return [x for x in lines if x] or None
    if isinstance(advice, dict):
        for key in ("requirements", "lines", "fixed_requirements", "dependencies", "packages"):
            val = advice.get(key)
            if isinstance(val, list):
                lines = [str(x).strip() for x in val]
                if any(lines):
                    return [x for x in lines if x]
    return None


def _legacy_advice_lines(req_file: Path, advice) -> Optional[list[str]]:
    """兼容旧的 {action, package, target_version} 形式，返回改写后的行；不可执行则 None。"""
    if not isinstance(advice, dict):
        return None
    action = advice.get("action")
    pkg = (advice.get("package") or "").strip().lower()
    if not pkg or action in (None, "none"):
        return None
    out_lines = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        if _pkg_name(line) == pkg:
            if action == "remove":
                continue
            if action == "downgrade" and advice.get("target_version"):
                out_lines.append(f"{advice['package']}=={advice['target_version']}")
                continue
            if action == "replace":
                continue
        out_lines.append(line)
    return out_lines


def _apply_advice(req_file: Optional[Path], advice) -> Optional[Path]:
    """按建议改写依赖清单，返回新文件；无可执行建议时原样返回（不产出坏文件）。

    主路径：agent 给出「修正后的完整清单」→ 直接写新文件（对任意修正都适用）。
    兼容路径：旧 {action, package, target_version} → 按行增删改。
    pyproject.toml/setup.py 类清单不按行改写（避免损坏）。
    """
    if req_file is None or not req_file.exists() or req_file.name.lower() in _PROJECT_MANIFESTS:
        return req_file
    lines = _advice_requirements(advice)
    if lines is None:
        lines = _legacy_advice_lines(req_file, advice)
    if lines is None:
        return req_file
    out = req_file.parent / f"requirements_fixed_{uuid.uuid4().hex[:8]}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _pkg_name(line: str) -> str:
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", line)
    return m.group(1).lower() if m else line.strip().lower()


def register() -> None:
    task_manager.register_handler(ENV_TASK_TYPE, _run_env_create)
