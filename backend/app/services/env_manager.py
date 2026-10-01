"""环境管理（模块详细设计 2.3）。

决策树: Dockerfile → 容器（可选扩展，需本机 docker）; environment.yml → conda; 否则 venv。
依赖修正循环: 安装失败 → agent 判断 → 降级/替换/移除 → 重试（上限 3 次）。
环境创建步骤与每次安装尝试写 run_record(env_install)，报错入 error 字段（供蒸馏知识提炼，需求六.1）。
"""
import asyncio
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import CONDA_PATH
from app.services import agent_service, knowledge_service, project_manager, task_manager

ENV_TASK_TYPE = "env_create"

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["downgrade", "remove", "replace", "none"]},
        "package": {"type": "string"},
        "target_version": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
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
        raise RuntimeError("项目带 Dockerfile，需容器环境；容器为可选扩展，未检测到 docker（本机未安装或未启动）")

    env_dir = ws / "env"
    created_at = _now()
    if env_type == "conda":
        cmd = _conda_create_cmd(source, env_dir)
    else:
        cmd = [sys.executable, "-m", "venv", str(env_dir)]
    try:
        await asyncio.to_thread(subprocess.run, cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"")
        err = err.decode(errors="ignore")[-2000:] if isinstance(err, bytes) else str(err)[-2000:]
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


async def _install_with_fix(
    source: Path, env_dir: Path, project_id: str, task_id: str, env_type: str
) -> bool:
    pip = _env_pip(env_dir)
    req_file = _find_requirements(source)
    versions = detect_versions(source, _env_python(env_dir))

    for attempt in range(1, 4):
        result = await asyncio.to_thread(_try_install, pip, req_file)
        _record_install(project_id, task_id, attempt, result, versions, env_type)
        if result["ok"]:
            return True
        if attempt >= 3:
            break
        advice = await _agent_fix_advice(source, result["error"])
        req_file = _apply_advice(req_file, advice)
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


def _try_install(pip: str, req_file: Optional[Path]) -> dict:
    if req_file is None or not req_file.exists():
        # 无依赖清单（如 conda 项目仅 environment.yml），跳过 pip 安装
        return {"ok": True, "error": None, "command": None, "started_at": _now(), "finished_at": _now()}
    cmd = [pip, "install", "-r", str(req_file)]
    started = _now()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        ok = proc.returncode == 0
        error = None if ok else (proc.stderr or proc.stdout)[-2000:]
    except subprocess.TimeoutExpired:
        ok, error = False, "pip install 超时"
    except Exception as e:  # noqa: BLE001
        ok, error = False, str(e)
    return {"ok": ok, "error": error, "command": " ".join(cmd), "started_at": started, "finished_at": _now()}


def _record_install(
    project_id: str, task_id: str, attempt: int, result: dict, versions: dict, env_type: str
) -> None:
    knowledge_service.record_run(
        {
            "project_id": project_id,
            "task_id": task_id,
            "run_type": "env_install",
            "environment": {"type": env_type, **versions},
            "params": {"attempt": attempt, "step": "pip_install"},
            "command": result.get("command"),
            "status": "success" if result["ok"] else "failed",
            "error": result.get("error"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        }
    )


async def _agent_fix_advice(source: Path, error: str) -> dict:
    prompt = (
        f"一个深度学习项目在安装依赖时失败，错误信息如下：\n\n{error}\n\n"
        f"请阅读项目代码（目录 {source}），判断代码实际 import 了哪些库，"
        "哪些依赖版本不匹配、可降级，哪些依赖实际未使用可移除，并给出修复建议。"
    )
    result = await agent_service.run_sync(prompt, cwd=str(source), output_schema=FIX_SCHEMA)
    return result.get("structured_output") or {}


def _apply_advice(req_file: Optional[Path], advice: dict) -> Optional[Path]:
    action = advice.get("action")
    pkg = (advice.get("package") or "").strip().lower()
    if not pkg or action in (None, "none"):
        return req_file

    lines = (req_file.read_text(encoding="utf-8").splitlines() if req_file and req_file.exists() else [])
    out_lines = []
    for line in lines:
        if _pkg_name(line) == pkg:
            if action == "remove":
                continue
            if action == "downgrade" and advice.get("target_version"):
                out_lines.append(f"{advice['package']}=={advice['target_version']}")
                continue
            if action == "replace":
                continue
        out_lines.append(line)

    out = (req_file.parent if req_file else Path(".")) / f"requirements_fixed_{uuid.uuid4().hex[:8]}.txt"
    out.write_text("\n".join(out_lines), encoding="utf-8")
    return out


def _pkg_name(line: str) -> str:
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", line)
    return m.group(1).lower() if m else line.strip().lower()


def register() -> None:
    task_manager.register_handler(ENV_TASK_TYPE, _run_env_create)
