"""模块一全流程（模块详细设计 3.3~3.7）。

任意项目加载（clone/挂载）、最小命令验证（3.5）、静态结构扫描（3.6）、
动态行为补充（3.7，仅当报告存在 uncertain 项时触发 agent）。
"""
import asyncio
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import PROJECT_ROOT
from app.services import agent_service, download_service, knowledge_service, project_manager, task_manager

VERIFY_TASK_TYPE = "verify"
ANALYZE_TASK_TYPE = "analyze"

SCAN_SCRIPT = PROJECT_ROOT / "scripts" / "scan_structure.py"

DYNAMIC_SCHEMA = {
    "type": "object",
    "properties": {
        "supplements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "reason": {"type": "string"},
                    "judgement": {"type": "string"},
                },
            },
        },
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_source(project_id: str, source_url: str) -> None:
    """3.3 任意项目加载：仓库地址 clone，本地路径挂载（软链，避免拷贝）。"""
    project = project_manager.get_project(project_id)
    source_dir = Path(project["workspace_path"]) / "source"

    if source_url.startswith(("http://", "https://", "git@", "git://", "ssh://")):
        if not download_service.verify_repo(source_url):
            raise RuntimeError(f"仓库不可达: {source_url}")
        download_service.clone_repo(source_url, source_dir)
    else:
        local = Path(source_url)
        if not local.exists():
            raise RuntimeError(f"本地路径不存在: {source_url}")
        _mount_local(local, source_dir)


def _mount_local(local: Path, source_dir: Path) -> None:
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if source_dir.exists():
        try:
            source_dir.rmdir()
        except OSError:
            pass
    try:
        source_dir.symlink_to(local, target_is_directory=True)
    except OSError:
        # Windows 无符号链接权限时回退复制（文档要求软链避免拷贝，此为兜底）
        import shutil

        shutil.copytree(local, str(source_dir))


def verify(project_id: str) -> str:
    project_manager.require_type(project_id, {"original"})
    return task_manager.create_task(VERIFY_TASK_TYPE, project_id=project_id, params={"project_id": project_id})


def analyze(project_id: str) -> str:
    project_manager.require_type(project_id, {"original"})
    return task_manager.create_task(ANALYZE_TASK_TYPE, project_id=project_id, params={"project_id": project_id})


def get_report(project_id: str) -> Optional[dict]:
    project = project_manager.get_project(project_id)
    if project is None:
        return None
    report_path = Path(project["workspace_path"]) / "reports" / "structure_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def _project_python(ws: Path) -> str:
    for candidate in (
        ws / "env" / "Scripts" / "python.exe",  # Windows venv
        ws / "env" / "python.exe",              # conda 环境（Windows）
        ws / "env" / "bin" / "python",          # Linux/mac venv
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


CMD_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string", "description": "命令行，如 python train.py"}},
    "required": ["command"],
}


def _locate_command(source: Path) -> Optional[list[str]]:
    """3.5 候选命令定位，优先级：README → 脚本目录 → setup.py console_scripts。"""
    cmd = _extract_from_readme(source)
    if cmd:
        return cmd
    for name in ("train.py", "main.py", "run.py"):
        if (source / name).exists():
            return [name]
    return None


def _extract_from_readme(source: Path) -> Optional[list[str]]:
    for name in ("README.md", "README.rst", "README", "readme.md", "README.txt"):
        p = source / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"(?:python|python3)\s+(-m\s+[\w.]+)", text)
        if m:
            return ["-m", m.group(1)]
        m = re.search(r"(?:python|python3)\s+([\w./-]+\.py(?:\s+[^\n\r`]*)?)", text)
        if m:
            return m.group(1).split()
    return None


async def _agent_construct_command(source: Path) -> Optional[list[str]]:
    prompt = (
        f"阅读项目代码（目录 {source}），确定最小可运行命令（用于验证代码能否跑通）。"
        "只返回命令行本身（如 python train.py），不要解释。"
    )
    result = await agent_service.run_sync(prompt, cwd=str(source), output_schema=CMD_SCHEMA)
    cmd = (result.get("structured_output") or {}).get("command")
    return cmd.split() if cmd else None


async def _run_verify(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    ws = Path(project["workspace_path"])
    source = ws / "source"

    cmd = _locate_command(source)
    if cmd is None:
        cmd = await _agent_construct_command(source)
    if cmd is None:
        knowledge_service.record_run(
            {"project_id": project_id, "task_id": task_id, "run_type": "smoke_run",
             "command": None, "status": "failed", "error": "无法定位最小可运行命令"}
        )
        raise RuntimeError("无法定位最小可运行命令")

    python = _project_python(ws)
    full_cmd = [python, *cmd]
    started = _now()
    proc = await asyncio.to_thread(
        subprocess.run, full_cmd, capture_output=True, text=True, timeout=300, cwd=str(source)
    )
    ok = proc.returncode == 0
    error = None if ok else (proc.stderr or proc.stdout)[-2000:]

    knowledge_service.record_run(
        {
            "project_id": project_id,
            "task_id": task_id,
            "run_type": "smoke_run",
            "command": " ".join(full_cmd),
            "status": "success" if ok else "failed",
            "error": error,
            "started_at": started,
            "finished_at": _now(),
        }
    )
    if not ok:
        raise RuntimeError(f"最小命令运行失败: {error}")


async def _run_analyze(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    ws = Path(project["workspace_path"])
    source = ws / "source"
    report_path = ws / "reports" / "structure_report.json"

    await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(SCAN_SCRIPT), str(source), str(report_path)],
        check=True,
        capture_output=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if report.get("uncertain"):
        supplement = await _dynamic_supplement(source, report["uncertain"])
        report["dynamic_supplement"] = supplement
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    project_manager.update_status(project_id, "analyzed")


async def _dynamic_supplement(source: Path, uncertain: list) -> dict:
    prompt = (
        f"以下是静态扫描无法确定的点（动态构建/条件分支），请阅读项目代码（{source}）补充判断：\n\n"
        f"{json.dumps(uncertain, ensure_ascii=False)}\n\n"
        "对每个 uncertain 项给出判断结果（所在文件、原因、判断结论），不要臆造。"
    )
    result = await agent_service.run_sync(prompt, cwd=str(source), output_schema=DYNAMIC_SCHEMA)
    return result.get("structured_output") or {}


def register() -> None:
    task_manager.register_handler(VERIFY_TASK_TYPE, _run_verify)
    task_manager.register_handler(ANALYZE_TASK_TYPE, _run_analyze)
