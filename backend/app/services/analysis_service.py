"""模块一全流程（模块详细设计 3.3~3.7）。

任意项目加载（clone/挂载）、最小命令验证（3.5）、静态结构扫描（3.6）、
动态行为补充（3.7，仅当报告存在 uncertain 项时触发 agent）。
"""
import asyncio
import json
import re
import subprocess
import sys
import threading
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


def _project_python(ws: Path) -> Optional[str]:
    """项目独立环境的解释器；环境未就绪返回 None（不退回宿主解释器，见 2.3 独立环境）。"""
    for candidate in (
        ws / "env" / "Scripts" / "python.exe",  # Windows venv
        ws / "env" / "python.exe",              # conda 环境（Windows）
        ws / "env" / "bin" / "python",          # Linux/mac venv
    ):
        if candidate.exists():
            return str(candidate)
    return None


CMD_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string", "description": "命令行，如 python train.py"}},
    "required": ["command"],
}


ENTRY_SCRIPTS = ("train.py", "main.py", "run.py")
# 「脚本目录」来源：扫描这些子目录下的常见入口（*.sh 与 ENTRY_SCRIPTS 中的 .py）
ENTRY_DIRS = ("scripts", "bin")


def _script_dir_candidates(source: Path) -> list[list[str]]:
    """扫描 scripts/、bin/ 下的常见入口：*.sh 用 bash 运行，main.py/run.py/train.py 直接运行。

    每个 .py 入口先给 `--help` 变体（快速验证环境/导入可用），再给直接运行变体。
    """
    found: list[list[str]] = []
    for dirname in ENTRY_DIRS:
        root = source / dirname
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(source).as_posix()
            if p.suffix.lower() == ".sh":
                found.append(["bash", rel])
            elif p.suffix.lower() == ".py" and p.name in ENTRY_SCRIPTS:
                found.append([rel, "--help"])
                found.append([rel])
    return found


def _candidate_commands(source: Path) -> list[list[str]]:
    """3.5 候选命令列表，按优先级：README → 根目录入口脚本 → scripts/bin 脚本目录。

    入口脚本先试 --help（快速验证环境/导入可用），再直接运行；三来源合并去重。
    全部失败则交 agent 构造（见 _run_verify）。
    """
    cands: list[list[str]] = []
    readme = _extract_from_readme(source)
    if readme:
        cands.append(readme)
    for name in ENTRY_SCRIPTS:
        if (source / name).exists():
            cands.append([name, "--help"])
    for name in ENTRY_SCRIPTS:
        if (source / name).exists() and [name] not in cands:
            cands.append([name])
    for cand in _script_dir_candidates(source):
        if cand not in cands:
            cands.append(cand)
    return cands


def _extract_from_readme(source: Path) -> Optional[list[str]]:
    for name in ("README.md", "README.rst", "README", "readme.md", "README.txt"):
        p = source / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        # 只取同一行内的脚本 token，避免 \s 跨行吞入后续 markdown 文本
        m = re.search(r"(?:python|python3)[ \t]+(-m[ \t]+[\w.]+)", text)
        if m:
            return m.group(1).split()
        m = re.search(r"(?:python|python3)[ \t]+([\w./-]+\.py)", text)
        if m:
            return [m.group(1)]
    return None


async def _agent_construct_command(source: Path) -> Optional[list[str]]:
    prompt = (
        f"阅读项目代码（目录 {source}），确定最小可运行命令（用于验证代码能否跑通）。"
        "只返回命令行本身（如 python train.py），不要解释。"
    )
    result = await agent_service.run_sync(prompt, cwd=str(source), output_schema=CMD_SCHEMA)
    cmd = (result.get("structured_output") or {}).get("command")
    return cmd.split() if cmd else None


GRACE_S = 60  # 启动判定宽限期（秒）：训练型项目启动满该时长且无导入/环境错误即判通过

_ENV_ERROR_PATTERNS = (
    "ModuleNotFoundError",
    "ImportError",
    "No module named",
    "is not recognized",   # Windows 命令未找到
    "command not found",
)


def _detect_env_error(text: str) -> Optional[str]:
    for pat in _ENV_ERROR_PATTERNS:
        if pat in text:
            return pat
    return None


def _run_with_grace(full_cmd: list[str], cwd: str, grace_s: int) -> dict:
    """运行命令并在宽限期内判定通过与否（3.5「跑通判定随项目类型」）。

    - 宽限期内退出且退出码 0 → 通过（快速脚本走此路径）；
    - 宽限期内退出且非 0 → 失败（附输出尾部）；
    - 宽限期满仍在运行且无导入/环境错误 → 视为「启动成功」，终止进程后判通过（训练型项目）；
    - 宽限期满仍在运行但输出含导入/环境错误 → 失败。
    """
    proc = subprocess.Popen(
        full_cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="ignore"
    )
    buf: list[str] = []

    def _reader() -> None:
        try:
            for line in proc.stdout:
                buf.append(line)
        except (ValueError, OSError):
            pass

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    try:
        proc.wait(timeout=grace_s)
        th.join(timeout=5)
        out = "".join(buf)
        ok = proc.returncode == 0
        return {"ok": ok, "error": None if ok else (out[-2000:] or f"退出码 {proc.returncode}"),
                "mode": "exited"}
    except subprocess.TimeoutExpired:
        th.join(timeout=2)
        out = "".join(buf)
        env_err = _detect_env_error(out)
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        if env_err:
            return {"ok": False, "error": f"启动即报错({env_err}):\n{out[-1500:]}", "mode": "startup_error"}
        return {"ok": True, "error": None, "mode": "started"}


async def _try_command(python: str, cmd: list[str], source: Path, grace_s: int = GRACE_S) -> dict:
    """运行一条候选命令，返回 {ok, error, mode, command, started_at, finished_at}。

    shell 脚本（`bash x.sh` 或 `x.sh`）**不能**再加 python 前缀——否则实际执行
    `python bash x.sh` 必然失败，scripts/bin 下的入口永远跑不通。
    """
    if cmd and cmd[0] in ("bash", "sh"):
        full_cmd = ["bash", *cmd[1:]]
    elif cmd and str(cmd[0]).lower().endswith((".sh", ".bash")):
        full_cmd = ["bash", *cmd]
    else:
        full_cmd = [python, *cmd]
    started = _now()
    try:
        result = await asyncio.to_thread(_run_with_grace, full_cmd, str(source), grace_s)
    except Exception as e:  # noqa: BLE001
        result = {"ok": False, "error": str(e), "mode": "exception"}
    result["command"] = " ".join(full_cmd)
    result["started_at"] = started
    result["finished_at"] = _now()
    return result


def _record_smoke(project_id: str, task_id: str, result: dict) -> None:
    knowledge_service.record_run(
        {
            "project_id": project_id,
            "task_id": task_id,
            "run_type": "smoke_run",
            "command": result.get("command"),
            "params": {"mode": result.get("mode")} if result.get("mode") else None,
            "status": "success" if result.get("ok") else "failed",
            "error": result.get("error"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        }
    )


async def _run_verify(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    ws = Path(project["workspace_path"])
    source = ws / "source"
    python = _project_python(ws)
    if python is None:
        _record_smoke(
            project_id, task_id,
            {"command": None, "ok": False,
             "error": "项目环境未就绪：未找到独立环境解释器（env 未创建成功）",
             "started_at": _now(), "finished_at": _now()},
        )
        raise RuntimeError("项目环境未就绪，无法在独立环境运行最小命令")

    last_error: Optional[str] = None
    for cmd in _candidate_commands(source):
        result = await _try_command(python, cmd, source)
        _record_smoke(project_id, task_id, result)
        if result["ok"]:
            return
        last_error = result["error"]

    # 全部候选失败 → agent 构造命令（3.5）
    agent_cmd = await _agent_construct_command(source)
    if agent_cmd:
        result = await _try_command(python, agent_cmd, source)
        _record_smoke(project_id, task_id, result)
        if result["ok"]:
            return
        last_error = result["error"]

    _record_smoke(
        project_id, task_id,
        {"command": None, "ok": False,
         "error": f"全部候选命令失败: {last_error}" if last_error else "无法定位最小可运行命令",
         "started_at": _now(), "finished_at": _now()},
    )
    raise RuntimeError(f"最小命令运行失败: {last_error or '无法定位最小可运行命令'}")


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
