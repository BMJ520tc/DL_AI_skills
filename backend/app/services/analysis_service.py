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

from app.config import PROJECT_ROOT, project_env_dir
from app.services import agent_service, download_service, knowledge_service, project_manager, prompts, task_manager

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
    # 模板（agents/prompts/dynamic_analysis.md）声明 supplements 为必需字段，
    # schema 与模板对齐：缺字段不能当「没有不确定项」静默通过
    "required": ["supplements"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_source(project_id: str, source_url: str) -> None:
    """3.3 任意项目加载：仓库地址 clone，本地路径挂载（软链，避免拷贝）。

    加载成功即把项目状态推进为 loaded（前端据此停止轮询）；失败直接抛错，
    由调用方（POST /api/projects）回滚这笔半成品创建。
    """
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

    project_manager.update_status(project_id, "loaded")


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


def _project_python(ws: Path, project_id: Optional[str] = None) -> Optional[str]:
    """项目独立环境的解释器；环境未就绪返回 None（不退回宿主解释器，见 2.3 独立环境）。

    环境落在**短路径根** `<ENV_ROOT>/<项目id前8位>`（规避 Windows 260 上限，见 config.project_env_dir）；
    同时兼容迁移前的老位置 `ws/env`（旧环境仍可用）。`ws` 的末段即项目 id，未显式传入时据此推断。
    """
    pid = project_id or Path(ws).name
    for root in (project_env_dir(pid), Path(ws) / "env"):
        for candidate in (
            root / "Scripts" / "python.exe",  # Windows venv
            root / "python.exe",              # conda 环境（Windows）
            root / "bin" / "python",          # Linux/mac venv
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

# `--help` 输出解析（需求一.2 的第三种来源）——固定代码 + 正则，不调大模型
HELP_TIMEOUT_S = 20  # 单次 `--help` 探测上限（秒）；超时按「拿不到 help」处理，不报错
# usage 段头：argparse 是 `usage:`，click 是 `Usage:`，同一正则大小写不敏感匹配
_USAGE_HEAD = re.compile(r"^[ \t]*usage:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
# usage 段里的选项 token（长选项或单字母短选项；`-h`/`--help` 后面单独排除）
_HELP_OPTION = re.compile(r"(?<![\w-])(--[A-Za-z][\w-]*|-[A-Za-z])")
_HELP_OPTION_SKIP = {"-h", "--help"}
# 选项后紧跟的取值占位符（metavar），如 `--data DIR`、`--config <path>`
_HELP_METAVAR = re.compile(r"[ \t=]+([A-Za-z_<][^\s\[\]|(){}]*)")
# metavar 形如目录/数据路径时，优先用项目里真实存在的 data/ 目录作为取值
_DIR_METAVARS = {"DIR", "DIRECTORY", "PATH", "ROOT", "DATA", "DATA_DIR", "DATA_PATH", "DATASET", "INPUT"}


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


# 顶层目录里**不是产品包**的常见名字：把它们当「导入即跑通」的候选没有信息量
# （`tests/` 尤其坑：`python -c "import tests"` 只是导入空壳包，恒成功 → 假阳性，
# 会把真正的 `import scgpt` 失败稀释掉。2026-10-05 scGPT 实测）。
NON_ENTRY_DIRS = {
    "tests", "test", "testing", "docs", "doc", "examples", "example", "samples",
    "scripts", "script", "bin", "build", "dist", "tools", "notebooks", "notebook",
    "demos", "demo", "experiments", "experiment", "assets", "data", "misc", "tutorials",
}


def _package_candidates(source: Path) -> list[list[str]]:
    """库型项目（无入口脚本，如 GEARS）的兜底候选：导入顶层包即证明环境可加载。

    3.5 要求「从 README、脚本目录或 --help 输出定位最小可运行命令，跑通一遍」——
    对没有可执行入口的库，`python -c "import <包>"` 是等价的最小跑通验证。
    跳过 `NON_ENTRY_DIRS`（tests/docs/examples…）这些无信息量的目录。
    """
    out: list[list[str]] = []
    try:
        entries = sorted(source.iterdir())
    except OSError:
        return out
    for p in entries:
        name = p.name
        if (p.is_dir() and name.isidentifier() and not name.startswith(".")
                and name.lower() not in NON_ENTRY_DIRS
                and (p / "__init__.py").exists()):
            out.append(["-c", f"import {name}"])
    return out


# ---- 装完但 import 缺包（依赖清单漏声明）→ 自动补装重试 ----

_MISSING_MODULE_RE = re.compile(r"No module named '([A-Za-z0-9_.]+)'")

# 模块名 ≠ 发行包名 的常见对照（其余按「下划线→连字符」交给 pip 归一）
MODULE_PACKAGE_ALIASES = {
    "cv2": "opencv-python", "sklearn": "scikit-learn", "skimage": "scikit-image",
    "yaml": "pyyaml", "PIL": "pillow", "IPython": "ipython", "Bio": "biopython",
    "dotenv": "python-dotenv", "dateutil": "python-dateutil", "attr": "attrs",
    "OpenSSL": "pyopenssl", "serial": "pyserial", "git": "GitPython",
    "google": "protobuf", "pkg_resources": "setuptools", "win32com": "pywin32",
}

# 一次验证最多自动补装几个包（防雪崩）
MAX_MISSING_DEP_FIXES = 3


def _missing_module(error: Optional[str]) -> Optional[str]:
    """从最小命令报错里取缺失的顶层模块名（`ModuleNotFoundError: No module named 'x.y'` → `x`）。"""
    m = _MISSING_MODULE_RE.search(error or "")
    return m.group(1).split(".")[0] if m else None


def _module_to_package(module: str) -> str:
    """模块名 → pip 发行包名（有别名用别名，否则按 PEP 503 把 `_` 归一为 `-`）。"""
    return MODULE_PACKAGE_ALIASES.get(module, module.replace("_", "-"))


def _candidate(tokens: list[str], source: str) -> dict:
    """候选命令条目：命令 token 序列 + **来源标注**（需求一.2 三来源可区分，随 run_record 复核）。"""
    return {"command": list(tokens), "source": source}


def _usage_block(text: str) -> Optional[str]:
    """取 `--help` 输出里的 usage 段（argparse/click 都会打印 `usage:`/`Usage:`），没有则 None。

    usage 常被折行（续行有缩进），顶格的下一节标题（如 `options:`）或空行即结束，
    因此按「缩进续行」收集后再拼成一行，供下面的必填参数解析使用。
    """
    if not text:
        return None
    m = _USAGE_HEAD.search(text)
    if not m:
        return None
    lines = [m.group(1).strip()]
    tail = text[m.end():]
    if "\n" not in tail:
        return lines[0]
    # 跳过 usage 首行的行尾残留（正则已吃光首行），只从下一行开始看续行
    for line in tail.split("\n", 1)[1].splitlines():
        if not line.strip():
            break            # 空行结束 usage 段
        if line[0] not in " \t":
            break            # 顶格的下一节（如 options:）结束 usage 段
        lines.append(line.strip())
    return " ".join(lines)


def _usage_option_contexts(block: str) -> list[tuple[str, int, Optional[int]]]:
    """扫描 usage 段，返回 [(选项, 起始位置, 所属 `(` 必填组起点或 None)]，只含**不在 `[]` 内**的选项。

    括号用栈配对：出现在 `[]` 内的选项是可选参数（不进候选）；`(...)` 组用于识别
    argparse 的必填互斥组 `(--a A | --b B)`。纯字符串扫描，确定性、可复核。
    """
    contexts: list[tuple[str, int, Optional[int]]] = []
    stack: list[tuple[str, int]] = []
    option_at = {m.start(): m.group(0) for m in _HELP_OPTION.finditer(block)}
    i = 0
    while i < len(block):
        ch = block[i]
        if ch in "[(":
            stack.append((ch, i))
            i += 1
            continue
        if ch in ")]":
            if stack:
                stack.pop()
            i += 1
            continue
        opt = option_at.get(i)
        if opt is not None:
            if opt not in _HELP_OPTION_SKIP and not any(c == "[" for c, _ in stack):
                # 不在 [] 内即必填；此时栈里只可能有 '('
                contexts.append((opt, i, stack[-1][1] if stack else None))
            i += len(opt)
            continue
        i += 1
    return contexts


def _help_arg_value(source: Optional[Path], metavar: str) -> str:
    """把 usage 里的 metavar 变成一个可尝试的实参（不臆造真实路径）。

    优先用项目里**真实存在**的同名条目（`--data DIR` 且项目有 `data/` → `data`）；
    metavar 是目录/数据类且项目有 `data/` 时用 `data`；都取不到就返回 `<METAVAR>` 占位，
    复核者一眼能看出这是待填占位符，而不是被编出来的路径。
    """
    name = metavar.strip().strip("<>")
    if not name:
        return "<value>"
    if source is not None:
        for cand in (name, name.lower()):
            if (source / cand).exists():
                return cand
        if name.upper() in _DIR_METAVARS and (source / "data").is_dir():
            return "data"
    return f"<{name}>"


def _required_option_tokens(block: str, source: Optional[Path] = None) -> list[str]:
    """从 usage 段里取**必填选项**并带上取值，拼成可追加到入口脚本后的 token 序列。

    固定正则 + 括号深度，不调大模型：
    - `[-h] --data DIR [--epochs N]` → `["--data", "DIR"]`（可选参数不进候选；
      若项目里存在 `data/` 则取真实值 `data`）；
    - 必填互斥组 `(--a A | --b B)` 只取第一个备选（一次只试一条命令）；
    - 裸必填位置参数（如 `{train,eval}`）无法推断取值，不臆造，跳过。
    """
    tokens: list[str] = []
    seen: set[str] = set()
    chosen_groups: set[int] = set()
    for opt, pos, paren in _usage_option_contexts(block):
        if opt in seen:
            continue
        if paren is not None:
            close = block.find(")", paren)
            group = block[paren:close if close != -1 else len(block)]
            if "|" in group:
                if paren in chosen_groups:
                    continue
                chosen_groups.add(paren)
        seen.add(opt)
        tokens.append(opt)
        mv = _HELP_METAVAR.match(block, pos + len(opt))
        if mv:
            tokens.append(_help_arg_value(source, mv.group(1)))
    return tokens


def _probe_help(python: str, target: list[str], cwd: Path, timeout_s: int = HELP_TIMEOUT_S) -> str:
    """跑一次 `<python> <入口> --help` 并返回合并输出；拿不到就返回空串，**绝不抛错**。

    `--help` 只用于推断候选命令：脚本没有 --help、不可执行、超时都在这里被吞掉并按
    「拿不到 help」处理（非零退出但打印了 usage 的仍会被解析），绝不把探测失败当错误冒泡。
    """
    try:
        proc = subprocess.run(
            [python, *target, "--help"],
            cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="ignore", timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


def _help_probe_targets(source: Path) -> list[list[str]]:
    """可探测 `--help` 的入口：README 抽到的 .py / `-m 包`、根目录入口脚本、scripts/bin 入口。

    shell 脚本与库导入没有 argparse 式 usage，探测它们只会拿到空输出，不列入。
    """
    targets: list[list[str]] = []

    def _add(tokens: list[str]) -> None:
        if tokens not in targets:
            targets.append(tokens)

    readme = _extract_from_readme(source)
    if readme and (readme[0] == "-m" or (len(readme) == 1 and readme[0].endswith(".py"))):
        _add(list(readme))
    for name in ENTRY_SCRIPTS:
        if (source / name).exists():
            _add([name])
    for dirname in ENTRY_DIRS:
        root = source / dirname
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.py")):
            if p.is_file() and p.name in ENTRY_SCRIPTS:
                _add([p.relative_to(source).as_posix()])
    return targets


async def _help_candidates(source: Path, python: Optional[str]) -> list[dict]:
    """需求一.2 的第三种来源：解析 `--help` 输出，推断「入口 + 必填参数」候选命令。

    口径：
    - 只在**项目独立环境解释器**（`python`，来自 `_project_python`）下探测；未就绪直接返回空，
      绝不回退宿主解释器（2.3 独立环境）；
    - 每个入口跑一次带超时的 `--help`，用固定代码 + 正则解析 usage 段里的必填选项
      （不调大模型），据此生成候选；
    - 解析不到（脚本无 --help、输出为空、无 usage、超时）**不报错**，返回空列表，
      调用方退回 README/脚本目录既有口径；
    - 候选一律带 `source="help"`，便于复核。
    """
    if not python:
        return []
    out: list[dict] = []
    for target in _help_probe_targets(source):
        text = await asyncio.to_thread(_probe_help, python, target, source)
        block = _usage_block(text)
        if not block:
            continue
        out.append(_candidate([*target, *_required_option_tokens(block, source)], "help"))
    return out


def _static_candidates(source: Path) -> tuple[list[dict], list[dict]]:
    """既有的静态候选（不含 `--help` 输出推断），按《模块详细设计》3.5 的来源优先级排列。

    返回 `(README/入口脚本/脚本目录候选, 库型项目兜底候选)`；`--help` 输出推断出的候选
    由 `_candidate_commands` 按文档顺序（README → 脚本目录 → `--help` 输出 → 库型兜底）
    插在两者之间。入口脚本先试 `--help`（快速验证环境/导入可用），再直接运行。
    """
    cands: list[dict] = []
    readme = _extract_from_readme(source)
    if readme:
        # 从 README 抽到的若是「裸脚本」（无参数），先试 `--help`（快速且无副作用），
        # 再按原文运行——避免一上来就真的跑一遍训练（宽限期内会被判「启动成功」但仍浪费时间）
        if len(readme) == 1 and readme[0].endswith(".py"):
            cands.append(_candidate([readme[0], "--help"], "readme"))
        cands.append(_candidate(readme, "readme"))
    for name in ENTRY_SCRIPTS:
        if (source / name).exists():
            cands.append(_candidate([name, "--help"], "entry"))
    for name in ENTRY_SCRIPTS:
        if (source / name).exists():
            cands.append(_candidate([name], "entry"))
    for cand in _script_dir_candidates(source):
        cands.append(_candidate(cand, "script_dir"))
    fallback = [_candidate(cand, "package") for cand in _package_candidates(source)]
    return cands, fallback


async def _candidate_commands(source: Path, python: Optional[str] = None) -> list[dict]:
    """3.5 候选命令列表，来源优先级（《模块详细设计》3.5）：README → 脚本目录（入口脚本
    先 `--help` 变体、后直接运行）→ **`--help` 输出推断的候选** → 库型项目兜底；
    全部失败则交 agent 构造（见 `_run_verify`）。

    需求一.2 的三种来源在候选里可区分（`source`: `readme` / `script_dir` / `help`）；
    `entry`、`package` 是既有口径的补充标注，`agent` 是最后的兜底构造。
    返回项形如 `{"command": ["train.py", "--data", "data"], "source": "help"}`。
    """
    static, package_fallback = _static_candidates(source)
    help_cands = await _help_candidates(source, python)
    out: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for cand in [*static, *help_cands, *package_fallback]:
        key = tuple(cand["command"])
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


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
    if not cmd:
        return None
    tokens = cmd.split()
    # 裸解释器（如仅 "python"）不是可运行命令——等于开一个 REPL，会以「打不开 python」失败
    if len(tokens) == 1 and tokens[0] in ("python", "python3", "py", "bash", "sh"):
        return None
    return tokens


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


def _env_script(name: str, python: str) -> Optional[str]:
    """项目环境 Scripts 下的同名可执行（如 `pytest` → `<env>/Scripts/pytest.exe`），没有则 None。"""
    if not name:
        return None
    scripts = Path(python).parent
    for cand in (scripts / name, scripts / f"{name}.exe", scripts / f"{name}.bat", scripts / f"{name}.cmd"):
        if cand.exists():
            return str(cand)
    return None


async def _try_command(python: str, cmd: list[str], source: Path, grace_s: int = GRACE_S) -> dict:
    """运行一条候选命令，返回 {ok, error, mode, command, started_at, finished_at}。

    shell 脚本（`bash x.sh` 或 `x.sh`）**不能**再加 python 前缀——否则实际执行
    `python bash x.sh` 必然失败，scripts/bin 下的入口永远跑不通。
    """
    if cmd and cmd[0] in ("bash", "sh"):
        full_cmd = ["bash", *cmd[1:]]
    elif cmd and str(cmd[0]).lower().endswith((".sh", ".bash")):
        full_cmd = ["bash", *cmd]
    elif cmd and cmd[0] in ("python", "python3", "py"):
        # agent/README 常把命令写成 "python x.py"：用**项目环境解释器替换**（不是前缀，
        # 否则会执行 `<env python> python x.py` → can't open file 'python'）
        if len(cmd) == 1:
            return {"ok": False, "mode": "invalid", "error": "候选命令只有解释器、缺少脚本，已跳过",
                    "command": " ".join(cmd), "started_at": _now(), "finished_at": _now()}
        full_cmd = [python, *cmd[1:]]
    elif cmd and (str(cmd[0]).lower().endswith(".py") or cmd[0] in ("-m", "-c")
                  or (source / str(cmd[0])).is_file()):
        full_cmd = [python, *cmd]              # 本地 .py / `-m 模块` / `-c 代码` / source 下的脚本文件
    else:
        # 其它裸命令（`pytest tests/…`、`make` 等）：**不能**加 python 前缀，否则变成
        # `python pytest …` → can't open file 'pytest'（2026-10-05 scGPT 实测）；
        # 优先用项目环境 Scripts 下的同名可执行，其次按原样跑。
        exe = _env_script(cmd[0] if cmd else "", python)
        full_cmd = [exe or cmd[0], *cmd[1:]]
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
    # 候选来源（readme/script_dir/help/entry/package/agent）一并落 run_record：
    # 「最小命令是怎么定位到的」不只看命令本身，来源也要可复核（需求一.2）
    params: dict = {}
    if result.get("mode"):
        params["mode"] = result["mode"]
    if result.get("source"):
        params["source"] = result["source"]
    knowledge_service.record_run(
        {
            "project_id": project_id,
            "task_id": task_id,
            "run_type": "smoke_run",
            "command": result.get("command"),
            "params": params or None,
            "status": "success" if result.get("ok") else "failed",
            "error": result.get("error"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        }
    )


def _missing_dep_to_fix(error: Optional[str], already: list[str]) -> Optional[str]:
    """报错缺包 → 待补装的发行包名；不适用（非缺包 / 已补过 / 超上限）返回 None。"""
    if len(already) >= MAX_MISSING_DEP_FIXES:
        return None
    module = _missing_module(error)
    if not module:
        return None
    package = _module_to_package(module)
    return None if package in already else package


async def _install_missing_dep(project_id: str, task_id: str, env_dir: Path, package: str) -> bool:
    """补装缺包并留痕（run_type=env_install，step=smoke_missing_dep）；返回是否装成功。"""
    from app.services import env_manager  # 延迟导入：避开 analysis_service ↔ env_manager 环

    started = _now()
    res = await env_manager.install_missing_package(env_dir, package)
    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "env_install",
        "params": {"step": "smoke_missing_dep", "package": package},
        "command": res.get("command"), "status": "success" if res.get("ok") else "failed",
        "error": res.get("error"), "started_at": started, "finished_at": _now(),
    })
    return bool(res.get("ok"))


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
    env_dir = project_env_dir(project_id)
    fixed: list[str] = []      # 已补装过的包（防重复、限数量）
    for cand in await _candidate_commands(source, python):
        result = await _try_command(python, cand["command"], source)
        result["source"] = cand["source"]          # 逐条如实标注来源，写进 run_record
        _record_smoke(project_id, task_id, result)
        if not result["ok"]:
            # 装完但 import 缺包（依赖清单漏声明，如 scGPT 用了 IPython 却没写进 pyproject）：
            # 按缺包名补装一次再重跑该命令 → 让这类项目自愈，不必人工装包。
            pkg = _missing_dep_to_fix(result["error"], fixed)
            if pkg:
                fixed.append(pkg)
                fix = await _install_missing_dep(project_id, task_id, env_dir, pkg)
                if fix:
                    result = await _try_command(python, cand["command"], source)
                    result["source"] = cand["source"]
                    _record_smoke(project_id, task_id, result)
        if result["ok"]:
            return
        last_error = result["error"]

    # 全部候选失败 → agent 构造命令（3.5）
    agent_cmd = await _agent_construct_command(source)
    if agent_cmd:
        result = await _try_command(python, agent_cmd, source)
        result["source"] = "agent"
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

    # 任务前知识带入（需求六.1、模块详细设计 8.2「模块一加载新项目」触发点）：
    # 以项目名作为模型维度检索已确认蒸馏结论，作为默认建议进任务进度（无命中静默跳过）。
    try:
        advice = knowledge_service.bring_advice_summary(model=project.get("name"))
    except Exception:  # noqa: BLE001 —— 带入失败不影响分析
        advice = {}
    if advice:
        task_manager.update_progress(task_id, {"knowledge_bring": advice})

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
    """动态行为补充判断：模板（`agents/prompts/dynamic_analysis.md`）+ 本次 uncertain 上下文。

    模板固定角色/任务/约束/字段说明，运行期上下文（项目目录、本次 uncertain 项）与
    结构化输出 schema 由代码追加，避免模板与 schema 漂移。
    """
    prompt = prompts.render_prompt(
        "dynamic_analysis",
        context=(
            f"### 项目代码目录\n\n{source}\n\n"
            "### 静态扫描的 uncertain 项\n\n"
            f"{json.dumps(uncertain, ensure_ascii=False, indent=2)}"
        ),
        schema=DYNAMIC_SCHEMA,
    )
    result = await agent_service.run_sync(prompt, cwd=str(source), output_schema=DYNAMIC_SCHEMA)
    return result.get("structured_output") or {}


def register() -> None:
    task_manager.register_handler(VERIFY_TASK_TYPE, _run_verify)
    task_manager.register_handler(ANALYZE_TASK_TYPE, _run_analyze)
