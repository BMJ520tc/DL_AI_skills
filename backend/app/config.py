"""全局配置与路径常量。

路径以 backend/ 为基准定位项目根与 data/ 运行时目录，
对齐《系统架构设计》六.1 monorepo 布局与《知识库与数据设计》二.3 数据目录。
"""
import os
import sys
from pathlib import Path

# 关闭 Claude Code CLI 自动更新（架构八.1 依赖管理），钉住内置 CLI 版本由 SDK 锁版保证
os.environ.setdefault("DISABLE_AUTOUPDATER", "1")

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# 打包形态（PyInstaller 冻结，一键封装 6.6）：数据目录默认挪到用户目录，静态产物从打包资源取。
IS_FROZEN = bool(getattr(sys, "frozen", False))


def _resolve_data_dir() -> Path:
    """数据目录解析：`DL_AI_DATA_DIR` 显式覆盖 > 打包缺省（用户数据目录）> 项目根 data/。

    打包产物解压目录可能只读/升级即被整体替换，数据必须与程序目录分离（探索稿 P4）。
    """
    env = os.getenv("DL_AI_DATA_DIR")
    if env:
        return Path(env)
    if IS_FROZEN:
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / "DL-AI-skills" / "data"
    return PROJECT_ROOT / "data"


DATA_DIR = _resolve_data_dir()
DB_PATH = DATA_DIR / "index.db"
PAPERS_DIR = DATA_DIR / "papers"
PROJECTS_DIR = DATA_DIR / "projects"
MODULES_DIR = DATA_DIR / "modules"
RUNS_DIR = DATA_DIR / "runs"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
DATASETS_DIR = DATA_DIR / "datasets"
AGENT_TASKS_DIR = DATA_DIR / "agent_tasks"

# agent 端点 (O2): 默认直连 Anthropic; 可选经 ANTHROPIC_BASE_URL 接 DeepSeek
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
DEFAULT_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL")
SMALL_MODEL = os.getenv("ANTHROPIC_DEFAULT_SMALL_MODEL")


def _detect_claude_cli() -> str | None:
    """探测原生 claude 可执行文件。

    Claude Agent SDK 出于安全拒绝执行 Windows 的 .bat/.cmd 批处理，
    需指向原生 claude.exe（CLAUDE_CLI_PATH 可覆盖）。
    """
    env = os.getenv("CLAUDE_CLI_PATH")
    if env:
        return env
    appdata = os.environ.get("APPDATA")
    if appdata:
        base = Path(appdata) / "npm" / "node_modules" / "@anthropic-ai" / "claude-code"
        candidates = [
            base / "bin" / "claude.exe",
            base / "node_modules" / "@anthropic-ai" / "claude-code-win32-x64" / "claude.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    return None


CLAUDE_CLI_PATH = _detect_claude_cli()


def _detect_conda() -> str | None:
    """探测 conda 可执行文件（CONDA_EXE 可覆盖）。"""
    env = os.getenv("CONDA_EXE")
    if env:
        return env
    userprofile = os.environ.get("USERPROFILE")
    candidates = []
    if userprofile:
        for name in ("miniconda3", "Miniconda3", "anaconda3", "Anaconda3"):
            candidates.append(Path(userprofile) / name / "Scripts" / "conda.exe")
    candidates.append(Path("C:/") / "miniconda3" / "Scripts" / "conda.exe")
    for c in candidates:
        if c.exists():
            return str(c)
    return None


CONDA_PATH = _detect_conda()

SCHEMA_VERSION = "1.0"

# 跨域（前端开发态）：Vite dev server 直连后端需要 CORS 头。
# 默认只放开本机来源（任意端口），可用 CORS_ORIGINS（逗号分隔的白名单）扩展；
# 生产部署走同源或反向代理时该配置不生效（同源请求无 Origin 校验）。
CORS_ORIGINS = [o.strip() for o in (os.getenv("CORS_ORIGINS") or "").split(",") if o.strip()]
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", r"http://(localhost|127\.0\.0\.1)(:\d+)?")

# 环境自建时的 pip 索引：PIP_INDEX_URL 显式指定主索引（未设则用 pip 自身配置）；
# 主索引取不到版本（如镜像 403）时回退到 PIP_FALLBACK_INDEX 重试，不与依赖冲突混为一谈。
PIP_INDEX_URL = os.getenv("PIP_INDEX_URL")
PIP_FALLBACK_INDEX = os.getenv("PIP_FALLBACK_INDEX", "https://pypi.org/simple")

# venv 环境创建所用的解释器：默认后端自身解释器；ENV_VENV_PYTHON 指定其他版本
# （真实项目常把依赖钉在旧 Python 上，用后端解释器会因无对应 wheel 而失败）。
ENV_VENV_PYTHON = os.getenv("ENV_VENV_PYTHON")

# 前端产物由后端同源服务（一键封装 P1）：开发默认关（沿用 vite dev/preview），
# `DL_AI_SERVE_STATIC=1` 显式开，打包形态（冻结）默认开。
SERVE_STATIC = os.getenv("DL_AI_SERVE_STATIC") == "1" or IS_FROZEN
# 静态产物目录：`DL_AI_DIST_DIR` 显式覆盖 > 打包资源（sys._MEIPASS/static）> 项目 frontend/dist。
_dist_env = os.getenv("DL_AI_DIST_DIR")
FRONTEND_DIST_DIR = (
    Path(_dist_env)
    if _dist_env
    else (Path(sys._MEIPASS) / "static" if IS_FROZEN else PROJECT_ROOT / "frontend" / "dist")
)


def ensure_data_dirs() -> None:
    """确保 data/ 下各运行时目录存在。"""
    for d in (
        DATA_DIR,
        PAPERS_DIR,
        PROJECTS_DIR,
        MODULES_DIR,
        RUNS_DIR,
        KNOWLEDGE_DIR,
        DATASETS_DIR,
        AGENT_TASKS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
