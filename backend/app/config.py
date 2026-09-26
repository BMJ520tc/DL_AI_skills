"""全局配置与路径常量。

路径以 backend/ 为基准定位项目根与 data/ 运行时目录，
对齐《系统架构设计》六.1 monorepo 布局与《知识库与数据设计》二.3 数据目录。
"""
import os
from pathlib import Path

# 关闭 Claude Code CLI 自动更新（架构八.1 依赖管理），钉住内置 CLI 版本由 SDK 锁版保证
os.environ.setdefault("DISABLE_AUTOUPDATER", "1")

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
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
