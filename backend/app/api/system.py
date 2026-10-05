"""系统自检端点（一键封装 6.6-a：首启自检引导）。

GET /api/system/env-check 每次**现查**（学生装完 git/Python 后点「重新检测」要能立刻看到）：
git / python（含 py launcher）/ conda / 数据目录可写性 / 静态服务 / 凭证配置 / pip 镜像。
"""
import os
import shutil
import subprocess
import sys

from fastapi import APIRouter

from app import settings_store
from app.config import DATA_DIR, PIP_INDEX_URL, SERVE_STATIC, _detect_conda
from app.services import long_paths

router = APIRouter(prefix="/api/system", tags=["system"])


def _git_info() -> dict:
    path = shutil.which("git")
    if not path:
        return {"found": False, "path": None, "version": None}
    version = None
    try:
        proc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
        version = (proc.stdout or proc.stderr).strip() or None
    except Exception:  # noqa: BLE001 —— 自检端点只报告，不因探测失败而报错
        pass
    return {"found": True, "path": path, "version": version}


def _python_info() -> dict:
    # venv 分支建环境用；py launcher 也算（`py -m venv` 可指定版本）
    return {
        "found": bool(shutil.which("python") or shutil.which("py")),
        "python": shutil.which("python"),
        "py_launcher": shutil.which("py"),
    }


def _claude_cli_info() -> dict:
    """Claude Code CLI 检测（大模型步骤经 claude_agent_sdk spawn 它）——**学生自装**，缺则引导。

    SDK 不带 CLI 二进制（`_bundled/` 为空），平台从 `CLAUDE_CLI_PATH` 或 npm 安装位探测；
    缺了它，复现/拆解/蒸馏/助手等大模型步骤会如实失败（规矩 7）。
    """
    from app.config import _detect_claude_cli

    path = _detect_claude_cli()
    version = None
    if path:
        try:
            proc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=15)
            version = (proc.stdout or proc.stderr).strip() or None
        except Exception:  # noqa: BLE001 —— 自检只报告
            pass
    return {"found": bool(path), "path": path, "version": version}


def _data_dir_info() -> dict:
    writable = False
    try:
        probe = DATA_DIR / ".writable_probe"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError:
        pass
    return {"path": str(DATA_DIR), "writable": writable}


@router.get("/env-check")
def env_check() -> dict:
    conda_path = _detect_conda()  # 现查（含 CONDA_EXE 覆盖），学生装完 conda 点「重新检测」即生效
    return {
        "git": _git_info(),
        "python": _python_info(),
        "conda": {"found": bool(conda_path), "path": conda_path},
        "claude_cli": _claude_cli_info(),
        "data_dir": _data_dir_info(),
        "static_served": SERVE_STATIC,
        "credentials_configured": settings_store.status()["configured"],
        "pip_index": PIP_INDEX_URL,
        "long_paths_enabled": long_paths.is_enabled(),
    }


@router.get("/long-paths")
def long_paths_status() -> dict:
    """Windows 长路径支持是否已开启（装深层依赖 >260 字符会失败，见 services/long_paths）。"""
    return {"enabled": long_paths.is_enabled(), "platform": sys.platform}


@router.post("/long-paths/enable")
def long_paths_enable() -> dict:
    """经 UAC 提权把 LongPathsEnabled 置 1（前端点「开启长路径支持」时调用）。"""
    return long_paths.enable()
