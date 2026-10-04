"""项目管理与工作区（模块详细设计 2.2，数据设计八.1，D12）。

两类项目: original（仅分析与运行）与 structured（画布可编辑）。
阶段1 只实现 original；structured 目录骨架预留，git 版本仓库归 7.6（阶段4）。
"""
import os
import stat
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import PROJECTS_DIR
from app.db.connection import get_connection

ORIGINAL_SUBDIRS = ["source", "env", "data", "reports", "runs"]
STRUCTURED_SUBDIRS = ["runs", "exports"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_project(
    project_type: str,
    source: Optional[str] = None,
    name: Optional[str] = None,
    parent_project_id: Optional[str] = None,
) -> str:
    if project_type not in ("original", "structured"):
        raise ValueError(f"invalid project_type: {project_type}")

    project_id = uuid.uuid4().hex
    ws = PROJECTS_DIR / project_id
    subdirs = ORIGINAL_SUBDIRS if project_type == "original" else STRUCTURED_SUBDIRS
    for d in subdirs:
        (ws / d).mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO project(project_id, project_type, source, parent_project_id, name, "
            "status, workspace_path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'loading', ?, ?, ?)",
            (project_id, project_type, source, parent_project_id, name, str(ws), _now(), _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return project_id


def get_project(project_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM project WHERE project_id = ?", (project_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def list_projects(project_type: Optional[str] = None) -> list[dict]:
    conn = get_connection()
    try:
        if project_type:
            rows = conn.execute(
                "SELECT * FROM project WHERE project_type = ? ORDER BY created_at DESC", (project_type,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM project ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def update_status(project_id: str, status: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE project SET status = ?, updated_at = ? WHERE project_id = ?",
            (status, _now(), project_id),
        )
        conn.commit()
    finally:
        conn.close()


def _is_link(path: Path) -> bool:
    """符号链接或 Windows 目录联接（junction）判定——两者都只能删链接本身，不能递归进去。"""
    try:
        if path.is_symlink():
            return True
        attrs = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return True  # 判不准时按链接处理（保守：宁可不删，也不跟随删除用户数据）
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and attrs & flag)


def _remove_link(path: Path) -> bool:
    """只删除链接本身（目录符号链接/联接用 os.rmdir 删重解析点，绝不进入目标）。"""
    try:
        os.rmdir(path)
        return True
    except OSError:
        pass
    try:
        path.unlink()
        return True
    except OSError:
        pass
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["cmd", "/c", "rmdir", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            return proc.returncode == 0
        except OSError:
            return False
    return False


def _force_unlink(path: Path) -> bool:
    """删除文件；Windows 上 git 对象/包文件是只读的，先清只读位再删。

    2026-10-04 实测缺陷：结构化项目工作区里含 `.git`，其 objects 下的文件带只读属性，
    原来的 `p.unlink()` 抛 PermissionError 后 `continue` 跳过 → 目录非空 → `rmdir` 失败，
    结果「删项目」只删了数据库记录、工作区（含版本历史）留在盘上。
    """
    try:
        path.unlink()
        return True
    except PermissionError:
        try:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()
            return True
        except OSError:
            return False
    except OSError:
        return False


def _force_rmdir(path: Path) -> bool:
    """删除空目录；同样先清只读位（Windows 目录也可能带只读属性）。"""
    try:
        path.rmdir()
        return True
    except PermissionError:
        try:
            os.chmod(path, stat.S_IWRITE)
            path.rmdir()
            return True
        except OSError:
            return False
    except OSError:
        return False


def _remove_dir_safely(root: Path) -> None:
    """递归删除目录树，但遇到符号链接/联接只删链接本身。

    任意项目加载会把工作区 `source` 软链到用户本机目录（_mount_local）。shutil.rmtree
    对目录联接（junction）的处理随 Python 版本而异（3.13 起不再跟随，更早版本会递归进
    目标删用户数据），故此处自行遍历、逐个判定，不依赖该行为；删不掉就留下，绝不越界。
    只读文件/目录（git 对象等）先清只读位再删，否则 Windows 上会留下半个工作区。
    """
    if _is_link(root):
        _remove_link(root)
        return
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        p = Path(entry.path)
        if _is_link(p):
            _remove_link(p)
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                _remove_dir_safely(p)
            else:
                _force_unlink(p)
        except OSError:
            continue
    _force_rmdir(root)


def delete_project(project_id: str) -> bool:
    """删除项目记录与工作区目录（失败补偿用，如入库链路中途失败要清掉半成品结构化项目）。

    工作区可能含指向用户本机目录的符号链接/联接（任意项目加载的本地挂载），
    删除时只删链接本身，安全性见 _remove_dir_safely。
    """
    project = get_project(project_id)
    if project is None:
        return False
    conn = get_connection()
    try:
        conn.execute("DELETE FROM project WHERE project_id = ?", (project_id,))
        conn.commit()
    finally:
        conn.close()
    ws = project.get("workspace_path")
    if ws:
        _remove_dir_safely(Path(ws))
    return True


def require_type(project_id: str, allowed_types: set[str]) -> dict:
    """类型权限校验（2.2）：分析/复现/拆解仅接受 original，画布/版本仅接受 structured。"""
    project = get_project(project_id)
    if project is None:
        raise LookupError("project not found")
    if project["project_type"] not in allowed_types:
        raise PermissionError(
            f"project_type={project['project_type']} not allowed, expected one of {sorted(allowed_types)}"
        )
    return project
