"""项目管理与工作区（模块详细设计 2.2，数据设计八.1，D12）。

两类项目: original（仅分析与运行）与 structured（画布可编辑）。
阶段1 只实现 original；structured 目录骨架预留，git 版本仓库归 7.6（阶段4）。
"""
import uuid
from datetime import datetime, timezone
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
