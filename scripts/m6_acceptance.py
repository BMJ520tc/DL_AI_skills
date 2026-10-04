# scripts/m6_acceptance.py — M6 验收驱动脚本（阶段5，模块六 自迭代闭环）。
#
# 严口径：自带临时库后端（端口 8022，不碰真实 index.db），用**真实任务**驱动闭环：
#   M6-1 闭环走通（知识入库 → 检索 → 带入新任务 → 运行 → 蒸馏回写）
#     · 用真实 env_create 任务失败路径**确定性**产出蒸馏草稿（env_manager 的 dependency_conflict，
#       不依赖大模型端点）；确认 → 检索 → POST /api/knowledge/bring 命中。
#   M6-2 依赖冲突预警在安装前生效并绕开真实冲突
#     · 预置一条**已确认**、可操作的依赖冲突（pkg=six, resolution=six==1.16.0）；
#       目标项目 requirements 钉了一个**不存在**的版本 six==0.0.0（真冲突）；
#       env_create 在安装前 precheck 命中 → 预应用版本钉 → 安装成功（真的绕开了冲突）。
#
# 用法:
#     D:\python.exe scripts/m6_acceptance.py

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
BACKEND_PORT = 8022
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

FAIL_PROJECT = "m6ac-fail-0001"
CONFLICT_PROJECT = "m6ac-conflict-0001"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def http_json(path: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{BACKEND_URL}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_task(task_id: str, timeout: float = 600.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = http_json(f"/api/tasks/{task_id}")
        if task.get("status") in ("success", "failed", "cancelled"):
            return task
        time.sleep(1.0)
    return http_json(f"/api/tasks/{task_id}")


def make_original_project(tmp_dir: Path, project_id: str, name: str, requirements: str) -> None:
    sys.path.insert(0, str(BACKEND))
    from app.db import connection
    from app.services import project_manager

    ws = Path(project_manager.PROJECTS_DIR) / project_id
    (ws / "source").mkdir(parents=True, exist_ok=True)
    (ws / "source" / "requirements.txt").write_text(requirements, encoding="utf-8")
    now = datetime.now().isoformat()
    conn = connection.get_connection()
    conn.execute(
        "INSERT INTO project(project_id, project_type, source, name, status, workspace_path, "
        "created_at, updated_at, schema_version) VALUES (?, 'original', 'local', ?, 'ready', ?, ?, ?, '1.0')",
        (project_id, name, str(ws), now, now),
    )
    conn.commit()
    conn.close()


def start_backend(tmp_dir: Path):
    sys.path.insert(0, str(BACKEND))
    from app.db import connection
    from app.services import project_manager

    connection.DB_PATH = tmp_dir / "index.db"
    connection.init_db()
    project_manager.PROJECTS_DIR = tmp_dir / "projects"
    Path(project_manager.PROJECTS_DIR).mkdir(parents=True, exist_ok=True)

    import uvicorn
    from app.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=BACKEND_PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(120):
        try:
            http_json("/api/projects")
            return server
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("backend did not start")


def main() -> int:
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="m6ac_"))
    print(f"[env] 临时工作目录：{tmp_dir}", flush=True)
    try:
        start_backend(tmp_dir)
        check("临时库后端就绪（不碰真实 index.db）", True)

        make_original_project(tmp_dir, FAIL_PROJECT, "M6 失败项目", "invalid requirement line\n")
        make_original_project(tmp_dir, CONFLICT_PROJECT, "M6 冲突项目", "six==0.0.0\n")

        # ---------------- M6-1 闭环：蒸馏回写（确定性） ----------------
        r = http_json(f"/api/projects/{FAIL_PROJECT}/env", method="POST")
        fail_task_id = r["task_id"]
        task = wait_task(fail_task_id)
        check("环境创建任务确实失败（无效依赖）", task.get("status") == "failed", task.get("error", "")[:80])

        drafts = http_json("/api/knowledge/list?data_type=knowledge&status=draft")
        draft = next((d for d in drafts if draft_source_task(d) == fail_task_id), None)
        check("任务结束后确定性蒸馏回写（dependency_conflict 草稿）", draft is not None,
              f"草稿数 {len(drafts)}")
        if draft is None:
            return finish()

        # 确认 → 检索 → 带入（闭环其余环节）
        http_json(f"/api/knowledge/confirm/{draft['knowledge_id']}?supersede=true", method="POST")
        item = http_json(f"/api/knowledge/items/knowledge/{draft['knowledge_id']}")
        check("草稿确认 → confirmed", item.get("status") == "confirmed")

        hits = http_json("/api/knowledge/search?types=knowledge&limit=200")
        check("检索命中该蒸馏知识", any(h["ref_id"] == draft["knowledge_id"] for h in hits),
              f"命中 {len(hits)} 条")

        brought = http_json("/api/knowledge/bring", method="POST")
        check("任务前带入命中该知识（bring）",
              any(k.get("knowledge_id") == draft["knowledge_id"] for k in brought.get("dependency_conflict", [])),
              f"dependency_conflict {len(brought.get('dependency_conflict', []))} 条")

        # ---------------- M6-2 安装前预检 + 绕开真实冲突 ----------------
        http_json("/api/knowledge/ingest", method="POST", body={
            "data_type": "knowledge",
            "data": {"type": "dependency_conflict", "title": "six 版本冲突（通用）",
                     "content": "six==0.0.0 不存在，改用 six==1.16.0。",
                     "structured": {"pkg": "six", "resolution": "six==1.16.0"},
                     "confidence": "high", "status": "confirmed"},
        })
        r = http_json(f"/api/projects/{CONFLICT_PROJECT}/env", method="POST")
        ctask = wait_task(r["task_id"])
        progress = ctask.get("progress")
        prog = json.loads(progress) if isinstance(progress, str) and progress else (progress or {})
        check("安装前预检生效（env_precheck 在任务进度里）", "env_precheck" in prog,
              f"progress keys: {sorted(prog) if isinstance(prog, dict) else progress}")
        check("预应用版本钉（env_precheck_applied）", "env_precheck_applied" in prog)
        check("绕开真实冲突：环境创建成功（env_ready）", ctask.get("status") == "success",
              (ctask.get("error") or "")[:120])

        return finish()
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def draft_source_task(draft: dict) -> str | None:
    s = draft.get("structured")
    if isinstance(s, str):
        try:
            s = json.loads(s)
        except (ValueError, TypeError):
            return None
    return s.get("source_task_id") if isinstance(s, dict) else None


def finish() -> int:
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n=== M6 验收：{passed}/{total} 通过 ===", flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
