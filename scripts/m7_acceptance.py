# scripts/m7_acceptance.py — M7 验收驱动脚本（阶段5，模块六 多模型综合分析，需求六.2 扩展）。
#
# 自带临时库后端（端口 8023，不碰真实 index.db），用**真实端点/任务队列**驱动：
#   两个模型在同一数据集上的评估产物（逐样本 predictions 契约 [{id,y_true,y_pred,prob,path}]，
#   与模块三 baseline_service 输出同形）→ POST /api/multi-model → 任务队列 → 报告 → fusion_insight 入库。
#
# 说明：预测为**契约同形的合成数据**（真实训练两个模型不在本轮验收范围）；接口、任务、报告结构、
# 同口径对齐、一致/分歧识别、融合与指标、结论入库均走真实代码路径。分歧归因需大模型端点，
# 失败不阻断报告（报告里如实记 attribution_error）。
#
# 用法:
#     D:\python.exe scripts/m7_acceptance.py

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
BACKEND_PORT = 8023
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

RUN_A = "mm-run-m1"
RUN_B = "mm-run-m2"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def http_json(path: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{BACKEND_URL}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_task(task_id: str, timeout: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = http_json(f"/api/tasks/{task_id}")
        if task.get("status") in ("success", "failed", "cancelled"):
            return task
        time.sleep(1.0)
    return http_json(f"/api/tasks/{task_id}")


def write_artifact(tmp_dir: Path, run_id: str, preds: list[dict]) -> str:
    art = tmp_dir / f"{run_id}.json"
    art.write_text(json.dumps({"metrics": {}, "predictions": preds}, ensure_ascii=False), encoding="utf-8")
    return str(art)


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

    tmp_dir = Path(tempfile.mkdtemp(prefix="m7ac_"))
    print(f"[env] 临时工作目录：{tmp_dir}", flush=True)
    try:
        start_backend(tmp_dir)
        check("临时库后端就绪（不碰真实 index.db）", True)

        # 两个模型在同一数据集上的逐样本预测（部分分歧）
        preds_a = [{"id": f"s{i}", "y_true": "A" if i % 2 == 0 else "B",
                    "y_pred": "A" if i % 2 == 0 else "B", "prob": {"A": 0.7, "B": 0.3}, "path": None}
                   for i in range(8)]
        preds_b = [dict(p) for p in preds_a]
        preds_b[2]["y_pred"] = "B"     # 分歧
        preds_b[5]["y_pred"] = "A"     # 分歧
        for p in preds_b:
            p["prob"] = {"A": 0.4, "B": 0.6}

        art_a = write_artifact(tmp_dir, RUN_A, preds_a)
        art_b = write_artifact(tmp_dir, RUN_B, preds_b)

        http_json("/api/knowledge/ingest", method="POST", body={"data_type": "run", "data": {
            "run_id": RUN_A, "project_id": "p-mm", "run_type": "eval", "status": "success",
            "params": {"model": "modelA", "task_type": "classification"}, "artifact_path": art_a}})
        http_json("/api/knowledge/ingest", method="POST", body={"data_type": "run", "data": {
            "run_id": RUN_B, "project_id": "p-mm", "run_type": "eval", "status": "success",
            "params": {"model": "modelB", "task_type": "classification"}, "artifact_path": art_b}})

        r = http_json("/api/multi-model", method="POST",
                      body={"run_ids": [RUN_A, RUN_B], "task_type": "classification"})
        task = wait_task(r["task_id"])
        check("多模型综合分析任务执行成功", task.get("status") == "success", (task.get("error") or "")[:120])
        if task.get("status") != "success":
            return finish()

        report = http_json(f"/api/multi-model/{r['task_id']}")
        check("报告含两模型", [m["name"] for m in report["models"]] == ["modelA", "modelB"],
              str([m["name"] for m in report["models"]]))
        check("同口径样本对齐（8 个共同样本）", report["n_common_samples"] == 8, str(report["n_common_samples"]))
        check("一致样本 6、分歧样本 2",
              len(report["consistent"]) == 6 and len(report["disagreements"]) == 2,
              f"一致 {len(report['consistent'])} 分歧 {len(report['disagreements'])}")
        check("融合前后指标齐备",
              report["metrics_after"]["accuracy"] is not None and len(report["metrics_before"]) == 2,
              json.dumps(report["metrics_after"], ensure_ascii=False))
        check("分歧归因字段就位（成功或如实记错）",
              bool(report.get("attribution", {}).get("summary")) or bool(report.get("attribution_error")),
              (report.get("attribution_error") or "有归因")[:80])

        kid = report.get("knowledge_id")
        item = http_json(f"/api/knowledge/items/knowledge/{kid}") if kid else {}
        check("结论以 fusion_insight 入库（draft）",
              item.get("type") == "fusion_insight" and item.get("status") == "draft",
              f"knowledge_id={kid}")

        return finish()
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def finish() -> int:
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n=== M7 验收：{passed}/{total} 通过 ===", flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
