# scripts/m4_acceptance.py — M4/M5 验收驱动脚本（阶段4，模块五；仅标准库 + 项目内 venv）。
#
# 前置：frontend 已 `npm run build`（本脚本不启动浏览器——界面验收已由
# ui_check_4a/4b/4c/4d2 完成）、data/_acceptance/venv_smoke 已建好（torch CPU）。
#
# 自动走完 M4/M5 端到端主链路（阶段4实施方案 七）并逐项断言：
#   M4：画布新建结构化项目（git 初始化）→ 混拼图保存（保存即提交）→ 参数调改
#      （再保存再提交）→ 导出代码（后端引擎，导出即所存即所训）→ 选数据集与环境
#      → 真实 CPU 训练 → 指标 + run_record（run_type=train）→ 运行即提交；
#   M5：版本树（演化关系 + 元数据摘要）→ 任意两版本对比（代码 + 参数差异）→
#      回退（画布内容变目标版本、回退记为新版本）→ 继续编辑保存（树上新节点）。
#
# 自带临时库后端（端口 8021，不碰真实 index.db）；产物落在 data/_acceptance/
# m4_acceptance/（方案九：验收产物与真实数据隔离；库为一次性临时库，工作区保留
# 供复核——版本仓库 git log 即 M5 证据）。
#
# 用法:
#     backend/.venv/Scripts/python.exe scripts/m4_acceptance.py

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
VENV_SMOKE = REPO_ROOT / "data" / "_acceptance" / "venv_smoke"
ACCEPT_DIR = REPO_ROOT / "data" / "_acceptance"

BACKEND_PORT = 8021
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

MODULE_ID = "mod_m4ac0001"
MODULE_COMPAT_ID = f"{MODULE_ID}:v1"
MODULE_NAME = "M4 验收样例模块"
DATASET_ID = "ds_m4ac0001"
DATASET_NAME = "M4 验收数值数据集"
ORIGINAL_NAME = "M4 验收原始项目"
NETWORK_NAME = "M4 验收样例网络"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def http_json(path: str) -> dict:
    with urllib.request.urlopen(f"{BACKEND_URL}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{BACKEND_URL}{path}", method=method,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_status(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """带状态码的调用（守卫断言用）。"""
    req = urllib.request.Request(
        f"{BACKEND_URL}{path}", method=method,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def git(ws: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ws), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{(proc.stderr or proc.stdout).strip()[-300:]}")
    return proc.stdout.strip()


def rmtree_win(path: Path) -> None:
    """Windows 下 .git 对象文件只读：先清只读位再删。"""

    def _onerror(func, p, _exc_info):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onerror=_onerror)


# ---------------------------------------------------------------------------
# 1. 临时库后端 + 种子（模块包 / 数据集 / 原始项目环境联接）
# ---------------------------------------------------------------------------

def seed_dataset_csv(data_dir: Path) -> None:
    import csv
    import random

    rng = random.Random(42)
    rows = []
    for i in range(40):
        feats = [round(rng.uniform(-1, 1), 4) for _ in range(4)]
        label = 1 if feats[0] + feats[1] > 0 else 0
        rows.append((i + 1, "train" if i < 32 else "test", label, json.dumps(feats)))
    with (data_dir / "preprocessed.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "split", "label", "input"])
        for rid, split, label, feats in rows:
            writer.writerow([rid, split, label, feats])


def start_backend(ws: Path):
    sys.path.insert(0, str(BACKEND))
    from app.db import connection
    from app.services import knowledge_service, project_manager

    connection.DB_PATH = ws / "index.db"
    connection.init_db()
    project_manager.PROJECTS_DIR = ws / "projects"
    Path(project_manager.PROJECTS_DIR).mkdir(parents=True, exist_ok=True)

    # 模块包：真实 module.py（torch Linear 包装类，根类在最后；与 ui_check_4c 同款）
    pkg = ws / "modules" / MODULE_ID / "v1"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "module.py").write_text(
        "import torch.nn as nn\n"
        "class _Inner(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 8)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "class M4acLinear(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.inner = _Inner()\n"
        "    def forward(self, x):\n"
        "        return self.inner(x)\n",
        encoding="utf-8",
    )
    now = datetime.now().isoformat()
    knowledge_service.record_module({
        "module_id": MODULE_ID,
        "module_version": "v1",
        "name": MODULE_NAME,
        "description": "M4 验收模块（Linear 4→8）",
        "source_project_id": None,
        "source_paper_id": None,
        "task_type": "tabular",
        "input_spec": None,
        "output_spec": None,
        "params_schema": {},
        "tags": ["m4-acceptance"],
        "verification": {"overall": "passed"},
        "saved_module_compat": {
            "id": MODULE_COMPAT_ID,
            "name": MODULE_NAME,
            "version": "v1",
            "description": "M4 验收模块",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
            "graph": {"nodes": [], "edges": []},
            "createdAt": now,
            "updatedAt": now,
        },
        "path": str(pkg),
    })

    # 数据集：模块三预处理产物（注册表 + 真实 preprocessed.csv）
    data_dir = ws / "datasets" / DATASET_ID
    data_dir.mkdir(parents=True, exist_ok=True)
    seed_dataset_csv(data_dir)
    knowledge_service.register_dataset({
        "dataset_id": DATASET_ID,
        "name": DATASET_NAME,
        "task_type": "tabular",
        "local_path": str(data_dir / "preprocessed.csv"),
    })

    import uvicorn
    from app.main import app

    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=BACKEND_PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            http_json("/api/projects")
            return server
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("backend did not start")


def seed_projects(ws: Path) -> tuple[str, str, Path]:
    """原始项目（环境 = venv_smoke 目录联接）+ 结构化网络（父项目）。"""
    original = http("POST", "/api/projects", {
        "project_type": "original", "source": "m4-acceptance", "name": ORIGINAL_NAME})
    original_id = original["project_id"]
    env_link = ws / "projects" / original_id / "env"
    subprocess.run(["cmd", "/c", "rmdir", str(env_link)], capture_output=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(env_link), str(VENV_SMOKE)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"目录联接失败：{r.stderr or r.stdout}")

    network = http("POST", "/api/projects", {
        "project_type": "structured", "name": NETWORK_NAME,
        "parent_project_id": original_id})
    return original_id, network["project_id"], env_link


def graph_v1() -> dict:
    """后端模块 + 标准节点混拼小模型：module_ref(Linear 4→8) → ReLU → Linear(8→2)。"""
    return {
        "nodes": [
            {"id": "m1", "type": "module_ref", "data": {
                "moduleId": MODULE_COMPAT_ID,
                "handles": {"inputs": ["in"], "outputs": ["out"]},
            }},
            {"id": "n2", "type": "relu_layer", "data": {}},
            {"id": "n3", "type": "linear_layer", "data": {
                "in_features": 8, "out_features": 2, "bias": True}},
        ],
        "edges": [
            {"id": "e1", "source": "m1", "sourceHandle": "out", "target": "n2",
             "targetHandle": "in-0", "data": {"label": "out_m1_out"}},
            {"id": "e2", "source": "n2", "sourceHandle": "out-0", "target": "n3",
             "targetHandle": "in-0", "data": {"label": "out_n2_out-0"}},
        ],
    }


def graph_v2() -> dict:
    """V2 = V1 调参（n3 out_features 2→3）+ 追加 relu n4（输出节点）。"""
    g = graph_v1()
    g["nodes"][2]["data"]["out_features"] = 3
    g["nodes"].append({"id": "n4", "type": "relu_layer", "data": {}})
    g["edges"].append(
        {"id": "e3", "source": "n3", "target": "n4", "targetHandle": "in-0",
         "data": {"label": "out_n3"}})
    return g


# ---------------------------------------------------------------------------
# 2. 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    smoke_python = VENV_SMOKE / "Scripts" / "python.exe"
    if not smoke_python.exists():
        print(f"FATAL: smoke 环境不存在：{smoke_python}（先建 venv_smoke 并装 torch）")
        return 2
    r = subprocess.run([str(smoke_python), "-c", "import torch; print(torch.__version__)"],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(f"FATAL: smoke 环境 torch 不可用：{r.stderr[-500:]}")
        return 2
    check("smoke 环境 torch 可用", True, r.stdout.strip())

    ws = ACCEPT_DIR / "m4_acceptance"
    if ws.exists():
        rmtree_win(ws)  # 上一次验收产物（git 只读位 + 可能残留联接均已按此清理）
    ws.mkdir(parents=True, exist_ok=True)
    print(f"验收工作区：{ws}")

    server = None
    env_link: Path | None = None
    try:
        server = start_backend(ws)
        check("临时库后端启动（模块/数据集种子，不碰真实库）", True, BACKEND_URL)
        original_id, network_id, env_link = seed_projects(ws)
        check("种子就绪：原始项目（smoke 环境联接）+ 画布网络", True,
              f"net={network_id} parent={original_id}")
        ws_dir = ws / "projects" / network_id

        # ================= M4：画布自建主链路（方案七） =================

        check("M4-1 画布新建结构化项目即 git 初始化（初始提交）",
              (ws_dir / ".git").exists() and git(ws_dir, "config", "user.name") == "DL-AI-skills"
              and git(ws_dir, "rev-list", "--count", "HEAD") == "1")

        r = http("PUT", f"/api/projects/{network_id}/graph", graph_v1())
        v1_commit = r["version"]["commit"]
        saved = http_json(f"/api/projects/{network_id}/graph")
        check("M4-2 混拼图保存即提交（参数往返一致）",
              v1_commit is not None and git(ws_dir, "rev-list", "--count", "HEAD") == "2"
              and saved["nodes"][2]["data"]["out_features"] == 2
              and saved["nodes"][0]["data"]["moduleId"] == MODULE_COMPAT_ID)

        r2 = http("PUT", f"/api/projects/{network_id}/graph", graph_v2())
        v2_commit = r2["version"]["commit"]
        check("M4-3 参数调改再保存再提交（保存即提交）",
              v2_commit is not None and v2_commit != v1_commit
              and git(ws_dir, "rev-list", "--count", "HEAD") == "3")

        code = http_json(f"/api/networks/{network_id}/export")["code"]
        export_ok = ("class GeneratedModel(nn.Module):" in code
                     and "class M4acLinear(nn.Module):" in code
                     and "nn.ReLU()" in code and "return out_n4" in code)
        check("M4-4 导出代码走后端引擎（导出即所存即所训，V2 四节点）",
              export_ok, f"{len(code)} 字符" + ("" if export_ok else f"，代码尾：{code[-160:]!r}"))

        opts = http_json(f"/api/networks/{network_id}/run-options")
        check("M4-5 运行面板数据源（数据集 + 目标环境）",
              any(d["dataset_id"] == DATASET_ID for d in opts["datasets"])
              and any(e["project_id"] == original_id for e in opts["environments"]))

        run_res = http("POST", f"/api/networks/{network_id}/run", {
            "dataset_id": DATASET_ID, "environment_project_id": original_id,
            "epochs": 2, "batch_size": 8, "learning_rate": 0.01,
        })
        task_id = run_res["task_id"]
        status = None
        error = None
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            task = http_json(f"/api/tasks/{task_id}")
            status, error = task["status"], task.get("error")
            if status in ("success", "failed", "cancelled"):
                break
            time.sleep(2)
        check("M4-6 真实 CPU 训练成功（smoke 环境，2 epochs）", status == "success",
              f"task={task_id} status={status}" + (f" error={error[-300:]}" if error else ""))

        runs = http_json(f"/api/networks/{network_id}/runs")
        rec = runs[0] if runs else None
        run_ok = bool(rec) and rec["run_type"] == "train" and rec["status"] == "success" \
            and "accuracy" in json.loads(rec["metrics"] or "{}")
        conn = sqlite3.connect(str(ws / "index.db"))
        db_runs = conn.execute(
            "SELECT COUNT(*) FROM run_record WHERE project_id = ? AND run_type = 'train'",
            (network_id,)).fetchone()[0]
        mod_rows = conn.execute(
            "SELECT COUNT(*) FROM module WHERE module_id = ? AND module_version = 'v1'",
            (MODULE_ID,)).fetchone()[0]
        conn.close()
        check("M4-7 训练指标 + run_record 落库（run_type=train 成功记录 + 模块条目）",
              run_ok and db_runs == 1 and mod_rows == 1,
              f"runs={len(runs)} db_runs={db_runs} modules={mod_rows}"
              + (f" metrics={rec['metrics']}" if rec else ""))

        check("M4-8 运行即提交（「训练运行 <task_id>」+ 指标摘要）",
              git(ws_dir, "log", "-1", "--format=%s") == f"训练运行 {task_id}"
              and git(ws_dir, "rev-list", "--count", "HEAD") == "4")

        run_dir = ws_dir / "runs" / task_id
        check("M4-9 运行产物齐备（model.py/train.log/train_metrics.json）",
              (run_dir / "model.py").exists() and (run_dir / "train.log").exists()
              and (run_dir / "train_metrics.json").exists(), str(run_dir))

        # ================= M5：版本管理（方案七） =================

        tree = http_json(f"/api/versions/{network_id}/tree")
        versions = tree["versions"]
        chain_ok = (len(versions) == 4
                    and tree["current"] == git(ws_dir, "rev-parse", "--short", "HEAD")
                    and all(a["parents"] == [b["commit"]]
                            for a, b in zip(versions, versions[1:]))
                    and versions[-1]["parents"] == [])
        counts = [v["meta"]["node_count"] for v in versions]
        check("M5-1 版本树展示演化关系（父子链 + 元数据摘要）",
              chain_ok and counts == [4, 4, 3, 0], f"node_counts={counts}")

        v1_full = git(ws_dir, "rev-parse", "HEAD~2")  # V1（3 节点）
        v0_full = git(ws_dir, "rev-parse", "HEAD~3")  # V0（初始空画布）
        cmp = http_json(f"/api/versions/{network_id}/compare?v1={v1_full}&v2={v2_commit}")
        pd = cmp["param_diff"]
        # diff 头两行是 ---/+++ 头，内容行从第 3 行起
        real_plus = [l for l in (cmp["code_diff"] or [])[2:] if l.startswith("+")]
        cmp_ok = (cmp["code_diff_error"] is None
                  and real_plus
                  and pd["nodes_added"] == [{"id": "n4", "type": "relu_layer"}]
                  and pd["nodes_changed"] == [{"id": "n3", "type": "linear_layer",
                                               "param_changes": [
                                                   {"key": "out_features", "old": 2, "new": 3}]}])
        cmp0 = http_json(f"/api/versions/{network_id}/compare?v1={v0_full}&v2={v1_full}")
        real_minus0 = [l for l in (cmp0["code_diff"] or [])[2:] if l.startswith("-")]
        all_added = (cmp0["code_diff_error"] is None
                     and cmp0["code_diff"]
                     and not real_minus0
                     and len(cmp0["param_diff"]["nodes_added"]) == 3)
        check("M5-2 任意两版本对比出代码与参数差异（V0→V1 纯新增、V1→V2 增节点+改参）",
              cmp_ok and all_added)

        rb = http("POST", f"/api/versions/{network_id}/rollback",
                  {"target_version": v1_full})
        back = http_json(f"/api/projects/{network_id}/graph")
        check("M5-3 回退后画布内容变目标版本、回退记为新提交",
              len(rb["graph"]["nodes"]) == 3
              and back["nodes"][2]["data"]["out_features"] == 2
              and git(ws_dir, "log", "-1", "--format=%s").startswith("回退到")
              and git(ws_dir, "rev-list", "--count", "HEAD") == "5")

        edited = http_json(f"/api/projects/{network_id}/graph")
        edited["nodes"][2]["data"]["out_features"] = 5
        r3 = http("PUT", f"/api/projects/{network_id}/graph", edited)
        tree2 = http_json(f"/api/versions/{network_id}/tree")
        check("M5-4 回退后继续编辑保存产生新版本节点（树上新版本）",
              r3["version"]["commit"] is not None
              and len(tree2["versions"]) == 6
              and tree2["versions"][0]["message"] == "保存画布"
              and git(ws_dir, "rev-list", "--count", "HEAD") == "6")

        st_cur, _ = http_status("POST", f"/api/versions/{network_id}/rollback",
                                {"target_version": git(ws_dir, "rev-parse", "HEAD")})
        st_bad, body_bad = http_status("POST", f"/api/versions/{network_id}/rollback",
                                       {"target_version": "deadbeef"})
        check("M5-5 回退守卫（目标即当前 400 / 版本不存在 404）",
              st_cur == 400 and st_bad == 404, f"{st_cur}/{st_bad} {body_bad.get('detail', '')}")
    finally:
        if server:
            server.should_exit = True
        if env_link and env_link.exists():
            # 先移除目录联接（防 rmtree 穿透删到 venv_smoke）；工作区本体保留供复核
            subprocess.run(["cmd", "/c", "rmdir", str(env_link)], capture_output=True)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nM4/M5 脚本化验收：{passed}/{len(RESULTS)} 项通过")
    if passed < len(RESULTS):
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  ✗ {name} — {detail}")
        print(f"（工作区保留供排查：{ws}）")
        return 1
    print(f"（工作区保留供复核：{ws}；版本仓库 git log 即 M5 证据）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
