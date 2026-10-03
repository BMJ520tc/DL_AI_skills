# scripts/ui_check_4c.py — 阶段4 4c 界面自检（AGENTS.md 三.2：构建通过 ≠ 界面能打开）。
#
# 覆盖 4c 验收判据（临时库后端 + 临时项目工作区，绝不触碰 data/ 真实数据）：
#   画布网络（结构化项目）打开构建产物界面 →「保存到项目 / 导出代码 / 运行训练」
#   三入口齐备 → 导出走后端引擎（导出即所训：再生成代码含内联模块与主类）→
#   运行面板（数据集下拉 = 模块三预处理产物、目标环境 = 项目独立环境、超参）→
#   启动训练 → **真实 CPU torch 训练**（smoke 环境 data/_acceptance/venv_smoke，
#   经目录联接挂为原始项目独立环境）→ 后台任务成功 → 面板显示指标 →
#   run_record 新增 run_type=train 成功记录；控制台 0 应用错误。
#
# 用法（需已执行 `npm run build`、且已建好 venv_smoke）：
#   D:\python.exe scripts/ui_check_4c.py

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
VENV_SMOKE = ROOT / "data" / "_acceptance" / "venv_smoke"

BACKEND_PORT = 8000
PREVIEW_PORT = 5199
CDP_PORT = 9222
APP_URL = f"http://127.0.0.1:{PREVIEW_PORT}/"
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

# --headful：用可见的 Edge 窗口跑（人眼核 + 截图）；默认无头。截图一律落 data/_acceptance/shots/
HEADFUL = "--headful" in sys.argv
SHOTS = ROOT / "data" / "_acceptance" / "shots"


def shot(cdp, name: str) -> None:
    """存一张页面截图（人眼核用）。"""
    import base64
    try:
        r = cdp.call("Page.captureScreenshot", {"format": "png"})
        SHOTS.mkdir(parents=True, exist_ok=True)
        path = SHOTS / f"{name}.png"
        path.write_bytes(base64.b64decode(r["data"]))
        print(f"  截图：{path}", flush=True)
    except Exception as e:  # noqa: BLE001 —— 截图失败不影响自检结论
        print(f"  截图失败（{name}）：{e}", flush=True)

MODULE_ID = "mod_ui4c0001"
MODULE_COMPAT_ID = f"{MODULE_ID}:v1"
MODULE_NAME = "UI4C 样例模块"
DATASET_ID = "ds_ui4c0001"
DATASET_NAME = "UI4C 数值数据集"
NETWORK_NAME = "UI4C 样例网络"
ORIGINAL_NAME = "UI4C 原始项目"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def http_json(path: str) -> dict:
    with urllib.request.urlopen(f"{BACKEND_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{BACKEND_URL}{path}", method=method,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# 1. 临时库后端 + 种子（模块包 / 数据集 / 原始项目环境联接 / 画布图）
# ---------------------------------------------------------------------------

def seed_dataset_csv(data_dir: Path) -> None:
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
            # input 单元格是含逗号的 JSON 数组，必须走 csv 转义，不能手工拼行
            writer.writerow([rid, split, label, feats])


def start_backend(tmp_dir: Path):
    sys.path.insert(0, str(BACKEND))
    from app.db import connection
    from app.services import knowledge_service, project_manager

    connection.DB_PATH = tmp_dir / "index.db"
    connection.init_db()
    project_manager.PROJECTS_DIR = tmp_dir / "projects"
    Path(project_manager.PROJECTS_DIR).mkdir(parents=True, exist_ok=True)

    # 模块包：真实 module.py（torch Linear 包装类，根类在最后）
    pkg = tmp_dir / "modules" / MODULE_ID / "v1"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "module.py").write_text(
        "import torch.nn as nn\n"
        "class _Inner(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 8)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "class Ui4cLinear(nn.Module):\n"
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
        "description": "4c 自检模块（Linear 4→8）",
        "source_project_id": None,
        "source_paper_id": None,
        "task_type": "tabular",
        "input_spec": None,
        "output_spec": None,
        "params_schema": {},
        "tags": ["ui-check"],
        "verification": {"overall": "passed"},
        "saved_module_compat": {
            "id": MODULE_COMPAT_ID,
            "name": MODULE_NAME,
            "version": "v1",
            "description": "4c 自检模块",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
            "graph": {"nodes": [], "edges": []},
            "createdAt": now,
            "updatedAt": now,
        },
        "path": str(pkg),
    })

    # 数据集：模块三预处理产物（注册表 + 真实 preprocessed.csv）
    data_dir = tmp_dir / "datasets" / DATASET_ID
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
    for _ in range(100):  # 等启动收敛
        try:
            http_json("/api/projects")
            return server
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("backend did not start")


def seed_projects(tmp_dir: Path) -> tuple[str, str, Path]:
    """原始项目（环境 = venv_smoke 目录联接）+ 结构化网络（父项目）+ 画布图。"""
    original = http("POST", "/api/projects", {
        "project_type": "original", "source": "ui-check-4c", "name": ORIGINAL_NAME})
    original_id = original["project_id"]
    env_link = tmp_dir / "projects" / original_id / "env"
    subprocess.run(["cmd", "/c", "rmdir", str(env_link)], capture_output=True)  # 移除空 env 目录
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(env_link), str(VENV_SMOKE)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"目录联接失败：{r.stderr or r.stdout}")

    network = http("POST", "/api/projects", {
        "project_type": "structured", "name": NETWORK_NAME,
        "parent_project_id": original_id})
    network_id = network["project_id"]

    # 后端模块 + 标准节点混拼小模型：module_ref(Linear 4→8) → ReLU → Linear(8→2)
    graph = {
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
            # 句柄/标签按画布 onConnect 口径：module_ref 源句柄 out、relu 源句柄 out-0
            {"id": "e1", "source": "m1", "sourceHandle": "out", "target": "n2",
             "targetHandle": "in-0", "data": {"label": "out_m1_out"}},
            {"id": "e2", "source": "n2", "sourceHandle": "out-0", "target": "n3",
             "targetHandle": "in-0", "data": {"label": "out_n2_out-0"}},
        ],
    }
    http("PUT", f"/api/projects/{network_id}/graph", graph)
    return original_id, network_id, env_link


# ---------------------------------------------------------------------------
# 2. CDP 客户端（与 ui_check_4a/4b 同款）
# ---------------------------------------------------------------------------

class CDP:
    def __init__(self, ws_url: str):
        from websockets.sync.client import connect
        self.ws = connect(ws_url, max_size=2 ** 24, legacy=True)
        self._id = 0
        self.events: list[dict] = []

    def call(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv(timeout=30))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method} failed: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)

    def evaluate(self, expr: str):
        result = self.call("Runtime.evaluate", {
            "expression": expr, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            desc = details.get("exception", {}).get("description") or details.get("text")
            raise RuntimeError(f"page JS error: {desc}")
        return result.get("result", {}).get("value")

    def close(self) -> None:
        self.ws.close()


def wait_for(cdp: CDP, js_cond: str, expect: bool, what: str, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if cdp.evaluate(js_cond) is expect:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    check(what, False, f"超时 {timeout}s 未满足：{js_cond[:80]}")
    return False


def poll(cdp: CDP, js_cond: str, expect: bool, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if cdp.evaluate(js_cond) is expect:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def click_button(cdp: CDP, text: str, exact: bool = False) -> bool:
    matcher = ("trim() ===" if exact else "includes")
    found = cdp.evaluate(
        f"(() => {{ const b = [...document.querySelectorAll('button')]"
        f".find(x => (x.textContent || '').{matcher}({json.dumps(text)}));"
        f" if (!b) return false; b.click(); return true; }})()")
    if not found:
        check(f"点击按钮「{text}」", False, "页面上找不到该按钮")
        return False
    return True


def dismiss_dialogs(cdp: CDP) -> list[str]:
    """导出失败会 alert() 阻塞页面：自动点掉并返回弹出的消息。"""
    messages = []
    for ev in cdp.events:
        if ev.get("method") == "Page.javascriptDialogOpening":
            messages.append(ev.get("params", {}).get("message", ""))
            try:
                cdp.call("Page.handleJavaScriptDialog", {"accept": True})
            except Exception:
                pass
    return messages


def console_errors(cdp: CDP, since: int = 0, extra_noise: tuple[str, ...] = ()) -> list[str]:
    errors: list[str] = []
    for ev in cdp.events[since:]:
        if ev.get("method") == "Runtime.exceptionThrown":
            details = ev.get("params", {}).get("exceptionDetails", {})
            text = details.get("exception", {}).get("description") or details.get("text")
            errors.append(f"exception: {text}")
        elif ev.get("method") == "Log.entryAdded":
            entry = ev.get("params", {}).get("entry", {})
            if entry.get("level") == "error":
                text = entry.get("text", "")
                if ("status of 404" in text or "status of 500" in text
                        or any(p in text for p in extra_noise)):
                    continue
                errors.append(f"log: {text}")
        elif ev.get("method") == "Runtime.consoleAPICalled":
            params = ev.get("params", {})
            if params.get("type") == "error":
                args = params.get("args", [])
                desc = args[0].get("description", "") if args else ""
                errors.append(f"console: {desc}")
    return [e for e in errors if "favicon" not in e.lower() and "vite" not in e.lower()]


def set_control(cdp: CDP, label_text: str, value: str, tag: str) -> str:
    """按 label 文本定位控件，回写 DOM 值并派发 input+change（React 19 受控组件，
    Playwright selectOption 同款写法：实例赋值 + 双事件）。
    返回回填后的 DOM 值：选项尚未渲染时浏览器会把受控 select 的值归一为空串，
    返回值不匹配即暴露该竞态（调用方必须先等选项渲染）。"""
    return cdp.evaluate(
        f"""(() => {{
            const lab = [...document.querySelectorAll('label')]
                .find(l => (l.textContent || '').includes({json.dumps(label_text)}));
            if (!lab) return 'NO_LABEL';
            const ctl = lab.nextElementSibling;
            if (!ctl || ctl.tagName !== {json.dumps(tag.upper())}) return 'NO_CTL';
            ctl.value = {json.dumps(value)};
            ctl.dispatchEvent(new Event('input', {{ bubbles: true }}));
            ctl.dispatchEvent(new Event('change', {{ bubbles: true }}));
            return ctl.value;
        }})()""")


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for path, what in ((Path(EDGE), "Edge"), (FRONTEND / "dist" / "index.html", "frontend/dist")):
        if not path.exists():
            print(f"FATAL: {what} 不存在：{path}")
            return 2
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="ui_check_4c_"))
    print(f"临时目录：{tmp_dir}")
    server = None
    preview: subprocess.Popen | None = None
    edge: subprocess.Popen | None = None
    cdp: CDP | None = None
    env_link: Path | None = None

    try:
        server = start_backend(tmp_dir)
        check("后端启动（临时库 + 模块/数据集种子）", True, f"http://127.0.0.1:{BACKEND_PORT}")
        original_id, network_id, env_link = seed_projects(tmp_dir)
        check("种子就绪：原始项目（smoke 环境联接）+ 结构化网络 + 画布图",
              True, f"net={network_id} parent={original_id}")

        preview_log = open(tmp_dir / "preview.log", "w", encoding="utf-8")
        preview = subprocess.Popen(
            ["npm.cmd", "run", "preview", "--", "--port", str(PREVIEW_PORT),
             "--strictPort", "--host", "127.0.0.1"],
            cwd=str(FRONTEND), stdout=preview_log, stderr=subprocess.STDOUT)
        served = False
        for _ in range(60):
            if preview.poll() is not None:
                preview_log.flush()
                raise RuntimeError(
                    f"vite preview 提前退出（exit={preview.poll()}），日志：\n"
                    + Path(tmp_dir / "preview.log").read_text(encoding="utf-8", errors="replace"))
            try:
                with urllib.request.urlopen(APP_URL, timeout=2) as resp:
                    if resp.status == 200 and b"<div id=\"root\"" in resp.read(4096):
                        served = True
                        break
            except Exception:
                pass
            time.sleep(0.3)
        if not served:
            raise RuntimeError("vite preview 未在 20s 内就绪")
        check("vite preview 服务构建产物", True, APP_URL)

        edge_args = [
            EDGE, f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={tmp_dir / 'edge-profile'}", "--no-first-run",
            "--no-default-browser-check", "--disable-gpu", "--disable-extensions",
            "--remote-allow-origins=*", "--window-size=1280,900", "about:blank",
        ]
        if not HEADFUL:
            edge_args.insert(1, "--headless=new")
        edge = subprocess.Popen(edge_args)
        ws_url = None
        for _ in range(80):
            try:
                targets = json.loads(urllib.request.urlopen(
                    f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=2).read())
                page = next((t for t in targets if t.get("type") == "page"), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not ws_url:
            check("无头 Edge 启动", False, "CDP 未就绪")
            return 1
        check("无头 Edge 启动（CDP）", True, f"port {CDP_PORT}")

        cdp = CDP(ws_url)
        cdp.call("Runtime.enable")
        cdp.call("Log.enable")
        cdp.call("Page.enable")
        cdp.call("Page.navigate", {"url": APP_URL})

        # ================= 阶段 A：画布打开与三入口 =================
        wait_for(cdp, "document.body.innerText.includes('创建原始项目')", True,
                 "项目列表渲染")
        if not click_button(cdp, "打开画布", exact=True):
            return 1
        wait_for(cdp, "document.body.innerText.includes('保存到项目')", True,
                 "画布视图打开（结构化网络）")
        trio = cdp.evaluate(
            "JSON.stringify(['保存到项目','导出代码','运行训练'].every("
            "t => [...document.querySelectorAll('button')].some(b => (b.textContent||'').trim() === t)))")
        check("画布右上三入口齐备（保存/导出/运行）", trio == "true")

        # 画布渲染三个节点（后端模块 + 标准节点混拼）
        nodes_count = cdp.evaluate(
            "document.querySelectorAll('.react-flow__node').length")
        check("画布渲染 3 个节点（module_ref + relu + linear）", nodes_count == 3,
              f"nodes={nodes_count}")
        shot(cdp, "4c-canvas")

        # ================= 阶段 B：导出代码（后端引擎，导出即所训） =================
        if not click_button(cdp, "导出代码", exact=True):
            return 1
        back = wait_for(cdp,
                        "[...document.querySelectorAll('button')].some(b => (b.textContent||'').trim() === '导出代码')",
                        True, "导出完成（按钮恢复）", timeout=30)
        dialogs = dismiss_dialogs(cdp)
        check("导出过程无错误弹窗", back and not dialogs, "；".join(dialogs))
        code = http_json(f"/api/networks/{network_id}/export")["code"]
        check("导出代码含主类与内联模块（导出即所训）",
              "class GeneratedModel(nn.Module):" in code
              and "class Ui4cLinear(nn.Module):" in code
              and "nn.ReLU()" in code and "return out_n3" in code,
              f"{len(code)} 字符")

        # ================= 阶段 C：运行面板 → 真实 CPU 训练 =================
        if not click_button(cdp, "运行训练", exact=True):
            return 1
        wait_for(cdp, "document.body.innerText.includes('训练运行')", True,
                 "运行面板打开")
        # 运行选项是异步加载的：必须等数据集选项渲染出来再回填，
        # 否则受控 select 收到不存在的 option 值会被浏览器归一为空串（历史竞态根因）
        wait_for(cdp, f"document.body.innerText.includes('{DATASET_NAME}')", True,
                 "运行选项加载（数据集/环境列表渲染）")

        ds_ok = set_control(cdp, "数据集", DATASET_ID, "select")
        env_ok = set_control(cdp, "目标环境", original_id, "select")
        check("运行面板下拉填充（数据集 + 目标环境）",
              ds_ok == DATASET_ID and env_ok == original_id,
              f"dataset={ds_ok!r} env={env_ok!r}")
        shot(cdp, "4c-run-panel")

        epochs_ok = cdp.evaluate(
            """(() => {
                const lab = [...document.querySelectorAll('label')]
                    .find(l => (l.textContent || '') === 'epochs');
                const ctl = lab.nextElementSibling;
                ctl.value = '2';
                ctl.dispatchEvent(new Event('input', { bubbles: true }));
                ctl.dispatchEvent(new Event('change', { bubbles: true }));
                return ctl.value === '2';
            })()""")
        check("训练超参可编辑（epochs → 2）", bool(epochs_ok))

        if not click_button(cdp, "启动训练", exact=True):
            return 1
        if not wait_for(cdp, "document.body.innerText.includes('训练中')", True,
                        "训练任务进入运行中（面板显示进度）", timeout=30):
            # 诊断：面板提示文本 + 下拉当前 DOM 值（React 状态未同步时重渲染会把它重置回空）
            diag = cdp.evaluate("document.body.innerText.slice(0, 1500)")
            ds_now = cdp.evaluate(
                """(() => {
                    const lab = [...document.querySelectorAll('label')]
                        .find(l => (l.textContent || '').includes('数据集'));
                    return lab ? lab.nextElementSibling.value : 'NO_LABEL';
                })()""")
            print(f"[diag] 面板文本片段：{diag!r}")
            print(f"[diag] 点击后数据集下拉值：{ds_now!r}")

        # 轮询后端任务至终态（真实 torch CPU 训练，smoke 规模 2 epochs）
        tasks = [t for t in http_json("/api/tasks")
                 if t.get("project_id") == network_id and t.get("task_type") == "network_train"]
        task_id = tasks[0]["task_id"] if tasks else None
        check("后端任务已建（POST run 入队）", bool(task_id), str(task_id))
        status = None
        error = None
        deadline = time.monotonic() + 300
        while task_id and time.monotonic() < deadline:
            task = http_json(f"/api/tasks/{task_id}")
            status, error = task["status"], task.get("error")
            if status in ("success", "failed", "cancelled"):
                break
            time.sleep(2)
        check("真实 CPU 训练任务成功", status == "success",
              f"status={status}" + (f" error={error[-300:]}" if error else ""))

        if status == "success":
            # 面板自动刷新运行记录并显示指标
            metrics_shown = wait_for(cdp,
                                     "document.body.innerText.includes('accuracy')",
                                     True, "面板显示训练指标（accuracy）", timeout=30)
            check("运行面板指标展示", metrics_shown)

            runs = http_json(f"/api/networks/{network_id}/runs")
            rec = runs[0] if runs else None
            run_ok = bool(rec) and rec["run_type"] == "train" and rec["status"] == "success" \
                and "accuracy" in json.loads(rec["metrics"] or "{}")
            check("run_record 新增 run_type=train 成功记录", run_ok,
                  f"runs={len(runs)}" + (f" metrics={rec['metrics']}" if rec else ""))

            # 工作区产物：model.py + train.log + train_metrics.json
            run_dir = tmp_dir / "projects" / network_id / "runs" / task_id
            artifacts = (run_dir / "model.py").exists() and (run_dir / "train.log").exists() \
                and (run_dir / "train_metrics.json").exists()
            check("运行目录产物齐备（model.py/train.log/train_metrics.json）", artifacts,
                  str(run_dir))

        # ================= 阶段 D：控制台 =================
        time.sleep(1.0)
        errors = console_errors(cdp)
        check("全程控制台 0 应用错误", not errors, "；".join(errors[:5]) if errors else "0 条")
    finally:
        if cdp:
            cdp.close()
        if server:
            server.should_exit = True
        for proc in (edge, preview):
            if proc and proc.poll() is None:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
        if env_link and env_link.exists():
            # 先移除目录联接再删临时目录（避免 rmtree 穿透删到 venv_smoke）
            subprocess.run(["cmd", "/c", "rmdir", str(env_link)], capture_output=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    passed = sum(1 for _, p, _ in CHECKS if p)
    print(f"\n4c 界面自检：{passed}/{len(CHECKS)} 项通过")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
