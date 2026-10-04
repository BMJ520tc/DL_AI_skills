# scripts/ui_check_5a.py — 阶段5 5a/5c 界面自检（AGENTS.md：构建通过 ≠ 界面能打开）。
#
# 全链路真实打开构建产物：
#   1. 进程内临时库后端（临时 SQLite + 临时项目目录，绝不触碰 data/ 真实数据），
#      种子：一条待确认蒸馏草稿（knowledge/draft）+ 两条评估运行（供多模型面板列表）；
#   2. `vite preview` 服务 frontend/dist；
#   3. 无头 Edge（CDP）断言：
#      - 新建模型进入画布 → 头部「知识库」「多模型」按钮；
#      - 知识库面板：检索/草稿双页签 → 草稿页签列出草稿 → 「确认入库」后草稿消失；
#      - 多模型面板打开并列出评估运行；
#      - 全程 console 零错误（非 favicon/vite 噪声）。
#
# 用法（需已执行 `npm run build`）：
#   D:\python.exe scripts/ui_check_5a.py

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

BACKEND_PORT = int(os.environ.get("UI_CHECK_BACKEND_PORT", "8000"))
PREVIEW_PORT = int(os.environ.get("UI_CHECK_PREVIEW_PORT", "5199"))
CDP_PORT = int(os.environ.get("UI_CHECK_CDP_PORT", "9222"))
APP_URL = f"http://127.0.0.1:{PREVIEW_PORT}/"
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

DRAFT_TITLE = "5a 自检草稿：分类任务建议 lr=1e-3"
RUN_A, RUN_B = "ui5a-run-a", "ui5a-run-b"
ORIG_PROJECT_ID = "ui5a-orig-0001"
ORIG_NAME = "5a 自检原始项目"
ADVICE_TITLE = "5a 带入建议：该项目建议用小学习率"

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def _assert_port_free(port: int, what: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(
                f"FATAL: {what} 端口 {port} 已被占用——本脚本需独占该端口起临时库后端/预览服务，"
                f"否则会把自检数据写进真实库。请释放端口或用 UI_CHECK_*_PORT 换端口。")


def http_json(path: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{BACKEND_URL}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def start_backend(tmp_dir: Path):
    sys.path.insert(0, str(BACKEND))
    from app.db import connection
    from app.services import knowledge_service, project_manager

    connection.DB_PATH = tmp_dir / "index.db"
    connection.init_db()
    project_manager.PROJECTS_DIR = tmp_dir / "projects"
    Path(project_manager.PROJECTS_DIR).mkdir(parents=True, exist_ok=True)

    knowledge_service.record_knowledge({
        "type": "param_advice", "title": DRAFT_TITLE, "content": "分类任务建议 lr=1e-3",
        "confidence": "medium", "scope": {"task_type": "classification"}, "status": "draft",
    })
    # 已确认的参数建议，scope 绑到原始项目名 → 打开该项目查看器时应出现「知识库带入建议」（②）
    knowledge_service.record_knowledge({
        "type": "param_advice", "title": ADVICE_TITLE, "content": "按此项目历史经验，学习率用 1e-4",
        "confidence": "high", "scope": {"model": ORIG_NAME}, "status": "confirmed",
    })
    # 原始项目（供查看器打开；仅需项目行 + 工作区目录）
    now = __import__("datetime").datetime.now().isoformat()
    orig_ws = Path(project_manager.PROJECTS_DIR) / ORIG_PROJECT_ID
    orig_ws.mkdir(parents=True, exist_ok=True)
    conn = connection.get_connection()
    conn.execute(
        "INSERT INTO project(project_id, project_type, source, name, status, workspace_path, "
        "created_at, updated_at, schema_version) VALUES (?, 'original', 'local', ?, 'ready', ?, ?, ?, '1.0')",
        (ORIG_PROJECT_ID, ORIG_NAME, str(orig_ws), now, now))
    conn.commit()
    conn.close()
    for rid, model in ((RUN_A, "modelA"), (RUN_B, "modelB")):
        art = tmp_dir / f"{rid}.json"
        art.write_text(json.dumps({"metrics": {}, "predictions": [
            {"id": "s1", "y_true": "A", "y_pred": "A"}]}, ensure_ascii=False), encoding="utf-8")
        knowledge_service.record_run({
            "run_id": rid, "project_id": "p-ui5a", "run_type": "eval", "status": "success",
            "params": {"model": model, "task_type": "classification"}, "artifact_path": str(art)})

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
        result = self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise RuntimeError(f"page JS error: {result['exceptionDetails'].get('text')}")
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


def console_errors(cdp: CDP) -> list[str]:
    """真实 console 错误；资源类 404 视为噪声（种子原始项目没有 report/ir/图表文件，属预期）。"""
    errors: list[str] = []
    for ev in cdp.events:
        if ev.get("method") == "Runtime.exceptionThrown":
            details = ev.get("params", {}).get("exceptionDetails", {})
            errors.append(f"exception: {details.get('exception', {}).get('description') or details.get('text')}")
        elif ev.get("method") == "Log.entryAdded":
            entry = ev.get("params", {}).get("entry", {})
            text = entry.get("text", "")
            if entry.get("level") == "error" and "status of 404" not in text:
                errors.append(f"log: {text}")
        elif ev.get("method") == "Runtime.consoleAPICalled":
            params = ev.get("params", {})
            if params.get("type") == "error":
                args = params.get("args", [])
                errors.append(f"console: {args[0].get('description', '') if args else ''}")
    return [e for e in errors if "favicon" not in e.lower() and "vite" not in e.lower()]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not Path(EDGE).exists():
        print(f"FATAL: Edge 不存在：{EDGE}")
        return 2
    if not (FRONTEND / "dist" / "index.html").exists():
        print("FATAL: frontend/dist 不存在，先执行 npm run build")
        return 2

    _assert_port_free(BACKEND_PORT, "临时库后端")
    _assert_port_free(PREVIEW_PORT, "预览服务")
    _assert_port_free(CDP_PORT, "无头浏览器调试")

    if BACKEND_PORT != 8000:
        print(f"[info] 后端端口非默认，按 VITE_API_BASE_URL={BACKEND_URL} 重建构建产物")
        subprocess.run(["npm.cmd", "run", "build"], cwd=str(FRONTEND),
                       env={**os.environ, "VITE_API_BASE_URL": BACKEND_URL}, check=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="ui_check_5a_"))
    print(f"临时目录：{tmp_dir}")
    preview = None
    edge = None
    cdp = None
    try:
        start_backend(tmp_dir)
        check("后端启动（临时库 + 草稿/评估运行种子）", True, BACKEND_URL)

        preview_log = open(tmp_dir / "preview.log", "w", encoding="utf-8")
        preview = subprocess.Popen(
            ["npm.cmd", "run", "preview", "--", "--port", str(PREVIEW_PORT), "--strictPort", "--host", "127.0.0.1"],
            cwd=str(FRONTEND), stdout=preview_log, stderr=subprocess.STDOUT)
        served = False
        for _ in range(60):
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

        edge = subprocess.Popen([
            EDGE, "--headless=new", f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={tmp_dir / 'edge-profile'}", "--no-first-run",
            "--no-default-browser-check", "--disable-gpu", "--disable-extensions",
            "--remote-allow-origins=*", "--window-size=1280,900", "about:blank",
        ])
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

        wait_for(cdp, "document.body.innerText.includes('创建原始项目')", True, "项目列表渲染")

        # --- 新建模型 → 画布（头部出现 知识库/多模型 按钮） ---
        wait_for(cdp, "[...document.querySelectorAll('button')].some(x => (x.textContent || '').includes('＋ 新建模型'))",
                 True, "出现「＋ 新建模型」按钮")
        click_button(cdp, "＋ 新建模型")
        wait_for(cdp, "document.body.innerText.includes('保存到项目')", True, "进入画布视图")

        # --- 知识库面板：检索 / 草稿 页签 ---
        if click_button(cdp, "知识库", exact=True):
            wait_for(cdp, "document.body.innerText.includes('📚 知识库')", True, "知识库面板打开")
            body = cdp.evaluate("document.body.innerText")
            check("面板含「检索」页签", "检索" in body)
            check("面板含「草稿」页签", "草稿" in body)
            if click_button(cdp, "草稿", exact=True):
                wait_for(cdp, f"document.body.innerText.includes({json.dumps(DRAFT_TITLE)})", True,
                         "草稿页签列出待确认草稿", timeout=15)
                if click_button(cdp, "确认入库", exact=True):
                    wait_for(cdp, "document.body.innerText.includes('暂无待确认草稿')", True,
                             "确认入库后草稿消失", timeout=15)
            click_button(cdp, "关闭", exact=True)

        # --- 多模型面板 ---
        if click_button(cdp, "多模型", exact=True):
            ok_panel = wait_for(cdp, "document.body.innerText.includes('多模型综合分析')", True, "多模型面板打开")
            if ok_panel:
                wait_for(cdp, "document.body.innerText.includes('ui5a-run-a')", True,
                         "多模型面板列出评估运行", timeout=15)
            click_button(cdp, "关闭", exact=True)

        # --- 返回列表 → 原始项目查看器：模块一「知识库带入建议」横幅（②，8.2）---
        if click_button(cdp, "← 返回"):
            wait_for(cdp, f"document.body.innerText.includes({json.dumps(ORIG_NAME)})", True,
                     "返回项目列表", timeout=15)
        if click_button(cdp, "模型查看器"):
            ok_viewer = wait_for(cdp, "document.body.innerText.includes('先拆解')", True,
                                 "模型查看器打开", timeout=15)
            if ok_viewer:
                wait_for(cdp, f"document.body.innerText.includes({json.dumps(ADVICE_TITLE)})", True,
                         "模块一显示「知识库带入建议」横幅（8.2）", timeout=15)

        errors = console_errors(cdp)
        check("全程 console 零错误", len(errors) == 0, "；".join(errors[:3]))
    finally:
        if cdp is not None:
            cdp.close()
        # npm.cmd 起的是 node 孙进程，terminate() 只杀 npm 外壳、node 会残留占端口 → 杀整棵进程树
        for proc in (edge, preview):
            if proc is not None:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True, check=False)
                except Exception:
                    proc.terminate()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"\n=== ui_check_5a：{passed}/{total} 通过 ===", flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
