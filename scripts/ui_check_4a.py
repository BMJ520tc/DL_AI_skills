# scripts/ui_check_4a.py — 阶段4 4a 界面自检（AGENTS.md 三.2：构建通过 ≠ 界面能打开）。
#
# 全链路真实打开构建产物：
#   1. 进程内启动后端（临时 SQLite + 临时项目目录，绝不触碰 data/ 真实数据），
#      种子：论文一条（含实验条目）、数据集一条、原始项目一条；
#   2. `vite preview` 服务 frontend/dist（当前构建产物，API 基址为默认 :8000）；
#   3. 无头 Edge（CDP）逐项断言：
#      - 项目列表渲染、无「后端连接失败」横幅；
#      - 「＋ 新建模型」创建结构化项目并直接打开画布（空图正常渲染，无报错）；
#      - 经 HTTP 复核：新项目 graph.json = {nodes:[], edges:[]}、status=ready；
#      - 返回列表 → 打开原始项目查看器：三并列入口（先复现/先使用/先拆解）齐备；
#      - 「先复现」面板：论文下拉含种子论文 → 实验条目表 → 点「确认」过 4.2 闸门
#        （按钮 ② 变为「已确认 1/1 条」）；
#      - 「先使用」面板：数据集下拉含种子数据集、四个操作按钮齐备；
#      - 全程收集浏览器 console 错误，非 favicon 类错误即判失败。
#
# 用法（需已执行 `npm run build`）：
#   D:\python.exe scripts/ui_check_4a.py

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
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# 端口可用 UI_CHECK_*_PORT 覆盖：本机可能已有真实后端跑在默认 8000 上。
BACKEND_PORT = int(os.environ.get("UI_CHECK_BACKEND_PORT", "8000"))
PREVIEW_PORT = int(os.environ.get("UI_CHECK_PREVIEW_PORT", "5199"))
CDP_PORT = int(os.environ.get("UI_CHECK_CDP_PORT", "9222"))
APP_URL = f"http://127.0.0.1:{PREVIEW_PORT}/"
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"


def _assert_port_free(port: int, what: str) -> None:
    """端口已被占用时必须中止。

    本脚本要在这些端口上起**自己的临时库后端 / 预览服务 / 无头浏览器**。若端口已被别人占用
    （例如用户本机正在跑真实后端 :8000），脚本会连上那个进程——自检数据就写进了**真实库**
    （2026-10-04 实际发生过：ui_check_4a 连到真实后端并建了一个结构化项目）。
    故改为「先探测、占用即中止」，并用 UI_CHECK_*_PORT 换端口。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(
                f"FATAL: {what} 端口 {port} 已被占用——本脚本需要独占该端口启动临时库后端/预览服务，"
                f"连到别人的进程会把自检数据写进真实库。请先释放端口，或用环境变量换端口："
                f"UI_CHECK_BACKEND_PORT / UI_CHECK_PREVIEW_PORT / UI_CHECK_CDP_PORT。"
            )


_assert_port_free(BACKEND_PORT, "临时库后端")
_assert_port_free(PREVIEW_PORT, "预览服务")
_assert_port_free(CDP_PORT, "无头浏览器调试")

PAPER_ID = "ui4a-paper-0001"
DATASET_ID = "ui4a-ds-0001"
ORIG_PROJECT_ID = "ui4a-orig-0001"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def http_json(path: str) -> dict:
    with urllib.request.urlopen(f"{BACKEND_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# 1. 临时库后端（进程内 uvicorn）
# ---------------------------------------------------------------------------

def start_backend(tmp_dir: Path):
    sys.path.insert(0, str(BACKEND))
    from app.db import connection
    from app.services import knowledge_service, project_manager

    connection.DB_PATH = tmp_dir / "index.db"
    connection.init_db()
    project_manager.PROJECTS_DIR = tmp_dir / "projects"
    Path(project_manager.PROJECTS_DIR).mkdir(parents=True, exist_ok=True)

    # 种子数据（只写临时库）
    now = datetime.now().isoformat()
    knowledge_service.record_paper({
        "paper_id": PAPER_ID, "title": "4a 自检论文", "source": "arxiv",
        "status": "downloaded", "created_at": now, "updated_at": now,
    })
    knowledge_service.record_experiment_items(PAPER_ID, [{
        "section_ref": "4.1", "dataset_name": "ImageNet", "metric_name": "Accuracy",
        "metric_value_reported": "0.931", "metric_unit": "%", "status": "extracted",
    }])
    knowledge_service.register_dataset({
        "dataset_id": DATASET_ID, "name": "4a 自检数据集", "source": "zenodo",
        "format": "CSV", "created_at": now, "updated_at": now,
    })
    orig_ws = Path(project_manager.PROJECTS_DIR) / ORIG_PROJECT_ID
    orig_ws.mkdir(parents=True, exist_ok=True)
    conn = connection.get_connection()
    conn.execute(
        """
        INSERT INTO project(project_id, project_type, source, name, status,
            workspace_path, created_at, updated_at, schema_version)
        VALUES (?, 'original', 'local', '4a 自检原始项目', 'ready', ?, ?, ?, '1.0')
        """,
        (ORIG_PROJECT_ID, str(orig_ws), now, now),
    )
    conn.commit()
    conn.close()

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


# ---------------------------------------------------------------------------
# 2. CDP 客户端（websockets 同步接口）
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


def console_errors(cdp: CDP) -> tuple[list[str], list[str]]:
    """返回 (真错误, 预期噪声)。预期噪声：种子原始项目尚无结构报告/IR 的 404 资源日志。"""
    errors: list[str] = []
    noise: list[str] = []
    for ev in cdp.events:
        if ev.get("method") == "Runtime.exceptionThrown":
            details = ev.get("params", {}).get("exceptionDetails", {})
            text = details.get("exception", {}).get("description") or details.get("text")
            errors.append(f"exception: {text}")
        elif ev.get("method") == "Log.entryAdded":
            entry = ev.get("params", {}).get("entry", {})
            if entry.get("level") == "error":
                text = entry.get("text", "")
                if "status of 404" in text:
                    noise.append(f"log: {text}")
                else:
                    errors.append(f"log: {text}")
        elif ev.get("method") == "Runtime.consoleAPICalled":
            params = ev.get("params", {})
            if params.get("type") == "error":
                args = params.get("args", [])
                desc = args[0].get("description", "") if args else ""
                errors.append(f"console: {desc}")
    return ([e for e in errors if "favicon" not in e.lower() and "vite" not in e.lower()],
            noise)


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not Path(EDGE).exists():
        print(f"FATAL: Edge 不存在：{EDGE}")
        return 2
    if not (FRONTEND / "dist" / "index.html").exists():
        print("FATAL: frontend/dist 不存在，先执行 npm run build")
        return 2

    if BACKEND_PORT != 8000:
        # 构建产物里的 API 基址是**编译期**注入的（默认 http://127.0.0.1:8000）。端口被真实后端
        # 占用而用 UI_CHECK_BACKEND_PORT 换端口时，若不按临时后端地址重建，浏览器里的前端仍会打到
        # :8000 —— 2026-10-04 实测把自检数据写进了真实库。故此处重建产物，确保打的是本次临时后端。
        print(f"[info] 后端端口非默认，按 VITE_API_BASE_URL={BACKEND_URL} 重建构建产物")
        subprocess.run(["npm.cmd", "run", "build"], cwd=str(FRONTEND),
                       env={**os.environ, "VITE_API_BASE_URL": BACKEND_URL}, check=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="ui_check_4a_"))
    print(f"临时目录：{tmp_dir}")
    server = None
    preview: subprocess.Popen | None = None
    edge: subprocess.Popen | None = None
    cdp: CDP | None = None

    try:
        server = start_backend(tmp_dir)
        check("后端启动（临时库 + 种子数据）", True, f"http://127.0.0.1:{BACKEND_PORT}")

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

        def dump_page(tag: str) -> None:
            info = cdp.evaluate(
                "JSON.stringify({title: document.title, url: location.href,"
                " ready: document.readyState,"
                " text: (document.body ? document.body.innerText : '(no body)').slice(0, 800)})")
            print(f"--- page dump [{tag}] ---\n{info}")
            tail = cdp.events[-12:]
            for ev in tail:
                method = ev.get("method", "?")
                params = ev.get("params", {})
                brief = params.get("entry", {}).get("text") or params.get("type") or ""
                print(f"  event {method}: {str(brief)[:100]}")

        # --- 项目列表 ---
        ok_list = wait_for(cdp, "document.body.innerText.includes('创建原始项目')", True,
                           "项目列表渲染")
        if not ok_list:
            dump_page("项目列表未渲染")
        if ok_list:
            body = cdp.evaluate("document.body.innerText")
            check("项目列表无后端错误横幅", "后端连接失败" not in body)
            check("列表含种子原始项目", "4a 自检原始项目" in body)
            check("「＋ 新建模型」按钮存在", "＋ 新建模型" in body)

        # --- 新建模型 → 画布 ---
        if click_button(cdp, "＋ 新建模型"):
            ok_canvas = wait_for(cdp, "document.body.innerText.includes('保存到项目')", True,
                                 "新建模型后进入画布视图")
            if ok_canvas:
                body = cdp.evaluate("document.body.innerText")
                has_rf = cdp.evaluate("!!document.querySelector('.react-flow')")
                check("空图画布渲染（无错误）",
                      "画布加载失败" not in body and "尚无画布快照" not in body,
                      f"react-flow 容器={'有' if has_rf else '无'}")
                structured = http_json("/api/projects?project_type=structured")
                newest = max(structured, key=lambda p: p["created_at"])
                graph = http_json(f"/api/projects/{newest['project_id']}/graph")
                check("新项目 graph.json 初始为空",
                      graph.get("nodes") == [] and graph.get("edges") == [],
                      f"{newest['project_id']} nodes={len(graph.get('nodes') or [])} edges={len(graph.get('edges') or [])}")
                check("新项目 status=ready", newest.get("status") == "ready", newest.get("status"))

        # --- 返回列表 → 原始项目查看器 ---
        if click_button(cdp, "← 返回"):
            wait_for(cdp, "document.body.innerText.includes('4a 自检原始项目')", True,
                     "返回项目列表（表格重载完成）")
        if click_button(cdp, "模型查看器"):
            ok_viewer = wait_for(cdp, "document.body.innerText.includes('先拆解')", True,
                                 "模型查看器打开")
            if ok_viewer:
                body = cdp.evaluate("document.body.innerText")
                for label in ("先复现", "先使用", "先拆解"):
                    check(f"三并列入口「{label}」存在", label in body)
                check("默认先拆解模式渲染", "原始项目操作目录" in body)

        # --- 先复现：选论文 → 看条目 → 过 4.2 确认闸门 ---
        if click_button(cdp, "先复现"):
            wait_for(cdp, "document.body.innerText.includes('抽取实验条目')", True,
                     "先复现面板渲染")
            wait_for(cdp, "document.body.innerText.includes('4a 自检论文')", True,
                     "论文下拉加载种子论文")
            picked = cdp.evaluate(f"""
                (() => {{
                    const sel = [...document.querySelectorAll('select')]
                        .find(s => [...s.options].some(o => o.textContent.includes('4a 自检论文')));
                    if (!sel) return false;
                    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set
                        .call(sel, {json.dumps(PAPER_ID)});
                    sel.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return true;
                }})()""")
            check("论文下拉含种子论文并选中", bool(picked))
            if picked:
                wait_for(cdp, "document.body.innerText.includes('Accuracy')", True,
                         "实验条目表渲染（种子条目）")
                if click_button(cdp, "确认", exact=True):
                    wait_for(cdp, "document.body.innerText.includes('已确认 1/1 条')", True,
                             "4.2 闸门：条目确认后计数更新")
                    wait_for(cdp, "document.body.innerText.includes('条目已确认')", True,
                             "确认成功提示")

        # --- 先使用：数据集下拉 + 按钮齐备 ---
        if click_button(cdp, "先使用"):
            wait_for(cdp, "document.body.innerText.includes('自带数据基准运行')", True,
                     "先使用面板渲染")
            wait_for(cdp, "document.body.innerText.includes('4a 自检数据集')", True,
                     "数据集下拉加载种子数据集")
            body = cdp.evaluate("document.body.innerText")
            check("数据集下拉含种子数据集", "4a 自检数据集" in body)
            for label in ("① 自带数据基准运行", "② 对齐所选数据集", "③ 结果对比与使用建议",
                          "④ 性能对比图", "④ 误差分布图", "④ 典型案例图"):
                check(f"按钮「{label}」存在", label in body)

        # --- 控制台错误汇总 ---
        time.sleep(1.0)
        errors, noise = console_errors(cdp)
        if noise:
            print(f"  预期噪声（种子项目无报告/IR 的 404 资源日志）：{len(noise)} 条")
        check("浏览器控制台无错误", not errors, "；".join(errors[:5]) if errors else "0 条")
        if errors:
            print("  完整错误清单：")
            for e in errors:
                print(f"    - {e}")
    finally:
        if cdp:
            cdp.close()
        if server:
            server.should_exit = True
        for proc in (edge, preview):
            if proc and proc.poll() is None:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    passed = sum(1 for _, p, _ in CHECKS if p)
    print(f"\n4a 界面自检：{passed}/{len(CHECKS)} 项通过")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
