# scripts/ui_check_4b.py — 阶段4 4b 界面自检（AGENTS.md 三.2：构建通过 ≠ 界面能打开）。
#
# 覆盖 4b 验收判据与全部守卫分支（临时库后端，绝不触碰 data/ 真实数据）：
#   A. 后端有模块：模块库注入 → 侧边栏可见、与本地同名模块并存且后端条目带
#      「后端模块库」徽标；损坏 saved_module_compat 的行静默跳过；真实拖拽后端
#      模块进画布 → 参数面板由 params_schema 生成（bool/int/float/str 四类控件
#      与默认值、list 不进面板、可选参数默认收起），改参数回写节点；localStorage
#      不被后端模块污染。
#   B. 模块接口 500（其余接口正常，桩服务模拟）：重载后画布照常打开、本地模块
#      仍在、后端模块不出现、控制台无应用异常。
#   C. 后端完全停机：项目列表显示「后端连接失败」横幅（可见降级、不崩溃）。
#
# 用法（需已执行 `npm run build` 且有 backend/.venv）：
#   backend/.venv/Scripts/python.exe scripts/ui_check_4b.py

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

BACKEND_PORT = 8000
PREVIEW_PORT = 5199
CDP_PORT = 9222
APP_URL = f"http://127.0.0.1:{PREVIEW_PORT}/"
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

MODULE_ID = "mod_ui4b0001"
MODULE_COMPAT_ID = f"{MODULE_ID}:v1"
MODULE_NAME = "UI4B 样例模块"
CORRUPT_NAME = "UI4B 损坏模块"
LOCAL_MODULE_ID = "mod-ui4b-local"
PARAMS = {
    "hidden_size": {"type": "int", "default": 128},
    "dropout": {"type": "float", "default": 0.5},
    "use_bias": {"type": "bool", "default": True},
    "act": {"type": "str", "default": "relu"},
    "norm": {"type": "list", "default": [3, 0.5, 1, 2]},
}
LOCAL_MODULE = {
    "id": LOCAL_MODULE_ID,
    "name": MODULE_NAME,
    "version": "v1",
    "graph": {"nodes": [], "edges": []},
    "handles": {"inputs": ["in"], "outputs": ["out"]},
    "description": "本地同名模块（守卫回归用）",
    "createdAt": "2026-10-03T00:00:00",
    "updatedAt": "2026-10-03T00:00:00",
    "origin": "local",
}

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def http_json(path: str) -> dict:
    with urllib.request.urlopen(f"{BACKEND_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# 1. 临时库后端（进程内 uvicorn，种子两条模块行：正常 + 损坏 compat）
# ---------------------------------------------------------------------------

def start_backend(tmp_dir: Path):
    sys.path.insert(0, str(BACKEND))
    from app.db import connection
    from app.services import knowledge_service, project_manager

    connection.DB_PATH = tmp_dir / "index.db"
    connection.init_db()
    project_manager.PROJECTS_DIR = tmp_dir / "projects"
    Path(project_manager.PROJECTS_DIR).mkdir(parents=True, exist_ok=True)

    now = datetime.now().isoformat()
    knowledge_service.record_module({
        "module_id": MODULE_ID,
        "module_version": "v1",
        "name": MODULE_NAME,
        "description": "4b 自检模块（参数面板联动用）",
        "source_project_id": None,
        "source_paper_id": None,
        "task_type": "vision",
        "input_spec": None,
        "output_spec": None,
        "params_schema": PARAMS,
        "tags": ["ui-check"],
        "verification": {"overall": "passed"},
        "saved_module_compat": {
            "id": MODULE_COMPAT_ID,
            "name": MODULE_NAME,
            "version": "v1",
            "description": "4b 自检模块",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
            "graph": {"nodes": [], "edges": []},
            "createdAt": now,
            "updatedAt": now,
        },
        "path": None,
    })
    # 损坏行直插 SQL：saved_module_compat 存非 JSON 文本，覆盖前端解析失败静默跳过分支
    conn = connection.get_connection()
    conn.execute(
        """
        INSERT INTO module(module_id, module_version, name, description,
            source_project_id, source_paper_id, task_type, input_spec, output_spec,
            params_schema, tags, verification, saved_module_compat, path,
            created_at, updated_at, schema_version)
        VALUES (?, 'v1', ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            ?, NULL, ?, ?, '1.0')
        """,
        ("mod_ui4b0002", CORRUPT_NAME, "{broken json（守卫分支：解析失败静默跳过）", now, now),
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
# 2. CDP 客户端（websockets 同步接口，与 ui_check_4a 同款）
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


def console_errors(cdp: CDP, since: int = 0, extra_noise: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """返回 (真错误, 预期噪声)。预期噪声：无报告/IR 的 404 资源日志、阶段 B 模拟的
    500、阶段 C 后端停机的连接拒绝日志。"""
    errors: list[str] = []
    noise: list[str] = []
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


NODE_JS = ("[...document.querySelectorAll('.react-flow__node')]"
           f".some(n => n.innerText.includes({json.dumps(MODULE_NAME)}))")


def module_entry_counts(cdp: CDP) -> str:
    return cdp.evaluate(
        f"""JSON.stringify((() => {{
            const entries = [...document.querySelectorAll('[draggable="true"]')]
                .filter(el => el.textContent.includes({json.dumps(MODULE_NAME)}));
            return {{
                entries: entries.length,
                badges: entries.filter(el => el.textContent.includes('后端模块库')).length,
                corrupt: document.body.innerText.includes({json.dumps(CORRUPT_NAME)}),
            }};
        }})())""")


def expand_group(cdp: CDP, group: str) -> bool:
    return cdp.evaluate(
        f"""(() => {{
            const el = [...document.querySelectorAll('div')].find(d =>
                d.textContent.includes({json.dumps(group)}) && d.style.cursor === 'pointer');
            if (!el) return false;
            el.click();
            return true;
        }})()""")


def try_cdp_drag(cdp: CDP) -> tuple[bool, str]:
    """CDP 真实拖拽：sidebar 条目 → 画布 pane（与浏览器原生 HTML5 DnD 等价）。"""
    item = cdp.evaluate(
        f"""JSON.stringify((() => {{
            const el = [...document.querySelectorAll('[draggable="true"]')]
                .find(x => x.textContent.includes({json.dumps(MODULE_NAME)})
                    && x.textContent.includes('后端模块库'));
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return {{ x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2) }};
        }})())""")
    pane = cdp.evaluate(
        """JSON.stringify((() => {
            const el = document.querySelector('.react-flow__pane');
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2) };
        })())""")
    if not item or not pane or item == "null" or pane == "null":
        return False, "找不到拖拽源或画布 pane"
    fx, fy = json.loads(item)["x"], json.loads(item)["y"]
    tx, ty = json.loads(pane)["x"], json.loads(pane)["y"]
    data = {"items": [
        {"mimeType": "application/reactflow", "data": "module_ref"},
        {"mimeType": "application/module-meta",
         "data": json.dumps({"moduleId": MODULE_COMPAT_ID})},
    ], "dragOperationsMask": 1}
    try:
        cdp.call("Input.dispatchMouseEvent", {
            "type": "mousePressed", "x": fx, "y": fy,
            "button": "left", "buttons": 1, "clickCount": 1})
        cdp.call("Input.dispatchDragEvent", {
            "type": "dragEnter", "x": fx, "y": fy, "data": data})
        cdp.call("Input.dispatchDragEvent", {
            "type": "dragOver", "x": tx, "y": ty, "data": data})
        cdp.call("Input.dispatchDragEvent", {
            "type": "drop", "x": tx, "y": ty, "data": data})
        cdp.call("Input.dispatchMouseEvent", {
            "type": "mouseReleased", "x": tx, "y": ty,
            "button": "left", "buttons": 0, "clickCount": 1})
        return True, f"({fx},{fy}) → ({tx},{ty})"
    except Exception as exc:
        return False, f"CDP 拖拽异常：{exc}"


def synthetic_drop(cdp: CDP) -> bool:
    """兜底：页面内合成 DragEvent（带 DataTransfer）派发给画布 pane。"""
    pane = cdp.evaluate(
        """JSON.stringify((() => {
            const el = document.querySelector('.react-flow__pane');
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2) };
        })())""")
    if not pane or pane == "null":
        return False
    pt = json.loads(pane)
    return bool(cdp.evaluate(
        f"""(() => {{
            const pane = document.querySelector('.react-flow__pane');
            if (!pane) return false;
            const dt = new DataTransfer();
            dt.setData('application/reactflow', 'module_ref');
            dt.setData('application/module-meta', JSON.stringify({{ moduleId: {json.dumps(MODULE_COMPAT_ID)} }}));
            const ev = new DragEvent('drop', {{ bubbles: true, cancelable: true,
                dataTransfer: dt, clientX: {pt['x']}, clientY: {pt['y']} }});
            pane.dispatchEvent(ev);
            return true;
        }})()"""))


# ---------------------------------------------------------------------------
# 3. 模块接口故障桩（阶段 B）：/api/modules → 500，其余接口按捕获的项目数据正常应答
# ---------------------------------------------------------------------------

class StubHandler(BaseHTTPRequestHandler):
    project: dict = {}

    def _send(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/projects":
            return self._send(200, [StubHandler.project])
        if path.endswith("/graph") and path.startswith("/api/projects/"):
            return self._send(200, {"nodes": [], "edges": []})
        if path.startswith("/api/projects/"):
            return self._send(200, StubHandler.project)
        if path == "/api/modules":
            return self._send(500, {"detail": "ui_check_4b 模拟模块接口故障"})
        return self._send(404, {"detail": f"ui_check_4b 桩未实现: {path}"})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


# ---------------------------------------------------------------------------
# 4. 主流程
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="ui_check_4b_"))
    print(f"临时目录：{tmp_dir}")
    server = None
    stub: ThreadingHTTPServer | None = None
    preview: subprocess.Popen | None = None
    edge: subprocess.Popen | None = None
    cdp: CDP | None = None

    try:
        server = start_backend(tmp_dir)
        check("后端启动（临时库 + 模块种子）", True, f"http://127.0.0.1:{BACKEND_PORT}")

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

        # ================= 阶段 A：后端有模块 =================
        ok_list = wait_for(cdp, "document.body.innerText.includes('创建原始项目')", True,
                           "项目列表渲染")
        if not ok_list:
            return 1

        # 种入本地同名模块（origin=local）后重载，模拟「本地 + 后端同名并存」
        cdp.evaluate(f"localStorage.setItem('customModules', "
                     f"JSON.stringify([{json.dumps(LOCAL_MODULE, ensure_ascii=False)}]))")
        cdp.call("Page.reload")
        wait_for(cdp, "document.body.innerText.includes('创建原始项目')", True,
                 "种入本地模块后重载（项目列表）")
        check("本地同名模块已种入 localStorage", True, LOCAL_MODULE_ID)

        if click_button(cdp, "＋ 新建模型"):
            wait_for(cdp, "document.body.innerText.includes('保存到项目')", True,
                     "新建模型后进入画布视图")
        structured = http_json("/api/projects?project_type=structured")
        newest = max(structured, key=lambda p: p["created_at"])
        StubHandler.project = newest
        check("画布新建模型进入（结构化项目 ready）", True, newest["project_id"])

        # --- 模块库注入：展开「自定义模块」区 ---
        if expand_group(cdp, "自定义模块"):
            wait_for(cdp, "document.body.innerText.includes('后端模块库')", True,
                     "模块库注入：后端模块条目出现")
        counts = json.loads(module_entry_counts(cdp))
        check("同名模块两条并存（后端 + 本地）", counts["entries"] == 2,
              f"entries={counts['entries']}")
        check("同名不同源展示可区分（后端条目带徽标）", counts["badges"] == 1,
              f"badges={counts['badges']}")
        check("损坏 compat 模块行静默跳过", not counts["corrupt"])

        # --- 拖拽后端模块进画布 ---
        drag_ok, drag_detail = try_cdp_drag(cdp)
        appeared = poll(cdp, NODE_JS, True)
        if not appeared:
            synthetic_drop(cdp)
            appeared = poll(cdp, NODE_JS, True)
        check("拖拽后端模块进画布", appeared,
              (drag_detail if drag_ok else "CDP 拖拽失败，已用合成事件兜底")
              + ("，节点已出现" if appeared else "，节点未出现"))

        if appeared:
            # --- 参数面板：默认收起（全部可选参数为默认值，与标准节点一致）---
            hidden = cdp.evaluate(
                f"""(() => {{ const n = [...document.querySelectorAll('.react-flow__node')]
                    .find(x => x.innerText.includes({json.dumps(MODULE_NAME)}));
                    return !!n && n.innerText.includes('+ 4 options'); }})()""")
            check("参数面板默认收起（+ 4 options）", bool(hidden))

            # 点击节点头部展开
            nrect = cdp.evaluate(
                f"""JSON.stringify((() => {{
                    const n = [...document.querySelectorAll('.react-flow__node')]
                        .find(x => x.innerText.includes({json.dumps(MODULE_NAME)}));
                    if (!n) return null;
                    const r = n.getBoundingClientRect();
                    return {{ x: Math.round(r.left + 50), y: Math.round(r.top + 12) }};
                }})())""")
            if nrect and nrect != "null":
                pt = json.loads(nrect)
                cdp.call("Input.dispatchMouseEvent", {
                    "type": "mousePressed", "x": pt["x"], "y": pt["y"],
                    "button": "left", "buttons": 1, "clickCount": 1})
                cdp.call("Input.dispatchMouseEvent", {
                    "type": "mouseReleased", "x": pt["x"], "y": pt["y"],
                    "button": "left", "buttons": 0, "clickCount": 1})
            expanded = poll(cdp,
                            "document.body.innerText.includes('hidden_size')", True)
            check("展开后参数面板显示 params_schema 字段", expanded)
            if expanded:
                vals = json.loads(cdp.evaluate(
                    f"""JSON.stringify((() => {{
                        const node = [...document.querySelectorAll('.react-flow__node')]
                            .find(n => n.innerText.includes({json.dumps(MODULE_NAME)}));
                        if (!node) return null;
                        const get = (name) => {{
                            const lab = [...node.querySelectorAll('label')]
                                .find(l => l.textContent === name);
                            if (!lab) return undefined;
                            const ctl = lab.nextElementSibling;
                            if (!ctl) return undefined;
                            return (ctl.tagName === 'INPUT' && ctl.type === 'checkbox')
                                ? ctl.checked : ctl.value;
                        }};
                        return {{
                            hidden_size: get('hidden_size'),
                            dropout: get('dropout'),
                            use_bias: get('use_bias'),
                            act: get('act'),
                            norm: node.innerText.includes('norm'),
                        }};
                    }})())"""))
                check("四类参数控件与默认值回填正确",
                      vals == {"hidden_size": "128", "dropout": "0.5",
                               "use_bias": True, "act": "relu", "norm": False},
                      json.dumps(vals, ensure_ascii=False))
                check("list 类型参数不进面板（norm 不显示）", vals is not None and not vals["norm"])

                # 修改 hidden_size → 256，验证面板联动回写节点数据
                edited = cdp.evaluate(
                    f"""(() => {{
                        const node = [...document.querySelectorAll('.react-flow__node')]
                            .find(n => n.innerText.includes({json.dumps(MODULE_NAME)}));
                        const lab = [...node.querySelectorAll('label')]
                            .find(l => l.textContent === 'hidden_size');
                        const input = lab.nextElementSibling;
                        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')
                            .set.call(input, '256');
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        return true;
                    }})()""")
                persisted = bool(edited) and poll(cdp, """(() => {
                    const node = [...document.querySelectorAll('.react-flow__node')]
                        .find(n => n.innerText.includes('UI4B 样例模块'));
                    const lab = [...node.querySelectorAll('label')]
                        .find(l => l.textContent === 'hidden_size');
                    return lab && lab.nextElementSibling.value === '256';
                })()""", True)
                check("修改参数回写节点（hidden_size → 256）", persisted)

        # --- localStorage 未被后端模块污染 ---
        stored_ids = cdp.evaluate(
            "JSON.stringify((JSON.parse(localStorage.getItem('customModules') || '[]')).map(m => m.id))")
        check("localStorage 未被后端模块污染", stored_ids == json.dumps([LOCAL_MODULE_ID]),
              stored_ids)

        time.sleep(1.0)
        errors, noise = console_errors(cdp)
        if noise:
            print(f"  预期噪声：{len(noise)} 条")
        check("阶段 A 控制台无错误", not errors, "；".join(errors[:5]) if errors else "0 条")

        # ================= 阶段 B：模块接口 500，其余正常 =================
        server.should_exit = True
        for _ in range(100):
            try:
                http_json("/api/projects")
                time.sleep(0.2)
            except Exception:
                break
        down = False
        try:
            http_json("/api/projects")
        except Exception:
            down = True
        check("原后端已停机", down)

        stub = ThreadingHTTPServer(("127.0.0.1", BACKEND_PORT), StubHandler)
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        stub_ready = False
        for _ in range(100):
            try:
                if http_json("/api/projects"):
                    stub_ready = True
                    break
            except Exception:
                time.sleep(0.2)
        check("模块接口故障桩就绪", stub_ready, "其余接口正常、/api/modules → 500")
        code500 = False
        try:
            urllib.request.urlopen(f"{BACKEND_URL}/api/modules", timeout=5)
        except urllib.error.HTTPError as exc:
            code500 = exc.code == 500
        check("桩确认：/api/modules 返回 500", code500)

        mark_b = len(cdp.events)
        cdp.call("Page.reload")
        # 新建模型的 name 可能为 null：以「打开画布」按钮出现为准（列表数据来自桩）
        wait_for(cdp, "document.body.innerText.includes('打开画布')", True,
                 "重载后项目列表可用（桩）")
        if click_button(cdp, "打开画布", exact=True):
            wait_for(cdp, "document.body.innerText.includes('保存到项目')", True,
                     "模块接口故障下画布照常打开")
            expand_group(cdp, "自定义模块")
            wait_for(cdp, f"document.body.innerText.includes({json.dumps(MODULE_NAME)})", True,
                     "本地模块条目仍在")
        counts_b = json.loads(module_entry_counts(cdp))
        check("本地模块仍在、后端模块不出现",
              counts_b["entries"] == 1 and counts_b["badges"] == 0,
              json.dumps(counts_b, ensure_ascii=False))
        time.sleep(1.0)
        errors_b, noise_b = console_errors(cdp, since=mark_b)
        if noise_b:
            print(f"  预期噪声（模拟 500 的资源日志）：{len(noise_b)} 条")
        check("阶段 B 控制台无应用错误", not errors_b,
              "；".join(errors_b[:5]) if errors_b else "0 条")

        # ================= 阶段 C：后端完全停机 =================
        stub.shutdown()
        stub = None
        mark_c = len(cdp.events)
        cdp.call("Page.reload")
        wait_for(cdp, "document.body.innerText.includes('后端连接失败')", True,
                 "后端停机：列表显示「后端连接失败」横幅")
        banner = cdp.evaluate("document.body.innerText.includes('后端连接失败')")
        check("可见降级、页面未崩溃", bool(banner))
        errors_c, noise_c = console_errors(cdp, since=mark_c,
                                           extra_noise=("ERR_CONNECTION_REFUSED", "net::ERR"))
        if noise_c:
            print(f"  预期噪声（连接拒绝资源日志）：{len(noise_c)} 条")
        check("阶段 C 控制台无应用异常", not errors_c,
              "；".join(errors_c[:5]) if errors_c else "0 条")
    finally:
        if cdp:
            cdp.close()
        if server:
            server.should_exit = True
        if stub:
            try:
                stub.shutdown()
            except Exception:
                pass
        for proc in (edge, preview):
            if proc and proc.poll() is None:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    passed = sum(1 for _, p, _ in CHECKS if p)
    print(f"\n4b 界面自检：{passed}/{len(CHECKS)} 项通过")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
