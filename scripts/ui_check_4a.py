# scripts/ui_check_4a.py — 阶段4 4a 界面自检（AGENTS.md 三.2：构建通过 ≠ 界面能打开）。
#
# 全链路真实打开构建产物：
#   1. 进程内启动后端（临时 SQLite + 临时项目目录，绝不触碰 data/ 真实数据），
#      种子：论文一条（含实验条目）、数据集一条、原始项目一条（含结构报告与 IR 种子）；
#   2. `vite preview` 服务 frontend/dist（当前构建产物，API 基址为默认 :8000）；
#   3. 无头 Edge（CDP）逐项断言：
#      - 项目列表渲染、无「后端连接失败」横幅；检索与下载入口存在；
#      - 「＋ 新建模型」创建结构化项目并直接打开画布（空图正常渲染，无报错）；
#      - 经 HTTP 复核：新项目 graph.json = {nodes:[], edges:[]}、status=ready；
#      - 返回列表 → 打开原始项目查看器：并列入口（先使用/先拆解）齐备，**不再有**「先复现」；
#      - 模块一新增入口：独立环境/最小可运行命令验证按钮、结构报告内容渲染（报告页签）、
#        参数区「入口输入规格」PUT 真写回后端；
#      - 模块四 6.2 延伸「结构」页签：结构校验提示、新增节点 / 新增边 / 删除边 / 删除节点
#        逐项经 HTTP 复核真写回后端（ir.json）；
#      - 初始界面「论文复现」卡片（模块二入口已从查看器迁到这里）：绑定项目 + 论文下拉
#        → 进入独立复现视图 → 实验条目五要素表 + 编辑入口 → 点「确认」过 4.2 闸门
#        （按钮 ② 变为「已确认 1/1 条」）→ 「⓪ 解析论文」真的发起 pdf_parse 任务（失败原因可见）；
#      - 「先使用」面板：数据集下拉含种子数据集、四个操作按钮齐备、「⓪ 数据预处理」真的发起
#        preprocess 任务（空路径先给可见校验提示）、公开数据检索入口存在；
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


def wait_for_task(task_type: str, project_id: str | None = None, timeout: float = 25.0) -> dict | None:
    """轮询 GET /api/tasks，等某类型任务出现（界面按钮真的发起了任务才算通过）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            for task in http_json("/api/tasks?limit=100"):
                if task.get("task_type") != task_type:
                    continue
                if project_id and task.get("project_id") != project_id:
                    continue
                return task
        except Exception:
            pass
        time.sleep(0.4)
    return None


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
    # 结构报告与 IR 种子（本轮新增界面入口的真实数据来源）：
    # 报告 → 查看器「报告」页签的内容渲染；IR → 参数区「入口输入规格」PUT 通道。
    reports = orig_ws / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "structure_report.json").write_text(json.dumps({
        "entry_points": ["train.py", "scripts/eval.sh"],
        "model_files": ["models/sample_net.py"],
        "train_flow": [{"file": "train.py", "function": "train"}],
        "inference_flow": [{"file": "train.py", "function": "evaluate"}],
        "module_hierarchy": [{"file": "models/sample_net.py", "class": "Net", "parent": "Module"}],
        "module_tree": [],
        "call_chain": [],
        "dependencies": [{"name": "torch", "version_spec": ">=2.0", "used_in": ["train.py"]}],
        "uncertain": [{"file": "models/sample_net.py", "reason": "dynamic getattr"}],
        "dynamic_supplement": {"supplements": [
            {"file": "models/sample_net.py", "reason": "dynamic getattr",
             "judgement": "条件分支构造的层，运行期才确定"},
        ]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "ir.json").write_text(json.dumps({
        "schema_version": "1.0",
        "source_file": "models/sample_net.py",
        "entry_class": "Net",
        "task_type": "classification",
        "input_spec": None,           # 故意留空：验证「补输入规格」这个补参通道
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None,
             "input_shape": [1, 4], "output_shape": [1, 8]},
            {"id": "fc1", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "net",
             "module_path": "fc1", "params": {"in_features": 4, "out_features": 8},
             "input_shape": [1, 4], "output_shape": [1, 8]},
        ],
        "edges": [{"from": "net", "to": "fc1", "tensor_shape": [1, 4]}],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
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


def set_input_by_placeholder(cdp: CDP, placeholder_prefix: str, value: str) -> bool:
    """按 placeholder 前缀定位输入框并回填（React 19 受控组件：实例赋值 + input/change 双事件）。"""
    return bool(cdp.evaluate(
        f"""(() => {{
            const el = [...document.querySelectorAll('input, textarea')]
                .find(i => (i.placeholder || '').startsWith({json.dumps(placeholder_prefix)}));
            if (!el) return false;
            const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, {json.dumps(value)});
            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            return el.value === {json.dumps(value)};
        }})()"""))


def console_errors(cdp: CDP) -> tuple[list[str], list[str]]:
    """返回 (真错误, 预期噪声)。预期噪声：资源类 404 日志（如尚未生成的图表文件）。"""
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
            check("项目列表有「检索与下载」入口（模块一 2.4）", "检索与下载" in body)
            parent_option = cdp.evaluate(
                "(() => { const s = [...document.querySelectorAll('option')]"
                ".find(o => (o.textContent || '').includes('父项目（可选'));"
                " return s ? s.textContent : ''; })()")
            check("新建模型可选父项目（父项目下拉存在）", bool(parent_option), str(parent_option))

        # --- 新建模型 → 画布 ---
        # 先等按钮**元素**真的渲染出来再点（新建模型流程会多发一次项目列表请求，直接点会撞竞态）
        wait_for(cdp,
                 "[...document.querySelectorAll('button')]"
                 ".some(x => (x.textContent || '').includes('＋ 新建模型'))",
                 True, "项目列表出现「＋ 新建模型」按钮")
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
                if not structured:
                    # 明确报因，不要抛 max() 的 traceback：最常见原因是预览产物没按本次临时后端地址重建
                    # （浏览器打到了别的后端，项目建在别处），其次是「＋ 新建模型」的交互/文案变了。
                    print("FATAL: 临时库中没有结构化项目——「＋ 新建模型」没有真的在本脚本的后端建项目。"
                          "请确认预览产物已按 VITE_API_BASE_URL=<本次临时后端> 重建（非默认端口时脚本会自动重建），"
                          "以及「＋ 新建模型」的交互是否仍能一次点通。")
                    return 2
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
                for label in ("先使用", "先拆解"):
                    check(f"查看器入口「{label}」存在", label in body)
                check("查看器不再有「先复现」（已迁到初始界面）", "先复现" not in body)
                check("默认先拆解模式渲染", "原始项目操作目录" in body)
                # --- 模块一：独立环境 / 最小可运行命令验证 / 结构报告内容 / 输入规格补参 ---
                check("原始项目查看器有「建独立环境」按钮", "建独立环境" in body or "重建独立环境" in body)
                check("原始项目查看器有「最小可运行命令验证」按钮", "最小可运行命令验证" in body)
                # IR 是异步加载的：等参数区把输入规格块渲染出来再断言
                wait_for(cdp, "document.body.innerText.includes('入口输入规格')", True,
                         "参数区「入口输入规格（input_spec）」入口渲染")
                check("参数区有「入口输入规格（input_spec）」入口",
                      "入口输入规格" in cdp.evaluate("document.body.innerText"))

                # 结构报告：点「报告」页签必须渲染内容（此前只判有没有）
                if click_button(cdp, "报告"):
                    wait_for(cdp, "document.body.innerText.includes('入口脚本')", True,
                             "结构报告内容渲染（页签）")
                    report_body = cdp.evaluate("document.body.innerText")
                    check("报告渲染入口脚本/模型文件/依赖版本/不确定项/动态补充",
                          all(k in report_body for k in ("入口脚本", "模型文件", "依赖及版本要求",
                                                         "不确定项", "动态补充", "train.py", "torch", ">=2.0")),
                          "缺失：" + "、".join(k for k in ("入口脚本", "模型文件", "依赖及版本要求",
                                                          "不确定项", "动态补充", "train.py", "torch", ">=2.0")
                                              if k not in report_body))

                # 输入规格补参：PUT /ir/input_spec（agent 给不出维度时的唯一通道）
                if click_button(cdp, "参数", exact=True):
                    wait_for(cdp, "document.body.innerText.includes('入口输入规格')", True,
                             "参数区输入规格入口渲染")
                    filled = set_input_by_placeholder(cdp, "shape，如", "1, 3, 8, 8")
                    check("输入规格 shape 可填写", filled)
                    if filled and click_button(cdp, "保存输入规格"):
                        wait_for(cdp, "document.body.innerText.includes('旧验证变 stale')", True,
                                 "输入规格保存后提示旧验证变 stale", timeout=15)
                        spec = http_json(f"/api/projects/{ORIG_PROJECT_ID}/ir")["ir"].get("input_spec")
                        # update_input_spec 会写入 user_edited=True（重拆解时用户补的 shape 优先于 agent 新值），
                        # 故只断言 shape 被写回、且带该标记，不要求对象逐键相等。
                        check("PUT /ir/input_spec 真的写回后端",
                              (spec or {}).get("shape") == [1, 3, 8, 8]
                              and spec.get("user_edited") is True, f"input_spec={spec}")

                # 最小可运行命令验证：确实发起 verify 任务；环境未就绪时失败原因与 run_record 都要可见
                if click_button(cdp, "最小可运行命令验证"):
                    verify_task = wait_for_task("verify", ORIG_PROJECT_ID, timeout=25)
                    check("最小可运行命令验证任务已发起（POST /verify）", verify_task is not None,
                          f"status={verify_task.get('status') if verify_task else 'None'}")
                    wait_for(cdp, "document.body.innerText.includes('最小可运行命令 run_record')"
                                  " || document.body.innerText.includes('环境未就绪')", True,
                             "验证结果/失败原因在界面可见", timeout=25)
                    verify_body = cdp.evaluate("document.body.innerText")
                    check("最小命令验证失败不静默（run_record 或失败横幅可见）",
                          ("最小可运行命令 run_record" in verify_body) or ("环境未就绪" in verify_body))

                # --- 模块四 6.2 延伸：IR 结构编辑（边/节点增删改，「结构」页签） ---
                if click_button(cdp, "结构", exact=True):
                    wait_for(cdp, "document.body.innerText.includes('新增节点')", True,
                             "「结构」页签渲染（新增节点入口）")
                    struct_body = cdp.evaluate("document.body.innerText")
                    check("「结构」页签显示结构校验提示",
                          "结构校验通过" in struct_body or "当前 IR 有" in struct_body)

                    # 新增节点 relu（叶子 nn.ReLU，顶层）
                    if click_button(cdp, "＋ 新增节点"):
                        set_input_by_placeholder(cdp, "如 mvc_decoder", "relu")
                        set_input_by_placeholder(cdp, "如 nn.Linear / MVCDecoder / add", "nn.ReLU")
                        if click_button(cdp, "添加节点"):
                            wait_for(cdp, "document.body.innerText.includes('节点已新增')", True,
                                     "新增节点成功提示", timeout=15)
                            ir_now = http_json(f"/api/projects/{ORIG_PROJECT_ID}/ir")["ir"]
                            check("POST /ir/nodes 真的写回后端（新增 relu）",
                                  any(n["id"] == "relu" for n in ir_now["nodes"]),
                                  f"nodes={[n['id'] for n in ir_now['nodes']]}")

                    # 新增边 net→relu（from/to 两个下拉 + 「添加边」）
                    edge_sel_ok = cdp.evaluate("""
                        (() => {
                            const sels = [...document.querySelectorAll('select')];
                            const from = sels.find(s => [...s.options].some(o => o.textContent === 'from…'));
                            const to = sels.find(s => [...s.options].some(o => o.textContent === 'to…'));
                            if (!from || !to) return false;
                            const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
                            set.call(from, 'net'); from.dispatchEvent(new Event('change', { bubbles: true }));
                            set.call(to, 'relu'); to.dispatchEvent(new Event('change', { bubbles: true }));
                            return from.value === 'net' && to.value === 'relu';
                        })()""")
                    check("边新增下拉（from/to）可选", bool(edge_sel_ok))
                    if edge_sel_ok and click_button(cdp, "添加边", exact=True):
                        wait_for(cdp, "document.body.innerText.includes('边已新增')", True,
                                 "新增边成功提示", timeout=15)
                        ir_now = http_json(f"/api/projects/{ORIG_PROJECT_ID}/ir")["ir"]
                        check("POST /ir/edges 真的写回后端（net→relu）",
                              any(e["from"] == "net" and e["to"] == "relu" for e in ir_now["edges"]),
                              f"edges={ir_now['edges']}")
                        # 删掉刚加的边（点该行 × 按钮）
                        if cdp.evaluate("""
                            (() => {
                                const b = [...document.querySelectorAll('button[title="删除边"]')]
                                    .find(x => (x.parentElement?.innerText || '').includes('net → relu'));
                                if (!b) return false; b.click(); return true;
                            })()"""):
                            wait_for(cdp, "document.body.innerText.includes('边已删除')", True,
                                     "删除边成功提示", timeout=15)
                            ir_now = http_json(f"/api/projects/{ORIG_PROJECT_ID}/ir")["ir"]
                            check("DELETE /ir/edges 真的写回后端",
                                  not any(e["from"] == "net" and e["to"] == "relu" for e in ir_now["edges"]),
                                  f"edges={ir_now['edges']}")

                    # 删除节点 relu：树里选中该行 → 「删除节点 relu」
                    if cdp.evaluate("""
                        (() => {
                            const els = [...document.querySelectorAll('div,span')]
                                .filter(x => (x.textContent || '').includes('nn.ReLU'));
                            if (!els.length) return false;
                            els.sort((a, b) => a.textContent.length - b.textContent.length)[0].click();
                            return true;
                        })()"""):
                        if click_button(cdp, "删除节点 relu", exact=True):
                            wait_for(cdp, "document.body.innerText.includes('节点已删除')", True,
                                     "删除节点成功提示", timeout=15)
                            ir_now = http_json(f"/api/projects/{ORIG_PROJECT_ID}/ir")["ir"]
                            check("DELETE /ir/nodes 真的写回后端",
                                  all(n["id"] != "relu" for n in ir_now["nodes"]),
                                  f"nodes={[n['id'] for n in ir_now['nodes']]}")

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

            # --- 模块三 5.1：数据预处理入口（真实发起 preprocess 任务） ---
            check("先使用面板有「⓪ 数据预处理」入口", "⓪ 数据预处理" in body)
            check("先使用面板有「运行预处理」按钮", "运行预处理" in body)
            if click_button(cdp, "运行预处理"):
                wait_for(cdp, "document.body.innerText.includes('请填写预处理输入路径')", True,
                         "预处理空路径校验提示可见", timeout=15)
            filled = set_input_by_placeholder(cdp, "输入路径",
                                              str(Path(tempfile.gettempdir()) / "ui4a_missing_input.csv"))
            check("预处理输入路径可填写", filled)
            if filled and click_button(cdp, "运行预处理"):
                preprocess_task = wait_for_task("preprocess", ORIG_PROJECT_ID, timeout=25)
                check("数据预处理任务已发起（POST /api/preprocess）", preprocess_task is not None,
                      f"status={preprocess_task.get('status') if preprocess_task else 'None'}")
                wait_for(cdp, "document.body.innerText.includes('preprocess')", True,
                         "预处理任务状态在界面可见", timeout=25)

            # --- 模块三 5.3：公开数据检索入口（存在性；真发外部检索依赖网络，故不在此触发） ---
            search_body = cdp.evaluate("document.body.innerText")
            check("先使用面板有公开数据检索入口",
                  "公开数据检索" in search_body and "检索公开数据" in search_body)

        # --- 论文复现（模块二 4.1~4.4）：入口已迁到初始界面「论文复现」卡片 ---
        if click_button(cdp, "← 返回"):
            wait_for(cdp, "document.body.innerText.includes('4a 自检原始项目')", True,
                     "返回项目列表")
        repro_body = cdp.evaluate("document.body.innerText")
        check("初始界面有「论文复现」入口卡片",
              "论文复现" in repro_body and "进入论文复现" in repro_body)
        # 绑定项目下拉：以占位符「绑定项目」定位，避开「创建结构化项目」里的父项目下拉
        picked_proj = cdp.evaluate(f"""
            (() => {{
                const sel = [...document.querySelectorAll('select')]
                    .find(s => [...s.options].some(o => o.textContent.startsWith('绑定项目')));
                if (!sel) return false;
                Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set
                    .call(sel, {json.dumps(ORIG_PROJECT_ID)});
                sel.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return true;
            }})()""")
        check("卡片含绑定项目下拉并选中种子项目", bool(picked_proj))
        if picked_proj:
            wait_for(cdp, "document.body.innerText.includes('4a 自检论文')", True,
                     "卡片论文下拉加载种子论文")
            picked = cdp.evaluate(f"""
                (() => {{
                    const sel = [...document.querySelectorAll('select')]
                        .find(s => [...s.options].some(o => o.textContent.startsWith('论文')));
                    if (!sel) return false;
                    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set
                        .call(sel, {json.dumps(PAPER_ID)});
                    sel.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return true;
                }})()""")
            check("卡片论文下拉含种子论文并选中", bool(picked))
            if picked and click_button(cdp, "进入论文复现"):
                wait_for(cdp, "document.body.innerText.includes('抽取实验条目')", True,
                         "论文复现面板渲染（独立视图）")
                wait_for(cdp, "document.body.innerText.includes('Accuracy')", True,
                         "实验条目表渲染（种子条目）")
                if click_button(cdp, "确认", exact=True):
                    wait_for(cdp, "document.body.innerText.includes('已确认 1/1 条')", True,
                             "4.2 闸门：条目确认后计数更新")
                    wait_for(cdp, "document.body.innerText.includes('条目已确认')", True,
                             "确认成功提示")
                # 模块二 4.1：解析入口（PDF→markdown）能发起任务；种子论文无 PDF，失败原因必须可见
                item_body = cdp.evaluate("document.body.innerText")
                check("论文复现面板有「解析论文（PDF→markdown）」入口", "解析论文" in item_body)
                check("论文复现面板显示条目五要素表头（数据集/划分方式/超参数/对比基线）",
                      all(k in item_body for k in ("数据集", "划分方式", "超参数", "对比基线")))
                check("论文复现面板有条目「编辑」入口", "编辑" in item_body)
                if click_button(cdp, "⓪ 解析论文"):
                    parse_task = wait_for_task("pdf_parse", None, timeout=25)
                    check("论文解析任务已发起（POST /papers/{id}/parse）", parse_task is not None,
                          f"status={parse_task.get('status') if parse_task else 'None'}")
                    wait_for(cdp, "document.body.innerText.includes('pdf_parse')"
                                  " || document.body.innerText.includes('PDF 不存在')", True,
                             "解析任务状态/失败原因可见", timeout=25)
                    parse_body = cdp.evaluate("document.body.innerText")
                    check("解析失败原因透出（不静默）",
                          ("pdf_parse" in parse_body) or ("PDF 不存在" in parse_body))

        # --- 控制台错误汇总 ---
        time.sleep(1.0)
        errors, noise = console_errors(cdp)
        if noise:
            print(f"  预期噪声（资源类 404 日志）：{len(noise)} 条")
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
