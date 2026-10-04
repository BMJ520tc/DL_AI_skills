# scripts/ui_check_4d2.py — 阶段4 4d-2 界面自检（AGENTS.md 三.2：构建通过 ≠ 界面能打开）。
#
# 覆盖 M5 三条验收判据（临时库后端 + 临时项目工作区，绝不触碰 data/ 真实数据；
# 无训练、不需要 venv_smoke）：
#   1. 版本树展示演化关系：初始空画布 + 两次保存共 3 个版本，面板按父子关系
#      缩进渲染、当前版本有标注，且每行标出它派生自哪个版本（父版本短号）；
#   2. 任意两版本对比出代码与参数差异：选 V1/V2 → 代码差异（再生成代码 diff，
#      与导出/训练同源）→ 参数差异表（新增节点 n4、n3 的 out_features 变化）；
#   3. 回退后画布内容变为目标版本、继续编辑保存后产生新版本节点：回退到 V1 →
#      画布变 3 节点 → 保存 → 树上出现新版本（共 5 个，回退本身也是新版本）；
#   4. 分叉可辨（需求五.3「多个版本以树或图呈现」）：应用内的保存/运行/回退都是向前
#      提交、历史上恒线性，故在临时工作区里用 git commit-tree 合成一个旁支版本与一个
#      合并提交（父提交不在日志相邻行）——面板应缩进该行并标「从 <short> 分出（分支）」。
#
# 用法（需已执行 `npm run build`）：
#   D:\python.exe scripts/ui_check_4d2.py

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
import urllib.error
import urllib.request
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
    """端口已被占用时必须中止（否则会连上别人的进程，把自检数据写进真实库）。"""
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

NETWORK_NAME = "UI4D2 样例网络"

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


def git(ws: Path, *args: str, env_extra: dict | None = None) -> str:
    """临时工作区里的 git 命令（仅自检用：合成旁支版本，验证分叉呈现）。"""
    proc = subprocess.run(
        ["git", "-C", str(ws), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", env={**os.environ, **(env_extra or {})})
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{(proc.stderr or proc.stdout).strip()[:300]}")
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# 1. 临时库后端 + 版本种子（结构化网络：初始空画布 → V1 三节点 → V2 四节点）
# ---------------------------------------------------------------------------

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


def chain_graph() -> dict:
    """Linear→ReLU→Linear 级连（与后端用例同口径的句柄/标签约定）。"""
    return {
        "nodes": [
            {"id": "n1", "type": "linear_layer", "data": {"in_features": 4, "out_features": 8}},
            {"id": "n2", "type": "relu_layer", "data": {}},
            {"id": "n3", "type": "linear_layer", "data": {"in_features": 8, "out_features": 2}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2", "targetHandle": "in-0",
             "data": {"label": "out_n1"}},
            {"id": "e2", "source": "n2", "sourceHandle": "out-0", "target": "n3",
             "targetHandle": "in-0", "data": {"label": "out_n2_out-0"}},
        ],
    }


def seed_network(tmp_dir: Path) -> tuple[str, dict]:
    """结构化网络 + 三个版本：创建即初始提交（空画布）→ V1 → V2（n3 参数变化 + 新增 n4）。"""
    network = http("POST", "/api/projects", {
        "project_type": "structured", "name": NETWORK_NAME})
    network_id = network["project_id"]
    http("PUT", f"/api/projects/{network_id}/graph", chain_graph())
    graph4 = chain_graph()
    graph4["nodes"][2]["data"]["out_features"] = 3
    graph4["nodes"].append({"id": "n4", "type": "relu_layer", "data": {}})
    graph4["edges"].append(
        {"id": "e3", "source": "n3", "target": "n4", "targetHandle": "in-0",
         "data": {"label": "out_n3"}})
    http("PUT", f"/api/projects/{network_id}/graph", graph4)
    tree = http_json(f"/api/versions/{network_id}/tree")
    if len(tree["versions"]) != 3:
        raise RuntimeError(f"版本种子异常：期望 3 个版本，实际 {len(tree['versions'])}")
    return network_id, tree


# ---------------------------------------------------------------------------
# 2. CDP 客户端（与 ui_check_4a/4b/4c 同款）
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


def click_version_row(cdp: CDP, short: str, child: str) -> bool:
    """点击某版本行的整行（child=''）或行内按钮（child='button'）。"""
    found = cdp.evaluate(
        f"""(() => {{
            const row = document.querySelector('[data-version={json.dumps(short)}]');
            if (!row) return false;
            const target = {json.dumps(child)} ? row.querySelector({json.dumps(child)}) : row;
            if (!target) return false;
            target.click();
            return true;
        }})()""")
    if not found:
        check(f"点击版本行 {short}{(' 的「' + child + '」') if child else ''}", False, "行不存在")
        return False
    return True


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

    if BACKEND_PORT != 8000:
        # 构建产物里的 API 基址是**编译期**注入的（默认 http://127.0.0.1:8000）。端口被真实后端
        # 占用而用 UI_CHECK_BACKEND_PORT 换端口时，若不按临时后端地址重建，浏览器里的前端仍会打到
        # :8000 —— 2026-10-04 实测把自检数据写进了真实库。故此处重建产物，确保打的是本次临时后端。
        print(f"[info] 后端端口非默认，按 VITE_API_BASE_URL={BACKEND_URL} 重建构建产物")
        subprocess.run(["npm.cmd", "run", "build"], cwd=str(FRONTEND),
                       env={**os.environ, "VITE_API_BASE_URL": BACKEND_URL}, check=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="ui_check_4d2_"))
    print(f"临时目录：{tmp_dir}")
    server = None
    preview: subprocess.Popen | None = None
    edge: subprocess.Popen | None = None
    cdp: CDP | None = None

    try:
        server = start_backend(tmp_dir)
        check("后端启动（临时库）", True, f"http://127.0.0.1:{BACKEND_PORT}")
        network_id, tree = seed_network(tmp_dir)
        v2_short = tree["versions"][0]["short"]
        v1_short = tree["versions"][1]["short"]
        v0_short = tree["versions"][2]["short"]  # 根版本（初始空画布提交）
        check("版本种子就绪：结构化网络 + 3 个版本（空画布→3 节点→4 节点）",
              True, f"net={network_id} v2={v2_short} v1={v1_short} v0={v0_short}")

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

        # ================= 阶段 A：画布打开，四入口齐备 =================
        wait_for(cdp, "document.body.innerText.includes('创建原始项目')", True,
                 "项目列表渲染")
        if not click_button(cdp, "打开画布", exact=True):
            return 1
        wait_for(cdp, "document.body.innerText.includes('保存到项目')", True,
                 "画布视图打开（结构化网络）")
        quartet = cdp.evaluate(
            "JSON.stringify(['保存到项目','导出代码','运行训练','版本'].every("
            "t => [...document.querySelectorAll('button')].some(b => (b.textContent||'').trim() === t)))")
        check("画布右上四入口齐备（保存/导出/运行/版本）", quartet == "true")

        nodes_count = cdp.evaluate(
            "document.querySelectorAll('.react-flow__node').length")
        check("画布渲染 4 个节点（当前版本 V2）", nodes_count == 4, f"nodes={nodes_count}")

        # ================= 阶段 B：版本面板 → 树与演化关系（M5 判据 1） =================
        if not click_button(cdp, "版本", exact=True):
            return 1
        wait_for(cdp, "document.body.innerText.includes('版本管理')", True,
                 "版本面板打开")
        if not wait_for(cdp,
                        "document.querySelectorAll('[data-version]').length === 3",
                        True, "版本面板列出 3 个版本"):
            diag = cdp.evaluate("document.body.innerText.slice(0, 1200)")
            print(f"[diag] 面板文本片段：{diag!r}")
            return 1
        check("版本面板列出 3 个版本", True)

        evolution = wait_for(cdp,
            f"""(() => {{
                const row = (s) => document.querySelector('[data-version=' + JSON.stringify(s) + ']');
                const r2 = row({json.dumps(v2_short)}), r1 = row({json.dumps(v1_short)});
                const rows = [...document.querySelectorAll('[data-version]')];
                const r0 = rows[2];
                const gutter = (r) => r && r.previousElementSibling;
                return !!(r2 && r1 && r0
                    && r2.textContent.includes('当前')          // 最新版本有标注
                    && gutter(r2).textContent.includes('●─')    // 链上提交：●─ 接竖线
                    && gutter(r1).textContent.includes('●─')
                    && gutter(r0).textContent.includes('●')     // 根版本：● 终点
                    && parseFloat(getComputedStyle(gutter(r2)).borderLeftWidth) > 0
                    && parseFloat(getComputedStyle(gutter(r0)).borderLeftWidth) === 0
                    // 4d-2 补强（需求五.3「看出某个版本是从哪个版本改出来的」）：
                    // 每行标出父版本短号与派生文案；线性历史不缩进、无分叉行
                    && r2.getAttribute('data-derived-from') === {json.dumps(v1_short)}
                    && r1.getAttribute('data-derived-from') === {json.dumps(v0_short)}
                    && r2.textContent.includes('↑ 上一版本 ' + {json.dumps(v1_short)})
                    && r1.textContent.includes('↑ 上一版本 ' + {json.dumps(v0_short)})
                    && r0.getAttribute('data-derived-from') === ''
                    && r0.textContent.includes('根版本（无父版本）')
                    && r2.getAttribute('data-branch') === 'false'
                    && r2.getAttribute('data-lane') === '0'
                    && rows.every(r => r.getAttribute('data-branch') === 'false'));
            }})()""", True, "版本树演化关系（竖线链 + ● + 当前标注 + 派生自父版本）")
        if not evolution:
            print("[diag] 演化关系子条件：" + str(cdp.evaluate(
                f"""(() => {{
                    const rows = [...document.querySelectorAll('[data-version]')];
                    const row = (s) => rows.find(x => x.getAttribute('data-version') === s);
                    const r2 = row({json.dumps(v2_short)}), r1 = row({json.dumps(v1_short)});
                    const r0 = rows[2];
                    const gutter = (r) => r && r.previousElementSibling;
                    return JSON.stringify({{
                        rows: rows.length,
                        has2: !!r2, has1: !!r1, has0: !!r0,
                        cur2: r2 ? r2.textContent.includes('当前') : null,
                        g2: gutter(r2) ? gutter(r2).textContent : null,
                        g1: gutter(r1) ? gutter(r1).textContent : null,
                        g0: gutter(r0) ? gutter(r0).textContent : null,
                        bl2: gutter(r2) ? getComputedStyle(gutter(r2)).borderLeftWidth : null,
                        bl0: gutter(r0) ? getComputedStyle(gutter(r0)).borderLeftWidth : null,
                        d2: r2 ? r2.getAttribute('data-derived-from') : null,
                        d1: r1 ? r1.getAttribute('data-derived-from') : null,
                        d0: r0 ? r0.getAttribute('data-derived-from') : null,
                        lane2: r2 ? r2.getAttribute('data-lane') : null,
                        branch: rows.map(x => x.getAttribute('data-branch')).join(','),
                        text2: r2 ? r2.textContent : null,
                        zoom: getComputedStyle(document.body).zoom || document.body.style.zoom || null,
                    }});
                }})()""")))
        check("版本树展示演化关系（竖线链 + ● 节点 + 当前标注 + 派生自父版本，新→旧）", bool(evolution))
        shot(cdp, "4d2-version-tree")

        # ================= 阶段 C：两版本对比（M5 判据 2） =================
        click_version_row(cdp, v1_short, "")
        click_version_row(cdp, v2_short, "")
        if not click_button(cdp, "对比所选版本", exact=True):
            return 1
        wait_for(cdp, "document.body.innerText.includes('+++ ')", True,
                 "代码差异渲染（再生成代码 diff）")
        # 内容 + 行（排除 diff 头行 +++ …）
        plus_lines = cdp.evaluate(
            "JSON.stringify([...document.querySelectorAll('pre div')]"
            ".filter(d => d.textContent.startsWith('+') && !d.textContent.startsWith('+++')).length)")
        check("对比出代码差异（+ 行/新增内容）", plus_lines not in (None, "0", "null"),
              f"+ 行数={plus_lines}")
        if not click_button(cdp, "参数差异", exact=True):
            return 1
        param_ok = wait_for(
            cdp,
            "document.body.innerText.includes('新增节点')"
            " && document.body.innerText.includes('n4')"
            " && document.body.innerText.includes('out_features')",
            True, "参数差异表渲染（新增 n4 + out_features 变化）")
        check("对比出参数差异（新增节点 n4、n3 out_features 变化）", param_ok)

        # ================= 阶段 D：回退 → 继续保存（M5 判据 3） =================
        if not click_version_row(cdp, v1_short, "button"):
            return 1
        wait_for(cdp, "document.body.innerText.includes('确认回退')", True,
                 "回退确认区出现")
        if not click_button(cdp, "确认回退", exact=True):
            return 1
        rolled = wait_for(cdp,
                          "document.querySelectorAll('.react-flow__node').length === 3",
                          True, "回退后画布变为目标版本（3 节点）", timeout=30)
        check("回退后画布内容变为目标版本", rolled)
        shot(cdp, "4d2-after-rollback")
        if not wait_for(cdp,
                        "document.querySelectorAll('[data-version]').length === 4"
                        " && document.querySelector('[data-version]').textContent.includes('回退到')",
                        True, "回退记为新版本（树 4 个版本，最新为回退提交）", timeout=30):
            diag = cdp.evaluate("document.body.innerText.slice(0, 1200)")
            print(f"[diag] 回退后面板文本片段：{diag!r}")
            return 1
        check("回退本身记为新版本（树 4 个版本）", True)

        if not click_button(cdp, "保存到项目", exact=True):
            return 1
        wait_for(cdp, "[...document.querySelectorAll('button')].some("
                      "b => (b.textContent||'').trim() === '已保存 ✓')",
                 True, "回退后继续保存成功", timeout=30)
        if not click_button(cdp, "刷新", exact=True):
            return 1
        grew = wait_for(cdp,
                        "document.querySelectorAll('[data-version]').length === 5",
                        True, "继续编辑保存后产生新版本节点（树 5 个版本）", timeout=30)
        tree_after = http_json(f"/api/versions/{network_id}/tree")
        top_msg = tree_after["versions"][0]["message"] if tree_after["versions"] else ""
        check("继续编辑保存后产生新版本节点（树 5 个版本，最新为保存画布）",
              grew and top_msg == "保存画布", top_msg)

        # ================= 阶段 E：分叉呈现（需求五.3「多个版本以树或图呈现」，4d-2 补强） =========
        # 应用内的保存/运行/回退都是向前提交，git 历史恒线性，天然造不出分支。为验证「画得出
        # 分叉」，在**临时**工作区里用 git commit-tree 合成两个提交：旁支版本 Y（父提交 = 最老的
        # 根版本 R，故它在 git log 里的父提交不是紧邻的下一行）与合并提交 M（父提交 = 当前
        # HEAD 与 Y），再把 HEAD 指向 M。面板应把 Y 缩进一层并标「从 <R> 分出（分支）」。
        from datetime import datetime, timedelta, timezone

        r_short = tree_after["versions"][-1]["short"]  # 根版本（初始空画布提交）
        head_short = tree_after["versions"][0]["short"]
        ws = tmp_dir / "projects" / network_id
        tree_hash = git(ws, "rev-parse", "HEAD^{tree}")
        now = datetime.now(timezone.utc)
        y_date = (now + timedelta(seconds=5)).isoformat()
        m_date = (now + timedelta(seconds=10)).isoformat()
        y_commit = git(
            ws, "commit-tree", tree_hash, "-p", tree_after["versions"][-1]["commit"],
            "-m", "旁支保存", env_extra={"GIT_AUTHOR_DATE": y_date, "GIT_COMMITTER_DATE": y_date})
        m_commit = git(
            ws, "commit-tree", tree_hash, "-p", tree_after["versions"][0]["commit"], "-p", y_commit,
            "-m", "合并旁支", env_extra={"GIT_AUTHOR_DATE": m_date, "GIT_COMMITTER_DATE": m_date})
        git(ws, "update-ref", "HEAD", m_commit)
        branch_tree = http_json(f"/api/versions/{network_id}/tree")
        check("分叉种子就绪：旁支版本 + 合并提交（父提交不在相邻行）",
              len(branch_tree["versions"]) == len(tree_after["versions"]) + 2
              and branch_tree["versions"][1]["short"] == y_commit[:7],
              f"rows={len(branch_tree['versions'])} 旁支={y_commit[:7]} 根={r_short}")

        if not click_button(cdp, "刷新", exact=True):
            return 1
        want_rows = len(tree_after["versions"]) + 2
        branched = wait_for(cdp,
            f"""(() => {{
                const rows = [...document.querySelectorAll('[data-version]')];
                if (rows.length !== {want_rows}) return false;
                const y = rows[1];                       // 旁支版本：父提交 = 根版本
                const m = rows[0];                       // 合并提交：父提交 = 上一版本 + 旁支
                const branches = rows.filter(r => r.getAttribute('data-branch') === 'true');
                const gutter = y.previousElementSibling;
                return branches.length === 1
                    && y.getAttribute('data-derived-from') === {json.dumps(r_short)}
                    && y.getAttribute('data-lane') === '1'
                    && y.textContent.includes('从 ' + {json.dumps(r_short)} + ' 分出（分支）')
                    && gutter.textContent.includes('└─')
                    && gutter.offsetWidth > 20               // 分叉行缩进（线性行 20px）
                    && m.getAttribute('data-lane') === '0'
                    && m.getAttribute('data-branch') === 'false'
                    && m.textContent.includes('↑ 上一版本 ' + {json.dumps(head_short)})
                    && m.textContent.includes('（合并 2 个父版本）')
                    && rows[2].getAttribute('data-lane') === '1';
            }})()""", True, "分叉版本缩进 + 「从 <short> 分出（分支）」+ 合并父版本标注")
        if not branched:
            print("[diag] 分叉子条件：" + str(cdp.evaluate(
                f"""(() => {{
                    const rows = [...document.querySelectorAll('[data-version]')];
                    return JSON.stringify(rows.map(r => ({{
                        v: r.getAttribute('data-version'),
                        from: r.getAttribute('data-derived-from'),
                        lane: r.getAttribute('data-lane'),
                        branch: r.getAttribute('data-branch'),
                        gw: r.previousElementSibling ? r.previousElementSibling.offsetWidth : null,
                        g: r.previousElementSibling ? r.previousElementSibling.textContent : null,
                        text: r.textContent,
                    }})));
                }})()""")))
        check("分叉可辨（旁支行缩进 + 「从 <short> 分出（分支）」+ 合并标注）", bool(branched))
        shot(cdp, "4d2-version-branch")

        # ================= 阶段 F：控制台 =================
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
        shutil.rmtree(tmp_dir, ignore_errors=True)

    passed = sum(1 for _, p, _ in CHECKS if p)
    print(f"\n4d-2 界面自检：{passed}/{len(CHECKS)} 项通过")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
