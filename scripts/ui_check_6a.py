"""6.6-a UI 自检：后端同源服务前端产物 + 设置弹窗（凭证页 / 环境自检 / 首启引导）。

口径（同 ui_check_4a）：
- 临时数据目录 + 临时库后端（DL_AI_DATA_DIR 覆盖），不碰真实 data/；
- 后端带 DL_AI_SERVE_STATIC=1 启动，页面与 API **同源**（不再起 vite preview）；
- 无头 Edge + CDP 收集 console 错误；端口被占即中止（避免连到真实后端）。

验证点：
  A. 静态服务：GET / 与深链回退 index.html；未知 /api 路径保持 404 JSON
  B. env-check / credentials 端点形状（data_dir=临时目录、static_served=True）
  C. 首启自动弹出设置页（未配置凭证 + 无跳过标记）
  D. 凭证保存：磁盘落 credentials.json（临时目录）、响应只回掩码不回明文；清除删除文件
  E. 「稍后再说」写跳过标记：无凭证时刷新不再弹；清掉标记刷新又弹
  F. 「⚙ 设置」按钮手动打开；环境自检区块渲染 + 「重新检测」可点
  G. 缺 git/python 模拟（受限 PATH 重启后端）：env-check 报缺、弹窗给安装指引链接
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# 端口可用 UI_CHECK_*_PORT 覆盖：本机可能已有真实后端跑在默认 8000 上。
BACKEND_PORT = int(os.environ.get("UI_CHECK_BACKEND_PORT", "8000"))
CDP_PORT = int(os.environ.get("UI_CHECK_CDP_PORT", "9222"))
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

# 自检用的凭证（绝不能与真实凭证相同：本脚本只碰临时目录）
TEST_KEY = "sk-secret-1234"
TEST_BASE_URL = "https://api.example.com"
TEST_MODEL = "test-model"
TEST_SMALL_MODEL = "small-1"


def _assert_port_free(port: int, what: str) -> None:
    """端口已被占用时必须中止（连到别人的进程会把自检数据写进真实库）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(
                f"FATAL: {what} 端口 {port} 已被占用——本脚本需要独占该端口启动临时库后端/无头浏览器。"
                f"请先释放端口，或用环境变量换端口：UI_CHECK_BACKEND_PORT / UI_CHECK_CDP_PORT。"
            )


_assert_port_free(BACKEND_PORT, "临时库后端")
_assert_port_free(CDP_PORT, "无头浏览器调试")

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


# ---------------------------------------------------------------------------
# 1. 临时库后端（进程内 uvicorn，SERVE_STATIC=1 同源服务前端产物）
# ---------------------------------------------------------------------------

def start_backend(tmp_dir: Path):
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True)
    # 必须在 import app.config 之前设置（DATA_DIR/SERVE_STATIC 是导入期求值）
    os.environ["DL_AI_DATA_DIR"] = str(data_dir)
    os.environ["DL_AI_SERVE_STATIC"] = "1"
    sys.path.insert(0, str(BACKEND))
    import uvicorn
    from app.main import app

    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=BACKEND_PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):  # 等启动收敛
        try:
            urllib.request.urlopen(f"{BACKEND_URL}/api/health", timeout=2)
            return server, thread, data_dir
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("backend did not start")


def http_raw(path: str) -> tuple[int, dict, bytes]:
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}{path}", timeout=10) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read(4096)
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read(4096)


def http_json(path: str) -> dict:
    with urllib.request.urlopen(f"{BACKEND_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_put_json(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BACKEND_URL}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="PUT")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# 2. CDP 客户端（websockets 同步接口，同 ui_check_4a）
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


def console_errors(cdp: CDP) -> list[str]:
    errors: list[str] = []
    for ev in cdp.events:
        if ev.get("method") == "Runtime.exceptionThrown":
            details = ev.get("params", {}).get("exceptionDetails", {})
            text = details.get("exception", {}).get("description") or details.get("text")
            errors.append(f"exception: {text}")
        elif ev.get("method") == "Log.entryAdded":
            entry = ev.get("params", {}).get("entry", {})
            if entry.get("level") == "error":
                text = entry.get("text", "")
                if "status of 404" not in text:
                    errors.append(f"log: {text}")
        elif ev.get("method") == "Runtime.consoleAPICalled":
            params = ev.get("params", {})
            if params.get("type") == "error":
                args = params.get("args", [])
                desc = args[0].get("description", "") if args else ""
                errors.append(f"console: {desc}")
    return [e for e in errors if "favicon" not in e.lower() and "vite" not in e.lower()]


DIALOG_OPEN = 'document.querySelector(\'[role="dialog"]\') !== null'
DIALOG_CLOSED = 'document.querySelector(\'[role="dialog"]\') === null'


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
        # 构建产物里的 API 基址是编译期注入的（默认 http://127.0.0.1:8000）。换端口时按「同源」
        # 重建（本模式页面由后端同源服务，空基址即打本页所在地址，端口无关）。
        print(f"[info] 后端端口非默认，按 VITE_API_BASE_URL=''（同源）重建构建产物")
        subprocess.run(["npm.cmd", "run", "build"], cwd=str(FRONTEND),
                       env={**os.environ, "VITE_API_BASE_URL": ""}, check=True)

    # 模拟「干净环境」：本机若带着 Claude Code 自己的 ANTHROPIC_* 环境变量，后端会判定
    # 凭证 env 来源已配置 → 首启弹窗流程测不了。剥掉 settings_store 读取的变量再起后端
    # （真实用户双击 start.bat 不会继承这些变量；本剥离只影响本脚本进程）。
    for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
               "ANTHROPIC_DEFAULT_MODEL", "ANTHROPIC_DEFAULT_SMALL_MODEL"):
        os.environ.pop(_k, None)

    tmp_dir = Path(tempfile.mkdtemp(prefix="ui_check_6a_"))
    print(f"临时目录：{tmp_dir}")
    edge: subprocess.Popen | None = None
    cdp: CDP | None = None

    try:
        server, _, data_dir = start_backend(tmp_dir)
        check("后端启动（临时库 + SERVE_STATIC）", True, BACKEND_URL)

        # --- A. 静态服务与 SPA 回退 ---
        status, headers, body = http_raw("/")
        check("GET / 返回构建产物页面",
              status == 200 and "text/html" in headers.get("content-type", "") and b'<div id="root"' in body,
              f"status={status}")
        status, _, body = http_raw("/some/deep/link")
        check("深链回退 index.html（SPA fallback）",
              status == 200 and b'<div id="root"' in body, f"status={status}")
        status, _, _ = http_raw("/api/unknown")
        check("未知 /api 路径保持 404（不回退页面）", status == 404, f"status={status}")

        # --- B. 端点形状 ---
        env = http_json("/api/system/env-check")
        check("env-check：static_served=True", env.get("static_served") is True)
        check("env-check：data_dir 为临时目录", env.get("data_dir", {}).get("path") == str(data_dir),
              env.get("data_dir", {}).get("path", ""))
        check("env-check：git/python/conda 三键齐全",
              all(k in env for k in ("git", "python", "conda")))
        cred = http_json("/api/settings/credentials")
        check("初始凭证未配置", cred.get("configured") is False)

        # --- C. 无头浏览器 + 首启自动弹凭证页 ---
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
        cdp.call("Page.navigate", {"url": BACKEND_URL + "/"})

        ok = wait_for(cdp, DIALOG_OPEN, True, "首启自动弹出设置页")
        if ok:
            body = cdp.evaluate("document.body.innerText")
            check("首启弹窗提示未配置凭证", "尚未配置模型接口凭证" in body)
            check("首启弹窗有「稍后再说」", "稍后再说" in body)
            check("弹窗含环境自检区块", "运行环境自检" in body)

        # --- D. 凭证保存/清除 ---
        check("回填 API Key 输入框", set_input_by_placeholder(cdp, "sk-...", TEST_KEY))
        check("回填接口地址输入框", set_input_by_placeholder(cdp, "留空使用默认", TEST_BASE_URL))
        check("回填默认模型输入框", set_input_by_placeholder(cdp, "如 deepseek-chat", TEST_MODEL))
        check("回填轻量模型输入框", set_input_by_placeholder(cdp, "小任务模型", TEST_SMALL_MODEL))
        click_button(cdp, "保存", exact=True)
        ok_saved = wait_for(cdp, "document.body.innerText.includes('当前已配置')", True, "保存后显示已配置")
        if ok_saved:
            saved = http_json("/api/settings/credentials")
            check("端点：configured=True 且来源 file",
                  saved.get("configured") is True and saved.get("source") == "file")
            check("端点：只回掩码", saved.get("key_mask") == "***1234", str(saved.get("key_mask")))
            check("端点：base_url/model/small_model 落库",
                  saved.get("base_url") == TEST_BASE_URL and saved.get("model") == TEST_MODEL
                  and saved.get("small_model") == TEST_SMALL_MODEL)
            raw = http_raw("/api/settings/credentials")[2]
            check("响应不含明文密钥（K2）", TEST_KEY.encode() not in raw)
            cred_file = data_dir / "credentials.json"
            check("凭证文件落在临时数据目录", cred_file.is_file(), str(cred_file))
            check("凭证文件含密钥（本地可用）", TEST_KEY in cred_file.read_text(encoding="utf-8"))

        # 表单口径「留空则保留现有密钥」：已配置后只改默认模型、密钥留空再保存
        check("改默认模型输入框（留空密钥）", set_input_by_placeholder(cdp, "如 deepseek-chat", "model-v2"))
        click_button(cdp, "保存", exact=True)
        wait_for(cdp, "document.body.innerText.includes('当前已配置')", True, "留空密钥保存后仍已配置")
        updated = http_json("/api/settings/credentials")
        check("留空密钥保存：旧密钥保留且字段已更新",
              updated.get("key_mask") == "***1234" and updated.get("model") == "model-v2")

        click_button(cdp, "清除凭证", exact=True)
        ok_cleared = wait_for(cdp, "document.body.innerText.includes('尚未配置模型接口凭证')", True, "清除后回到未配置")
        if ok_cleared:
            check("清除后凭证文件已删除", not (data_dir / "credentials.json").exists())
            check("清除后端点未配置", http_json("/api/settings/credentials").get("configured") is False)

        # --- E. 「稍后再说」跳过标记 ---
        set_input_by_placeholder(cdp, "sk-...", "sk-mini-9999")
        click_button(cdp, "保存", exact=True)
        wait_for(cdp, "document.body.innerText.includes('当前已配置')", True, "再次保存（最小凭证）")
        click_button(cdp, "稍后再说", exact=True)
        wait_for(cdp, DIALOG_CLOSED, True, "点「稍后再说」关闭弹窗")
        check("跳过标记已写入 localStorage",
              cdp.evaluate("localStorage.getItem('dlai_credentials_skip')") == "1")
        # 服务端把凭证清掉：没有标记时刷新应该再弹；有标记时不弹——验证标记真的生效
        http_put_json("/api/settings/credentials", {"api_key": ""})
        cdp.call("Page.reload")
        time.sleep(2.5)
        check("有跳过标记：无凭证刷新也不弹", cdp.evaluate(DIALOG_CLOSED) is True)
        cdp.evaluate("localStorage.removeItem('dlai_credentials_skip')")
        cdp.call("Page.reload")
        wait_for(cdp, DIALOG_OPEN, True, "清掉标记：无凭证刷新再弹")
        click_button(cdp, "稍后再说", exact=True)
        wait_for(cdp, DIALOG_CLOSED, True, "关闭弹窗")

        # --- F. 「⚙ 设置」手动入口 + 重新检测 ---
        click_button(cdp, "⚙ 设置")
        ok_gear = wait_for(cdp, DIALOG_OPEN, True, "右下角「⚙ 设置」打开弹窗")
        if ok_gear:
            check("手动打开无「稍后再说」", "稍后再说" not in cdp.evaluate("document.body.innerText"))
            click_button(cdp, "重新检测", exact=True)
            wait_for(cdp, "document.body.innerText.includes('Git')", True, "重新检测后自检区块渲染")
            body = cdp.evaluate("document.body.innerText")
            check("自检区块含数据目录行", "数据目录" in body and str(data_dir) in body)
            click_button(cdp, "完成", exact=True)
            wait_for(cdp, DIALOG_CLOSED, True, "「完成」关闭弹窗")

        # --- G. 缺 git/python 模拟（方案验收判据：「缺 git」下正确报缺并给指引） ---
        # 说明：本段同进程停旧后端起新后端，停旧后端时 stderr 会出现一条
        # 「Task exception was never retrieved … Queue is bound to a different event loop」
        # ——uvicorn 生命周期结束后 task_manager 残留 worker 的 asyncio 噪声，仅在本脚本
        # 这种「同进程二次启动后端」场景出现，真实运行（一次生命周期）不存在，不影响断言。
        orig_path = os.environ.get("PATH", "")
        try:
            # 先让页面离开，避免「停旧后端 → 起新后端」窗口期页面发请求打出 ERR_CONNECTION_REFUSED
            cdp.call("Page.navigate", {"url": "about:blank"})
            server.should_exit = True  # 停第一个后端，释放端口再起受限 PATH 的新后端
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    if sock.connect_ex(("127.0.0.1", BACKEND_PORT)) != 0:
                        break
                time.sleep(0.3)
            # 只留 System32：找不到 git / python / py launcher / conda（py.exe 在 C:\Windows 而非 System32）
            os.environ["PATH"] = r"C:\Windows\System32"
            server2, _, _ = start_backend(tmp_dir / "missing-tools")
            check("缺工具模拟：后端重启（受限 PATH）", True, BACKEND_URL)
            env2 = http_json("/api/system/env-check")
            check("缺工具模拟：env-check 报 git 缺失", env2.get("git", {}).get("found") is False)
            check("缺工具模拟：env-check 报 python 缺失", env2.get("python", {}).get("found") is False)
            cdp.call("Page.navigate", {"url": BACKEND_URL + "/"})
            wait_for(cdp, "document.body.innerText.includes('⚙ 设置')", True, "缺工具模拟：页面重新加载")
            click_button(cdp, "⚙ 设置")
            wait_for(cdp, DIALOG_OPEN, True, "缺工具模拟：打开设置弹窗")
            # env-check 结果异步拉取，等 git 行渲染出指引链接再断言
            wait_for(cdp, "document.body.innerText.includes('git-scm.com/downloads/win')", True,
                     "缺工具模拟：git 行报缺并给安装指引")
            check("缺工具模拟：python 行报缺并给安装指引",
                  "python.org/downloads" in cdp.evaluate("document.body.innerText"))
            click_button(cdp, "完成", exact=True)
            wait_for(cdp, DIALOG_CLOSED, True, "缺工具模拟：关闭弹窗")
            server2.should_exit = True
        finally:
            os.environ["PATH"] = orig_path

        # --- console 错误收口（全程累积事件一次性检查） ---
        errors = console_errors(cdp)
        check("页面 console 无错误", not errors, "; ".join(errors[:5]) if errors else "干净")

    finally:
        if cdp:
            try:
                cdp.close()
            except Exception:
                pass
        if edge:
            edge.terminate()

    passed = sum(1 for _, p, _ in CHECKS if p)
    print(f"\n===== ui_check_6a: {passed}/{len(CHECKS)} 通过 =====")
    for name, p, detail in CHECKS:
        if not p:
            print(f"  FAIL: {name} — {detail}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
