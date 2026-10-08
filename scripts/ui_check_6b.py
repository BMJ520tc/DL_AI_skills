"""6.6-b 打包产物自检：解压 zip → 起 exe（干净环境模拟）→ 同源服务 + 首启流程断言。

口径（沿用 ui_check_6a）：
- 干净环境模拟（K1 本机）：临时 `%LOCALAPPDATA%`/`%USERPROFILE%` + PATH 剔除 git/Python/conda + 无凭证 env；
- 起打包 exe（`--no-browser`，避免弹真实浏览器），从 stdout 解析 `SERVING http://127.0.0.1:<port>`；
- 无头 Edge + CDP 收集 console 错误；
- 结束后 `taskkill /F /T` 杀进程树。

验证点：
  A. exe 起得来、解析到端口；GET / 与深链回退 index.html；未知 /api 保持 404
  B. env-check：static_served=True、data_dir 落在临时 LOCALAPPDATA 下
  C. 缺 git/Python（受限 PATH）时首启设置页仍弹、env-check 如实报缺
  D. 控制台零报错
  E. 截图（`--headful` 时出可见窗口并把截图落 data/_acceptance/shots/）

用法：
  D:\\python.exe scripts/ui_check_6b.py [zip路径] [--headful]
  默认 zip：D:\\releases\\DL-AI-skills-0.1.0-win64.zip
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CDP_PORT = int(os.environ.get("UI_CHECK_CDP_PORT", "9222"))
SHOTS = ROOT / "data" / "_acceptance" / "shots"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


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


def _clean_env(tmp: Path) -> dict:
    """干净环境：临时 LOCALAPPDATA/USERPROFILE + PATH 剔 git/Python/conda + 剥凭证 env。"""
    env = dict(os.environ)
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
              "ANTHROPIC_DEFAULT_MODEL", "ANTHROPIC_DEFAULT_SMALL_MODEL"):
        env.pop(k, None)
    env["LOCALAPPDATA"] = str(tmp / "localappdata")
    env["USERPROFILE"] = str(tmp / "userprofile")
    (tmp / "localappdata").mkdir(parents=True, exist_ok=True)
    (tmp / "userprofile").mkdir(parents=True, exist_ok=True)
    keep = []
    for part in env.get("PATH", "").split(os.pathsep):
        low = part.lower()
        if any(x in low for x in ("git", "python", "conda", "miniconda", "anaconda")):
            continue
        # 目录本身若直接含 python/py/git 可执行文件也剥掉（本机 python 就在 D:\ 根、py 在 C:\Windows）
        if part and any(os.path.exists(os.path.join(part, exe))
                        for exe in ("python.exe", "py.exe", "git.exe")):
            continue
        keep.append(part)
    env["PATH"] = os.pathsep.join(keep)
    return env


def _http(url: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read(8192)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(8192)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:]]
    headful = "--headful" in args
    args = [a for a in args if a != "--headful"]
    zip_path = Path(args[0]) if args else Path(r"D:\releases\DL-AI-skills-0.1.0-win64.zip")
    if not zip_path.is_file():
        print(f"FATAL: 找不到打包产物 {zip_path}")
        return 2
    if not Path(EDGE).exists():
        print(f"FATAL: Edge 不存在：{EDGE}")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="ui_check_6b_"))
    print(f"临时目录：{tmp}")
    exe_proc: subprocess.Popen | None = None
    edge: subprocess.Popen | None = None
    cdp: CDP | None = None
    try:
        # --- 解压 ---
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp / "pkg")
        app_dir = tmp / "pkg" / "DL-AI-skills"
        exe = app_dir / "DL-AI-skills.exe"
        check("zip 解压出 exe", exe.is_file(), str(exe))
        if not exe.is_file():
            return 1
        check("包内含内置 claude.exe", (app_dir / "claude" / "claude.exe").is_file())
        check("包内含前端产物 static/", (app_dir / "_internal" / "static" / "index.html").is_file())

        # --- 起 exe（干净环境） ---
        env = _clean_env(tmp)
        exe_proc = subprocess.Popen(
            [str(exe), "--no-browser"], env=env, cwd=str(app_dir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)

        base = None
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            line = exe_proc.stdout.readline()
            if not line:
                if exe_proc.poll() is not None:
                    break
                continue
            if line.startswith("SERVING "):
                base = line.strip().split()[1]
                break
        check("exe 启动并打印服务地址", bool(base), base or "未解析到 SERVING 行")
        if not base:
            return 1

        # 等就绪
        for _ in range(60):
            try:
                st, _, _ = _http(base + "/api/health", 3)
                if st == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)

        # --- A. 静态服务与 SPA 回退 ---
        st, hdr, body = _http(base + "/")
        check("GET / 返回构建产物页面",
              st == 200 and "text/html" in hdr.get("content-type", "") and b'<div id="root"' in body,
              f"status={st}")
        st, _, body = _http(base + "/some/deep/link")
        check("深链回退 index.html（SPA fallback）", st == 200 and b'<div id="root"' in body, f"status={st}")
        st, _, _ = _http(base + "/api/unknown")
        check("未知 /api 路径保持 404", st == 404, f"status={st}")

        # --- B. env-check（干净环境：git/python 应报缺；数据目录在临时 LOCALAPPDATA 下） ---
        st, _, raw = _http(base + "/api/system/env-check")
        envj = json.loads(raw.decode("utf-8"))
        check("env-check：static_served=True", envj.get("static_served") is True)
        dpath = envj.get("data_dir", {}).get("path", "")
        check("env-check：data_dir 落在临时 LOCALAPPDATA 下",
              str(tmp / "localappdata") in dpath, dpath)
        check("env-check：受限 PATH 下如实报缺 git",
              envj.get("git", {}).get("found") is False, str(envj.get("git")))
        check("env-check：受限 PATH 下如实报缺 python",
              envj.get("python", {}).get("found") is False, str(envj.get("python")))

        # --- C/D. 无头浏览器：首启设置页 + 控制台零错 ---
        profile = tmp / "edge-profile"
        edge_args = [
            EDGE, f"--remote-debugging-port={CDP_PORT}", f"--user-data-dir={profile}",
            "--no-first-run", "--no-default-browser-check", "--disable-gpu",
            "--disable-extensions", "--remote-allow-origins=*", "--window-size=1280,900",
        ]
        if not headful:
            edge_args.append("--headless=new")
        edge_args.append("about:blank")
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
        check("无头 Edge 启动（CDP）", bool(ws_url), f"port {CDP_PORT}")
        if not ws_url:
            return 1
        cdp = CDP(ws_url)
        cdp.call("Runtime.enable")
        cdp.call("Log.enable")
        cdp.call("Page.enable")
        cdp.call("Page.navigate", {"url": base + "/"})
        dialog = False
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if cdp.evaluate('document.querySelector(\'[role="dialog"]\') !== null'):
                dialog = True
                break
            time.sleep(0.3)
        check("首启自动弹出设置页（未配置凭证）", dialog)
        if not dialog:
            try:
                txt = cdp.evaluate("document.body.innerText")
                ls = cdp.evaluate("JSON.stringify(Object.keys(localStorage))")
                print(f"[debug] body.innerText[:300]={str(txt)[:300]!r}")
                print(f"[debug] localStorage keys={ls}")
            except Exception as e:  # noqa: BLE001
                print(f"[debug] 诊断失败：{e}")
        if headful:
            SHOTS.mkdir(parents=True, exist_ok=True)
            shot = SHOTS / "ui_check_6b_frozen.png"
            data = cdp.call("Page.captureScreenshot", {"format": "png"}).get("data")
            if data:
                shot.write_bytes(base64.b64decode(data))
                print(f"[info] 截图：{shot}")

        errs = []
        for ev in cdp.events:
            if ev.get("method") == "Runtime.exceptionThrown":
                errs.append(str(ev.get("params", {}).get("exceptionDetails", {}).get("text")))
            elif ev.get("method") == "Runtime.consoleAPICalled" and ev.get("params", {}).get("type") == "error":
                errs.append("console error")
        errs = [e for e in errs if "favicon" not in e.lower()]
        check("控制台零报错", not errs, "; ".join(errs[:3]))

        passed = sum(1 for _, ok, _ in CHECKS if ok)
        print(f"\n==== ui_check_6b: {passed}/{len(CHECKS)} ====")
        return 0 if passed == len(CHECKS) else 1
    finally:
        if cdp:
            try:
                cdp.close()
            except Exception:
                pass
        if edge:
            edge.terminate()
        if exe_proc and exe_proc.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(exe_proc.pid)],
                           capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
