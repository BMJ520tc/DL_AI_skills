"""打包形态入口（PyInstaller onedir，一键封装 6.6-b）。

职责（顺序要紧）：
  ① 派发模式：`--run-py <script> [args...]`（用内嵌解释器跑随包脚本）、`--mcp-knowledge`（当知识库 stdio MCP）。
  ② 置 `CLAUDE_CLI_PATH` 指向随包内置的 claude.exe（**必须在 import app.config 之前**，config 导入即固化）。
  ③ 首启数据目录迁移提示（旧数据目录 + 新目录空 → 一次性选择）。
  ④ 选空闲端口 → 打印 `SERVING http://127.0.0.1:<port>` → 就绪后开浏览器 + 打印就绪自检。
  ⑤ `uvicorn.run(app)`。
"""
import os
import runpy
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _exe_dir() -> Path:
    return Path(sys.executable).resolve().parent


def _dispatch() -> None:
    """`--run-py` / `--mcp-knowledge` 两种派发模式（不启动服务）。"""
    if "--mcp-knowledge" in sys.argv:
        from app.mcp.knowledge_mcp import main as mcp_main

        mcp_main()
        raise SystemExit(0)

    if "--run-py" in sys.argv:
        i = sys.argv.index("--run-py")
        script, rest = sys.argv[i + 1], sys.argv[i + 2:]
        # 模拟 `python script.py`：sys.path[0] = 脚本所在目录（脚本常 `from _viz_common import ...`）。
        sys.path.insert(0, str(Path(script).resolve().parent))
        sys.argv = [script, *rest]
        runpy.run_path(script, run_name="__main__")
        raise SystemExit(0)


def _set_claude_cli() -> None:
    if os.getenv("CLAUDE_CLI_PATH"):
        return
    for c in (_exe_dir() / "claude" / "claude.exe", _exe_dir().parent / "claude" / "claude.exe"):
        if c.exists():
            os.environ["CLAUDE_CLI_PATH"] = str(c)
            return


def _maybe_migrate() -> None:
    """首启数据目录迁移提示（6.6-a 遗留）：旧数据目录有内容、新目录空、且未标记 → 让用户二选一。"""
    if os.getenv("DL_AI_DATA_DIR"):
        return
    local = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "DL-AI-skills" / "data"
    legacy = _exe_dir() / "data"
    marker = local / ".migrated"
    if marker.exists() or not legacy.is_dir():
        return
    try:
        if not any(legacy.iterdir()):
            return
        new_empty = (not local.exists()) or (not any(local.iterdir()))
    except OSError:
        return
    if not new_empty:
        return

    print("[首启] 检测到随程序目录的既有数据：%s" % legacy, flush=True)
    print("[首启]   新的用户数据目录（%s）为空。" % local, flush=True)
    print("[首启]   1) 沿用随程序目录的数据（配置指向它，不复制）", flush=True)
    print("[首启]   2) 初始化新的用户数据目录（默认）", flush=True)
    if sys.stdin and sys.stdin.isatty():
        try:
            choice = input("[首启] 请选择 [1/2]（默认 2）：").strip()
        except EOFError:
            choice = "2"
    else:
        choice = "2"
    if choice == "1":
        os.environ["DL_AI_DATA_DIR"] = str(legacy)
        print("[首启] 已沿用 %s" % legacy, flush=True)
    else:
        print("[首启] 使用新数据目录 %s" % local, flush=True)
    try:
        local.mkdir(parents=True, exist_ok=True)
        marker.write_text("migrated", encoding="utf-8")
    except OSError:
        pass


def _free_port() -> int:
    for _ in range(20):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        if port:
            return port
    return 8000


def _setup_logging(data_dir: Path) -> None:
    import logging
    from logging.handlers import RotatingFileHandler

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    try:
        log_dir = data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass


def _open_when_ready(url: str, open_browser: bool) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=2):
                break
        except Exception:  # noqa: BLE001 —— 后端还没起来
            time.sleep(0.5)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    _dispatch()
    _set_claude_cli()
    _maybe_migrate()

    from app.config import DATA_DIR, FRONTEND_DIST_DIR, resource_path

    _setup_logging(DATA_DIR)
    import logging

    log = logging.getLogger("launcher")
    if not FRONTEND_DIST_DIR.is_dir():
        log.warning("未找到前端产物目录：%s（界面将不可用）", FRONTEND_DIST_DIR)

    port = _free_port()
    url = "http://127.0.0.1:%d" % port
    # 供验收脚本解析；改变格式前先改 scripts/ui_check_6b.py。
    print("SERVING %s" % url, flush=True)
    log.info("数据目录：%s", DATA_DIR)
    log.info("服务地址：%s", url)

    open_browser = "--no-browser" not in sys.argv
    threading.Thread(target=_open_when_ready, args=(url, open_browser), daemon=True).start()

    # 就绪自检（等后端起来后打印一份 git/Python/conda/claude CLI/凭证 摘要），与上方并行。
    try:
        subprocess.Popen([sys.executable, "--run-py", str(resource_path("scripts/startup_check.py")), url])
    except OSError:
        pass

    from app.main import app
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
