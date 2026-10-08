"""启动就绪自检（一键启动 start.bat 用）。

等后端起来后调一次 GET /api/system/env-check，把「git / Python / conda / claude CLI /
模型凭证 / 静态页面 / 数据目录」打印成一页摘要，让用户一眼看清 **claude 有没有起、
凭证缺不缺**。只读、不花 token、不改任何状态。

用法（start.bat 会调）：python scripts/startup_check.py [http://127.0.0.1:8000]
"""
import json
import sys
import time
import urllib.request


def _ok(flag) -> str:
    return "[OK]" if flag else "[!!]"


def _fetch(url: str, timeout_s: int = 45):
    """轮询后端直到就绪；返回 env-check 的 dict，超时返回 None。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/system/env-check", timeout=3) as resp:
                if getattr(resp, "status", 200) == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 —— 后端还没起来属正常，继续轮询
            pass
        time.sleep(1)
    return None


def _match_console_encoding() -> None:
    """让输出编码与控制台代码页一致（cmd 默认 936=GBK）；非控制台/拿不到则不改。"""
    try:
        import ctypes

        cp = ctypes.windll.kernel32.GetConsoleOutputCP()
        if cp:
            enc = "utf-8" if cp == 65001 else ("cp%d" % cp)
            sys.stdout.reconfigure(encoding=enc, errors="replace")
    except Exception:  # noqa: BLE001 —— 拿不到就退回默认行为
        pass


def main() -> int:
    _match_console_encoding()
    url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    data = _fetch(url)

    if data is None:
        print("")
        print("================ 启动自检 ================")
        print("  [!!] 后端在 45s 内未就绪，请查看窗口里的 uvicorn 报错。")
        print("==========================================")
        return 1

    git = data.get("git", {})
    py = data.get("python", {})
    conda = data.get("conda", {})
    cli = data.get("claude_cli", {})
    creds = bool(data.get("credentials_configured"))
    data_dir = data.get("data_dir", {})

    print("")
    print("================ 启动自检（后端已就绪）================")
    print("  地址          : %s" % url)
    print("  git           : %s %s" % (_ok(git.get("found")), git.get("version") or ""))
    print("  Python        : %s %s" % (_ok(py.get("found")), py.get("python") or py.get("py_launcher") or ""))
    print("  conda         : %s %s" % (_ok(conda.get("found")), conda.get("path") or ""))
    print("  claude CLI    : %s %s" % (_ok(cli.get("found")), cli.get("version") or ""))
    print("  模型凭证      : %s %s" % (_ok(creds), "已配置" if creds else "未配置"))
    print("  静态页面      : %s" % _ok(data.get("static_served")))
    print("  数据目录      : %s %s" % (_ok(data_dir.get("writable")), data_dir.get("path") or ""))
    print("  ------------------------------------------------------")

    if not cli.get("found"):
        print("  [需处理] 未找到 claude CLI —— 大模型步骤（复现/拆解/蒸馏/助手）不能用。")
        print("           装法：先装 Node.js，再执行  npm i -g @anthropic-ai/claude-code")
    if not creds:
        print("  [需处理] 未配置模型接口凭证 —— 大模型步骤不能用。")
        print("           在浏览器界面里打开「设置」填写端点与密钥（本机不需要改脚本）。")
    if cli.get("found") and creds:
        print("  一切就绪：大模型步骤（复现/拆解/蒸馏/助手）可用。")
    print("======================================================")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
