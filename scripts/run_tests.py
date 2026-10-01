"""后端测试入口：在 backend/ 下用当前解释器跑 pytest。

用法：

    D:\\python.exe scripts/run_tests.py              # 全部用例
    D:\\python.exe scripts/run_tests.py -k task      # 透传 pytest 参数
    D:\\python.exe scripts/run_tests.py tests/test_api_smoke.py -v

退出码：pytest 的退出码（0=全通过；1=有失败；2=中断；5=未收集到用例）；
未安装 pytest 时返回 2 并提示安装命令。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"


def main() -> int:
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(
            "缺少 pytest，请先执行：\n"
            f"  {sys.executable} -m pip install -r backend/requirements-dev.txt",
            file=sys.stderr,
        )
        return 2
    if not BACKEND.is_dir():
        print(f"未找到后端目录：{BACKEND}", file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "pytest", *sys.argv[1:]]
    print("$ " + " ".join(cmd) + f"   (cwd={BACKEND})", flush=True)
    return subprocess.call(cmd, cwd=BACKEND)


if __name__ == "__main__":
    raise SystemExit(main())
