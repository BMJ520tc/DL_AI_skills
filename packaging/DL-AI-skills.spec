# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir 打包配置（一键封装 6.6-b）。

产物：dist/DL-AI-skills/DL-AI-skills.exe + _internal/ + static/ + agents/ + backend/ + scripts/。
运行期随包资源按**仓库相对布局**打进 `sys._MEIPASS`（= dist/DL-AI-skills），与
`app.config.resource_path(rel)` 的 `rel` 一一对应；claude.exe 另由 build 脚本拷到
`dist/DL-AI-skills/claude/claude.exe`（launcher 置 CLAUDE_CLI_PATH）。
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent          # 仓库根（packaging/ 的上一级）
BACKEND = ROOT / "backend"

datas = [
    (str(ROOT / "frontend" / "dist"), "static"),
    (str(ROOT / "agents" / "prompts"), "agents/prompts"),
    (str(BACKEND / "templates" / "train.py"), "backend/templates"),
    (str(BACKEND / "app" / "db" / "schema.sql"), "backend/app/db"),
    (str(BACKEND / "app" / "vendor" / "echarts.min.js"), "backend/app/vendor"),
]
# 随包脚本（含 _viz_common/_model_loader/_pipeline_common/_iter_calls/_data_import 等本地模块）：
# 后端经 `--run-py` 或项目环境解释器执行它们，文件须在包内。
for _p in sorted((ROOT / "scripts").glob("*.py")):
    datas.append((str(_p), "scripts"))

hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto", "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.lifespan.on", "uvicorn.lifespan.off",
    "anyio._backends._asyncio",
]
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("app")   # api/services 经 router 动态加载，保底全收

binaries = []
for _pkg in ("claude_agent_sdk", "fitz", "pymupdf4llm", "pdfplumber", "pdfminer"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h
# mcp：仅 claude_agent_sdk 需要（本项目 MCP server 是手写 stdio，不 import mcp）；
# 排除 mcp.cli —— 它 import 未安装的 typer。
hiddenimports += collect_submodules("mcp", filter=lambda n: not n.startswith("mcp.cli"))
datas += collect_data_files("mcp")

excludes = ["torch", "torchvision", "torchaudio", "matplotlib", "tkinter", "IPython", "notebook"]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(BACKEND), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DL-AI-skills",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DL-AI-skills",
)
