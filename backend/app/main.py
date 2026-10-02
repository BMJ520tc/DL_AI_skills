"""FastAPI 入口。

服务层是全部能力的中枢，前端不直接访问文件系统与知识库（架构三.1）。
各功能模块路由在后续任务中通过 include_router 挂载。
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import CLAUDE_CLI_PATH, CORS_ORIGINS, CORS_ORIGIN_REGEX, ensure_data_dirs
from app.db.connection import init_db
from app.services import (
    task_manager, agent_service, env_manager, analysis_service,
    preprocess_service, baseline_service, dataset_service, compare_service, download_service,
    paper_service, decompose_service,
)
from app.api import (
    tasks, projects, knowledge, agents, environments, search, analysis, preprocess, datasets, papers,
    decompose, modules,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    init_db()
    _validate_claude()
    agent_service.register()
    download_service.register()
    env_manager.register()
    analysis_service.register()
    preprocess_service.register()
    baseline_service.register()
    dataset_service.register()
    compare_service.register()
    paper_service.register()
    decompose_service.register()
    await task_manager.start()
    yield
    await task_manager.stop()


app = FastAPI(title="DL-AI-skills", lifespan=lifespan)

# 前端开发态跨域：Vite dev server（任意本机端口）直连后端时必需；生产走同源/反代（架构三.1）。
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_claude() -> None:
    """启动时校验 claude CLI 可用（0.5、架构八.1 版本锁定）。失败仅告警不阻塞。"""
    import subprocess

    if not CLAUDE_CLI_PATH:
        print("[warn] 未找到 claude 可执行文件，agent 任务将失败")
        return
    try:
        proc = subprocess.run([CLAUDE_CLI_PATH, "--version"], capture_output=True, text=True, timeout=15)
        print(f"[claude] {proc.stdout.strip() or proc.stderr.strip()}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] claude --version 校验失败: {e}")

app.include_router(tasks.router)
app.include_router(projects.router)
app.include_router(knowledge.router)
app.include_router(agents.router)
app.include_router(environments.router)
app.include_router(search.router)
app.include_router(analysis.router)
app.include_router(preprocess.router)
app.include_router(datasets.router)
app.include_router(papers.router)
app.include_router(decompose.router)
app.include_router(decompose.ir_router)
app.include_router(modules.router)


@app.post("/api/torchlens")
def torchlens_unavailable() -> dict:
    """沙盒画布的形状追踪通道（基底 torchlens）。

    实现位于独立进程 `backend/runner.py`（容器通道，属可选扩展，默认不启用）。
    本端点显式返回 503（而非 404），让前端能给出「追踪不可用，需启动 runner」的明确提示。
    """
    raise HTTPException(
        status_code=503,
        detail=(
            "形状追踪不可用：torchlens 通道需独立启动 backend/runner.py（容器为可选扩展，默认未启用）；"
            "独立的形状追踪请用模块四的 POST /api/projects/{id}/decompose/trace"
        ),
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
