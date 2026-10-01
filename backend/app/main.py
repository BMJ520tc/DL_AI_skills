"""FastAPI 入口。

服务层是全部能力的中枢，前端不直接访问文件系统与知识库（架构三.1）。
各功能模块路由在后续任务中通过 include_router 挂载。
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import CLAUDE_CLI_PATH, ensure_data_dirs
from app.db.connection import init_db
from app.services import (
    task_manager, agent_service, env_manager, analysis_service,
    preprocess_service, baseline_service, dataset_service, compare_service, download_service,
    paper_service,
)
from app.api import (
    tasks, projects, knowledge, agents, environments, search, analysis, preprocess, datasets, papers,
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
    await task_manager.start()
    yield
    await task_manager.stop()


app = FastAPI(title="DL-AI-skills", lifespan=lifespan)


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


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
