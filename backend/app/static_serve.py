"""前端构建产物由后端同源服务（一键封装 6.6-a；探索稿 P1）。

开发默认关（沿用 vite dev/preview 双进程），`DL_AI_SERVE_STATIC=1` 显式开、
打包形态（PyInstaller 冻结）默认开（见 `app.config.SERVE_STATIC`）。

挂载带 SPA fallback：未知路径回退到 index.html（前端无 react-router 服务器路由，
深链/刷新不能 404）；`/api/*` 除外——未知 API 路径保持 404 JSON，不回退到页面。
"""
from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException  # StaticFiles 抛的是 starlette 基类，fastapi.HTTPException（子类）捕不到
from starlette.staticfiles import StaticFiles

from app.config import FRONTEND_DIST_DIR, SERVE_STATIC


class SPAStaticFiles(StaticFiles):
    """静态产物服务 + SPA 回退（未知非 API 路径返回 index.html）。"""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or scope.get("path", "").startswith("/api"):
                raise
            return await super().get_response("index.html", scope)


def mount_static(app: FastAPI) -> bool:
    """按配置把 frontend/dist（或打包资源 static/）挂到根路径；返回是否已挂载。"""
    if not SERVE_STATIC:
        return False
    dist = Path(FRONTEND_DIST_DIR)
    if not dist.is_dir():
        print(f"[warn] 静态产物目录不存在（SERVE_STATIC 开启）：{dist}")
        return False
    app.mount("/", SPAStaticFiles(directory=dist, html=True), name="static")
    return True
