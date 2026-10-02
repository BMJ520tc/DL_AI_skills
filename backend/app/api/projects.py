"""项目管理 API（模块详细设计 2.2）。"""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.services import analysis_service, project_manager, version_service

router = APIRouter(prefix="/api/projects", tags=["projects"])

# 可经 HTTP 打开的画布图表文件图型（模块三 5.5 三张图；与 visualize_service 的图型白名单一致）
FIGURE_CHARTS = ("performance", "error_dist", "cases")


class ProjectCreate(BaseModel):
    project_type: str
    source: str | None = None
    name: str | None = None
    parent_project_id: str | None = None
    source_url: str | None = None  # 仓库地址或本地路径，提供则创建后加载（3.3）


@router.post("")
def create_project(body: ProjectCreate) -> dict:
    try:
        project_id = project_manager.create_project(
            body.project_type, body.source, body.name, body.parent_project_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 画布新建模型（阶段4 4a）：初始化空画布快照并直接置 ready（结构化项目无异步加载流程）
    version: dict | None = None
    version_error: str | None = None
    if body.project_type == "structured":
        project = project_manager.get_project(project_id)
        if project is not None:
            empty_graph = {"nodes": [], "edges": []}
            _graph_path(project).write_text(
                json.dumps(empty_graph, ensure_ascii=False), encoding="utf-8"
            )
            project_manager.update_status(project_id, "ready")
            # 4d-1：git init + 初始提交（空画布）；失败透出在响应里，不连坐项目创建
            try:
                version = version_service.commit_graph(project_id, empty_graph)
            except Exception as e:  # noqa: BLE001
                version_error = f"版本仓库初始化失败（保存画布时会自动重试）: {e}"

    if body.source_url:
        try:
            analysis_service.load_source(project_id, body.source_url)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"加载失败: {e}")

    response: dict = {"project_id": project_id, "status": "loading"}
    if body.project_type == "structured":
        response.update({"version": version, "version_error": version_error})
    return response


@router.get("")
def list_projects(project_type: str | None = None) -> list[dict]:
    return project_manager.list_projects(project_type)


@router.get("/{project_id}")
def get_project(project_id: str) -> dict:
    project = project_manager.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _require_structured(project_id: str) -> dict:
    try:
        return project_manager.require_type(project_id, {"structured"})
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _graph_path(project: dict) -> Path:
    return Path(project["workspace_path"]) / "graph.json"


@router.get("/{project_id}/graph")
def get_graph(project_id: str) -> dict:
    """结构化项目画布快照（GraphIR v2，模块四 6.5/7.1）。"""
    project = _require_structured(project_id)
    p = _graph_path(project)
    if not p.exists():
        raise HTTPException(status_code=404, detail="graph not found")
    return json.loads(p.read_text(encoding="utf-8"))


@router.put("/{project_id}/graph")
def put_graph(project_id: str, body: dict) -> dict:
    """画布保存（GraphIR v2 全量覆盖，模块四 7.1 最小闭环）。

    阶段4 4d-1：保存即提交——graph.json + network_version.json 入工作区 git 仓库
    （未初始化则在此懒初始化）。版本提交失败不连坐保存本身（图已落盘），
    错误透出在响应的 version_error 字段。
    """
    project = _require_structured(project_id)
    if not isinstance(body.get("nodes"), list) or not isinstance(body.get("edges"), list):
        raise HTTPException(status_code=400, detail="非法 GraphIR：需含 nodes/edges 数组")
    _graph_path(project).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    project_manager.update_status(project_id, "ready")
    try:
        version = version_service.commit_graph(project_id, body)
        return {"status": "saved", "version": version}
    except Exception as e:  # noqa: BLE001
        return {"status": "saved", "version": None, "version_error": f"版本提交失败（图已保存）: {e}"}


@router.get("/{project_id}/figures/{chart_type}")
def get_figure(project_id: str, chart_type: str) -> HTMLResponse:
    """模块三 5.5 图表文件服务：自包含 HTML 直接以页面打开（图表路径落盘为服务器本地文件，
    前端无法直接引用，故经本项目端点转发；图型白名单与可视化服务一致，杜绝路径注入）。"""
    if chart_type not in FIGURE_CHARTS:
        raise HTTPException(status_code=404, detail=f"unknown chart type: {chart_type}")
    project = project_manager.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    p = Path(project["workspace_path"]) / "reports" / "figures" / f"{chart_type}.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="figure not found（先运行该图型的可视化）")
    return HTMLResponse(content=p.read_text(encoding="utf-8"), media_type="text/html")
