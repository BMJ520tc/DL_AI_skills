"""模块四 API（模块详细设计 6.1/6.2/6.3/6.4）。

任务端点收拢在 /decompose/* 前缀下（实施约定：/verify 已被模块一占用，
analysis.py:22）；IR 读写与调参在 /ir/* 下（ir_router，前缀 /api/projects/{id}）。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import decompose_service, project_manager
from app.services.ir_codegen import IrIncompleteError

router = APIRouter(prefix="/api/projects/{project_id}/decompose", tags=["decompose"])
ir_router = APIRouter(prefix="/api/projects/{project_id}", tags=["decompose"])


def _require_original(project_id: str) -> None:
    try:
        project_manager.require_type(project_id, {"original"})
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))


class DecomposeBody(BaseModel):
    entry_class: str | None = None  # 可选：指定入口类（论文库二次验证等场景，规避多根类歧义）


@router.post("")
def decompose(project_id: str, body: DecomposeBody | None = None) -> dict:
    """触发 agent 解析（前置：structure_report.json 存在，缺失即 409）。"""
    try:
        decompose_service.decompose_precheck(project_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {
        "task_id": decompose_service.decompose(project_id, body.entry_class if body else None),
        "status": "queued",
    }


@router.post("/trace")
def trace(project_id: str) -> dict:
    """项目环境跑模型 hook 回填缺失形状。"""
    _require_original(project_id)
    return {"task_id": decompose_service.trace(project_id), "status": "queued"}


@router.post("/regenerate")
def regenerate(project_id: str) -> dict:
    """同步再生成（6.3）：IR 不完整 → 400 附缺失项清单。"""
    _require_original(project_id)
    try:
        code = decompose_service.regenerate(project_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except IrIncompleteError as e:
        missing = [m for m in str(e).split("；") if m]
        raise HTTPException(status_code=400, detail={"error": "IR 不完整，无法再生成", "missing": missing})
    return {"code": code}


@router.post("/verify")
def verify(project_id: str) -> dict:
    """两步验证（结构+数值），结果落 reports/verification.json（比对不过任务仍 success）。"""
    _require_original(project_id)
    return {"task_id": decompose_service.verify(project_id), "status": "queued"}


@router.get("/verify")
def get_verification(project_id: str) -> dict:
    """最近一次验证记录（未知项目 404、类型不符 400——原先未映射会 500）。"""
    try:
        decompose_service.verification_precheck(project_id)
        verification = decompose_service.get_verification(project_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if verification is None:
        raise HTTPException(status_code=404, detail="verification not found：请先 POST /decompose/verify")
    return verification


@ir_router.get("/ir")
def get_ir(project_id: str) -> dict:
    """IR 快照 + 验证新鲜度（none=未验证 / valid=一致 / stale=调参后未重验）+ 可再生成性校验。"""
    _require_original(project_id)
    ir = decompose_service.read_ir(project_id)
    if ir is None:
        raise HTTPException(status_code=404, detail="ir not found：请先 POST /decompose")
    status, verification = decompose_service.verification_status(project_id)
    return {"ir": ir, "verification_status": status, "verification": verification,
            "ir_errors": decompose_service.edit_errors(ir),
            "ir_warnings": decompose_service.edit_warnings(ir)}


class NodeEditBody(BaseModel):
    """改节点（PUT /ir/nodes/{id}）：只回写显式给出的字段（exclude_unset）。

    parent_id/code_hint 传 `null` = 清除该字段；params 传 `{...}` = 整体替换。
    """

    kind: str | None = None
    class_name: str | None = None
    parent_id: str | None = None
    params: dict | None = None
    code_hint: str | None = None
    module_path: str | None = None
    module_file: str | None = None


class NodeCreateBody(BaseModel):
    """新增节点（POST /ir/nodes）。"""

    id: str
    kind: str
    class_name: str
    parent_id: str | None = None
    params: dict | None = None
    code_hint: str | None = None
    module_path: str | None = None
    module_file: str | None = None


class EdgeBody(BaseModel):
    """新增边（POST /ir/edges）。`from` 是 Python 关键字，用 alias 收。"""

    from_: str = Field(alias="from")
    to: str
    tensor_shape: list[int] | None = None

    model_config = {"populate_by_name": True}


class InputSpecBody(BaseModel):
    shape: list[int]
    dtype: str | None = None
    # 多输入模型的额外入参（如 scGPT 的 forward(src, values, src_key_padding_mask)）；[] = 清除
    extra: list[dict] | None = None
    # forward 关键字参数（如 scGPT 的 CLS/MVC/ECS 开关）；{} = 清除
    forward_kwargs: dict | None = None
    # 多输入模型：按调用顺序列出吃外部输入的节点 id（root 的 forward 形参）；[] = 清除
    inputs: list[str] | None = None


@ir_router.put("/ir/input_spec")
def update_input_spec(project_id: str, body: InputSpecBody) -> dict:
    """修正入口输入规格（agent 给不出具体维度时的补参通道）；写回后旧验证变 stale。"""
    _require_original(project_id)
    try:
        return decompose_service.update_input_spec(
            project_id, body.shape, body.dtype, body.extra, body.forward_kwargs, body.inputs)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class EntryArgsBody(BaseModel):
    entry_args: dict


@ir_router.put("/ir/entry_args")
def update_entry_args(project_id: str, body: EntryArgsBody) -> dict:
    """修正入口类构造参数（模型需要运行期配置/外部数据时的补参通道）；写回后旧验证变 stale。"""
    _require_original(project_id)
    try:
        return decompose_service.update_entry_args(project_id, body.entry_args)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@ir_router.put("/ir/nodes/{node_id}")
def update_node(project_id: str, node_id: str, body: NodeEditBody) -> dict:
    """改节点（6.2 调参 + 结构编辑）；写回后旧验证经 ir_hash 变 stale，入库前需重新验证。"""
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(status_code=400, detail="patch 为空：请至少给出一个要改的字段")
    _require_original(project_id)
    try:
        return decompose_service.update_node(project_id, node_id, patch)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@ir_router.post("/ir/nodes")
def add_node(project_id: str, body: NodeCreateBody) -> dict:
    """新增节点；结构改动后 IR 变 stale，响应回传当前校验结果。"""
    _require_original(project_id)
    try:
        return decompose_service.add_node(project_id, body.model_dump())
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@ir_router.delete("/ir/nodes/{node_id}")
def delete_node(project_id: str, node_id: str, recursive: bool = False) -> dict:
    """删除节点及关联边（根节点不可删；有子节点须 recursive=true）。"""
    _require_original(project_id)
    try:
        return decompose_service.delete_node(project_id, node_id, recursive)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@ir_router.post("/ir/edges")
def add_edge(project_id: str, body: EdgeBody) -> dict:
    """新增边 from→to（两端须存在、非自环、无重边、不成环）。"""
    _require_original(project_id)
    try:
        return decompose_service.add_edge(project_id, body.from_, body.to, body.tensor_shape)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@ir_router.delete("/ir/edges/{from_node}/{to_node}")
def delete_edge(project_id: str, from_node: str, to_node: str) -> dict:
    """删除 from→to 的边。"""
    _require_original(project_id)
    try:
        return decompose_service.delete_edge(project_id, from_node, to_node)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
