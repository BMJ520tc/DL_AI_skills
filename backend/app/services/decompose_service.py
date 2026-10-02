"""模块四：模型拆解→可视化→再生成→两步验证→标准化入库（模块详细设计六章 6.1~6.6）。

四个后台任务 + 两个同步入口：
- decompose        6.1 agent 解析模型 → IR（validate_ir 硬校验后落盘 reports/ir.json）
- decompose_trace  6.2 项目环境跑模型 hook 回填缺失形状（scripts/trace_shapes.py）
- decompose_verify 6.4 结构+数值两步验证（scripts/verify_decompose.py，项目环境执行）
- module_ingest    6.5 验证通过 → 模块包入库 + 结构化项目（画布可编辑，graph.json）
- regenerate（同步端点）6.3 IR → 自包含 PyTorch 代码（ir_codegen.py，不产生任务）
- 查询/调参（同步）: read_ir / get_verification / update_node_params

验证不过（overall=failed）时任务本身仍 success，结果经 run_record(status=failed,
error=failure_reason) 可检索供 agent/用户改进；入库前置强校验 ir_hash 防 stale。
"""
import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import MODULES_DIR, PROJECT_ROOT
from app.services import (
    agent_service, analysis_service, ir_codegen, ir_graphir, ir_schema,
    knowledge_service, proc_util, project_manager, task_manager, version_service,
)
from app.services.ir_codegen import IrIncompleteError
from app.services.ir_schema import (
    SCHEMA_VERSION, TASK_TYPES, _as_ir, ir_hash, nodes_by_id, normalize_class_name, validate_ir,
)

logger = logging.getLogger(__name__)

TASK_DECOMPOSE = "decompose"
TASK_TRACE = "decompose_trace"
TASK_VERIFY = "decompose_verify"
TASK_INGEST = "module_ingest"

TRACE_SCRIPT = PROJECT_ROOT / "scripts" / "trace_shapes.py"
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_decompose.py"

# 数值比对阈值（实施约定；仿 REPRO_DEVIATION_* 环境变量覆盖先例）
DECOMPOSE_NUM_RTOL = float(os.getenv("DECOMPOSE_NUM_RTOL", "1e-5"))
DECOMPOSE_NUM_ATOL = float(os.getenv("DECOMPOSE_NUM_ATOL", "1e-6"))
DECOMPOSE_NUM_SEEDS = os.getenv("DECOMPOSE_NUM_SEEDS", "42,1337,2024")
DECOMPOSE_AGENT_TIMEOUT_S = 1800
DECOMPOSE_AGENT_RETRIES = int(os.getenv("DECOMPOSE_AGENT_RETRIES", "3"))  # 结构化输出失败重试次数（3.9 风险）
DECOMPOSE_AGENT_BUDGET_S = int(os.getenv("DECOMPOSE_AGENT_BUDGET_S", "3600"))  # 重试总预算
DECOMPOSE_TRACE_TIMEOUT_S = 600
DECOMPOSE_VERIFY_TIMEOUT_S = 1800


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ws(project: dict) -> Path:
    return Path(project["workspace_path"])


def _require_original(project_id: str) -> dict:
    return project_manager.require_type(project_id, {"original"})


def _read_ir(project: dict) -> Optional[dict]:
    p = _ws(project) / "reports" / "ir.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_ir(project_id: str) -> Optional[dict]:
    return _read_ir(_require_original(project_id))


def _write_ir(project: dict, ir: dict) -> None:
    p = _ws(project) / "reports" / "ir.json"
    p.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")


def get_verification(project_id: str) -> Optional[dict]:
    p = _ws(_require_original(project_id)) / "reports" / "verification.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def verification_precheck(project_id: str) -> None:
    """查询前置：项目存在且为原始项目（供 API 映射 404/400，避免内部异常冒泡成 500）。"""
    _require_original(project_id)


def verification_status(project_id: str) -> tuple[str, Optional[dict]]:
    """验证新鲜度：none=未验证 / valid=ir_hash 一致 / stale=IR 已修改未重验。"""
    ir = read_ir(project_id)
    verification = get_verification(project_id)
    if verification is None:
        return "none", None
    if ir is not None and verification.get("ir_hash") == ir_hash(ir):
        return "valid", verification
    return "stale", verification


# --------------------------- 同步入口 ---------------------------

def decompose(project_id: str, entry_class: str | None = None) -> str:
    _require_original(project_id)
    params: dict = {"project_id": project_id}
    if entry_class:
        params["entry_class"] = entry_class
    return task_manager.create_task(TASK_DECOMPOSE, project_id=project_id, params=params)


def trace(project_id: str) -> str:
    _require_original(project_id)
    return task_manager.create_task(TASK_TRACE, project_id=project_id, params={"project_id": project_id})


def verify(project_id: str) -> str:
    _require_original(project_id)
    return task_manager.create_task(TASK_VERIFY, project_id=project_id, params={"project_id": project_id})


def ingest(project_id: str) -> str:
    _require_original(project_id)
    return task_manager.create_task(TASK_INGEST, project_id=project_id, params={"project_id": project_id})


def decompose_precheck(project_id: str) -> None:
    """POST /decompose 的同步前置（3.1）：结构分析报告须已存在，否则 409（映射在 API 层）。"""
    project = _require_original(project_id)
    if not (_ws(project) / "reports" / "structure_report.json").exists():
        raise FileNotFoundError("尚未完成代码结构分析：请先 POST /api/projects/{id}/analyze")


def ingest_precheck(project_id: str) -> dict:
    """POST /api/modules 的同步前置（3.5）：IR 存在 → 验证 passed → ir_hash 未 stale。

    返回 {"module_id", "module_version"}（供 API 提示）；任一条件不满足即抛错，
    API 层映射为 404（缺产物）/400（验证未过）/409（已过期或同版本已入库）。
    """
    project = _require_original(project_id)
    ir = _read_ir(project)
    if ir is None:
        raise LookupError("尚未拆解：请先 POST /api/projects/{id}/decompose")
    verification = get_verification(project_id)
    if verification is None:
        raise LookupError("尚未验证：请先 POST /api/projects/{id}/decompose/verify")
    if verification.get("overall") != "passed":
        raise PermissionError("验证未通过，不能入库；请调整 IR 后重新验证")
    if verification.get("ir_hash") != ir_hash(ir):
        raise ValueError("IR 已修改（调参）但未重新验证，验证结果已过期；请重新执行 verify")
    module_id = "mod_" + _module_signature(ir)
    module_version = knowledge_service.next_module_version(module_id)
    if knowledge_service.get_module(module_id, module_version):
        raise ValueError(f"模块 {module_id} 的 {module_version} 已入库，请确认后再入库")
    # 需求/设计 6.5 异常与边界「重复模块（同项目同结构）→ 提示已有，支持覆盖为新版本」：
    # 不阻断，但把已有版本回给调用方，由界面提示「同结构此前已入库 vN，本次生成新版本」
    return {
        "module_id": module_id,
        "module_version": module_version,
        "existing_versions": knowledge_service.list_module_versions(module_id),
    }


def regenerate(project_id: str) -> str:
    """6.3 同步再生成：IR 结构错误/不完整 → IrIncompleteError（message 为缺失项清单）。"""
    ir = read_ir(project_id)
    if ir is None:
        raise LookupError("ir not found：请先 POST /api/projects/{id}/decompose")
    return ir_codegen.generate(ir)


def update_input_spec(project_id: str, shape: list, dtype: Optional[str] = None) -> dict:
    """修正入口输入规格（6.1/6.2）：agent 对「尺寸由运行期构造参数决定」的模型给不出具体维度时，
    由用户/工具补上；写回后 ir_hash 变化 → 旧验证变 stale（与调参同口径）。

    shape 必须是非空正整数数组（不允许 null 维度：无法据此构造输入）。
    """
    project = _require_original(project_id)
    ir = _read_ir(project)
    if ir is None:
        raise LookupError("ir not found：请先 POST /api/projects/{id}/decompose")
    if not isinstance(shape, list) or not shape or not all(isinstance(d, int) and d > 0 for d in shape):
        raise ValueError("shape 必须是非空正整数数组（如 [1, 64]），不接受 null/非正数维度")
    spec = dict(ir.get("input_spec") or {})
    spec["shape"] = shape
    if dtype:
        spec["dtype"] = dtype
    ir["input_spec"] = spec
    _write_ir(project, ir)
    return spec


def update_node_params(project_id: str, node_id: str, params: dict) -> dict:
    """PUT 调参回写（6.2「用户能手动调节层级的参数」）：写回后旧验证经 ir_hash 变 stale。"""
    project = _require_original(project_id)
    ir = _read_ir(project)
    if ir is None:
        raise LookupError("ir not found：请先 POST /api/projects/{id}/decompose")
    node = nodes_by_id(ir).get(node_id)
    if node is None:
        raise LookupError(f"node not found: {node_id}")
    node["params"] = params
    _write_ir(project, ir)
    return node


# --------------------------- 6.1 拆解 ---------------------------

def _decompose_prompt(hierarchy: list[dict], entry_class: str | None = None) -> str:
    roots = [h for h in hierarchy if h.get("parent") == "Module"]
    parts = [
        "任务：阅读深度学习项目代码，把入口 PyTorch 模型精确拆解为结构化 IR（中间表示），"
        "用于后续确定性代码再生成与两步验证。\n\n"
        "【红线】只允许读取文件（Read/Glob/Grep），禁止修改、创建、删除项目内任何文件。\n\n"
        "【静态结构报告】识别到的 nn.Module 子类：\n"
        f"{json.dumps(hierarchy, ensure_ascii=False)}\n",
    ]
    if entry_class:
        parts.append(
            f"【指定入口类】用户已指定 entry_class={entry_class}，必须以此类为入口模型根节点"
            "（其他类按第 3 条作为子模块展开）。\n"
        )
    elif roots:
        parts.append(
            f"其中直接继承 nn.Module 的类：{json.dumps([h['class'] for h in roots], ensure_ascii=False)}；"
            "请从这些类中确定模型的实际入口类作为 entry_class（通常就是它）。\n"
        )
    parts.append(
        "【IR 规范】\n"
        "1. source_file=入口类所在文件（相对路径）；entry_class=入口类名；task_type 从 "
        f"{json.dumps(list(TASK_TYPES), ensure_ascii=False)} 中选；"
        'input_spec 填入口模型输入（shape 数组 + dtype），如 {"shape":[1,3,32,32],"dtype":"float32"}；'
        "root_id 指向入口类对应的节点。\n"
        "2. 每个 nn.Module 子类实例一个节点。入口模型实例为根节点（kind=module，parent_id=null，module_path=\"\"）。\n"
        "3. 自定义 nn.Module 子类（如 BasicBlock）必须作为 kind=module 节点，其内部子模块作为子节点"
        "（parent_id 指向它），逐层展开到叶子；class_name 用源码类名。\n"
        "4. 叶子层为 torch.nn 内置层：kind=leaf，class_name 用 nn.X 形式（nn.Conv2d、nn.BatchNorm2d、"
        "nn.ReLU、nn.Dropout 等）。params 为该层构造参数（PyTorch 构造器关键字，如 "
        '{"in_channels":64,"out_channels":128,"kernel_size":3}），必须从源码读出真实值，不要臆造；'
        "读不出就留空并置 uncertain=true（后续可人工补参数）。\n"
        "5. nn.Sequential 为 kind=container、class_name=\"nn.Sequential\"，其成员按顺序作为子节点。"
        "nn.ModuleList 展开为父模块的直接子节点（不建 container）。\n"
        "6. forward 中的函数式操作建 kind=op 节点：class_name 用小写名（白名单: add/sub/mul/div/"
        "matmul/bmm/cat/relu/sigmoid/tanh/softmax/flatten/mean/max/min/sum/view/reshape/permute）；"
        "白名单外的操作填 code_hint 内联表达式模板，输入变量用 {inputs} 占位。op 是叶子（无子节点）；"
        "op 可有多条入边（如残差相加），非 op 节点最多 1 条入边。\n"
        "7. 数据流用 edges 表达（from/to 为节点 id）。同层节点按 forward 执行顺序连边；残差/跳跃连接"
        "直接连到汇合 op（如 add）。Sequential 内部成员之间不连边（顺序由声明序决定），Sequential 整体"
        "与外部节点的边连在 container 节点上。\n"
        "8. 节点 id 全局唯一，须为合法 Python 标识符（[A-Za-z_][A-Za-z0-9_]*，不能含点/连字符）；"
        "建议直接用源码中的属性名（如 conv1、bn1、layer1），重名时加前缀区分。\n"
        "9. module_path 填该实例在 named_modules() 中的完整路径（如 layer1.0.conv1、downsample.0），"
        "用于形状追踪与验证对齐；根节点 module_path=\"\"。\n"
        "10. input_shape/output_shape 能静态推出就填数组（元素为整数），推不出填 null。拿不准的节点置 uncertain=true。\n"
        "11. 输出必须是单层 JSON 对象（不要用 {\"ir\":...} 包裹），顶层键：source_file、entry_class、"
        "task_type、input_spec、root_id、nodes、edges。\n"
        '示例（片段）：{"root_id":"net","nodes":[{"id":"net","kind":"module","class_name":"Net",'
        '"parent_id":null,"module_path":""},{"id":"conv1","kind":"leaf","class_name":"nn.Conv2d",'
        '"parent_id":"net","module_path":"conv1","params":{"in_channels":3,"out_channels":16,"kernel_size":3}},'
        '{"id":"skip_add","kind":"op","class_name":"add","parent_id":"net"}],'
        '"edges":[{"from":"conv1","to":"skip_add"}]}'
    )
    return "".join(parts)


async def _run_decompose(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = _require_original(project_id)
    ws = _ws(project)
    source = ws / "source"
    report_path = ws / "reports" / "structure_report.json"
    if not report_path.exists():
        raise RuntimeError("尚未完成代码结构分析：请先 POST /api/projects/{id}/analyze")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hierarchy = report.get("module_hierarchy") or []
    if not hierarchy:
        raise RuntimeError("结构分析报告未识别到 nn.Module 子类，无法拆解")

    started = _now()
    entry_class = params.get("entry_class")
    # 任务前带入知识（需求六.1、数据设计三.3）：已确认的蒸馏结论附进 prompt
    try:
        knowledge = knowledge_service.bring_knowledge(model=entry_class)
    except Exception:  # noqa: BLE001 —— 带入失败不影响拆解
        knowledge = {}
    hint = _knowledge_hint(knowledge)
    prompt = _decompose_prompt(hierarchy, entry_class) + hint
    deadline = time.monotonic() + DECOMPOSE_AGENT_BUDGET_S
    ir: Optional[dict] = None
    last_reason = "agent 未产出有效 IR 结构（structured_output 缺失）"

    def _record_fail(reason: str, bad_ir: Optional[dict] = None) -> None:
        """失败也要留 run_record（架构九.4：失败可检索供 agent 改进）。"""
        entry: dict = {
            "project_id": project_id, "task_id": task_id, "run_type": "decompose",
            "command": "agent parse (structure → IR)", "status": "failed", "error": reason,
            "started_at": started, "finished_at": _now(),
        }
        if bad_ir is not None:
            entry["metrics"] = {"nodes": len(bad_ir.get("nodes") or []),
                                "edges": len(bad_ir.get("edges") or [])}
        knowledge_service.record_run(entry)

    for attempt in range(1, DECOMPOSE_AGENT_RETRIES + 1):
        budget = deadline - time.monotonic()
        if attempt > 1 and budget <= 30:
            last_reason = (
                f"agent 连续 {attempt - 1} 次未产出可用 IR，重试预算（{DECOMPOSE_AGENT_BUDGET_S}s）已用尽"
            )
            break
        # 重试时把上一次的失败原因回喂，否则同样的 prompt 只会得到同样的结果
        ask = prompt if attempt == 1 else prompt + f"\n\n【上一次尝试未通过，请据此修正】{last_reason}"
        try:
            result = await agent_service.run_sync(
                ask,
                cwd=str(source),
                output_schema=ir_schema.IR_SCHEMA,
                max_turns=60,
                timeout_s=int(min(DECOMPOSE_AGENT_TIMEOUT_S, budget)),
            )
        except Exception as e:  # noqa: BLE001 —— agent 失败记 run_record 供检索（先例 4.3）
            _record_fail(f"第 {attempt}/{DECOMPOSE_AGENT_RETRIES} 次尝试失败: {e}")
            raise
        candidate = _as_ir(result.get("structured_output"))
        if candidate is None:
            last_reason = (
                f"agent 未产出有效 IR 结构（structured_output 缺失，第 {attempt}/{DECOMPOSE_AGENT_RETRIES} 次）"
            )
            continue
        errors = validate_ir(candidate)
        if errors:
            last_reason = "IR 结构校验失败: " + "；".join(errors)
            _record_fail(last_reason, candidate)
            continue
        # 结构自检：能否再生成（纯函数、秒级）。agent 对「结构由运行期构造参数决定」的模型
        # （如 GEARS 的 MLP）产出不稳定（漏边、空容器、参数写成 `sizes[0]` 这类表达式），
        # 这类问题先前只到用户点「再生成」才暴露 → 提前拦下并重试，使链路对抖动可自愈。
        try:
            ir_codegen.generate(candidate)
        except (IrIncompleteError, ValueError) as e:
            last_reason = "IR 结构自检未通过（再生成失败）: " + str(e)
            _record_fail(last_reason, candidate)
            continue
        ir = candidate
        break
    if ir is None:
        _record_fail(last_reason)
        raise RuntimeError(last_reason)
    ir["schema_version"] = SCHEMA_VERSION
    ir["project_id"] = project_id
    ir_path = ws / "reports" / "ir.json"
    ir_path.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")

    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "decompose",
        "command": "agent parse (structure → IR)",
        "params": {"entry_class": ir.get("entry_class"), "source_file": ir.get("source_file")},
        "status": "success",
        "metrics": {
            "nodes": len(ir["nodes"]),
            "edges": len(ir["edges"]),
            "uncertain": sum(1 for n in ir["nodes"] if n.get("uncertain")),
            "knowledge_brought": {
                "param_advice": len(knowledge.get("param_advice") or []),
                "dependency_conflict": len(knowledge.get("dependency_conflict") or []),
            },
        },
        "artifact_path": str(ir_path),
        "started_at": started, "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {
        "entry_class": ir.get("entry_class"),
        "nodes": len(ir["nodes"]),
        "edges": len(ir["edges"]),
        "ir_path": str(ir_path),
        # 形状缺失 → 前端提示可跑「补形状」（3.2）
        "shapes_missing": any(
            n.get("input_shape") is None or n.get("output_shape") is None for n in ir["nodes"]
        ),
    })


# --------------------------- 6.2 形状追踪 ---------------------------

# 维度类参数（in_features/out_channels/…）在构造签名里没有默认值，因此必然出现在 trace 的
# `params_delta` 中；agent 给错（或给成 `'sizes[0]'` 这类表达式）时由回填改写为模型实际值，
# 保证原模型与再生成模型同维（比对才有意义）。


def _same_param(a, b) -> bool:
    """参数等价判定：3 与 (3,3)、1 与 (1,1) 视为同一值（避免无谓改写 `params` 让验证变 stale）。"""
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same_param(x, y) for x, y in zip(a, b))
    if isinstance(a, (tuple, list)) and a and all(x == a[0] for x in a):
        return _same_param(a[0], b)
    if isinstance(b, (tuple, list)) and b and all(x == b[0] for x in b):
        return _same_param(a, b[0])
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b


def _merge_shapes(ir: dict, trace_output: dict) -> int:
    """把 trace 捕获的形状与层构造参数回填 IR 的缺失项。

    trace 产物两种形态都接受：`{path: {input_shape, output_shape}}`（旧）与
    `{"shapes": {...}, "order": [{"path","class_name"}]}`（新，附类名以便兜底对齐）。
    兜底（3.2）：节点没有 module_path 时，按「class_name + 出现次序」与 traced 的
    同层同类模块配对，避免 agent 漏填 module_path 就整棵树补不上形状。
    补参（3.1）：库型模型的层尺寸由运行期构造参数决定，agent 给出的值为 null 或 `'sizes[0]'`
    这类表达式；trace 从实例化模型上读出真实构造参数，按 module_path 回填**仅当当前值缺失、
    不可用、或与模型实际值不等价**——与构造默认值等价的键不回填，避免平白改动 `params`
    使既有验证无故 stale（trace 另给出 `params_delta`＝其中与默认值不等价的部分）。
    另把「模型实际暴露的参数」原样记入节点的 `params_model`（不参与 ir_hash）：模块结构签名
    取它，从而与 agent 是否写出可选参数（ReLU 的 inplace 等）解耦（R27）。
    input_spec.shape 缺失/含非法维度时，用本次 trace 实际使用的输入形状（`input_shape`）补齐。
    """
    if "shapes" in trace_output and isinstance(trace_output.get("shapes"), dict):
        shapes: dict = trace_output["shapes"]
        order: list[dict] = trace_output.get("order") or []
    else:
        shapes, order = trace_output, []
    by_class: dict[str, list[str]] = {}
    for item in order:
        cls = (item.get("class_name") or "").split(".")[-1]
        by_class.setdefault(cls, []).append(item.get("path", ""))
    used: dict[str, int] = {}
    # 已被「自带 module_path 的节点」占用的 traced 路径，兜底配对时不得再用
    claimed = {n.get("module_path") for n in ir["nodes"] if n.get("module_path")}

    filled = 0
    for n in ir["nodes"]:
        mp = n.get("module_path")
        if mp is None:
            if n["id"] == ir["root_id"]:
                mp = ""
            elif order:
                # 无 module_path 的非根节点：按类名+次序兜底（跳过已被占用的路径）
                cls = normalize_class_name(n.get("class_name") or "").split(".")[-1]
                candidates = [p for p in (by_class.get(cls) or []) if p not in claimed]
                idx = used.get(cls, 0)
                if idx >= len(candidates):
                    continue
                used[cls] = idx + 1
                mp = candidates[idx]
                n["module_path"] = mp
                claimed.add(mp)
            else:
                continue
        rec = shapes.get(mp)
        if not rec:
            continue
        if n.get("input_shape") is None and rec.get("input_shape"):
            n["input_shape"] = rec["input_shape"]
            filled += 1
        if n.get("output_shape") is None and rec.get("output_shape"):
            n["output_shape"] = rec["output_shape"]
            filled += 1

    # 层构造参数回填（3.1 补参）：以模型实际值为准的只有两类键——① agent 没写、且该键与
    # 构造默认值不等价（trace 的 `params_delta`；默认值等价的键补了也等于没补，反而平白改动
    # `params` 让既有验证无故 stale）；② agent 写了但与模型实际值不等价（含 `'sizes[0]'`
    # 这类写进生成代码会 NameError 的表达式）。等价字面量（3 vs (3,3)）保持原样不动。
    # 兼容旧 trace 产物（无 params_delta 时按全量 params 处理）。
    trace_params = trace_output.get("params") or {}
    trace_delta = trace_output.get("params_delta")
    if not isinstance(trace_delta, dict):
        trace_delta = trace_params
    for n in ir["nodes"]:
        params = n.get("params")
        if not isinstance(params, dict):
            continue
        src = trace_params.get(n.get("module_path"))
        if not src:
            continue
        # 记录「模型实际暴露的参数」，供结构签名使用（`params` 仍保持「可写进生成代码」的语义）
        n["params_model"] = dict(src)
        delta = trace_delta.get(n.get("module_path")) or {}
        for k, v in src.items():
            cur = params.get(k)
            if cur is None:
                if k in delta:
                    params[k] = v
                    filled += 1
            elif not _same_param(cur, v):
                params[k] = v
                filled += 1

    # 输入规格回填：shape 缺失/含非法维度时，用本次 trace 实际使用的输入形状
    used_shape = trace_output.get("input_shape")
    if used_shape:
        spec = dict(ir.get("input_spec") or {})
        cur = spec.get("shape") or []
        if not cur or not all(isinstance(d, int) and d > 0 for d in cur):
            spec["shape"] = list(used_shape)
            spec.setdefault("dtype", "float32")
            ir["input_spec"] = spec
            filled += 1
    return filled


def _infer_missing_shapes(ir: dict) -> int:
    """兜底回填：op 节点不是 nn.Module，forward hook 永远抓不到形状。

    按其入边上游的输出形状回填（多输入 op 取声明序第一条：add/残差正确，cat 沿通道拼接为近似）。
    trace 之后调用，只为仍为 null 的项兜底，不覆盖已知值。
    """
    filled = 0
    node_map = nodes_by_id(ir)
    for n in ir["nodes"]:
        if n.get("input_shape") and n.get("output_shape"):
            continue
        ins = [e for e in ir["edges"] if e["to"] == n["id"]]
        if not ins:
            continue
        src = node_map.get(ins[0]["from"]) or {}
        shape = src.get("output_shape") or src.get("input_shape")
        if not shape:
            continue
        if n.get("input_shape") is None:
            n["input_shape"] = shape
            filled += 1
        if n.get("output_shape") is None:
            n["output_shape"] = shape
            filled += 1
    return filled


def _knowledge_hint(knowledge: dict) -> str:
    """把「任务前带入的知识」（需求六.1、数据设计三.3）渲染为 prompt 附注，限长避免挤占上下文。"""
    lines: list[str] = []
    for label, items in (("参数建议", knowledge.get("param_advice") or []),
                         ("已知依赖冲突", knowledge.get("dependency_conflict") or [])):
        for it in items[:5]:
            content = (it.get("content") or "").strip().replace("\n", " ")
            lines.append(f"- [{label}] {it.get('title') or ''}：{content[:200]}")
    if not lines:
        return ""
    return ("\n\n【知识库带入（历史已确认结论，供参考；与本次代码冲突时以代码为准）】\n"
            + "\n".join(lines[:12])[:2000])


def _warn_canonical_unavailable(task_id: str, reason: str) -> None:
    """仅补记模型参数失败时的进度警示（run_record 由调用方按真实原因记录）。"""
    task_manager.update_progress(task_id, {
        "skipped": True,
        "warning": f"{reason}；未能补记模型实际参数，结构签名暂用 agent 参数（module_id 可能随可选参数写法变化）",
    })


def _record_canonical_unavailable(project_id: str, task_id: str, reason: str) -> None:
    """形状/补参/输入规格都已齐备，仅「模型实际参数」补记失败：如实留痕但不阻断任务（R27）。

    此时 IR 仍可用（结构签名回落到 agent 参数），但 module_id 可能对 agent 的可选参数写法敏感，
    因此记一条 failed 的 run_record 与进度警示，便于事后发现「同一模型两个 module_id」的根因。
    """
    now = _now()
    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "decompose_trace",
        "command": "trace canonical params skipped", "status": "failed",
        "error": f"{reason}；未能补记模型实际参数（结构签名暂用 agent 参数，module_id 可能不稳定）",
        "started_at": now, "finished_at": now,
    })
    _warn_canonical_unavailable(task_id, reason)


async def _run_trace(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = _require_original(project_id)
    ws = _ws(project)
    source = ws / "source"
    ir = _read_ir(project)
    if ir is None:
        raise RuntimeError("尚未拆解：请先 POST /api/projects/{id}/decompose")
    need_shape = any(n.get("input_shape") is None or n.get("output_shape") is None for n in ir["nodes"])
    need_params = any(
        n.get("kind") == "leaf" and any(v is None for v in (n.get("params") or {}).values())
        for n in ir["nodes"]
    )
    spec_shape = (ir.get("input_spec") or {}).get("shape") or []
    need_input = not spec_shape or not all(isinstance(d, int) and d > 0 for d in spec_shape)
    # 结构签名所需的「模型实际暴露的参数」是否已具备（R27）：缺失时即使形状/参数都齐也要跑一次
    # trace，否则 module_id 会随 agent 是否写出可选参数（inplace 等）而变化。
    need_canonical = any(
        n.get("kind") == "leaf" and not isinstance(n.get("params_model"), dict)
        for n in ir["nodes"]
    )
    if not (need_shape or need_params or need_input or need_canonical):
        # 3.2「读 IR 统计形状缺失（无缺失直接跳过）」：形状、补参、输入规格与模型参数均已齐备，
        # 省掉一次项目环境跑模型
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose_trace",
            "command": "trace skipped (shapes complete)", "status": "success",
            "metrics": {"skipped": True, "reason": "IR 形状已完整"},
            "started_at": _now(), "finished_at": _now(),
        })
        task_manager.update_progress(task_id, {"skipped": True, "reason": "IR 形状已完整，无需追踪"})
        return
    # 仅为了补记模型参数才需要追踪时，不因环境/模型不可用而中断（IR 本身已完整）
    only_canonical = not (need_shape or need_params or need_input)
    python = analysis_service._project_python(ws)
    if python is None:
        if not only_canonical:
            raise RuntimeError("项目环境未就绪：未找到独立环境解释器，请先完成环境创建（模块一 env）")
        _record_canonical_unavailable(project_id, task_id, "项目环境未就绪（未找到解释器）")
        return

    run_dir = ws / "runs" / "decompose" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "shapes.json"
    ir_path = ws / "reports" / "ir.json"
    started = _now()
    command = f"{python} {TRACE_SCRIPT} <source> {ir_path.name} {out_json.name}"
    try:
        rc, log = await proc_util.run_command(
            [python, str(TRACE_SCRIPT), str(source), str(ir_path), str(out_json)],
            cwd=str(run_dir), timeout=DECOMPOSE_TRACE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose_trace",
            "command": command, "status": "failed",
            "error": f"形状追踪超时（>{DECOMPOSE_TRACE_TIMEOUT_S}s）",
            "started_at": started, "finished_at": _now(),
        })
        if only_canonical:
            _warn_canonical_unavailable(task_id, "形状追踪超时")
            return
        raise RuntimeError("形状追踪超时")
    if rc != 0:
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose_trace",
            "command": command, "status": "failed", "error": log[-2000:],
            "log_path": None, "started_at": started, "finished_at": _now(),
        })
        if only_canonical:
            _warn_canonical_unavailable(task_id, "形状追踪执行失败（模型/环境不可用）")
            return
        raise RuntimeError("形状追踪失败: " + log[-1500:])

    shapes = json.loads(out_json.read_text(encoding="utf-8"))
    filled = _merge_shapes(ir, shapes)
    inferred = _infer_missing_shapes(ir)  # op 节点兜底（hook 抓不到）
    _write_ir(project, ir)
    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "decompose_trace",
        "command": command, "status": "success",
        "metrics": {"captured_paths": len(shapes.get("shapes", shapes)) if isinstance(shapes, dict) else 0,
                    "filled": filled, "inferred": inferred},
        "artifact_path": str(out_json),
        "started_at": started, "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {
        "captured_paths": len(shapes.get("shapes", shapes)) if isinstance(shapes, dict) else 0,
        "filled": filled, "inferred": inferred,
    })


# --------------------------- 6.4 两步验证 ---------------------------

def _seeds() -> list[int]:
    return [int(s) for s in DECOMPOSE_NUM_SEEDS.split(",") if s.strip()]


async def _run_verify(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = _require_original(project_id)
    ws = _ws(project)
    source = ws / "source"
    ir = _read_ir(project)
    if ir is None:
        raise RuntimeError("尚未拆解：请先 POST /api/projects/{id}/decompose")
    python = analysis_service._project_python(ws)
    if python is None:
        raise RuntimeError("项目环境未就绪：未找到独立环境解释器，请先完成环境创建（模块一 env）")

    # 再生成失败（IR 不完整）→ 任务失败，报缺失项
    try:
        code = ir_codegen.generate(ir)
    except IrIncompleteError as e:
        raise RuntimeError(f"IR 不完整，无法再生成代码: {e}")

    run_dir = ws / "runs" / "decompose" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    regen_path = run_dir / "regenerated.py"
    regen_path.write_text(code, encoding="utf-8")
    out_json = run_dir / "verification_result.json"
    ir_path = ws / "reports" / "ir.json"
    started = _now()
    command = (
        f"{python} {VERIFY_SCRIPT} <source> {ir_path.name} regenerated.py {out_json.name} "
        f"<seeds> <rtol> <atol>"
    )
    seeds = _seeds()
    try:
        rc, log = await proc_util.run_command(
            [python, str(VERIFY_SCRIPT), str(source), str(ir_path), str(regen_path),
             str(out_json), DECOMPOSE_NUM_SEEDS, str(DECOMPOSE_NUM_RTOL), str(DECOMPOSE_NUM_ATOL)],
            cwd=str(run_dir), timeout=DECOMPOSE_VERIFY_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose_verify",
            "command": command, "status": "failed",
            "error": f"验证运行超时（>{DECOMPOSE_VERIFY_TIMEOUT_S}s）",
            "started_at": started, "finished_at": _now(),
        })
        raise RuntimeError("验证运行超时")
    if rc != 0:
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose_verify",
            "command": command, "status": "failed", "error": log[-2000:],
            "started_at": started, "finished_at": _now(),
        })
        raise RuntimeError("验证脚本异常: " + log[-1500:])

    result = json.loads(out_json.read_text(encoding="utf-8"))
    verification = {
        "schema_version": "1.0",
        "verified_at": _now(),
        "project_id": project_id,
        "ir_hash": ir_hash(ir),
        "seeds": seeds,
        "tolerance": {"rtol": DECOMPOSE_NUM_RTOL, "atol": DECOMPOSE_NUM_ATOL},
        **result,
    }
    verification_path = ws / "reports" / "verification.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = verification["overall"]
    # 比对不过：任务 success、run_record failed（可检索供 agent 改进）
    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "decompose_verify",
        "command": command,
        "params": {"seeds": seeds, "rtol": DECOMPOSE_NUM_RTOL, "atol": DECOMPOSE_NUM_ATOL},
        "status": "success" if overall == "passed" else "failed",
        "error": verification.get("failure_reason"),
        "metrics": {
            "overall": overall,
            "max_rel_err": max((r["max_rel_err"] for r in verification["numeric"]["per_seed"]), default=None),
            "max_abs_err": max((r["max_abs_err"] for r in verification["numeric"]["per_seed"]), default=None),
        },
        "artifact_path": str(verification_path),
        "started_at": started, "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {
        "overall": overall,
        "structure_passed": verification["structure"]["passed"],
        "numeric_passed": verification["numeric"]["passed"],
        "failure_reason": verification.get("failure_reason"),
    })


# --------------------------- 6.5 模块入库 ---------------------------

def _module_signature(ir: dict) -> str:
    """结构签名（实施约定）：入口类+input_spec+节点(类名,排序参数)+边(类对)，不含 id/位置/形状。

    节点清单按签名的 JSON 文本排序：同一结构无论 agent 的节点声明序如何，module_id 都一样。
    参数取「agent 的 params」∪「模型实际暴露的 params_model」（后者优先，由 trace 回填）：
    agent 是否写出可选参数（ReLU 的 inplace、BatchNorm 的 affine 等）不再影响 module_id（R27）；
    仅在从未跑过 trace 时才回落到纯 agent 参数。
    """
    node_map = nodes_by_id(ir)

    def _node_params(n) -> dict:
        params = dict(n.get("params") or {})
        model_params = n.get("params_model")
        if isinstance(model_params, dict):
            params.update(model_params)  # 模型实际值优先
        # op 节点是表达式、模型不暴露参数：按再生成引擎的同一份占位符默认值补齐，
        # 使「写全默认值」与「省略默认值」两种 agent 写法得到同一签名（如 flatten 的 end_dim）
        if n.get("kind") == "op":
            for k in ir_codegen.op_param_names(n):
                if params.get(k) is None:
                    params[k] = ir_codegen.OP_PARAM_DEFAULTS[k]
        return dict(sorted(params.items(), key=lambda kv: str(kv[0])))

    def _node_sig(n):
        return [n.get("kind"), ir_schema.normalize_class_name(n.get("class_name") or ""), _node_params(n)]

    def _edge_sig(e):
        return [
            ir_schema.normalize_class_name(node_map[e["from"]].get("class_name") or ""),
            ir_schema.normalize_class_name(node_map[e["to"]].get("class_name") or ""),
        ]

    def _sort_key(sig) -> str:
        return json.dumps(sig, ensure_ascii=False, sort_keys=True, default=str)

    parts = [
        ir["entry_class"],
        ir.get("input_spec") or {},
        sorted((_node_sig(n) for n in ir["nodes"]), key=_sort_key),
        sorted(_edge_sig(e) for e in ir["edges"]),
    ]
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _params_schema(params: dict) -> dict:
    """根节点 params → {参数名: {type, default}}（3.5 字段表口径）。"""
    def _type_of(v) -> str:
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "str"
        if isinstance(v, list):
            return "list"
        if isinstance(v, dict):
            return "dict"
        return "null" if v is None else type(v).__name__

    return {k: {"type": _type_of(v), "default": v} for k, v in (params or {}).items()}


def _param_count_of(verification: dict) -> Optional[int]:
    """verification.structure.param_count（新口径为 {original, regenerated}，兼容旧标量）。"""
    pc = ((verification or {}).get("structure") or {}).get("param_count")
    if isinstance(pc, dict):
        return pc.get("original")
    return pc if isinstance(pc, int) else None


async def _run_ingest(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = _require_original(project_id)
    ws = _ws(project)
    ir = _read_ir(project)
    if ir is None:
        raise RuntimeError("尚未拆解：请先 POST /api/projects/{id}/decompose")
    verification = get_verification(project_id)
    if verification is None:
        raise RuntimeError("尚未验证：请先 POST /api/projects/{id}/decompose/verify")
    if verification.get("overall") != "passed":
        raise RuntimeError("验证未通过，不能入库；请调整 IR 后重新验证")
    if verification.get("ir_hash") != ir_hash(ir):
        raise RuntimeError("IR 已修改（调参）但未重新验证，验证结果已过期；请重新执行 verify")

    started = _now()
    code = ir_codegen.generate(ir)  # 验证已通过，此处不应再缺项；缺则任务失败
    module_id = "mod_" + _module_signature(ir)
    root = nodes_by_id(ir)[ir["root_id"]]
    param_count = _param_count_of(verification)
    description = f"从项目 {project_id} 拆解的 {ir['entry_class']}（{ir.get('task_type')}），经两步验证通过"

    # 结构化项目（画布可编辑，graph.json 落工作区根）；失败要清掉半成品（补偿见 except）
    structured_id = project_manager.create_project(
        "structured", source="decompose", name=ir["entry_class"], parent_project_id=project_id
    )
    package_dir: Optional[Path] = None
    tmp_dir: Optional[Path] = None
    module_version = ""
    recorded = False
    try:
        graph = ir_graphir.ir_to_graphir(ir)
        graph_path = _ws(project_manager.get_project(structured_id)) / "graph.json"
        graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
        project_manager.update_status(structured_id, "ready")
        # 4d-1：git init + 初始提交（拆解图快照）；失败不连坐入库（保存画布时懒初始化自愈）
        try:
            version_service.commit_graph(structured_id, graph)
        except Exception:  # noqa: BLE001
            logger.exception("结构化项目 git 版本初始化失败（保存画布时会自动重试）：%s", structured_id)

        tmp_dir = MODULES_DIR / f".tmp_{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # 目录布局（数据设计七.1）：module.py（再生成代码）+ module.json + ir.json（IR 快照）+ assets/（预留权重）
        (tmp_dir / "module.py").write_text(code, encoding="utf-8")
        (tmp_dir / "ir.json").write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp_dir / "assets").mkdir(exist_ok=True)

        # 版本解析与写入：并发下版本可能被抢占（复合主键冲突）→ 递增重试
        for attempt in range(3):
            module_version = knowledge_service.next_module_version(module_id)
            package_dir = MODULES_DIR / module_id / module_version
            module_json = {
                "module_id": module_id,
                "module_version": module_version,
                "name": ir["entry_class"],
                "description": description,
                "source_project_id": project_id,
                "source_paper_id": None,  # 实施约定：本阶段 project 表无论文关联列，置 null
                "task_type": ir.get("task_type"),
                "input_spec": ir.get("input_spec") or {},
                "output_spec": {"shape": root.get("output_shape")} if root.get("output_shape") else None,
                "params_schema": _params_schema(root.get("params") or {}),
                "tags": [t for t in (ir.get("task_type"), project.get("source"), "decompose") if t],
                "verification": verification,
                "index_summary": (
                    f"{ir.get('task_type')} 模块 | 参数量 {param_count} | 验证通过 | 来源项目 {project_id}"
                ),
                "index_keywords": " ".join(
                    str(x) for x in (ir["entry_class"], ir.get("task_type"), "decompose") if x
                ),
                "saved_module_compat": {
                    "id": f"{module_id}:{module_version}",
                    "name": ir["entry_class"],
                    "version": module_version,
                    "description": description,
                    "handles": {"inputs": ["in"], "outputs": ["out"]},
                    "graph_ref": structured_id,
                    "graph": graph,  # 兼容基底 SavedModule.graph 必填字段（与 graph_ref 同指结构化项目）
                    "createdAt": _now(),
                    "updatedAt": _now(),
                },
                # path = 模块包目录（module.py / module.json / assets 的所在目录）
                "path": str(package_dir),
            }
            (tmp_dir / "module.json").write_text(
                json.dumps(module_json, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                knowledge_service.record_module(module_json)
                recorded = True
                break
            except sqlite3.IntegrityError:
                if attempt == 2:
                    raise RuntimeError(f"模块 {module_id} 的版本号连续冲突，入库失败（请重试）")
                continue

        # 临时目录 → 正式目录（rename 不创建父目录，先建 module_id 一级）
        package_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.rename(package_dir)
    except Exception:
        if recorded:
            knowledge_service.delete_module(module_id, module_version)  # DB 行已写但包未落盘 → 回滚
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        project_manager.delete_project(structured_id)  # 不残留半成品结构化项目
        raise

    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "module_ingest",
        "command": "module package → MODULES_DIR",
        "status": "success",
        "metrics": {
            "module_id": module_id,
            "module_version": module_version,
            "structured_project_id": structured_id,
        },
        "artifact_path": str(package_dir / "module.json"),
        "started_at": started, "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {
        "module_id": module_id,
        "module_version": module_version,
        "structured_project_id": structured_id,
        "path": str(package_dir),
    })


def register() -> None:
    task_manager.register_handler(TASK_DECOMPOSE, _run_decompose)
    task_manager.register_handler(TASK_TRACE, _run_trace)
    task_manager.register_handler(TASK_VERIFY, _run_verify)
    task_manager.register_handler(TASK_INGEST, _run_ingest)
