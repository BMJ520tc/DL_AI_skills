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
FIDELITY_SCRIPT = PROJECT_ROOT / "scripts" / "ir_fidelity_probe.py"

# 数值比对阈值（实施约定；仿 REPRO_DEVIATION_* 环境变量覆盖先例）
DECOMPOSE_NUM_RTOL = float(os.getenv("DECOMPOSE_NUM_RTOL", "1e-5"))
DECOMPOSE_NUM_ATOL = float(os.getenv("DECOMPOSE_NUM_ATOL", "1e-6"))
DECOMPOSE_NUM_SEEDS = os.getenv("DECOMPOSE_NUM_SEEDS", "42,1337,2024")
DECOMPOSE_AGENT_TIMEOUT_S = 1800
# 「跑到稳定为止」：重试上限给得足够高，让**时间预算**（DECOMPOSE_AGENT_BUDGET_S）成为真正的
# 限制；每次失败都把「差多少 + 真实模型的带参层清单」回喂，agent 带着反馈修正。
DECOMPOSE_AGENT_RETRIES = int(os.getenv("DECOMPOSE_AGENT_RETRIES", "12"))
DECOMPOSE_AGENT_BUDGET_S = int(os.getenv("DECOMPOSE_AGENT_BUDGET_S", "3600"))  # 重试总预算
DECOMPOSE_TRACE_TIMEOUT_S = 600
DECOMPOSE_VERIFY_TIMEOUT_S = 1800
DECOMPOSE_FIDELITY_TIMEOUT_S = int(os.getenv("DECOMPOSE_FIDELITY_TIMEOUT_S", "600"))
# 节点数上限：一次要模型吐出的 IR 越长越不稳（漏字段/漏边/整份不产出）。把重复与标准结构
# 用 code_hint 折叠后，真实模型通常 20~40 个节点就够；超上限即判失败并带原因重试。
DECOMPOSE_MAX_NODES = int(os.getenv("DECOMPOSE_MAX_NODES", "60"))
# **分步生成**（默认**关**，置 1 启用）：先模块骨架、再逐模块并行展开、最后合并。
# 实测能显著改善稳定性（第 1 次就产出结构合法的 IR），但**每次拆解的会话数 × 模块数**，
# token 消耗高得多；未经验证优于单次路径前，默认仍走单次生成。
DECOMPOSE_STEPWISE = os.getenv("DECOMPOSE_STEPWISE", "0") != "0"
# 重试时是否续接上一次 agent 会话。默认**不续接**（每轮全新会话）：实测续接的长会话到后期
# 常直接「不产出 IR」（12 次里 8 次），换新会话给干净上下文；置 1 可切回续接（省一轮读源码）。
DECOMPOSE_RETRY_RESUME = os.getenv("DECOMPOSE_RETRY_RESUME", "0") != "0"
# 保真度自检开关：默认开。要求项目环境 + entry_args 才做（起不来就跳过，不误判）
DECOMPOSE_FIDELITY_CHECK = os.getenv("DECOMPOSE_FIDELITY_CHECK", "1") != "0"


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


def _clean_extra_inputs(extra: list) -> list:
    """校验并规范化 `input_spec.extra`（多输入模型的额外入参）：`[{shape, dtype?}, …]`。"""
    if not isinstance(extra, list):
        raise ValueError("extra 必须是数组（每个元素形如 {shape, dtype}）")
    cleaned: list[dict] = []
    for i, e in enumerate(extra):
        if not isinstance(e, dict):
            raise ValueError(f"extra[{i}] 必须是对象，如 {{\"shape\":[1,1200],\"dtype\":\"float32\"}}")
        eshape = e.get("shape")
        if not isinstance(eshape, list) or not eshape or not all(isinstance(d, int) and d > 0 for d in eshape):
            raise ValueError(f"extra[{i}].shape 必须是非空正整数数组")
        item: dict = {"shape": list(eshape)}
        if e.get("dtype"):
            item["dtype"] = str(e["dtype"])
        cleaned.append(item)
    return cleaned


def _clean_inputs(ir: dict, inputs: list) -> list:
    """校验 `input_spec.inputs`：按**调用顺序**列出吃外部输入的节点 id（root 的 forward 形参）。"""
    if not isinstance(inputs, list):
        raise ValueError("inputs 必须是数组（按调用顺序列出吃外部输入的节点 id）")
    ids = {n["id"] for n in ir.get("nodes") or []}
    cleaned: list[str] = []
    for i, v in enumerate(inputs):
        if not isinstance(v, str) or not v:
            raise ValueError(f"inputs[{i}] 必须是节点 id 字符串")
        if v not in ids:
            raise ValueError(f"inputs[{i}] 的节点 id 不存在于 IR：{v}")
        cleaned.append(v)
    return cleaned


def update_input_spec(project_id: str, shape: list, dtype: Optional[str] = None,
                      extra: Optional[list] = None,
                      forward_kwargs: Optional[dict] = None,
                      inputs: Optional[list] = None) -> dict:
    """修正入口输入规格（6.1/6.2）：agent 对「尺寸由运行期构造参数决定」的模型给不出具体维度时，
    由用户/工具补上；写回后 ir_hash 变化 → 旧验证变 stale（与调参同口径）。

    shape 必须是非空正整数数组（不允许 null 维度：无法据此构造输入）。
    `extra` 是**多输入模型**的补充入参（如 scGPT 的 forward(src, values, src_key_padding_mask)）：
    按序追加在主输入之后；传 `[]` 表示清除，传 None 表示不改。
    `forward_kwargs` 是传给 **forward 的关键字参数**（如 scGPT 的 `CLS/MVC/ECS` 开关）：
    默认关着的分支不执行 → hook 抓不到形状；打开开关即可让那些节点被捕获。传 {} 清除。
    """
    project = _require_original(project_id)
    ir = _read_ir(project)
    if ir is None:
        raise LookupError("ir not found：请先 POST /api/projects/{id}/decompose")
    if not isinstance(shape, list) or not shape or not all(isinstance(d, int) and d > 0 for d in shape):
        raise ValueError("shape 必须是非空正整数数组（如 [1, 64]），不接受 null/非正数维度")
    spec = dict(ir.get("input_spec") or {})
    spec["shape"] = shape
    spec["user_edited"] = True       # 标记：重拆解时这版 shape/dtype 优先于 agent 的新值
    if dtype:
        spec["dtype"] = dtype
    if extra is not None:
        cleaned = _clean_extra_inputs(extra)
        if cleaned:
            spec["extra"] = cleaned
        else:
            spec.pop("extra", None)          # 传空数组 = 清除
    if forward_kwargs is not None:
        if not isinstance(forward_kwargs, dict):
            raise ValueError("forward_kwargs 必须是 JSON 对象（键为 forward 的形参名）")
        bad = [k for k in forward_kwargs if not isinstance(k, str) or not k.isidentifier()]
        if bad:
            raise ValueError(f"forward_kwargs 的键必须是合法标识符：{bad}")
        if forward_kwargs:
            spec["forward_kwargs"] = forward_kwargs
        else:
            spec.pop("forward_kwargs", None)  # 传空对象 = 清除
    if inputs is not None:
        cleaned_inputs = _clean_inputs(ir, inputs)
        if cleaned_inputs:
            spec["inputs"] = cleaned_inputs
        else:
            spec.pop("inputs", None)          # 传空数组 = 清除（退回单输入）
    ir["input_spec"] = spec
    _write_ir(project, ir)
    return spec


def update_entry_args(project_id: str, entry_args: dict) -> dict:
    """修正入口类构造参数（agent 给不出、模型需要外部数据时的补参通道）。

    形状追踪/两步验证都要实例化入口类；像 scGPT 的 `TransformerModel(ntoken, d_model, nhead,
    d_hid, nlayers, vocab=…)` 参数来自运行期配置与数据，脚本猜不到（`_model_loader` 的固定
    猜测列表必然失败）——由用户在这里给出最小可构造的 args（如 `{"ntoken":1000,...,
    "vocab":{"<pad>":0}}`），trace/verify 会优先用它实例化。写回后 ir_hash 变化 → 旧验证 stale。
    """
    project = _require_original(project_id)
    ir = _read_ir(project)
    if ir is None:
        raise LookupError("ir not found：请先 POST /api/projects/{id}/decompose")
    if not isinstance(entry_args, dict):
        raise ValueError("entry_args 必须是一个 JSON 对象（键为构造参数名）")
    for k in entry_args:
        if not isinstance(k, str) or not k.isidentifier():
            raise ValueError(f"构造参数名必须是合法标识符：{k!r}")
    if entry_args:
        ir["entry_args"] = entry_args
    else:
        ir.pop("entry_args", None)          # 传空对象 = 清除
    _write_ir(project, ir)
    return entry_args


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

# few-shot：一份**完整且合法**的最小 IR（module + leaf + op + 边 + input_spec/root_id 齐备）。
# 只给「片段」时模型常写出漏边/缺必填参数/op 无入边的结构，触发 IR 校验或再生成自检失败 →
# 整轮重试（真库实测重试是主要时间乘数）。给一份可照抄的完整样例以降低失败率。
# 该样例在用例中被 validate_ir + ir_codegen.generate 校验，防止示例本身失效。
_IR_EXAMPLE: dict = {
    "source_file": "model.py",
    "entry_class": "Net",
    "task_type": "classification",
    "input_spec": {"shape": [1, 3, 32, 32], "dtype": "float32"},
    "root_id": "net",
    "nodes": [
        {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None, "module_path": ""},
        {"id": "conv1", "kind": "leaf", "class_name": "nn.Conv2d", "parent_id": "net",
         "module_path": "conv1",
         "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1},
         "input_shape": [1, 3, 32, 32], "output_shape": [1, 16, 32, 32]},
        {"id": "relu1", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net",
         "module_path": "relu1"},
        {"id": "flatten", "kind": "op", "class_name": "flatten", "parent_id": "net"},
        {"id": "fc", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "net",
         "module_path": "fc", "params": {"in_features": 16, "out_features": 10}},
    ],
    "edges": [
        {"from": "conv1", "to": "relu1"},
        {"from": "relu1", "to": "flatten"},
        {"from": "flatten", "to": "fc"},
    ],
}


# 重拆解时要带过去的**用户补参**字段：这些 agent 从不产出，只可能由用户经
# PUT /ir/entry_args 与 PUT /ir/input_spec 补上；不带走的话，用户补完再点一次「拆解」
# 就静默清空（2026-10-05 scGPT 实测：重拆解后 entry_args 丢失 → 补形状直接失败）。
_CARRY_SPEC_KEYS = ("extra", "forward_kwargs", "inputs")


def _carry_over_user_specs(prev_ir: Optional[dict], ir: dict) -> None:
    """把上一版 IR 里的**用户补参**带进新 IR（重拆解不该让用户白填一次）。"""
    if not prev_ir:
        return
    if prev_ir.get("entry_args") and not ir.get("entry_args"):
        ir["entry_args"] = prev_ir["entry_args"]
    prev_spec = prev_ir.get("input_spec") or {}
    spec = dict(ir.get("input_spec") or {})
    for key in _CARRY_SPEC_KEYS:
        if prev_spec.get(key) and not spec.get(key):
            spec[key] = prev_spec[key]
    if prev_spec.get("user_edited"):
        # 用户改过的 shape/dtype 优先于 agent 的新值（agent 每次产出的维度可能不同）
        if prev_spec.get("shape"):
            spec["shape"] = prev_spec["shape"]
        if prev_spec.get("dtype"):
            spec["dtype"] = prev_spec["dtype"]
        spec["user_edited"] = True
    if spec:
        ir["input_spec"] = spec


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
        "   **入口有多个输入时**（如 `forward(src, values, src_key_padding_mask)`）再加 `inputs`："
        "**按调用顺序列出 root 的直接子节点里接收外部输入的那些节点的 id**"
        '（是**节点 id**，如 `"inputs":["encoder","value_encoder"]`；'
        "**不要**写 forward 的参数名或描述文字）；单输入模型省略此项。\n"
        "2. 每个 nn.Module 子类实例一个节点。入口模型实例为根节点（kind=module，parent_id=null，module_path=\"\"）。\n"
        "3. 自定义 nn.Module 子类（如 BasicBlock）必须作为 kind=module 节点，其内部子模块作为子节点"
        "（parent_id 指向它），逐层展开到叶子；class_name 用源码类名。\n"
        "4. 叶子层为 torch.nn 内置层：kind=leaf，class_name 用 nn.X 形式（nn.Conv2d、nn.BatchNorm2d、"
        "nn.ReLU、nn.Dropout 等）。params 为该层构造参数（PyTorch 构造器关键字，如 "
        '{"in_channels":64,"out_channels":128,"kernel_size":3}），必须从源码读出真实值，不要臆造；'
        "**必填构造参数必须给全**（缺了会被结构校验直接拒绝）：nn.Linear→in_features/out_features、"
        "nn.Embedding→num_embeddings/embedding_dim、nn.Conv*d→in_channels/out_channels/kernel_size、"
        "nn.LayerNorm→normalized_shape、nn.BatchNorm*d→num_features、池化→kernel_size；"
        "这些值源码里一定有（构造调用或默认配置），从源码读准；**确实读不出**的次要参数可留空并置 "
        "uncertain=true（它代替不了必填参数）。\n"
        "   叶子白名单：nn.Linear/Conv1d~3d/ConvTranspose*/BatchNorm*/LayerNorm/GroupNorm/Embedding/"
        "MaxPool*/AvgPool*/Adaptive*Pool*/Upsample/ReLU 系/Sigmoid/Tanh/Softmax/Dropout*/Flatten/Identity。\n"
        "   **叶子的调用只接受一个输入张量**；因此**内部含多输入子层**的复合模块（如 "
        "nn.MultiheadAttention、nn.TransformerEncoder/EncoderLayer、nn.CosineSimilarity）**不要展开成**"
        "显式叶子，也不要硬塞进白名单——用**一个节点 + code_hint 给出完整构造表达式整体折叠**"
        "（其内部子模块视为已覆盖，参数量自然对齐）；展开反而表达不了、且前向调用会参数不符。\n"
        "   若某个子模块的 **forward 需要多个输入**（如 scGPT 的 MVCDecoder(cell_emb, gene_embs)）："
        "**直接用 module 节点 + 多条入边**——入边顺序就是它 forward 的形参顺序，父模块会按序传参"
        "（这是 module 节点与叶子/容器的区别：叶子/容器只能一条入边）。\n"
        "   **白名单外的 nn.* 类**（如 nn.TransformerEncoder、nn.MultiheadAttention、nn.TransformerEncoderLayer）："
        "优先继续展开为子模块；若整体当一层用，**必须给 code_hint 完整构造表达式、不要用 {params} 占位符**"
        "（如 \"code_hint\": \"nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6)\"），"
        "否则该节点会被拒绝。\n"
        "5. nn.Sequential 为 kind=container、class_name=\"nn.Sequential\"，其成员按顺序作为子节点。"
        "nn.ModuleList 展开为父模块的直接子节点（不建 container）。\n"
        "6. forward 中的函数式操作建 kind=op 节点：class_name 用小写名（白名单: add/sub/mul/div/"
        "matmul/bmm/cat/relu/sigmoid/tanh/softmax/flatten/mean/max/min/sum/view/reshape/permute）；"
        "白名单外的操作填 code_hint 内联表达式模板，输入变量用 {inputs} 占位。op 是叶子（无子节点）；"
        "op 可有多条入边（如残差相加），非 op 节点最多 1 条入边。\n"
        "   **占位符规则**：多入边时 `{inputs}` 展开成「a, b, c」逗号串（适合 `torch.add({inputs})`、"
        "`f({inputs})` 这种整体传参）；要**单独取某一个操作数**必须写 `{inputs[N]}`"
        "（如 `{inputs[0]} / {inputs[1]}`、`dict(pred={inputs[0]}, aux={inputs[1]})`），"
        "**不要**写 `{inputs}[0]`——那会展开成「a, b, c[0]」而语法错误。\n"
        "7. 数据流用 edges 表达（from/to 为节点 id）。同层节点按 forward 执行顺序连边；残差/跳跃连接"
        "直接连到汇合 op（如 add）。Sequential 内部成员之间不连边（顺序由声明序决定），Sequential 整体"
        "与外部节点的边连在 container 节点上。\n"
        "   **每个 module 节点的直接子节点必须只有一个汇点（唯一输出）**；若 forward 有多个输出分支"
        "（多个汇总点），必须用一个 op 节点（如 cat/add）把它们汇合为该模块的唯一输出，再连到该模块之外。\n"
        "8. 节点 id 全局唯一，须为合法 Python 标识符（[A-Za-z_][A-Za-z0-9_]*，不能含点/连字符）；"
        "建议直接用源码中的属性名（如 conv1、bn1、layer1），重名时加前缀区分。\n"
        "9. module_path 填该实例在 named_modules() 中的完整路径（如 layer1.0.conv1、downsample.0），"
        "用于形状追踪与验证对齐；根节点 module_path=\"\"。\n"
        "10. input_shape/output_shape 能静态推出就填数组（元素为整数），推不出填 null。拿不准的节点置 uncertain=true。\n"
        "11. 输出必须是单层 JSON 对象（不要用 {\"ir\":...} 包裹），顶层键：source_file、entry_class、"
        "task_type、input_spec、root_id、nodes、edges。\n"
        "12. **提交前自查这 6 条高频被拒原因**（此前实测最容易踩，逐条确认再输出）：\n"
        "   a) **节点 id 不能是路径**——`encoder.embedding` 非法；id 用属性名（`enc_embedding`），"
        "路径只放 `module_path`；\n"
        "   b) **不要建空的 module 节点**——某类没有任何子模块时，要么继续展开出子节点，"
        "要么整体折叠为一个 leaf（+code_hint）；空 module 会被拒；\n"
        "   c) **每个 op 节点至少一条入边**（没有入边的表达式无意义）；\n"
        "   d) **leaf 的必填构造参数必须给全**（见第 4 条清单）；\n"
        "   e) **入边数**：叶子/容器**最多 1 条**；**module 可多条**（多输入，按入边序成形参）；"
        "op 可多条；\n"
        "   f) **code_hint 里不要写 `self.xxx`**（除非 xxx 是该节点的子节点 id 或它的 params 键）；"
        "取多入边里的某一个操作数写 `{inputs[0]}`，**不要**写 `{inputs}[0]`。\n"
        "13. **节点要「粗」不要「细」**（重要，直接决定成败）：单次输出有长度限制，节点越多越容易"
        "漏字段、漏边，甚至整份 JSON 都吐不出来。因此——\n"
        "   · **同一构造重复多次的结构**（如 `encoder.layers.0..11` 共 12 层）**只建 1 个节点**，"
        "用 code_hint 写构造表达式并按真实层数填参数，**不要**展开成 12 个节点；\n"
        "   · **标准复合层**（torch.nn 自带：nn.TransformerEncoder / TransformerEncoderLayer / "
        "MultiheadAttention / nn.Sequential 等）**整体折叠成 1 个节点**；\n"
        "   · **关键：折叠后的节点写成 kind=`leaf` + code_hint（给完整构造表达式），"
        "不要写成 kind=`module`**——module 的含义是「它的子模块要逐个展开列出」，"
        "而 torch.nn 自带层的内部子模块**不需要也不应该**列出来（写空 module 会被直接拒绝）；\n"
        "   · **code_hint 只能引用 `nn.*` / `torch.*` / `torch.nn.functional`**：再生成代码是"
        "**自包含**的（只 import torch，不 import 你项目里的模块）——写项目里的类名"
        "（如 `FlashTransformerEncoderLayer`）会在再生成时 `NameError`；\n"
        "   · 判据：**项目源码里自定义的 nn.Module 子类 → kind=module（展开子节点）；"
        "torch.nn 提供的标准层 → 折叠成 leaf/container（一条 code_hint 说完）**；\n"
        "   · 经验值：一个 12 层 Transformer 的 IR **20~40 个节点**足够；"
        f"**节点数上限 {DECOMPOSE_MAX_NODES}**，超了会被判失败打回。\n"
        "【完整示例（照此结构与粒度产出；数值须换成你读到的真实值）】\n"
        f"{json.dumps(_IR_EXAMPLE, ensure_ascii=False)}"
    )
    return "".join(parts)


# 通用规则：**每一步都会随 prompt 重发**，所以尽量短（每字都是 token×步数）。
_COMMON_RULES = (
    "【规则】"
    "① 节点 id 用属性名（合法标识符、不含点），路径只放 module_path；"
    "② 叶子只收一个输入张量，必填构造参数给全（Linear→in_features/out_features、"
    "Embedding→num_embeddings/embedding_dim、LayerNorm→normalized_shape）；"
    "③ torch.nn 自带的标准层（TransformerEncoder/Layer、MultiheadAttention、CosineSimilarity…）"
    "折叠成 leaf+code_hint（完整构造式，只用 nn.*/torch.*；不展开、不建空 module）；"
    "④ 项目自定义的 nn.Module 子类才是 module（其子节点由展开步给）；"
    "⑤ 函数式操作建 op（白名单外给 code_hint；第 N 个操作数写 {inputs[N]}），每个 op 至少一条入边；"
    "⑥ code_hint 不引用 self.xxx。\n"
)

_SKELETON_SCHEMA = {
    "type": "object",
    "properties": {
        "source_file": {"type": "string"}, "entry_class": {"type": "string"},
        "task_type": {"type": "string"}, "root_id": {"type": "string"},
        "input_spec": {"type": "object"},
        "modules": {"type": "array"},
    },
    "required": ["source_file", "entry_class", "root_id", "modules"],
}

_MODULE_SCHEMA = {
    "type": "object",
    "properties": {"nodes": {"type": "array"}, "edges": {"type": "array"}},
    "required": ["nodes", "edges"],
}


def _skeleton_prompt(hierarchy: list[dict], entry_class: str | None, hint: str,
                     feedback: str = "") -> str:
    """第 1 步：只要**模块骨架**（项目自定义 nn.Module 的层级 + 入口输入规格），输出很短。"""
    fb = f"\n【上一次尝试失败原因（务必避开）】{feedback}\n" if feedback else ""
    return (
        "任务（第 1 步 / 共 2 步）：阅读项目代码，先给出入口模型的**模块骨架**——"
        "只列**项目自定义的 nn.Module 子类实例**及其层级，**先不要列叶子层**。\n"
        "【红线】只允许读取文件（Read/Glob/Grep），禁止修改项目内任何文件。\n\n"
        "【静态结构报告】识别到的 nn.Module 子类：\n"
        f"{json.dumps(hierarchy, ensure_ascii=False)}\n\n"
        + (f"【指定入口类】entry_class={entry_class}，必须作为根模块。\n\n" if entry_class else "")
        + "【输出】单层 JSON：\n"
        '{"source_file":"入口类所在文件(相对路径)","entry_class":"入口类名",'
        '"task_type":"classification|regression|generation|embedding|other 之一",'
        '"input_spec":{"shape":[1,...],"dtype":"float32",'
        '"inputs":["按调用顺序列出根的直接子模块中接收外部输入的节点 id（单输入省略）"]},'
        '"root_id":"根模块 id","modules":[{"id","class_name","parent_id","module_path"}]}\n'
        "【要求】\n"
        "· modules 只含**项目自定义的 nn.Module 子类实例**（含入口模型本身）；"
        "torch.nn 自带的标准层**不要**列进来（下一步会折叠表达）；\n"
        "· parent_id 指向所属模块的 id（根为 null）；module_path 填 named_modules 路径"
        "（如 layer1.0 / encoder.embedding 所属模块），根为 \"\"；\n"
        "· 不要建空的模块——此步只列**类的层级**，别放叶子。\n" + _COMMON_RULES + hint + fb
    )


def _extract_class_source(source_dir: Path, class_name: Optional[str], cap: int = 6000) -> str:
    """在项目里定位 `class <class_name>` 并**把其源码切片塞进 prompt**。

    动机（省 token）：分步展开时，agent 为了看某个类会反复 Read/Grep——那些文件内容都是**输入
    token**，才是这一步真正的大头。宿主直接用 `ast` 定位类定义、把它作为文本给出去，agent
    多数情况就不必再读了（也确实要求它「先用给到的源码」）。
    """
    if not class_name or not source_dir.is_dir():
        return ""
    import ast as _ast

    for p in sorted(source_dir.rglob("*.py")):
        if any(part in (".git", "build", "dist", "__pycache__", "node_modules") for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if f"class {class_name}" not in text:
            continue
        try:
            tree = _ast.parse(text)
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ClassDef) and node.name == class_name:
                seg = _ast.get_source_segment(text, node) or ""
                if seg:
                    rel = p.relative_to(source_dir).as_posix()
                    return f"# --- {rel} :: class {class_name} ---\n{seg[:cap]}"
    return ""


def _module_prompt(parent: dict, child_modules: list[dict], module_path: str,
                   class_source: str = "") -> str:
    """第 2 步：展开**一个**模块的直接子节点与模块内边（每次输出都很短）。"""
    known = [{"id": c["id"], "class_name": c.get("class_name"), "module_path": c.get("module_path")}
             for c in child_modules]
    src_block = (f"\n【该类的源码（已替你取出）】**优先只依据它作答，不要再去读文件**"
                 "（省时间与额度）；确实缺信息时才读，且最多 1 次。\n"
                 f"{class_source}\n" if class_source else "")
    return (
        f"任务（第 2 步 / 共 2 步）：展开模块 `{parent['id']}`"
        f"（类 {parent.get('class_name')}，module_path=\"{module_path}\"）的**直接子节点**"
        "与这些子节点之间的数据流边。\n"
        "【红线】只允许读取文件（Read/Glob/Grep），禁止修改项目内任何文件。\n"
        + src_block +
        "\n【已由骨架确定的子模块（**不要**在 nodes 里重复列出）】\n"
        f"{json.dumps(known, ensure_ascii=False)}\n\n"
        "【输出】单层 JSON：\n"
        '{"nodes":[{"id","kind":"leaf|op|container","class_name","module_path",'
        '"parent_id":"' + str(parent['id']) + '","params":{...},"code_hint":"(可选)"}],'
        '"edges":[{"from":"节点 id","to":"节点 id"}]}\n'
        "【要求】\n"
        "· nodes 只放**叶子/函数式操作/Sequential 容器**（module 子节点来自骨架，不要重复）；\n"
        "· edges 覆盖本模块内**全部**数据流：从本模块的外部输入（或上一步给的子模块）"
        "到子节点、子节点之间、子节点到子模块、以及最终汇点；\n"
        "· 本模块有**多个输入**时，各输入分别连给它真正消费的那个子节点/子模块"
        "（不要把所有输入接到同一个节点上）；\n"
        "· 本模块的直接子节点（含子模块）**必须只有一个汇点**，多分支要用 op 汇合。\n"
        + _COMMON_RULES
    )


def _merge_stepwise(skeleton: dict, expansions: dict[str, dict]) -> dict:
    """确定性合并：骨架的模块 + 各模块展开出的子节点与边 → 完整 IR。

    `expansions[module_id] = {"nodes": [...], "edges": [...]}`（该模块的直接子节点与其内部边）。
    """
    nodes: list[dict] = [{"id": m["id"], "kind": "module", "class_name": m.get("class_name"),
                          "parent_id": m.get("parent_id"), "module_path": m.get("module_path") or ""}
                         for m in skeleton.get("modules") or []]
    seen = {n["id"] for n in nodes}
    edges: list[dict] = []
    for items in expansions.values():
        for n in items.get("nodes") or []:
            if n.get("id") in seen:            # 去重（骨架里已有的模块节点不重复）
                continue
            seen.add(n["id"])
            nodes.append(n)
        edges.extend(items.get("edges") or [])
    return {
        "source_file": skeleton.get("source_file"), "entry_class": skeleton.get("entry_class"),
        "task_type": skeleton.get("task_type"), "input_spec": skeleton.get("input_spec") or {},
        "root_id": skeleton.get("root_id"), "nodes": nodes, "edges": edges,
    }


async def _stepwise_decompose(source: Path, hierarchy: list[dict], entry_class: Optional[str],
                              hint: str, task_id: str, deadline: float,
                              on_event=None, set_stage=None, feedback: str = "") -> Optional[dict]:
    """**分步生成** IR：先骨架，再逐个模块展开，最后合并。

    为什么：一次吐出 90 个节点的整份 IR 太不稳（漏字段/漏边/整份不产出）。分步后每步输出都短，
    单步出错只重跑那一步（这里简化为「整体重来一次」，由外层重试循环兜底）。
    """
    set_stage = set_stage
    # 第 1 步：骨架（自带 3 次重试）
    skeleton = None
    reason = ""
    for i in range(3):
        budget = deadline - time.monotonic()
        if budget <= 30:
            break
        r = await agent_service.run_sync(
            _skeleton_prompt(hierarchy, entry_class, hint), cwd=str(source),
            output_schema=_SKELETON_SCHEMA, max_turns=40, timeout_s=int(min(900, budget)),
            on_event=on_event)
        skeleton = r.get("structured_output")
        errs = _skeleton_errors(skeleton)
        if not errs:
            break
        reason = "；".join(errs)
        skeleton = None
    if skeleton is None:
        raise RuntimeError(f"分步拆解：模块骨架未通过（{reason or '未产出'}）")

    # 第 2 步：逐模块展开。**模块之间彼此独立**（各自只看自己的类源码与子模块清单）→ 并发跑，
    # 墙钟从「模块数 × 单步」降到约「单步」（9 个模块 ≈ 10 分钟 → ≈ 2 分钟）；限 3 并发躲限流。
    expansions: dict[str, dict] = {}
    by_id = {m["id"]: m for m in skeleton["modules"]}
    sem = asyncio.Semaphore(int(os.getenv("DECOMPOSE_MODULE_CONCURRENCY", "3")))

    async def _expand(m: dict) -> tuple[str, dict]:
        children = [c for c in skeleton["modules"] if c.get("parent_id") == m["id"]]
        prompt = _module_prompt(m, children, m.get("module_path") or "",
                                class_source=_extract_class_source(source, m.get("class_name")))
        # 模块级重试 3 次：实测 agent 会偶发「某一步返回空」（没写结果文件也没回文本），
        # 重试这一模块比让整条分步流程从头再来便宜得多。
        async with sem:
            for _ in range(3):
                budget = deadline - time.monotonic()
                if budget <= 30:
                    raise RuntimeError("分步拆解：预算用尽")
                if set_stage:
                    set_stage(f"拆解中：展开模块 {m['id']}（{m.get('class_name')}）…")
                r = await agent_service.run_sync(
                    prompt, cwd=str(source), output_schema=_MODULE_SCHEMA,
                    max_turns=40, timeout_s=int(min(900, budget)), on_event=on_event)
                got = r.get("structured_output")
                if isinstance(got, dict) and isinstance(got.get("nodes"), list):
                    return m["id"], got
        raise RuntimeError(f"分步拆解：模块 {m['id']} 展开失败（未产出）")

    results = await asyncio.gather(*(_expand(m) for m in skeleton["modules"]),
                                   return_exceptions=True)
    for item in results:
        if isinstance(item, BaseException):
            raise item
        mid, got = item
        expansions[mid] = got
    ir = _merge_stepwise(skeleton, expansions)
    logger.info("分步拆解完成：模块 %d 个，节点 %d，边 %d", len(by_id), len(ir["nodes"]), len(ir["edges"]))
    return ir


def _skeleton_errors(skeleton) -> list[str]:
    """骨架的形态校验（便宜）：id/父子引用/根存在。"""
    if not isinstance(skeleton, dict):
        return ["骨架不是 JSON 对象"]
    mods = skeleton.get("modules")
    if not isinstance(mods, list) or not mods:
        return ["modules 缺失或为空"]
    errs: list[str] = []
    ids = [m.get("id") for m in mods if isinstance(m, dict)]
    for i, m in enumerate(mods):
        if not isinstance(m, dict):
            errs.append(f"modules[{i}] 不是对象")
            continue
        nid = m.get("id")
        if not isinstance(nid, str) or not nid.isidentifier():
            errs.append(f"模块 id 非法（须为 Python 标识符）：{nid!r}")
        elif not isinstance(m.get("class_name"), str) or not m.get("class_name"):
            errs.append(f"模块 {nid} 缺 class_name")
    if skeleton.get("root_id") not in ids:
        errs.append(f"root_id {skeleton.get('root_id')!r} 不在 modules 里")
    for m in mods:
        pid = m.get("parent_id")
        if pid is not None and pid not in ids:
            errs.append(f"模块 {m.get('id')} 的 parent_id {pid!r} 不存在")
    return errs[:6]


def _attempt_stage(attempt: int, total: int, last_reason: str) -> str:
    """重试循环的阶段文案（任务进度 banner 显示）。"""
    if attempt <= 1:
        return f"拆解中（第 1/{total} 次尝试）：读取源码并生成 IR…"
    tail = (last_reason or "").strip().replace("\n", " ")
    return f"第 {attempt}/{total} 次尝试：按上次失败原因修正中…（上次：{tail[:100]}）"


def _attempt_prompt(base: str, last_reason: str, prev_session: Optional[str], attempt: int,
                    allow_resume: bool = True):
    """返回 (prompt, resume)。

    第 2 次起两种打法：
    - `allow_resume=True`（默认）：**能续接就只发修正指令**（`resume=prev_session`），复用上一次
      已读进上下文的源码、省掉整轮重读；
    - `allow_resume=False`：**每轮全新会话**（完整 prompt + 修正要点）。用于「长会话退化」——
      实测 12 次重试里 8 次 agent 根本没产出 IR（resume 的会话越堆越长就越容易丢步），
      换新会话给它干净的上下文。
    """
    if attempt <= 1:
        return base, None
    reason = (last_reason or "").strip()
    if allow_resume and prev_session:
        msg = ("【上一次尝试未通过，请据此修正后，重新输出完整、合法的 IR JSON】\n"
               f"修正要点：{reason}\n"
               "**必须用 Write 工具把完整的 IR JSON 一次性写入下方系统指令给出的结果文件路径**"
               "（不要只在回复里说明；忽略上一次尝试用过的旧路径）；可先再读文件确认。")
        return msg, prev_session
    return base + f"\n\n【上一次尝试未通过，请据此修正】{reason}", None


def _progress_reporter(task_id: str):
    """返回 (set_stage, on_event)：把拆解的阶段与工具调用写进任务进度（进度打点）。

    `task_manager.update_progress` 是**整体覆盖**，故 on_event 每次都带上当前 stage，
    避免滚动 activity 时把 stage 冲掉。
    """
    state = {"stage": "拆解中…"}

    def set_stage(text: str) -> None:
        state["stage"] = text
        task_manager.update_progress(task_id, {"stage": text})

    def on_event(kind: str, data: dict) -> None:
        if kind != "tool":
            return
        name = (data or {}).get("name") or ""
        if name:
            task_manager.update_progress(task_id, {"stage": state["stage"],
                                                   "activity": f"调用工具 {name}"})

    return set_stage, on_event


def _real_inventory_hint(modules: list) -> str:
    """把探针给出的**真实带参层清单**渲染成回喂给 agent 的提示。

    **按路径模式归并**（数字段折叠成 `{i}` 并计数）：真实模型常是「N 个同构层」（scGPT 的
    `transformer_encoder.layers.{i}.*` 有 12 份），平铺 87 行既撑 prompt 又看不出规律；
    归并后是「`…layers.{i}.self_attn` ×12 :: nn.MultiheadAttention（每个 787968 参数）」——
    直接告诉 agent 要建几份、每份多大。
    """
    if not modules:
        return ""
    groups: dict[str, dict] = {}
    for m in modules:
        # 只折叠**整段是数字**的路径段（`layers.0.` → `layers.{i}.`）：`linear1`/`linear2`
        # 里的数字是名字的一部分，一并折叠会把两种不同的层混成一种，反而把信息抹平。
        pattern = ".".join("{i}" if seg.isdigit() else seg
                           for seg in str(m.get("path") or "").split("."))
        g = groups.setdefault(pattern, {"n": 0, "cls": m.get("class"), "params": m.get("params")})
        g["n"] += 1
    rows = []
    for pattern, g in list(groups.items())[:48]:
        times = f" ×{g['n']}" if g["n"] > 1 else ""
        rows.append(f"- {pattern}{times} :: {g['cls']}（每个 {g['params']} 参数）")
    more = f"\n（共 {len(groups)} 种路径模式、{len(modules)} 个带参层，仅列前 48 种）" if len(groups) > 48 else ""
    return ("\n【真实模型的带参层清单（**以此为准**重建 IR：module_path 取这里的 path，"
            "带数字段的写成同一模式、按 ×N 建 N 份）】\n" + "\n".join(rows) + more)


async def _fidelity_issues(source: Path, ws: Path, ir: dict, task_id: str,
                           attempt: int) -> tuple[list[str], str, str]:
    """把候选 IR 与**真实模型**比一遍 → `(差异清单, 备注)`。

    差异为空即通过。真实模型起不来（项目环境未就绪、缺 entry_args、依赖装不上）时**跳过**而不是
    误判「不忠实」，但**把跳过原因回传给调用方留痕**（否则「跳过」与「通过」在结果里长得一样）。
    比对口径与 6.4 两步验证一致（带参层数 / 参数量 / state_dict 形状 / 模块覆盖）。
    """
    if not DECOMPOSE_FIDELITY_CHECK:
        return [], "保真度自检已关闭（DECOMPOSE_FIDELITY_CHECK=0）", ""
    python = analysis_service._project_python(ws)
    if python is None:
        return [], "项目环境未就绪，跳过保真度自检", ""
    run_dir = ws / "runs" / "decompose" / task_id / f"fidelity_a{attempt}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        ir_path = run_dir / "candidate_ir.json"
        regen_path = run_dir / "regenerated.py"
        args_path = run_dir / "entry_args.json"
        out_path = run_dir / "fidelity.json"
        ir_path.write_text(json.dumps(ir, ensure_ascii=False), encoding="utf-8")
        regen_path.write_text(ir_codegen.generate(ir), encoding="utf-8")
        args_path.write_text(json.dumps(ir.get("entry_args") or {}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 —— 自检自身的准备失败不该阻断拆解
        return [], f"保真度自检准备失败，跳过：{e}", ""
    try:
        rc, log = await proc_util.run_command(
            [python, str(FIDELITY_SCRIPT), str(source), str(ir_path), str(regen_path),
             str(out_path), str(args_path)],
            cwd=str(run_dir), timeout=DECOMPOSE_FIDELITY_TIMEOUT_S)
    except asyncio.TimeoutError:
        return [], "保真度自检超时，跳过", ""
    if rc != 0:
        return [], f"保真度自检脚本异常（rc={rc}），跳过：{(log or '')[-200:]}", ""
    try:
        res = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [], f"保真度自检结果不可解析，跳过：{e}", ""
    if res.get("skipped"):
        return [], f"跳过（{res.get('reason')}）", ""
    if res.get("ok"):
        return [], "通过", ""
    return list(res.get("issues") or []), "不通过", _real_inventory_hint(res.get("real_param_modules") or [])


async def _run_decompose(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = _require_original(project_id)
    ws = _ws(project)
    prev_ir = _read_ir(project)          # 重拆解前先留一份：用户的补参要带过去（见 _carry_over_user_specs）
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
    prev_session: Optional[str] = None   # 上一次尝试的 agent 会话 id（重试时续接复用已读上下文）
    fidelity_note = "未执行"             # 保真度自检结论（通过/跳过原因），随 run_record 留痕
    set_stage, on_event = _progress_reporter(task_id)

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
        set_stage(_attempt_stage(attempt, DECOMPOSE_AGENT_RETRIES, last_reason))
        # 重试时把上一次的失败原因回喂（能续接则只发修正指令，复用上一轮已读的源码上下文）
        if DECOMPOSE_STEPWISE:
            # **分步生成**：骨架 → 逐模块展开 → 合并（每步输出短，稳定性好得多）
            hint_all = prompt[len(_decompose_prompt(hierarchy, entry_class)):]  # 知识带入等附注
            try:
                candidate = await _stepwise_decompose(
                    source, hierarchy, entry_class, hint_all, task_id, deadline,
                    on_event=on_event, set_stage=set_stage, feedback=last_reason)
            except Exception as e:  # noqa: BLE001
                last_reason = f"分步拆解失败：{e}"
                _record_fail(last_reason)
                continue
        else:
            ask, resume = _attempt_prompt(prompt, last_reason, prev_session, attempt,
                                          allow_resume=DECOMPOSE_RETRY_RESUME)

            async def _one_run(ask: str, resume_: Optional[str]) -> dict:
                return await agent_service.run_sync(
                    ask,
                    cwd=str(source),
                    output_schema=ir_schema.IR_SCHEMA,
                    max_turns=60,
                    timeout_s=int(min(DECOMPOSE_AGENT_TIMEOUT_S, max(30, budget))),
                    resume=resume_,
                    on_event=on_event,
                )

            try:
                try:
                    result = await _one_run(ask, resume)
                except Exception as exc:  # noqa: BLE001
                    if resume and agent_service.is_stale_session_error(exc):
                        # 续接目标已失效 → 退回完整 prompt 重跑一次（不带续接）
                        result = await _one_run(prompt, None)
                    else:
                        raise
                if resume and _as_ir(result.get("structured_output")) is None:
                    # 续接后**没写出结果文件**（模型只回了文字 / 写去了旧路径）→ 用完整 prompt 立刻重跑，
                    # 不占用一次重试名额、也不静默失败（2026-10-05 实测踩过：续接那轮 2s 空手而归）
                    result = await _one_run(prompt, None)
            except Exception as e:  # noqa: BLE001 —— agent 失败记 run_record 供检索（先例 4.3）
                _record_fail(f"第 {attempt}/{DECOMPOSE_AGENT_RETRIES} 次尝试失败: {e}")
                raise
            candidate = _as_ir(result.get("structured_output"))
            if DECOMPOSE_RETRY_RESUME:      # 不续接时不必记会话 id
                prev_session = result.get("session_id") or prev_session
        if candidate is None:
            last_reason = (
                f"agent 未产出有效 IR 结构（structured_output 缺失，第 {attempt}/{DECOMPOSE_AGENT_RETRIES} 次）"
            )
            _record_fail(last_reason)   # 也要留痕：否则「没产出」这类尝试在 run_record 里查不到
            # 仍拿不到结果文件 → 下一轮退回完整 prompt，避免反复踩同一坑
            prev_session = None
            continue
        errors = validate_ir(candidate)
        if errors:
            last_reason = "IR 结构校验失败: " + "；".join(errors)
            _record_fail(last_reason, candidate)
            continue
        # input_spec.inputs 的**形态**校验（便宜且确定）：必须是 root 直接子节点的 id——
        # 实测 agent 会填成 forward 的参数名/描述（`["src (gene token ids)", …]`），匹配不上任何节点
        # → 所有外部输入都落到第一个形参（value encoder 收到 token id → 前向 dtype 错）。
        bad_inputs = [x for x in ((candidate.get("input_spec") or {}).get("inputs") or [])
                      if x not in {n["id"] for n in candidate.get("nodes") or []}]
        if bad_inputs:
            last_reason = (
                f"input_spec.inputs 必须是 root 子节点的**节点 id**（按调用顺序），但 {bad_inputs} "
                "不是任何节点 id——不要写 forward 的参数名或描述文字；单输入模型请省略该项"
            )
            _record_fail(last_reason, candidate)
            continue
        # 节点数上限（便宜且确定，先于保真度自检）：IR 越长越不稳——漏字段/漏边/整份不产出都
        # 与输出长度相关。把重复结构折叠掉，既短又稳（参数量不受影响：code_hint 实例化同一构造）。
        if len(candidate.get("nodes") or []) > DECOMPOSE_MAX_NODES:
            last_reason = (
                f"节点数 {len(candidate['nodes'])} 超过上限 {DECOMPOSE_MAX_NODES}：单次输出越长越容易"
                "漏字段/漏边甚至整份不产出——请把**同一构造重复多次的结构**（如 encoder 的 N 层）"
                "与**标准复合层**（nn.TransformerEncoder/EncoderLayer 等）用 code_hint **折叠成 1 个节点**，"
                "节点数控制在 40 以内再提交"
            )
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
        # **保真度自检前先把用户补参带进来**（entry_args/extra/forward_kwargs/inputs）：
        # agent 不产出这些，而真实模型往往正是靠它们才实例化得起来；不带就必然「跳过」自检
        # ——那样等于把「静默放行坏 IR」原样保留（本函数第一版就踩了这个坑，靠独立复跑才发现）。
        _carry_over_user_specs(prev_ir, candidate)
        # **保真度自检**：与真实模型比参数量/带参层/逐层形状/模块覆盖。上面两道自检只看
        # 「IR 内部合法 + 能再生成」，从不与真实模型比对——少拆子树（scGPT 漏 mvc_decoder）、
        # 尺寸算错都会被放行，拖到「⑤ 两步验证」才暴露，且换一次拆解复现一次。这里拦下并重试。
        set_stage(f"第 {attempt}/{DECOMPOSE_AGENT_RETRIES} 次尝试：保真度自检（与真实模型比对结构/前向）…")
        issues, fidelity_note, fidelity_hint = await _fidelity_issues(source, ws, candidate, task_id, attempt)
        if issues:
            last_reason = ("IR 保真度自检未通过（与真实模型比对）：" + "；".join(issues) + fidelity_hint)
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
            "fidelity": fidelity_note,
            "knowledge_brought": {
                "param_advice": len(knowledge.get("param_advice") or []),
                "dependency_conflict": len(knowledge.get("dependency_conflict") or []),
            },
        },
        "artifact_path": str(ir_path),
        "started_at": started, "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {
        "stage": f"拆解完成：{ir.get('entry_class')}（{len(ir['nodes'])} 节点 / {len(ir['edges'])} 边）",
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


def _uncovered_modules(ir: dict, trace_output: dict) -> list[str]:
    """trace **实际执行到**、但 IR 里没有对应节点的子模块路径（agent 漏拆的子树）。

    用「真实实例 + 真前向」的结果做完备性核对，而不是静态报告——后者列出源码里**所有** nn.Module
    子类（scGPT 41 个里有 20 个属于别的模型变体/测试替身），照它校验会满屏假阳性。
    实测价值：scGPT 的 IR 漏了整棵 `mvc_decoder`（2 个 Linear / 52.5 万参数），此前要等到「⑤ 两步
    验证」看到「87 vs 85 层」才间接暴露；这里当场点名。
    """
    shapes = trace_output.get("shapes") if isinstance(trace_output.get("shapes"), dict) else trace_output
    covered = {str(n.get("module_path") or "") for n in ir["nodes"]}
    covered.add("")

    def _is_covered(path: str) -> bool:
        # 命中某个 IR 节点自身，**或落在它的子树里**：`nn.TransformerEncoder` 由一条 code_hint
        # 节点折叠表达时，它下面 100+ 个真实子模块不该被算成「漏拆」。
        return any(path == mp or path.startswith(mp + ".") for mp in covered)

    return sorted(p for p in (shapes or {}) if not _is_covered(p))


def _fill_edge_shapes(ir: dict) -> int:
    """把边上流动的张量形状写入 `edges[].tensor_shape`。

    背景：IR schema 早就有 `tensor_shape` 字段、`ir_to_graphir` 也在消费它（`_edge_kind` 附近），
    但此前**没有任何写入方**——节点形状有 forward hook 捕获与 op 兜底，边形状一直是 null，
    画布/查看器拿不到「这条线上流的是什么形状」。

    口径：边上流过的就是**源节点的输出**，取源节点 `output_shape`；缺失时退回 `input_shape`；
    两者都未知就保持 null（不猜、不臆造）。已存在的值不覆盖（与节点形状的「只补缺」一致）。

    与 `ir_hash` 的关系：`canonical_ir` 的边投影只有 `from`/`to`，`tensor_shape` 不进哈希，
    因此本次回填**不会**让已验证的 IR 无故变 stale（与 6.6-4「回填不产生 stale」一致）。
    """
    filled = 0
    node_map = nodes_by_id(ir)
    for e in ir.get("edges") or []:
        if e.get("tensor_shape"):
            continue
        src = node_map.get(e.get("from")) or {}
        shape = src.get("output_shape") or src.get("input_shape")
        if not shape:
            continue
        e["tensor_shape"] = list(shape)
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
        task_manager.update_progress(task_id, {
            "stage": "补形状：IR 形状已完整，无需追踪（跳过）",
            "skipped": True, "reason": "IR 形状已完整，无需追踪"})
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
    task_manager.update_progress(task_id, {"stage": "补形状：在项目环境运行模型抓取各层形状…"})
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
    edges_filled = _fill_edge_shapes(ir)  # 边形状（此前无写入方，见该函数 docstring）
    uncovered = _uncovered_modules(ir, shapes)   # 实际执行到、IR 却没拆的子树（agent 漏拆）
    _write_ir(project, ir)
    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "decompose_trace",
        "command": command, "status": "success",
        "metrics": {"captured_paths": len(shapes.get("shapes", shapes)) if isinstance(shapes, dict) else 0,
                    "filled": filled, "inferred": inferred, "edges_filled": edges_filled,
                    "uncovered": len(uncovered), "uncovered_examples": uncovered[:5]},
        "artifact_path": str(out_json),
        "started_at": started, "finished_at": _now(),
    })
    _captured = len(shapes.get("shapes", shapes)) if isinstance(shapes, dict) else 0
    _gap = (f"；⚠ IR 未覆盖 {len(uncovered)} 个实际子模块（如 {uncovered[0]}），"
            "建议重新拆解或手工补节点" if uncovered else "")
    task_manager.update_progress(task_id, {
        "stage": f"补形状完成：捕获 {_captured} 个模块，回填 {filled + inferred + edges_filled} 处{_gap}",
        "captured_paths": _captured,
        "filled": filled, "inferred": inferred, "edges_filled": edges_filled,
        "uncovered": len(uncovered), "uncovered_examples": uncovered[:5],
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
    task_manager.update_progress(task_id, {"stage": "验证：再生成代码并与原模型逐 seed 数值比对…"})
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
        "stage": f"验证完成：{'通过' if overall == 'passed' else '未通过'}"
                 + (f"（{verification.get('failure_reason')}）" if overall != "passed" else ""),
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
    started = _now()

    def _record_fail(reason: str) -> None:
        """失败也要留 run_record（架构九.4：失败可检索供 agent 改进）。

        与同章 decompose/trace/verify 同口径：入库链路每一处失败（IR 缺失/不完整、
        验证未过或不新鲜、写盘失败）都**先记一条 failed、再原样抛出**，
        异常类型与消息不变，调用方（task_manager）行为不受影响。
        """
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "module_ingest",
            "command": "module package → MODULES_DIR", "status": "failed", "error": reason,
            "started_at": started, "finished_at": _now(),
        })

    project = _require_original(project_id)
    ws = _ws(project)
    ir = _read_ir(project)
    if ir is None:
        reason = "尚未拆解：请先 POST /api/projects/{id}/decompose"
        _record_fail(reason)
        raise RuntimeError(reason)
    verification = get_verification(project_id)
    if verification is None:
        reason = "尚未验证：请先 POST /api/projects/{id}/decompose/verify"
        _record_fail(reason)
        raise RuntimeError(reason)
    if verification.get("overall") != "passed":
        reason = "验证未通过，不能入库；请调整 IR 后重新验证"
        _record_fail(reason)
        raise RuntimeError(reason)
    if verification.get("ir_hash") != ir_hash(ir):
        reason = "IR 已修改（调参）但未重新验证，验证结果已过期；请重新执行 verify"
        _record_fail(reason)
        raise RuntimeError(reason)

    try:
        code = ir_codegen.generate(ir)  # 验证已通过，此处不应再缺项；缺则任务失败
    except Exception as e:  # noqa: BLE001 —— 留痕后再原样抛出（缺项清单仍是 IrIncompleteError）
        _record_fail(f"IR 不完整，无法再生成代码: {e}")
        raise
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
    except Exception as e:  # noqa: BLE001 —— 先留痕再抛出：补偿自身失败也不会丢掉这次失败记录
        _record_fail(f"入库失败: {e}")
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
        "stage": f"入库完成：{module_id}:{module_version}",
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
