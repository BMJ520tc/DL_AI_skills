"""画布网络代码再生成（阶段4 4c，模块详细设计 7.5）。

把结构化项目的画布快照（GraphIR v2）生成**可运行**的 PyTorch 代码：

- 标准节点：语义忠实移植前端 `codeCompile.ts` 的 `compileGraphToScript` +
  `generateMainCode`（同一套连线变量命名、拓扑序、init/forward 行格式），
  节点代码模板见 `NODE_TABLE`（与前端各节点 getInitCode/getForwardCode 对应）。
- module_ref 节点：内联模块包的代码文件（`module.py`；v4 之前的历史包为 `model.py`，两者都探测），
  实例化无参调用；画布上改过固化的参数则拒绝导出（不静默丢弃用户修改）。
- 前端画布导出与训练运行共用本引擎——导出即所训，两端一致。

纯函数、无 IO（module.py 读取除外）、无 subprocess；同图恒产出同代码。
不支持的图（控制流节点 / 拆解视图 ir 节点 / 本地模块）抛 `ExportError`，
message 面向用户，禁止静默产出错误代码。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.services import knowledge_service

HEADER = "import torch\nimport torch.nn as nn"
MAIN_CLASS = "GeneratedModel"


class ExportError(ValueError):
    """画布图无法导出为代码（结构/节点类型/模块引用问题）。"""


# ---------------------------------------------------------------------------
# 名称清洗与取值语义（与前端 codeCompile.ts / BaseClass.tsx 逐一对齐）
# ---------------------------------------------------------------------------

def sanitize_ident(name: str) -> str:
    """与前端 sanitizeIdent 一致：非字母数字下划线替换为 _；数字开头补 _；空串给 _x。"""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned:
        return "_x"
    return cleaned if re.match(r"^[A-Za-z_]", cleaned) else f"_{cleaned}"


def get_param(schema: dict[str, dict], data: dict, key: str) -> Any:
    """与前端 getParamValue 一致：值缺省回退 paramSchema 默认值；number 类型只认真数值。"""
    spec = schema.get(key) or {}
    val = data.get(key)
    if spec.get("type") == "number":
        # JS: typeof val === "number"（bool 不算 number）才用；否则回退默认值
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return val
        return spec.get("defaultValue")
    # JS: val !== undefined 即用（含 ""、null）；Python 侧 None 视为缺省（JSON 图不存 undefined）
    if val is not None:
        return val
    return spec.get("defaultValue")


def _edge_var_name(edge: dict) -> str:
    """边的变量名：优先画布写入的 label（onConnect 固定写 out_<source>[_<handle>]）；
    无 label 退回按边 id 生成稳定唯一名（与 codeCompile.ts edgeVarName 一致）。"""
    data = edge.get("data")
    label = data.get("label") if isinstance(data, dict) else None
    if label:
        return sanitize_ident(str(label))
    return sanitize_ident(f"edge_{edge['id']}") if edge.get("id") else sanitize_ident("edge")


# ---------------------------------------------------------------------------
# 节点代码模板表
# ---------------------------------------------------------------------------
# 每个节点类型一条：handles（连线句柄规格）、params（paramSchema 默认值表，供
# get_param 回退）、init/forward（与前端该节点 getInitCode/getForwardCode 语义一致）。
# 表由 scripts 侧对照 frontend/src/nodes/ 逐节点移植，修改前端节点时须同步。
# 控制流节点（repeat_layer / module_list）含内部子图，不支持服务端导出，见 UNSUPPORTED。

# ---------------------------------------------------------------------------
# 标准节点模板（逐节点移植 frontend/src/nodes/** 的 paramSchema / getInitCode /
# getForwardCode；编译口径对齐 codeCompile.ts）
# ---------------------------------------------------------------------------
# - init(data, name) / forward(data, name, inputs, outputs) 与框架调用约定一致，
#   返回不带前导缩进的代码行；多行模板（残差块）内部自带 8 空格续行缩进。
# - handles 取自前端类的**静态 handles 字段**：codeCompile 只读取它决定输出变量名
#   （ClassRef.handles），未定义静态 handles 的节点 sourceHandles 为空、输出名改为
#   逐条出边取 label；故这类节点此处 sources 记 []。targets 仅记录组件渲染句柄，
#   代码生成不使用。
# - params 为该节点 paramSchema 的默认值表（type/defaultValue 与 FieldSpec 一致），
#   供 get_param 回退与 buildInitString 移植共用。


def _js_num_str(val: Any) -> str:
    """JS 模板插值里的 Number→string（定点/指数阈值与 Python str() 不同）。"""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        if abs(val) < 1e21:
            return str(val)
        val = float(val)
    if val != val:
        return "NaN"
    if val == float("inf"):
        return "Infinity"
    if val == float("-inf"):
        return "-Infinity"
    if val == 0:
        return "0"
    sign = "-" if val < 0 else ""
    if val.is_integer() and abs(val) < 1e21:
        return sign + str(int(abs(val)))
    r = repr(abs(val))
    if "e" in r:
        mant, _, exp_str = r.partition("e")
        digits = mant.replace(".", "")
        n = int(exp_str) + 1
    else:
        int_part, _, frac_part = r.partition(".")
        all_digits = int_part + frac_part
        lead = len(all_digits) - len(all_digits.lstrip("0"))
        digits = all_digits[lead:]
        n = len(int_part) - lead
    if not digits:
        return "0"
    k = len(digits)
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mant_out = digits if k == 1 else digits[0] + "." + digits[1:]
        body = f"{mant_out}e{'+' if e >= 0 else '-'}{abs(e)}"
    return sign + body


def _val_str(val: Any) -> str:
    """JS 模板插值 ${val}：数值按 Number→string，布尔按小写字面，其余 str()。"""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return _js_num_str(val)
    return str(val)


def _js_eq(a: Any, b: Any) -> bool:
    """JS 严格相等（===）近似：布尔/数值/字符串按类型区分（bool 不与 0/1 相等）。"""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    return a == b


def _to_py(spec: dict, val: Any) -> str:
    """buildInitString 的 toPython：boolean 打 True/False，其余按模板插值。"""
    if spec.get("type") == "boolean":
        return "True" if val else "False"
    return _val_str(val)


def _build_init(
    module: str, name: str, schema: dict[str, dict], required: tuple[str, ...], data: dict
) -> str:
    """buildInitString 移植：required 参数恒写 key=value（缺省回退默认值）；
    可选参数仅在值存在且与默认值不严格相等时写出。"""
    args: list[str] = []
    for key, spec in schema.items():
        val = data.get(key)
        if key in required:
            args.append(f"{key}={_to_py(spec, val if val is not None else spec.get('defaultValue'))}")
        elif val is not None and not _js_eq(val, spec.get("defaultValue")):
            args.append(f"{key}={_to_py(spec, val)}")
    return f"self.{name} = {module}({', '.join(args)})"


def _in0(inputs: list[str]) -> str:
    """inputs[0] || "x"（JS 假值回退）。"""
    return inputs[0] if inputs and inputs[0] else "x"


def _out0(outputs: list[str]) -> str:
    """outputs[0] || "x"（JS 假值回退）。"""
    return outputs[0] if outputs and outputs[0] else "x"


def _at(items: list[str], idx: int, fallback: str) -> str:
    """items[idx] || fallback（越界与空串都回退）。"""
    return items[idx] if len(items) > idx and items[idx] else fallback


def _module_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    """通用 forward：out = self.<name>(in)（绝大多数层节点共用）。"""
    return f"{_out0(outputs)} = self.{name}({_in0(inputs)})"


# 前端 handles 常量：targets 仅记录渲染句柄；sources 为静态 handles 的 sources。
_HANDLES_UNARY = {"targets": ["in-0"], "sources": ["out-0"]}
_HANDLES_BINARY = {"targets": ["in-0", "in-1"], "sources": ["out-0"]}
_HANDLES_UNSPEC = {"targets": ["in-0"], "sources": []}
_HANDLES_UNSPEC_BINARY = {"targets": ["in-0", "in-1"], "sources": []}
_HANDLES_CONST = {"targets": [], "sources": ["out-0"]}
_HANDLES_INPUT = {"targets": [], "sources": ["out-0"]}
_HANDLES_LOSS_PRED = {"targets": ["pred", "target"], "sources": ["out-0"]}
_HANDLES_LOSS_LOGITS = {"targets": ["logits", "target"], "sources": ["out-0"]}
_HANDLES_MHA = {"targets": ["query", "key", "value", "mask"], "sources": ["out-0"]}
_HANDLES_RESIDUAL = {"targets": ["in-0"], "sources": ["out_main", "out_skip"]}

# 各节点 paramSchema（仅代码生成路径用到的键，顺序与前端一致）
_P_CONCAT = {"dim": {"type": "number", "defaultValue": 1}}
_P_POW = {"exponent": {"type": "number", "defaultValue": 2}}
_P_CLIP = {
    "min": {"type": "number", "defaultValue": 0},
    "max": {"type": "number", "defaultValue": 1},
}
_P_REDUCE = {"dim": {"type": "number", "defaultValue": -1}}
_P_SOFTMAX = {"dim": {"type": "number", "defaultValue": -1}}
_P_RESHAPE = {"target_shape": {"type": "text", "defaultValue": "-1"}}
_P_TRANSPOSE = {"perm": {"type": "text", "defaultValue": "0,2,3,1"}}
_P_CONST_TENSOR = {"shape": {"type": "text", "defaultValue": "1,1"}}
_P_BATCHNORM = {
    "num_features": {"type": "number", "defaultValue": 32},
    "eps": {"type": "number", "defaultValue": 1e-5},
    "momentum": {"type": "number", "defaultValue": 0.1},
    "affine": {"type": "boolean", "defaultValue": True},
}
_P_GROUPNORM = {
    "num_groups": {"type": "number", "defaultValue": 8},
    "num_features": {"type": "number", "defaultValue": 32},
    "eps": {"type": "number", "defaultValue": 1e-5},
    "affine": {"type": "boolean", "defaultValue": True},
}
_P_LAYERNORM = {
    "normalized_shape": {"type": "number", "defaultValue": 128},
    "eps": {"type": "number", "defaultValue": 1e-5},
    "affine": {"type": "boolean", "defaultValue": True},
}
_P_RMSNORM = {
    "normalized_shape": {"type": "number", "defaultValue": 128},
    "eps": {"type": "number", "defaultValue": 1e-6},
}
_P_DROPOUT = {"p": {"type": "number", "defaultValue": 0.5}}
_P_STOCHASTIC_DEPTH = {"p": {"type": "number", "defaultValue": 0.1}}
_P_LINEAR = {
    "in_features": {"type": "number", "defaultValue": 1},
    "out_features": {"type": "number", "defaultValue": 1},
    "bias": {"type": "boolean", "defaultValue": True},
}
_P_CONV = {
    "in_channels": {"type": "number", "defaultValue": 1},
    "out_channels": {"type": "number", "defaultValue": 1},
    "kernel_size": {"type": "number", "defaultValue": 3},
    "stride": {"type": "number", "defaultValue": 1},
    "padding": {"type": "number", "defaultValue": 0},
    "bias": {"type": "boolean", "defaultValue": True},
}
_P_DEPTHWISE = {
    "in_channels": {"type": "number", "defaultValue": 1},
    "depth_multiplier": {"type": "number", "defaultValue": 1},
    "kernel_size": {"type": "number", "defaultValue": 3},
    "stride": {"type": "number", "defaultValue": 1},
    "padding": {"type": "number", "defaultValue": 0},
    "dilation": {"type": "number", "defaultValue": 1},
    "bias": {"type": "boolean", "defaultValue": True},
}
_P_POINTWISE = {
    "in_channels": {"type": "number", "defaultValue": 1},
    "out_channels": {"type": "number", "defaultValue": 1},
    "bias": {"type": "boolean", "defaultValue": True},
}
_P_CONVT = {
    "in_channels": {"type": "number", "defaultValue": 1},
    "out_channels": {"type": "number", "defaultValue": 1},
    "kernel_size": {"type": "number", "defaultValue": 3},
    "stride": {"type": "number", "defaultValue": 1},
    "padding": {"type": "number", "defaultValue": 0},
    "output_padding": {"type": "number", "defaultValue": 0},
    "dilation": {"type": "number", "defaultValue": 1},
    "bias": {"type": "boolean", "defaultValue": True},
}
_P_UPSAMPLE = {
    "scale_factor": {"type": "number", "defaultValue": 2},
    "mode": {"type": "select", "defaultValue": "nearest"},
}
_P_RESIDUAL = {
    "channels": {"type": "number", "defaultValue": 64},
    "kernel_size": {"type": "number", "defaultValue": 3},
    "use_bn": {"type": "boolean", "defaultValue": True},
}
_P_POOL = {
    "kernel_size": {"type": "number", "defaultValue": 2},
    "stride": {"type": "number", "defaultValue": 2},
}
_P_ADAPTIVE_POOL = {"output_size": {"type": "number", "defaultValue": 1}}
_P_EMBEDDING = {
    "num_embeddings": {"type": "number", "defaultValue": 1000},
    "embedding_dim": {"type": "number", "defaultValue": 128},
    "padding_idx": {"type": "number", "defaultValue": -1},
}
_P_RECURRENT = {
    "input_size": {"type": "number", "defaultValue": 128},
    "hidden_size": {"type": "number", "defaultValue": 128},
    "num_layers": {"type": "number", "defaultValue": 1},
    "bidirectional": {"type": "boolean", "defaultValue": False},
}
_P_MHA = {
    "embed_dim": {"type": "number", "defaultValue": 128},
    "num_heads": {"type": "number", "defaultValue": 8},
}
_P_POS_ENC = {"dim": {"type": "number", "defaultValue": 128}}

# buildInitString 合成 schema（Depthwise / Pointwise 在 init 时拼接额外参数）
_P_DEPTHWISE_INIT = {
    **_P_DEPTHWISE,
    "out_channels": {"type": "number", "defaultValue": 1},
    "groups": {"type": "number", "defaultValue": 1},
}
_P_POINTWISE_INIT = {
    **_P_POINTWISE,
    "kernel_size": {"type": "number", "defaultValue": 1},
    "stride": {"type": "number", "defaultValue": 1},
    "padding": {"type": "number", "defaultValue": 0},
}

# buildInitString 的 required 集合（与前端 schema.required 一致）
_R_CONV = ("in_channels", "out_channels", "kernel_size")
_R_BATCHNORM = ("num_features",)
_R_GROUPNORM = ("num_groups", "num_features")
_R_LAYERNORM = ("normalized_shape",)
_R_STOCHASTIC_DEPTH = ("p",)
_R_DEPTHWISE_INIT = ("in_channels", "kernel_size", "out_channels", "groups")
_R_POINTWISE_INIT = ("in_channels", "out_channels", "kernel_size", "stride", "padding")


# ---- inputs ----
# 前端: nodes/inputs/InputNode.tsx
def _input_init(data: dict, name: str) -> str:
    return "# input layer does not require initialization"


def _input_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    return f"{_out0(outputs)} = {_in0(inputs)}  # input passthrough"


# ---- torch_ops ----
# 前端: nodes/pytorch_core/AddNode.tsx
def _add_init(data: dict, name: str) -> str:
    return "# add is functional"


def _add_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    args = [i for i in inputs if i]
    if not args:
        return ""
    sum_expr = " + ".join(args)
    return f"{_out0(outputs)} = {sum_expr}"


# 前端: nodes/pytorch_core/ConcatNode.tsx
def _concat_init(data: dict, name: str) -> str:
    return "# concat has no module; handled in forward"


def _concat_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    args = ", ".join(i for i in inputs if i)
    dim = data.get("dim")
    if dim is None:
        dim = _P_CONCAT["dim"]["defaultValue"]
    return f"{_out0(outputs)} = torch.cat([{args}], dim={_val_str(dim)})"


# 前端: nodes/pytorch_core/ElementwiseBinaryNode.tsx（Sub/Mul/Div 工厂）
def _binary_entry(label: str, op: str) -> dict:
    def _init(data: dict, name: str) -> str:
        return f"# {label} uses functional op"

    def _forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
        out = _out0(outputs)
        args = [i for i in inputs if i]
        if len(args) < 2:
            return f"{out} = {args[0] if args else 'x'}"
        sep = f" {op} "
        return f"{out} = {sep.join(args)}"

    return {"handles": _HANDLES_BINARY, "params": {}, "init": _init, "forward": _forward}


# 前端: nodes/pytorch_core/UnaryElementwiseNode.tsx（Exp/Log/Sqrt 工厂）
def _unary_fn_entry(label: str, torch_fn: str) -> dict:
    def _init(data: dict, name: str) -> str:
        return f"# {label} uses functional torch op"

    def _forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
        return f"{_out0(outputs)} = torch.{torch_fn}({_in0(inputs)})"

    return {"handles": _HANDLES_UNARY, "params": {}, "init": _init, "forward": _forward}


# 前端: nodes/pytorch_core/PowNode.tsx
def _pow_init(data: dict, name: str) -> str:
    return "# pow uses functional torch.pow"


def _pow_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    exp = get_param(_P_POW, data, "exponent")
    return f"{_out0(outputs)} = torch.pow({_in0(inputs)}, {_val_str(exp)})"


# 前端: nodes/pytorch_core/ClipNode.tsx
def _clip_init(data: dict, name: str) -> str:
    return "# clip uses torch.clamp"


def _clip_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    mn = get_param(_P_CLIP, data, "min")
    mx = get_param(_P_CLIP, data, "max")
    args: list[str] = []
    if mn is not None:
        args.append(f"min={_val_str(mn)}")
    if mx is not None:
        args.append(f"max={_val_str(mx)}")
    suffix = ", " + ", ".join(args) if args else ""
    return f"{_out0(outputs)} = torch.clamp({_in0(inputs)}{suffix})"


# 前端: nodes/pytorch_core/MatMulNode.tsx
def _matmul_init(data: dict, name: str) -> str:
    return "# matmul uses torch.matmul"


def _matmul_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    left = _at(inputs, 0, "x")
    right = _at(inputs, 1, "y")
    return f"{_out0(outputs)} = torch.matmul({left}, {right})"


# 前端: nodes/pytorch_core/ReductionNode.tsx（Sum/Mean/Prod 工厂）
def _reduction_entry(label: str, torch_fn: str) -> dict:
    def _init(data: dict, name: str) -> str:
        return f"# {label} uses torch.{torch_fn}"

    def _forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
        dim = get_param(_P_REDUCE, data, "dim")
        return f"{_out0(outputs)} = torch.{torch_fn}({_in0(inputs)}, dim={_val_str(dim)})"

    return {"handles": _HANDLES_UNARY, "params": _P_REDUCE, "init": _init, "forward": _forward}


# 前端: nodes/pytorch_core/ArgExtremaNodes.tsx（Max/Min/ArgMax/ArgMin）
def _extrema_entry(torch_fn: str, values_suffix: bool) -> dict:
    def _init(data: dict, name: str) -> str:
        tail = "(...).values" if values_suffix else ""
        return f"# {torch_fn} uses torch.{torch_fn}{tail}"

    def _forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
        dim = get_param(_P_REDUCE, data, "dim")
        tail = ".values" if values_suffix else ""
        return f"{_out0(outputs)} = torch.{torch_fn}({_in0(inputs)}, dim={_val_str(dim)}){tail}"

    return {"handles": _HANDLES_UNARY, "params": _P_REDUCE, "init": _init, "forward": _forward}


# ---- tensor_shape ----
# 前端: nodes/pytorch_core/ReshapeNode.tsx
def _reshape_init(data: dict, name: str) -> str:
    return "# reshape handled in forward"


def _reshape_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    dims = data.get("target_shape") or "-1"
    return f"{_out0(outputs)} = {_in0(inputs)}.view({dims})"


# 前端: nodes/pytorch_core/TransposeNode.tsx
def _transpose_init(data: dict, name: str) -> str:
    return "# transpose handled in forward"


def _transpose_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    perm = data.get("perm") or ""
    return f"{_out0(outputs)} = {_in0(inputs)}.permute({perm})"


# 前端: nodes/pytorch_core/FlattenNode.tsx
def _flatten_init(data: dict, name: str) -> str:
    return f"self.{name} = nn.Flatten()"


# 前端: nodes/pytorch_core/Identity.tsx（pass_layer）
def _pass_init(data: dict, name: str) -> str:
    return f"self.{name} = nn.Identity()"


def _pass_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    inp = inputs[0] if inputs else "x"
    out = outputs[0] if outputs else "x"
    return f"{out} = {inp}"


# ---- tensor_create ----
# 前端: nodes/pytorch_core/ConstantTensorNodes.tsx（Zeros/Ones/Rand 工厂）
def _constant_entry(torch_fn: str) -> dict:
    def _init(data: dict, name: str) -> str:
        return "# constant tensor created in forward"

    def _forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
        shape = data.get("shape") or ""
        return f"{_out0(outputs)} = torch.{torch_fn}({shape})"

    return {"handles": _HANDLES_CONST, "params": _P_CONST_TENSOR, "init": _init, "forward": _forward}


# ---- activations ----
# 前端: nodes/pytorch_core/UnaryActivationNode.tsx / activations.tsx
def _activation_entry(init_expr: str) -> dict:
    def _init(data: dict, name: str) -> str:
        return f"self.{name} = {init_expr}"

    def _forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
        return f"{_out0(outputs)} = self.{name}({_in0(inputs)})"

    return {"handles": _HANDLES_UNARY, "params": {}, "init": _init, "forward": _forward}


# ---- 归一化 / softmax ----
# 前端: nodes/pytorch_core/SoftmaxNode.tsx（Softmax/LogSoftmax）
def _softmax_entry(module: str) -> dict:
    def _init(data: dict, name: str) -> str:
        dim = get_param(_P_SOFTMAX, data, "dim")
        return f"self.{name} = nn.{module}(dim={_val_str(dim)})"

    return {"handles": _HANDLES_UNARY, "params": _P_SOFTMAX, "init": _init, "forward": _module_forward}


# 前端: nodes/pytorch_core/NormNodes.tsx、RegNodes.tsx、vision/conv/*、dense/* 等
# buildInitString 路径的统一入口（schema/required 与前端逐键一致）
def _build_init_entry(
    module: str, params: dict, required: tuple[str, ...], handles: dict | None = None
) -> dict:
    def _init(data: dict, name: str) -> str:
        return _build_init(module, name, params, required, data)

    return {
        "handles": _HANDLES_UNSPEC if handles is None else handles,
        "params": params,
        "init": _init,
        "forward": _module_forward,
    }


# ---- dense ----
# 前端: nodes/dense/LinearLayer.tsx
def _linear_init(data: dict, name: str) -> str:
    in_f = data.get("in_features") or _P_LINEAR["in_features"]["defaultValue"]
    out_f = data.get("out_features") or _P_LINEAR["out_features"]["defaultValue"]
    bias_arg = ", bias=False" if data.get("bias") is False else ""
    return (
        f"self.{name} = nn.Linear(in_features={_val_str(in_f)}, "
        f"out_features={_val_str(out_f)}{bias_arg})"
    )


def _linear_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    inp = inputs[0] if inputs else "x"
    out = outputs[0] if outputs else "x"
    return f"{out} = self.{name}({inp})"


# ---- vision_conv ----
# 前端: nodes/vision/conv/DepthwiseConv2dNode.tsx
def _depthwise_init(data: dict, name: str) -> str:
    in_ch = get_param(_P_DEPTHWISE, data, "in_channels")
    mult = get_param(_P_DEPTHWISE, data, "depth_multiplier")
    merged = {**data, "out_channels": in_ch * mult, "groups": in_ch}
    return _build_init("nn.Conv2d", name, _P_DEPTHWISE_INIT, _R_DEPTHWISE_INIT, merged)


# 前端: nodes/vision/conv/PointwiseConv2dNode.tsx
def _pointwise_init(data: dict, name: str) -> str:
    merged = {**data, "kernel_size": 1, "stride": 1, "padding": 0}
    return _build_init("nn.Conv2d", name, _P_POINTWISE_INIT, _R_POINTWISE_INIT, merged)


# 前端: nodes/vision/conv/UpsampleNode.tsx
def _upsample_init(data: dict, name: str) -> str:
    scale = get_param(_P_UPSAMPLE, data, "scale_factor")
    mode = get_param(_P_UPSAMPLE, data, "mode")
    return f'self.{name} = nn.Upsample(scale_factor={_val_str(scale)}, mode="{_val_str(mode)}")'


# 前端: nodes/vision/blocks/ResidualBlock.tsx
def _residual_init(data: dict, name: str) -> str:
    c = get_param(_P_RESIDUAL, data, "channels")
    k = get_param(_P_RESIDUAL, data, "kernel_size")
    use_bn = get_param(_P_RESIDUAL, data, "use_bn")
    padding = k // 2  # 与前端 Math.floor(k / 2) 一致（含负值）
    bn_line = (
        f"\n        self.{name}_bn = nn.BatchNorm2d(num_features={_val_str(c)})" if use_bn else ""
    )
    return (
        f"self.{name}_conv = nn.Conv2d(in_channels={_val_str(c)}, out_channels={_val_str(c)}, "
        f"kernel_size={_val_str(k)}, padding={_val_str(padding)}){bn_line}"
    )


def _residual_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    x = _in0(inputs)
    out_main = _at(outputs, 0, f"{name}_out_main")
    out_skip = _at(outputs, 1, f"{name}_out_skip")
    use_bn = get_param(_P_RESIDUAL, data, "use_bn")
    conv = f"{out_main} = self.{name}_conv({x})"
    bn = f"\n        {out_main} = self.{name}_bn({out_main})" if use_bn else ""
    act = f"\n        {out_main} = torch.relu({out_main})"
    skip = f"\n        {out_skip} = {x} + {out_main}"
    return f"{conv}{bn}{act}{skip}"


# ---- vision_pool ----
# 前端: nodes/vision/pooling/{Max,Avg}Pool{1,2,3}dNode.tsx
def _pool_entry(module: str) -> dict:
    def _init(data: dict, name: str) -> str:
        k = get_param(_P_POOL, data, "kernel_size")
        s = get_param(_P_POOL, data, "stride")
        return f"self.{name} = nn.{module}(kernel_size={_val_str(k)}, stride={_val_str(s)})"

    return {"handles": _HANDLES_UNSPEC, "params": _P_POOL, "init": _init, "forward": _module_forward}


# 前端: nodes/vision/pooling/Adaptive{Avg,Max}Pool2dNode.tsx
def _adaptive_pool_entry(module: str) -> dict:
    def _init(data: dict, name: str) -> str:
        out = get_param(_P_ADAPTIVE_POOL, data, "output_size")
        return f"self.{name} = nn.{module}({_val_str(out)})"

    return {
        "handles": _HANDLES_UNSPEC,
        "params": _P_ADAPTIVE_POOL,
        "init": _init,
        "forward": _module_forward,
    }


# 前端: nodes/vision/pooling/Global{Avg,Max}Pool2dNode.tsx
def _global_pool_entry(module: str) -> dict:
    def _init(data: dict, name: str) -> str:
        return f"self.{name} = nn.{module}(1)"

    return {"handles": _HANDLES_UNSPEC, "params": {}, "init": _init, "forward": _module_forward}


# ---- sequence ----
# 前端: nodes/sequence/EmbeddingNode.tsx
def _embedding_init(data: dict, name: str) -> str:
    vocab = get_param(_P_EMBEDDING, data, "num_embeddings")
    dim = get_param(_P_EMBEDDING, data, "embedding_dim")
    pad = get_param(_P_EMBEDDING, data, "padding_idx")
    pad_arg = f", padding_idx={_val_str(pad)}" if pad is not None and pad >= 0 else ""
    return (
        f"self.{name} = nn.Embedding(num_embeddings={_val_str(vocab)}, "
        f"embedding_dim={_val_str(dim)}{pad_arg})"
    )


# 前端: nodes/sequence/{RNN,LSTM,GRU}Node.tsx
def _recurrent_entry(module: str) -> dict:
    def _init(data: dict, name: str) -> str:
        input_size = get_param(_P_RECURRENT, data, "input_size")
        hidden_size = get_param(_P_RECURRENT, data, "hidden_size")
        layers = get_param(_P_RECURRENT, data, "num_layers")
        bidir = get_param(_P_RECURRENT, data, "bidirectional")
        return (
            f"self.{name} = nn.{module}(input_size={_val_str(input_size)}, "
            f"hidden_size={_val_str(hidden_size)}, num_layers={_val_str(layers)}, "
            f"bidirectional={'True' if bidir else 'False'}, batch_first=True)"
        )

    def _forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
        return f"{_out0(outputs)}, _ = self.{name}({_in0(inputs)})"

    return {
        "handles": _HANDLES_UNARY,
        "params": _P_RECURRENT,
        "init": _init,
        "forward": _forward,
    }


# 前端: nodes/sequence/MultiheadAttentionNode.tsx
def _mha_init(data: dict, name: str) -> str:
    embed = get_param(_P_MHA, data, "embed_dim")
    heads = get_param(_P_MHA, data, "num_heads")
    return (
        f"self.{name} = nn.MultiheadAttention(embed_dim={_val_str(embed)}, "
        f"num_heads={_val_str(heads)}, batch_first=True)"
    )


def _mha_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    q = _at(inputs, 0, "x")
    k = _at(inputs, 1, q)
    v = _at(inputs, 2, q)
    mask = inputs[3] if len(inputs) > 3 else None
    mask_arg = f", attn_mask={mask}" if mask else ""
    return f"{_out0(outputs)}, _ = self.{name}({q}, {k}, {v}{mask_arg})"


# 前端: nodes/sequence/PositionalEncodingNode.tsx（原样保留其 "\\n        " 连接符：
# 前端模板里是转义后的反斜杠+n 字面量，移植时不能改成真换行）
def _pos_enc_init(data: dict, name: str) -> str:
    return "# positional encoding computed in forward (sinusoidal)"


def _pos_enc_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    x = _in0(inputs)
    out = _out0(outputs)
    dim = get_param(_P_POS_ENC, data, "dim")
    arange = f"torch.arange(0, {_val_str(dim)}, 2, device={x}.device)"
    log = f"torch.log(torch.tensor(10000.0, device={x}.device))"
    lines = [
        f"seq_len = {x}.shape[1]",
        f"position = torch.arange(seq_len, device={x}.device).unsqueeze(1)",
        f"div_term = torch.exp({arange} * (-{log} / {_val_str(dim)}))",
        f"pe = torch.zeros(1, seq_len, {_val_str(dim)}, device={x}.device)",
        "pe[0, :, 0::2] = torch.sin(position * div_term)",
        "pe[0, :, 1::2] = torch.cos(position * div_term)",
        f"{out} = {x} + pe",
    ]
    return "\n        ".join(lines)


# ---- losses ----
# 前端: nodes/losses/{MSE,CrossEntropy,BCE}LossNode.tsx
def _loss_entry(module: str, first: str, handles: dict) -> dict:
    def _init(data: dict, name: str) -> str:
        return f"self.{name} = nn.{module}()"

    def _forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
        a = _at(inputs, 0, first)
        b = _at(inputs, 1, "target")
        return f"{_at(outputs, 0, 'loss')} = self.{name}({a}, {b})"

    return {"handles": handles, "params": {}, "init": _init, "forward": _forward}


# ---- metrics ----
# 前端: nodes/metrics/AccuracyNode.tsx
def _accuracy_init(data: dict, name: str) -> str:
    return "# accuracy is computed in forward"


def _accuracy_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    logits = _at(inputs, 0, "logits")
    target = _at(inputs, 1, "target")
    return f"{_at(outputs, 0, 'acc')} = (torch.argmax({logits}, dim=-1) == {target}).float().mean()"


NODE_TABLE: dict[str, dict] = {
    # ================= inputs =================
    # 前端: nodes/inputs/InputNode.tsx
    "input_layer": {
        "handles": _HANDLES_INPUT,
        "params": {},
        "init": _input_init,
        "forward": _input_forward,
    },
    # ================= torch_ops =================
    # 前端: nodes/pytorch_core/AddNode.tsx（组件 targetHandles=2，无静态 handles）
    "add_layer": {
        "handles": _HANDLES_UNSPEC_BINARY,
        "params": {},
        "init": _add_init,
        "forward": _add_forward,
    },
    # 前端: nodes/pytorch_core/ConcatNode.tsx（组件 targetHandles=2，无静态 handles）
    "concat_layer": {
        "handles": _HANDLES_UNSPEC_BINARY,
        "params": _P_CONCAT,
        "init": _concat_init,
        "forward": _concat_forward,
    },
    # 前端: nodes/pytorch_core/ElementwiseBinaryNode.tsx
    "sub_layer": _binary_entry("Sub", "-"),
    "mul_layer": _binary_entry("Mul", "*"),
    "div_layer": _binary_entry("Div", "/"),
    # 前端: nodes/pytorch_core/UnaryElementwiseNode.tsx
    "exp_layer": _unary_fn_entry("Exp", "exp"),
    "log_layer": _unary_fn_entry("Log", "log"),
    "sqrt_layer": _unary_fn_entry("Sqrt", "sqrt"),
    # 前端: nodes/pytorch_core/PowNode.tsx
    "pow_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": _P_POW,
        "init": _pow_init,
        "forward": _pow_forward,
    },
    # 前端: nodes/pytorch_core/ClipNode.tsx
    "clip_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": _P_CLIP,
        "init": _clip_init,
        "forward": _clip_forward,
    },
    # 前端: nodes/pytorch_core/MatMulNode.tsx
    "matmul_layer": {
        "handles": _HANDLES_BINARY,
        "params": {},
        "init": _matmul_init,
        "forward": _matmul_forward,
    },
    # 前端: nodes/pytorch_core/ReductionNode.tsx
    "sum_layer": _reduction_entry("Sum", "sum"),
    "mean_layer": _reduction_entry("Mean", "mean"),
    # 前端: nodes/pytorch_core/ArgExtremaNodes.tsx
    "prod_layer": _reduction_entry("Prod", "prod"),
    "max_layer": _extrema_entry("max", True),
    "min_layer": _extrema_entry("min", True),
    "argmax_layer": _extrema_entry("argmax", False),
    "argmin_layer": _extrema_entry("argmin", False),
    # ================= tensor_shape =================
    # 前端: nodes/pytorch_core/ReshapeNode.tsx
    "reshape_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": _P_RESHAPE,
        "init": _reshape_init,
        "forward": _reshape_forward,
    },
    # 前端: nodes/pytorch_core/TransposeNode.tsx
    "transpose_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": _P_TRANSPOSE,
        "init": _transpose_init,
        "forward": _transpose_forward,
    },
    # 前端: nodes/pytorch_core/FlattenNode.tsx
    "flatten_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": {},
        "init": _flatten_init,
        "forward": _module_forward,
    },
    # 前端: nodes/pytorch_core/Identity.tsx
    "pass_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": {},
        "init": _pass_init,
        "forward": _pass_forward,
    },
    # ================= tensor_create =================
    # 前端: nodes/pytorch_core/ConstantTensorNodes.tsx
    "zeros_layer": _constant_entry("zeros"),
    "ones_layer": _constant_entry("ones"),
    "rand_layer": _constant_entry("rand"),
    # ================= activations =================
    # 前端: nodes/pytorch_core/activations.tsx
    "relu_layer": _activation_entry("nn.ReLU()"),
    "leakyrelu_layer": _activation_entry("nn.LeakyReLU()"),
    "gelu_layer": _activation_entry("nn.GELU()"),
    "elu_layer": _activation_entry("nn.ELU()"),
    "selu_layer": _activation_entry("nn.SELU()"),
    "tanh_layer": _activation_entry("nn.Tanh()"),
    "sigmoid_layer": _activation_entry("nn.Sigmoid()"),
    "softplus_layer": _activation_entry("nn.Softplus()"),
    "softsign_layer": _activation_entry("nn.Softsign()"),
    "hardswish_layer": _activation_entry("nn.Hardswish()"),
    "hardsigmoid_layer": _activation_entry("nn.Hardsigmoid()"),
    # 前端: nodes/pytorch_core/SoftmaxNode.tsx
    "softmax_layer": _softmax_entry("Softmax"),
    "logsoftmax_layer": _softmax_entry("LogSoftmax"),
    # ================= normalization =================
    # 前端: nodes/pytorch_core/NormNodes.tsx
    "batchnorm2d_layer": _build_init_entry("nn.BatchNorm2d", _P_BATCHNORM, _R_BATCHNORM),
    "instancenorm2d_layer": _build_init_entry("nn.InstanceNorm2d", _P_BATCHNORM, _R_BATCHNORM),
    "groupnorm_layer": _build_init_entry("nn.GroupNorm", _P_GROUPNORM, _R_GROUPNORM),
    "layernorm_layer": _build_init_entry("nn.LayerNorm", _P_LAYERNORM, _R_LAYERNORM),
    "rmsnorm_layer": _build_init_entry("nn.RMSNorm", _P_RMSNORM, _R_LAYERNORM),
    # ================= regularization =================
    # 前端: nodes/pytorch_core/RegNodes.tsx
    "dropout_layer": _build_init_entry("nn.Dropout", _P_DROPOUT, ()),
    "spatialdropout2d_layer": _build_init_entry("nn.Dropout2d", _P_DROPOUT, ()),
    "alphadropout_layer": _build_init_entry("nn.AlphaDropout", _P_DROPOUT, ()),
    "stochasticdepth_layer": _build_init_entry(
        "nn.StochasticDepth", _P_STOCHASTIC_DEPTH, _R_STOCHASTIC_DEPTH, _HANDLES_UNARY
    ),
    # ================= dense =================
    # 前端: nodes/dense/LinearLayer.tsx
    "linear_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": _P_LINEAR,
        "init": _linear_init,
        "forward": _linear_forward,
    },
    # ================= vision_conv =================
    # 前端: nodes/vision/conv/Conv{1,2,3}dNode.tsx
    "conv1d_layer": _build_init_entry("nn.Conv1d", _P_CONV, _R_CONV),
    "conv2d_layer": _build_init_entry("nn.Conv2d", _P_CONV, _R_CONV),
    "conv3d_layer": _build_init_entry("nn.Conv3d", _P_CONV, _R_CONV),
    # 前端: nodes/vision/conv/DepthwiseConv2dNode.tsx
    "depthwiseconv2d_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": _P_DEPTHWISE,
        "init": _depthwise_init,
        "forward": _module_forward,
    },
    # 前端: nodes/vision/conv/PointwiseConv2dNode.tsx
    "pointwiseconv2d_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": _P_POINTWISE,
        "init": _pointwise_init,
        "forward": _module_forward,
    },
    # 前端: nodes/vision/conv/ConvTranspose2dNode.tsx
    "convtranspose2d_layer": _build_init_entry("nn.ConvTranspose2d", _P_CONVT, _R_CONV),
    # 前端: nodes/vision/conv/UpsampleNode.tsx
    "upsample_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": _P_UPSAMPLE,
        "init": _upsample_init,
        "forward": _module_forward,
    },
    # 前端: nodes/vision/blocks/ResidualBlock.tsx（普通层，无内部子图）
    "residual_block": {
        "handles": _HANDLES_RESIDUAL,
        "params": _P_RESIDUAL,
        "init": _residual_init,
        "forward": _residual_forward,
    },
    # ================= vision_pool =================
    # 前端: nodes/vision/pooling/{Max,Avg}Pool{1,2,3}dNode.tsx
    "maxpool1d_layer": _pool_entry("MaxPool1d"),
    "maxpool2d_layer": _pool_entry("MaxPool2d"),
    "maxpool3d_layer": _pool_entry("MaxPool3d"),
    "avgpool1d_layer": _pool_entry("AvgPool1d"),
    "avgpool2d_layer": _pool_entry("AvgPool2d"),
    "avgpool3d_layer": _pool_entry("AvgPool3d"),
    # 前端: nodes/vision/pooling/Adaptive{Avg,Max}Pool2dNode.tsx
    "adaptiveavgpool2d_layer": _adaptive_pool_entry("AdaptiveAvgPool2d"),
    "adaptivemaxpool2d_layer": _adaptive_pool_entry("AdaptiveMaxPool2d"),
    # 前端: nodes/vision/pooling/Global{Avg,Max}Pool2dNode.tsx
    "globalavgpool2d_layer": _global_pool_entry("AdaptiveAvgPool2d"),
    "globalmaxpool2d_layer": _global_pool_entry("AdaptiveMaxPool2d"),
    # ================= sequence =================
    # 前端: nodes/sequence/EmbeddingNode.tsx
    "embedding_layer": {
        "handles": _HANDLES_UNARY,
        "params": _P_EMBEDDING,
        "init": _embedding_init,
        "forward": _module_forward,
    },
    # 前端: nodes/sequence/{RNN,LSTM,GRU}Node.tsx
    "rnn_layer": _recurrent_entry("RNN"),
    "lstm_layer": _recurrent_entry("LSTM"),
    "gru_layer": _recurrent_entry("GRU"),
    # 前端: nodes/sequence/MultiheadAttentionNode.tsx
    "multihead_attention_layer": {
        "handles": _HANDLES_MHA,
        "params": _P_MHA,
        "init": _mha_init,
        "forward": _mha_forward,
    },
    # 前端: nodes/sequence/PositionalEncodingNode.tsx
    "positional_encoding_layer": {
        "handles": _HANDLES_UNSPEC,
        "params": _P_POS_ENC,
        "init": _pos_enc_init,
        "forward": _pos_enc_forward,
    },
    # ================= losses =================
    # 前端: nodes/losses/{MSE,CrossEntropy,BCE}LossNode.tsx
    "mse_loss": _loss_entry("MSELoss", "pred", _HANDLES_LOSS_PRED),
    "cross_entropy_loss": _loss_entry("CrossEntropyLoss", "logits", _HANDLES_LOSS_LOGITS),
    "bce_loss": _loss_entry("BCELoss", "pred", _HANDLES_LOSS_PRED),
    # ================= metrics =================
    # 前端: nodes/metrics/AccuracyNode.tsx
    "accuracy_metric": {
        "handles": _HANDLES_LOSS_LOGITS,
        "params": {},
        "init": _accuracy_init,
        "forward": _accuracy_forward,
    },
}


UNSUPPORTED_TYPES: dict[str, str] = {
    "repeat_layer": "控制流节点（重复块）暂不支持服务端导出",
    "module_list": "控制流节点（模块列表）暂不支持服务端导出",
}

# ---------------------------------------------------------------------------
# module_ref：内联模块包代码
# ---------------------------------------------------------------------------

def _module_class_names(module_py: str) -> list[str]:
    """module.py 中定义的类名列表（模块四再生成代码：根类在最后）。"""
    return re.findall(r"^class\s+(\w+)\s*\(", module_py, flags=re.MULTILINE)


def _rename_classes(module_py: str, mapping: dict[str, str]) -> str:
    """按映射整体替换内联模块里的类名（词边界匹配，一次扫描，避免新旧名互相误伤）。"""
    if not mapping:
        return module_py
    # 长名优先，避免 Decomp_a 抢掉 Decomp_ab 的匹配
    names = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(c) for c in names) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(0)], module_py)


def _module_ref_block(ref: str, node_id: str, data: dict) -> tuple[str, str, str]:
    """解析模块引用 → (内联代码块, 实例化类名, 排重键)。

    引用格式 module_id:module_version（saved_module_compat.id 口径）。
    """
    if ":" not in ref:
        raise ExportError(
            f"模块节点 {node_id} 引用的模块 {ref} 不在后端模块库（本地模块）——"
            "本地模块只存在于浏览器，请在模块编辑器里导出代码"
        )
    module_id, _, version = ref.partition(":")
    row = knowledge_service.get_module(module_id, version)
    if row is None:
        raise ExportError(f"模块节点 {node_id} 引用的模块 {ref} 在模块库中不存在（可能已删除）")
    # 兼容历史包：早期记录把 path 记成了 module.json 的**文件**路径（R11 修复前的产物），
    # 代码文件名也曾在 v4 才由 model.py 改名为 module.py——两者都按包目录探测，不因版本老而拒绝。
    raw_path = row.get("path")
    pkg = Path(raw_path) if raw_path else None
    if pkg is not None and pkg.is_file():
        pkg = pkg.parent
    code_path = next(
        (pkg / name for name in ("module.py", "model.py") if pkg and (pkg / name).exists()),
        None,
    )
    if code_path is None:
        raise ExportError(
            f"模块 {ref} 的代码文件缺失（包目录下既没有 module.py 也没有 model.py，path={raw_path}）")
    module_py = code_path.read_text(encoding="utf-8").strip()
    classes = _module_class_names(module_py)
    if not classes:
        raise ExportError(f"模块 {ref} 的 {code_path.name} 中没有类定义，无法内联")

    # 画布上若改过固化参数：不静默丢弃修改，拒绝导出并说明原因
    schema = json.loads(row.get("params_schema") or "{}")
    edited = [
        k for k, spec in schema.items()
        if data.get(k) is not None and data[k] != spec.get("default")
    ]
    if edited:
        raise ExportError(
            f"模块节点 {node_id}（{ref}）的参数 {', '.join(edited)} 已在画布中修改，"
            "但该模块的构造参数在拆解入库时已固化（再生成代码把参数写死在层内），"
            "导出代码无法体现这些修改——请把参数改回默认值，或走模块四链路重新拆解入库"
        )
    return module_py, classes[-1], f"{module_id}:{version}"


def _module_ref_init(node_id: str, data: dict,
                     renames_by_key: dict[str, dict[str, str]] | None = None) -> str:
    """module_ref 的 __init__ 行：实例化内联模块的根类（无参——参数已固化在模块内）。

    类名冲突时 `generate()` 会给内联类加模块键后缀，这里按同一个映射取实际类名。
    """
    ref = data.get("moduleId")
    if not ref:
        raise ExportError(f"模块节点 {node_id} 缺少 moduleId")
    _, root_class, key = _module_ref_block(str(ref), node_id, data)
    if renames_by_key:
        root_class = renames_by_key.get(key, {}).get(root_class, root_class)
    return f"self.{sanitize_ident(node_id)}_layer = {root_class}()"


def _module_ref_forward(data: dict, name: str, inputs: list[str], outputs: list[str]) -> str:
    """module_ref 的 forward 行（与前端 ModuleRefNode.getForwardCode 一致）。"""
    call_args = ", ".join(inputs) if inputs else ""
    call = f"({call_args})" if call_args else "()"
    if len(outputs) > 1:
        return f"{', '.join(outputs)} = self.{name}{call}"
    out = outputs[0] if outputs else "x"
    return f"{out} = self.{name}{call}"


# ---------------------------------------------------------------------------
# 连线编译（compileGraphToScript 移植）
# ---------------------------------------------------------------------------

def _handles_of(node_type: str, data: dict) -> dict:
    """节点句柄规格：默认单入单出；module_ref 从节点数据取。"""
    if node_type == "module_ref":
        h = data.get("handles") or {}
        inputs = h.get("inputs") or ["in"]
        outputs = h.get("outputs") or ["out"]
        return {"targets": inputs, "sources": outputs}
    spec = NODE_TABLE.get(node_type)
    return spec["handles"] if spec else {"targets": ["in"], "sources": ["out"]}


def _ordered_in_edges(in_edges: list[dict]) -> list[dict]:
    """按 `targetHandle` 的序号排序输入边（in-0 / in-1 / …）。

    不对称算子（sub/div/matmul/cross_entropy/accuracy/multihead）按**位置**消费输入；
    若按边数组顺序绑定，用户接线的句柄顺序与数组顺序不一致时操作数会颠倒。
    缺 `targetHandle` 的边（老图/无句柄）排在带序号的之后，彼此保持相对顺序。
    """

    def key(item: tuple[int, dict]) -> tuple[int, int, int]:
        idx, edge = item
        handle = edge.get("targetHandle")
        matched = re.search(r"(\d+)\s*$", handle) if isinstance(handle, str) else None
        return (0, int(matched.group(1)), idx) if matched else (1, idx, 0)

    return [edge for _, edge in sorted(enumerate(in_edges), key=key)]


def _forward_line(
    node_type: str,
    data: dict,
    nid: str,
    layer_name: str,
    incoming: dict[str, list[dict]],
    outgoing: dict[str, list[dict]],
    node_output_map: dict[str, list[str]],
) -> str:
    """单个节点的 forward 行（含多句柄节点的输出变量命名，与前端一致）。"""
    in_edges = _ordered_in_edges(incoming[nid])
    input_names = [_edge_var_name(e) for e in in_edges] if in_edges else ["x"]

    out_edges = outgoing[nid]
    handles = _handles_of(node_type, data)
    source_handles = handles.get("sources") or []
    if source_handles:
        # 先按 sourceHandle 认领出边；旧图/历史图的边可能没有 sourceHandle，此时按顺序
        # 认领一条——否则会凭空造名，与消费侧（按 label/边 id 取名）对不上，导出的 forward
        # 会引用未定义变量（与前端 compileGraphToScript 同口径）。
        pending = list(out_edges)
        output_names = []
        for idx, hid in enumerate(source_handles):
            match_idx = next((i for i, e in enumerate(pending) if e.get("sourceHandle") == hid), -1)
            if match_idx < 0 and pending:
                match_idx = 0
            if match_idx >= 0:
                output_names.append(_edge_var_name(pending.pop(match_idx)))
            else:
                output_names.append(sanitize_ident(f"out_{nid}_{hid if hid is not None else idx}"))
    elif not out_edges:
        output_names = [sanitize_ident(f"out_{nid}")]
    else:
        output_names = [_edge_var_name(e) for e in out_edges]
    node_output_map[nid] = output_names

    if node_type == "module_ref":
        return f"        {_module_ref_forward(data, layer_name, input_names, output_names)}"
    spec = NODE_TABLE[node_type]
    return f"        {spec['forward'](data, layer_name, input_names, output_names)}"


def compile_graph(
    nodes: list[dict], edges: list[dict],
    class_renames: dict[str, dict[str, str]] | None = None,
) -> tuple[list[str], list[str], str]:
    """把一张图画布编译为 (init 行, forward 行, 返回变量)。

    `class_renames`：模块键 → {原类名: 重命名后的类名}，用于内联类同名冲突时的实例化对齐。
    """
    if not nodes:
        return [], [], "x"

    # 1. 邻接表与入度（与前端一致的迭代序：节点按出现序，边按出现序）
    adj: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        if e.get("source") in adj:
            adj[e["source"]].append(e["target"])
        if e.get("target") in in_degree:
            in_degree[e["target"]] += 1

    # 2. 拓扑排序（有环兜底：未定序节点按出现序补齐）
    queue = [n["id"] for n in nodes if in_degree[n["id"]] == 0]
    sorted_ids: list[str] = []
    while queue:
        u = queue.pop(0)
        sorted_ids.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    sorted_ids += [n["id"] for n in nodes if n["id"] not in sorted_ids]
    node_map = {n["id"]: n for n in nodes}

    incoming: dict[str, list[dict]] = {n["id"]: [] for n in nodes}
    outgoing: dict[str, list[dict]] = {n["id"]: [] for n in nodes}
    for e in edges:
        if e.get("target") in incoming:
            incoming[e["target"]].append(e)
        if e.get("source") in outgoing:
            outgoing[e["source"]].append(e)

    # 3. 种子行：无入边节点的每条出边 ← 模块输入 x
    seed_lines: list[str] = []
    for n in nodes:
        if not incoming[n["id"]]:
            for e in outgoing[n["id"]]:
                seed_lines.append(f"        {_edge_var_name(e)} = x  # input passthrough")

    # 4. init / forward 行
    init_lines: list[str] = []
    forward_lines: list[str] = []
    node_output_map: dict[str, list[str]] = {}
    for node in (node_map[nid] for nid in sorted_ids):
        nid = node["id"]
        layer_name = f"{sanitize_ident(nid)}_layer"
        node_type = node.get("type")
        data = node.get("data") if isinstance(node.get("data"), dict) else {}

        if node.get("parentId"):
            raise ExportError(
                f"节点 {nid}：嵌套子节点（容器内部图）暂不支持服务端导出"
            )
        if node_type in UNSUPPORTED_TYPES:
            raise ExportError(
                f"节点 {nid}：{UNSUPPORTED_TYPES[node_type]}（类型 {node_type}）"
            )

        if node_type == "module_ref":
            init_lines.append(f"        {_module_ref_init(nid, data, class_renames)}")
            forward_lines.append(
                _forward_line(node_type, data, nid, layer_name, incoming, outgoing, node_output_map)
            )
        elif node_type in NODE_TABLE:
            init_lines.append(f"        {NODE_TABLE[node_type]['init'](data, layer_name)}")
            forward_lines.append(
                _forward_line(node_type, data, nid, layer_name, incoming, outgoing, node_output_map)
            )
        else:
            raise ExportError(
                f"节点 {nid} 的类型 {node_type} 暂不支持导出代码"
                "（拆解视图 ir 节点请走模块四的再生成代码链路）"
            )

    # 5. 返回变量：顶层无出边节点的输出（单输出直接返回，多输出返回元组）
    terminal_outputs: list[str] = []
    for nid in sorted_ids:
        if not outgoing[nid]:
            terminal_outputs.extend(node_output_map.get(nid, []))
    if len(terminal_outputs) == 1:
        return_var = terminal_outputs[0]
    elif len(terminal_outputs) > 1:
        return_var = f"({', '.join(terminal_outputs)})"
    else:
        return_var = "x"
    return init_lines, forward_lines, return_var


# ---------------------------------------------------------------------------
# 入口：GraphIR → 完整代码
# ---------------------------------------------------------------------------

def generate(graph: dict) -> str:
    """GraphIR v2 → 自包含 PyTorch 代码（class GeneratedModel）。

    输出布局与前端 recursiveCodeGenerator 一致：
    import 头 + 各被引用模块的内联代码块 + 主类。
    """
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ExportError("非法 GraphIR：需含 nodes/edges 数组")

    # 先收集模块内联块（排重、保持引用顺序），失败即整体拒绝。
    # 不同模块的内联类可能同名（模块四按 `Decomp_<节点 id>` 命名，根节点 id 常相同）——
    # 同名类会后者遮蔽前者、两个引用实例化同一个模型，故冲突时给该类名加模块键后缀。
    module_blocks: list[str] = []
    seen: set[str] = set()
    used_classes: set[str] = set()
    class_renames: dict[str, dict[str, str]] = {}
    for n in nodes:
        if n.get("type") != "module_ref":
            continue
        data = n.get("data") if isinstance(n.get("data"), dict) else {}
        ref = data.get("moduleId")
        if not ref:
            raise ExportError(f"模块节点 {n.get('id')} 缺少 moduleId")
        block, _, key = _module_ref_block(str(ref), n.get("id", "?"), data)
        if key in seen:
            continue
        seen.add(key)
        classes = _module_class_names(block)
        if any(c in used_classes for c in classes):
            mapping = {c: sanitize_ident(f"{c}_{sanitize_ident(key)}") for c in classes}
            block = _rename_classes(block, mapping)
            classes = [mapping[c] for c in classes]
            class_renames[key] = mapping
        used_classes.update(classes)
        module_blocks.append(block)

    init_lines, forward_lines, return_var = compile_graph(nodes, edges, class_renames)

    main_lines = [
        f"class {MAIN_CLASS}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        *init_lines,
        "",
        "    def forward(self, x):",
        *forward_lines,
        f"        return {return_var}",
    ]
    blocks = module_blocks + ["\n".join(main_lines)]
    return HEADER + "\n\n\n" + "\n\n\n".join(blocks) + "\n"
