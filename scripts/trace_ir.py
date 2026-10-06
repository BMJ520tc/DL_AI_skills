"""由**真实追踪**生成模块四 IR（治本：不再让 LLM 猜结构）。

为什么：LLM 产出的 IR 会漏模块（scGPT 丢 `creterion_cce`）、把模块晾成孤立死块（`mvc_decoder`），
再生成模型与真实模型对不上，而旧判据放行。追踪（`torch.export` + 节点自带 `nn_module_stack`）
能给出**按构造保真**的层级与数据流。

做法（一次导出拿到全部）：
  torch.export.export(model, 示例输入) → flat FX 图；每个节点带 `meta["nn_module_stack"]`
  （`{'': 'TransformerModel', 'encoder': 'GeneEncoder', 'encoder.embedding': 'Embedding'}`）
  → 模块树 = 栈里的路径集合；算子的**归属模块** = 栈里最深的那条路径；
  数据流边 = 节点参数引用了谁（再上抬到 IR 节点）。

节点分类：
  - `nn.*` 在叶子白名单（`ir_schema.LEAF_REQUIRED_ARGS`）→ **leaf**（参数由 `getattr(module, 名)` 取）
  - `nn.Sequential` → **container**
  - 复合 `nn.*`（如 `nn.TransformerEncoder`）→ 用**折叠模板**（`_FOLD`）写成 leaf 的 `code_hint`（按 ×N 折叠）
  - 其余（自定义 `nn.Module`）→ **module**，其子模块成为子节点
  - 归属到 **module** 层的算子 → **op** 节点（`code_hint` 由 aten 名渲染）

用法: python trace_ir.py <source_dir> <入参IR|-> <out_ir.json>
（入参 IR 只为取 `entry_class`/`source_file`/`input_spec`/`entry_args`；`-` 表示用最小默认值。）
"""
import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _model_loader import (  # noqa: E402
    accepted_kwargs, accepted_positional, call_kwargs, instantiate,
    load_entry_class, build_inputs, prepare_torch,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.ir_schema import LEAF_REQUIRED_ARGS  # noqa: E402

# 导出伪影：无真实数据流，直接跳过（注意**别把真算子写进来**——曾误把 `arange` 当伪影跳过，
# 结果位置编码整链断掉、下游算子全缺入边）
_SKIP_ATEN = ("_assert_tensor_metadata", "to.dtype_layout", "lift_fresh_copy")

# **元数据算子**：只读形状/元素个数、**不消费数值**（`aten.sym_size.int` 等）。动态批量维下
# dynamo 会把 `cos_sim.size(0)` 规范成 `src.size(0)`，于是根层出现一个「读用户输入形状」的算子——
# 两个后果：① 它会被当成「根层消费了该用户输入」→ 把 `input_spec.inputs` 里那个输入模块挤掉；
# ② `src` 在根层没有变量名 → 渲染成 `{ext:src}`，根类平白多一个错位形参。故单独处理（见 `meta_src`）。
_META_ATEN = ("sym_size", "sym_numel", "sym_stride")

# `aten.slice.Tensor` 的「到末尾」哨兵（dynamo 把 `x[:, 0, :]` 写成 `slice(dim=0, 0, INT64_MAX)`）。
_SLICE_UNBOUNDED = 2 ** 62

# 复合 nn.* 层的折叠模板：从**实例属性**拼出构造表达式（按 num_layers 之类折叠成 1 个节点）。
# 只列项目里真会遇到的；命中不了的复合层会走「展开子模块」，展开会让节点数暴涨 → 由调用方决定。
_FOLD: dict[str, str] = {
    "nn.TransformerEncoder": (
        "nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, "
        "dim_feedforward={dim_feedforward}, dropout={dropout}, batch_first=True), num_layers={num_layers})"
    ),
}

# aten 名 → 代码表达式模板（`{a}`/`{b}`… 是入参占位，生成时替换成 {inputs[N]}）
_ATEN_EXPR: dict[str, str] = {
    "add.Tensor": "torch.add({a}, {b})",
    "add.Scalar": "torch.add({a}, {b})",
    "sub.Tensor": "torch.sub({a}, {b})",
    "mul.Tensor": "torch.mul({a}, {b})",
    "div.Tensor": "torch.div({a}, {b})",
    "matmul.default": "torch.matmul({a}, {b})",
    "mm.default": "torch.mm({a}, {b})",
    "bmm.default": "torch.bmm({a}, {b})",
    "cat.default": "torch.cat([{a}, {b}], dim=0)",
    "relu.default": "torch.relu({a})",
    "sigmoid.default": "torch.sigmoid({a})",
    "tanh.default": "torch.tanh({a})",
    "softmax.int": "torch.softmax({a}, dim=-1)",
    # 标量实参**从真实实参取**（写死会让不同调用塌成同一个节点、且语义错的——实测 scGPT 的
    # `cell1.unsqueeze(1)`/`cell2.unsqueeze(0)` 曾被模板统一成 `unsqueeze(-1)`）
    "unsqueeze.default": "torch.unsqueeze({a}, {b})",
    "squeeze.dim": "torch.squeeze({a}, {b})",
    "squeeze.default": "torch.squeeze({a})",
    "select.int": "torch.select({a}, {b}, {c})",
    "slice.Tensor": "torch.narrow({a}, {b}, {c}, {d})",
    "clamp.default": "torch.clamp({a}, max=512)",
    "clamp_min.default": "torch.clamp_min({a}, 1e-12)",
    "permute.default": "torch.permute({a}, (0, 2, 1))",
    "flatten.using_ints": "torch.flatten({a}, 1)",
    "cross_entropy.default": "torch.nn.functional.cross_entropy({a}, {b})",
    "cross_entropy_loss.default": "torch.nn.functional.cross_entropy({a}, {b})",
    "cosine_similarity.default": "torch.nn.functional.cosine_similarity({a}, {b}, dim=-1)",
    "linalg_vector_norm.default": "torch.linalg.vector_norm({a}, dim=-1, keepdim=True)",
    "expand_as.default": "{a}.expand_as({b})",
    # 元数据算子（动态批量维下 dynamo 会引入）：读形状/元素个数，返回的是 **int 不是张量**
    "sym_size.int": "{a}.size({b})",
    "sym_numel.default": "{a}.numel()",
}


# aten 名 → torch 调用（以 `.` 开头表示**方法调用**，如 `.to` → `a.to(...)`）。
# 表里没有的算子走「按真实实参渲染」的通用路径（见 `_render_aten`）。
_ATEN_CALL: dict[str, str] = {
    "add.Tensor": "torch.add", "add.Scalar": "torch.add", "sub.Tensor": "torch.sub",
    "mul.Tensor": "torch.mul", "mul.Scalar": "torch.mul", "div.Tensor": "torch.div",
    "div.Scalar": "torch.div", "matmul.default": "torch.matmul", "mm.default": "torch.mm",
    "bmm.default": "torch.bmm", "cat.default": "torch.cat", "stack.default": "torch.stack",
    "relu.default": "torch.relu", "sigmoid.default": "torch.sigmoid", "tanh.default": "torch.tanh",
    "cos.default": "torch.cos", "sin.default": "torch.sin", "exp.default": "torch.exp",
    "clamp.default": "torch.clamp", "clamp_min.default": "torch.clamp_min",
    "arange.default": "torch.arange", "eye.default": "torch.eye", "zeros.default": "torch.zeros",
    "ones.default": "torch.ones", "linalg_vector_norm.default": "torch.linalg.vector_norm",
    "cosine_similarity.default": "torch.nn.functional.cosine_similarity",
    "cross_entropy_loss.default": "torch.nn.functional.cross_entropy",
    "scaled_dot_product_attention.default": "torch.nn.functional.scaled_dot_product_attention",
    "linear.default": "torch.nn.functional.linear", "softmax.int": "torch.softmax",
    # 方法调用
    ".t": ".t", ".to": ".to", ".contiguous": ".contiguous", ".view": ".view",
    ".reshape": ".reshape", ".transpose": ".transpose", ".expand": ".expand",
    ".expand_as": ".expand_as", ".unsqueeze": ".unsqueeze", ".squeeze": ".squeeze",
    ".select": ".select", ".permute": ".permute", ".flatten": ".flatten",
    # `torch.*` 函数形式（**不是** Tensor 方法——`rsub` 就没有 `Tensor.rsub`，回退成方法会 AttributeError）
    "rsub.Scalar": "torch.rsub", "rsub.Tensor": "torch.rsub",
    "remainder.Scalar": "torch.remainder", "fmod.Scalar": "torch.fmod",
    "normalize.default": "torch.nn.functional.normalize",
}


# Tensor **方法式**算子（`a.to(...)`/`a.t()`…）：aten 名带重载后缀（`to.dtype`），按 head 回退
_TENSOR_METHODS = frozenset({
    "to", "t", "contiguous", "view", "reshape", "transpose", "permute", "expand",
    "expand_as", "unsqueeze", "squeeze", "select", "flatten", "clone", "detach", "type_as",
    # 元素级/掩码/规约（实测 scGPT 位置编码块会用到 masked_fill / pow 等）
    "masked_fill", "masked_fill_", "masked_select", "pow", "log", "log2", "log10", "exp",
    "sqrt", "rsqrt", "abs", "neg", "sign", "floor", "ceil", "round",
    "eq", "ne", "lt", "gt", "le", "ge", "logical_and", "logical_or", "logical_not",
    "softmax", "log_softmax", "repeat", "roll", "flip", "triu", "tril", "diagonal",
    "norm", "addmm", "addbmm", "index_select", "gather", "scatter",
    "mean", "sum", "max", "min", "cumsum", "argsort", "sort", "topk", "argmax", "argmin",
    "chunk", "split", "unbind", "stack", "clamp_", "fill_", "zero_", "add_", "mul_",
    "float", "long", "half", "double", "type",
    "rsub", "remainder", "fmod", "clamp_max", "clamp_min", "lerp", "addcmul", "addcdiv",
}) - {"rsub"}   # `Tensor.rsub` 不存在（`torch.rsub` 才是函数）——已在 `_ATEN_CALL` 显式给出


def _call_repr(a) -> str | None:
    """FX 实参 → 代码字面量（标量/dtype/None/表）；不可表达的返回 None。"""
    if a is None:
        return "None"
    if isinstance(a, bool):
        return repr(a)
    if isinstance(a, (int, float)):
        return repr(a)
    if isinstance(a, str):
        return repr(a)
    import torch as _t

    if isinstance(a, _t.dtype):
        return f"torch.{str(a).split('.')[-1]}"
    if isinstance(a, (list, tuple)):
        parts = [_call_repr(x) for x in a]
        if any(p is None for p in parts):
            return None
        return "(" + ", ".join(parts) + ")" if isinstance(a, tuple) else "[" + ", ".join(parts) + "]"
    return None


def _render_aten(n, user_inputs: list[str], ph_module: dict[str, str],
                 op_anc: set[str], resolve) -> tuple[str, list[str]] | None:
    """按**真实实参**渲染一个 aten 调用 → `(code_hint, 用到的外部输入名)`；表达不了时 None。

    操作数是图节点 → `{inputs[j]}`（j 只数「会产生入边」的操作数，与 IR 入边顺序一致）；
    是**本模块输入**的用户占位（`ph_module` 认领且属于本模块）→ `{inputs[j]}`（由 `模块→算子` 边喂入）；
    否则是根层外部输入 → `{ext:名}`；是标量/dtype/None/表 → 字面量；是参数/缓冲 → 表达不了。
    """
    aten = str(n.target).replace("torch.ops.", "").replace("aten.", "")
    fn = _ATEN_CALL.get(aten)
    if fn is None:
        # 方法形式的算子（`to.dtype`/`t.default`/`view.default`…）：按**方法名**回退到 `.<head>(...)`。
        # 白名单之外再用 `hasattr(torch.Tensor, head)` 兜一层（真方法才回退——`rsub` 这类会踩 AttributeError）。
        import torch as _t
        head = aten.split(".")[0]
        fn = ("." + head) if (head in _TENSOR_METHODS or hasattr(_t.Tensor, head)) else None
    if fn is None:
        return None
    parts: list[str] = []
    exts: list[str] = []
    node_idx = 0
    for a in n.args:
        if hasattr(a, "name") and hasattr(a, "op"):          # FX 节点
            a = resolve(a)                                   # 透传节点（to.dtype_layout）解析到真数据源
            if a.op == "placeholder":
                nm = str(a.name)
                if nm not in user_inputs:
                    return None                              # 参数/缓冲被当数据用 → 表达不了
                tm = ph_module.get(nm)
                if tm is not None and tm in op_anc:
                    parts.append("{inputs[%d]}" % node_idx)  # 本模块的输入（由模块→算子边喂入）
                    node_idx += 1
                    continue
                parts.append(f"{{ext:{nm}}}")
                if nm not in exts:
                    exts.append(nm)
                continue
            if a.op == "get_attr":
                return None          # **模型属性**（如 `self.batch_labels`）不是数据流，IR 表达不了
            parts.append("{inputs[%d]}" % node_idx)
            node_idx += 1
            continue
        lit = _call_repr(a)
        if lit is None:
            return None
        parts.append(lit)
    kw: list[str] = []
    for k, v in n.kwargs.items():
        if k in ("device", "layout", "memory_format", "pin_memory"):   # 与语义无关，带不进生成代码
            continue
        lit = _call_repr(v)
        if lit is None:
            return None
        kw.append(f"{k}={lit}")
    if fn.startswith("."):                                   # 方法调用：第一个操作数做接收者
        if not parts:
            return None
        expr = f"{parts[0]}{fn}({', '.join(parts[1:] + kw)})"
    else:
        expr = f"{fn}({', '.join(parts + kw)})"
    return expr, exts


_CALL_SUFFIX_RE = re.compile(r"@\d+")


def _strip_calls(path: str) -> str:
    """去掉**重复调用**后缀（段级）：`value_encoder@1.linear1` → `value_encoder.linear1`。

    查真实模块（`named_modules`）时要用它；**建节点**时不用——同一模块被调用两次是两次调用，
    各建一套节点（否则同一叶子会有多条入边，IR 判非法，实测 scGPT 的 `value_encoder`/`encoder`）。
    """
    return _CALL_SUFFIX_RE.sub("", path)


def _norm_stack(stack: dict) -> list[str]:
    """`nn_module_stack` → 模块路径列表（`L__self__encoder.embedding` → `encoder.embedding`）。

    **去掉重复调用后缀 `@N`**：同一模块被调用多次时 torch 会给后一次打 `@N`（`value_encoder@1`）。
    实测这些调用**输入完全相同**（如 scGPT 两次 `value_encoder(values)`），故按**实例**建节点
    是忠实的（参数不翻倍）；重复调用产生的重复算子由**按 IR 级键复用**消化（见算子循环）。
    """
    return [k.split("__")[-1].split("@")[0] for k in (stack or {})]


def _ident(path: str, root_tag: str = "model") -> str:
    """模块路径 → IR 节点 id（合法标识符）。"""
    if not path:
        return root_tag
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in path)


def _nn_name(cls) -> str:
    """给 `torch.nn` 里认得的类名（沿 MRO 回溯，避免报出内部子类名）。

    **必须排除 `nn.Module` 本身**：用户自定义的 `nn.Module` 子类（如 scGPT 的 `Similarity`）MRO 回溯
    到的第一个「torch.nn 里有同名属性」的基类就是 `nn.Module` → 会被误判成「无参 `nn.*`」而**折进
    父层、整棵子树丢失**（实测 `Sim`/`Similarity` 因此消失 → loss_cce 分支丢）。
    """
    import torch

    for base in cls.__mro__:
        if base.__name__ in ("Module", "object"):
            continue
        if getattr(base, "__module__", "").startswith("torch.nn") and hasattr(torch.nn, base.__name__):
            return f"nn.{base.__name__}"
    return cls.__name__


# 有 `bias` 构造参数的叶子。实例上是 **Parameter 或 None**（不是构造参数那个 bool）——
# **没有偏置才写 `bias=False`**：漏掉会静默用默认 True 构造、凭空多出一个 bias 参数（实测
# scGPT 的 `MVCDecoder.W = nn.Linear(..., bias=False)` 参数量差 +512，且该随机 bias 参与前向
# → `mvc_output` 数值对不上 3.9%）。
_BIAS_LEAVES = frozenset({
    "nn.Linear", "nn.Conv1d", "nn.Conv2d", "nn.Conv3d", "nn.ConvTranspose1d", "nn.ConvTranspose2d",
})

# 卷积类：形状语义相关的属性**一律带上**（`stride`/`padding` 等实例上是 tuple，与标量默认值归一
# 比较很啰嗦，直接按实例值写全更稳）。
_CONV_ATTRS: dict[str, tuple[str, ...]] = {
    "nn.Conv1d": ("stride", "padding", "dilation", "groups"),
    "nn.Conv2d": ("stride", "padding", "dilation", "groups"),
    "nn.Conv3d": ("stride", "padding", "dilation", "groups"),
    "nn.ConvTranspose1d": ("stride", "padding", "output_padding", "dilation", "groups"),
    "nn.ConvTranspose2d": ("stride", "padding", "output_padding", "dilation", "groups"),
}

# 其余叶子：**值与默认不同才写**（写默认值只是噪音、还会让 IR 无故膨胀）。
_LEAF_OPTIONAL: dict[str, dict[str, object]] = {
    "nn.Embedding": {"padding_idx": None},
    "nn.LeakyReLU": {"negative_slope": 0.01},
    "nn.ELU": {"alpha": 1.0},
}


def _jsonable(val):
    """实例属性 → JSON 可存的值（tuple→list；不可表达的返回 None）。

    `nn.Linear.bias`/卷积 `bias` 在实例上是 **Parameter 或 None**（不是构造参数那个 bool），
    故用 `bias is not None` 判「有没有偏置」。
    """
    if val is None:
        return None
    if isinstance(val, bool) or isinstance(val, (int, float, str)):
        return val
    if isinstance(val, (tuple, list)):
        return list(val)
    return None


def _leaf_params(nn_name: str, mod) -> dict:
    """叶子构造参数：白名单里的必填名 + 非默认/形状相关的可选属性，直接从实例属性取。"""
    out: dict = {}
    for key in LEAF_REQUIRED_ARGS.get(nn_name, ()):
        val = getattr(mod, key, None)
        if val is None:
            continue
        out[key] = list(val) if isinstance(val, (tuple, list)) else (
            val if isinstance(val, (int, float, bool, str)) else str(val))
    if nn_name.startswith("nn.Dropout"):
        out["p"] = float(getattr(mod, "p", 0.0))
    if nn_name in _BIAS_LEAVES and getattr(mod, "bias", None) is None:
        out["bias"] = False
    if nn_name in _CONV_ATTRS:
        for key in _CONV_ATTRS[nn_name]:
            val = _jsonable(getattr(mod, key, None))
            if val is not None:
                out[key] = val
        return out
    for key, default in _LEAF_OPTIONAL.get(nn_name, {}).items():
        if key == "padding_idx":
            if isinstance(getattr(mod, "padding_idx", None), int):
                out[key] = getattr(mod, "padding_idx")
            continue
        val = _jsonable(getattr(mod, key, None))
        if val is not None and val != default:
            out[key] = val
    return out


def _fold_hint(nn_name: str, mod) -> str | None:
    """复合层折叠成 code_hint（如 nn.TransformerEncoder(...) 一条）。取不到必要属性就返回 None。"""
    tpl = _FOLD.get(nn_name)
    if not tpl:
        return None
    try:
        if nn_name == "nn.TransformerEncoder":
            layers = list(mod.layers)
            layer = layers[0]
            attn = layer.self_attn
            vals = {
                "d_model": getattr(attn, "embed_dim", None),
                "nhead": getattr(attn, "num_heads", None),
                "dim_feedforward": getattr(layer.linear1, "out_features", None),
                "dropout": float(getattr(layer.dropout, "p", 0.0)),
                "num_layers": len(layers),
            }
            if any(v is None for v in vals.values()):
                return None
            return tpl.format(**vals)
    except Exception:  # noqa: BLE001
        return None
    return None


def _with_batch(spec: dict, batch: int) -> dict:
    """spec 的**张量入参 dim0 全换成** `batch`（主形状 + `extra`）——动态批量维下的示例输入用。"""
    s = dict(spec)
    sh = list(s.get("shape") or [])
    if sh:
        s["shape"] = [batch, *sh[1:]]
    if s.get("extra"):
        s["extra"] = [{**e, "shape": [batch, *list(e.get("shape") or [1])[1:]]} for e in s["extra"]]
    return s


def _dynamic_shapes(model, args: tuple, kwargs: dict) -> dict | None:
    """每个**张量**实参的 dim0 → 同一个符号 `batch`；非张量（bool 开关等）→ None。

    dict 形态必须列出**全部实参名**（含 kwargs），否则 `torch.export` 直接 UserError。
    形参名按 `model.forward` 的签名取（位置实参按序命名）。
    """
    import inspect

    import torch
    from torch.export import Dim

    try:
        names = [p for p in inspect.signature(model.forward).parameters if p != "self"][:len(args)]
    except (TypeError, ValueError):
        return None
    if len(names) != len(args):
        return None
    dyn: dict = {}
    for nm, val in zip(names, args):
        dyn[nm] = {0: Dim("batch")} if isinstance(val, torch.Tensor) and val.dim() > 0 else None
    for nm in kwargs:
        dyn[nm] = None
    return dyn


def _export_traced(model, spec: dict, kwargs: dict, notes: list[str],
                   source_dir: str | None = None):
    """导出：**优先把批量维声明为动态**，失败则回退静态（并登记原因）。

    为什么要动态：`labels = torch.arange(cos_sim.size(0))`、`mask = torch.eye(cos_sim.size(0))` 这类
    **数据依赖**的常量，在静态导出下会被**用示例批量维折叠掉**（`arange(1)`/`eye(1)`）→ 生成的模型
    批量维被写死：换 batch 直接崩（实测 scGPT 导出模型 batch>1 报 `cross_entropy` 尺寸不符），
    且 batch=1 时 `loss_cce`/`loss_ecs` 恒为常数（退化）。声明动态后图表里留 `sym_size`，
    IR 渲染成 `x0.size(0)`，模型对任意 batch 成立。
    两条实测口径：① dynamo 会**特化常数 1**（示例批量=1 必失败）→ dummy 用 **2**；
                  ② 模型若把 dim0 当别的语义（各入参 dim0 不一致）会导出失败 → **回退静态**。
    """
    import torch

    def _export(spec_use: dict, dynamic: bool):
        dt_ = getattr(torch, str(spec_use.get("dtype") or "float32").replace("torch.", ""), torch.float32)
        xs = build_inputs(model, spec_use, source_dir)
        a = accepted_positional(model.forward, xs)
        dyn = _dynamic_shapes(model, a, kwargs) if dynamic else None
        if dynamic and dyn is None:
            return None
        return torch.export.export(model, a, kwargs=kwargs or None, strict=False, dynamic_shapes=dyn)

    try:
        ep = _export(_with_batch(spec, 2), True)
        if ep is not None:
            notes.append("导出用**动态批量维**（dim0 符号化）：IR 对任意 batch 成立")
            return ep
    except Exception as e:  # noqa: BLE001 —— 动态失败很常见（模型把 dim0 当别的语义）→ 回退静态
        notes.append(f"动态批量维导出失败，回退静态（批量维会被示例值写死）：{type(e).__name__}: {str(e)[:160]}")
    return _export(spec, False)


def build_ir(model, *, entry_class: str, source_file: str, spec: dict, task_type: str = "other",
             entry_args: dict | None = None, source_dir: str | None = None) -> dict:
    """在**已实例化**的真实模型上生成 IR。返回 (ir, notes)：notes 是如实登记的「没能表达」清单。"""
    import torch

    notes: list[str] = []
    kwargs = accepted_kwargs(model.forward, call_kwargs(spec))
    # 导出本身会跑一次 forward → 顺手用 hook 捕获**真实返回值**（多输出模型的 dict 键名从这来，
    # 不必再调一次模型：重复调用在导出之后可能因实参个数解析不同而失败）。
    captured: dict = {}
    _h = model.register_forward_hook(lambda _m, _i, o: captured.__setitem__("out", o))
    try:
        ep = _export_traced(model, spec, kwargs, notes, source_dir)
    finally:
        _h.remove()
    nodes = list(ep.graph_module.graph.nodes)

    # **透传**节点：`arange(...).long()` 这类会被导出成 `to.dtype` → `to.dtype_layout`（dtype 已相同，
    # 只是布局归一的**导出伪影**）。它在 `_SKIP_ATEN` 里、不建节点，但它的**消费者**若直接拿它当操作数
    # 就会凭空缺一个入边 → 整条分支被当碎片删掉（实测 scGPT 的 `cross_entropy(cos_sim, labels)` 与
    # `masked_fill(mm, mask)` 因此丢了 CCE/ECS 两条输出分支）。故把它当**别名**解析到真正的数据源。
    passthrough: dict[int, object] = {}
    for n in nodes:
        if "to.dtype_layout" in str(n.target):
            ins = [a for a in n.args if hasattr(a, "op")]
            if len(ins) == 1:
                passthrough[id(n)] = ins[0]

    def resolve(node):
        seen: set[int] = set()
        while id(node) in passthrough and id(node) not in seen:
            seen.add(id(node))
            node = passthrough[id(node)]
        return node

    # **外部输入**：`torch.export` 的 graph_signature 把「参数/缓冲」记为 PARAMETER，其余即用户输入
    # （如 scGPT 的 `src`/`values`/`src_key_padding_mask`）。顺序即 forward 形参顺序。
    user_inputs: list[str] = []
    for s in getattr(ep.graph_signature, "input_specs", []):
        if "USER_INPUT" in str(getattr(s, "kind", "")).upper():
            nm = getattr(getattr(s, "arg", None), "name", None)
            if nm:
                user_inputs.append(str(nm))

    # ---- 1) 模块树：从 nn_module_stack 收集路径 ----
    paths: set[str] = set()
    owner_of: dict[int, str] = {}
    for n in nodes:
        stack = _norm_stack(n.meta.get("nn_module_stack"))
        owner = stack[-1] if stack else ""
        owner_of[id(n)] = owner
        for p in stack:
            if p:
                paths.add(p)

    # ---- 2) 分类 ----
    root_id = "model"
    kind_of: dict[str, str] = {"": "module"}
    class_of: dict[str, str] = {"": entry_class}
    mods = dict(model.named_modules())
    # 节点顺序**按原模型的 `named_modules()` 顺序**（即声明/实例化序）——IR 节点顺序决定再生成模型的
    # 子模块顺序 → `state_dict` 顺序 → ⑤ 结构比对**按位置**对齐与**按位置拷权重**。按名称字典序排
    # （`sorted(paths)`）会让顺序与原模型不符（实测 `enc` 排到了 `head` 前面），权重逐位置错配。
    _mod_order = {p: i for i, (p, _m) in enumerate(model.named_modules())}

    def _order_key(p: str) -> int:
        return _mod_order.get(_strip_calls(p), len(_mod_order))

    def _leaf_ancestor(p: str) -> bool:
        """祖先里是否有**已被折叠成叶子**的层（如 nn.TransformerEncoder）——它的子孙不再建节点。"""
        cur = p
        while "." in cur:
            cur = cur.rsplit(".", 1)[0]
            if kind_of.get(cur) == "leaf":
                return True
        return False

    for p in sorted(paths, key=_order_key):
        if _leaf_ancestor(p):
            continue
        sub = mods.get(_strip_calls(p))       # 节点身份带 `@N`，查模块要用归一化路径
        if sub is None:
            continue
        nn_name = _nn_name(type(sub))
        if nn_name == "nn.Sequential":
            kind_of[p], class_of[p] = "container", "nn.Sequential"
        elif nn_name in LEAF_REQUIRED_ARGS:
            kind_of[p], class_of[p] = "leaf", nn_name
        elif _fold_hint(nn_name, sub) is not None:
            kind_of[p], class_of[p] = "leaf", nn_name          # 折叠成 leaf + code_hint
        elif nn_name.startswith("nn.") and not list(sub.parameters()) and not list(sub.buffers()):
            # **无参的 nn.\* 模块**（如 nn.CrossEntropyLoss / nn.CosineSimilarity）当**函数**用：
            # 折进父层——它的算子落在父模块的 forward 里，这样它才可能引用根类的外部输入
            # （如 `F.cross_entropy(cos_sim, labels)`；若建成子模块，`labels` 得穿过子模块 forward，
            # 而 IR 不允许叶子/容器多输入）。不建节点。
            continue
        else:
            kind_of[p], class_of[p] = "module", type(sub).__name__

    def nearest_ir_path(p: str) -> str:
        """把任意模块路径上抬到最近的**已建节点**路径（叶子/容器/模块都算）。"""
        cur = p
        while cur and cur not in kind_of:
            cur = cur.rsplit(".", 1)[0] if "." in cur else ""
        return cur

    def _parent_path(raw: str) -> str:
        """子节点的父路径。torch 只给**被重入的那一段**打 `@N`（`encoder.embedding@1`），
        父段不带——所以末段带 `@N` 时要挂到**同一次调用的父**（`encoder@1`）上，
        否则两次调用（`encoder`/`encoder@1`）的子节点会混到同一个父下面、把父模块搞成多汇点。"""
        if "." not in raw:
            return ""
        head, _, tail = raw.rpartition(".")
        m = re.search(r"@(\d+)$", tail)
        if m and f"{head}@{m.group(1)}" in kind_of:
            return f"{head}@{m.group(1)}"
        return head

    # ---- 3) IR 节点 ----
    ir_nodes: list[dict] = []
    node_by_path: dict[str, str] = {"": root_id}
    ir_nodes.append({"id": root_id, "kind": "module", "class_name": entry_class,
                     "parent_id": None, "module_path": ""})
    for p in sorted(kind_of, key=_order_key):
        if p == "":
            continue
        nid = _ident(p)
        parent = nearest_ir_path(_parent_path(p))
        node: dict = {"id": nid, "kind": kind_of[p], "class_name": class_of[p],
                      "parent_id": node_by_path.get(parent, root_id),
                      "module_path": _strip_calls(p)}   # 覆盖比对用归一化路径
        sub = mods.get(_strip_calls(p))
        if kind_of[p] == "leaf":
            hint = _fold_hint(class_of[p], sub) if class_of[p] not in LEAF_REQUIRED_ARGS else None
            if hint:
                node["code_hint"] = hint
            else:
                node["params"] = _leaf_params(class_of[p], sub)
        ir_nodes.append(node)
        node_by_path[p] = nid

    # 折叠后仍**没有子节点**的 module 节点：实测同一模块的**第二次调用**，torch 不再展开其内部
    # 模块栈（`encoder@1` 直接承载算子）→ 无法重构内部结构。删除并如实登记（其下游会因此缺入边）。
    # **但**「有算子归属」的模块不算空——它的子节点是**算子**、在第 4 段才创建（实测 scGPT 的
    # `sim`(Similarity) 子模块全是无参 `nn.CosineSimilarity`（折进父层不建节点），此处被误删 →
    # 整条 CCE 分支（loss_cce）丢失）。
    _op_owners = set()
    for _n in nodes:
        if _n.op != "call_function":
            continue
        _aten = str(_n.target).replace("torch.ops.", "").replace("aten.", "")
        if any(s in _aten for s in _SKIP_ATEN):
            continue
        _op_owners.add(node_by_path.get(nearest_ir_path(owner_of[id(_n)])))
    _parents = {n.get("parent_id") for n in ir_nodes}
    dropped = {n["id"] for n in ir_nodes
               if n["kind"] == "module" and n["id"] != root_id and n["id"] not in _parents
               and n["id"] not in _op_owners}
    if dropped:
        notes.append(f"{len(dropped)} 个 module 节点无子节点（重复调用未展开内部栈）——已删除：{sorted(dropped)[:4]}")
        ir_nodes = [n for n in ir_nodes if n["id"] not in dropped]
        for _p, _nid in list(node_by_path.items()):
            if _nid in dropped:
                node_by_path.pop(_p, None)

    # ---- 3.5) 输入接线：用户输入 placeholder → 承载它的**顶层模块** ----
    # 真实 forward 的外部输入（`src`/`values`…）在导出图里是 placeholder，被某个模块子树的算子消费。
    # 承载它的**顶层 module 节点**（root 的直接子节点）就是这份输入的模块级入口 → 写进
    # `input_spec.inputs`，根类据此把它作为形参按序传给该模块；模块内部的算子把它当**自己的输入**
    # 用（`{inputs[N]}` + `模块→算子` 边，见第 4/5 段），**不再写 `{ext:…}`**——那样在子模块的
    # forward 里是未定义名（实测 scGPT 的 `Decomp_value_encoder.forward` 出现裸 `values`）。
    # 若某 placeholder 还被**根层/折叠叶子**消费（无顶层模块可承载），则整份按外部输入处理
    # （`{ext:…}` + `input_spec.external`），与单输入模型的既有行为一致。
    node_by_id = {n["id"]: n for n in ir_nodes}
    _chain_cache: dict[str, list[str]] = {}

    def _owner_chain(owner_path: str) -> list[str]:
        """模块路径 → 从最近 IR 节点起的祖先 id 链（自身在前、root 在末）。"""
        if owner_path not in _chain_cache:
            chain: list[str] = []
            cur = nearest_ir_path(owner_path)
            while cur:
                nid = node_by_path.get(cur)
                if nid:
                    chain.append(nid)
                cur = _parent_path(cur)
            _chain_cache[owner_path] = chain
        return _chain_cache[owner_path]

    def _top_module(owner_path: str) -> str | None:
        """消费点所属的顶层 module 节点（root 的直接子节点）；消费在根层/折叠叶子内则 None。"""
        for nid in _owner_chain(owner_path):
            n = node_by_id.get(nid)
            if n is not None and n.get("parent_id") == root_id and n["kind"] == "module":
                return nid
        return None

    ph_module: dict[str, str] = {}          # placeholder → 承载它的顶层模块 id
    ph_at_root: set[str] = set()            # 还被根层消费的 placeholder（整份按外部输入处理）
    ph_conflict: dict[str, str] = {}
    for n in nodes:
        if n.op in ("placeholder", "output"):
            continue
        if any(k in str(n.target) for k in _META_ATEN):
            continue      # 元数据算子只读形状、不消费数值 → 不算「根层消费了该用户输入」
        for a in list(n.args) + list(n.kwargs.values()):
            if not (hasattr(a, "op") and a.op == "placeholder"):
                continue
            nm = str(a.name)
            if nm not in user_inputs:
                continue
            tm = _top_module(owner_of[id(n)])
            if tm is None:
                ph_at_root.add(nm)
            elif nm not in ph_module:
                ph_module[nm] = tm
            elif ph_module[nm] != tm:
                ph_conflict[nm] = f"{ph_module[nm]} / {tm}"
    for nm in ph_at_root:
        ph_module.pop(nm, None)
    # `input_spec.inputs`：按 forward 形参顺序列出承载外部输入的顶层模块（去重保序）
    input_mods: list[str] = []
    for nm in user_inputs:
        tm = ph_module.get(nm)
        if tm and tm not in input_mods:
            input_mods.append(tm)
    for nm, why in ph_conflict.items():
        notes.append(f"外部输入 `{nm}` 被多个顶层模块消费（{why}）——按首个模块接线，其余引用可能落空")
    if input_mods:
        notes.append("输入接线：" + "、".join(f"`{nm}`→{ph_module[nm]}" for nm in user_inputs if nm in ph_module))
    _unwired = [nm for nm in user_inputs if nm not in ph_module and nm not in ph_at_root]
    if _unwired:
        notes.append(f"用户输入 {_unwired} 只被折叠叶子/未建模处消费——未接模块输入（其算子不参与再生成）")

    # 每个 IR 节点的祖先 id 集（判「某模块是不是该节点的祖先」用）。**按需算**——op 节点在其后的
    # 第 4 段才创建，预先算好的快照会漏掉它们（曾因此漏建 `模块→算子` 的输入边、算子被当碎片删）。
    def _anc_of(nid: str) -> set[str]:
        pmap = {x["id"]: x.get("parent_id") for x in ir_nodes}
        s: set[str] = set()
        cur = pmap.get(nid)
        while cur:
            s.add(cur)
            cur = pmap.get(cur)
        return s

    # **元数据算子的操作数改接**：`src.size(0)`（动态批量维下 dynamo 把 `cos_sim.size(0)` 规范成它）
    # 里的 `src` 是**用户输入 placeholder**，在根层没有变量名。改接**承载该输入的模块节点**——
    # `var_encoder.size(0)` 与 `x0.size(0)` 批量维一致，且根层本来就有这个变量。
    meta_src: dict[int, str] = {}          # 图节点 id → 操作数应取的 IR 节点 id
    for n in nodes:
        if n.op != "call_function" or not any(k in str(n.target) for k in _META_ATEN):
            continue
        ph = [a for a in n.args if hasattr(a, "op") and a.op == "placeholder"]
        if len(ph) == 1:
            tm = ph_module.get(str(ph[0].name))
            if tm:
                meta_src[id(n)] = node_by_path[tm]

    # ---- 4) 模块层算子 → op 节点（叶子/容器内部的操作由它们自己承载，不再展开）----
    op_of_grapnode: dict[int, str] = {}
    skipped: set[int] = set()
    used_exts: list[str] = []          # 实际被算子引用的外部输入（只声明用到的）
    # 只把「归属模块**确有其节点**」的图节点映射到 IR 节点；其余（已删除的模块里的算子）**不建边**
    # ——否则会被错挂到根上、造出假边。
    ir_node_of: dict[int, str] = {}
    for _n in nodes:
        _pid = node_by_path.get(nearest_ir_path(owner_of[id(_n)]))
        if _pid is not None:
            ir_node_of[id(_n)] = _pid
    seen_ops: dict[tuple, str] = {}       # (owner, aten, code_hint) → 复用的 op 节点 id
    used_ids: set[str] = {n["id"] for n in ir_nodes}   # id 全局唯一（模块/叶子/容器先建、算子后建）

    def _op_id(owner_path: str, aten: str) -> str:
        """算子节点 id：`<归属模块末段>_<算子头>`（如 `value_encoder_unsqueeze`），重名加 `_2`/`_3`。

        原来是 `op1`/`op12`——在查看器/画布上读不出内容，而这些 id 还会成为再生成代码的变量名。
        """
        seg = owner_path.rsplit(".", 1)[-1] if owner_path else "root"
        base = re.sub(r"[^0-9A-Za-z_]", "_", f"{seg}_{aten.split('.')[0]}")
        if not base or base[0].isdigit():
            base = "op_" + base
        oid, k = base, 1
        while oid in used_ids:
            k += 1
            oid = f"{base}_{k}"
        used_ids.add(oid)
        return oid

    for n in nodes:
        if n.op != "call_function":
            continue
        aten = str(n.target).replace("torch.ops.", "").replace("aten.", "")
        owner = nearest_ir_path(owner_of[id(n)])
        if kind_of.get(owner) != "module" or owner not in node_by_path:
            continue                                  # 叶子/容器内部、或已删除的模块 → 不建算子
        if any(s in aten for s in _SKIP_ATEN):
            skipped.add(id(n))
            continue
        # 渲染路径：**先内置模板**（可读性好），不行再走**按真实实参的通用渲染**。
        # 两者都失败 → 跳过该算子并如实登记（它的下游会因此缺入边，下游问题随之暴露）。
        hint: str | None = None
        # 本算子的祖先 id 集（含其归属模块）：判「某模块是不是它的祖先」——即该模块的输入是否直喂它
        owner_id = node_by_path[owner]
        op_anc = {owner_id} | _anc_of(owner_id)
        if id(n) in meta_src:
            # 元数据算子、且操作数是**用户输入 placeholder**（dynamo 把 `cos.size(0)` 规范成了
            # `src.size(0)`）：该 placeholder 在根层没有变量名，操作数已改接成**承载它的模块节点**
            # （见 `meta_src` 的注释）→ 直接渲染 `{inputs[0]}.size(dim)`。
            # 操作数是普通中间张量的情形由下面的 `_ATEN_EXPR["sym_size.int"]` 模板覆盖。
            dim = _call_repr(n.args[1]) if len(n.args) > 1 else "0"
            hint = "{inputs[0]}.numel()" if "numel" in aten else "{inputs[0]}.size(%s)" % dim
            oid = _op_id(owner, aten)
            ir_nodes.append({"id": oid, "kind": "op", "class_name": aten,
                             "parent_id": node_by_path.get(owner, root_id), "code_hint": hint})
            op_of_grapnode[id(n)] = oid
            ir_node_of[id(n)] = oid
            continue
        if aten == "slice.Tensor":
            # `aten.slice.Tensor(self, dim, start, end, step)`。两种情形分开处理：
            # ① **整维切片**（start=0 且 end 是「到末尾」哨兵）是**恒等** —— 动态形状下 dynamo 会把
            #    `layer[:, 0, :]` 写成 `slice(dim=0, 0, INT64_MAX)` + `select(1, 0)`；照 `torch.narrow`
            #    渲染会拼出 `narrow(x, 0, 0, INT64_MAX)`，运行期 `start+length exceeds dimension size`。
            #    故当**别名**跳过（消费者直接接到被切的那个张量）。
            # ② 其余：`torch.narrow(a, dim, start, end-start)`（`narrow` 的第 4 参是**长度**，不是 end）。
            _a = list(n.args)
            if _a and hasattr(_a[0], "op"):
                _rest = [x for x in _a[1:] if not hasattr(x, "op")]
                _dim = _rest[0] if _rest and isinstance(_rest[0], int) else 0
                _start = _rest[1] if len(_rest) > 1 and isinstance(_rest[1], int) else 0
                _end = _rest[2] if len(_rest) > 2 else None
                if _start == 0 and isinstance(_end, int) and _end >= _SLICE_UNBOUNDED:
                    passthrough[id(n)] = _a[0]
                    skipped.add(id(n))          # 不建节点（同 `to.dtype_layout`：`ir_node_of` 会被剔除）
                    continue
                if isinstance(_end, int) and _end > _start:
                    hint = "torch.narrow({inputs[0]}, %d, %d, %d)" % (_dim, _start, _end - _start)
        if hint is not None:
            oid = _op_id(owner, aten)
            ir_nodes.append({"id": oid, "kind": "op", "class_name": aten,
                             "parent_id": node_by_path.get(owner, root_id), "code_hint": hint})
            op_of_grapnode[id(n)] = oid
            ir_node_of[id(n)] = oid
            continue
        tpl = _ATEN_EXPR.get(aten)
        if tpl is not None:
            nargs = sum(tpl.count(f"{{{c}}}") for c in "abcde")
            args = list(n.args)
            reps: list[str] = []
            ok = len(args) >= nargs
            node_idx = 0
            for a in args[:nargs] if ok else []:
                if hasattr(a, "op"):
                    a = resolve(a)
                if hasattr(a, "op") and a.op == "placeholder":
                    nm = str(a.name)
                    if nm not in user_inputs:
                        ok = False
                        break
                    tm = ph_module.get(nm)
                    if tm is not None and tm in op_anc:
                        reps.append("{inputs[%d]}" % node_idx)   # 本模块的输入
                        node_idx += 1
                        continue
                    reps.append(f"{{ext:{nm}}}")
                    if nm not in used_exts:
                        used_exts.append(nm)
                    continue
                if hasattr(a, "op") and a.op == "get_attr":
                    ok = False               # 模型属性（非数据流）→ 该算子表达不了
                    break
                if hasattr(a, "op"):
                    reps.append("{inputs[%d]}" % node_idx)
                    node_idx += 1
                    continue
                lit = _call_repr(a)
                if lit is None:
                    ok = False
                    break
                reps.append(lit)
            if ok:
                hint = tpl
                for i, ph in enumerate(("{a}", "{b}", "{c}", "{d}", "{e}")):
                    if ph in hint and i < len(reps):
                        hint = hint.replace(ph, reps[i])
        if hint is None:
            gen = _render_aten(n, user_inputs, ph_module, op_anc, resolve)
            if gen is not None:
                hint, gexts = gen
                for e in gexts:
                    if e not in used_exts:
                        used_exts.append(e)
        if hint is None:
            notes.append(f"算子 `{aten}`（{owner or 'root'} 层）无法渲染（实参不可表达或缺模板）——已跳过")
            skipped.add(id(n))
            continue
        # **同一算子被重复调用**（`unsqueeze`/`unsqueeze_13`）会产生多个图节点，但 IR 级输入完全相同
        # （同一模块、同一算子、同一组 IR 入参 → code_hint 一字不差）→ **复用同一个 op 节点**。
        # 否则下游叶子会出现多条入边（IR 判非法）、节点数也虚高。
        key = (owner, aten, hint)
        if key in seen_ops:
            oid = seen_ops[key]
        else:
            # class_name 用**完整 aten 名**（如 `div.Scalar`）：`div.Scalar` 只有一个张量操作数，
            # 不该按「二元算子（OP_BINARY 的 `div`）」校验；语义由 code_hint 承载。
            oid = _op_id(owner, aten)
            node = {"id": oid, "kind": "op", "class_name": aten,
                    "parent_id": node_by_path.get(owner, root_id), "code_hint": hint}
            # 归属模块**自己没有节点**时（无参 `nn.*` 折进父层当函数用，如 `nn.CrossEntropyLoss`），
            # 把**真实模块路径**记在该 op 上：保真度自检与补形状都按 `module_path` 查「真实调用到的
            # 模块是否被 IR 覆盖」，不记就会被误判成「漏拆 creterion_cce」而拒收（实测踩到）。
            raw_owner = owner_of[id(n)]
            if raw_owner and raw_owner != owner:
                node["module_path"] = _strip_calls(raw_owner)
            ir_nodes.append(node)
            seen_ops[key] = oid
        op_of_grapnode[id(n)] = oid
        ir_node_of[id(n)] = oid          # 该图节点今后代表这个 op

    # ---- 5) 边：图节点参数引用了谁（上抬到 IR 节点）----
    # 没建节点的算子（未收录/导出伪影）**不建边**——否则会把边错挂到它所在的模块上
    # （实测造出 `model → transformer_encoder` 这类假边）。
    for k in list(ir_node_of):
        if k in skipped:
            ir_node_of.pop(k, None)

    # 容器（`nn.Sequential`）的子节点不与外界连边：整体输入/输出挂在**容器**节点上（validate_ir 的容器规则）。
    parent_of = {n["id"]: n.get("parent_id") for n in ir_nodes}
    kind_by_id = {n["id"]: n["kind"] for n in ir_nodes}

    def _lift(nid: str) -> str:
        cur, p = nid, parent_of.get(nid)
        while p and kind_by_id.get(p) == "container":
            cur, p = p, parent_of.get(p)
        return cur

    def _top_child(nid: str) -> str:
        """节点所属的**顶层子节点**（root 的直接子节点；自己就是则返回自己，root 返回 root）。"""
        cur = nid
        while True:
            p = parent_of.get(cur)
            if not p or p == root_id:
                return cur
            cur = p

    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    live_ids = {n["id"] for n in ir_nodes}
    # 元数据算子（`sym_size`）在 IR 里的节点 id —— 它们的值**不是张量**，不能喂进折叠叶子/容器
    meta_ids = {ir_node_of[k] for k in meta_src if k in ir_node_of}

    def _emit(s: str, t: str) -> None:
        if s != t and s in live_ids and t in live_ids and (s, t) not in seen_edges:
            seen_edges.add((s, t))
            edges.append({"from": s, "to": t})

    _meta_leaf_skips: set[str] = set()      # 被忽略的「元数据 → 折叠层」边的目标（循环后统一登记）
    for n in nodes:
        if n.op in ("placeholder", "output"):
            continue
        tgt = ir_node_of.get(id(n))
        if tgt is None:
            continue
        tgt = _lift(tgt)
        for src in n.args:
            if not hasattr(src, "name"):
                continue
            if id(n) in meta_src:
                # 元数据算子：操作数固定改接承载该输入的模块节点（见 meta_src 的注释）
                _emit(_lift(meta_src[id(n)]), tgt)
                break
            if hasattr(src, "op"):
                src = resolve(src)      # 透传节点 → 真数据源（否则消费者缺入边）
            if src.op == "placeholder":
                # `torch.export` 把**参数/缓冲**也提升成 graph 输入（placeholder）——那是权重不是数据流；
                # 但**用户输入**若由消费点所属的顶层模块承载，则补一条 `模块 → 消费节点` 的边
                # （codegen 据 `_ext_var` 把该模块的输入形参喂给它；未连边 = 该模块的输入未被使用）。
                nm = str(src.name)
                tm = ph_module.get(nm)
                if tm is None:
                    continue
                if tm not in _anc_of(tgt):
                    continue
                s = tm
            else:
                s = ir_node_of.get(id(src))
                if s is None:
                    continue
            s = _lift(s)
            # **元数据值流进折叠叶子/容器**（如融合的 `_transformer_encoder_layer_fwd` 内部要用形状）：
            # 叶子是**整体实例化**的黑盒（code_hint 一条构造式），多喂一个 int 会让 `self.X(...)` 多一个
            # 参数、运行期直接报错。这类边不建（叶子内部本就未建模）。
            if s in meta_ids and kind_by_id.get(tgt) in ("leaf", "container"):
                _meta_leaf_skips.add(tgt)
                continue
            # 跨「顶层子节点」的边：
            # - **目标是模块**（内层节点）：`_module_class` 只在**直接子节点**间接线，故必须补一条
            #   `源模块 → 目标模块` 的边（追踪得到的边挂在最深消费节点：`transformer_encoder →
            #   decoder_fc`，而 root 需要 `transformer_encoder → decoder`）。目标模块**多输入**时
            #   （如 scGPT 的 `MVCDecoder(cell_emb, gene_embs)`），内部节点还要知道「自己吃哪个参数」
            #   → 再保留一条 `源模块 → 该内部节点` 的路由边（codegen 的 `_ext_var` 按来源在
            #   `sources` 里的位置取形参）。
            # - **目标不是模块**（根层算子/叶子）：**原样保留**。上抬会把模块内的汇点算子（如 `Sim`
            #   里的 `cosine_similarity`）变成「IR 级无人消费」→ 被断头清理误删；保持原样则由
            #   codegen 的 `_sink_of` 解析成该模块的输出变量（内部源只会是模块汇点）。
            st, tt = _top_child(s), _top_child(tgt)
            if st == tt:
                _emit(s, tgt)
            elif kind_by_id.get(tt) == "module":
                _emit(st, tt)
                _emit(st, tgt)
            else:
                _emit(s, tgt)
    if _meta_leaf_skips:
        notes.append(f"元数据算子（sym_size）的值流进折叠层 {sorted(_meta_leaf_skips)}——该边已忽略"
                     "（叶子内部未建模）")

    # ---- 6) 多输出模型（如 scGPT 返回 `dict(mlm_output=…, cell_emb=…, …)`）----
    # root 会有**多个汇点**，而 IR 要求模块唯一汇点 → 合成一个「装配输出」算子把各分支汇成
    # **真实返回结构**（键名取自真实返回值，不是猜的）。
    out_keys: list[str] | None = None
    _real = captured.get("out")
    if isinstance(_real, dict) and len(_real) > 1:
        out_keys = [str(k) for k in _real.keys() if str(k).isidentifier()]
    elif isinstance(_real, (tuple, list)) and len(_real) > 1:
        out_keys = [f"out{i}" for i in range(len(_real))]
    # 真实返回的**每个输出**对应哪个 IR 节点（键名 + 产出节点），装配算子在清理之后按存活分支建
    def _flatten_out(a) -> list:
        """output 节点的参数可能是**嵌套 dict/list**（dict 返回时 export 保留键结构）——
        按**插入顺序**展平成张量节点序列，才能与真实返回的 key 顺序对齐。"""
        if hasattr(a, "name"):
            return [a]
        if isinstance(a, dict):
            got: list = []
            for v in a.values():
                got.extend(_flatten_out(v))
            return got
        if isinstance(a, (list, tuple)):
            got = []
            for v in a:
                got.extend(_flatten_out(v))
            return got
        return []

    o_node = next((x for x in nodes if x.op == "output"), None)
    _arg_seq = (o_node.args[0] if (o_node is not None and o_node.args) else [])
    # **产出图的输出**的 IR 节点：它们天然「没有被消费」，**永不算断头**（单输出模型也一样）
    out_producers = [t for t in (ir_node_of.get(id(a))
                                 for a in _flatten_out(_arg_seq) if hasattr(a, "name")) if t]
    pairs: list[tuple[str, str]] = []
    if out_keys:
        if len(out_keys) != len(out_producers):
            notes.append(f"真实返回 {len(out_keys)} 个输出、图里解析出 {len(out_producers)} 个——按较短对齐")
        pairs = list(zip(out_keys, out_producers))[:min(len(out_keys), len(out_producers))]

    # ---- 7) 两轮清理（都源于「重复调用被删」留下的碎片）----
    live = {n["id"] for n in ir_nodes}

    def _indeg(es: list[dict]) -> dict[str, int]:
        d: dict[str, int] = {}
        for e in es:
            if e.get("to") in live:
                d[e["to"]] = d.get(e["to"], 0) + 1
        return d

    # 7a) **断头**算子：输出没有任何消费者（会被算作模块汇点 → 「多汇点」报错）
    consumed = {e["from"] for e in edges}
    keep = set(out_producers) | {t for _k, t in pairs}   # 图输出/装配分支**不算断头**
    dead = {n["id"] for n in ir_nodes
            if n["kind"] == "op" and n["id"] not in consumed and n["id"] not in keep}
    # 7b) **缺操作数**算子：code_hint 引用的 `{inputs[N]}` 超过实际入边数（某个上游被丢掉了）
    deg = _indeg(edges)
    for n in ir_nodes:
        if n["kind"] != "op" or n["id"] in dead:
            continue
        idxs = [int(m) for m in re.findall(r"\{inputs\[(\d+)\]\}", str(n.get("code_hint") or ""))]
        if idxs and max(idxs) >= deg.get(n["id"], 0):
            dead.add(n["id"])
    # 迭代到**不动点**：删掉一批碎片后，下游可能又变成「缺操作数」——级联清理。
    dropped_total: set[str] = set()
    for _ in range(8):
        if not dead:
            break
        dropped_total |= dead
        ir_nodes = [n for n in ir_nodes if n["id"] not in dead]
        edges = [e for e in edges if e["from"] not in dead and e["to"] not in dead]
        live = {n["id"] for n in ir_nodes}
        pairs = [(k, t) for k, t in pairs if t in live]
        keep = {t for _k, t in pairs}
        consumed = {e["from"] for e in edges}
        dead = {n["id"] for n in ir_nodes
                if n["kind"] == "op" and n["id"] not in consumed and n["id"] not in keep}
        deg = _indeg(edges)
        for n in ir_nodes:
            if n["kind"] != "op" or n["id"] in dead:
                continue
            idxs = [int(m) for m in re.findall(r"\{inputs\[(\d+)\]\}", str(n.get("code_hint") or ""))]
            if idxs and max(idxs) >= deg.get(n["id"], 0):
                dead.add(n["id"])
    if dropped_total:
        notes.append(f"删除 {len(dropped_total)} 个碎片算子（断头/缺操作数，级联）：{sorted(dropped_total)[:5]}")

    # ---- 8) 装配输出算子：把存活的多输出分支汇成**真实返回结构**（root 因此唯一汇点）----
    if len(pairs) > 1:
        hint = "dict(" + ", ".join(f"{k}={{inputs[{i}]}}" for i, (k, _) in enumerate(pairs)) + ")"
        oid = "output" if "output" not in used_ids else "output_"
        ir_nodes.append({"id": oid, "kind": "op", "class_name": "build_output",
                         "parent_id": root_id, "code_hint": hint})
        for _k, t in pairs:
            edges.append({"from": t, "to": oid})

    spec_out = dict(spec)
    # `inputs`（root 的模块级形参）：**由追踪推导**——按 forward 形参顺序列出承载外部输入的顶层模块。
    # 推导不出（没有任何输入落在模块内，如单输入模型的输入只在根层被消费）时保留参考 IR 的值/缺省，
    # 此时 root 只有默认形参 `x`，`{ext:…}` 与 no-in-edge 子节点都落到它上面，行为与既有单输入模型一致。
    if input_mods:
        spec_out["inputs"] = input_mods
    if used_exts:
        spec_out["external"] = [{"name": n} for n in used_exts]
    else:
        spec_out.pop("external", None)
    ir = {
        "schema_version": "1.0",
        "source_file": source_file,
        "entry_class": entry_class,
        "task_type": task_type,
        "input_spec": spec_out,
        "root_id": root_id,
        "nodes": ir_nodes,
        "edges": edges,
    }
    # 入口类构造参数（用户补的）**必须带过**：它决定原模型怎么实例化（`verify`/`trace` 都靠它），
    # 也在 `ir_hash` 里——漏了会让验证脚本无法构造原模型。
    if entry_args:
        ir["entry_args"] = entry_args
    return ir, notes


def _load_ctors(source_dir: str) -> dict[str, str]:
    """读 `<ws>/reports/module_ctors.py` 的 CTORS（缺文件/缺 CTORS → 空表 = 不折叠）。"""
    path = Path(source_dir).resolve().parent / "reports" / "module_ctors.py"
    if not path.exists():
        return {}
    import importlib.util

    spec_mod = importlib.util.spec_from_file_location("_module_ctors", path)
    if spec_mod is None or spec_mod.loader is None:
        return {}
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    ctors = getattr(mod, "CTORS", None)
    return {str(k): str(v) for k, v in ctors.items()} if isinstance(ctors, dict) else {}


def _output_is_tensor(model, spec: dict, source_dir: str | None) -> dict[str, bool]:
    """跑一次前向，记录每个顶层子模块的输出**是否为单个张量**（hook，失败返回空表）。

    黑盒节点在 IR 里代表「一路张量」，而真实子模块的 forward 可能返回**容器**
    （实测 scGPT 的 `ExprDecoder.forward` 返回 `dict(pred=…)`）——展开版在追踪时把这层
    拆掉了，黑盒直接调真实类就会把整个 dict 交给下游。这类模块**不能黑盒**。
    """
    import torch

    import _model_loader as ml

    try:
        inputs = ml.build_inputs(model, _with_batch(spec, 2), source_dir)
        kwargs = ml.call_kwargs(spec)
    except Exception:  # noqa: BLE001 —— 探测失败就不限制（宁可保守展开）
        return {}
    seen: dict[str, bool] = {}
    handles = []
    for name, child in model.named_children():
        def _hook(_m, _a, out, _n=name):  # noqa: ANN001
            seen.setdefault(_n, isinstance(out, torch.Tensor))

        handles.append(child.register_forward_hook(_hook))
    try:
        with torch.no_grad():
            model(*inputs, **kwargs)
    except Exception:  # noqa: BLE001
        return {}
    finally:
        for h in handles:
            h.remove()
    return seen


def collapse_blackboxes(ir: dict, ctors: dict[str, str],
                        tensor_out: dict[str, bool] | None = None) -> tuple[dict, list[str]]:
    """把**有构造契约的顶层子模块**整棵子树折叠成单个「黑盒」leaf 节点。

    为什么：追踪展开到叶子会让真实模型的 IR 过大（画布不可读、节点数撞上限）。而
    「这个子模块怎么造」这件事，`<ws>/reports/module_ctors.py` 已经给出了**可验证**的
    构造表达式（`scripts/module_ctors_check.py` 逐项比对类名/子模块序列/参数量）。
    两者一合：把展开的子树换成一个 `leaf` + `code_hint` = 契约表达式，IR 既小又忠实。

    折叠规则：
    - 只折叠 **root 的直接子节点**、且其 `module_path` 在契约里（嵌套模块不折——它的
      构造表达式无从验证）；
    - 折叠 = 删掉该子节点的**整棵子树**，在原位留一个 `leaf` 节点（id/class_name/
      module_path 保留，`code_hint` = 契约表达式）；
    - 边重接：跨子树边折到黑盒上，子树内部的边删除；去重、去自环；
    - 多入边对「带 code_hint 的 leaf」合法（回退后按入边序传参），不额外处理。
    """
    notes: list[str] = []
    if not ctors:
        return ir, notes
    root_id = ir["root_id"]
    nodes, edges = ir["nodes"], ir["edges"]
    kids: dict[str, list[str]] = {}
    for n in nodes:
        if n.get("parent_id"):
            kids.setdefault(n["parent_id"], []).append(n["id"])

    def _subtree(nid: str) -> set[str]:
        out: set[str] = set()
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(kids.get(cur, []))
        return out

    new_nodes: list[dict] = []
    fold: dict[str, str] = {}          # 被折叠掉的节点 id → 黑盒节点 id
    indeg: dict[str, int] = {}
    for e in edges:
        indeg[e["to"]] = indeg.get(e["to"], 0) + 1
    for n in nodes:
        if n["id"] in fold:
            continue
        is_top = n.get("parent_id") == root_id
        mod_path = str(n.get("module_path") or "")
        if is_top and mod_path in ctors and indeg.get(n["id"], 0) > 1:
            # **多输入模块不黑盒**：IR 的入边顺序来自「内部消费序」，而黑盒是直接调用
            # 真实类——真实 `forward` 的**位置实参顺序**数据流里没有（实测 scGPT 的
            # `MVCDecoder.forward(cell_emb, gene_embs)` 与入边序正好相反，黑盒传反 →
            # `bmm` 报 batch1 must be a 3D tensor）。宁可展开，不可传错。
            notes.append(f"黑盒跳过 {n['id']}（{n.get('class_name')}）：多输入"
                         f"（{indeg.get(n['id'], 0)} 条入边）——位置实参顺序无法从数据流推出")
            new_nodes.append(n)          # 保持展开（节点本身必须留下，否则其子节点成孤儿）
            continue
        if is_top and mod_path in ctors and (tensor_out or {}).get(n["id"]) is False:
            # **输出不是单个张量**（dict/tuple）——黑盒代表的是「一路张量」，展开版在追踪时
            # 已把这层容器拆掉（如 scGPT 的 `ExprDecoder` 返回 `dict(pred=…)`，原模型取 `["pred"]`）。
            notes.append(f"黑盒跳过 {n['id']}（{n.get('class_name')}）：forward 返回容器而非单个张量")
            new_nodes.append(n)
            continue
        if is_top and mod_path in ctors:
            sub = _subtree(n["id"])
            for sid in sub - {n["id"]}:
                fold[sid] = n["id"]
            new_nodes.append({**n, "kind": "leaf", "code_hint": ctors[mod_path]})
            notes.append(f"黑盒折叠 {n['id']}（{n.get('class_name')}）：{len(sub)} 个节点 → 1 个")
            continue
        new_nodes.append(n)
    if not fold:
        return ir, notes

    live = {n["id"] for n in new_nodes}
    seen: set[tuple[str, str]] = set()
    new_edges: list[dict] = []
    for e in edges:
        f, t = fold.get(e["from"], e["from"]), fold.get(e["to"], e["to"])
        if f not in live or t not in live or f == t or (f, t) in seen:
            continue
        seen.add((f, t))
        new_edges.append({**e, "from": f, "to": t})

    out = {**ir, "nodes": new_nodes, "edges": new_edges}
    if isinstance(out.get("input_spec"), dict):
        spec = dict(out["input_spec"])
        # 被折叠的顶层模块若在 `inputs` 里，它的 id 仍然存在（就是黑盒那个节点），无需改
        out["input_spec"] = spec
    return out, notes


def main() -> None:
    source_dir, ir_in, out_path = sys.argv[1:4]
    spec = {"shape": [1, 4], "dtype": "float32"}
    entry_class, source_file, entry_args, task_type = "Net", "model.py", None, "other"
    if ir_in != "-":
        ref = json.loads(Path(ir_in).read_text(encoding="utf-8"))
        spec = ref.get("input_spec") or spec
        entry_class = ref.get("entry_class") or entry_class
        source_file = ref.get("source_file") or source_file
        entry_args = ref.get("entry_args")
        task_type = ref.get("task_type") or task_type

    prepare_torch()
    cls = load_entry_class(source_dir, source_file, entry_class)
    model = instantiate(cls, entry_args, source_dir).eval()
    ir, notes = build_ir(model, entry_class=entry_class, source_file=source_file,
                         spec=spec, task_type=task_type, entry_args=entry_args,
                         source_dir=source_dir)
    # 顶层「黑盒折叠」：有构造契约的顶层子模块折成一个 leaf+code_hint（IR 从展开态变小）。
    # 契约由 `<ws>/reports/module_ctors.py` 给出、由 `scripts/module_ctors_check.py` 验证。
    ctors = _load_ctors(source_dir)
    ir, fold_notes = collapse_blackboxes(
        ir, ctors, _output_is_tensor(model, spec, source_dir) if ctors else None)
    notes.extend(fold_notes)
    Path(out_path).write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written: {out_path}（{len(ir['nodes'])} 节点 / {len(ir['edges'])} 边）")
    for x in notes[:20]:
        print("  note:", x)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
