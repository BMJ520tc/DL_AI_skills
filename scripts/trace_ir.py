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
    load_entry_class, make_dummy_input, make_extra_inputs, prepare_torch,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.ir_schema import LEAF_REQUIRED_ARGS  # noqa: E402

# 导出伪影：无真实数据流，直接跳过（注意**别把真算子写进来**——曾误把 `arange` 当伪影跳过，
# 结果位置编码整链断掉、下游算子全缺入边）
_SKIP_ATEN = ("_assert_tensor_metadata", "to.dtype_layout", "lift_fresh_copy")

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
    "unsqueeze.default": "torch.unsqueeze({a}, -1)",
    "squeeze.dim": "torch.squeeze({a}, -1)",
    "squeeze.default": "torch.squeeze({a})",
    "select.int": "torch.select({a}, 1, 0)",
    "slice.Tensor": "torch.narrow({a}, 1, 0, 1)",
    "clamp.default": "torch.clamp({a}, max=512)",
    "clamp_min.default": "torch.clamp_min({a}, 1e-12)",
    "permute.default": "torch.permute({a}, (0, 2, 1))",
    "flatten.using_ints": "torch.flatten({a}, 1)",
    "cross_entropy.default": "torch.nn.functional.cross_entropy({a}, {b})",
    "cross_entropy_loss.default": "torch.nn.functional.cross_entropy({a}, {b})",
    "cosine_similarity.default": "torch.nn.functional.cosine_similarity({a}, {b}, dim=-1)",
    "linalg_vector_norm.default": "torch.linalg.vector_norm({a}, dim=-1, keepdim=True)",
    "expand_as.default": "torch.expand_as({a}, {b})",
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
})


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


def _render_aten(n, user_inputs: list[str]) -> tuple[str, list[str]] | None:
    """按**真实实参**渲染一个 aten 调用 → `(code_hint, 用到的外部输入名)`；表达不了时 None。

    操作数是图节点 → `{inputs[j]}`（j 只数「非外部输入」的操作数，与 IR 入边顺序一致）；
    是用户输入占位 → `{ext:名}`；是标量/dtype/None/表 → 字面量；是参数/缓冲 → 表达不了。
    """
    aten = str(n.target).replace("torch.ops.", "").replace("aten.", "")
    fn = _ATEN_CALL.get(aten)
    if fn is None:
        # 方法形式的算子（`to.dtype`/`t.default`/`view.default`…）：按**方法名**回退到 `.<head>(...)`
        head = aten.split(".")[0]
        fn = ("." + head) if head in _TENSOR_METHODS else None
    if fn is None:
        return None
    parts: list[str] = []
    exts: list[str] = []
    node_idx = 0
    for a in n.args:
        if hasattr(a, "name") and hasattr(a, "op"):          # FX 节点
            if a.op == "placeholder":
                nm = str(a.name)
                if nm not in user_inputs:
                    return None                              # 参数/缓冲被当数据用 → 表达不了
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
    """给 `torch.nn` 里认得的类名（沿 MRO 回溯，避免报出内部子类名）。"""
    import torch

    for base in cls.__mro__:
        if getattr(base, "__module__", "").startswith("torch.nn") and hasattr(torch.nn, base.__name__):
            return f"nn.{base.__name__}"
    return cls.__name__


def _leaf_params(nn_name: str, mod) -> dict:
    """叶子构造参数：白名单里的必填名直接从实例属性取（nn.Linear.in_features 等）。"""
    out: dict = {}
    for key in LEAF_REQUIRED_ARGS.get(nn_name, ()):
        val = getattr(mod, key, None)
        if val is None:
            continue
        out[key] = list(val) if isinstance(val, (tuple, list)) else (
            val if isinstance(val, (int, float, bool, str)) else str(val))
    if nn_name == "nn.Dropout" or nn_name.startswith("nn.Dropout"):
        out["p"] = float(getattr(mod, "p", 0.0))
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


def build_ir(model, *, entry_class: str, source_file: str, spec: dict, task_type: str = "other") -> dict:
    """在**已实例化**的真实模型上生成 IR。返回 (ir, notes)：notes 是如实登记的「没能表达」清单。"""
    import torch

    notes: list[str] = []
    dt = getattr(torch, str(spec.get("dtype") or "float32").replace("torch.", ""), torch.float32)
    xs = (make_dummy_input(spec.get("shape") or [1, 100], dt), *make_extra_inputs(spec))
    args = accepted_positional(model.forward, xs)
    kwargs = accepted_kwargs(model.forward, call_kwargs(spec))
    # 导出本身会跑一次 forward → 顺手用 hook 捕获**真实返回值**（多输出模型的 dict 键名从这来，
    # 不必再调一次模型：重复调用在导出之后可能因实参个数解析不同而失败）。
    captured: dict = {}
    _h = model.register_forward_hook(lambda _m, _i, o: captured.__setitem__("out", o))
    try:
        ep = torch.export.export(model, args, kwargs=kwargs or None, strict=False)
    finally:
        _h.remove()
    nodes = list(ep.graph_module.graph.nodes)

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

    def _leaf_ancestor(p: str) -> bool:
        """祖先里是否有**已被折叠成叶子**的层（如 nn.TransformerEncoder）——它的子孙不再建节点。"""
        cur = p
        while "." in cur:
            cur = cur.rsplit(".", 1)[0]
            if kind_of.get(cur) == "leaf":
                return True
        return False

    for p in sorted(paths):
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
    for p in sorted(kind_of):
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
    _parents = {n.get("parent_id") for n in ir_nodes}
    dropped = {n["id"] for n in ir_nodes
               if n["kind"] == "module" and n["id"] != root_id and n["id"] not in _parents}
    if dropped:
        notes.append(f"{len(dropped)} 个 module 节点无子节点（重复调用未展开内部栈）——已删除：{sorted(dropped)[:4]}")
        ir_nodes = [n for n in ir_nodes if n["id"] not in dropped]
        for _p, _nid in list(node_by_path.items()):
            if _nid in dropped:
                node_by_path.pop(_p, None)

    # ---- 4) 模块层算子 → op 节点（叶子/容器内部的操作由它们自己承载，不再展开）----
    # **外部输入**：`torch.export` 的 graph_signature 把「参数/缓冲」记为 PARAMETER，其余即用户输入
    # （如 scGPT 的 `src`/`values`/`labels`）。op 引用它们时写 `{ext:名}`，根类会加同名形参。
    user_inputs: list[str] = []
    for s in getattr(ep.graph_signature, "input_specs", []):
        if "USER_INPUT" in str(getattr(s, "kind", "")).upper():
            nm = getattr(getattr(s, "arg", None), "name", None)
            if nm:
                user_inputs.append(str(nm))

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
    counter = 0
    seen_ops: dict[tuple, str] = {}       # (owner, aten, code_hint) → 复用的 op 节点 id
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
        counter += 1
        oid = f"op{counter}"
        hint: str | None = None
        tpl = _ATEN_EXPR.get(aten)
        if tpl is not None:
            nargs = tpl.count("{a}") + tpl.count("{b}") + tpl.count("{c}")
            args = list(n.args)
            reps: list[str] = []
            ok = len(args) >= nargs
            node_idx = 0
            for a in args[:nargs] if ok else []:
                if hasattr(a, "op") and a.op == "placeholder":
                    nm = str(a.name)
                    if nm not in user_inputs:
                        ok = False
                        break
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
                for i, ph in enumerate(("{a}", "{b}", "{c}")):
                    if ph in hint and i < len(reps):
                        hint = hint.replace(ph, reps[i])
        if hint is None:
            gen = _render_aten(n, user_inputs)
            if gen is not None:
                hint, gexts = gen
                for e in gexts:
                    if e not in used_exts:
                        used_exts.append(e)
        if hint is None:
            notes.append(f"算子 `{aten}`（{owner or 'root'} 层）无法渲染（实参不可表达或缺模板）——已跳过")
            skipped.add(id(n))
            counter -= 1
            continue
        # **同一算子被重复调用**（`unsqueeze`/`unsqueeze_13`）会产生多个图节点，但 IR 级输入完全相同
        # （同一模块、同一算子、同一组 IR 入参 → code_hint 一字不差）→ **复用同一个 op 节点**。
        # 否则下游叶子会出现多条入边（IR 判非法）、节点数也虚高。
        key = (owner, aten, hint)
        if key in seen_ops:
            oid = seen_ops[key]
            counter -= 1
        else:
            # class_name 用**完整 aten 名**（如 `div.Scalar`）：`div.Scalar` 只有一个张量操作数，
            # 不该按「二元算子（OP_BINARY 的 `div`）」校验；语义由 code_hint 承载。
            ir_nodes.append({"id": oid, "kind": "op",
                             "class_name": aten, "parent_id": node_by_path.get(owner, root_id),
                             "code_hint": hint})
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

    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    live_ids = {n["id"] for n in ir_nodes}
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
            if src.op == "placeholder":
                # `torch.export` 把**参数/缓冲**也提升成 graph 输入（placeholder）——那是权重不是数据流；
                # 真实外部输入也一样（IR 约定：无入边的节点即吃外部输入）。都不建边。
                continue
            s = ir_node_of.get(id(src))
            if s is None:
                continue
            s = _lift(s)
            if s != tgt and s in live_ids and tgt in live_ids and (s, tgt) not in seen_edges:
                seen_edges.add((s, tgt))
                edges.append({"from": s, "to": tgt})

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
        oid = "output" if all(n["id"] != "output" for n in ir_nodes) else "output_"
        ir_nodes.append({"id": oid, "kind": "op", "class_name": "build_output",
                         "parent_id": root_id, "code_hint": hint})
        for _k, t in pairs:
            edges.append({"from": t, "to": oid})

    spec_out = dict(spec)
    if used_exts:
        spec_out["external"] = [{"name": n} for n in used_exts]
    # 注：`input_spec.inputs`（root 的模块级形参）目前仍沿用参考 IR 的值——从追踪里推它还需要
    # 想清楚「模块级输入」与「外部输入(ext)」的分工（草稿版把叶子节点当成输入，反而更差，已回退）。
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
    return ir, notes


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
    model = instantiate(cls, entry_args).eval()
    ir, notes = build_ir(model, entry_class=entry_class, source_file=source_file,
                         spec=spec, task_type=task_type)
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
