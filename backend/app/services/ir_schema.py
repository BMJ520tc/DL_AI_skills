"""模块四 IR：schema、校验与规范化（模块详细设计 6.1）。

IR 是拆解模块的核心数据结构（思想与基底 graphIR 一致，7.1）。
层级树由 parent_id 隐式表达 + root_id 标入口，不双写显式树（实施约定）。

校验分两层：validate_ir 为结构硬错误（decompose 闸门）；incomplete_ir 为
再生成阻断项（regenerate/verify/ingest 闸门，6.3「IR 不完整 → 拒绝生成」）。
"""
import hashlib
import json
import keyword
import re
from typing import Optional

SCHEMA_VERSION = "1.0"

# 节点 id 将用作再生成代码的类名/属性名（Decomp_{id}、self.{id}），必须为合法标识符
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

KINDS = ("module", "leaf", "container", "op")

TASK_TYPES = (
    "classification", "regression", "segmentation", "detection",
    "generation", "embedding", "other",
)

# op 节点无需 code_hint 即可再生成的已知类名（与 ir_codegen._OP_TEMPLATES 保持一致）
OP_WHITELIST = frozenset({
    "add", "sub", "mul", "div", "matmul", "bmm", "cat",
    "relu", "sigmoid", "tanh", "softmax",
    "flatten", "mean", "max", "min", "sum",
    "view", "reshape", "permute",
})

# 需要多个输入变量的 op（入边不足 → 生成的表达式调用必失败，再生成前拦下）
OP_BINARY = frozenset({"add", "sub", "mul", "div", "matmul", "bmm"})

# 单输入 op：多一条入边就会生成 `torch.relu(a, b)` 这类运行期必错表达式，同样拦下。
# 注：max/min 不在其列——`torch.max(a, b)` 是合法的逐元素取大/取小，允许两条入边。
OP_UNARY = frozenset({"relu", "sigmoid", "tanh", "softmax", "flatten", "mean", "sum",
                      "view", "reshape", "permute"})

# leaf 白名单（6.3/6.6-2「叶子白名单直接实例化 nn.X(**params)」）→ 必填构造参数。
# 不在表内的 nn.* 类需用 code_hint 给出构造表达式，否则拒绝再生成（避免生成代码运行期 AttributeError）。
LEAF_REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "nn.Linear": ("in_features", "out_features"),
    "nn.Conv1d": ("in_channels", "out_channels", "kernel_size"),
    "nn.Conv2d": ("in_channels", "out_channels", "kernel_size"),
    "nn.Conv3d": ("in_channels", "out_channels", "kernel_size"),
    "nn.ConvTranspose1d": ("in_channels", "out_channels", "kernel_size"),
    "nn.ConvTranspose2d": ("in_channels", "out_channels", "kernel_size"),
    "nn.BatchNorm1d": ("num_features",),
    "nn.BatchNorm2d": ("num_features",),
    "nn.BatchNorm3d": ("num_features",),
    "nn.GroupNorm": ("num_groups", "num_channels"),
    "nn.LayerNorm": ("normalized_shape",),
    "nn.InstanceNorm2d": ("num_features",),
    "nn.Embedding": ("num_embeddings", "embedding_dim"),
    "nn.MaxPool1d": ("kernel_size",),
    "nn.MaxPool2d": ("kernel_size",),
    "nn.MaxPool3d": ("kernel_size",),
    "nn.AvgPool1d": ("kernel_size",),
    "nn.AvgPool2d": ("kernel_size",),
    "nn.AvgPool3d": ("kernel_size",),
    "nn.AdaptiveAvgPool1d": ("output_size",),
    "nn.AdaptiveAvgPool2d": ("output_size",),
    "nn.AdaptiveAvgPool3d": ("output_size",),
    "nn.AdaptiveMaxPool1d": ("output_size",),
    "nn.AdaptiveMaxPool2d": ("output_size",),
    "nn.Upsample": (),
    "nn.ReLU": (), "nn.ReLU6": (), "nn.LeakyReLU": (), "nn.GELU": (), "nn.SiLU": (),
    "nn.Sigmoid": (), "nn.Tanh": (), "nn.ELU": (), "nn.PReLU": (),
    "nn.Softmax": (), "nn.LogSoftmax": (),
    "nn.Dropout": (), "nn.Dropout1d": (), "nn.Dropout2d": (), "nn.Dropout3d": (),
    "nn.Flatten": (), "nn.Identity": (),
}

# op 白名单里**必填**参数（缺失会生成运行期必错的表达式，如 `x.view(())`）
OP_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "view": ("shape",),
    "reshape": ("shape",),
    "permute": ("dims",),
    "cat": ("dim",),  # 模板默认 dim=1 会静默改变语义，必须显式给出
}


def leaf_ctor_error(node: dict) -> Optional[str]:
    """leaf 节点能否安全实例化：白名单 + 必填参数齐备；None = 通过。

    给了 code_hint 就按「自描述构造表达式」放行——**再生成引擎同样会用它**
    （ir_codegen._leaf_expr 优先 code_hint），两边判据一致，避免「校验放行、生成时
    把类名截断成 nn.lock() 这类运行期必错的代码」。
    """
    if node.get("code_hint"):
        if "{params}" in str(node.get("code_hint")):
            return (f"leaf 节点 {node.get('id')} 的 code_hint 不应含 {{params}} 占位符"
                    "（leaf 请直接给出完整构造表达式，如 nn.Conv2d(3, 8, 3)）")
        return None
    cls = normalize_class_name(node.get("class_name") or "")
    if not cls.startswith("nn."):
        return f"leaf 节点 {node.get('id')} 的 class_name 应为 nn.* 形式: {node.get('class_name')}"
    required = LEAF_REQUIRED_ARGS.get(cls)
    if required is None:
        return (
            f"leaf 节点 {node.get('id')} 的 {cls} 不在叶子白名单（nn.Conv2d/Linear/BatchNorm2d/"
            "MaxPool2d/ReLU/…；如确需该层，请在 code_hint 给出构造表达式）"
        )
    params = node.get("params") or {}
    if cls == "nn.Upsample":  # size 与 scale_factor 二选一
        if not params.get("size") and not params.get("scale_factor"):
            return f"leaf 节点 {node.get('id')}（nn.Upsample）需要 size 或 scale_factor"
        return None
    absent = [k for k in required if k not in params]
    if absent:
        return f"leaf 节点 {node.get('id')}（{cls}）缺少必填构造参数: {', '.join(absent)}"
    return None

IR_SCHEMA = {
    "type": "object",
    "properties": {
        "source_file": {"type": "string", "description": "模型定义文件（相对 source 根的路径）"},
        "entry_class": {"type": "string", "description": "入口 nn.Module 子类名"},
        "task_type": {"type": "string", "enum": list(TASK_TYPES)},
        "input_spec": {
            "type": "object",
            "properties": {
                "shape": {"type": "array", "items": {"type": "integer"}},
                "dtype": {"type": "string"},
            },
        },
        "root_id": {"type": "string"},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "class_name": {"type": "string"},
                    "module_file": {"type": "string"},
                    "module_path": {"type": "string", "description": "named_modules 路径，如 layer1.0.conv1"},
                    "params": {"type": "object"},
                    "parent_id": {"type": ["string", "null"]},
                    "input_shape": {"type": ["array", "null"], "items": {"type": "integer"}},
                    "output_shape": {"type": ["array", "null"], "items": {"type": "integer"}},
                    "code_hint": {"type": ["string", "null"], "description": "op 节点内联表达式模板，输入用 {inputs} 占位"},
                    "uncertain": {"type": "boolean"},
                },
                "required": ["id", "kind", "class_name"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "tensor_shape": {"type": ["array", "null"], "items": {"type": "integer"}},
                },
                "required": ["from", "to"],
            },
        },
    },
    "required": ["source_file", "entry_class", "task_type", "root_id", "nodes", "edges"],
}

_IR_WRAPPER_KEYS = ("ir", "result", "model_ir")


def _as_ir(structured) -> Optional[dict]:
    """规整 agent 解析输出（形状漂移兼容，先例 paper_service._as_items）。

    兼容：裸 IR 对象、{"ir": {...}}/{"result": {...}} 包装、以及含 nodes 键的列表元素。
    """
    if isinstance(structured, dict):
        if "nodes" in structured and "edges" in structured:
            return structured
        for key in _IR_WRAPPER_KEYS:
            val = structured.get(key)
            if isinstance(val, dict):
                return _as_ir(val)
    if isinstance(structured, list):
        for item in structured:
            if isinstance(item, dict):
                found = _as_ir(item)
                if found:
                    return found
    return None


def nodes_by_id(ir: dict) -> dict[str, dict]:
    return {n["id"]: n for n in ir.get("nodes", [])}


def children_of(ir: dict, node_id: str) -> list[dict]:
    """某节点的子节点（parent_id 归属，保持 IR 原始顺序）。"""
    return [n for n in ir.get("nodes", []) if n.get("parent_id") == node_id]


def in_edges(ir: dict, node_id: str) -> list[dict]:
    """指向某节点的边（保持 edges 声明顺序，多输入的顺序语义由声明序决定）。

    用 .get 取键：缺 from/to 的畸形边由 validate_ir 报错，这里不应 KeyError 崩掉校验本身。
    """
    return [e for e in ir.get("edges", []) if e.get("to") == node_id]


def out_edges(ir: dict, node_id: str) -> list[dict]:
    return [e for e in ir.get("edges", []) if e.get("from") == node_id]


def normalize_class_name(name: str) -> str:
    """归一化类名：去 torch. 前缀（leaf/container 另去 nn. 前缀）。"""
    return name[len("torch."):] if name.startswith("torch.") else name


def validate_ir(ir: dict) -> list[str]:
    """结构硬错误校验（decompose 闸门），返回错误清单，空 = 通过。"""
    errors: list[str] = []
    nodes = ir.get("nodes") or []
    edges = ir.get("edges") or []

    if not nodes:
        return ["nodes 为空"]
    ids = [n.get("id") for n in nodes]
    if len(set(ids)) != len(ids):
        errors.append("节点 id 不唯一")
    id_set = set(ids)

    root_id = ir.get("root_id")
    if root_id not in id_set:
        errors.append(f"root_id 不存在: {root_id}")
    if not ir.get("entry_class"):
        errors.append("entry_class 缺失")

    for n in nodes:
        nid = n.get("id", "?")
        if not _ID_RE.match(str(nid)) or keyword.iskeyword(str(nid)):
            errors.append(f"节点 id 非法（须为 Python 标识符且非关键字，再生成代码将用作类名/变量名）: {nid}")
        params = n.get("params")
        if isinstance(params, dict):
            for key in params:
                if not _ID_RE.match(str(key)) or keyword.iskeyword(str(key)):
                    errors.append(f"节点 {nid} 的 params 键 {key!r} 不是合法关键字参数名（再生成代码将无法调用）")
        if n.get("kind") not in KINDS:
            errors.append(f"节点 {nid} kind 非法: {n.get('kind')}")
            continue
        pid = n.get("parent_id")
        if pid is not None and pid not in id_set:
            errors.append(f"节点 {nid} 的 parent_id 不存在: {pid}")

        if n["kind"] == "leaf":
            ctor_error = leaf_ctor_error(n)
            if ctor_error:
                errors.append(ctor_error)
        elif n["kind"] == "container":
            if normalize_class_name(n.get("class_name") or "") != "nn.Sequential":
                errors.append(
                    f"container 节点 {nid} 仅支持 nn.Sequential（nn.ModuleList 请展开为父模块子节点）: {n.get('class_name')}"
                )
            for c in nodes:
                if c.get("parent_id") == nid and c.get("kind") == "op":
                    errors.append(f"container 节点 {nid} 的子节点不能是 op: {c.get('id')}")
        elif n["kind"] == "module":
            if not any(c.get("parent_id") == nid for c in nodes):
                errors.append(f"module 节点 {nid} 无子节点，无法重构其内部结构")
        elif n["kind"] == "op":
            n_in = len(in_edges(ir, nid))
            # 引用了**外部输入**（`{ext:名字}`）的算子可以没有入边——它的操作数直接来自根 forward 的实参
            exts = re.findall(r"\{ext:(\w+)\}", str(n.get("code_hint") or ""))
            if exts:
                declared = {str(e.get("name")) for e in
                            ((ir.get("input_spec") or {}).get("external") or [])
                            if isinstance(e, dict) and e.get("name")}
                unknown = [x for x in exts if x not in declared]
                if unknown:
                    errors.append(
                        f"op 节点 {nid} 的 code_hint 引用了未声明的外部输入 {unknown}"
                        "（应在 input_spec.external 里声明，根类会为它加一个同名形参）")
            # 需要入边的判据 = code_hint **引用了操作数**（`{inputs…}`）。纯常量表达式
            # （如 `torch.arange(0, 1200)`、`torch.eye(4)`——从标量造张量）本来就不消费任何节点。
            hint_txt = str(n.get("code_hint") or "")
            consumes = "{inputs}" in hint_txt or "{inputs[" in hint_txt
            if not n_in and not exts and consumes:
                errors.append(f"op 节点 {nid} 至少需要一条入边")
            cls = normalize_class_name(n.get("class_name") or "")
            if cls in OP_BINARY and n_in + len(exts) < 2:
                errors.append(f"op 节点 {nid}（{cls}）需要至少两条入边，当前 {n_in} 条")
            if cls in OP_UNARY and n_in > 1:
                errors.append(f"op 节点 {nid}（{cls}）是单输入算子，当前有 {n_in} 条入边"
                              "（多输入须用 add/cat 等汇合算子）")
            params = n.get("params") or {}
            absent = [k for k in OP_REQUIRED_PARAMS.get(cls, ()) if not params.get(k)]
            if absent and not n.get("code_hint"):
                errors.append(
                    f"op 节点 {nid}（{cls}）缺少必填参数: {', '.join(absent)}"
                    "（缺失会生成运行期必错的表达式，如 x.view(())）"
                )

        # 多输入：`op` 天然多操作数；**module** 按入边序成为 forward 形参（如 MVCDecoder(cell_emb, gene_embs)）；
        # **带 code_hint 的复合叶子**同样合法——如折叠出来的
        # `nn.TransformerEncoder(src, mask, src_key_padding_mask)`（白名单叶子/容器仍是单张量）。
        if len(in_edges(ir, nid)) > 1 and n["kind"] not in ("op", "module"):
            hint_leaf = (
                n["kind"] == "leaf" and bool(n.get("code_hint"))
                and normalize_class_name(n.get("class_name") or "") not in LEAF_REQUIRED_ARGS
            )
            if not hint_leaf:
                errors.append(
                    f"非 op 节点 {nid} 有多条入边（叶子/容器按 PyTorch 单张量输入建模；"
                    "多输入请用 **module 节点**——多个输入会按入边顺序成为它的 forward 形参，"
                    "或用一个 op 节点汇合）")

    # container 子节点的边只能在同 Sequential 子节点之间：进出 Sequential 的
    # 数据流由 container 节点自身的边表达（codegen 将子节点内联进 Sequential，
    # 子节点不可被外部引用）
    for n in nodes:
        if n.get("kind") != "container":
            continue
        cids = {c.get("id") for c in nodes if c.get("parent_id") == n["id"]}
        for cid in cids:
            for e in edges:
                if e.get("to") == cid and e.get("from") not in cids:
                    errors.append(
                        f"container 子节点 {cid} 的入边必须来自同 Sequential 内"
                        f"（整体输入由 container 节点 {n['id']} 的边表达）: {e}"
                    )
                if e.get("from") == cid and e.get("to") not in cids:
                    errors.append(
                        f"container 子节点 {cid} 的出边必须指向同 Sequential 内"
                        f"（整体输出由 container 节点 {n['id']} 的边表达）: {e}"
                    )

    for e in edges:
        if not e.get("from") or not e.get("to"):
            errors.append(f"边缺少 from/to: {e}")
        elif e["from"] not in id_set or e["to"] not in id_set:
            errors.append(f"边引用不存在的节点: {e}")
        elif e["from"] == e["to"]:
            errors.append(f"边自环: {e}")

    return errors


def incomplete_ir(ir: dict) -> list[str]:
    """再生成阻断项（6.3「IR 不完整 → 拒绝生成」），返回缺失项清单，空 = 可生成。

    leaf 的完整性（白名单 + 必填构造参数）由 validate_ir 的 leaf_ctor_error 统一把关，
    此处只管 op 的自描述是否足够（uncertain 仅作提示，不单独阻断）。
    """
    items: list[str] = []
    for n in ir.get("nodes") or []:
        if n.get("kind") != "op":
            continue
        cls = normalize_class_name(n.get("class_name") or "")
        if cls not in OP_WHITELIST and not n.get("code_hint"):
            items.append(f"op 节点 {n.get('id')}（{n.get('class_name')}）既不在白名单也无 code_hint")
    return items


def external_input_conflicts(ir: dict) -> list[str]:
    """「外部输入模块」的子树里却又存在**外来入边** → 该边在再生成时被忽略（死边），返回警告清单。

    `input_spec.inputs` 列出的节点直接吃根 forward 的外部输入，其子树的数据流由该输入起头；
    若子树里某个节点还有一条来自子树**之外**的入边，再生成时那条边不会生效（模块拿的是外部输入），
    画布上画出来是误导。实测 scGPT 的 IR 有 `encoder -> val_prep`（`val_prep` 属 `value_encoder`
    的子树，而 `value_encoder` 在 `input_spec.inputs` 里）——生成代码里 `value_encoder(x1)` 直接吃 x1，
    这条边被完全忽略。**只作警告、不阻断**（结构/数值验证都只看层与数值，不看边）。
    """
    spec = ir.get("input_spec") or {}
    roots = [s for s in (spec.get("inputs") or []) if isinstance(s, str)]
    if not roots:
        return []
    nodes = ir.get("nodes") or []
    ids = {n.get("id") for n in nodes}

    def subtree(root: str) -> set:
        out = {root}
        stack = [root]
        while stack:
            cur = stack.pop()
            for n in nodes:
                if n.get("parent_id") == cur and n.get("id") not in out:
                    out.add(n["id"])
                    stack.append(n["id"])
        return out

    warns: list[str] = []
    for root in roots:
        if root not in ids:
            continue
        sub = subtree(root)
        for e in ir.get("edges") or []:
            if e.get("to") in sub and e.get("from") not in sub:
                warns.append(
                    f"外部输入模块 {root} 的子树节点 {e.get('to')} 还有一条外来入边 "
                    f"{e.get('from')} → {e.get('to')}；该边在再生成时**不生效**"
                    f"（{root} 直接吃外部输入），画布上的这根连线是误导——请删除或改接线"
                )
    return warns


def isolated_blocks(ir: dict) -> list[str]:
    """既无入边也无出边的节点（非根、且父不是容器）→ 再生成时是**死模块**，返回警告清单。

    实测 scGPT 的 IR 里 `mvc_decoder` 整棵子树孤立——真实模型里它**确实被调用**（forward hook 实测，
    输出进返回 dict），但再生成的模型 `self.mvc_decoder` 只被实例化、从不调用；而结构比对看的是
    state_dict/层序列、数值比对只比首个张量 → **两关都没发现**（2026-10-06 坐实）。
    容器（`nn.Sequential`）的子节点本来就不带边（数据流隐式），故排除。
    """
    nodes = ir.get("nodes") or []
    by_id = {n.get("id"): n for n in nodes}
    targets = {e.get("to") for e in ir.get("edges") or []}
    sources = {e.get("from") for e in ir.get("edges") or []}
    warns: list[str] = []
    for n in nodes:
        nid = n.get("id")
        if nid == ir.get("root_id") or nid in targets or nid in sources:
            continue
        parent = by_id.get(n.get("parent_id"))
        if parent is not None and parent.get("kind") == "container":
            continue
        warns.append(
            f"节点 {nid}（{n.get('class_name')}）既无入边也无出边 → 再生成时是**死模块**"
            "（会被实例化但从不调用）：请接上它的输入/输出，或删除该节点"
        )
    return warns


def canonical_ir(ir: dict) -> dict:
    """规范化副本（ir_hash 计算用）：去掉易漂移字段，键排序保证稳定。

    不含 `module_path`：它是 trace 兜底回填的定位锚点（与形状同类），若计入哈希，
    「先验证后补形状」会把验证无故变 stale（与 6.6-4「回填不产生 stale」矛盾）。
    """
    out = {k: ir.get(k) for k in
           ("schema_version", "project_id", "source_file", "entry_class", "task_type", "input_spec", "root_id")}
    # entry_args（入口类构造参数，用户补的）也要进哈希：改了它 → 旧验证 stale。
    # 仅在**存在且非空**时加入，避免给既有 IR 凭空插键、把所有历史验证打成 stale。
    if ir.get("entry_args"):
        out["entry_args"] = ir["entry_args"]
    out["nodes"] = [
        {k: n.get(k) for k in
         ("id", "kind", "class_name", "module_file", "params", "parent_id", "code_hint")}
        for n in ir.get("nodes") or []
    ]
    out["edges"] = [
        {k: e.get(k) for k in ("from", "to")} for e in ir.get("edges") or []
    ]
    return out


def ir_hash(ir: dict) -> str:
    """IR 规范化哈希（PUT 调参后变化 → 旧验证变 stale 的判据）。"""
    return hashlib.sha256(
        json.dumps(canonical_ir(ir), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
