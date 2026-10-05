"""模块四代码再生成引擎（模块详细设计 6.3；实施约定：后端 Python 单一确定性实现）。

按 IR 连接关系生成**自包含** PyTorch 代码（继承 nn.Module 的类），不 import 源项目——
入库后脱离原项目必须能独立运行；数值比对同时证明模块包的自包含性。
纯函数、无 IO/无 subprocess，宿主解释器内执行；同 IR 恒产出同代码。

与前端 codeCompile.ts 的关系（实施约定）：IR schema 与画布 GraphIR 不同、移植无收益，
本引擎为唯一实现，前端仅用 CodeViewer 展示返回代码。
"""
import json
import re

from app.services.ir_schema import (
    OP_BINARY, OP_WHITELIST, children_of, in_edges, incomplete_ir, nodes_by_id,
    normalize_class_name, out_edges, validate_ir,
)

HEADER = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n\n\n"
)


class IrIncompleteError(ValueError):
    """IR 不完整：拒绝生成（6.3 异常与边界），message 携带缺失项清单。"""

    def __init__(self, message):
        if isinstance(message, (list, tuple)):
            message = "；".join(message)
        super().__init__(message)


# 已知 op 类名 → 表达式模板；{inputs} 代输入变量，{dim}/{start_dim}/{end_dim}/{shape}/{dims} 代 params
_OP_TEMPLATES = {
    "add": "torch.add({inputs})",
    "sub": "torch.sub({inputs})",
    "mul": "torch.mul({inputs})",
    "div": "torch.div({inputs})",
    "matmul": "torch.matmul({inputs})",
    "bmm": "torch.bmm({inputs})",
    "cat": "torch.cat([{inputs}], dim={dim})",
    "relu": "torch.relu({inputs})",
    "sigmoid": "torch.sigmoid({inputs})",
    "tanh": "torch.tanh({inputs})",
    "softmax": "torch.softmax({inputs}, dim={dim})",
    "flatten": "torch.flatten({inputs}, start_dim={start_dim}, end_dim={end_dim})",
    "mean": "torch.mean({inputs}, dim={dim})",
    "max": "torch.max({inputs})",
    "min": "torch.min({inputs})",
    "sum": "torch.sum({inputs}, dim={dim})",
    "view": "{inputs}.view({shape})",
    "reshape": "{inputs}.reshape({shape})",
    "permute": "{inputs}.permute({dims})",
}
assert set(_OP_TEMPLATES) == set(OP_WHITELIST), "op 模板与 ir_schema.OP_WHITELIST 不同步"

# op 模板占位符的默认值（`_render_op` 与结构签名共用同一份，避免两处口径漂移）。
# 结构签名用它补齐 op 节点未写的参数：op 是表达式、模型不暴露参数，「写全默认值」与
# 「省略默认值」必须得到同一 module_id（如 flatten 的 end_dim=-1）。
OP_PARAM_DEFAULTS: dict = {"dim": 1, "start_dim": 1, "end_dim": -1, "shape": [], "dims": []}


def op_param_names(node: dict) -> list:
    """节点表达式模板用到的参数占位符名（不含 `{inputs}`）；无模板（走 code_hint）时为空。"""
    tpl = _OP_TEMPLATES.get(normalize_class_name(node.get("class_name") or ""))
    if not tpl:
        return []
    return [k for k in OP_PARAM_DEFAULTS if "{" + k + "}" in tpl]


def _py_value(v) -> str:
    """JSON 值 → Python 字面量（数组转元组，PyTorch 构造参数大多要求 tuple）。"""
    if isinstance(v, list):
        return "(" + ", ".join(_py_value(x) for x in v) + ")" if v else "()"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_py_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _render_kwargs(params: dict) -> str:
    return ", ".join(f"{k}={_py_value(v)}" for k, v in (params or {}).items())


def _leaf_name(class_name: str) -> str:
    """leaf 类名去 nn./torch.nn. 前缀（validate 已保证前缀合法）。"""
    return normalize_class_name(class_name)[len("nn."):]


def _leaf_expr(node: dict) -> str:
    """leaf 构造表达式：优先 code_hint（agent 自描述的完整表达式），否则 nn.X(**params)。

    与 ir_schema.leaf_ctor_error 的判据一致——校验放行的写法，这里必须能同样渲染出来。
    """
    hint = node.get("code_hint")
    if hint:
        return str(hint)
    return f"nn.{_leaf_name(node['class_name'])}({_render_kwargs(node.get('params') or {})})"


def _render_op(node: dict, in_vars: list[str]) -> str:
    cls = normalize_class_name(node.get("class_name") or "")
    tpl = _OP_TEMPLATES.get(cls) or node.get("code_hint")
    if not tpl:
        raise IrIncompleteError([f"op 节点 {node['id']}（{cls}）既不在白名单也无 code_hint"])
    if "{inputs}" not in tpl and "{inputs[" not in tpl:
        raise IrIncompleteError([f"op 节点 {node['id']} 的 code_hint 必须含 {{inputs}} 占位符: {tpl}"])
    if cls in OP_BINARY and len(in_vars) < 2:
        raise IrIncompleteError(
            [f"op 节点 {node['id']}（{cls}）需要至少两条入边，当前 {len(in_vars)} 条（生成的表达式调用会失败）"]
        )
    params = node.get("params") or {}

    def _p(key):
        v = params.get(key)
        return OP_PARAM_DEFAULTS[key] if v is None else v

    # `{inputs[N]}` = 第 N 个操作数（多入边时 `{inputs}` 会展开成「a, b, c」逗号串，
    # 想单独取某一个操作数必须用索引形式，如 `{inputs[0]} + {inputs[1]}`）。
    def _sub_input_index(m: "re.Match") -> str:
        i = int(m.group(1))
        if i >= len(in_vars):
            raise IrIncompleteError(
                [f"op 节点 {node['id']} 的 code_hint 用了 {{inputs[{i}]}}，但只有 {len(in_vars)} 条入边"])
        return in_vars[i]

    rendered = re.sub(r"\{inputs\[(\d+)\]\}", _sub_input_index, tpl)
    return (
        rendered.replace("{inputs}", in_vars[0] if len(in_vars) == 1 else ", ".join(in_vars))
        .replace("{dim}", _py_value(_p("dim")))
        .replace("{start_dim}", _py_value(_p("start_dim")))
        .replace("{end_dim}", _py_value(_p("end_dim")))
        .replace("{shape}", _py_value(_p("shape")))
        .replace("{dims}", _py_value(_p("dims")))
    )


def _depth(ir: dict, node_id: str) -> int:
    d, cur = 0, nodes_by_id(ir).get(node_id)
    while cur and cur.get("parent_id"):
        d += 1
        cur = nodes_by_id(ir).get(cur["parent_id"])
    return d


def _subtree_ids(ir: dict, root_id: str) -> set[str]:
    """某节点子树内全部节点 id（不含根自身）。"""
    out = set()
    for c in children_of(ir, root_id):
        out.add(c["id"])
        out |= _subtree_ids(ir, c["id"])
    return out


def _level_order(ir: dict, ids: set[str]) -> list[dict]:
    """同一层内按边拓扑序（隔离节点按声明序）排列给定节点集合。"""
    node_map = nodes_by_id(ir)
    decl_index = {n["id"]: i for i, n in enumerate(ir["nodes"])}
    edge_set = {(e["from"], e["to"]) for e in ir["edges"] if e["from"] in ids and e["to"] in ids}
    indeg = {nid: 0 for nid in ids}
    for _f, t in edge_set:
        indeg[t] += 1
    ready = sorted((nid for nid in ids if indeg[nid] == 0), key=lambda nid: decl_index[nid])
    order: list[str] = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for nid in ids:
            if (cur, nid) in edge_set:
                indeg[nid] -= 1
                if indeg[nid] == 0:
                    ready.append(nid)
                    ready.sort(key=lambda nid: decl_index[nid])
    # 兜底（有环/未定序）：按声明序补齐，保证同 IR 恒定产出同代码（set 迭代顺序不稳定）
    order += [nid for nid in sorted(ids, key=lambda x: decl_index[x]) if nid not in order]
    return [node_map[nid] for nid in order]


def _inline_init(ir: dict, node: dict) -> str:
    """container 子节点内联实例化（Sequential 参数位置，不单独注册属性——
    同一 module 实例注册两次会导致 named_modules/state_dict 重复，破坏结构比对）。"""
    nid = node["id"]
    if node["kind"] == "leaf":
        return _leaf_expr(node)
    if node["kind"] == "module":
        return f"Decomp_{nid}()"
    if node["kind"] == "container":
        gc = [c["id"] for c in children_of(ir, nid)]
        return f"nn.Sequential({', '.join(_inline_init(ir, nodes_by_id(ir)[c]) for c in gc)})"
    raise IrIncompleteError([f"container 子节点 {nid} kind 非法: {node['kind']}"])


def _subtree_sink(ir: dict, root_id: str) -> str:
    """某节点子树的「唯一输出」节点 id（规则与 _module_class 的汇点判断一致）。

    用于跨层边的合法性判定：只有「子树输出节点」的输出才等价于该子模块的输出。
    叶子/无子节点返回自身；汇点不唯一时返回自身（由调用方的汇点校验另行报错）。
    """
    kid_ids = [c["id"] for c in children_of(ir, root_id)]
    if not kid_ids:
        return root_id
    owner: dict[str, str] = {}
    for cid in kid_ids:
        owner[cid] = cid
        for nid in _subtree_ids(ir, cid):
            owner[nid] = cid
    sinks = [cid for cid in kid_ids
             if not any((ow := owner.get(e.get("to"))) is not None and ow != cid
                        for m in ({cid} | _subtree_ids(ir, cid)) for e in out_edges(ir, m))]
    return sinks[0] if len(sinks) == 1 else root_id


def _module_class(ir: dict, node: dict) -> str:
    """一个自定义 module → 一个自包含类：只实例化其**直接子节点**。

    直接子节点若是 module/container，交由 Decomp_<id>()/nn.Sequential 承载，
    其内部层级由该子节点自己的类展开——父类**不重复展开孙层**（否则嵌套子模块
    的层在父类里被二次实例化，参数量翻倍、结构比对必败）。
    """
    node_map = nodes_by_id(ir)
    children = children_of(ir, node["id"])
    child_ids = [c["id"] for c in children]
    child_set = set(child_ids)

    # 跨层边归属：任一节点 → 它所属的那个直接子节点（用于把孙层发出的边折算到子节点上）
    owner: dict[str, str] = {}
    for cid in child_ids:
        owner[cid] = cid
        for nid in _subtree_ids(ir, cid):
            owner[nid] = cid

    # container 的直接子节点内联进 Sequential：不单独实例化/不单独出 forward 行
    container_children: set[str] = set()
    for c in children:
        if c["kind"] == "container":
            container_children |= {x["id"] for x in children_of(ir, c["id"])}

    ordered = _level_order(ir, child_set)
    order_index = {n["id"]: i for i, n in enumerate(ordered)}

    init_lines, fwd_lines = [], []
    for n in ordered:
        nid = n["id"]
        if nid in container_children:
            continue
        if n["kind"] == "leaf":
            init_lines.append(f"        self.{nid} = {_leaf_expr(n)}")
        elif n["kind"] == "container":
            gc = [c["id"] for c in children_of(ir, nid)]
            gc.sort(key=lambda cid: order_index.get(cid, 0))
            init_lines.append(
                "        self.{0} = nn.Sequential({1})".format(
                    nid, ", ".join(_inline_init(ir, node_map[c]) for c in gc))
            )
        elif n["kind"] == "module":
            init_lines.append(f"        self.{nid} = Decomp_{nid}()")

    sink_cache: dict[str, str] = {}

    def _sink_of(cid: str) -> str:
        if cid not in sink_cache:
            sink_cache[cid] = _subtree_sink(ir, cid)
        return sink_cache[cid]

    # 汇点：出边全部止于自身子树内部（或模块之外）的直接子节点 = 本模块的输出。
    # 注意「孤立子节点」：既无入边也无出边（典型如定义了但 forward 未使用的 self.relu），
    # 不应算作汇点——否则真实仓库里这类模型会被判「多汇点」而无法再生成。
    def _isolated(cid: str) -> bool:
        # 「未参与数据流」= 整棵子树不与任何边相连（如定义了但 forward 未使用的 self.relu）。
        # 注意不能只看「模块内」的边：模块的输入子节点只有一条来自父模块（owner 之外）的入边，
        # 若把 owner 之外的边排除，输入子节点会被误判为孤立。
        members = {cid} | _subtree_ids(ir, cid)
        return not any(in_edges(ir, m) or out_edges(ir, m) for m in members)

    candidates = [cid for cid in child_ids
                  if cid not in container_children and not _isolated(cid)]
    if not candidates:      # 全是孤立子节点：唯一子节点视为输出
        candidates = [cid for cid in child_ids if cid not in container_children]
    sinks = []
    for cid in candidates:
        members = {cid} | _subtree_ids(ir, cid)
        consumed = any(
            (ow := owner.get(e["to"])) is not None and ow != cid
            for m in members for e in out_edges(ir, m)
        )
        if not consumed:
            sinks.append(cid)
    if len(sinks) != 1:
        raise IrIncompleteError(
            [f"module 节点 {node['id']} 的直接子节点应有唯一输出，实际汇点 {len(sinks)} 个: {sinks}"
             "（多分支输出请用 op 节点汇合）"]
        )

    for n in ordered:
        nid = n["id"]
        if nid in container_children:
            continue
        if _isolated(nid) and nid not in sinks:
            continue      # 未参与数据流的属性（如未使用的 self.relu）：只实例化，不出 forward 行
        in_vars = []
        for e in in_edges(ir, nid):
            src = e.get("from")
            ow = owner.get(src)
            if ow is None:
                in_vars.append("x")  # 模块之外的输入（含来自父层的残差）
            elif src == ow or src == _sink_of(ow):
                in_vars.append(f"var_{ow}")  # 直接子节点，或该子节点子树的输出节点
            else:
                # 孙层的「内部节点」（不是其所属子模块的输出）被当成模块输出连线：
                # 生成出来的接线必然是错的，直接拒绝而不是静默产出错误代码
                raise IrIncompleteError(
                    [f"module 节点 {node['id']} 的子节点 {nid} 有一条跨层入边来自 {src}"
                     f"（属于 {ow} 的内部节点，不是 {ow} 的输出）：请改从 {ow} 自身或其子树输出节点连边"]
                )
        if n["kind"] == "op":
            fwd_lines.append(f"        var_{nid} = {_render_op(n, in_vars)}")
        else:
            in_var = in_vars[0] if in_vars else "x"
            fwd_lines.append(f"        var_{nid} = self.{nid}({in_var})")

    return (
        f"class Decomp_{node['id']}(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "\n".join(init_lines) + "\n\n"
        "    def forward(self, x):\n"
        + "\n".join(fwd_lines) + "\n"
        f"        return var_{sinks[0]}\n"
    )


def generate(ir: dict) -> str:
    """IR → 自包含 PyTorch 代码。IR 结构错误或缺失再生成所需项 → IrIncompleteError。"""
    errors = validate_ir(ir) + incomplete_ir(ir)
    if errors:
        raise IrIncompleteError("；".join(errors))
    node_map = nodes_by_id(ir)
    if node_map[ir["root_id"]]["kind"] != "module":
        raise IrIncompleteError(
            [f"root 节点 {ir['root_id']} 应为 kind=module（当前 {node_map[ir['root_id']]['kind']}）"]
        )
    module_nodes = [n for n in ir["nodes"] if n["kind"] == "module"]
    module_nodes.sort(key=lambda n: -_depth(ir, n["id"]))
    code = HEADER + "\n\n".join(_module_class(ir, n) for n in module_nodes) + "\n"
    _check_syntax(code)
    return code


def _check_syntax(code: str) -> None:
    """生成代码必须**能编译**——否则是 IR 里某条 code_hint 拼错了（如 `{inputs}[0]` 被展开成
    逗号串、`dict(a=x, b, c)` 这类位置参数跟在关键字参数之后）。

    此前不校验：坏代码一路走到「⑤ 两步验证」才以脚本 SyntaxError 爆出，且拆解的自检
    （`ir_codegen.generate`）放行 → 白跑一轮。这里提前拦下，报出错行，让拆解能带着
    原因重试、再生成能给出可读错误。
    """
    try:
        compile(code, "<ir_codegen>", "exec")
    except SyntaxError as e:
        lines = code.splitlines()
        bad = lines[e.lineno - 1].strip() if (e.lineno and 0 < e.lineno <= len(lines)) else ""
        raise IrIncompleteError(
            [f"再生成代码存在语法错误（第 {e.lineno} 行）：{e.msg}；该行: {bad[:160]}"]
        ) from e
