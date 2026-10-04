"""静态结构扫描（模块详细设计 3.6，T4 固定脚本）。

识别入口脚本、模型定义文件（nn.Module 子类）、训练/推理流程、模块层级与第三方依赖，
不确定点（动态构建/条件分支）标 uncertain，交由 agent 补充（3.7）。

层级与调用链口径（3.6 步骤 3「类嵌套、调用链」，全部基于 AST 静态信息，拿不到就留空/标 source）:
- `module_hierarchy`（原字段，保持不变）: 每个 nn.Module 子类一条 {file, class, parent}，
  parent=直接基类名（继承关系）。
- `module_tree`: 组合嵌套树。类内 `self.x = SomeClass(...)` 是父子边；名字解析到本仓库的
  nn.Module 子类才建子节点（同名优先同文件，跨文件同名歧义不猜），nn./torch. 前缀记
  source=builtin、其余记 source=unknown 且无子节点；`nn.Sequential/ModuleList/ModuleDict`
  的直接元素调用（列表/元组的元素、dict 的值）作为它的子节点（带 element_index）；
  条件分支里的赋目标 conditional=true；自引用标 recursive、超深（MAX_TREE_DEPTH）标 truncated。
- `call_chain`: 调用**顺序**链（basis=ast_call_order），不是数据流依赖链——数据流仍需 agent/动态追踪。
  forward 链收 `self.<attr>(...)`（module_call，能解析出该属性对应的仓库类时带 resolved_class）
  与点号调用（call，如 F.relu / torch.optim.Adam）；训练/推理入口函数链额外收裸名调用里能解析到
  本仓库 nn.Module 子类的**实例化**（kind=instantiate）与本函数内由该类赋值出的局部变量的调用
  （kind=instance_call，如 `model = Net(8)` 后的 `model(x)`）。裸名其他调用（print/len 等）不入链；
  顺序按源码求值顺序（参数的调用先于外层调用，分支体内按分支体顺序），不代表运行期必然发生；
  assign_to 标该调用所在赋值语句的左值（语句级归属，非数据流归属）。

用法: python scan_structure.py <source_dir> <output.json>
"""
import ast
import json
import re
import sys
from pathlib import Path

EXCLUDE_DIRS = {"venv", ".venv", "env", ".git", "node_modules", "__pycache__", ".idea", "build", "dist", "data"}
ENTRY_HINTS = ("train", "main", "run", "eval", "infer", "test", "predict")
TRAIN_HINTS = ("train", "fit")
INFER_HINTS = ("eval", "infer", "predict", "valid", "test")
DYNAMIC_FUNCS = {"getattr", "eval", "exec", "type", "__import__"}
# module_tree 递归展开深度上限（防自引用/超深嵌套把报告撑爆）
MAX_TREE_DEPTH = 6
# 容器类调用：其直接元素调用作为子节点展开
CONTAINER_CALLS = ("nn.Sequential", "Sequential", "nn.ModuleList", "ModuleList",
                   "nn.ModuleDict", "ModuleDict")
BUILTIN_CALL_ROOTS = ("nn", "torch", "torchvision", "F", "functional")


def _conditional_module_attrs(class_node: ast.ClassDef) -> list[str]:
    """条件/循环/异常分支里的 `self.x = <Call>(...)` 属性名。

    这类模块在构造期是否真的被赋值取决于运行时条件（如 `if stride != 1: self.shortcut = ...`），
    静态等价描述不可靠 → 交 agent 动态补充（需求一.3「条件分支等」）。
    """
    out: list[str] = []

    def scan(stmts: list, in_branch: bool) -> None:
        for st in stmts:
            if isinstance(st, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                scan(st.body, True)
                scan(getattr(st, "orelse", []), True)
            elif isinstance(st, ast.Try):
                scan(st.body, True)
                for handler in st.handlers:
                    scan(handler.body, True)
                scan(st.finalbody, True)
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if st.name == "__init__":
                    scan(st.body, in_branch)  # 构造期赋值就在 __init__ 里，继续按当前条件上下文扫
                continue
            elif isinstance(st, ast.ClassDef):
                continue
            elif isinstance(st, ast.Assign) and in_branch:
                for target in st.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and isinstance(st.value, ast.Call)):
                        out.append(target.attr)

    scan(class_node.body, False)
    return out


def _parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
    except SyntaxError:
        return None


def _is_module_class(node: ast.ClassDef) -> bool:
    """识别 nn.Module 子类，兼容 nn.Module / torch.nn.Module / t.nn.Module /
    from torch.nn import Module 后的裸 Module。"""
    for base in node.bases:
        if isinstance(base, ast.Attribute) and base.attr == "Module":
            v = base.value
            if isinstance(v, ast.Name) and v.id == "nn":
                return True
            if isinstance(v, ast.Attribute) and v.attr == "nn":
                return True
        if isinstance(base, ast.Name) and base.id == "Module":
            return True
    return False


def _parent_class(node: ast.ClassDef):
    for base in node.bases:
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
    return None


def _is_entry(path: Path, tree: ast.AST) -> bool:
    if path.stem.lower() in ENTRY_HINTS:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            t = node.test
            if isinstance(t.left, ast.Name) and t.left.id == "__name__":
                if any(isinstance(c, ast.Constant) and c.value == "__main__" for c in t.comparators):
                    return True
    return False


def _top_module(name: str) -> str:
    return name.split(".")[0]


# ---- 层级树（module_tree）与调用顺序链（call_chain）：纯 AST 静态信息，不臆造 ----


def _dotted_name(node: ast.AST) -> "str | None":
    """AST 里的点号名（Name / Attribute / Call 的 func）；取不到返回 None（不猜）。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return None


def _self_attr(name: "str | None") -> "str | None":
    """`self.<attr>` → attr；其余（含 self.a.b 之类）返回 None。"""
    if name and name.startswith("self.") and name.count(".") == 1:
        return name.split(".", 1)[1]
    return None


def _is_builtin_callee(callee: "str | None") -> bool:
    return bool(callee) and callee.split(".")[0] in BUILTIN_CALL_ROOTS


def _container_elements(call: ast.Call, attr: str, conditional: bool) -> list[dict]:
    """容器调用（Sequential/ModuleList/ModuleDict）的直接元素调用 → 子条目。

    只取**直接**的 Call 元素（`Block(...)`）；列表推导/循环变量等动态写法不猜。
    element_index 是元素在容器里的位置（跨参数与 dict 值连续计数）。
    """
    out: list[dict] = []

    def add(item: ast.AST) -> None:
        if isinstance(item, ast.Call):
            out.append({"attr": attr, "callee": _dotted_name(item.func),
                        "line": getattr(item, "lineno", None),
                        "element_index": len(out), "conditional": conditional, "elements": []})

    for arg in call.args:
        items = arg.elts if isinstance(arg, (ast.List, ast.Tuple, ast.Set)) else [arg]
        for item in items:
            add(item)
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Dict):
            for val in kw.value.values:
                add(val)
    return out


def _instantiation_entries(class_node: ast.ClassDef) -> list[dict]:
    """类内 `self.<attr> = <Call>(...)`（含条件分支，标 conditional）→ 实例化条目。

    entry: {attr, callee, line, conditional, elements}；容器调用把直接元素调用放进 elements。
    动态写法（getattr/add_module/循环变量/列表推导）不猜，交由 uncertain 与 agent。
    """
    out: list[dict] = []

    def add(attr: str, call: ast.Call, conditional: bool) -> None:
        callee = _dotted_name(call.func)
        entry = {"attr": attr, "callee": callee, "line": getattr(call, "lineno", None),
                 "conditional": conditional, "elements": []}
        if callee in CONTAINER_CALLS:
            entry["elements"] = _container_elements(call, attr, conditional)
        out.append(entry)

    def scan(stmts: list, conditional: bool) -> None:
        for st in stmts:
            if isinstance(st, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                scan(st.body, True)
                scan(getattr(st, "orelse", []), True)
            elif isinstance(st, ast.Try):
                scan(st.body, True)
                for handler in st.handlers:
                    scan(handler.body, True)
                scan(st.finalbody, True)
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if st.name == "__init__":
                    scan(st.body, conditional)   # 构造期赋值就在 __init__ 里
                continue
            elif isinstance(st, ast.ClassDef):
                continue
            elif isinstance(st, ast.Assign):
                for target in st.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and isinstance(st.value, ast.Call)):
                        add(target.attr, st.value, conditional)
            elif isinstance(st, ast.AnnAssign):
                if (isinstance(st.target, ast.Attribute)
                        and isinstance(st.target.value, ast.Name)
                        and st.target.value.id == "self"
                        and isinstance(st.value, ast.Call)):
                    add(st.target.attr, st.value, conditional)

    scan(class_node.body, False)
    return out


def _find_class_node(callee: "str | None", file: str, class_nodes: list[dict]) -> "dict | None":
    """把被调用名解析到本仓库的 nn.Module 子类节点：同名优先同文件，跨文件同名歧义返回 None。"""
    if not callee:
        return None
    name = callee.split(".")[-1]
    candidates = [c for c in class_nodes if c["class"] == name]
    if not candidates:
        return None
    same_file = [c for c in candidates if c["file"] == file]
    if same_file:
        return same_file[0]
    return candidates[0] if len(candidates) == 1 else None


def _node_for_entry(entry: dict, file: str, class_nodes: list[dict],
                    ancestors: frozenset, depth: int) -> dict:
    callee = entry.get("callee")
    target = _find_class_node(callee, file, class_nodes)
    node: dict = {"class": callee, "attr": entry.get("attr"), "line": entry.get("line"),
                  "children": []}
    if entry.get("element_index") is not None:
        node["element_index"] = entry["element_index"]
    if entry.get("conditional"):
        node["conditional"] = True
    if target is None:
        node.update({"file": None, "parent": None,
                     "source": "builtin" if _is_builtin_callee(callee) else "unknown"})
        # 容器（nn.Sequential 等）即使不在本仓库，也按静态元素展开一层，children 为 builtin/unknown
        if entry.get("elements") and depth < MAX_TREE_DEPTH:
            node["children"] = [_node_for_entry(e, file, class_nodes, ancestors, depth + 1)
                                for e in entry["elements"]]
        return node
    node.update({"file": target["file"], "parent": target["parent"], "source": "repo"})
    if target["key"] in ancestors:
        node["recursive"] = True
        return node
    if depth >= MAX_TREE_DEPTH:
        node["truncated"] = True
        return node
    if entry.get("elements"):
        node["children"] = [_node_for_entry(e, target["file"], class_nodes, ancestors, depth + 1)
                            for e in entry["elements"]]
    else:
        node["children"] = [_node_for_entry(e, target["file"], class_nodes,
                                            ancestors | {target["key"]}, depth + 1)
                            for e in target["instantiations"]]
    return node


def _build_module_tree(class_nodes: list[dict]) -> list[dict]:
    """组合嵌套树：容器类 → 子模块（`self.x = SomeClass(...)`，容器元素展开），递归到 builtin 叶子。

    根=没有被其他仓库模块类实例化的模块类；同名跨文件歧义、未知名、动态写法一律不猜（如实标 source）。
    """
    referenced = set()
    for c in class_nodes:
        for entry in c["instantiations"]:
            candidates = [entry.get("callee")]
            candidates += [e.get("callee") for e in entry.get("elements") or []]
            for callee in candidates:
                target = _find_class_node(callee, c["file"], class_nodes)
                if target and target["key"] != c["key"]:   # 自引用不算「被别人嵌套」
                    referenced.add(target["key"])

    roots = [c for c in class_nodes if c["key"] not in referenced]
    if class_nodes and not roots:
        roots = list(class_nodes)   # 纯环（互相嵌套）：全部作根，recursive 标记保证不无限展开

    tree: list[dict] = []
    for c in roots:
        tree.append({
            "class": c["class"], "file": c["file"], "parent": c["parent"], "attr": None,
            "line": None, "source": "repo",
            "children": [_node_for_entry(e, c["file"], class_nodes, frozenset({c["key"]}), 1)
                         for e in c["instantiations"]],
        })
    return tree


def _iter_calls(node: ast.AST):
    """按**求值顺序**产出表达式/语句里的 Call（被调用者的参数先于该调用本身）。"""
    if isinstance(node, ast.Call):
        yield from _iter_calls(node.func)
        for arg in node.args:
            yield from _iter_calls(arg)
        for kw in node.keywords:
            yield from _iter_calls(kw.value)
        yield node
        return
    for child in ast.iter_child_nodes(node):
        yield from _iter_calls(child)


def _target_names(stmt: ast.stmt) -> list[str]:
    """简单赋值的左值名（`model = ...` / `model: T = ...`）；解构/属性赋值不给名。"""
    if isinstance(stmt, ast.Assign):
        return [t.id for t in stmt.targets if isinstance(t, ast.Name)]
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id]
    return []


def _collect_calls(body: list, class_attrs: dict, repo_names: set) -> list[dict]:
    """按源码求值顺序收集调用（AST 事实，非数据流推断）：

    ① `self.<attr>(...)` → kind=module_call（能解析出属性对应的仓库类时带 resolved_class）；
    ② 点号调用（F.relu / torch.optim.Adam / model.train）→ kind=call；
    ③ 裸名调用：名字是仓库 nn.Module 子类 → kind=instantiate；名字是本函数内由该类实例化
       赋值的局部变量（`model = Net(8)` 后的 `model(x)`）→ kind=instance_call + resolved_class；
       裸名其他调用（print/len/range）不入链。
    assign_to=该调用所在赋值语句的左值（同一语句的多个调用共享，如实标注语句而非数据流归属）。
    """
    out: list[dict] = []
    order = 0
    bindings: dict = {}

    def visit(stmts: list) -> None:
        nonlocal order
        for st in stmts:
            names = _target_names(st)
            assign_to = names[0] if names else None
            if isinstance(st, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                # 条件/迭代对象先于分支体求值
                parts = [getattr(st, "test", None), getattr(st, "iter", None)]
                parts += [it.context_expr for it in getattr(st, "items", [])]
                for part in parts:
                    if part is not None:
                        for call in _iter_calls(part):
                            order = _append_call(out, call, class_attrs, repo_names, bindings,
                                                 order, None)
                visit(st.body)
                visit(getattr(st, "orelse", []))
                visit(getattr(st, "finalbody", []))
                continue
            if isinstance(st, ast.Try):
                visit(st.body)
                for handler in st.handlers:
                    visit(handler.body)
                visit(st.finalbody)
                continue
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue          # 内层定义的调用顺序属于它自己，不混进当前链
            for call in _iter_calls(st):
                order = _append_call(out, call, class_attrs, repo_names, bindings, order, assign_to)

    visit(body)
    return out


def _append_call(out: list, call: ast.Call, class_attrs: dict, repo_names: set,
                 bindings: dict, order: int, assign_to: "str | None") -> int:
    """按口径决定一次调用是否入链；入链则 order+1。"""
    callee = _dotted_name(call.func)
    attr = _self_attr(callee)
    entry: dict = {"order": order + 1, "callee": callee, "line": getattr(call, "lineno", None)}
    if attr is not None:
        entry["kind"] = "module_call"
        entry["module_attr"] = attr
        resolved = class_attrs.get(attr)
        if resolved:
            entry["resolved_class"] = resolved
    elif callee and "." in callee:
        entry["kind"] = "call"
    elif callee in repo_names:
        entry["kind"] = "instantiate"
        entry["resolved_class"] = callee
        if assign_to:
            bindings[assign_to] = callee      # 供本函数内后续 `model(x)` 解析
    elif callee in bindings:
        entry["kind"] = "instance_call"
        entry["resolved_class"] = bindings[callee]
    else:
        return order      # 裸名其他调用（print/len/range…）：不入链，避免噪声
    if assign_to:
        entry["assign_to"] = assign_to
    out.append(entry)
    return order + 1


def _method_node(class_node: ast.ClassDef, name: str) -> "ast.AST | None":
    for st in class_node.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name == name:
            return st
    return None


def _build_call_chain(class_nodes: list[dict], func_nodes: dict, report: dict) -> list[dict]:
    """调用顺序链：forward 链 + 训练/推理入口函数链（口径见文件 docstring）。"""
    repo_names = {c["class"] for c in class_nodes}
    chains: list[dict] = []
    for c in class_nodes:
        forward = _method_node(c["node"], "forward")
        if forward is None:
            continue
        attrs: dict = {}
        for entry in c["instantiations"]:
            target = _find_class_node(entry.get("callee"), c["file"], class_nodes)
            attrs[entry["attr"]] = target["class"] if target else None
        chains.append({"file": c["file"], "class": c["class"], "function": "forward",
                       "kind": "forward", "basis": "ast_call_order",
                       "calls": _collect_calls(forward.body, attrs, repo_names)})
    for kind, key in (("train", "train_flow"), ("inference", "inference_flow")):
        for item in report.get(key) or []:
            node = func_nodes.get((item["file"], item["function"]))
            if node is None:
                continue
            chains.append({"file": item["file"], "class": None, "function": item["function"],
                           "kind": kind, "basis": "ast_call_order",
                           "calls": _collect_calls(node.body, {}, repo_names)})
    return chains


def _parse_requirements(source: Path) -> dict[str, str]:
    """解析 requirements.txt，返回 {包名(小写): 版本约束}。"""
    spec: dict[str, str] = {}
    req = source / "requirements.txt"
    if not req.exists():
        return spec
    for line in req.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-", "git+", "http")):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)(.*)$", line)
        if m:
            spec[m.group(1).lower()] = m.group(2).strip()
    return spec


def _local_modules(source: Path, py_files: list[Path]) -> set[str]:
    """项目内顶层模块名（.py 文件 stem + 顶层包目录名），用于从依赖中排除。"""
    local: set[str] = set()
    for p in py_files:
        rel = p.relative_to(source)
        if len(rel.parts) == 1:
            local.add(rel.stem.lower())
        elif rel.parts:
            local.add(rel.parts[0].lower())
    return local


def scan(source_dir: str) -> dict:
    # resolve 软链（本地路径挂载用符号链接，rglob 默认不跟随）
    source = Path(source_dir).resolve()
    py_files = []
    for p in source.rglob("*.py"):
        rel_parts = p.relative_to(source).parts
        if not any(part in EXCLUDE_DIRS for part in rel_parts):
            py_files.append(p)

    report: dict = {
        "entry_points": [],
        "model_files": [],
        "train_flow": [],
        "inference_flow": [],
        "module_hierarchy": [],
        "module_tree": [],
        "call_chain": [],
        "dependencies": [],
        "uncertain": [],
    }

    stdlib = set(sys.stdlib_module_names)
    imports_seen: dict[str, list[str]] = {}
    class_nodes: list[dict] = []
    func_nodes: dict[tuple, ast.AST] = {}

    for py_file in py_files:
        tree = _parse(py_file)
        if tree is None:
            continue
        rel = str(py_file.relative_to(source))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_module_class(node):
                if rel not in report["model_files"]:
                    report["model_files"].append(rel)
                report["module_hierarchy"].append(
                    {"file": rel, "class": node.name, "parent": _parent_class(node)}
                )
                class_nodes.append({
                    "key": (rel, node.name), "file": rel, "class": node.name,
                    "parent": _parent_class(node), "node": node,
                    "instantiations": _instantiation_entries(node),
                })
                for attr in _conditional_module_attrs(node):
                    report["uncertain"].append({
                        "file": rel,
                        "reason": f"conditional module instantiation: {node.name}.{attr}",
                    })

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                low = node.name.lower()
                if any(k in low for k in TRAIN_HINTS):
                    report["train_flow"].append({"file": rel, "function": node.name})
                    func_nodes.setdefault((rel, node.name), node)
                if any(k in low for k in INFER_HINTS):
                    report["inference_flow"].append({"file": rel, "function": node.name})
                    func_nodes.setdefault((rel, node.name), node)

            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = _top_module(alias.name)
                    if top and top not in stdlib:
                        imports_seen.setdefault(top, []).append(rel)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    continue  # 相对导入（from .x import y），本地模块，非第三方
                if node.module:
                    top = _top_module(node.module)
                    if top and top not in stdlib:
                        imports_seen.setdefault(top, []).append(rel)

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DYNAMIC_FUNCS:
                report["uncertain"].append({"file": rel, "reason": f"dynamic {node.func.id}"})

        if _is_entry(py_file, tree) and rel not in report["entry_points"]:
            report["entry_points"].append(rel)

    req_spec = _parse_requirements(source)
    local_mods = _local_modules(source, py_files)

    for name, files in imports_seen.items():
        if name in local_mods:
            continue  # 排除本地模块
        report["dependencies"].append(
            {"name": name, "version_spec": req_spec.get(name, ""), "used_in": files}
        )

    # 层级树与调用顺序链：两阶段构建（先收齐全仓库模块类，再解析跨文件引用）
    report["module_tree"] = _build_module_tree(class_nodes)
    report["call_chain"] = _build_call_chain(class_nodes, func_nodes, report)

    return report


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python scan_structure.py <source_dir> <output.json>", file=sys.stderr)
        sys.exit(2)
    source_dir, output = sys.argv[1], sys.argv[2]
    report = scan(source_dir)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"scan done: {len(report['model_files'])} model files, "
        f"{len(report['dependencies'])} deps, {len(report['uncertain'])} uncertain"
    )


if __name__ == "__main__":
    main()
