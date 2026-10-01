"""静态结构扫描（模块详细设计 3.6，T4 固定脚本）。

识别入口脚本、模型定义文件（nn.Module 子类）、训练/推理流程、模块层级与第三方依赖，
不确定点（动态构建/条件分支）标 uncertain，交由 agent 补充（3.7）。

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
        "dependencies": [],
        "uncertain": [],
    }

    stdlib = set(sys.stdlib_module_names)
    imports_seen: dict[str, list[str]] = {}

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

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                low = node.name.lower()
                if any(k in low for k in TRAIN_HINTS):
                    report["train_flow"].append({"file": rel, "function": node.name})
                if any(k in low for k in INFER_HINTS):
                    report["inference_flow"].append({"file": rel, "function": node.name})

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
