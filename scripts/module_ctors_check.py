"""验证 `<ws>/reports/module_ctors.py`（顶层黑盒子模块的构造契约）。

**为什么需要它**：顶层黑盒节点把「这个子模块怎么造出来」的知识外包给了一个
agent 生成、人可手改的契约文件。契约若不经验证就是「信任 LLM」——一个写错的构造
表达式，会一路走到再生成/⑤ 才以参数量不符爆出。这里在**拆解当场**把它验掉。

判据（四条全过才算该黑盒可信）：
  1. 表达式可求值——命名空间**只有** `torch`/`nn`/`F`（再生成代码头部就这三行 import）+ Python 内置；
  2. 类名与真实子模块相同；
  3. `named_modules()` 的 **相对路径 + 类名序列**完全相同（内部结构一致）；
  4. 参数张量数、参数总量、buffer 数相同（权重能逐位置拷过去的前提）。

用法（仓库根目录）:
    python scripts/module_ctors_check.py --project-id <原始项目 id>
    python scripts/module_ctors_check.py --project-id <id> --ctor "sim"   # 只查一个
    python scripts/module_ctors_check.py --project-id <id> --json          # 机器可读

**不修改**项目任何文件。缺 `module_ctors.py` 时如实报「无契约」（不是失败）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load_ctors(path: Path) -> dict[str, str] | None:
    """按路径导入契约文件，返回其 CTORS 字典（缺文件/缺 CTORS → None）。"""
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_module_ctors", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    ctors = getattr(mod, "CTORS", None)
    return ctors if isinstance(ctors, dict) else None


def _fingerprint(module: nn.Module) -> dict:
    """比对指纹：类名 + 相对子模块序列 + 参数/buffer 计数。"""
    return {
        "class": type(module).__name__,
        "modules": [(p, type(m).__name__) for p, m in module.named_modules() if p],
        "param_tensors": sum(1 for _ in module.parameters()),
        "param_numel": sum(int(p.numel()) for p in module.parameters()),
        "buffer_tensors": sum(1 for _ in module.buffers()),
    }


def _first_diff(a: list, b: list) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"第 {i} 项不同: 契约 {x!r} vs 真实 {y!r}"
    return f"长度不同: 契约 {len(a)} vs 真实 {len(b)}"


def _compare(name: str, expr: str, real: nn.Module) -> dict:
    """求值一个构造表达式并与真实子模块比对，返回结果字典。"""
    out = {"name": name, "ok": False, "reason": "", "class_expected": type(real).__name__}
    ns = {"torch": torch, "nn": nn, "F": F}
    try:
        built = eval(expr, ns)  # noqa: S307 —— 契约就是「可求值的表达式」，这是它的定义
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"表达式求值失败（{type(e).__name__}: {e}）"
        return out
    if not isinstance(built, nn.Module):
        out["reason"] = f"表达式求值结果不是 nn.Module，而是 {type(built).__name__}"
        return out

    want, got = _fingerprint(real), _fingerprint(built)
    out["class_built"] = got["class"]
    if want["class"] != got["class"]:
        out["reason"] = f"类名不同：契约 {got['class']} vs 真实 {want['class']}"
        return out
    if want["modules"] != got["modules"]:
        out["reason"] = "内部结构与真实不同（named_modules 路径+类名序列）：" + _first_diff(
            [f"{p}:{c}" for p, c in got["modules"]], [f"{p}:{c}" for p, c in want["modules"]])
        return out
    for key, label in (("param_tensors", "参数张量数"), ("param_numel", "参数总量"),
                       ("buffer_tensors", "buffer 数")):
        if want[key] != got[key]:
            out["reason"] = f"{label}不同：契约 {got[key]} vs 真实 {want[key]}"
            return out
    out["ok"] = True
    out["reason"] = (f"一致（{len(want['modules'])} 个子模块、{want['param_tensors']} 个参数张量、"
                     f"{want['param_numel']} 个参数）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="验证 module_ctors.py 的构造契约")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--ctors", default=None, help="契约文件路径（默认 <ws>/reports/module_ctors.py）")
    ap.add_argument("--ctor", default=None, help="只验证某一个顶层子模块")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    ws = REPO_ROOT / "data" / "projects" / args.project_id
    ir_path = ws / "reports" / "ir.json"
    if not ir_path.exists():
        print(f"[失败] 找不到 IR：{ir_path}")
        return 1
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    ctor_path = Path(args.ctors) if args.ctors else ws / "reports" / "module_ctors.py"
    ctors = _load_ctors(ctor_path)
    if ctors is None:
        msg = f"无构造契约：{ctor_path} 不存在或缺 CTORS"
        print(json.dumps({"status": "absent", "detail": msg}, ensure_ascii=False) if args.json else f"[跳过] {msg}")
        return 0

    import _model_loader as ml  # noqa: E402

    ml.prepare_torch()
    cls = ml.load_entry_class(str(ws / "source"), ir["source_file"], ir["entry_class"])
    model = ml.instantiate(cls, ir.get("entry_args"), str(ws / "source"))
    if model is None:
        print("[失败] 真实模型构造失败（instantiate 返回 None）")
        return 1

    children = dict(model.named_children())
    names = [args.ctor] if args.ctor else list(ctors.keys())
    results = []
    for name in names:
        if name not in ctors:
            results.append({"name": name, "ok": False, "reason": "契约里没有这一项"})
            continue
        if name not in children:
            results.append({"name": name, "ok": False,
                            "reason": f"真实模型没有名为 {name} 的顶层子模块"})
            continue
        results.append(_compare(name, ctors[name], children[name]))

    ok = sum(1 for r in results if r["ok"])
    if args.json:
        print(json.dumps({"status": "ok" if ok == len(results) else "mismatch",
                          "passed": ok, "total": len(results), "results": results},
                         ensure_ascii=False, indent=2))
    else:
        print(f"契约文件：{ctor_path}")
        for r in results:
            mark = "✓" if r["ok"] else "✗"
            print(f"  {mark} {r['name']:22s} {r['reason']}")
        print(f"\n{ok}/{len(results)} 通过")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
