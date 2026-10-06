"""从**真实模型追踪**出结构（治本：不再让 LLM 猜 IR 的模块集与依赖）。

背景（2026-10-06 用户实测）：LLM 产出的 IR 会漏模块、把模块晾成孤立死块——scGPT 的 IR 丢了
`creterion_cce`、`mvc_decoder` 变成「只实例化不调用」，而 ⑤ 原先只比首个张量、照样判通过。

链路（在**项目独立环境**执行）：
  torch.export.export（`fx.symbolic_trace` 对数据依赖分支会失败，实测 scGPT 必失败）
  → torch.export.unflatten（还原模块层级）
  → 遍历图：call_module → 模块调用（path/class/参数量），call_function → 算子（aten 名 + 输入引用）
  → 每个节点「引用了谁」即一条依赖边。
输出 JSON（**结构真相**，供宿主与 IR 比对 / 后续据此生成 IR）。

用法: python trace_structure.py <source_dir> <ir.json> <out.json>
退出码：0 成功；非 0 异常（宿主记 run_record）。追踪不可用（缺包/起不来）时写 skipped 并 0 退出。
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _model_loader import (  # noqa: E402
    accepted_kwargs, accepted_positional, call_kwargs, instantiate,
    load_entry_class, build_inputs, prepare_torch,
)

# 导出伪影：只用于元数据断言/布局，不产生真实数据流，比对与生成都要跳过
_SKIP_ATEN = ("_assert_tensor_metadata", "to.dtype_layout", "lift_fresh_copy")


def _refs(node) -> list[str]:
    """本节点的输入引用了哪些图节点（名字）。"""
    out = []
    for a in list(node.args) + list(node.kwargs.values()):
        if hasattr(a, "name") and hasattr(a, "op"):
            out.append(a.name)
        elif isinstance(a, (list, tuple)):
            out.extend(x.name for x in a if hasattr(x, "name") and hasattr(x, "op"))
    return out


def trace_model(model, spec: dict, source_dir: str | None = None) -> dict:
    """在**已实例化**的真实模型上追踪结构（供保真度自检复用，避免二次加载模型）。

    返回 `{"skipped": bool, "reason"?: str, "modules": [...], "ops": [...], "edges": [...]}`。
    追踪不可用（torch.export 失败）时 `skipped=True` 并给原因——**跳过而非误判**。
    """
    import torch
    from torch.export import unflatten

    try:
        dt = getattr(torch, str(spec.get("dtype") or "float32").replace("torch.", ""), torch.float32)
        xs = build_inputs(model, spec, source_dir)
        args = accepted_positional(model.forward, xs)
        kwargs = accepted_kwargs(model.forward, call_kwargs(spec))
        ep = torch.export.export(model, args, kwargs=kwargs or None, strict=False)
        gm = unflatten(ep)
    except Exception as e:  # noqa: BLE001
        return {"skipped": True, "reason": f"{type(e).__name__}: {str(e)[:200]}"}

    # unflatten 会把模块包成 InterpreterModule，类名要从**原始模型**取
    orig_mods = dict(model.named_modules())
    submods = dict(gm.named_modules())

    modules: dict[str, dict] = {}
    ops: list[dict] = []
    edges: list[dict] = []
    name2target: dict[str, str] = {}

    for n in gm.graph.nodes:
        if n.op == "call_module":
            target = str(n.target)
            m = submods.get(target)
            om = orig_mods.get(target)
            entry = modules.setdefault(target, {
                "module_path": target,
                "class_name": type(om).__name__ if om is not None else "?",
                "param_numel": sum(p.numel() for p in m.parameters()) if m is not None else 0,
                "parent": target.rsplit(".", 1)[0] if "." in target else None,
                "calls": 0,
            })
            entry["calls"] += 1
            name2target[n.name] = target
        elif n.op == "call_function":
            aten = str(n.target).replace("torch.ops.", "")
            if any(s in aten for s in _SKIP_ATEN):
                continue
            ops.append({"graph_name": n.name, "aten": aten, "inputs": _refs(n)})
            name2target[n.name] = f"op:{n.name}"
        elif n.op == "placeholder":
            name2target[n.name] = f"input:{n.name}"

    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output"):
            continue
        for src in _refs(n):
            edges.append({"from": name2target.get(src, src), "to": name2target.get(n.name, n.name)})

    return {
        "skipped": False,
        "n_graph_nodes": len(list(gm.graph.nodes)),
        "modules": sorted(modules.values(), key=lambda m: m["module_path"]),
        "ops": ops,
        "edges": edges,
    }


def main() -> None:
    source_dir, ir_path, out_path = sys.argv[1:4]
    ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    spec = ir.get("input_spec") or {}
    out = Path(out_path)

    def _write(payload: dict) -> None:
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {out}")

    prepare_torch()
    try:
        cls = load_entry_class(source_dir, ir.get("source_file"), ir.get("entry_class"))
        model = instantiate(cls, ir.get("entry_args"), source_dir).eval()
    except Exception as e:  # noqa: BLE001 —— 模型起不来属「跳过」而非失败
        _write({"skipped": True, "reason": f"{type(e).__name__}: {e}"})
        return

    traced = trace_model(model, spec, source_dir)
    traced.update({"schema_version": "1.0", "entry_class": ir.get("entry_class")})
    _write(traced)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 —— 脚本异常非 0 退出，宿主记 run_record
        traceback.print_exc()
        sys.exit(1)
