"""IR 保真度自检（模块四 6.1）：把「IR 再生成的模型」与**真实模型**比一遍，对不上就报差异。

为什么需要它：拆解原有的自检（`validate_ir` + `ir_codegen.generate`）只查「IR 内部合法 + 能再生成」，
**从不与真实模型比对**——于是 agent 少拆一棵子树（scGPT 漏 mvc_decoder）、把 `transformer_encoder` 的
尺寸算错（再生成只剩 287 万参数 vs 真实 2129 万）都被放行，一直拖到「⑤ 两步验证」才暴露，而且
**换一次拆解就复现一次**。把这个比对挪进拆解循环，就能带着差异清单重试、自动收敛。

宿主（decompose_service）先把候选 IR 生成成 `regen.py` 并写好 entry_args，本脚本只做比对，
因此**只依赖 `_model_loader` + torch**，不需要后端包。

用法: python ir_fidelity_probe.py <source_dir> <ir.json> <regen.py> <out.json> [entry_args.json]
退出码: 0=检查完成（结论在 JSON 的 ok/issues）；2=用法错误。
真实模型起不来（缺 entry_args / 依赖未装）时 JSON 里 skipped=1 —— 宿主据此**跳过**本轮检查。
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _model_loader import (   # noqa: E402
    accepted_kwargs, accepted_positional, call_kwargs, first_tensor, instantiate,
    load_entry_class, make_dummy_input, make_extra_inputs, prepare_torch,
)

MAX_ISSUES = 8
MAX_EXAMPLES = 6


def _param_modules(model):
    """带参层（直接持有参数的子模块）——与 6.4 两步验证同一口径。"""
    return [(p, m) for p, m in model.named_modules()
            if p and next(m.parameters(recurse=False), None) is not None]


def _nn_name(cls) -> str:
    """给出「`torch.nn` 里认得的」类名。

    torch 有些层是**内部子类**，`type(m).__name__` 报出来的名字在 `torch.nn` 里并不存在
    （如 `NonDynamicallyQuantizableLinear` 只是 `nn.Linear` 的量化变体）——把它当构造表达式
    回喂，agent 就会写出 `nn.NonDynamicallyQuantizableLinear(...)` 让再生成代码直接 AttributeError
    （拆解循环实测踩到）。故沿 MRO 回溯到最近的、`torch.nn` 真正暴露的基类。
    """
    import torch

    for base in cls.__mro__:
        if getattr(base, "__module__", "").startswith("torch.nn") and hasattr(torch.nn, base.__name__):
            return f"nn.{base.__name__}"
    return cls.__name__


def _covered_by_ir(ir: dict, path: str) -> bool:
    """真实模块路径是否被 IR 覆盖（命中某节点 module_path，或落在它的子树里）。"""
    for n in ir["nodes"]:
        mp = str(n.get("module_path") or "")
        if path == mp or (mp and path.startswith(mp + ".")):
            return True
    return False


def _write(out_path: str, out: dict) -> None:
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


def main() -> None:
    if len(sys.argv) < 5:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    source_dir, ir_path, regen_path, out_path = sys.argv[1:5]
    entry_args_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] not in ("", "-") else None

    ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    entry_args = None
    if entry_args_path and Path(entry_args_path).is_file():
        try:
            entry_args = json.loads(Path(entry_args_path).read_text(encoding="utf-8")) or None
        except json.JSONDecodeError:
            entry_args = None

    # 真实模型：起不来就跳过本轮（缺 entry_args / 依赖未装 → 无从比对，不能误判为「不忠实」）
    try:
        cls = load_entry_class(source_dir, ir["source_file"], ir["entry_class"])
        orig = instantiate(cls, entry_args).eval()
    except Exception as e:  # noqa: BLE001
        _write(out_path, {"ok": True, "skipped": True,
                          "reason": f"真实模型无法实例化（跳过保真度自检）：{type(e).__name__}: {str(e)[:200]}"})
        return

    spec = importlib.util.spec_from_file_location("regen_probe", regen_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        regen = getattr(mod, f"Decomp_{ir['root_id']}")().eval()
    except Exception as e:  # noqa: BLE001 —— 再生成侧起不来 = IR 的问题，如实报
        _write(out_path, {"ok": False, "issues": [f"再生成模型无法实例化：{type(e).__name__}: {str(e)[:300]}"]})
        return

    issues: list[str] = []
    o_layers, g_layers = _param_modules(orig), _param_modules(regen)
    if len(o_layers) != len(g_layers):
        issues.append(f"带参层数不一致：真实 {len(o_layers)} / 再生成 {len(g_layers)}")
    # 逐层**类名序列**（位置对齐）：结构是否忠实。
    seq_o = [type(m).__name__ for _, m in o_layers]
    seq_g = [type(m).__name__ for _, m in g_layers]
    if len(seq_o) == len(seq_g) and seq_o != seq_g:
        first = next(i for i, (a, b) in enumerate(zip(seq_o, seq_g)) if a != b)
        issues.append(f"逐层类型不一致（第 {first} 层起）：真实 {seq_o[first]} / 再生成 {seq_g[first]}")

    # **不比参数数值/形状**：`num_embeddings` 这类由**运行期**决定（实例按 entry_args 构造，
    # 与源码里的真实词表大小不同），而「③ 补形状」本就会从真实实例把这些参数回填进 IR；
    # 拆解阶段拿「参数量精确相等」当门槛会永远判不过。数值交给 ③ 对齐、由「⑤ 两步验证」终判。
    o_sd, g_sd = list(orig.state_dict().items()), list(regen.state_dict().items())
    if len(o_sd) != len(g_sd):
        issues.append(f"参数张量条目数不一致（结构差异）：真实 {len(o_sd)} / 再生成 {len(g_sd)}")
    o_params = sum(p.numel() for p in orig.parameters())      # 仅作参考信息（不参与判定）
    g_params = sum(p.numel() for p in regen.parameters())

    # 只盯**带参**子模块：那才是「⑤ 两步验证」的结构比对真正看得见的（state_dict / 带参层）。
    # 无参模块（如 schema 表达不了的 `nn.CrossEntropyLoss` 这类多操作数叶子）不计——否则
    # 把判据搞得比 ⑤ 更严，会让「⑤ 本来能过」的 IR 在拆解阶段被判死。
    uncovered = [p for p, m in orig.named_modules()
                 if p and next(m.parameters(recurse=False), None) is not None
                 and not _covered_by_ir(ir, p)]
    if uncovered:
        issues.append(f"真实模型有 {len(uncovered)} 个带参层未被 IR 覆盖：{', '.join(uncovered[:MAX_EXAMPLES])}")

    # **真实调用结构**（`torch.export` + `unflatten`，见 trace_structure）：带参层覆盖之外，还要看
    # **无参但被调用**的模块在不在 IR 里——实测 scGPT 真实前向会调 `creterion_cce`（nn.CrossEntropyLoss，
    # 0 参数）而 IR 整条没有，「只看带参层」永远发现不了（2026-10-06 用户实测坐实）。
    # 追踪不可用（export 失败）时跳过、不误判。
    from trace_structure import trace_model

    traced = trace_model(orig, ir.get("input_spec") or {})
    traced_modules = traced.get("modules") or []
    if not traced.get("skipped"):
        missing_mods = [m for m in traced_modules if not _covered_by_ir(ir, m["module_path"])]
        if missing_mods:
            names = ", ".join(f"{m['module_path']}({m['class_name']})" for m in missing_mods[:MAX_EXAMPLES])
            issues.append(
                f"真实模型调用了 {len(missing_mods)} 个模块，但 IR 里没有对应节点：{names}"
                "（**含无参模块**；`nn.CrossEntropyLoss` 这类用 op 节点 + code_hint 表达，"
                "如 code_hint=\"F.cross_entropy({inputs[0]}, {inputs[1]})\"，并给它接上真实输入）"
            )

    # **前向自检**：结构对了不代表**接线**对——scGPT 实测过「参数完全相同、逐层类型全对，
    # 但前向 `mat1 and mat2 shapes cannot be multiplied (1x1200 and 1x512)`」：某条边把错的
    # 张量接进了某个 Linear。只看 state_dict 看不出来，必须真跑一遍。
    spec = ir.get("input_spec") or {}
    shape = list(spec.get("shape") or [])
    forward_note = ""
    if shape and all(isinstance(d, int) and d > 0 for d in shape):
        import torch

        prepare_torch()
        dtype = getattr(torch, str(spec.get("dtype") or "float32"), torch.float32)
        fkw = call_kwargs(spec)
        try:
            torch.manual_seed(0)
            xs = (make_dummy_input(shape, dtype), *make_extra_inputs(spec))
            with torch.no_grad():
                yo = first_tensor(orig(*accepted_positional(orig.forward, xs),
                                       **accepted_kwargs(orig.forward, fkw)))
                yr = first_tensor(regen(*accepted_positional(regen.forward, xs),
                                        **accepted_kwargs(regen.forward, fkw)))
        except Exception as e:  # noqa: BLE001
            # 把**再生成代码里的出错行**一并回喂（那行就是问题节点，如 `var_x = self.cv_linear1(x)`
            # ——实测 agent 漏了它前面的 unsqueeze，只给「形状乘不了」它不知道改哪儿）。
            import traceback as _tb

            frames = [ln.strip() for ln in _tb.format_exc().splitlines()
                      if "gen" in ln and ".py" in ln or ln.strip().startswith("var_")]
            where = " | ".join(frames[-3:])[:300]
            issues.append(f"前向执行失败：{type(e).__name__}: {str(e)[:160]}"
                          + (f"；出错节点附近：{where}" if where else ""))
        else:
            if yo is None or yr is None:
                issues.append("前向输出不是张量（无法比对形状）")
            elif tuple(yo.shape) != tuple(yr.shape):
                issues.append(f"前向输出形状不一致：真实 {list(yo.shape)} / 再生成 {list(yr.shape)}")
    else:
        forward_note = "input_spec.shape 缺失/非法 → 未做前向自检"

    # 真实带参层清单：回喂给 agent 重建结构用（比只给「差多少」有用得多）
    inventory = [
        {"path": p, "class": _nn_name(type(m)), "params": sum(x.numel() for x in m.parameters(recurse=False))}
        for p, m in o_layers
    ]
    _write(out_path, {
        "ok": not issues,
        "issues": issues[:MAX_ISSUES],
        "original_params": o_params,
        "regenerated_params": g_params,
        "original_param_layers": len(o_layers),
        "regenerated_param_layers": len(g_layers),
        "uncovered": uncovered[:MAX_EXAMPLES],
        "real_param_modules": inventory[:120],
        # 真实调用结构（含无参模块与算子），供宿主回喂 / 后续据其生成 IR
        "traced_modules": traced_modules[:120],
        "traced_skipped": None if not traced.get("skipped") else traced.get("reason"),
    })


if __name__ == "__main__":
    main()
