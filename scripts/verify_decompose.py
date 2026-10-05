"""模块四两步验证（模块详细设计 6.4，T7 固定脚本）：结构比对 + 数值比对。

以项目独立环境 python 运行（原模型依赖项目环境；再生成代码自包含，仅依赖 torch）。
用法: python verify_decompose.py <source_dir> <ir.json> <regenerated.py> <out.json> <seeds> <rtol> <atol>
退出码约定（实施约定）：比对不过 = 0 退出（业务结果在 JSON）；脚本异常 = 非 0。
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import torch

from _model_loader import (
    accepted_kwargs, accepted_positional, call_kwargs, first_tensor, _shape_of, instantiate,
    load_entry_class, make_dummy_input, make_extra_inputs, prepare_torch,
)

_NUM_DIFF_LIMIT = 10  # 数值失败定位的差异层上报上限


def _load_generated(path: Path):
    """按文件路径加载再生成代码（类名 Decomp_{root_id}，由 ir_codegen 保证）。"""
    spec = importlib.util.spec_from_file_location("regenerated", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法解析再生成代码: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_modules(model):
    """named_modules 顺序的子模块列表（不含根）：(path, module)，位置即对齐关系。"""
    return [(p, m) for p, m in model.named_modules() if p]


def _param_modules(model):
    """带参数的子模块序列（named_modules 顺序，不含根），用于逐层类型比对。

    注意：不能用 `any(sub.parameters(...))`——parameters() 产出的是张量，
    对多元素张量求 bool 会抛「Boolean value of Tensor ... is ambiguous」。
    """
    return [(p, m) for p, m in _all_modules(model) if next(m.parameters(recurse=False), None) is not None]


def _compare_structure(orig, regen, output_shape_match: bool) -> dict:
    """结构比对（6.4 判定算法 1）：**按位置**逐层对应 + 参数量 + state_dict 形状 + 输出形状。

    层数按「带参数的层」计（`_param_modules`）：无参层（ReLU/Dropout/空 Sequential 等）不携带
    任何权重，静态等价描述下允许被拆分或复制（如复用同一 ReLU 实例的模型被展开为多个 ReLU），
    故不计入严格层数与模块计数——否则样例 ResNet 这类「共享 relu 实例」的模型永远无法通过。
    模块总数仍作为参考信息输出（module_count），不参与判定。
    """
    o = _param_modules(orig)
    g = _param_modules(regen)
    layer_sequence_match = (
        len(o) == len(g)
        and all(oc.__class__.__name__ == gc.__class__.__name__ for (_, oc), (_, gc) in zip(o, g))
    )
    param_count = sum(p.numel() for p in orig.parameters())
    regen_param_count = sum(p.numel() for p in regen.parameters())

    o_items = list(orig.state_dict().items())
    g_items = list(regen.state_dict().items())
    param_shapes_match = (
        len(o_items) == len(g_items)
        and all(v1.shape == v2.shape for (_, v1), (_, v2) in zip(o_items, g_items))
    )

    diff: list[dict] = []
    for (op, oc), (_gp, gc) in zip(o, g):
        if oc.__class__.__name__ != gc.__class__.__name__:
            diff.append({
                "path": op,
                "original_class": oc.__class__.__name__,
                "regenerated_class": gc.__class__.__name__,
            })
    if len(o) != len(g):
        diff.append({"path": None, "original_layers": len(o), "regenerated_layers": len(g)})
    for (k1, v1), (k2, v2) in zip(o_items, g_items):
        if v1.shape != v2.shape:
            diff.append({
                "path": k1,
                "original_shape": list(v1.shape),
                "regenerated_shape": list(v2.shape),
            })

    return {
        "passed": layer_sequence_match and param_shapes_match
        and param_count == regen_param_count and output_shape_match,
        "layer_count": {"original": len(o), "regenerated": len(g)},
        "param_count": {"original": param_count, "regenerated": regen_param_count},
        "module_count": {
            "original": len(list(orig.named_modules())),
            "regenerated": len(list(regen.named_modules())),
            "note": "含无参层，参考信息，不参与判定",
        },
        "layer_sequence_match": layer_sequence_match,
        "param_shapes_match": param_shapes_match,
        "output_shape_match": output_shape_match,
        "diff_layers": diff[:_NUM_DIFF_LIMIT],
    }


def _run_capture(model, inputs, kwargs: dict | None = None):
    """前向并捕获各子模块输出（named_modules 顺序的 (path, 张量) 列表）。

    用位置列表而非路径字典：两模型的子模块命名不同（原模型 layer1.0.conv1 vs 再生成
    代码的 n4.n5），按路径求交集只剩根节点，定位不到差异层。
    """
    captured: list[tuple[str, torch.Tensor]] = []

    def _hook(path):
        def _fn(_module, _args, output):
            # dict/tuple 输出取第一个张量；NestedTensor（MHA 快速路径）不支持 .shape → 跳过
            t = first_tensor(output)
            if t is not None and _shape_of(t) is not None:
                captured.append((path, t.detach()))

        return _fn

    handles = [sub.register_forward_hook(_hook(path)) for path, sub in model.named_modules()]
    try:
        with torch.no_grad():
            y = model(*inputs, **(kwargs or {}))
    finally:
        for h in handles:
            h.remove()
    return y, captured


def _max_errors(ref: torch.Tensor, test: torch.Tensor):
    """（max_rel_err, max_abs_err）：相对误差按 6.4 口径 max(|test−ref|/(|ref|+1e-8))。"""
    diff = (test - ref).abs()
    rel = diff / (ref.abs() + 1e-8)
    return float(rel.max()), float(diff.max())


def _numeric_pass(rel_max: float, abs_max: float, rtol: float, atol: float) -> bool:
    """逐元素相对+绝对误差双判据（满足其一即通过，两者皆超限才算不一致）。"""
    return rel_max <= rtol or abs_max <= atol


def _diff_paths(ref_cap: list, test_cap: list, rtol: float, atol: float) -> list[dict]:
    """按位置对齐两模型的内部输出，返回不一致的层（最深者在前），路径取原模型命名。"""
    out: list[dict] = []
    for (path, y_ref), (_p2, y_test) in zip(ref_cap, test_cap):
        if y_ref.shape != y_test.shape:
            out.append({"path": path, "max_abs_err": None, "note": "shape mismatch"})
            continue
        rel_max, abs_max = _max_errors(y_ref, y_test)
        if not _numeric_pass(rel_max, abs_max, rtol, atol):
            out.append({"path": path, "max_rel_err": rel_max, "max_abs_err": abs_max})
    out.sort(key=lambda d: d["path"].count("."), reverse=True)
    return out[:_NUM_DIFF_LIMIT]


def main() -> None:
    if len(sys.argv) != 8:
        print(
            "usage: python verify_decompose.py <source_dir> <ir.json> <regenerated.py> "
            "<out.json> <seeds> <rtol> <atol>",
            file=sys.stderr,
        )
        sys.exit(2)
    source_dir, ir_path, regen_path, out_path = sys.argv[1:5]
    seeds = [int(s) for s in sys.argv[5].split(",") if s.strip()]
    rtol, atol = float(sys.argv[6]), float(sys.argv[7])

    prepare_torch()
    ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    cls = load_entry_class(source_dir, ir["source_file"], ir["entry_class"])
    spec = ir.get("input_spec") or {}
    shape = list(spec.get("shape") or [1, 3, 32, 32])
    if not shape or not all(isinstance(d, int) and d > 0 for d in shape):
        print("input_spec.shape 含非正整数维度（如 null），无法构造输入："
              "请先 PUT /api/projects/{id}/ir/input_spec 指定具体形状（如 [1, 64]）", file=sys.stderr)
        sys.exit(3)
    dtype = getattr(torch, str(spec.get("dtype") or "float32"), torch.float32)

    orig = instantiate(cls, ir.get("entry_args")).eval()
    regen = getattr(_load_generated(Path(regen_path)), f"Decomp_{ir['root_id']}")().eval()

    # 权重同源（实施约定 6.6-3）：两模型各自随机初始化，直接比对必然不等。按结构比对
    # 所依据的位置对应关系，把原模型的权重/缓冲逐一张量拷入再生成模型；数值比对检验的
    # 才是「连接关系与拓扑顺序是否忠实」，而不是初始化巧合。
    with torch.no_grad():
        for v_src, v_dst in zip(orig.state_dict().values(), regen.state_dict().values()):
            if v_src.shape == v_dst.shape:
                v_dst.copy_(v_src)

    # 输出形状（种子 0 单次前向）+ 逐种子数值比对。
    # 前向失败（IR 内部不一致，例如改了 conv 的 out_channels 却未同步其后的 BN）→ 记业务失败，
    # 不让脚本异常退出：验证「不过」是正常结果，宿主据此落 run_record(status=failed)。
    forward_error: str | None = None
    yo = yr = None
    torch.manual_seed(seeds[0])
    fkw = call_kwargs(spec)          # forward 关键字参数（如 CLS/MVC 分支开关）
    fkw_o = accepted_kwargs(orig.forward, fkw)    # 原模型按开关跑对应分支
    fkw_g = accepted_kwargs(regen.forward, fkw)   # 再生成模型的 forward 是 IR 生成的，未必有这些开关
    x0 = (make_dummy_input(shape, dtype), *make_extra_inputs(spec))
    x0_o = accepted_positional(orig.forward, x0)    # 原模型可能消费多输入（src/values/mask）
    x0_g = accepted_positional(regen.forward, x0)   # 再生成模型只声明它消费的输入
    try:
        with torch.no_grad():
            yo = first_tensor(orig(*x0_o, **fkw_o))   # dict/tuple 输出取首个张量（如 scGPT 返回 Mapping）
            yr = first_tensor(regen(*x0_g, **fkw_g))
    except Exception as e:  # noqa: BLE001
        forward_error = f"{type(e).__name__}: {e}"
    output_shape_match = yo is not None and yr is not None and list(yo.shape) == list(yr.shape)

    structure = _compare_structure(orig, regen, output_shape_match)
    if forward_error:
        structure["forward_error"] = forward_error

    per_seed: list[dict] = []
    first_fail: dict | None = None
    for seed in seeds:
        if forward_error:
            break
        torch.manual_seed(seed)
        x = (make_dummy_input(shape, dtype), *make_extra_inputs(spec))
        if seed == seeds[0]:
            y_ref, y_test = yo, yr
        else:
            y_ref = first_tensor(_run_capture(orig, accepted_positional(orig.forward, x), fkw_o)[0])
            y_test = first_tensor(_run_capture(regen, accepted_positional(regen.forward, x), fkw_g)[0])
        rel_max, abs_max = _max_errors(y_ref, y_test)
        passed = _numeric_pass(rel_max, abs_max, rtol, atol)
        record = {"seed": seed, "max_rel_err": rel_max, "max_abs_err": abs_max, "passed": passed}
        if not passed:
            _y, ref_cap = _run_capture(orig, x, fkw)
            _y2, test_cap = _run_capture(regen, x, fkw)
            record["diff_layers"] = _diff_paths(ref_cap, test_cap, rtol, atol)
            if first_fail is None:
                first_fail = record
        per_seed.append(record)

    numeric = {"passed": not forward_error and all(r["passed"] for r in per_seed),
               "weight_source": "original",  # 实施约定 6.6-3：比对前已把原模型权重拷入再生成模型
               "per_seed": per_seed}
    if forward_error:
        numeric["error"] = f"前向执行失败，跳过数值比对：{forward_error}"

    failure_reason = None
    if forward_error and not structure["passed"]:
        pc, rpc = structure["param_count"]["original"], structure["param_count"]["regenerated"]
        failure_reason = (
            f"前向执行失败（原模型与再生成模型结构不一致，无法比对数值）: {forward_error}; "
            f"结构比对失败: layer_sequence_match={structure['layer_sequence_match']}, "
            f"param_shapes_match={structure['param_shapes_match']}, param_count={pc}/{rpc}; "
            f"差异: {json.dumps(structure['diff_layers'][:3], ensure_ascii=False)}"
        )
    elif not structure["passed"]:
        pc, rpc = structure["param_count"]["original"], structure["param_count"]["regenerated"]
        failure_reason = (
            f"结构比对失败: layer_sequence_match={structure['layer_sequence_match']}, "
            f"param_shapes_match={structure['param_shapes_match']}, "
            f"param_count={pc}/{rpc}, "
            f"output_shape_match={structure['output_shape_match']}; "
            f"差异: {json.dumps(structure['diff_layers'][:3], ensure_ascii=False)}"
        )
    elif not numeric["passed"] and first_fail is not None:
        head = (
            f"数值比对失败: seed={first_fail['seed']} "
            f"max_rel_err={first_fail['max_rel_err']:.3e} max_abs_err={first_fail['max_abs_err']:.3e}"
        )
        failure_reason = (
            f"{head}; 差异层: {json.dumps(first_fail['diff_layers'], ensure_ascii=False)}"
            if first_fail.get("diff_layers") else
            f"{head}（两模型内部输出按位置对齐后无差异层可定位）"
        )

    result = {
        "structure": structure,
        "numeric": numeric,
        "forward_error": forward_error,
        "overall": "passed" if structure["passed"] and numeric["passed"] else "failed",
        "failure_reason": failure_reason,
    }
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verify done: overall={result['overall']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 —— 脚本异常非 0 退出，宿主记 run_record
        traceback.print_exc()
        sys.exit(1)
