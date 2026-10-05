"""模块四形状追踪（模块详细设计 6.1）：项目环境中实例化入口模型，注册 forward hook
捕获各子模块的输入/输出形状，按 named_modules 路径回填 IR 的缺失形状。

用法: python trace_shapes.py <source_dir> <ir.json> <out.json>
宿主侧编排：以项目独立环境 python 运行本脚本（decompose_service._run_trace）。
退出码: 0=成功; 1=模型加载/前向失败（错误信息入 stderr，宿主记 run_record 供检索）。
"""
import inspect
import json
import sys
from pathlib import Path

import torch

from _model_loader import (
    call_kwargs, first_tensor, _shape_of, instantiate, load_entry_class, make_dummy_input,
    make_extra_inputs, prepare_torch,
)

# 从实例化出来的层上读取构造参数。两个用途：
# ① 回填 agent 给不出的值（库型模型的层尺寸由运行期构造参数决定）；
# ② 作为「模型实际暴露的参数」用于模块结构签名 —— agent 是否写出**可选参数**
#    （如 ReLU 的 inplace、BatchNorm 的 affine）并不稳定，若签名取 agent 的写法，
#    同一模型两次拆解可能得到不同 module_id（见《阶段3实施方案》11.6 R27）。
# 因此这里覆盖常见可选开关（inplace/affine/…）与结构类参数（start_dim/mode/…），
# **只取「属性名即构造参数名」的键**，保证回填进 params 后可直接进生成代码。
_PARAM_ATTRS = (
    # 维度类
    "in_features", "out_features", "num_features", "in_channels", "out_channels",
    "num_embeddings", "embedding_dim", "num_groups", "num_channels", "normalized_shape",
    # 卷积/池化/上采样形状类
    "kernel_size", "stride", "padding", "output_padding", "dilation", "groups",
    "output_size", "size", "scale_factor",
    # 归一化/正则/激活的可选开关与标量
    "bias", "affine", "track_running_stats", "elementwise_affine", "eps", "momentum",
    "p", "inplace", "negative_slope", "alpha", "num_parameters",
    # 展平/维度语义/卷积补齐方式
    "start_dim", "end_dim", "dim", "mode", "align_corners", "padding_mode",
)


def _ctor_param_names(cls) -> set | None:
    """类 `__init__` 接受的关键字参数名；有 `**kwargs` 或取不到签名时返回 None（不限制）。

    必要的一道闸门：属性名不等于构造参数名。典型反例 `nn.Conv2d.output_padding`——
    它由 `_ConvNd` 统一持有，但普通卷积的 `__init__` 不接受该参数，
    直接回填会生成 `nn.Conv2d(..., output_padding=(0, 0))` 这种一实例化就 TypeError 的代码
    （实测：样例 ResNet 再生成即崩）。故只保留 `__init__` 真正接受的键。
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    names = set()
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            names.add(name)
    return names


def _norm_value(v):
    return list(v) if isinstance(v, (tuple, list)) else v


def _same_value(a, b) -> bool:
    """参数等价判定：标量与同值元组（`stride=1` 与 `(1, 1)`）视为同一值。"""
    a, b = _norm_value(a), _norm_value(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, list) and a and all(x == a[0] for x in a):
        return _same_value(a[0], b)
    if isinstance(b, list) and b and all(x == b[0] for x in b):
        return _same_value(a, b[0])
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b


def _ctor_defaults(cls) -> dict:
    """类 `__init__` 的默认值表（无法取签名时为空）。"""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}
    return {
        name: p.default for name, p in sig.parameters.items()
        if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
        and p.default is not inspect.Parameter.empty
    }


def _module_params(module) -> dict:
    """实例化层上「可直接作为构造参数」的键值（None 与不可序列化值不取）。"""
    allowed = _ctor_param_names(type(module))
    out: dict = {}
    for k in _PARAM_ATTRS:
        if allowed is not None and k not in allowed:
            continue
        if not hasattr(module, k):
            continue
        v = getattr(module, k)
        if v is None:
            continue
        if isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif (isinstance(v, (tuple, list)) and v
              and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
            out[k] = list(v)
    return out


def _non_default_params(module, params: dict) -> dict:
    """params 中「与构造默认值不等价」的子集。

    只有这部分值得在 agent 没写时回填进 IR 的 `params`：默认值等价的键（agent 不写也等价）
    若也写进去，会平白改动 `params` 从而让既有验证记录变 stale（与「形状回填不产生 stale」
    的口径冲突）。完整参数仍写入 `params` 输出，供模块结构签名使用（见 `main`）。
    """
    defaults = _ctor_defaults(type(module))
    return {k: v for k, v in params.items()
            if k not in defaults or not _same_value(v, defaults[k])}


def _derive_input_shape(model) -> list[int]:
    """input_spec.shape 缺失/非法时，按模型首个带维度的层推导一个可用输入形状。"""
    for m in model.modules():
        for attr, shape in (("in_features", lambda n: [1, n]),
                            ("in_channels", lambda n: [1, n, 32, 32]),
                            ("num_embeddings", lambda n: [1, n])):
            n = getattr(m, attr, None)
            if isinstance(n, int) and n > 0:
                return shape(n)
    return [1, 3, 32, 32]


def _capture(model, inputs, kwargs: dict | None = None) -> dict:
    """注册 hook 捕获各子模块（含根，路径 ""）的输入/输出形状。

    产物：{"shapes": {path: {input_shape, output_shape}}, "order": [{path, class_name}…]}。
    order 按 named_modules 顺序并带类名，供宿主在 IR 节点缺 module_path 时按「类名+次序」兜底对齐。
    """
    captured: dict[str, dict] = {}

    def _hook(path):
        def _fn(_module, args, output):
            rec = captured.setdefault(path, {})
            if args:
                t = first_tensor(args[0])           # 入参也可能是 dict/tuple
                shape = _shape_of(t) if t is not None else None   # NestedTensor 等无 .shape → None
                if shape:
                    rec["input_shape"] = shape
            t = first_tensor(output)                # 输出可能是 dict/tuple（取第一个张量）
            shape = _shape_of(t) if t is not None else None
            if shape:
                rec["output_shape"] = shape

        return _fn

    modules = list(model.named_modules())
    handles = [sub.register_forward_hook(_hook(path)) for path, sub in modules]
    try:
        with torch.no_grad():
            model(*inputs, **(kwargs or {}))
    finally:
        for h in handles:
            h.remove()
    return {
        "shapes": captured,
        "order": [{"path": p, "class_name": type(m).__name__} for p, m in modules if p],
    }


def main() -> None:
    if len(sys.argv) < 4:
        print("usage: python trace_shapes.py <source_dir> <ir.json> <out.json>", file=sys.stderr)
        sys.exit(2)
    source_dir, ir_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    prepare_torch()
    ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    cls = load_entry_class(source_dir, ir["source_file"], ir["entry_class"])
    model = instantiate(cls, ir.get("entry_args")).eval()

    spec = ir.get("input_spec") or {}
    shape = list(spec.get("shape") or [])
    if not shape or not all(isinstance(d, int) and d > 0 for d in shape):
        # agent 给不出具体维度（如入口模型尺寸由运行期构造参数决定）→ 按模型首层推导，
        # 与本次 trace 用的是同一份实例化，故回填的形状与后续再生成/验证保持一致。
        shape = _derive_input_shape(model)
        print(f"input_spec.shape 缺失/非法 → 按模型首层自动推导 {shape}"
              "（可用 PUT /api/projects/{id}/ir/input_spec 覆盖）", file=sys.stderr)
    dtype = getattr(torch, str(spec.get("dtype") or "float32"), torch.float32)
    torch.manual_seed(0)
    inputs = (make_dummy_input(shape, dtype), *make_extra_inputs(spec))

    result = _capture(model, inputs, call_kwargs(spec))
    result["input_shape"] = shape
    params: dict = {}
    delta: dict = {}
    for p, m in model.named_modules():
        if not p:
            continue
        pm = _module_params(m)
        params[p] = pm
        nd = _non_default_params(m, pm)
        if nd:
            delta[p] = nd
    result["params"] = params          # 模型实际暴露的构造参数（供结构签名，不写回 IR params）
    result["params_delta"] = delta     # 其中与默认值不等价的部分（等价于「agent 不写会丢语义」）
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"trace done: {len(result['shapes'])} module paths captured, "
          f"{len(delta)} paths with non-default params")


if __name__ == "__main__":
    main()
