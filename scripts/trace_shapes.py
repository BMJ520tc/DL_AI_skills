"""模块四形状追踪（模块详细设计 6.1）：项目环境中实例化入口模型，注册 forward hook
捕获各子模块的输入/输出形状，按 named_modules 路径回填 IR 的缺失形状。

用法: python trace_shapes.py <source_dir> <ir.json> <out.json>
宿主侧编排：以项目独立环境 python 运行本脚本（decompose_service._run_trace）。
退出码: 0=成功; 1=模型加载/前向失败（错误信息入 stderr，宿主记 run_record 供检索）。
"""
import json
import sys
from pathlib import Path

import torch

from _model_loader import instantiate, load_entry_class

# 从实例化出来的层上读取构造参数（回填 agent 给不出的值：库型模型的层尺寸由运行期构造参数决定）
_PARAM_ATTRS = ("in_features", "out_features", "num_features", "in_channels", "out_channels",
                "kernel_size", "stride", "padding", "dilation", "groups", "bias",
                "num_embeddings", "embedding_dim", "p", "eps")


def _module_params(module) -> dict:
    out: dict = {}
    for k in _PARAM_ATTRS:
        if not hasattr(module, k):
            continue
        v = getattr(module, k)
        if isinstance(v, (bool, int, float)):
            out[k] = v
        elif isinstance(v, (tuple, list)) and v and all(isinstance(x, int) for x in v):
            out[k] = list(v)
    return out


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


def _capture(model, x) -> dict:
    """注册 hook 捕获各子模块（含根，路径 ""）的输入/输出形状。

    产物：{"shapes": {path: {input_shape, output_shape}}, "order": [{path, class_name}…]}。
    order 按 named_modules 顺序并带类名，供宿主在 IR 节点缺 module_path 时按「类名+次序」兜底对齐。
    """
    captured: dict[str, dict] = {}

    def _hook(path):
        def _fn(_module, args, output):
            rec = captured.setdefault(path, {})
            if args and isinstance(args[0], torch.Tensor):
                rec["input_shape"] = list(args[0].shape)
            if isinstance(output, torch.Tensor):
                rec["output_shape"] = list(output.shape)

        return _fn

    modules = list(model.named_modules())
    handles = [sub.register_forward_hook(_hook(path)) for path, sub in modules]
    try:
        with torch.no_grad():
            model(x)
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

    ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    cls = load_entry_class(source_dir, ir["source_file"], ir["entry_class"])
    model = instantiate(cls).eval()

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
    x = torch.randn(*shape, dtype=dtype)

    result = _capture(model, x)
    result["input_shape"] = shape
    result["params"] = {p: _module_params(m) for p, m in model.named_modules() if p}
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"trace done: {len(result['shapes'])} module paths captured")


if __name__ == "__main__":
    main()
