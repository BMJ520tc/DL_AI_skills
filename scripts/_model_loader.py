"""模块四脚本共用：在项目环境中加载入口模型类并实例化（6.1 形状追踪 / 6.4 两步验证）。

被 trace_shapes.py / verify_decompose.py 以项目独立环境 python 运行（sys.path 含本
脚本目录，from _model_loader import ...）。加载策略：
1. model_file 为包内模块（含 / 且父目录有 __init__.py）→ 常规 import；
2. 否则按文件路径加载（spec_from_file_location），兼容无 __init__.py 的平铺项目。
实例化策略：优先无参构造；失败时按常见关键字签名（num_classes/sizes 等）依次尝试；
可用环境变量 DECOMPOSE_ENTRY_ARGS（JSON dict）前置覆盖（论文库模型常需外部数据，
如 GEARS_Model(args)，可给最小 args 字典）。均失败 RuntimeError 说明签名与原因。
"""
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

_INSTANTIATE_FALLBACKS = (
    {"num_classes": 10},
    {"num_labels": 10},
    {"n_classes": 10},
    {"sizes": [64, 128, 10]},
    {"dims": [64, 128, 10]},
    {"hidden_size": 64},
)


def _fallbacks() -> tuple[dict, ...]:
    override = os.environ.get("DECOMPOSE_ENTRY_ARGS")
    if override:
        try:
            parsed = json.loads(override)
            if isinstance(parsed, dict):
                return (parsed,) + _INSTANTIATE_FALLBACKS
        except json.JSONDecodeError:
            pass
    return _INSTANTIATE_FALLBACKS


def load_entry_class(source_dir: str, model_file: str, entry_class: str):
    """返回入口类（模块与类均找不到时 RuntimeError，错误信息供任务 error 展示）。"""
    source = Path(source_dir).resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    fpath = (source / model_file).resolve()
    # model_file 来自 IR（agent 产出）：绝对路径或用 .. 越出 source 一律拒绝，只允许项目内文件
    if not fpath.is_relative_to(source):
        raise RuntimeError(f"模型定义文件必须位于项目目录内: {model_file}")
    if not fpath.exists():
        raise RuntimeError(f"模型定义文件不存在: {model_file}")

    # 包内模块优先走常规 import（相对导入可用）
    if "/" in model_file.replace("\\", "/"):
        dotted = model_file.replace("/", ".").replace("\\", ".").removesuffix(".py")
        try:
            module = importlib.import_module(dotted)
        except (ImportError, ModuleNotFoundError):
            module = _load_by_file(fpath)
    else:
        module = _load_by_file(fpath)

    cls = getattr(module, entry_class, None)
    if cls is None:
        raise RuntimeError(f"模型文件 {model_file} 中未找到入口类 {entry_class}")
    return cls


def _load_by_file(fpath: Path):
    spec = importlib.util.spec_from_file_location(f"decomp_entry_{fpath.stem}", fpath)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法解析模型文件: {fpath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dummy_input(shape: list, dtype):
    """按 dtype 造一张假输入。

    浮点/复数用 `randn`（数值比对需要非平凡输入）；整型/bool 用**全 0**——
    `torch.randn` 不支持整型 dtype（`input_spec.dtype=int64` 的 token id 模型会直接崩），
    而 0 一定落在合法索引范围内（不会越 Embedding 边界）。
    """
    import torch

    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(*shape, dtype=dtype)
    return torch.zeros(*shape, dtype=dtype)


def prepare_torch() -> None:
    """关掉 `nn.MultiheadAttention` 的 NestedTensor 快速路径。

    该快速路径会把子模块输出变成 `NestedTensorImpl`——它**不支持 `.shape`**，forward hook 里
    `args[0].shape` 直接 RuntimeError（scGPT 补形状实测：`Internal error: NestedTensorImpl
    doesn't support sizes`）。关掉后走普通张量路径，形状追踪正常。
    """
    import torch

    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:  # noqa: BLE001 —— 旧版 torch 没有该开关
        pass


def accepted_kwargs(fn, kwargs: dict) -> dict:
    """只保留 `fn` 签名接受的键。

    `input_spec.forward_kwargs` 是给**原模型**的开关（如 scGPT 的 CLS/MVC/CCE/ECS）：原模型要按
    这些开关跑对应分支，而**再生成模型**的 forward 由 IR 生成、并不含这些参数——原样传过去会
    `TypeError: got an unexpected keyword argument 'CLS'`，让验证连比对都做不了。
    """
    import inspect

    if not kwargs:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def accepted_positional(fn, inputs: tuple) -> tuple:
    """按 `fn` 的位置形参个数截断入参。

    再生成模型的 forward 由 IR 生成，**只声明它真正消费的输入**（如单输入图的 `def forward(self, x)`），
    原样把多输入（scGPT 的 src/values/mask）传过去会 `takes 2 positional arguments but 4 were given`，
    验证连比对都做不了。截断后能跑出真实结论（IR 若不消费某输入，比对**本就应该**不过）。
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return inputs
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return inputs
    n = sum(1 for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))
    return tuple(inputs[:n])


def first_tensor(obj):
    """从嵌套输出（dict / list / tuple）里取第一个张量；取不到返回 None。

    很多模型 forward 返回 **dict 或 tuple**（如 scGPT 返回 `Mapping[str, Tensor]`、
    注意力层返回 `(attn_out, weights)`）——只认「顶层就是张量」会让这些节点永远缺输出形状。
    深度优先取第一个张量作为该节点的输出形状代表（近似，但比「未知」有用）。
    """
    import torch

    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = first_tensor(v)
            if t is not None:
                return t
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            t = first_tensor(v)
            if t is not None:
                return t
    return None


def _shape_of(t):
    """张量形状；不支持 `.shape` 的对象（如 NestedTensor）返回 None，不让 hook 抛错中断追踪。"""
    try:
        return list(t.shape)
    except Exception:  # noqa: BLE001
        return None


def call_kwargs(spec: dict) -> dict:
    """`input_spec.forward_kwargs`：传给模型 forward 的**关键字参数**。

    与 `entry_args`（构造参数）分开：很多模型的前向带开关（如 scGPT 的
    `forward(..., CLS=False, MVC=False, ECS=False)`），默认关着的分支不会被 hook 捕获 →
    节点形状补不上；在这里打开对应开关即可（`{"CLS": true, "MVC": true}`）。
    """
    fk = (spec or {}).get("forward_kwargs")
    return dict(fk) if isinstance(fk, dict) else {}


def make_extra_inputs(spec: dict) -> list:
    """`input_spec.extra` 描述的**额外输入**（多输入模型的补参通道）。

    例：scGPT 的 `forward(src, values, src_key_padding_mask)` 需要 3 个入参，而 `input_spec`
    的主 shape 只描述第一个；`extra` 里按顺序补 `[{"shape":[1,1200],"dtype":"float32"},
    {"shape":[1,1200],"dtype":"bool"}]`，调用方按 `model(x, *extras)` 传入。
    """
    import torch

    out = []
    for e in (spec.get("extra") or []):
        shape = list(e.get("shape") or [])
        dtype = getattr(torch, str(e.get("dtype") or "float32"), torch.float32)
        out.append(make_dummy_input(shape, dtype))
    return out


def instantiate(cls, entry_args: dict | None = None):
    """实例化入口类：**IR 里用户补的 entry_args 优先** → 无参 → 常见关键字签名兜底。

    `entry_args` 来自 IR（`PUT /api/projects/{id}/ir/entry_args`）：像 scGPT 的
    `TransformerModel(ntoken, d_model, nhead, d_hid, nlayers, vocab=…)`，参数取自运行期配置与
    数据，固定猜测列表必然失败——由用户给出最小可构造的 args 后，trace/verify 才能实例化。
    均失败时 RuntimeError 说明签名与已尝试的候选。
    """
    attempts: list[str] = []
    last: Exception | None = None
    candidates: list[dict] = []
    if isinstance(entry_args, dict) and entry_args:
        candidates.append(entry_args)      # IR 里用户补的优先（最可能对）
    candidates += [{}] + list(_fallbacks())
    for kwargs in candidates:
        attempts.append("无参" if not kwargs else json.dumps(kwargs, ensure_ascii=False))
        try:
            return cls(**kwargs)
        except TypeError as e:
            if last is None:
                last = e  # 首个（无参构造）TypeError 最富信息：列出缺失的位置参数
        except Exception as e:  # noqa: BLE001 —— 非 TypeError（数据缺失等）优先保留为最终原因
            last = e
    raise RuntimeError(f"入口类 {cls.__name__} 无法实例化（尝试: {attempts}）: {last}")
