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
import re
import sys
from collections.abc import Mapping
from pathlib import Path

_INSTANTIATE_FALLBACKS = (
    {"num_classes": 10},
    {"num_labels": 10},
    {"n_classes": 10},
    {"sizes": [64, 128, 10]},
    {"dims": [64, 128, 10]},
    {"hidden_size": 64},
)


MAKE_INPUTS_FILENAME = "make_inputs.py"
MAKE_MODEL_FILENAME = "make_model.py"


def make_model_path(source_dir: "str | None") -> "Path | None":
    """项目约定：`<工作区>/reports/make_model.py`（工作区 = source 的父目录）。"""
    if not source_dir:
        return None
    path = Path(source_dir).resolve().parent / "reports" / MAKE_MODEL_FILENAME
    return path if path.is_file() else None


def _load_make_model(path: Path):
    """加载约定脚本里的 `build_model(entry_class, checkpoint=None)`；拿不到就 None。"""
    try:
        spec = importlib.util.spec_from_file_location("_project_make_model", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {path.name} 加载失败（{type(e).__name__}: {e}）", file=sys.stderr)
        return None
    fn = getattr(module, "build_model", None)
    return fn if callable(fn) else None


def make_inputs_path(source_dir: "str | None") -> "Path | None":
    """项目约定：`<工作区>/reports/make_inputs.py`（工作区 = source 的父目录）。"""
    if not source_dir:
        return None
    path = Path(source_dir).resolve().parent / "reports" / MAKE_INPUTS_FILENAME
    return path if path.is_file() else None


def _load_make_inputs(path: Path):
    """加载约定脚本里的 `build_inputs(model)`；没有该函数 / 加载失败 → None（并如实提示）。"""
    try:
        spec = importlib.util.spec_from_file_location("_project_make_inputs", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {path.name} 加载失败（{type(e).__name__}: {e}），回退默认 dummy 输入",
              file=sys.stderr)
        return None
    fn = getattr(module, "build_inputs", None)
    if not callable(fn):
        print(f"[warn] {path.name} 未定义 build_inputs(model)，回退默认 dummy 输入", file=sys.stderr)
        return None
    return fn


def build_inputs(model, spec: dict, source_dir: "str | None" = None) -> tuple:
    """造喂给 `model(...)` 的**位置实参**。

    优先用项目的 `make_inputs.py`（`build_inputs(model) -> tuple`）——真实仓库的 forward 常常
    不吃「一个张量」：boltz 要的是 `feats` **特征字典**（键多、维度各异，由 featurizer 造），
    平台按 shape+dtype 喂的 dummy 张量必然 `IndexError: too many indices`（实测踩到）。
    没有约定脚本时，退回原来的 shape+dtype dummy 张量 + `input_spec.extra` 额外输入。
    """
    path = make_inputs_path(source_dir)
    if path is not None:
        build = _load_make_inputs(path)
        if build is not None:
            produced = build(model)
            return produced if isinstance(produced, tuple) else (produced,)
    import torch

    shape = list(spec.get("shape") or [])
    dtype = getattr(torch, str(spec.get("dtype") or "float32"), torch.float32)
    return (make_dummy_input(shape, dtype), *make_extra_inputs(spec))


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


def _find_checkpoint(source_dir: "str | None") -> "Path | None":
    """找一个可用的 checkpoint 文件。

    顺序：显式环境变量 `DECOMPOSE_CHECKPOINT` → 项目工作区 `data/` 下的 `*.ckpt`/`*.pt`
    （工作区 = source 的父目录；项目数据按约定放那儿）→ 工作区顶层。
    """
    hint = os.environ.get("DECOMPOSE_CHECKPOINT")
    if hint:
        p = Path(hint)
        if p.is_file():
            return p
    if not source_dir:
        return None
    ws = Path(source_dir).resolve().parent
    for root, recursive in ((ws / "data", True), (ws, False)):
        if not root.is_dir():
            continue
        found = sorted(root.glob("**/*.ckpt") if recursive else root.glob("*.ckpt"))
        found += sorted(root.glob("**/*.pt") if recursive else root.glob("*.pt"))
        if found:
            return found[0]
    return None


_EXCESS_KWARG_RE = re.compile(r"unexpected keyword argument '([^']+)'")


def _drop_key_deep(obj, key: str) -> bool:
    """从嵌套配置里删掉某个键（深度优先，删第一个命中）。返回是否删到。

    嵌套配置实测是 OmegaConf 的 `DictConfig`（Mapping，但不是 dict），所以按 `Mapping` 认。
    """
    if isinstance(obj, Mapping):
        if key in obj:
            try:
                del obj[key]
            except Exception:  # noqa: BLE001 —— 只读/struct 模式的配置删不动，当没删到
                return False
            return True
        for value in list(obj.values()):
            if _drop_key_deep(value, key):
                return True
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            if _drop_key_deep(value, key):
                return True
    return False


def _checkpoint_entry_args(source_dir: "str | None") -> "dict | None":
    """从 checkpoint 的 `hyper_parameters` 取**权威构造参数**（Lightning 的 save_hyperparameters 存这里）。

    为什么需要：Lightning 系仓库（boltz 就是）的 `__init__` 要一二十个结构化参数
    （`atom_s`/`training_args`/`msa_args`/…），代码里**没有直接构造调用**——只有
    `Model.load_from_checkpoint(...)`，参数藏在 ckpt 里。agent 猜不出来 → 实例化失败 →
    真实追踪起不来 → 只能退回「让 LLM 猜 IR」（实测拆出 7 个孤立死模块）。
    """
    ckpt = _find_checkpoint(source_dir)
    if ckpt is None:
        return None
    try:
        import torch

        obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 —— 权重读不了就当没有，不改变原有报错路径
        return None
    if not isinstance(obj, dict):
        return None
    for key in ("hyper_parameters", "hparams"):
        hp = obj.get(key)
        if isinstance(hp, dict) and hp:
            # validators 不可序列化；Boltz2 的 save_hyperparameters 本来就 ignore 它
            return {k: v for k, v in hp.items() if k != "validators"}
    return None


def _prefer_repo_package(fpath: Path) -> None:
    """把仓库源码的**包根父目录**插到 `sys.path` 最前，让 `import <包名>` 解析到仓库这版。

    为什么需要：很多仓库把自己的包也发到 PyPI，依赖清单一引就把**安装版**装进 site-packages
    （boltz 实测装成 `boltz-2.2.1` 实体包，不是 editable）。此时从**仓库文件**加载入口类，
    它内部的 `import boltz.xxx` 却会命中 site-packages 那版 → 两版混着跑，报
    `AttentionPairBias.forward() got an unexpected keyword argument` 这种"缝合怪"错误
    （实测踩到：追踪的根本不是仓库这份代码）。
    """
    pkg = fpath.parent
    top: Path | None = None
    while (pkg / "__init__.py").is_file():
        top = pkg
        pkg = pkg.parent
    if top is None:
        return
    root = str(top.parent)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


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

    _prefer_repo_package(fpath)

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


def instantiate(cls, entry_args: dict | None = None, source_dir: "str | None" = None):
    """实例化入口类：**IR/用户补的 entry_args** → 无参 → 常见关键字签名 → **checkpoint 的 hparams**。

    `entry_args` 来自 IR（`PUT /api/projects/{id}/ir/entry_args`）：像 scGPT 的
    `TransformerModel(ntoken, d_model, nhead, d_hid, nlayers, vocab=…)`，参数取自运行期配置与
    数据，固定猜测列表必然失败——由用户给出最小可构造的 args 后，trace/verify 才能实例化。

    都失败时再试 **checkpoint 的 `hyper_parameters`**（Lightning 的 save_hyperparameters 存的
    权威构造参数）——boltz 这类「只 load_from_checkpoint、代码里不直接构造」的仓库靠它才能起来。
    `source_dir` 用于定位权重（见 `_find_checkpoint`）；读权重较贵，故放在最后兜底。
    """
    attempts: list[str] = []
    last: Exception | None = None

    # ① 项目自己的构造脚本最权威：仓库常把构造方式写在 `load_from_checkpoint(ckpt, **覆盖)` 里，
    #    ckpt 的 hparams 并不足以还原一个能跑的模型（实测 boltz：ckpt 的 pairformer_args 没有 v2，
    #    而当前代码的 forward 无条件用 v2 才有的参数）。
    builder = None
    model_script = make_model_path(source_dir)
    if model_script is not None:
        builder = _load_make_model(model_script)
    if builder is not None:
        attempts.append("make_model.py")
        try:
            built = builder(cls, _find_checkpoint(source_dir))
            if built is not None:
                return built
            attempts[-1] = "make_model.py（返回 None，回退其它方式）"
        except Exception as e:  # noqa: BLE001
            last = e
            attempts[-1] = f"make_model.py（失败: {type(e).__name__}: {str(e)[:120]}）"

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

    hparams = _checkpoint_entry_args(source_dir)
    if hparams is not None:
        # ckpt 可能是**另一个版本的代码**存下来的：hparams 里会有当前 __init__ 不认的键
        # （boltz 实测 `chain_sampling_args`）→ 按签名过滤，别为一个多余键整条路失败。
        filtered = accepted_kwargs(cls.__init__, hparams)
        attempts.append(
            f"checkpoint hparams（{len(hparams)} 个，签名接受 {len(filtered)} 个）"
        )
        try:
            return cls(**filtered)
        except TypeError as e:
            last = e
            # 嵌套配置（如 diffusion_process_args）里也可能有当前代码不认的键——
            # 错误信息会点名那个键，从嵌套里删掉再试。ckpt 与代码版本不一致时很常见，
            # 删掉的键本来就用不了，退回当前代码的默认值反而更贴近本仓库的结构。
            for _ in range(30):
                m = _EXCESS_KWARG_RE.search(str(last))
                if not m or not _drop_key_deep(filtered, m.group(1)):
                    break
                try:
                    return cls(**filtered)
                except TypeError as e2:
                    last = e2
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"入口类 {cls.__name__} 无法实例化（尝试: {attempts}）: {last}")
