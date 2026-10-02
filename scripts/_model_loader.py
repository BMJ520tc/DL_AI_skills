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


def instantiate(cls):
    """实例化入口类：无参优先，常见关键字签名兜底；均失败时 RuntimeError 说明签名。"""
    attempts: list[str] = []
    last: Exception | None = None
    candidates: list[dict] = [{}] + list(_fallbacks())
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
