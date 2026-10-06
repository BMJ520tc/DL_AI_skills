"""画布网络训练入口（阶段4 4c，模块详细设计 7.5）。

由 backend 拷贝进结构化项目工作区，并以**项目独立环境**执行：

    <env-python> train.py <data_dir> <epochs> <batch_size> <lr> <out_json>

- 数据：<data_dir>/preprocessed.csv（模块三统一列 schema：id/split/label/input [+meta_*]），
  input 单元格为纯字符串。本脚本当前只支持**数值输入**（单值 / 逗号或空格分隔 /
  JSON 数组），图像路径或文本输入会明确报错退出（不静默产出垃圾指标）。
- **多输入模型**：导出代码里的模块级 `MODEL_INPUTS` 声明了入口形参个数与各自 dtype
  （如 scGPT 的 `forward(src, values, src_key_padding_mask)`）。此时第 1 个输入仍读 `input`
  列，其余依次读 `input_1`、`input_2`… 列，并按声明 dtype 造张量（int64→long、bool→bool）；
  缺列或值不合 dtype 一律明确报错。**没有 `MODEL_INPUTS` 的旧导出**按单输入 float32 处理（行为不变）。
- 标签：数值且整型化 → 分类（交叉熵 + accuracy）；数值非整型 → 回归
  （MSE + mae/mse/rmse/r2）；字符串 → 按字典序编码为类别（交叉熵 + accuracy）。
- 切分：split 列为 train 的行作训练集，其余行（test/val 等）作评估集；
  没有评估行时用训练集评估（指标里 eval_samples 与 train_samples 相同）。
- **设备**：自动选——有可用 CUDA 就在 GPU 上训练（模型与每个 batch 的张量都建/搬到 device），
  否则 CPU；启动即打印 `[info] device=…`，并写进结果 JSON 的 `device` 字段（随 run_record 可检索）。
  环境里是否有 CUDA 版 torch 由环境创建时决定（`env_manager.plan_cuda`：有 NVIDIA 驱动就装 `+cuXXX` 轮子）。
- 输出：训练结束把指标写入 <out_json>（CANONICAL_METRICS 键），供 backend 落 run_record；
  训练过程按 epoch 打印进度（进入 train.log）。

依赖 torch（项目独立环境），backend 宿主环境不需要装。
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from model import GeneratedModel

try:                       # 多输入模型由导出代码声明（旧导出没有这个常量 → 单输入 float32）
    from model import MODEL_INPUTS
except ImportError:        # noqa: N812
    MODEL_INPUTS = [{"name": "input", "dtype": "float32"}]


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float32": torch.float32, "float64": torch.float64, "float16": torch.float16,
    "long": torch.long, "int64": torch.long, "int32": torch.int32,
    "int16": torch.int16, "int8": torch.int8, "uint8": torch.uint8, "bool": torch.bool,
}
_INT_DTYPES = {"long", "int", "int64", "int32", "int16", "int8", "uint8"}


def _torch_dtype(name: str) -> torch.dtype:
    return _DTYPE_MAP.get((name or "float32").lower().replace("torch.", ""), torch.float32)


def _flatten_raw(raw) -> list:
    """JSON 数组拍平（允许嵌套一层的矩阵写法），保留原始标量（不做类型转换）。"""
    out: list = []
    stack = [raw]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _split_cell(cell: str) -> list[str]:
    """输入单元格 → 原始值字符串列表（JSON 数组 / 逗号或空格分隔）。"""
    cell = (cell or "").strip()
    if not cell:
        raise SystemExit("preprocessed.csv 存在空输入单元格")
    if cell.startswith("["):
        try:
            raw = json.loads(cell)
        except json.JSONDecodeError as e:
            raise SystemExit(f"输入不是合法 JSON 数组：{cell[:60]}") from e
        if not isinstance(raw, list):
            raise SystemExit(f"输入的 JSON 需为数组：{cell[:60]}")
        return [str(v) for v in _flatten_raw(raw)]
    return cell.replace(",", " ").split()


def parse_cell(cell: str, dtype: str) -> list:
    """按声明的 dtype 解析一个输入单元格（多输入模型各输入可能 dtype 不同）。"""
    values = _split_cell(cell)
    dt = (dtype or "float32").lower().replace("torch.", "")
    if dt == "bool":
        return [v.strip().lower() in ("1", "true", "yes") for v in values]
    if dt in _INT_DTYPES:
        try:
            return [int(float(v)) for v in values]
        except ValueError as e:
            raise SystemExit(f"输入需要整数（dtype={dtype}）：{cell[:60]}") from e
    try:
        return [float(v) for v in values]
    except ValueError as e:
        raise SystemExit(
            "训练脚本当前只支持数值型输入：input 单元格需为单值、逗号/空格分隔或 JSON 数组"
            "（图像路径 / 文本输入暂不支持，请先在模块三预处理成数值特征）"
        ) from e


# ---------------------------------------------------------------------------
# 标签编码
# ---------------------------------------------------------------------------

def parse_labels(raw: list[str]) -> tuple[str, torch.Tensor, list[str]]:
    """→ (mode, labels 张量, 类别名表)。数值整型/字符串 → 分类；数值非整型 → 回归。"""
    try:
        floats = [float(v) for v in raw]
    except ValueError:
        classes = sorted(set(raw))
        return "classification", torch.tensor(
            [classes.index(v) for v in raw], dtype=torch.long
        ), classes

    integral = all(f == int(f) for f in floats)
    if integral:
        classes = sorted({int(f) for f in floats})
        return "classification", torch.tensor(
            [classes.index(int(f)) for f in floats], dtype=torch.long
        ), [str(c) for c in classes]
    return "regression", torch.tensor(floats, dtype=torch.float32), []


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def _output_tensors(out) -> list[torch.Tensor]:
    """模型输出 → 候选张量列表（dict 取所有张量值；tuple 取全部张量；单张量取自身）。"""
    if isinstance(out, dict):
        return [v for v in out.values() if torch.is_tensor(v)]
    if isinstance(out, tuple):
        return [t for t in out if torch.is_tensor(t)]
    return [out] if torch.is_tensor(out) else []


def pick_output(out, mode: str, num_classes: int) -> tuple[int, str]:
    """按任务在候选输出里挑一个头，返回 `(下标, 口径)`；挑不出就**明确报错**（不猜）。

    多输出头模型（实测 scGPT 的根节点返回 `dict(mlm_output=…, cell_emb=…, cls_output=…,
    mvc_output=…)`）该用哪个头是建模决策，这里只做**保守的按形状匹配**：
    - 分类：优先「最后一维 = 类别数」（交叉熵）；否则二分类时取「最后一维 = 1」（BCE logit）；
    - 回归：优先「每样本 1 个值」；只有唯一候选时用它。
    都不匹配即报错并列出候选形状，让用户改任务类型或模型，而不是抛张量维度错误。
    """
    cands = _output_tensors(out)
    if not cands:
        raise SystemExit(f"模型输出里没有张量（{type(out).__name__}）；无法训练")
    if mode == "classification":
        for i, t in enumerate(cands):
            if t.dim() >= 2 and int(t.shape[-1]) == num_classes:
                return i, "ce"
        if num_classes == 2:
            for i, t in enumerate(cands):
                if t.dim() >= 2 and int(t.shape[-1]) == 1:
                    return i, "bce"
        raise SystemExit(
            f"没有可用的分类输出头（类别数 {num_classes}）：候选形状 "
            f"{[tuple(t.shape) for t in cands]}；请改用回归任务或调整模型")
    for i, t in enumerate(cands):
        if int(t.shape[-1]) == 1:
            return i, "mse"
    if len(cands) == 1:
        return 0, "mse"
    raise SystemExit(f"没有可用的回归输出头：候选形状 {[tuple(t.shape) for t in cands]}；请调整模型")


def _predict(out, head_idx: int, head_kind: str, mode: str) -> torch.Tensor:
    """按已选定的输出头出预测（分类给类别下标 / 回归给数值）。"""
    t = _output_tensors(out)[head_idx]
    if mode == "regression":
        return t.reshape(-1)
    if head_kind == "bce":
        return (torch.sigmoid(t.reshape(-1)) > 0.5).long()
    return t.argmax(dim=1)


def _loss_of(out, yb, head_idx: int, head_kind: str, mode: str, criterion) -> torch.Tensor:
    t = _output_tensors(out)[head_idx]
    if mode == "regression":
        return criterion(t.reshape(-1), yb)
    if head_kind == "bce":
        return criterion(t.reshape(-1), yb.float())
    return criterion(t, yb)


def _pad_stack(rows: list[list], dtype: torch.dtype,
               device: torch.device | None = None) -> torch.Tensor:
    """变长序列右补零成批（标量特征为长度 1）；bool 也用 0/False 补齐；张量直接建在目标设备上。"""
    dev = device or torch.device("cpu")
    tensors = [torch.tensor(r, dtype=dtype, device=dev) for r in rows]
    length = max(t.shape[0] for t in tensors)
    padded = [
        t if t.shape[0] == length
        else torch.cat([t, torch.zeros(length - t.shape[0], dtype=dtype, device=dev)])
        for t in tensors
    ]
    return torch.stack(padded)


def make_batch(idx: list[int], xs: list[list[list]], dtypes: list[str],
               y: torch.Tensor, device: torch.device | None = None,
               ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """每个输入各成一批（多输入模型逐个喂；单输入时长度 1 的列表）。

    张量**直接建在目标设备上**（`device` 缺省 CPU）：模型在 GPU 而数据在 CPU 会报设备不一致。
    """
    dev = device or torch.device("cpu")
    tensors = [_pad_stack([xs[k][i] for i in idx], _torch_dtype(dtypes[k]), dev)
               for k in range(len(xs))]
    return tensors, y[idx].to(dev)


def eval_metrics(mode: str, model: nn.Module, xs: list[list[list]], dtypes: list[str],
                 idx: list[int], y: torch.Tensor, batch_size: int, loss: float,
                 num_classes: int, head_idx: int, head_kind: str,
                 device: torch.device | None = None) -> dict:
    """评估集指标（CANONICAL_METRICS 键）。"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            xb, yb = make_batch(batch_idx, xs, dtypes, y, device)
            preds.append(_predict(model(*xb), head_idx, head_kind, mode))
            labels.append(yb)
    preds = torch.cat(preds)
    labels = torch.cat(labels)

    if mode == "classification":
        correct = int((preds == labels).sum().item())
        metrics = {"loss": loss, "accuracy": correct / max(len(labels), 1)}
    else:
        diff = preds - labels
        mse = float((diff ** 2).mean().item())
        ss_res = float((diff ** 2).sum().item())
        ss_tot = float(((labels - labels.mean()) ** 2).sum().item())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
        metrics = {
            "loss": loss,
            "mae": float(diff.abs().mean().item()),
            "mse": mse,
            "rmse": mse ** 0.5,
            "r2": r2,
        }
    model.train()
    return metrics


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit("用法: train.py <data_dir> <epochs> <batch_size> <lr> <out_json>")
    data_dir = Path(sys.argv[1])
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    lr = float(sys.argv[4])
    out_json = Path(sys.argv[5])

    csv_path = data_dir / "preprocessed.csv"
    if not csv_path.exists():
        raise SystemExit(f"数据文件不存在：{csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"数据为空：{csv_path}")
    if rows and rows[0].get("label") is None:
        raise SystemExit("preprocessed.csv 缺少 label 列")
    # 多输入模型：MODEL_INPUTS 逐个声明列名与 dtype（第 1 个用 input，其余 input_1/input_2…）
    names = [str((m or {}).get("name") or "input") for m in (MODEL_INPUTS or [])] or ["input"]
    dtypes = [str((m or {}).get("dtype") or "float32") for m in (MODEL_INPUTS or [])] or ["float32"]
    missing = [n for n in names if rows[0].get(n) is None]
    if missing:
        raise SystemExit(
            f"该模型是 {len(names)} 输入，需要列 {'、'.join(names)}；preprocessed.csv 缺少：{'、'.join(missing)}"
            "（第 1 个输入读 input 列，其余依次读 input_1、input_2…）")
    xs = [[parse_cell(r[n], dt) for r in rows] for n, dt in zip(names, dtypes)]
    mode, y, classes = parse_labels([r["label"].strip() for r in rows])
    num_classes = len(classes) if mode == "classification" else 0
    if mode == "classification" and num_classes < 2:
        raise SystemExit(f"分类标签只有 {num_classes} 类，无法训练分类模型")

    train_idx = [i for i, r in enumerate(rows) if (r.get("split") or "train") == "train"]
    eval_idx = [i for i in range(len(rows)) if i not in train_idx] or train_idx

    torch.manual_seed(42)
    # **自动选设备**：有可用的 CUDA 就用 GPU（环境装了 CUDA 版 torch 才为真——见 env_manager 的
    # `plan_cuda`：有 NVIDIA 驱动就装 `+cuXXX` 轮子）。启动即打印，进 train.log，作为可见证据。
    # 设备：默认自动（有 CUDA 用 GPU）；`TRAIN_DEVICE=cpu|cuda` 可显式覆盖
    # （Windows 上 `CUDA_VISIBLE_DEVICES=""` 不一定真能隐藏 GPU，实测无效 → 这个开关才是确定的）。
    want = (os.environ.get("TRAIN_DEVICE") or "auto").strip().lower()
    if want == "cpu":
        device = torch.device("cpu")
    elif want == "cuda" and not torch.cuda.is_available():
        raise SystemExit("TRAIN_DEVICE=cuda 但本机/本环境没有可用的 CUDA")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # 再生成的模型里有**常量算子**（`torch.arange(size(0))` 造 labels、`torch.eye(...)` 造掩码），
        # 它们不带 device → 默认建在 **CPU**，与 GPU 上的张量混算会报
        # `Expected all tensors to be on the same device`（实测 scGPT 的 CCE 分支）。把默认设备设成
        # cuda（PyTorch 2.0+ 官方 API），这些常量就跟着建在 GPU 上；显式给了 device 的调用不受影响。
        torch.set_default_device(device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[info] device=cuda（{torch.cuda.get_device_name(0)}，显存 {free / 2**30:.1f}/"
              f"{total / 2**30:.1f} GiB 可用）")
    else:
        print("[info] device=cpu（无可用 CUDA：装的是 CPU 版 torch 或本机没有 GPU）")
    model = GeneratedModel().to(device)
    # 先探一次输出、选定输出头：多输出头模型按形状匹配选（选不出即明确报错，不猜）
    with torch.no_grad():
        probe_xb, _ = make_batch((train_idx or [0])[:1], xs, dtypes, y, device)
        head_idx, head_kind = pick_output(model(*probe_xb), mode, num_classes)
    if mode == "classification" and head_kind == "bce":
        print("[info] 分类输出头最后一维为 1 → 按二分类 BCE 处理（>0.5 判正类）")
    criterion = (nn.BCEWithLogitsLoss() if head_kind == "bce"
                 else nn.CrossEntropyLoss() if mode == "classification" else nn.MSELoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        _t0 = time.time()
        total, count = 0.0, 0
        order = train_idx[:]
        random.Random(epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            xb, yb = make_batch(batch_idx, xs, dtypes, y, device)
            optimizer.zero_grad()
            loss = _loss_of(model(*xb), yb, head_idx, head_kind, mode, criterion)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_idx)
            count += len(batch_idx)
        last_loss = total / max(count, 1)
        _dt = time.time() - _t0
        print(f"[epoch {epoch}/{epochs}] loss={last_loss:.6f}  用时 {_dt:.1f}s")


    metrics = eval_metrics(mode, model, xs, dtypes, eval_idx, y, batch_size, last_loss,
                           num_classes, head_idx, head_kind, device)
    result = {
        "mode": mode,
        "classes": classes,
        "device": str(device),          # 用了 CPU 还是 GPU：随指标进 run_record，可检索
        "output_head": {"index": head_idx, "kind": head_kind},
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_samples": len(train_idx),
        "eval_samples": len(eval_idx),
        "metrics": metrics,
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics -> {out_json}")


if __name__ == "__main__":
    main()
