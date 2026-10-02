"""画布网络训练入口（阶段4 4c，模块详细设计 7.5）。

由 backend 拷贝进结构化项目工作区，并以**项目独立环境**执行：

    <env-python> train.py <data_dir> <epochs> <batch_size> <lr> <out_json>

- 数据：<data_dir>/preprocessed.csv（模块三统一列 schema：id/split/label/input [+meta_*]），
  input 单元格为纯字符串。本脚本当前只支持**数值输入**（单值 / 逗号或空格分隔 /
  JSON 数组），图像路径或文本输入会明确报错退出（不静默产出垃圾指标）。
- 标签：数值且整型化 → 分类（交叉熵 + accuracy）；数值非整型 → 回归
  （MSE + mae/mse/rmse/r2）；字符串 → 按字典序编码为类别（交叉熵 + accuracy）。
- 切分：split 列为 train 的行作训练集，其余行（test/val 等）作评估集；
  没有评估行时用训练集评估（指标里 eval_samples 与 train_samples 相同）。
- 输出：训练结束把指标写入 <out_json>（CANONICAL_METRICS 键），供 backend 落 run_record；
  训练过程按 epoch 打印进度（进入 train.log）。

依赖 torch（项目独立环境），backend 宿主环境不需要装。
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

from model import GeneratedModel


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------

def _flatten(raw) -> list[float]:
    """JSON 数组拍平（允许嵌套一层的矩阵写法）。"""
    out: list[float] = []
    stack = [raw]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            out.append(float(item))
    return out


def _to_floats(cells: list[str]) -> list[float]:
    try:
        return [float(c) for c in cells]
    except ValueError as e:
        raise SystemExit(
            "训练脚本当前只支持数值型输入：input 单元格需为单值、逗号/空格分隔或 JSON 数组"
            "（图像路径 / 文本输入暂不支持，请先在模块三预处理成数值特征）"
        ) from e


def parse_input(cell: str) -> list[float]:
    """input 单元格 → 特征向量。"""
    cell = (cell or "").strip()
    if not cell:
        raise SystemExit("preprocessed.csv 存在空 input 单元格")
    if cell.startswith("["):
        try:
            raw = json.loads(cell)
        except json.JSONDecodeError as e:
            raise SystemExit(f"input 不是合法 JSON 数组：{cell[:60]}") from e
        if not isinstance(raw, list):
            raise SystemExit(f"input 的 JSON 需为数组：{cell[:60]}")
        return _flatten(raw)
    parts = cell.replace(",", " ").split()
    if len(parts) > 1:
        return _to_floats(parts)
    return _to_floats([cell])


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

def make_batch(idx: list[int], x: list[list[float]], y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """变长序列右补零成批（标量特征为长度 1）。"""
    tensors = [torch.tensor(x[i], dtype=torch.float32) for i in idx]
    length = max(t.shape[0] for t in tensors)
    padded = torch.stack([nn.functional.pad(t, (0, length - t.shape[0])) for t in tensors])
    return padded, y[idx]


def eval_metrics(mode: str, model: nn.Module, x: list[list[float]], y: torch.Tensor,
                 batch_size: int, loss: float, num_classes: int) -> dict:
    """评估集指标（CANONICAL_METRICS 键）。"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch_idx = list(range(start, min(start + batch_size, len(x))))
            xb, yb = make_batch(batch_idx, x, y)
            out = model(xb)
            if isinstance(out, tuple):
                out = out[0]
            if mode == "classification":
                preds.append(out.argmax(dim=1))
            else:
                preds.append(out.reshape(-1))
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
    if any(r.get("input") is None or r.get("label") is None for r in rows):
        raise SystemExit("preprocessed.csv 缺少 input/label 列")

    x = [parse_input(r["input"]) for r in rows]
    mode, y, classes = parse_labels([r["label"].strip() for r in rows])
    num_classes = len(classes) if mode == "classification" else 0
    if mode == "classification" and num_classes < 2:
        raise SystemExit(f"分类标签只有 {num_classes} 类，无法训练分类模型")

    train_idx = [i for i, r in enumerate(rows) if (r.get("split") or "train") == "train"]
    eval_idx = [i for i in range(len(rows)) if i not in train_idx] or train_idx

    torch.manual_seed(42)
    model = GeneratedModel()
    criterion = nn.CrossEntropyLoss() if mode == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        order = train_idx[:]
        random.Random(epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            xb, yb = make_batch(batch_idx, x, y)
            optimizer.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                out = out[0]
            if mode == "classification":
                loss = criterion(out, yb)
            else:
                loss = criterion(out.reshape(-1), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_idx)
            count += len(batch_idx)
        last_loss = total / max(count, 1)
        print(f"[epoch {epoch}/{epochs}] loss={last_loss:.6f}")

    metrics = eval_metrics(mode, model, x, y, batch_size, last_loss, num_classes)
    result = {
        "mode": mode,
        "classes": classes,
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
