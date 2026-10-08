"""训练模板 `templates/train.py` 的回归用例。

重点覆盖 **BatchNorm + batch=1**：含 BatchNorm 的模型在 train 模式下单样本前向会抛
`ValueError: Expected more than 1 value per channel when training`。模板有两处会撞上——
① 选输出头的**探针**（固定用 1 个样本）；② 训练循环的**尾批只剩 1 个样本**。
两处都应改走 `model.eval()`（BN 用滑动统计），不丢样本、不改变其余批行为。
（实测来源：GEARS 模块含 BatchNorm1d，在画布上训练即崩。）

需要本机有 torch（`pytest.importorskip` 守门；本机 D:\\python.exe 已装）。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOKEN = pytest.importorskip("torch")
BACKEND = Path(__file__).resolve().parent.parent
TEMPLATE = BACKEND / "templates" / "train.py"

_BN_MODEL = (
    "import torch.nn as nn\n"
    "\n"
    "\n"
    "class GeneratedModel(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(1, 4)\n"
    "        self.bn = nn.BatchNorm1d(4)\n"
    "        self.out = nn.Linear(4, 2)\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.out(self.bn(self.fc(x)))\n"
)


def _run_template(tmp_path: Path, rows: str, epochs: int, batch: int) -> subprocess.CompletedProcess:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model.py").write_text(_BN_MODEL, encoding="utf-8")
    shutil.copyfile(TEMPLATE, run_dir / "train.py")
    data_dir = tmp_path / "ds"
    data_dir.mkdir()
    (data_dir / "preprocessed.csv").write_text(
        "id,split,label,input\n" + rows, encoding="utf-8")
    out = run_dir / "m.json"
    return subprocess.run(
        [sys.executable, "train.py", str(data_dir), str(epochs), str(batch), "0.01", str(out)],
        cwd=str(run_dir), capture_output=True, text=True, timeout=300)


def test_train_template_batchnorm_probe_single_sample(tmp_path):
    """探针固定用 1 个样本（train 模式下 BN 会炸）——2 行 train、batch 4（单批 ≤1 也命中）。"""
    r = _run_template(tmp_path, "1,train,0,0.1\n2,train,1,0.9\n", epochs=2, batch=4)
    assert r.returncode == 0, f"stdout={r.stdout[-1500:]}\nstderr={r.stderr[-1500:]}"


def test_train_template_batchnorm_tail_batch_one(tmp_path):
    """5 行 train、batch=4 → 尾批恰 1 个样本（训练循环的 BN 崩溃点）。"""
    rows = "".join(f"{i},train,{i % 2},0.{i}\n" for i in range(1, 6))
    r = _run_template(tmp_path, rows, epochs=2, batch=4)
    assert r.returncode == 0, f"stdout={r.stdout[-1500:]}\nstderr={r.stderr[-1500:]}"
    assert "Expected more than 1 value per channel" not in (r.stdout + r.stderr)
