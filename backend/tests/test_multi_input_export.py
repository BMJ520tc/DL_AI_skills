"""多输入模型的「画布导出入口 + 训练模板」回归（2026-10-06 用户实测的坏结构化项目）。

背景：scGPT 的结构化项目（`599a39f6…`，父项目 scGPT）导出的代码**跑不起来**——
根类 `Decomp_transformer_model.forward(self, x0, x1)` 是两输入，而画布入口 `GeneratedModel`
写死 `def forward(self, x): return self.root(x)`，实跑即
`TypeError: … missing 1 required positional argument: 'x1'`；而训练模板只认 `GeneratedModel`。
（⑤ 两步验证能过，因为它直接实例化根类 `Decomp_<root>` 并按 input_spec 造多个输入。）

口径：
- 入口 `forward` 形参个数 = 根节点 forward 形参个数（`input_spec.inputs`），并写出模块级
  `MODEL_INPUTS`（列名 + dtype）供训练模板按多输入/按 dtype 喂数据；
- 训练模板第 1 个输入读 `input` 列，其余读 `input_1`…；缺列 / 值不合 dtype / 输出形状与任务不符
  一律**明确报错**，不静默；
- 旧导出（无 `MODEL_INPUTS`）行为不变（单输入 float32）。
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.services import network_export
from app.services.ir_schema import external_input_conflicts

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "train.py"


# ---------------------------------------------------------------- A 入口签名

def _ir(inputs, extras, root="net"):
    return {
        "schema_version": "1.0", "source_file": "m.py", "entry_class": "Net",
        "task_type": "classification", "root_id": root,
        "input_spec": {"shape": [1, 4], "dtype": "int64", "extra": extras, "inputs": inputs},
        "nodes": [{"id": root, "kind": "module", "class_name": "Net", "parent_id": None, "module_path": ""}],
        "edges": [],
    }


def test_entry_inputs_single_when_no_inputs_declared():
    assert network_export._entry_inputs(_ir([], [])) == [{"name": "input", "dtype": "int64"}]


def test_entry_inputs_multi_uses_extra_dtypes():
    got = network_export._entry_inputs(_ir(["a", "b"], [{"shape": [1, 4], "dtype": "float32"}]))
    assert got == [{"name": "input", "dtype": "int64"}, {"name": "input_1", "dtype": "float32"}]


def test_entry_class_signature_matches_root_arity_and_writes_manifest():
    code = network_export._with_entry_class("import torch\n", _ir(["a", "b"], [{"shape": [1, 4], "dtype": "float32"}]))
    assert "def forward(self, x0, x1):" in code
    assert "return self.root(x0, x1)" in code
    assert '"name": "input_1"' in code and '"dtype": "float32"' in code
    # 单输入模型仍是一个形参
    single = network_export._with_entry_class("import torch\n", _ir([], []))
    assert "def forward(self, x0):" in single


# ---------------------------------------------------------------- C 死边检测

def test_external_input_conflicts_flags_foreign_in_edge():
    ir = _ir(["enc", "val"], [])
    ir["nodes"] += [
        {"id": "enc", "kind": "module", "class_name": "E", "parent_id": "net", "module_path": "enc"},
        {"id": "val", "kind": "module", "class_name": "V", "parent_id": "net", "module_path": "val"},
        {"id": "val_prep", "kind": "op", "class_name": "unsqueeze", "parent_id": "val"},
    ]
    ir["edges"] = [{"from": "enc", "to": "val_prep"}]
    warns = external_input_conflicts(ir)
    assert len(warns) == 1 and "val_prep" in warns[0] and "不生效" in warns[0]
    # 子树内部连线不算
    ir["edges"] = [{"from": "val_prep", "to": "val"}]
    assert external_input_conflicts(ir) == []


# ---------------------------------------------------------------- B 训练模板（真跑）

_MODEL_MULTI = '''
import torch
import torch.nn as nn

MODEL_INPUTS = [{"name": "input", "dtype": "int64"}, {"name": "input_1", "dtype": "float32"}]


class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.fc = nn.Linear(4 + 1, 2)

    def forward(self, x0, x1):
        e = self.emb(x0).mean(dim=1)
        v = x1.mean(dim=1, keepdim=True)
        return self.fc(torch.cat([e, v], dim=1))
'''


def _write_case(tmp: Path, model_src: str, rows: list[dict], header: list[str]) -> Path:
    (tmp / "model.py").write_text(model_src, encoding="utf-8")
    (tmp / "train.py").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    data = tmp / "data"
    data.mkdir(exist_ok=True)
    with (data / "preprocessed.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return data


def _run(tmp: Path, data: Path, epochs: int = 2) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(tmp / "train.py"), str(data), str(epochs), "4", "0.05", str(tmp / "out.json")],
        cwd=str(tmp), capture_output=True, text=True, timeout=300,
    )


def test_train_template_feeds_multi_inputs(tmp_path):
    """多输入模型：input + input_1 两列 → 训练跑通并产出指标。"""
    rows = [
        {"id": i, "split": "train" if i < 8 else "test", "label": i % 2,
         "input": "[1, 2, 3, 4]" if i % 2 == 0 else "[4, 3, 2, 1]",
         "input_1": "[0.1, 0.2, 0.3, 0.4]" if i % 2 == 0 else "[0.9, 0.8, 0.7, 0.6]"}
        for i in range(10)
    ]
    tmp = tmp_path / "multi"
    tmp.mkdir()
    data = _write_case(tmp, _MODEL_MULTI, rows, ["id", "split", "label", "input", "input_1"])

    res = _run(tmp, data)
    assert res.returncode == 0, res.stdout + res.stderr
    metrics = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "classification"
    assert 0.0 <= metrics["metrics"]["accuracy"] <= 1.0


_MODEL_DICT = '''
import torch
import torch.nn as nn

MODEL_INPUTS = [{"name": "input", "dtype": "int64"}, {"name": "input_1", "dtype": "float32"}]


class GeneratedModel(nn.Module):
    """多输出头（dict）模型，仿 scGPT：首个张量不是分类头。"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.fc = nn.Linear(4 + 1, 1)

    def forward(self, x0, x1):
        e = self.emb(x0).mean(dim=1)
        v = x1.mean(dim=1, keepdim=True)
        logit = self.fc(torch.cat([e, v], dim=1))
        return {"expr": torch.zeros(e.shape[0], 1200), "cls": logit}
'''


def test_train_template_selects_head_for_dict_output(tmp_path):
    """多输出头（dict）：按形状选中「最后一维=1」的分类头 → 二分类 BCE，训练跑通。

    回归 scGPT 那种 `dict(mlm_output=[B,1200], cls_output=[B,1], …)`：原先按「首个张量」取会
    与标签形状不符；现在按任务形状匹配选头，并在结果里记 `output_head`。
    """
    rows = [
        {"id": i, "split": "train" if i < 8 else "test", "label": i % 2,
         "input": "[1, 2, 3, 4]" if i % 2 == 0 else "[4, 3, 2, 1]",
         "input_1": "[0.1, 0.2, 0.3, 0.4]" if i % 2 == 0 else "[0.9, 0.8, 0.7, 0.6]"}
        for i in range(10)
    ]
    tmp = tmp_path / "dict"
    tmp.mkdir()
    data = _write_case(tmp, _MODEL_DICT, rows, ["id", "split", "label", "input", "input_1"])

    res = _run(tmp, data)
    assert res.returncode == 0, res.stdout + res.stderr
    metrics = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
    assert metrics["output_head"]["kind"] == "bce"          # 选中 cls 头（最后一维 1）
    assert 0.0 <= metrics["metrics"]["accuracy"] <= 1.0


def test_train_template_reports_missing_input_column(tmp_path):
    """缺 input_1 列 → 明确报错（而不是 TypeError / 静默）。"""
    rows = [{"id": i, "split": "train", "label": i % 2, "input": "[1, 2, 3, 4]"} for i in range(6)]
    tmp = tmp_path / "missing"
    tmp.mkdir()
    data = _write_case(tmp, _MODEL_MULTI, rows, ["id", "split", "label", "input"])

    res = _run(tmp, data, epochs=1)
    assert res.returncode != 0
    assert "input_1" in (res.stdout + res.stderr)


def test_train_template_single_input_backward_compatible(tmp_path):
    """无 MODEL_INPUTS 的旧导出：单输入 float32，行为不变。"""
    model = (
        "import torch\nimport torch.nn as nn\n\n"
        "class GeneratedModel(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 2)\n"
        "    def forward(self, x):\n        return self.fc(x)\n"
    )
    rows = [{"id": i, "split": "train", "label": i % 2,
             "input": "[1, 2, 3, 4]"} for i in range(6)]
    tmp = tmp_path / "single"
    tmp.mkdir()
    data = _write_case(tmp, model, rows, ["id", "split", "label", "input"])

    res = _run(tmp, data, epochs=2)
    assert res.returncode == 0, res.stdout + res.stderr
    assert json.loads((tmp / "out.json").read_text(encoding="utf-8"))["metrics"]["accuracy"] >= 0.0
