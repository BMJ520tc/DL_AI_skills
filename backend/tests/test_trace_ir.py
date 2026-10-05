"""由**真实追踪**生成 IR（`scripts/trace_ir.py`）的回归用例。

链路：真实模型 → `torch.export` + 节点自带的 `nn_module_stack` → 模块树/算子归属/数据流 → IR。
这是「不再让 LLM 猜结构」的治本方向：模块树、边、叶子参数都由追踪机械取到。

本用例只锁**已经跑通的那档**（简单模型：结构合法 + 能再生成）；复杂模型（多输入叶子、
只有参数/外部输入的算子、容器子节点边规则）仍受 IR 表达力限制，属已知边界。
用真实 torch 跑（`D:\\python.exe` 带 torch）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.config import PROJECT_ROOT
from app.services import ir_codegen
from app.services.ir_schema import incomplete_ir, validate_ir

SCRIPT = PROJECT_ROOT / "scripts" / "trace_ir.py"

SOURCE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "\n"
    "class Block(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.lin = nn.Linear(4, 4)\n"
    "        self.act = nn.ReLU()\n"
    "    def forward(self, x):\n"
    "        return self.act(self.lin(x))\n"
    "\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.block = Block()\n"
    "        self.drop = nn.Dropout(0.1)\n"
    "    def forward(self, x):\n"
    "        return self.drop(self.block(x))\n"
)

# 已知边界：根层算子吃**外部输入**（`add(..., x)`）——当前 IR 没有「外部输入」节点，表达不了
SOURCE_EXTERNAL_INPUT_OP = SOURCE.replace(
    "        return self.drop(self.block(x))\n",
    "        return torch.add(self.drop(self.block(x)), x)\n")


def _run(tmp_path: Path, source: str = SOURCE) -> dict:
    src = tmp_path / "source"
    src.mkdir()
    (src / "model.py").write_text(source, encoding="utf-8")
    ref = {"entry_class": "Net", "source_file": "model.py", "task_type": "classification",
           "input_spec": {"shape": [1, 4], "dtype": "float32"}}
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    out = tmp_path / "ir.json"
    res = subprocess.run([sys.executable, str(SCRIPT), str(src), str(ref_path), str(out)],
                         capture_output=True, text=True, timeout=300)
    assert out.exists(), f"trace_ir 未产出：rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    return json.loads(out.read_text(encoding="utf-8"))


def test_trace_ir_generates_valid_ir_from_real_model(tmp_path):
    """真实模型（自定义子模块 + 叶子 + 根层算子）→ 结构合法、可再生成的 IR。"""
    ir = _run(tmp_path)

    # 模块树由 `nn_module_stack` 机械取到：根 + Block + Block 的两个叶子 + Dropout
    ids = {n["id"] for n in ir["nodes"]}
    assert {"model", "block", "block_lin", "block_act", "drop"} <= ids
    assert ir["root_id"] == "model"

    # 叶子参数由实例属性取（nn.Linear 的 in/out_features），不靠猜
    lin = next(n for n in ir["nodes"] if n["id"] == "block_lin")
    assert lin["kind"] == "leaf" and lin["class_name"] == "nn.Linear"
    assert lin["params"] == {"in_features": 4, "out_features": 4}

    # 数据流边来自图依赖（lin → act 是 Block 内部的一条）
    assert {"from": "block_lin", "to": "block_act"} in ir["edges"]

    # 结构合法 + 能再生成（与拆解链路同一道闸门）
    assert validate_ir(ir) == []
    assert incomplete_ir(ir) == []
    code = ir_codegen.generate(ir)
    assert "class Decomp_block" in code and "self.block_lin" in code


def test_trace_ir_expresses_external_input_in_op(tmp_path):
    """**算子吃外部输入**（`add(..., x)`）：op 的 code_hint 写 `{ext:x}`，`input_spec.external`
    声明它，根类为此加形参——结构合法且能再生成。scGPT 的 `creterion_cce(cos_sim, labels)` 同理。
    """
    ir = _run(tmp_path, SOURCE_EXTERNAL_INPUT_OP)

    add_op = next(n for n in ir["nodes"] if n["kind"] == "op" and n["class_name"].startswith("add"))
    assert "{ext:x}" in add_op["code_hint"]
    assert {"name": "x"} in ir["input_spec"]["external"]

    assert validate_ir(ir) == []
    code = ir_codegen.generate(ir)
    assert "x" in code.split("def forward(")[1].split(")")[0]     # 根 forward 有该形参
