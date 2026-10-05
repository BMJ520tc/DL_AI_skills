"""拆解保真度自检（`scripts/ir_fidelity_probe.py`）的回归用例。

它把「IR 再生成的模型」与**真实模型**比一遍（带参层数 / 参数量 / state_dict 形状 / 模块覆盖），
供拆解循环在写盘前拦下不忠实的 IR —— 此前这类问题要到「⑤ 两步验证」才暴露，且换一次拆解复现一次。
本用例用真实 torch 跑（本机 `D:\\python.exe` 带 torch），不 mock。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.services import ir_codegen

PROBE = PROJECT_ROOT / "scripts" / "ir_fidelity_probe.py"

SOURCE = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(4, 4)\n"
    "        self.act = nn.ReLU()\n"
    "    def forward(self, x):\n"
    "        return self.act(self.fc(x))\n"
)


def _src(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    src.mkdir()
    (src / "model.py").write_text(SOURCE, encoding="utf-8")
    return src


def _ir(*, with_fc: bool = True, with_act: bool = True) -> dict:
    nodes = [{"id": "net", "kind": "module", "class_name": "Net", "parent_id": None, "module_path": ""}]
    edges = []
    if with_fc:
        nodes.append({"id": "fc", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "net",
                      "module_path": "fc", "params": {"in_features": 4, "out_features": 4}})
    if with_act:
        nodes.append({"id": "act", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net",
                      "module_path": "act"})
    if with_fc and with_act:
        edges.append({"from": "fc", "to": "act"})
    return {"source_file": "model.py", "entry_class": "Net", "task_type": "classification",
            "input_spec": {"shape": [1, 4], "dtype": "float32"}, "root_id": "net",
            "nodes": nodes, "edges": edges}


def _run_probe(tmp_path: Path, ir: dict) -> dict:
    src = _src(tmp_path)
    ir_path = tmp_path / "ir.json"
    regen_path = tmp_path / "regenerated.py"
    out_path = tmp_path / "fidelity.json"
    ir_path.write_text(json.dumps(ir), encoding="utf-8")
    regen_path.write_text(ir_codegen.generate(ir), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(PROBE), str(src), str(ir_path), str(regen_path), str(out_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert out_path.exists(), f"探针未产出结果：rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_probe_passes_on_faithful_ir(tmp_path):
    res = _run_probe(tmp_path, _ir())
    assert res["ok"] is True, res.get("issues")
    assert res["original_params"] == res["regenerated_params"]
    assert res["uncovered"] == []


def test_probe_flags_missing_layer(tmp_path):
    """少拆一层（如 scGPT 漏 mvc_decoder）→ 参数量对不上 + 模块未覆盖，两处都点名。"""
    res = _run_probe(tmp_path, _ir(with_fc=False))
    assert res["ok"] is False
    joined = "；".join(res["issues"])
    assert "带参层数不一致" in joined
    assert "未被 IR 覆盖" in joined and "fc" in joined


def test_probe_tolerates_value_differences(tmp_path):
    """**只查结构，不查参数数值**：`num_embeddings` 这类值由运行期决定（实例按 entry_args 构造），
    而「③ 补形状」本就会从真实实例回填——拆解阶段拿「参数精确相等」当门槛会永远判不过。
    这里改一个**不影响前向形状**的值（ReLU 的 inplace），结构一致即应放行。"""
    ir = _ir()
    for n in ir["nodes"]:
        if n["id"] == "act":
            n["params"] = {"inplace": True}      # 真实实例是 inplace=False，但不改变结构/形状
    res = _run_probe(tmp_path, ir)
    assert res["ok"] is True, res.get("issues")


def test_probe_skips_when_real_model_cannot_instantiate(tmp_path):
    """真实模型起不来（缺依赖/参数）→ skipped，宿主据此**跳过**而不是误判不忠实。"""
    ir = _ir()
    ir["entry_class"] = "DoesNotExist"     # 真实模型侧加载失败 → 无从比对，应跳过
    res = _run_probe(tmp_path, ir)
    assert res.get("skipped") is True and res["ok"] is True
