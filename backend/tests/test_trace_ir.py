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


def _run(tmp_path: Path, source: str = SOURCE, spec_extra: dict | None = None) -> dict:
    src = tmp_path / "source"
    src.mkdir()
    (src / "model.py").write_text(source, encoding="utf-8")
    ref = {"entry_class": "Net", "source_file": "model.py", "task_type": "classification",
           "input_spec": {"shape": [1, 4], "dtype": "float32", **(spec_extra or {})}}
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


# ---------------------------------------------------------------- 端到端（多输入 + loss 分支）

# 覆盖 2026-10-06 追踪链路暴露的一批缺陷：外部输入按模块接线、多输入子模块的内部路由、
# 无偏置叶子（`bias=False`）、多输出 + loss 分支（`arange().long()` 的 `to.dtype_layout` 透传、
# `cross_entropy` 的常量操作数）、唯一子模块是无参 `nn.*` 的模块（`Sim`——子节点只有算子，
# 不能被「无子模块节点」清理误删）。
SOURCE_PIPELINE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "\n"
    "class Head(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(4, 4, bias=False)\n"
    "    def forward(self, x):\n"
    "        return self.fc(x)\n"
    "\n"
    "class Enc(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(4, 4)\n"
    "    def forward(self, y):\n"
    "        return self.fc(y)\n"
    "\n"
    "class Writer(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.lin_a = nn.Linear(4, 4)\n"
    "        self.lin_b = nn.Linear(4, 4)\n"
    "    def forward(self, a, b):\n"
    "        q = self.lin_b(b)\n"          # 先吃第二个形参 → 图上先出现 b（签名序 ≠ 图序）
    "        r = self.lin_a(a)\n"
    "        return q + r\n"
    "\n"
    "class Sim(nn.Module):\n"              # 唯一子模块是无参 nn.*（折进父层不建节点）→ Sim 必须保留
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.cos = nn.CosineSimilarity(dim=-1)\n"
    "    def forward(self, a, b):\n"
    "        return self.cos(a, b)\n"
    "\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.head = Head()\n"
    "        self.enc = Enc()\n"
    "        self.writer = Writer()\n"
    "        self.sim = Sim()\n"
    "    def forward(self, x, y):\n"
    "        h = self.head(x)\n"
    "        e = self.enc(y)\n"
    "        w = self.writer(h, e)\n"
    "        logits = h + w\n"
    "        sim = self.sim(h, e) / 0.5\n"
    "        labels = torch.arange(logits.size(0)).long()\n"
    "        loss = F.cross_entropy(logits, labels)\n"
    "        return {\"logits\": logits, \"sim\": sim, \"loss\": loss}\n"
)


def test_trace_ir_keeps_batch_dim_dynamic(tmp_path):
    """**批量维必须保持动态**：`arange(cos.size(0))` 这类**数据依赖**的常量，静态导出会被示例批量
    **常量折叠**成 `arange(1)` → 生成的模型换 batch 直接崩（实测 scGPT 导出模型 batch>1 报
    `cross_entropy` 尺寸不符，且 batch=1 时 `loss_cce`/`loss_ecs` 恒为常数）。锁：IR 里是 `.size(0)`
    而不是常数，且再生成模型在 batch=3 能跑出正确形状。
    """
    ir = _run(tmp_path, SOURCE_DYNAMIC_BATCH)
    assert validate_ir(ir) == []
    hints = " ".join(str(n.get("code_hint") or "") for n in ir["nodes"])
    assert ".size(0)" in hints, hints
    assert "torch.arange(1)" not in hints

    regen = tmp_path / "regenerated.py"
    regen.write_text(ir_codegen.generate(ir), encoding="utf-8")
    out = _forward_at_batch(regen, ir["root_id"], batch=3)
    assert list(out["logits"].shape) == [3, 3]


SOURCE_DYNAMIC_BATCH = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(4, 4)\n"
    "    def forward(self, x):\n"
    "        n = F.normalize(self.fc(x), p=2, dim=1)\n"
    "        cos = torch.mm(n, n.t())\n"
    "        labels = torch.arange(cos.size(0)).long()\n"      # 数据依赖 → 静态导出会被折叠成 arange(1)
    "        loss = F.cross_entropy(cos, labels)\n"
    "        return {\"logits\": cos, \"loss\": loss}\n"
)


def _forward_at_batch(regen_path: Path, root_id: str, batch: int):
    """加载再生成代码，在给定 batch 上跑一次前向（测试环境自带 torch）。"""
    import importlib.util

    import torch

    spec = importlib.util.spec_from_file_location("regen_dyn_batch", regen_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = getattr(mod, f"Decomp_{root_id}")().eval()
    with torch.no_grad():
        return model(torch.randn(batch, 4))


def test_trace_ir_multi_input_loss_model_passes_two_step_verification(tmp_path):
    """端到端：追踪产出的 IR 能再生成、并通过 ⑤ 两步验证（结构 + **逐键**数值比对）。"""
    ir = _run(tmp_path, SOURCE_PIPELINE,
              spec_extra={"extra": [{"shape": [1, 4], "dtype": "float32"}]})

    # 外部输入按模块接线：x→head、y→enc（顺序取 forward 形参序）
    assert ir["input_spec"]["inputs"] == ["head", "enc"]
    # 无偏置叶子记进 params（漏了会凭空多一个 bias 参数、数值也对不上）
    head_fc = next(n for n in ir["nodes"] if n["id"] == "head_fc")
    assert head_fc["params"].get("bias") is False
    # 唯一子模块是无参 nn.* 的 `Sim` 不能被误删（子节点是算子，第 4 段才建）
    assert "sim" in {n["id"] for n in ir["nodes"]}
    # 所有算子都该活下来（碎片清理为 0）
    assert validate_ir(ir) == []
    assert incomplete_ir(ir) == []

    regen = tmp_path / "regenerated.py"
    regen.write_text(ir_codegen.generate(ir), encoding="utf-8")

    out = tmp_path / "verify.json"
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_decompose.py"),
         str(tmp_path / "source"), str(tmp_path / "ir.json"), str(regen), str(out),
         "0,1", "1e-5", "1e-6"],
        capture_output=True, text=True, timeout=300)
    assert out.exists(), f"verify 未产出：rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["overall"] == "passed", result.get("failure_reason")
    assert result["structure"]["output_keys_match"] is True
