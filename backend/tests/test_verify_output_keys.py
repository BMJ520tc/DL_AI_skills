"""⑤ 两步验证的「输出分支逐键比对」回归（2026-10-06 用户实测的假通过）。

背景：`verify_decompose.py` 原先对 dict 输出只取 `first_tensor(...)` 比对——多输出头模型（如 scGPT
返回 `{mlm_output, cell_emb, cls_output, mvc_output, loss_ecs, …}`）**丢掉整条分支也判 passed**：
实测再生成模型缺 `mvc_output`/`loss_ecs`、`mvc_decoder` 变成只实例化不调用的死模块，而 ⑤ 仍「通过」。
现要求**逐键比对**：一侧非 Mapping、键集不一致、或任一键形状/数值不符 → `overall=failed`。

用真实 torch 跑 `verify_decompose.py`（脚本本身是被宿主按文件执行的，需按文件路径验证）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
VERIFY = SCRIPTS / "verify_decompose.py"

_SOURCE = """
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return {"a": self.fc(x), "b": self.fc(x) + 1.0}
"""

_REGENERATED_OK = """
import torch
import torch.nn as nn


class Decomp_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return {"a": self.fc(x), "b": self.fc(x) + 1.0}
"""

# 少了 "b" 这条分支（等价于实测里丢掉的 mvc_output）
_REGENERATED_MISSING_KEY = """
import torch
import torch.nn as nn


class Decomp_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return {"a": self.fc(x)}
"""

# 键都在，但 "b" 的值不对（数值仍要能拦下）
_REGENERATED_WRONG_VALUE = """
import torch
import torch.nn as nn


class Decomp_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return {"a": self.fc(x), "b": self.fc(x) + 5.0}
"""

_IR = {
    "schema_version": "1.0", "source_file": "model.py", "entry_class": "Net",
    "task_type": "classification", "root_id": "net",
    "input_spec": {"shape": [1, 4], "dtype": "float32"},
    "nodes": [{"id": "net", "kind": "module", "class_name": "Net", "parent_id": None, "module_path": ""}],
    "edges": [],
}


def _run(tmp: Path, regenerated: str) -> dict:
    src = tmp / "source"
    src.mkdir(parents=True, exist_ok=True)
    (src / "model.py").write_text(_SOURCE, encoding="utf-8")
    (tmp / "ir.json").write_text(json.dumps(_IR), encoding="utf-8")
    (tmp / "regenerated.py").write_text(regenerated, encoding="utf-8")
    out = tmp / "out.json"
    res = subprocess.run(
        [sys.executable, str(VERIFY), str(src), str(tmp / "ir.json"),
         str(tmp / "regenerated.py"), str(out), "42", "1e-5", "1e-6"],
        capture_output=True, text=True, timeout=300,
    )
    assert out.exists(), res.stdout + res.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_verify_passes_when_all_output_keys_match(tmp_path):
    r = _run(tmp_path / "ok", _REGENERATED_OK)
    assert r["overall"] == "passed", r.get("failure_reason")
    assert r["structure"]["output_keys_match"] is True


def test_verify_fails_when_a_branch_is_missing(tmp_path):
    """生成模型少一个输出键（丢整条分支）→ 判不通过（原先只比首个张量会漏过）。"""
    r = _run(tmp_path / "missing", _REGENERATED_MISSING_KEY)
    assert r["overall"] == "failed"
    assert r["structure"]["output_key_diff"]["missing"] == ["b"]
    assert "输出分支不一致" in (r.get("failure_reason") or "")


def test_verify_fails_on_per_key_value_mismatch(tmp_path):
    """键齐全但某个分支数值不符 → 仍判不通过（逐键数值比对生效）。"""
    r = _run(tmp_path / "wrong", _REGENERATED_WRONG_VALUE)
    assert r["overall"] == "failed"
    per_key = r["numeric"]["per_seed"][0]["per_key"]
    assert per_key["a"]["passed"] is True and per_key["b"]["passed"] is False
