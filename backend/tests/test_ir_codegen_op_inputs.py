"""IR 再生成：op 占位符与生成代码可编译性（2026-10-05 scGPT 两步验证暴露）。

- `{inputs}` 在多入边时展开成「a, b, c」逗号串（适合 `torch.add({inputs})`）；
- 单独取第 N 个操作数要写 `{inputs[N]}`；agent 误写 `{inputs}[0]` 会拼出
  `dict(a=x, b, c[0])` 这种「位置参数跟在关键字参数之后」的语法错误；
- 生成代码**必须能编译**：不能编译就在 `generate` 处报错（拆解可带原因重试），
  而不是一路走到「⑤ 两步验证」才以脚本 SyntaxError 爆出。
"""
from __future__ import annotations

import pytest

from app.services import ir_codegen
from app.services.ir_codegen import IrIncompleteError


def _ir(op_hint: str, n_inputs: int = 2) -> dict:
    nodes = [
        {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None, "module_path": ""},
        {"id": "a", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "net",
         "params": {"in_features": 4, "out_features": 4}},
        {"id": "b", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net"},
        {"id": "op1", "kind": "op", "class_name": "f", "parent_id": "net", "code_hint": op_hint},
    ]
    edges = [{"from": "a", "to": "op1"}]
    if n_inputs >= 2:
        edges.append({"from": "b", "to": "op1"})
    return {"source_file": "m.py", "entry_class": "Net", "task_type": "classification",
            "input_spec": {"shape": [1, 4], "dtype": "float32"}, "root_id": "net",
            "nodes": nodes, "edges": edges}


def test_indexed_placeholder_picks_nth_input():
    """`{inputs[0]}` / `{inputs[1]}` → 第 0/1 个操作数（不是逗号串）。"""
    code = ir_codegen.generate(_ir("torch.max({inputs[0]}, {inputs[1]})"))
    assert "torch.max(var_a, var_b)" in code


def test_indexed_placeholder_out_of_range_errors():
    with pytest.raises(IrIncompleteError, match=r"inputs\[3\]"):
        ir_codegen.generate(_ir("torch.add({inputs[3]})"))


def test_bare_inputs_still_expands_to_comma_list():
    code = ir_codegen.generate(_ir("torch.add({inputs})"))
    assert "torch.add(var_a, var_b)" in code


def test_generate_rejects_uncompilable_code():
    """`{inputs}[0]` 会拼出位置参数跟在关键字参数之后 → generate 必须拦下并报出错行。"""
    with pytest.raises(IrIncompleteError, match="语法错误"):
        ir_codegen.generate(_ir("dict(pred={inputs}[0], aux={inputs}[1])"))


def test_check_syntax_reports_line():
    with pytest.raises(IrIncompleteError, match="第 2 行"):
        ir_codegen._check_syntax("x = 1\ndict(a=1, b)\n")
