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


# ---------------------------------------------------------------- 多输入（边界 ⑯）

def _two_input_ir(**spec_extra) -> dict:
    """root 有两个「模块之外」的源节点 a/b——单输入时二者都拿到 x（旧行为）。"""
    return {
        "source_file": "m.py", "entry_class": "Net", "task_type": "classification",
        "input_spec": {"shape": [1, 4], "dtype": "float32", **spec_extra},
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None, "module_path": ""},
            {"id": "a", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "net",
             "params": {"in_features": 4, "out_features": 4}},
            {"id": "b", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net"},
            {"id": "c", "kind": "op", "class_name": "add", "parent_id": "net"},
        ],
        "edges": [{"from": "a", "to": "c"}, {"from": "b", "to": "c"}],
    }


def test_multi_input_declaration_generates_multiple_forward_params():
    """`input_spec.inputs` 按序给 root 的 forward 声明多个形参，源节点各取自己的那个。"""
    code = ir_codegen.generate(_two_input_ir(inputs=["a", "b"]))
    assert "def forward(self, x0, x1):" in code
    assert "var_a = self.a(x0)" in code
    assert "var_b = self.b(x1)" in code


def test_multi_input_undeclared_source_falls_back_to_first():
    """只声明一个时，未声明的源节点退回第一个形参（向后兼容，不炸）。"""
    code = ir_codegen.generate(_two_input_ir(inputs=["a"]))
    assert "def forward(self, x0):" in code
    assert "var_a = self.a(x0)" in code and "var_b = self.b(x0)" in code


def test_single_input_signature_unchanged_without_declaration():
    code = ir_codegen.generate(_two_input_ir())
    assert "def forward(self, x):" in code
    assert "var_a = self.a(x)" in code and "var_b = self.b(x)" in code


def test_leaf_with_two_in_edges_is_still_rejected():
    """**叶子**仍是单张量输入（多操作数叶子表达不了）；要多输入用 **module 节点**（见下）。"""
    ir = _two_input_ir()
    ir["nodes"].append({"id": "cos", "kind": "leaf", "class_name": "nn.CosineSimilarity",
                        "parent_id": "net", "code_hint": "nn.CosineSimilarity(dim=-1)"})
    ir["edges"] = [{"from": "a", "to": "cos"}, {"from": "b", "to": "cos"}]
    with pytest.raises(IrIncompleteError, match="多条入边"):
        ir_codegen.generate(ir)


def test_multi_input_module_gets_params_matching_its_in_edges():
    """**多输入模块**：入边即 forward 形参（`MVCDecoder(cell_emb, gene_embs)` 这类），
    父模块按序传参。此前 schema 只许一条入边 → 这类模块要么表达不了、要么被迫不连边丢分支。"""
    ir = {
        "source_file": "m.py", "entry_class": "Net", "task_type": "classification",
        "input_spec": {"shape": [1, 4], "dtype": "float32"}, "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None, "module_path": ""},
            {"id": "a", "kind": "leaf", "class_name": "nn.Identity", "parent_id": "net"},
            {"id": "b", "kind": "leaf", "class_name": "nn.Identity", "parent_id": "net"},
            {"id": "mv", "kind": "module", "class_name": "MVCDecoder", "parent_id": "net",
             "module_path": "mv"},
            {"id": "mv1", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "mv",
             "params": {"in_features": 4, "out_features": 4}},
            {"id": "mv2", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "mv",
             "params": {"in_features": 4, "out_features": 4}},
        ],
        "edges": [{"from": "a", "to": "mv"}, {"from": "b", "to": "mv"}, {"from": "mv1", "to": "mv2"}],
    }
    code = ir_codegen.generate(ir)
    assert "var_mv = self.mv(var_a, var_b)" in code          # 父模块按序把两个输入传下去
    assert "class Decomp_mv(nn.Module):" in code
    assert "def forward(self, x0, x1):" in code              # 子模块两个形参
    assert "var_mv1 = self.mv1(x0)" in code                  # 内部无入边的子节点用第一个形参




def test_module_params_emitted_as_attributes():
    """模块节点的标量 params → `self.<k> = 值`（agent 的 code_hint 常引用它们，如 self.max_value）。"""
    ir = _ir("torch.clamp({inputs}, max=self.max_value)")   # 两个 op 入边
    for n in ir["nodes"]:
        if n["id"] == "net":
            n["params"] = {"max_value": 512, "ratio": 0.5, "nested": {"a": 1}}   # 嵌套非字面量应跳过
    code = ir_codegen.generate(ir)
    assert "self.max_value = 512" in code
    assert "self.ratio = 0.5" in code
    assert "self.nested" not in code


def test_code_hint_self_ref_must_refer_to_child_or_param():
    """code_hint 引用未定义的 `self.<name>` → 再生成时拦下并给出可操作提示（不是运行期 AttributeError）。"""
    ir = _ir("torch.clamp({inputs}, max=self.nope)")
    with pytest.raises(IrIncompleteError, match=r"self\.nope"):
        ir_codegen.generate(ir)




def test_uncovered_modules_flags_ir_gaps():
    """trace 实际执行到、IR 却没拆的子树要如实点名（如 scGPT 的 mvc_decoder）。"""
    from app.services import decompose_service

    ir = {"nodes": [{"id": "net", "module_path": ""}, {"id": "a", "module_path": "encoder"}]}
    trace = {"shapes": {"encoder": {}, "mvc_decoder.W": {}, "mvc_decoder.gene2query": {}}}
    assert decompose_service._uncovered_modules(ir, trace) == ["mvc_decoder.W", "mvc_decoder.gene2query"]
    # 全覆盖 → 空
    assert decompose_service._uncovered_modules(ir, {"shapes": {"encoder": {}}}) == []
    # **子树也算覆盖**：code_hint 折叠的模块（如 nn.TransformerEncoder）内部子模块不应被算漏拆
    ir2 = {"nodes": [{"id": "net", "module_path": ""},
                     {"id": "te", "module_path": "transformer_encoder"}]}
    trace2 = {"shapes": {"transformer_encoder": {},
                         "transformer_encoder.layers.0.self_attn": {},
                         "transformer_encoder.layers.0.linear1": {}}}
    assert decompose_service._uncovered_modules(ir2, trace2) == []
