"""模块结构签名与「模型实际参数」回填（R27）。

已知缺陷：`module_id` 由结构签名算出，而签名此前只取 agent 写进 IR 的 `params`；
agent 是否写出**可选参数**（ReLU 的 `inplace`、BatchNorm 的 `affine` 等）并不稳定 →
同一模型两次拆解可能得到不同 module_id（实测 mod_611e10504a9b0e61 与 mod_fa06d7bcc04e0319 并存）。

修法：trace 把「模型实际暴露的参数」记入节点 `params_model`（不参与 ir_hash），
签名取 `params` ∪ `params_model`（模型值优先）。本用例覆盖：可选参数漏写不再改变签名、
冲突时模型值优先、未跑 trace 时回落 agent 参数、以及「回填不产生 stale」的口径。
"""
from __future__ import annotations

from app.services.decompose_service import _merge_shapes, _module_signature
from app.services.ir_schema import ir_hash


def _net_ir(relu_params: dict, relu_model_params: dict | None = None) -> dict:
    """最小可签名 IR：Net(root, module) → fc(Linear) → act(ReLU)。"""
    act: dict = {
        "id": "act",
        "kind": "leaf",
        "class_name": "nn.ReLU",
        "module_path": "act",
        "params": dict(relu_params),
        "parent_id": "net",
        "input_shape": [1, 8],
        "output_shape": [1, 8],
    }
    if relu_model_params is not None:
        act["params_model"] = dict(relu_model_params)
    return {
        "schema_version": "1.0",
        "project_id": "p1",
        "source_file": "model.py",
        "entry_class": "Net",
        "task_type": "classification",
        "input_spec": {"shape": [1, 8], "dtype": "float32"},
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "params": {}, "parent_id": None},
            {"id": "fc", "kind": "leaf", "class_name": "nn.Linear", "module_path": "fc",
             "params": {"in_features": 8, "out_features": 8}, "parent_id": "net"},
            act,
        ],
        "edges": [{"from": "fc", "to": "act"}],
    }


def _conv_ir(agent_kernel_size, model_kernel_size) -> dict:
    """单 leaf Conv2d 模型：用于验证「冲突时模型值优先」与「签名对结构差异仍敏感」。"""
    return {
        "schema_version": "1.0",
        "source_file": "model.py",
        "entry_class": "Net",
        "task_type": "classification",
        "input_spec": {"shape": [1, 3, 32, 32], "dtype": "float32"},
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "params": {}, "parent_id": None},
            {"id": "conv", "kind": "leaf", "class_name": "nn.Conv2d", "module_path": "conv",
             "params": {"in_channels": 3, "out_channels": 8, "kernel_size": agent_kernel_size},
             "params_model": {"kernel_size": model_kernel_size}, "parent_id": "net"},
        ],
        "edges": [],
    }


def test_optional_param_omission_does_not_change_module_id():
    """agent 漏写 inplace（模型实际为 True）与写出 inplace=True 必须得到同一 module_id。"""
    written = _module_signature(_net_ir({"inplace": True}, {"inplace": True}))
    omitted = _module_signature(_net_ir({}, {"inplace": True}))
    assert written == omitted


def test_without_trace_signature_falls_back_to_agent_params():
    """未跑过 trace（无 params_model）时仍回落 agent 参数——如实记录「签名仍可能不稳定」。"""
    written = _module_signature(_net_ir({"inplace": True}))
    omitted = _module_signature(_net_ir({}))
    assert written != omitted


def test_model_params_win_on_conflict():
    """agent 的等价写法（kernel_size 3 vs (3,3)）被模型值归一，且不掩盖真实结构差异。"""
    as_int = _module_signature(_conv_ir(3, [3, 3]))
    as_list = _module_signature(_conv_ir([3, 3], [3, 3]))
    assert as_int == as_list
    assert _module_signature(_conv_ir([3, 3], [5, 5])) != as_list


def test_merge_shapes_shapes_only_keeps_hash():
    """形状回填不改变 ir_hash（6.6-4「先验证后补形状」不得把验证无故变 stale）。"""
    ir = _net_ir({"inplace": True})
    for n in ir["nodes"]:
        n["input_shape"] = None
        n["output_shape"] = None
    before = ir_hash(ir)
    trace = {
        "shapes": {
            "fc": {"input_shape": [1, 8], "output_shape": [1, 8]},
            "act": {"input_shape": [1, 8], "output_shape": [1, 8]},
        },
        "order": [{"path": "fc", "class_name": "Linear"}, {"path": "act", "class_name": "ReLU"}],
    }
    filled = _merge_shapes(ir, trace)
    assert filled == 4
    assert ir_hash(ir) == before


def test_merge_shapes_records_model_params_without_changing_hash():
    """params_model 只在签名使用，不参与 ir_hash；agent 已写全的 params 不被无谓改写。"""
    ir = _net_ir({"inplace": True})
    fc = next(n for n in ir["nodes"] if n["id"] == "fc")
    fc["params"]["bias"] = True  # 与模型实际值一致 → 回填不产生改动
    before = ir_hash(ir)
    trace = {
        "shapes": {
            "fc": {"input_shape": [1, 8], "output_shape": [1, 8]},
            "act": {"input_shape": [1, 8], "output_shape": [1, 8]},
        },
        "order": [{"path": "fc", "class_name": "Linear"}, {"path": "act", "class_name": "ReLU"}],
        "params": {
            "fc": {"in_features": 8, "out_features": 8, "bias": True},
            "act": {"inplace": True},
        },
        "params_delta": {},
        "input_shape": [1, 8],
    }
    filled = _merge_shapes(ir, trace)
    assert ir_hash(ir) == before
    assert filled == 2  # 仅补 fc 的输入/输出形状；params 与模型值一致，不产生改动
    assert next(n for n in ir["nodes"] if n["id"] == "act")["params_model"] == {"inplace": True}
    assert fc["params_model"] == {"in_features": 8, "out_features": 8, "bias": True}


def test_merge_shapes_only_backfills_non_default_delta():
    """默认值等价的键不回填进 params（避免验证无故 stale），但完整值仍进 params_model。"""
    ir = _net_ir({})
    before = ir_hash(ir)
    trace = {
        "shapes": {"act": {"input_shape": [1, 8], "output_shape": [1, 8]},
                   "fc": {"input_shape": [1, 8], "output_shape": [1, 8]}},
        "order": [{"path": "fc", "class_name": "Linear"}, {"path": "act", "class_name": "ReLU"}],
        "params": {"fc": {"in_features": 8, "out_features": 8, "bias": True},
                   "act": {"inplace": False}},
        "params_delta": {"fc": {"in_features": 8, "out_features": 8}},  # bias/inplace 都是默认值
        "input_shape": [1, 8],
    }
    _merge_shapes(ir, trace)
    assert ir_hash(ir) == before
    act = next(n for n in ir["nodes"] if n["id"] == "act")
    assert act["params"] == {}          # inplace=False 是默认值 → 不回填
    assert act["params_model"] == {"inplace": False}
    fc = next(n for n in ir["nodes"] if n["id"] == "fc")
    assert fc["params"] == {"in_features": 8, "out_features": 8}  # 未写入 bias
    assert fc["params_model"] == {"in_features": 8, "out_features": 8, "bias": True}


def test_merge_shapes_backfills_optional_param_into_params():
    """agent 完全没写可选参数、且该值与默认值不等价（padding=1）→ 回填进 params。"""
    ir = _net_ir({})
    trace = {
        "shapes": {"act": {"input_shape": [1, 8], "output_shape": [1, 8]}},
        "order": [{"path": "act", "class_name": "ReLU"}],
        "params": {"act": {"inplace": True, "padding": [1, 1]}},
        "params_delta": {"act": {"inplace": True, "padding": [1, 1]}},
        "input_shape": [1, 8],
    }
    _merge_shapes(ir, trace)
    act = next(n for n in ir["nodes"] if n["id"] == "act")
    assert act["params"] == {"inplace": True, "padding": [1, 1]}
    assert act["params_model"] == {"inplace": True, "padding": [1, 1]}


def test_merge_shapes_keeps_equivalent_literal():
    """agent 写 3、模型实际 (3,3)：等价 → 不改写 params（旧口径正是为此保留原样）。"""
    ir = _net_ir({})
    conv = {"id": "conv", "kind": "leaf", "class_name": "nn.Conv2d", "module_path": "conv",
            "params": {"in_channels": 3, "out_channels": 8, "kernel_size": 3}, "parent_id": "net"}
    ir["nodes"].append(conv)
    trace = {
        "shapes": {},
        "order": [],
        "params": {"conv": {"in_channels": 3, "out_channels": 8, "kernel_size": [3, 3]}},
        "params_delta": {"conv": {"in_channels": 3, "out_channels": 8, "kernel_size": [3, 3]}},
        "input_shape": [1, 3, 32, 32],
    }
    _merge_shapes(ir, trace)
    assert conv["params"]["kernel_size"] == 3
    assert conv["params_model"]["kernel_size"] == [3, 3]


def test_merge_shapes_replaces_non_literal_expression():
    """agent 写 `'sizes[0]'` 这类表达式（不可用）→ 用模型实际值替换。"""
    ir = _net_ir({})
    fc = next(n for n in ir["nodes"] if n["id"] == "fc")
    fc["params"]["in_features"] = "sizes[0]"
    trace = {
        "shapes": {},
        "order": [],
        "params": {"fc": {"in_features": 8, "out_features": 8}},
        "params_delta": {"fc": {"in_features": 8}},
        "input_shape": [1, 8],
    }
    _merge_shapes(ir, trace)
    assert fc["params"]["in_features"] == 8


def test_op_node_optional_param_does_not_change_module_id():
    """op 节点模型不暴露参数：写全 `end_dim=-1` 与省略它必须得到同一 module_id。"""
    def _ir_with_flatten(end_dim_written: bool) -> dict:
        params = {"start_dim": 1}
        if end_dim_written:
            params["end_dim"] = -1
        return {
            "schema_version": "1.0",
            "source_file": "model.py",
            "entry_class": "Net",
            "task_type": "classification",
            "input_spec": {"shape": [1, 8], "dtype": "float32"},
            "root_id": "net",
            "nodes": [
                {"id": "net", "kind": "module", "class_name": "Net", "params": {}, "parent_id": None},
                {"id": "fc", "kind": "leaf", "class_name": "nn.Linear", "module_path": "fc",
                 "params": {"in_features": 8, "out_features": 8}, "parent_id": "net"},
                {"id": "flatten", "kind": "op", "class_name": "flatten", "params": params,
                 "parent_id": "net"},
            ],
            "edges": [{"from": "fc", "to": "flatten"}],
        }

    assert _module_signature(_ir_with_flatten(True)) == _module_signature(_ir_with_flatten(False))


def test_op_node_real_param_difference_still_detected():
    """补齐默认值不能把真实结构差异抹平：flatten 的 start_dim 不同仍应得到不同签名。"""
    def _ir(start_dim: int) -> dict:
        return {
            "schema_version": "1.0",
            "source_file": "model.py",
            "entry_class": "Net",
            "task_type": "classification",
            "input_spec": {"shape": [1, 8], "dtype": "float32"},
            "root_id": "net",
            "nodes": [
                {"id": "net", "kind": "module", "class_name": "Net", "params": {}, "parent_id": None},
                {"id": "fc", "kind": "leaf", "class_name": "nn.Linear", "module_path": "fc",
                 "params": {"in_features": 8, "out_features": 8}, "parent_id": "net"},
                {"id": "flatten", "kind": "op", "class_name": "flatten",
                 "params": {"start_dim": start_dim}, "parent_id": "net"},
            ],
            "edges": [{"from": "fc", "to": "flatten"}],
        }

    assert _module_signature(_ir(1)) != _module_signature(_ir(2))


def test_merge_shapes_fixes_bogus_expression_on_default_valued_key():
    """默认值键（不在 delta）也被 agent 写成表达式时，同样以模型值纠正（不放过错代码）。"""
    ir = _net_ir({"inplace": "cfg"})
    trace = {
        "shapes": {},
        "order": [],
        "params": {"act": {"inplace": True}},   # 与默认值等价 → delta 为空
        "params_delta": {},
        "input_shape": [1, 8],
    }
    _merge_shapes(ir, trace)
    act = next(n for n in ir["nodes"] if n["id"] == "act")
    assert act["params"]["inplace"] is True
