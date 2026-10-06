"""边张量形状的写入方（`_fill_edge_shapes`）与它不产生 stale 的口径。

背景（本次修复的缺陷）：IR schema 早就有 `edges[].tensor_shape`、`ir_to_graphir` 也在消费它
（`ir_graphir.py:247`），但**没有任何写入方**——节点形状有 forward hook 捕获与 op 兜底，
边形状一直是 null，画布/查看器拿不到「这条线上流的是什么形状」。

口径：
- 边上流过的就是源节点的输出 → 取源节点 `output_shape`，缺失时退回 `input_shape`；
  两者都未知时**保持 null（不猜）**；
- 已存在的值不覆盖（与节点形状「只补缺」一致）；
- `tensor_shape` 不进 `ir_hash`（`canonical_ir` 的边投影只有 from/to），
  因此回填**不会**让已验证的 IR 无故 stale（与《模块详细设计》6.6-4 一致）。

只用内存构造的 IR，不碰真实 data/，不跑模型。
"""
from __future__ import annotations

from app.services import decompose_service, ir_schema
from app.services.ir_graphir import ir_to_graphir


def _base_ir() -> dict:
    """最小可用 IR：根容器 net → fc（有输出形状）→ relu（无形状）。"""
    return {
        "schema_version": "1.0",
        "project_id": "test-project",
        "source_file": "model.py",
        "entry_class": "Net",
        "task_type": "classification",
        "root_id": "net",
        "input_spec": {"shape": [1, 4], "dtype": "float32"},
        "nodes": [
            {
                "id": "net", "kind": "module", "class_name": "Net", "module_path": "",
                "params": {}, "parent_id": None,
                "input_shape": [1, 4], "output_shape": [1, 2],
            },
            {
                "id": "fc", "kind": "leaf", "class_name": "nn.Linear", "module_path": "fc",
                "params": {"in_features": 4, "out_features": 2}, "parent_id": "net",
                "input_shape": [1, 4], "output_shape": [1, 2],
            },
            {
                "id": "act", "kind": "leaf", "class_name": "nn.ReLU", "module_path": "act",
                "params": {}, "parent_id": "net",
                "input_shape": None, "output_shape": None,
            },
        ],
        "edges": [
            {"from": "fc", "to": "act"},
        ],
    }


def test_fill_edge_shapes_uses_source_output_shape():
    """边形状取源节点的输出形状。"""
    ir = _base_ir()

    filled = decompose_service._fill_edge_shapes(ir)

    assert filled == 1
    assert ir["edges"][0]["tensor_shape"] == [1, 2]


def test_fill_edge_shapes_falls_back_to_input_shape():
    """源节点没有输出形状时退回它的输入形状（不臆造）。"""
    ir = _base_ir()
    ir["nodes"][1]["output_shape"] = None  # fc 只有 input_shape

    decompose_service._fill_edge_shapes(ir)

    assert ir["edges"][0]["tensor_shape"] == [1, 4]


def test_fill_edge_shapes_stays_null_when_unknown():
    """源节点形状完全未知时保持 null：不写字段、不计入回填数。"""
    ir = _base_ir()
    ir["nodes"][1]["output_shape"] = None
    ir["nodes"][1]["input_shape"] = None

    filled = decompose_service._fill_edge_shapes(ir)

    assert filled == 0
    assert "tensor_shape" not in ir["edges"][0]


def test_fill_edge_shapes_does_not_overwrite_existing():
    """已有值不覆盖（agent 明确写出的形状优先）。"""
    ir = _base_ir()
    ir["edges"][0]["tensor_shape"] = [9, 9]

    filled = decompose_service._fill_edge_shapes(ir)

    assert filled == 0
    assert ir["edges"][0]["tensor_shape"] == [9, 9]


def test_fill_edge_shapes_does_not_change_ir_hash():
    """边形状不进 ir_hash：回填前后 ir_hash 必须一致（否则「补形状」会把验证无故变 stale）。"""
    ir = _base_ir()
    before = ir_schema.ir_hash(ir)

    decompose_service._fill_edge_shapes(ir)

    assert ir["edges"][0]["tensor_shape"] == [1, 2]
    assert ir_schema.ir_hash(ir) == before


def test_ir_to_graphir_surfaces_edge_tensor_shape():
    """消费方确认：回填后 GraphIR 的边 data 里带上 tensor_shape（画布/查看器据此展示）。"""
    ir = _base_ir()
    decompose_service._fill_edge_shapes(ir)

    graph = ir_to_graphir(ir)

    assert graph["edges"][0]["data"]["tensor_shape"] == [1, 2]


# ---------------------------------------------------------------- 画布红线：形状元数据不能编造

def test_infer_missing_shapes_does_not_fabricate_output_shape():
    """op 的形状兜底**只填输入、不编造输出**。

    上游形状对 `select`/`squeeze`/`unsqueeze`/`mean`/`to` 这类**改形状**的算子并不是它的输出，
    编出来会让画布 `verifyIRShapes` 误报「输入形状不匹配」（实测 scGPT 追踪 IR 16 条红线里 14 条
    由此而来，用户据此以为「模型根本跑不了」）。形状未知时画布按「缺形状不算失败」放行。
    """
    ir = {
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None},
            {"id": "a", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "net",
             "input_shape": [1, 4], "output_shape": [1, 8]},
            {"id": "sel", "kind": "op", "class_name": "select.int", "parent_id": "net",
             "code_hint": "{inputs[0]}.select(1, 0)"},
            {"id": "out", "kind": "op", "class_name": "build_output", "parent_id": "net",
             "code_hint": "dict(y={inputs[0]})"},
        ],
        "edges": [{"from": "a", "to": "sel"}, {"from": "sel", "to": "out"}],
    }
    decompose_service._infer_missing_shapes(ir)
    by = {n["id"]: n for n in ir["nodes"]}
    assert by["sel"]["input_shape"] == [1, 8]        # 输入 = 上游输出（确凿）
    assert by["sel"].get("output_shape") is None     # 输出**不编造**（select 会改形状）
    assert by["out"].get("input_shape") is None      # 装配输出算子（dict）没有单一张量形状


def test_infer_missing_shapes_ignores_metadata_ops():
    """元数据算子（`sym_size` 产出的是 int 不是张量）不当形状来源。"""
    ir = {
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None},
            {"id": "sz", "kind": "op", "class_name": "sym_size.int", "parent_id": "net",
             "code_hint": "{inputs[0]}.size(0)", "input_shape": [1, 1200, 512]},
            {"id": "ar", "kind": "op", "class_name": "arange.default", "parent_id": "net",
             "code_hint": "torch.arange({inputs[0]})"},
        ],
        "edges": [{"from": "net", "to": "sz"}, {"from": "sz", "to": "ar"}],
    }
    decompose_service._infer_missing_shapes(ir)
    assert {n["id"]: n for n in ir["nodes"]}["ar"].get("input_shape") is None


def test_drop_ambiguous_module_input_shape():
    """多入边 module 的单一 `input_shape` 只代表**第一个实参** → 清掉。

    否则第二条入边（`MVCDecoder(cell_emb, gene_embs)` 里喂 `gene_embs` 的那条）会被误标红线。
    """
    ir = {
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None},
            {"id": "enc", "kind": "module", "class_name": "Enc", "parent_id": "net",
             "input_shape": [1, 512], "output_shape": [1, 1200, 512]},
            {"id": "one", "kind": "module", "class_name": "One", "parent_id": "net",
             "input_shape": [1, 512]},
        ],
        "edges": [{"from": "net", "to": "enc"}, {"from": "one", "to": "enc"},
                  {"from": "net", "to": "one"}],
    }
    assert decompose_service._drop_ambiguous_module_input_shape(ir) == 1
    by = {n["id"]: n for n in ir["nodes"]}
    assert by["enc"]["input_shape"] is None                      # 两入边 → 清
    assert by["enc"]["output_shape"] == [1, 1200, 512]           # 输出保留
    assert by["one"]["input_shape"] == [1, 512]                  # 单入边 → 不动


def test_infer_missing_shapes_authoritatively_clears_stale_op_shapes():
    """**先清后填**：op 的旧值（早期版本把上游输出同时写成输入与输出）必须被清掉。

    否则修复"治不了已有的 IR"——实测残留一条假红线
    `cls_decoder_out_layer(out=[1,1]) → build_output(in=[1,1200,1])`（`build_output` 旧值没被清）。
    """
    ir = {
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None},
            {"id": "a", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "net",
             "input_shape": [1, 4], "output_shape": [1, 1]},
            # 旧版编造：装配算子被塞了一个张量形状
            {"id": "out", "kind": "op", "class_name": "build_output", "parent_id": "net",
             "code_hint": "dict(y={inputs[0]})", "input_shape": [1, 1200, 1], "output_shape": [1, 1200, 1]},
            # 元数据算子下游同样被旧版编造过
            {"id": "sz", "kind": "op", "class_name": "sym_size.int", "parent_id": "net",
             "code_hint": "{inputs[0]}.size(0)"},
            {"id": "ar", "kind": "op", "class_name": "arange.default", "parent_id": "net",
             "code_hint": "torch.arange({inputs[0]})", "input_shape": [1, 1200, 512]},
        ],
        "edges": [{"from": "a", "to": "out"}, {"from": "net", "to": "sz"}, {"from": "sz", "to": "ar"}],
    }
    decompose_service._infer_missing_shapes(ir)
    by = {n["id"]: n for n in ir["nodes"]}
    assert by["out"].get("input_shape") is None       # 旧值被清掉（装配算子没有单一张量形状）
    assert by["out"].get("output_shape") is None
    assert by["ar"].get("input_shape") is None        # 元数据算子下游不编造
    assert by["ar"].get("output_shape") is None
