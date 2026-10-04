"""拆解 ir 图的画布导出/训练：GraphIR(ir) → IR 反向映射与同源再生成（6.3/7.5）。

缺陷背景：模块四入库生成的结构化项目，画布节点是 `type="ir"`（ir_graphir.ir_to_graphir）。
旧实现里 network_export 对这类图直接拒绝（`parentId` 非空；节点类型不在 `NODE_TABLE`），
于是「画布可编辑、可保存、可出版本」但「导出代码与发起训练必 400」。本文件锁定修复后的契约：

- 反向映射 `ir_graphir.graphir_to_ir`：`parentId` → `parent_id`、`data.params` → `params`、
  边 → `edges`，还原后必须通过 `ir_schema.validate_ir`；
- **画布上的调参必须体现在导出代码里**（本修复的核心价值）；
- 导出与训练同源：训练落盘的 `model.py` 与导出端点返回的代码逐字节相同，且带
  `GeneratedModel` 入口（`templates/train.py` 的 import 契约）；
- 不合法的 ir 图 → 400 + 定位到节点/字段，不 500。

工作区与知识库都是临时目录，训练脚本执行被 monkeypatch 掉，不碰真实 data/（AGENTS.md 规矩 5）。
"""
from __future__ import annotations

import asyncio
import json
import py_compile
from pathlib import Path

import pytest

from app.services import ir_codegen, ir_graphir, ir_schema, network_export


# ---------------------------------------------------------------------------
# 夹具图：最小 IR（1 container + 2 leaf + 1 op）与嵌套/跳连 IR
# ---------------------------------------------------------------------------

def _minimal_ir() -> dict:
    """最小合法 IR：container(Linear→ReLU) + 直连 Linear + 汇合算子 add。

    结构：net(module) ├ block(container){fc1(Linear 4→8), act(ReLU)}
                      ├ head(Linear 8→8)
                      └ fuse(op add, 入边 block/head)
    """
    return {
        "schema_version": "1.0",
        "source_file": "models/sample_net.py",
        "entry_class": "Net",
        "task_type": "classification",
        "input_spec": {"shape": [1, 4], "dtype": "float32"},
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None,
             "input_shape": [1, 4], "output_shape": [1, 8]},
            {"id": "block", "kind": "container", "class_name": "nn.Sequential", "parent_id": "net",
             "module_path": "block"},
            {"id": "fc1", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "block",
             "module_path": "block.0", "params": {"in_features": 4, "out_features": 8},
             "input_shape": [1, 4], "output_shape": [1, 8]},
            {"id": "act", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "block",
             "module_path": "block.1", "params": {}},
            {"id": "head", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "net",
             "module_path": "head", "params": {"in_features": 8, "out_features": 8}},
            {"id": "fuse", "kind": "op", "class_name": "add", "parent_id": "net", "params": {}},
        ],
        "edges": [
            {"from": "block", "to": "head", "tensor_shape": [1, 8]},
            {"from": "block", "to": "fuse", "tensor_shape": [1, 8]},
            {"from": "head", "to": "fuse", "tensor_shape": [1, 8]},
        ],
    }


def _nested_ir() -> dict:
    """三层嵌套（net → sub → sub2 → lin2）+ 同父跳跃边（a → c 非相邻兄弟）。

    结构：net(module) ├ sub(module) └ sub2(module) └ lin2(Leaf)
                      ├ a(ReLU) ├ b(ReLU) ├ c(ReLU) └ d(op add)
    边：sub→a、a→b、a→c（跳跃）、b→d、c→d。
    """
    return {
        "schema_version": "1.0",
        "source_file": "models/nested_net.py",
        "entry_class": "Net",
        "task_type": "classification",
        "input_spec": {"shape": [1, 4]},
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None,
             "input_shape": [1, 4], "output_shape": [1, 4]},
            {"id": "sub", "kind": "module", "class_name": "Sub", "parent_id": "net",
             "module_path": "sub"},
            {"id": "sub2", "kind": "module", "class_name": "Sub2", "parent_id": "sub",
             "module_path": "sub.sub2"},
            {"id": "lin2", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "sub2",
             "module_path": "sub.sub2.lin2", "params": {"in_features": 4, "out_features": 4}},
            {"id": "a", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net",
             "module_path": "a", "params": {}},
            {"id": "b", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net",
             "module_path": "b", "params": {}},
            {"id": "c", "kind": "leaf", "class_name": "nn.ReLU", "parent_id": "net",
             "module_path": "c", "params": {}},
            {"id": "d", "kind": "op", "class_name": "add", "parent_id": "net", "params": {}},
        ],
        "edges": [
            {"from": "sub", "to": "a", "tensor_shape": [1, 4]},
            {"from": "a", "to": "b", "tensor_shape": [1, 4]},
            {"from": "a", "to": "c", "tensor_shape": [1, 4]},
            {"from": "b", "to": "d", "tensor_shape": [1, 4]},
            {"from": "c", "to": "d", "tensor_shape": [1, 4]},
        ],
    }


def _canvas_of(ir: dict) -> dict:
    """IR → 画布 GraphIR（后端与拆解入库同一条正向映射）。"""
    graph = ir_graphir.ir_to_graphir(ir)
    assert ir_schema.validate_ir(ir) == [], "夹具 IR 本身必须合法"
    return graph


def _node_by_id(graph: dict, node_id: str) -> dict:
    return next(n for n in graph["nodes"] if n["id"] == node_id)


def _edit_params(graph: dict, node_id: str, patch: dict) -> None:
    """模拟画布参数面板写回 `data.params`（IrNode.updateParams 的口径）。"""
    data = _node_by_id(graph, node_id)["data"]
    data["params"] = {**(data.get("params") or {}), **patch}


# ---------------------------------------------------------------------------
# 往返一致性（service 层）
# ---------------------------------------------------------------------------

def test_roundtrip_restores_canvas_param_edits():
    """IR → GraphIR → 画布改参 → IR：参数改动进结果、层级/连线保真、校验通过、出码可见。"""
    ir0 = _minimal_ir()
    graph = _canvas_of(ir0)

    # 画布上改两个 leaf 的构造参数（宽度一并改到 16，保证图形状自洽）
    _edit_params(graph, "fc1", {"out_features": 16})
    _edit_params(graph, "head", {"in_features": 16})

    ir1 = ir_graphir.graphir_to_ir(graph)
    assert ir_schema.validate_ir(ir1) == []
    nodes = ir_schema.nodes_by_id(ir1)

    # 画布改参原样进 IR（本修复的核心价值）
    assert nodes["fc1"]["params"] == {"in_features": 4, "out_features": 16}
    assert nodes["head"]["params"] == {"in_features": 16, "out_features": 8}
    # 结构与层级保真
    assert nodes["fc1"]["parent_id"] == "block"
    assert nodes["fc1"]["class_name"] == "nn.Linear"
    assert nodes["fc1"]["module_path"] == "block.0"
    assert nodes["fuse"]["kind"] == "op" and nodes["block"]["kind"] == "container"
    # 边保真（含 tensor_shape）
    assert {(e["from"], e["to"]) for e in ir1["edges"]} == \
           {(e["from"], e["to"]) for e in ir0["edges"]}
    assert all(e.get("tensor_shape") == [1, 8] for e in ir1["edges"])
    # IR 级元数据随图携带（画布本身没有 entry_class/source_file，靠 data.ir_meta 还原）
    assert ir1["root_id"] == "net"
    assert ir1["entry_class"] == "Net"
    assert ir1["task_type"] == "classification"
    assert ir1["source_file"] == "models/sample_net.py"
    assert ir1["input_spec"] == {"shape": [1, 4], "dtype": "float32"}

    # 再生成代码体现改后的参数（旧值不再出现）
    code = ir_codegen.generate(ir1)
    assert "nn.Linear(in_features=4, out_features=16)" in code
    assert "nn.Linear(in_features=4, out_features=8)" not in code
    assert "nn.Linear(in_features=16, out_features=8)" in code

    # 导出/训练唯一出口 network_export.generate：同一张图 → 同一份代码 + GeneratedModel 入口
    exported = network_export.generate(graph)
    assert "nn.Linear(in_features=4, out_features=16)" in exported
    assert "class GeneratedModel(nn.Module):" in exported
    assert "self.root = Decomp_net()" in exported
    assert network_export.generate(graph) == exported  # 纯函数：同图恒同码


def test_roundtrip_nested_modules_and_skip_edges():
    """多层 parentId 嵌套 + skip 边：往返后 validate_ir 通过、层级不丢、skip 语义复现。"""
    ir0 = _nested_ir()
    graph = _canvas_of(ir0)
    kinds = {(e["source"], e["target"]): e["kind"] for e in graph["edges"]}
    assert kinds[("a", "c")] == "skip", "夹具必须真的产出跳跃边（否则本用例没覆盖 skip）"
    # 画布边的 kind（data/skip）是由 IR 结构派生的：与 _edge_kind 重算结果逐条一致，
    # 故反向映射不需要它参与（它不携带超出 from/to 的语义）
    for e in graph["edges"]:
        assert e["kind"] == ir_graphir._edge_kind(ir0, {"from": e["source"], "to": e["target"]})

    # 老 graph.json 兼容：抹掉后端新增的承载字段（根标记 + IR 元数据）
    for n in graph["nodes"]:
        n["data"].pop("ir_root", None)
        n["data"].pop("ir_meta", None)

    ir1 = ir_graphir.graphir_to_ir(graph)
    assert ir_schema.validate_ir(ir1) == []
    parents = {n["id"]: n["parent_id"] for n in ir1["nodes"]}
    assert parents["sub"] == "net" and parents["sub2"] == "sub" and parents["lin2"] == "sub2"
    assert parents["a"] == "net" and parents["d"] == "net"
    # 无根标记/元数据时的兜底推导：唯一顶层节点 = 根，入口类名取根节点类名
    assert ir1["root_id"] == "net"
    assert ir1["entry_class"] == "Net"
    assert ir1["task_type"] == "other"          # 元数据缺失时的缺省，不伪造结论
    # skip 边由结构重算复现（kind 是派生字段，不进 IR）
    assert ir_graphir._edge_kind(ir1, {"from": "a", "to": "c"}) == "skip"
    # 多输入算子入边顺序按 targetHandle 还原：add(b, c) 不能颠倒
    assert [e["from"] for e in ir_schema.in_edges(ir1, "d")] == ["b", "c"]
    code = ir_codegen.generate(ir1)
    assert "torch.add(var_b, var_c)" in code
    assert "class Decomp_sub(nn.Module):" in code and "class Decomp_sub2(nn.Module):" in code


# ---------------------------------------------------------------------------
# 导出端点（HTTP）
# ---------------------------------------------------------------------------

def _client_with_tmp_projects(app_client, tmp_path, monkeypatch):
    """项目工作区落到临时目录（路径在请求期读取，故即时生效）。"""
    from app.services import project_manager

    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    return app_client


def _create_original(client, name: str = "原始项目") -> str:
    r = client.post("/api/projects", json={"project_type": "original", "source": "local-test", "name": name})
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def _create_structured(client, parent_project_id: str | None = None) -> str:
    body: dict = {"project_type": "structured", "name": "拆解网络"}
    if parent_project_id:
        body["parent_project_id"] = parent_project_id
    r = client.post("/api/projects", json=body)
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def _fake_env_python(project_id: str, tmp_path) -> str:
    """在项目工作区里放一个假解释器，满足 _project_python 的就绪探测。"""
    exe = tmp_path / "projects" / project_id / "env" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    return str(exe)


def _record_dataset(tmp_path, name: str = "ir_smoke") -> str:
    from app.services import knowledge_service as ks

    data_dir = tmp_path / f"ds_{name}"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "preprocessed.csv").write_text(
        "id,split,label,input\n1,train,0,0.5\n2,test,1,0.9\n", encoding="utf-8"
    )
    return ks.register_dataset({
        "dataset_id": f"ds_{name}",
        "name": name,
        "task_type": "tabular",
        "local_path": str(data_dir / "preprocessed.csv"),
    })


def _edited_canvas() -> dict:
    graph = _canvas_of(_minimal_ir())
    _edit_params(graph, "fc1", {"out_features": 16})
    _edit_params(graph, "head", {"in_features": 16})
    return graph


def test_export_ir_graph_endpoint_uses_canvas_edits(app_client, tmp_path, monkeypatch):
    """ir 图经保存后导出 200：代码含预期类名/参数，且参数是**画布上改后**的值，可编译。"""
    client = _client_with_tmp_projects(app_client, tmp_path, monkeypatch)
    project_id = _create_structured(client)
    graph = _edited_canvas()
    assert client.put(f"/api/projects/{project_id}/graph", json=graph).status_code == 200

    r = client.get(f"/api/networks/{project_id}/export")
    assert r.status_code == 200, r.text
    code = r.json()["code"]
    assert "class Decomp_net(nn.Module):" in code
    assert "class GeneratedModel(nn.Module):" in code     # train.py 的 import 契约
    assert "self.root = Decomp_net()" in code
    assert "self.block = nn.Sequential(nn.Linear(in_features=4, out_features=16), nn.ReLU())" in code
    assert "self.head = nn.Linear(in_features=16, out_features=8)" in code
    assert "torch.add(var_block, var_head)" in code

    model_py = tmp_path / "exported_ir_model.py"
    model_py.write_text(code, encoding="utf-8")
    py_compile.compile(str(model_py), doraise=True)

    # 再改一次参数（画布保存 → 导出跟着变：导出即所存）
    _edit_params(graph, "fc1", {"out_features": 4})
    _edit_params(graph, "head", {"in_features": 4})
    assert client.put(f"/api/projects/{project_id}/graph", json=graph).status_code == 200
    code2 = client.get(f"/api/networks/{project_id}/export").json()["code"]
    assert "nn.Linear(in_features=4, out_features=4)" in code2
    assert "out_features=16" not in code2


def test_export_ir_graph_invalid_is_400(app_client, tmp_path, monkeypatch):
    """ir 图缺字段/根不唯一/与标准节点混拼 → 400 + 定位到节点，不 500。"""
    client = _client_with_tmp_projects(app_client, tmp_path, monkeypatch)
    project_id = _create_structured(client)

    def _put(graph: dict) -> None:
        assert client.put(f"/api/projects/{project_id}/graph", json=graph).status_code == 200

    def _detail() -> str:
        r = client.get(f"/api/networks/{project_id}/export")
        assert r.status_code == 400, f"应 400（可读原因），实际 {r.status_code}: {r.text}"
        return r.json()["detail"]

    # ① kind 非法
    g = _edited_canvas()
    _node_by_id(g, "fc1")["data"]["kind"] = "bogus"
    _put(g)
    detail = _detail()
    assert "kind 缺失/非法" in detail and "fc1" in detail

    # ② leaf 缺必填构造参数
    g = _edited_canvas()
    _node_by_id(g, "fc1")["data"]["params"] = {"in_features": 4}
    _put(g)
    detail = _detail()
    assert "out_features" in detail and "fc1" in detail

    # ③ 根节点不唯一（抹掉承载字段后有两个顶层节点）
    g = _edited_canvas()
    for n in g["nodes"]:
        n["data"].pop("ir_root", None)
        n["data"].pop("ir_meta", None)
    _node_by_id(g, "block")["parentId"] = None
    _put(g)
    detail = _detail()
    assert "根节点" in detail and "block" in detail

    # ④ 参数不是键值对
    g = _edited_canvas()
    _node_by_id(g, "fc1")["data"]["params"] = "in_features=4"
    _put(g)
    detail = _detail()
    assert "params" in detail and "fc1" in detail

    # ⑤ ir 节点与标准画布节点混拼（两个引擎都放不下，明确拒绝而不是各挑一半）
    g = _edited_canvas()
    g["nodes"].append({"id": "std1", "type": "relu_layer", "data": {}})
    _put(g)
    detail = _detail()
    assert "同时含" in detail and "标准画布节点" in detail


# ---------------------------------------------------------------------------
# 训练链路：同源代码 + 入队前守卫
# ---------------------------------------------------------------------------

def test_train_ir_graph_uses_same_code_as_export(app_client, tmp_path, monkeypatch):
    """训练落盘的 model.py 与导出端点返回的代码**逐字节相同**（导出即所存即所训）。"""
    client = _client_with_tmp_projects(app_client, tmp_path, monkeypatch)
    original_id = _create_original(client)
    project_id = _create_structured(client, original_id)
    _fake_env_python(original_id, tmp_path)
    dataset_id = _record_dataset(tmp_path)
    graph = _edited_canvas()
    assert client.put(f"/api/projects/{project_id}/graph", json=graph).status_code == 200

    from app.services import network_service, proc_util, project_manager

    async def fake_run(cmd, *, cwd, timeout):
        # 假训练脚本：把指标写进 argv 末位的 out_json（与既有用例同风格）
        run_dir = Path(cwd)
        assert (run_dir / "train.py").exists()
        Path(cmd[6]).write_text(
            json.dumps({"metrics": {"loss": 0.2, "accuracy": 0.9}}), encoding="utf-8"
        )
        return 0, "[epoch 1/1] loss=0.200000"

    monkeypatch.setattr(proc_util, "run_command", fake_run)

    params = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "environment_project_id": original_id,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.01,
    }
    asyncio.run(network_service._run_train(params, "task-ir-001"))

    ws = Path(project_manager.get_project(project_id)["workspace_path"])
    model_py = (ws / "runs" / "task-ir-001" / "model.py").read_text(encoding="utf-8")
    assert "class GeneratedModel(nn.Module):" in model_py
    assert "nn.Linear(in_features=4, out_features=16)" in model_py   # 训练用的是画布改后的参数
    assert model_py == client.get(f"/api/networks/{project_id}/export").json()["code"]

    runs = client.get(f"/api/networks/{project_id}/runs").json()
    assert [r["status"] for r in runs] == ["success"]
    assert json.loads(runs[0]["metrics"]) == {"loss": 0.2, "accuracy": 0.9}


def test_run_rejects_broken_ir_graph_before_enqueue(app_client, tmp_path, monkeypatch):
    """发起训练：不合法 ir 图入队前即 400（可定位），合法 ir 图正常入队（守卫不误伤）。"""
    client = _client_with_tmp_projects(app_client, tmp_path, monkeypatch)
    original_id = _create_original(client)
    project_id = _create_structured(client, original_id)
    _fake_env_python(original_id, tmp_path)
    dataset_id = _record_dataset(tmp_path)

    broken = _edited_canvas()
    _node_by_id(broken, "fc1")["data"]["params"] = {"in_features": 4}   # 缺 out_features
    assert client.put(f"/api/projects/{project_id}/graph", json=broken).status_code == 200
    r = client.post(f"/api/networks/{project_id}/run", json={"dataset_id": dataset_id})
    assert r.status_code == 400, r.text
    assert "fc1" in r.json()["detail"] and "out_features" in r.json()["detail"]

    # 正向对照：合法 ir 图能入队（monkeypatch 掉建任务，避免后台 worker 真跑）
    from app.services import task_manager

    calls: list[dict] = []

    def fake_create(task_type, project_id=None, params=None):
        calls.append({"task_type": task_type, "project_id": project_id, "params": params})
        return "task-ir-queued"

    monkeypatch.setattr(task_manager, "create_task", fake_create)
    assert client.put(f"/api/projects/{project_id}/graph",
                      json=_edited_canvas()).status_code == 200
    r = client.post(f"/api/networks/{project_id}/run", json={"dataset_id": dataset_id})
    assert r.status_code == 200, r.text
    assert r.json()["task_id"] == "task-ir-queued"
    assert calls and calls[0]["task_type"] == "network_train"


def test_roundtrip_carries_code_hint_for_non_whitelist_op():
    """白名单外的算子靠 code_hint 才能再生成：往返必须带上它（丢了就 400，这是不可逆字段）。"""
    ir = {
        "schema_version": "1.0",
        "source_file": "models/head_net.py",
        "entry_class": "Net",
        "task_type": "classification",
        "input_spec": {"shape": [1, 4, 2]},
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "parent_id": None},
            {"id": "lin", "kind": "leaf", "class_name": "nn.Linear", "parent_id": "net",
             "params": {"in_features": 4, "out_features": 8}},
            {"id": "gi", "kind": "op", "class_name": "getitem", "parent_id": "net",
             "params": {}, "code_hint": "torch.flatten({inputs}, 1)", "uncertain": True},
        ],
        "edges": [{"from": "lin", "to": "gi", "tensor_shape": [1, 8]}],
    }
    assert ir_schema.validate_ir(ir) == []
    graph = _canvas_of(ir)
    assert _node_by_id(graph, "gi")["data"]["code_hint"] == "torch.flatten({inputs}, 1)"

    ir2 = ir_graphir.graphir_to_ir(graph)
    assert ir_schema.validate_ir(ir2) == []
    gi = ir_schema.nodes_by_id(ir2)["gi"]
    assert gi["code_hint"] == "torch.flatten({inputs}, 1)"
    assert gi.get("uncertain") is True
    code = network_export.generate(graph)
    assert "var_gi = torch.flatten(var_lin, 1)" in code

    # 反向对照：code_hint 一丢，再生成引擎就拒（400 的可定位原因）——所以必须随 data 携带
    for n in graph["nodes"]:
        n["data"].pop("code_hint", None)
    ir3 = ir_graphir.graphir_to_ir(graph)
    assert ir_schema.validate_ir(ir3) == []
    with pytest.raises(network_export.ExportError, match="getitem"):
        network_export.generate(graph)


def test_standard_graph_is_not_dispatched_to_ir_path():
    """标准画布图（含 module_ref）不被误判为 ir 图，仍走原画布引擎。"""
    standard = {
        "nodes": [
            {"id": "n1", "type": "linear_layer", "data": {"in_features": 4, "out_features": 8}},
            {"id": "n2", "type": "relu_layer", "data": {}},
        ],
        "edges": [{"id": "e1", "source": "n1", "target": "n2", "targetHandle": "in-0",
                   "data": {"label": "out_n1"}}],
    }
    assert network_export.is_ir_graph(standard) is False
    code = network_export.generate(standard)
    assert "self.n1_layer = nn.Linear(in_features=4, out_features=8)" in code
    assert "Decomp_" not in code

    # 空图仍然走原路径（拆解项目刚创建、画布还没放节点）
    empty = {"nodes": [], "edges": []}
    assert network_export.is_ir_graph(empty) is False
    assert "class GeneratedModel(nn.Module):" in network_export.generate(empty)
