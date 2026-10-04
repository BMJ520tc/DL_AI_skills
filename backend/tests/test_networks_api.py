"""画布网络 API（阶段4 4c）：导出同源、运行参数校验、训练编排与运行记录。

工作区目录经 monkeypatch 指向临时路径，测试不触碰真实 data/projects（AGENTS.md 规矩 5）。
训练脚本执行经 monkeypatch 掉 proc_util.run_command，不真实跑 torch。
"""
from __future__ import annotations

import asyncio
import json
import re

import pytest


@pytest.fixture()
def tmp_networks(app_client, tmp_path, monkeypatch):
    """项目工作区落到临时目录（应用已随 app_client 启动，路径在请求期读取故即时生效）。"""
    from app.services import project_manager

    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    return app_client, tmp_path


def _create_original(client, name: str = "原始项目") -> str:
    r = client.post("/api/projects", json={"project_type": "original", "source": "local-test", "name": name})
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def _create_structured(client, parent_project_id: str | None) -> str:
    body: dict = {"project_type": "structured", "name": "画布网络"}
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


def _record_dataset(tmp_path, name: str = "smoke") -> str:
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


def _record_module_ref(path) -> None:
    """入库一个双类模块包（根类在后），供 module_ref 内联。"""
    from app.services import knowledge_service as ks

    pkg = path / "mod_ref_0001"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "module.py").write_text(
        "import torch.nn as nn\n"
        "class _Inner(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 8)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "class ModRef(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.inner = _Inner()\n"
        "    def forward(self, x):\n"
        "        return self.inner(x)\n",
        encoding="utf-8",
    )
    ks.record_module({
        "module_id": "mod_ref_0001",
        "module_version": "v1",
        "name": "ModRef",
        "description": None,
        "source_project_id": None,
        "source_paper_id": None,
        "task_type": None,
        "input_spec": None,
        "output_spec": None,
        "params_schema": {"hidden_size": {"type": "int", "default": 128}},
        "tags": None,
        "verification": None,
        "saved_module_compat": json.dumps({
            "id": "mod_ref_0001:v1", "name": "ModRef", "version": "v1",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
            "graph": {"nodes": [], "edges": []},
        }),
        "path": str(pkg),
    })


# ---------------------------------------------------------------------------
# 导出（保存收口 + 两端一致：前端 PUT graph 落库，后端同一引擎再生成）
# ---------------------------------------------------------------------------

def test_export_standard_chain(tmp_networks):
    """标准节点混拼（Linear→ReLU→Linear）经 PUT/GET graph 保存后导出再生成代码。"""
    client, _ = tmp_networks
    project_id = _create_structured(client, None)

    body = {
        "nodes": [
            {"id": "n1", "type": "linear_layer", "data": {"in_features": 4, "out_features": 8, "bias": True}},
            {"id": "n2", "type": "relu_layer", "data": {}},
            {"id": "n3", "type": "linear_layer", "data": {"in_features": 8, "out_features": 2, "bias": False}},
        ],
        "edges": [
            # 句柄/标签按画布 onConnect 口径：linear 无静态源句柄（label 无后缀），
            # relu 源句柄 out-0（label 带后缀，再生成时清洗为下划线）
            {"id": "e1", "source": "n1", "target": "n2", "targetHandle": "in-0",
             "data": {"label": "out_n1"}},
            {"id": "e2", "source": "n2", "sourceHandle": "out-0", "target": "n3",
             "targetHandle": "in-0", "data": {"label": "out_n2_out-0"}},
        ],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    # 保存收口：读回与写入一致（module_ref 混拼见下一条）
    saved = client.get(f"/api/projects/{project_id}/graph").json()
    assert [n["id"] for n in saved["nodes"]] == ["n1", "n2", "n3"]

    r = client.get(f"/api/networks/{project_id}/export")
    assert r.status_code == 200, r.text
    code = r.json()["code"]
    # 金标准整串比对（阶段4 4c「导出即所存即所训」；与前端生成器逐字节一致由
    # scripts/export_parity.py 复核，见《模块详细设计》7.4）
    assert code == (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class GeneratedModel(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.n1_layer = nn.Linear(in_features=4, out_features=8)\n"
        "        self.n2_layer = nn.ReLU()\n"
        "        self.n3_layer = nn.Linear(in_features=8, out_features=2, bias=False)\n"
        "\n"
        "    def forward(self, x):\n"
        "        out_n1 = self.n1_layer(x)\n"
        "        out_n2_out_0 = self.n2_layer(out_n1)\n"
        "        out_n3 = self.n3_layer(out_n2_out_0)\n"
        "        return out_n3\n"
    )


def test_export_tensor_repeat_separate_from_control_flow_repeat(tmp_networks):
    """张量 Repeat（键 repeat_tensor）可导出；控制流重复块（键 repeat_layer）仍拒绝导出。

    基底遗留缺陷：两个节点曾共用类型键 `repeat_layer`，注册表后写覆盖前写——张量 Repeat
    算子拖不出来，且导出被 `UNSUPPORTED_TYPES` 当作控制流拒绝。拆键后两者互不影响。
    """
    client, _ = tmp_networks
    project_id = _create_structured(client, None)

    # 1) 张量 torch.repeat 算子：按前端 nodes/pytorch_core/RepeatNode.tsx 语义
    #    （forward `out = in.repeat(<repeats>)`，init 仅注释）导出。
    body = {
        "nodes": [
            {"id": "a", "type": "input_layer", "data": {}},
            {"id": "b", "type": "repeat_tensor", "data": {"repeats": "2,3"}},
        ],
        "edges": [
            {"id": "e1", "source": "a", "target": "b", "targetHandle": "in-0",
             "data": {"label": "out_a"}},
        ],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    r = client.get(f"/api/networks/{project_id}/export")
    assert r.status_code == 200, r.text
    code = r.json()["code"]
    assert "# repeat handled in forward" in code
    assert "out_b = out_a.repeat(2,3)" in code
    assert "return out_b" in code

    # 2) 控制流「重复块」容器（键名保持 repeat_layer）：**空内部图**也能导出（`#Empty Loop` 直通）。
    control_body = {
        "nodes": [
            {"id": "a", "type": "input_layer", "data": {}},
            {"id": "loop", "type": "repeat_layer", "data": {"repetitions": 3}},
        ],
        "edges": [
            {"id": "e1", "source": "a", "target": "loop", "targetHandle": "in-external",
             "data": {"label": "out_a"}},
        ],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=control_body).status_code == 200
    empty_loop = client.get(f"/api/networks/{project_id}/export")
    assert empty_loop.status_code == 200, empty_loop.text
    loop_code = empty_loop.json()["code"]
    assert "#Empty Loop" in loop_code
    assert "out_loop = out_a # Empty Loop" in loop_code
    assert "return out_loop" in loop_code


def _control_flow_graph(kind: str) -> dict:
    """控制流容器图（形状照真实画布：子节点同时在顶层 nodes 且带 parentId，边界边也在顶层 edges）。"""
    child = {"id": "c1", "type": "linear_layer", "parentId": "ctr", "extent": "parent",
             "position": {"x": 20, "y": 40},
             "data": {"in_features": 4, "out_features": 4}}
    internal_edges = [
        {"id": "ie1", "source": "ctr", "sourceHandle": "in-internal", "target": "c1",
         "targetHandle": "in-0", "data": {"label": "c1_in"}},
        {"id": "ie2", "source": "c1", "target": "ctr", "targetHandle": "out-internal",
         "data": {"label": "out_c1"}},
    ]
    return {
        "nodes": [
            {"id": "in1", "type": "input_layer", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "ctr", "type": kind, "position": {"x": 200, "y": 0}, "data": {
                "repetitions": 2, "internalNodes": [child], "internalEdges": internal_edges}},
            child,
        ],
        "edges": [
            {"id": "e1", "source": "in1", "target": "ctr",
             "targetHandle": "in-external" if kind == "repeat_layer" else "in",
             "data": {"label": "out_in1"}},
            *internal_edges,
        ],
    }


def test_export_control_flow_containers(tmp_networks):
    """控制流容器（repeat_layer / module_list）可导出、可编译，且模型真的用上容器输出。

    两条钉死口径（与前端逐字节一致所必需，见 scripts/export_parity.py）：
    - `repeat_layer` 的子层 init **出现两次**（容器内联一份 + 外层平铺一份，幂等赘余）；
    - `module_list` 的子层 init **只出现一次**（`encapsulatesChildInit`：子层只在 ModuleList 内）。
    同时守住「容器的内部边界边不算它的输入/输出边」——否则 forward 末行会写成内部边名字、
    整图找不到输出节点，导出代码变成 `return x`（模型不接输入）。
    """
    import py_compile

    client, tmp_path = tmp_networks

    for kind, expected_init_count, expected_ret in (
        ("repeat_layer", 2, "return out_ctr"),
        ("module_list", 0, "return out_ctr_out"),
    ):
        project_id = _create_structured(client, None)
        graph = _control_flow_graph(kind)
        assert client.put(f"/api/projects/{project_id}/graph", json=graph).status_code == 200
        r = client.get(f"/api/networks/{project_id}/export")
        assert r.status_code == 200, r.text
        code = r.json()["code"]
        assert code.count("self.c1_layer = nn.Linear") == expected_init_count, code
        assert expected_ret in code, code
        assert "\n        return x\n" not in code, code
        p = tmp_path / f"{kind}.py"
        p.write_text(code, encoding="utf-8")
        py_compile.compile(str(p), doraise=True)


def test_export_rejects_ir_standard_mix(tmp_networks):
    """ir 节点与标准节点混拼 → 400，且文案说明「两类节点走不同引擎」。"""
    client, _ = tmp_networks
    project_id = _create_structured(client, None)
    body = {
        "nodes": [
            {"id": "ir1", "type": "ir", "data": {"kind": "module"}},
            {"id": "n1", "type": "relu_layer", "data": {}},
        ],
        "edges": [],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    r = client.get(f"/api/networks/{project_id}/export")
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "ir" in detail and "标准" in detail


def test_export_positional_encoding_is_executable(tmp_networks):
    """含位置编码层的导出代码必须可编译（阶段4：join 写死字面 '\\n' 会 SyntaxError）。"""
    import py_compile

    client, tmp_path = tmp_networks
    project_id = _create_structured(client, None)
    body = {
        "nodes": [
            {"id": "a", "type": "input_layer", "data": {}},
            {"id": "b", "type": "positional_encoding_layer", "data": {"dim": 64}},
        ],
        "edges": [{"id": "e1", "source": "a", "target": "b", "targetHandle": "in-0",
                   "data": {"label": "out_a"}}],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    code = client.get(f"/api/networks/{project_id}/export").json()["code"]
    p = tmp_path / "exported_model.py"
    p.write_text(code, encoding="utf-8")
    py_compile.compile(str(p), doraise=True)


def test_put_graph_rejects_malformed_graph(tmp_networks):
    """非法 GraphIR 在入口即 400（节点/边元素形态校验），不落盘、不留后续 500。"""
    client, _ = tmp_networks
    project_id = _create_structured(client, None)
    cases = [
        {"nodes": ["x"], "edges": []},                       # 节点非对象
        {"nodes": [{"type": "relu_layer"}], "edges": []},     # 节点缺 id
        {"nodes": [{"id": "a"}], "edges": []},                # 节点缺 type
        {"nodes": [], "edges": [{"source": "a"}]},            # 边缺 target
    ]
    for body in cases:
        r = client.put(f"/api/projects/{project_id}/graph", json=body)
        assert r.status_code == 400, f"{body} -> {r.status_code}"
    # 未落盘：仍是创建时的空图
    assert client.get(f"/api/projects/{project_id}/graph").json() == {"nodes": [], "edges": []}


def test_create_structured_project_is_ready(tmp_networks):
    """画布新建结构化项目：创建即 ready（4a），响应与库内一致。"""
    client, _ = tmp_networks
    r = client.post("/api/projects", json={"project_type": "structured", "name": "新模型"})
    assert r.status_code == 200, r.text
    project_id = r.json()["project_id"]
    assert r.json()["status"] == "ready"
    assert client.get(f"/api/projects/{project_id}").json()["status"] == "ready"


def test_export_module_ref_and_edited_params_rejected(tmp_networks):
    """module_ref 内联模块代码；画布改过固化参数则拒绝导出（不静默丢弃）。"""
    client, tmp_path = tmp_networks
    _record_module_ref(tmp_path)
    project_id = _create_structured(client, None)

    body = {
        "nodes": [
            {"id": "m1", "type": "module_ref", "data": {
                "moduleId": "mod_ref_0001:v1",
                "handles": {"inputs": ["in"], "outputs": ["out"]},
            }},
        ],
        "edges": [],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    saved = client.get(f"/api/projects/{project_id}/graph").json()
    assert saved["nodes"][0]["data"]["moduleId"] == "mod_ref_0001:v1"

    r = client.get(f"/api/networks/{project_id}/export")
    assert r.status_code == 200, r.text
    code = r.json()["code"]
    assert "class ModRef(nn.Module):" in code          # 模块代码内联
    assert "self.m1_layer = ModRef()" in code          # 无参实例化（参数已固化）

    # 画布上改了参数 → 400 并说明原因
    body["nodes"][0]["data"]["hidden_size"] = 999
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    r = client.get(f"/api/networks/{project_id}/export")
    assert r.status_code == 400
    assert "已固化" in r.json()["detail"]


def test_export_binds_multi_input_by_handle(tmp_networks):
    """不对称算子按 targetHandle 绑定输入，不随边数组顺序颠倒（复核 D1）。

    边数组顺序与句柄顺序相反时，sub 的 forward 仍必须是 «in-0 的值 - in-1 的值»。
    """
    client, _ = tmp_networks
    project_id = _create_structured(client, None)
    body = {
        "nodes": [
            {"id": "a", "type": "linear_layer", "data": {"in_features": 4, "out_features": 4}},
            {"id": "b", "type": "linear_layer", "data": {"in_features": 4, "out_features": 4}},
            {"id": "s", "type": "sub_layer", "data": {}},
        ],
        # in-1 的边排在 in-0 之前——按数组序绑定会算成 out_b - out_a
        "edges": [
            {"id": "eb", "source": "b", "target": "s", "targetHandle": "in-1",
             "data": {"label": "out_b"}},
            {"id": "ea", "source": "a", "target": "s", "targetHandle": "in-0",
             "data": {"label": "out_a"}},
        ],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    code = client.get(f"/api/networks/{project_id}/export").json()["code"]
    assert "out_a - out_b" in code, code
    assert "out_b - out_a" not in code


def test_export_no_undefined_vars_for_handleless_edges(tmp_networks):
    """边缺 sourceHandle（旧图/导入图）时，生产者与消费者取名必须一致（复核 D3）。

    有声明源句柄的节点若凭空造名（`out_<id>_<handle>`），消费侧却按 label/边 id 取名，
    导出的 forward 会引用未定义变量——两端逐字节一致也照样跑不起来。
    """
    client, _ = tmp_networks
    project_id = _create_structured(client, None)
    body = {
        "nodes": [
            {"id": "a", "type": "input_layer", "data": {}},
            {"id": "u", "type": "upsample_layer", "data": {"mode": "bilinear", "scale": 2}},
        ],
        # 无 sourceHandle 的边（历史/导入图口径）
        "edges": [
            {"id": "e1", "source": "a", "target": "u", "targetHandle": "in-0",
             "data": {"label": "out_a"}},
        ],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    code = client.get(f"/api/networks/{project_id}/export").json()["code"]

    assigned = {
        line.strip().split("=")[0].strip()
        for line in code.splitlines()
        if "=" in line and not line.strip().startswith(("#", "def "))
        and " " not in line.strip().split("=")[0].strip()
    }
    used = set(re.findall(r"\bout_[A-Za-z0-9_]+\b", code))
    assert used - assigned == set(), f"未定义引用：{used - assigned}\n{code}"


def test_export_isolates_duplicate_module_class_names(tmp_networks):
    """两个模块内联出同名类时给第二个加模块键后缀，避免遮蔽（复核 D2）。

    模块四按 `Decomp_<根节点 id>` 命名，不同模块根 id 相同时会撞名；撞名会让两个引用
    实例化同一个类（模型错）。
    """
    client, tmp_path = tmp_networks
    project_id = _create_structured(client, None)
    _record_module_ref(tmp_path)  # mod_ref_0001:v1，根类 ModRef

    # 第二个模块：不同 id、**同样的类名**
    from app.services import knowledge_service as ks

    pkg = tmp_path / "mod_ref_0002"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "module.py").write_text(
        "import torch.nn as nn\n"
        "class ModRef(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(8, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n",
        encoding="utf-8",
    )
    ks.record_module({
        "module_id": "mod_ref_0002", "module_version": "v1", "name": "ModRef",
        "description": None, "source_project_id": None, "source_paper_id": None,
        "task_type": None, "input_spec": None, "output_spec": None,
        "params_schema": None, "tags": None, "verification": None,
        "saved_module_compat": json.dumps({
            "id": "mod_ref_0002:v1", "name": "ModRef", "version": "v1",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
            "graph": {"nodes": [], "edges": []},
        }),
        "path": str(pkg),
    })

    body = {
        "nodes": [
            {"id": "m1", "type": "module_ref", "data": {
                "moduleId": "mod_ref_0001:v1", "handles": {"inputs": ["in"], "outputs": ["out"]}}},
            {"id": "m2", "type": "module_ref", "data": {
                "moduleId": "mod_ref_0002:v1", "handles": {"inputs": ["in"], "outputs": ["out"]}}},
        ],
        "edges": [],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    code = client.get(f"/api/networks/{project_id}/export").json()["code"]

    # 第一个保持原名；第二个带模块键后缀，两处实例化引用的类名不相同
    assert code.count("class ModRef(nn.Module):") == 1
    assert "class ModRef_mod_ref_0002_v1(nn.Module):" in code, code
    assert "self.m1_layer = ModRef()" in code
    assert "self.m2_layer = ModRef_mod_ref_0002_v1()" in code


def test_export_accepts_legacy_module_package(tmp_networks):
    """历史模块包（v4 前）也能内联：代码文件名是 `model.py`，且 `path` 记成了 module.json 的**文件**路径。

    画布上引用这类老版本不能让导出直接 400（阶段3 早期产物 ↔ 阶段4 消费的兼容）。
    """
    client, tmp_path = tmp_networks
    project_id = _create_structured(client, None)

    from app.services import knowledge_service as ks

    pkg = tmp_path / "mod_legacy_0001"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "model.py").write_text(
        "import torch.nn as nn\n"
        "class LegacyNet(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n",
        encoding="utf-8",
    )
    (pkg / "module.json").write_text(
        json.dumps({"module_id": "mod_legacy_0001"}), encoding="utf-8")
    ks.record_module({
        "module_id": "mod_legacy_0001", "module_version": "v1", "name": "LegacyNet",
        "description": None, "source_project_id": None, "source_paper_id": None,
        "task_type": None, "input_spec": None, "output_spec": None,
        "params_schema": None, "tags": None, "verification": None,
        "saved_module_compat": json.dumps({
            "id": "mod_legacy_0001:v1", "name": "LegacyNet", "version": "v1",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
            "graph": {"nodes": [], "edges": []},
        }),
        "path": str(pkg / "module.json"),  # 旧记录口径：文件路径而非包目录
    })

    body = {
        "nodes": [{"id": "m1", "type": "module_ref", "data": {
            "moduleId": "mod_legacy_0001:v1", "handles": {"inputs": ["in"], "outputs": ["out"]}}}],
        "edges": [],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    r = client.get(f"/api/networks/{project_id}/export")
    assert r.status_code == 200, r.text
    code = r.json()["code"]
    assert "class LegacyNet(nn.Module):" in code
    assert "self.m1_layer = LegacyNet()" in code


def test_export_rejects_non_network_projects(tmp_networks):
    """画布网络接口只接受结构化项目：original 400（7.7-1 与 2.2 同口径）、不存在 404。"""
    client, _ = tmp_networks
    original_id = _create_original(client)
    assert client.get(f"/api/networks/{original_id}/export").status_code == 400
    assert client.get("/api/networks/not-exist/export").status_code == 404


# ---------------------------------------------------------------------------
# 运行参数校验（POST run 前置闸门）
# ---------------------------------------------------------------------------

def test_run_rejects_bad_inputs(tmp_networks):
    """数据集缺失/无预处理产物、环境未就绪、超参越界：全部 400 并给出引导。"""
    client, tmp_path = tmp_networks
    original_id = _create_original(client)
    project_id = _create_structured(client, original_id)

    # 数据集不存在
    r = client.post(f"/api/networks/{project_id}/run", json={"dataset_id": "nope"})
    assert r.status_code == 400

    # 数据集存在但无本地产物（只有 registry 行）
    from app.services import knowledge_service as ks

    ks.register_dataset({"dataset_id": "ds_no_local", "name": "x", "local_path": None})
    r = client.post(f"/api/networks/{project_id}/run", json={"dataset_id": "ds_no_local"})
    assert r.status_code == 400
    assert "预处理" in r.json()["detail"]

    # 环境未就绪（父项目没有独立环境）→ 引导先走模块一建环境
    dataset_id = _record_dataset(tmp_path)
    r = client.post(f"/api/networks/{project_id}/run", json={"dataset_id": dataset_id})
    assert r.status_code == 400
    assert "建环境" in r.json()["detail"]

    # 指定了不存在的环境项目
    r = client.post(f"/api/networks/{project_id}/run", json={
        "dataset_id": dataset_id, "environment_project_id": "not-exist",
    })
    assert r.status_code == 400

    # 超参越界
    _fake_env_python(original_id, tmp_path)
    r = client.post(f"/api/networks/{project_id}/run", json={
        "dataset_id": dataset_id, "epochs": 0,
    })
    assert r.status_code == 400
    assert "epochs" in r.json()["detail"]


def test_run_requires_env_for_parentless_network(tmp_networks):
    """画布新建的网络没有父项目环境可复用：不指定环境即报错引导。"""
    client, tmp_path = tmp_networks
    project_id = _create_structured(client, None)
    dataset_id = _record_dataset(tmp_path)

    r = client.post(f"/api/networks/{project_id}/run", json={"dataset_id": dataset_id})
    assert r.status_code == 400
    assert "environment_project_id" in r.json()["detail"]


def test_run_options_lists_envs_and_datasets(tmp_networks):
    """运行面板初始化数据：父项目 + 就绪环境（假解释器）+ 有产物的数据集。"""
    client, tmp_path = tmp_networks
    original_id = _create_original(client)
    project_id = _create_structured(client, original_id)
    _fake_env_python(original_id, tmp_path)
    dataset_id = _record_dataset(tmp_path)

    r = client.get(f"/api/networks/{project_id}/run-options")
    assert r.status_code == 200, r.text
    opts = r.json()
    assert opts["parent_project_id"] == original_id
    assert [e["project_id"] for e in opts["environments"]] == [original_id]
    assert [d["dataset_id"] for d in opts["datasets"]] == [dataset_id]

    # 只有本地文件、没有 preprocessed.csv 的数据集不进面板（7.5「有预处理产物的注册条目」）
    from app.services import knowledge_service as ks

    bare = tmp_path / "ds_bare"
    bare.mkdir(exist_ok=True)
    (bare / "raw.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    ks.register_dataset({
        "dataset_id": "ds_bare", "name": "bare", "task_type": "tabular",
        "local_path": str(bare / "raw.csv"),
    })
    opts2 = client.get(f"/api/networks/{project_id}/run-options").json()
    assert [d["dataset_id"] for d in opts2["datasets"]] == [dataset_id]


# ---------------------------------------------------------------------------
# 训练编排（monkeypatch 掉训练脚本执行，handler 直跑——与 test_dataset_alignment 同风格）
# ---------------------------------------------------------------------------

def _insert_running_task(task_id: str, project_id: str, params: dict) -> None:
    """直接落一条 running 任务行，供 handler 直跑后查询任务进度。

    不经队列：app_client 走 lifespan 起的后台 worker 会并发执行同一任务，
    直插任务行可让「handler 直跑 + 任务进度查询」这条断言链完全确定。
    """
    from app.db.connection import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO task(task_id, task_type, project_id, params, status, created_at, updated_at) "
            "VALUES (?, 'network_train', ?, ?, 'running', '2026-10-04T00:00:00+00:00', "
            "'2026-10-04T00:00:00+00:00')",
            (task_id, project_id, json.dumps(params, ensure_ascii=False)),
        )
        conn.commit()
    finally:
        conn.close()


def _prepare_train(client, tmp_path) -> tuple[str, str, str]:
    """建「父原始项目 + 结构化网络 + 假环境 + 预处理数据集 + 模块包」并保存一次画布。"""
    original_id = _create_original(client)
    project_id = _create_structured(client, original_id)
    _fake_env_python(original_id, tmp_path)
    dataset_id = _record_dataset(tmp_path)
    _record_module_ref(tmp_path)
    body = {
        "nodes": [{"id": "m1", "type": "module_ref", "data": {
            "moduleId": "mod_ref_0001:v1",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
        }}],
        "edges": [],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200
    return original_id, project_id, dataset_id


def _train_params(project_id: str, dataset_id: str, original_id: str) -> dict:
    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "environment_project_id": original_id,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.01,
    }


async def _fake_ok_run(cmd, *, cwd, timeout):
    """假训练脚本：把指标写进 argv 末位的 out_json（与既有用例同风格）。"""
    from pathlib import Path

    Path(cmd[6]).write_text(
        json.dumps({"metrics": {"loss": 0.1, "accuracy": 0.95}}), encoding="utf-8"
    )
    return 0, "[epoch 1/1] loss=0.100000"


def test_train_orchestration_writes_run_record(tmp_networks, monkeypatch):
    """任务 handler 全链路：导出落盘 → 假执行产出指标 → run_record(run_type=train)。"""
    client, tmp_path = tmp_networks
    original_id = _create_original(client)
    project_id = _create_structured(client, original_id)
    _fake_env_python(original_id, tmp_path)
    dataset_id = _record_dataset(tmp_path)
    _record_module_ref(tmp_path)

    from app.services import network_service, proc_util, project_manager

    async def fake_run(cmd, *, cwd, timeout):
        # 校验落盘产物：model.py 与 train.py 均已写入运行目录
        from pathlib import Path

        run_dir = Path(cwd)
        assert (run_dir / "model.py").read_text(encoding="utf-8").startswith("import torch")
        assert (run_dir / "train.py").exists()
        # 假训练：把指标写进 argv 末位的 out_json
        Path(cmd[6]).write_text(
            json.dumps({"metrics": {"loss": 0.1, "accuracy": 0.95}}), encoding="utf-8"
        )
        return 0, "[epoch 1/2] loss=0.100000"

    monkeypatch.setattr(proc_util, "run_command", fake_run)

    body = {
        "nodes": [{"id": "m1", "type": "module_ref", "data": {
            "moduleId": "mod_ref_0001:v1",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
        }}],
        "edges": [],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200

    params = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "environment_project_id": original_id,
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.01,
    }
    asyncio.run(network_service._run_train(params, "task-net-001"))

    # 运行目录产物：模型代码 + 训练日志
    from pathlib import Path

    ws = Path(project_manager.get_project(project_id)["workspace_path"])
    assert (ws / "runs" / "task-net-001" / "model.py").exists()
    assert "class ModRef" in (ws / "runs" / "task-net-001" / "model.py").read_text(encoding="utf-8")
    assert (ws / "runs" / "task-net-001" / "train.log").exists()

    # run_record 落库：run_type=train、状态成功、指标齐备
    runs = client.get(f"/api/networks/{project_id}/runs").json()
    assert len(runs) == 1
    rec = runs[0]
    assert rec["run_type"] == "train"
    assert rec["status"] == "success"
    assert json.loads(rec["metrics"]) == {"loss": 0.1, "accuracy": 0.95}
    assert json.loads(rec["params"])["epochs"] == 2
    assert json.loads(rec["environment"])["environment_project_id"] == original_id
    assert rec["task_id"] == "task-net-001"
    assert rec["duration_s"] is not None


def test_train_task_endpoint_and_failure_path(tmp_networks, monkeypatch):
    """POST run 建任务入队；训练脚本失败 → 任务 failed、不写成功 run_record，
    但**落一条 failed 记录可检索**（《知识库与数据设计》五.2「报错只写 run_record」）。"""
    client, tmp_path = tmp_networks
    original_id = _create_original(client)
    project_id = _create_structured(client, original_id)
    _fake_env_python(original_id, tmp_path)
    dataset_id = _record_dataset(tmp_path)
    _record_module_ref(tmp_path)

    from app.services import network_service, proc_util

    async def failing_run(cmd, *, cwd, timeout):
        return 1, "Traceback: boom"

    monkeypatch.setattr(proc_util, "run_command", failing_run)

    body = {
        "nodes": [{"id": "m1", "type": "module_ref", "data": {
            "moduleId": "mod_ref_0001:v1",
            "handles": {"inputs": ["in"], "outputs": ["out"]},
        }}],
        "edges": [],
    }
    assert client.put(f"/api/projects/{project_id}/graph", json=body).status_code == 200

    r = client.post(f"/api/networks/{project_id}/run", json={
        "dataset_id": dataset_id, "epochs": 1,
    })
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    assert client.get(f"/api/tasks/{task_id}").status_code == 200

    # handler 直跑：脚本失败 → RuntimeError 带日志尾部；成功 run_record 不产生
    params = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "environment_project_id": original_id,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.01,
    }
    with pytest.raises(RuntimeError, match="退出码 1"):
        asyncio.run(network_service._run_train(params, task_id))
    # 成功记录不产生（/runs 只列 success）；失败记录可检索（五.2）。
    # 注：POST run 入队后后台 worker 可能也执行了一次同一任务，故按「至少一条且都属于本任务」断言。
    assert client.get(f"/api/networks/{project_id}/runs").json() == []
    from app.services import knowledge_service as ks

    failed = ks.list_runs(project_id, "train", status="failed")
    assert failed, "训练失败应落一条可检索的 run_record"
    assert all(r["task_id"] == task_id for r in failed), failed
    assert all("退出码 1" in (r["error"] or "") for r in failed), failed


# ---------------------------------------------------------------------------
# 运行即提交失败透出（4d-1 补强）：不连坐训练结果，但版本提交失败必须可查、不静默
# ---------------------------------------------------------------------------

def test_train_version_commit_failure_is_transparent(tmp_networks, monkeypatch):
    """commit_run 抛异常 → 任务仍 success、成功 run_record 仍在（既有口径保持），
    且失败透出在任务进度 version_error 与 run_record 失败留痕里（不静默）。"""
    client, tmp_path = tmp_networks
    original_id, project_id, dataset_id = _prepare_train(client, tmp_path)

    from app.services import network_service, proc_util, version_service

    def broken_commit_run(_project_id, _task_id, _metrics):
        raise RuntimeError("git commit 失败（模拟）")

    monkeypatch.setattr(proc_util, "run_command", _fake_ok_run)
    monkeypatch.setattr(version_service, "commit_run", broken_commit_run)

    params = _train_params(project_id, dataset_id, original_id)
    task_id = "task-net-vc-fail"
    _insert_running_task(task_id, project_id, params)

    # 不抛出：版本提交失败与训练结果解耦（不连坐）
    asyncio.run(network_service._run_train(params, task_id))

    # 既有断言保持：训练成功记录照常、指标齐备
    runs = client.get(f"/api/networks/{project_id}/runs").json()
    assert len(runs) == 1
    assert runs[0]["status"] == "success"
    assert runs[0]["task_id"] == task_id
    assert json.loads(runs[0]["metrics"]) == {"loss": 0.1, "accuracy": 0.95}

    # 新增：失败透出到任务进度（GET /api/tasks/{id} → progress.version_error）
    task = client.get(f"/api/tasks/{task_id}").json()
    progress = json.loads(task["progress"])
    assert progress["stage"] == "完成"
    assert "版本提交失败" in progress["version_error"]
    assert "训练结果已保留" in progress["version_error"]
    assert "版本节点未生成" in progress["version_error"]
    assert "git commit 失败（模拟）" in progress["version_error"]

    # 新增：run_record 留一条失败留痕（可检索），且不污染成功记录列表
    from app.services import knowledge_service as ks

    failed = ks.list_runs(project_id, "train", status="failed")
    assert len(failed) == 1
    assert failed[0]["task_id"] == task_id
    assert "版本提交失败" in (failed[0]["error"] or "")
    assert "git commit 失败（模拟）" in (failed[0]["error"] or "")


def test_train_version_commit_reports_written_but_uncommitted(tmp_networks, monkeypatch):
    """边界（4d-1 实施要点 3）：commit_run 先写盘 network_version.json、再 git 提交；
    只有提交步骤失败时文件已写盘、而版本节点未生成——文案应如实说明这一状态。"""
    client, tmp_path = tmp_networks
    original_id, project_id, dataset_id = _prepare_train(client, tmp_path)

    from app.services import network_service, proc_util, version_service

    real_run_git = version_service._run_git

    def git_fails_on_commit(ws, *args, **kwargs):
        if args and args[0] == "commit":
            raise RuntimeError("git commit 失败（模拟）")
        return real_run_git(ws, *args, **kwargs)

    monkeypatch.setattr(proc_util, "run_command", _fake_ok_run)
    monkeypatch.setattr(version_service, "_run_git", git_fails_on_commit)

    params = _train_params(project_id, dataset_id, original_id)
    task_id = "task-net-vc-uncommitted"
    _insert_running_task(task_id, project_id, params)
    asyncio.run(network_service._run_train(params, task_id))

    # 版本内容在：network_version.json 已写盘且带本次 run_summary
    ws = tmp_path / "projects" / project_id
    written = json.loads((ws / "network_version.json").read_text(encoding="utf-8"))
    assert written["run_summary"]["task_id"] == task_id
    # 版本节点不在：git 提交失败，版本树没有「训练运行」这一版
    tree = version_service.version_tree(project_id)
    assert all("训练运行" not in v["message"] for v in tree["versions"])

    progress = json.loads(client.get(f"/api/tasks/{task_id}").json()["progress"])
    assert "版本提交失败" in progress["version_error"]
    assert "已写盘但未提交" in progress["version_error"]
    assert "版本内容在、版本节点不在" in progress["version_error"]


def test_train_version_commit_success_has_no_version_error(tmp_networks, monkeypatch):
    """反向用例：commit_run 正常时不出现 version_error（防误报），也不写失败留痕。"""
    client, tmp_path = tmp_networks
    original_id, project_id, dataset_id = _prepare_train(client, tmp_path)

    from app.services import knowledge_service as ks, network_service, proc_util, version_service

    monkeypatch.setattr(proc_util, "run_command", _fake_ok_run)

    params = _train_params(project_id, dataset_id, original_id)
    task_id = "task-net-vc-ok"
    _insert_running_task(task_id, project_id, params)
    asyncio.run(network_service._run_train(params, task_id))

    # 任务进度不带 version_error（缺省或 null 均视为「无误报」，与项目创建响应同口径）
    progress = json.loads(client.get(f"/api/tasks/{task_id}").json()["progress"])
    assert progress.get("version_error") is None
    assert progress["stage"] == "完成"
    # 无失败留痕；版本节点已生成且带本次运行摘要
    assert ks.list_runs(project_id, "train", status="failed") == []
    tree = version_service.version_tree(project_id)
    assert tree["versions"][0]["meta"]["run_summary"]["task_id"] == task_id


# ---------------------------------------------------------------------------
# 需求五.3「每次运行生成一个版本节点」：训练失败也生成版本节点（4d-2 补强）
# ---------------------------------------------------------------------------

def test_train_failure_creates_version_node(tmp_networks, monkeypatch):
    """训练脚本失败 → 除 run_record(failed) 外，版本树里还要有一条「训练失败 <task_id>」版本节点
    （run_summary.status="failed"、错误摘要非空、无 metrics）；同一用例再跑一次成功训练，
    证明成功路径口径不变（提交信息仍为「训练运行 <task_id>」、status=success、metrics 齐备）。"""
    client, tmp_path = tmp_networks
    original_id, project_id, dataset_id = _prepare_train(client, tmp_path)

    from app.services import knowledge_service as ks, network_service, proc_util, version_service

    async def failing_run(cmd, *, cwd, timeout):
        return 1, "Traceback: boom"

    monkeypatch.setattr(proc_util, "run_command", failing_run)

    params = _train_params(project_id, dataset_id, original_id)
    task_id = "task-net-fail-ver"
    _insert_running_task(task_id, project_id, params)

    # 原始训练失败原因原样抛出：版本提交与否都不改变任务结果
    with pytest.raises(RuntimeError, match="退出码 1"):
        asyncio.run(network_service._run_train(params, task_id))

    # 既有口径保持：失败记录可检索
    assert [r["task_id"] for r in ks.list_runs(project_id, "train", status="failed")] == [task_id]

    # 新增：失败也生成版本节点（提交信息 + run_summary 形状）
    tree = version_service.version_tree(project_id)
    head = tree["versions"][0]
    assert head["message"] == f"训练失败 {task_id}"
    assert tree["current"] == head["short"]
    summary = head["meta"]["run_summary"]
    assert summary["status"] == "failed"
    assert summary["task_id"] == task_id
    assert summary["error"] and "退出码 1" in summary["error"]
    assert "metrics" not in summary
    # 线性演化：失败版本节点的父提交即下一个版本（与既有版本树口径一致）
    assert head["parents"] == [tree["versions"][1]["commit"]]

    # 成功路径不受影响：同样的链路成功时仍是「训练运行 …」+ status=success + metrics
    monkeypatch.setattr(proc_util, "run_command", _fake_ok_run)
    ok_task = "task-net-ok-ver"
    _insert_running_task(ok_task, project_id, params)
    asyncio.run(network_service._run_train(params, ok_task))

    tree2 = version_service.version_tree(project_id)
    ok_head = tree2["versions"][0]
    assert ok_head["message"] == f"训练运行 {ok_task}"
    assert ok_head["meta"]["run_summary"]["status"] == "success"
    assert ok_head["meta"]["run_summary"]["metrics"] == {"loss": 0.1, "accuracy": 0.95}
    # 失败版本节点仍留在树上（两个任务各一版）；成功列表不被失败版本污染
    assert any(v["message"] == f"训练失败 {task_id}" for v in tree2["versions"])
    runs = client.get(f"/api/networks/{project_id}/runs").json()
    assert [r["task_id"] for r in runs] == [ok_task]


def test_train_failure_version_commit_error_does_not_mask_failure(tmp_networks, monkeypatch):
    """版本提交本身失败时：原始训练失败原因照旧抛出、run_record(failed) 照旧落库（不掩盖、不连坐）。"""
    client, tmp_path = tmp_networks
    original_id, project_id, dataset_id = _prepare_train(client, tmp_path)

    from app.services import knowledge_service as ks, network_service, proc_util, version_service

    async def failing_run(cmd, *, cwd, timeout):
        return 1, "Traceback: boom"

    def broken_commit_run(*_args, **_kwargs):
        raise RuntimeError("git commit 失败（模拟）")

    monkeypatch.setattr(proc_util, "run_command", failing_run)
    monkeypatch.setattr(version_service, "commit_run", broken_commit_run)

    params = _train_params(project_id, dataset_id, original_id)
    task_id = "task-net-fail-vc"
    _insert_running_task(task_id, project_id, params)

    with pytest.raises(RuntimeError, match="退出码 1"):  # 原始失败原因，不是版本提交错误
        asyncio.run(network_service._run_train(params, task_id))

    failed = ks.list_runs(project_id, "train", status="failed")
    assert [r["task_id"] for r in failed] == [task_id]
    assert "退出码 1" in (failed[0]["error"] or "")
    # 版本节点确实没生成（提交失败），但这件事只体现在日志里，不改任务结论
    assert all("训练失败" not in v["message"]
               for v in version_service.version_tree(project_id)["versions"])
