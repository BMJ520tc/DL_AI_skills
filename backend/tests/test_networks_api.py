"""画布网络 API（阶段4 4c）：导出同源、运行参数校验、训练编排与运行记录。

工作区目录经 monkeypatch 指向临时路径，测试不触碰真实 data/projects（AGENTS.md 规矩 5）。
训练脚本执行经 monkeypatch 掉 proc_util.run_command，不真实跑 torch。
"""
from __future__ import annotations

import asyncio
import json

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
    assert "import torch" in code
    assert "class GeneratedModel(nn.Module):" in code
    assert "nn.Linear(in_features=4, out_features=8)" in code
    assert "nn.ReLU()" in code
    assert "out_n2_out_0 = self.n2_layer(out_n1)" in code
    assert "return out_n3" in code


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


def test_export_rejects_non_network_projects(tmp_networks):
    """画布网络接口只接受结构化项目：original 403、不存在 404。"""
    client, _ = tmp_networks
    original_id = _create_original(client)
    assert client.get(f"/api/networks/{original_id}/export").status_code == 403
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


# ---------------------------------------------------------------------------
# 训练编排（monkeypatch 掉训练脚本执行，handler 直跑——与 test_dataset_alignment 同风格）
# ---------------------------------------------------------------------------

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
    """POST run 建任务入队；训练脚本失败 → 任务 failed、不写成功 run_record。"""
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
    assert client.get(f"/api/networks/{project_id}/runs").json() == []
