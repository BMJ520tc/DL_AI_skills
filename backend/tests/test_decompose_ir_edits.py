"""模块四「IR 调参 / 补输入规格 / 验证新鲜度 / 入库前置」的回归用例。

完备性核查登记：拆解链路此前只有模块签名与模块列表两条契约用例，
`verification_status` / `update_node_params` / `update_input_spec` / `ingest_precheck` /
`decompose_precheck` / `regenerate` 的守卫与新鲜度语义**没有任何自动化用例**，
而这几个函数正是「调参后旧验证必须变 stale、入库必须挡住过期验证」的闸门，
一旦回归只能靠人工发现。本文件补上这些判据。

口径（与《模块详细设计》6.2/6.4/6.5 一致）：
- 验证新鲜度：none（未验证）/ valid（ir_hash 一致）/ stale（IR 已改未重验）；
- 调参与补 input_spec 都会改动 ir_hash → 旧验证转 stale；
- 入库前置按 缺产物→LookupError、未通过→PermissionError、过期/已入库→ValueError 区分。

只使用临时库（conftest.isolated_db）与临时工作区/模块目录
（monkeypatch project_manager.PROJECTS_DIR / decompose_service.MODULES_DIR），不碰真实 data/。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import decompose_service, knowledge_service, project_manager
from app.services.ir_schema import ir_hash


@pytest.fixture()
def ir_env(isolated_db, tmp_path, monkeypatch):
    """原始项目 + 临时工作区/模块目录。"""
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    monkeypatch.setattr(decompose_service, "MODULES_DIR", modules_dir)
    project_id = project_manager.create_project("original", source="local-test")
    return {"project_id": project_id, "modules_dir": modules_dir}


def _ir(project_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "project_id": project_id,
        "source_file": "model.py",
        "entry_class": "Net",
        "task_type": "classification",
        "input_spec": {"shape": [1, 8], "dtype": "float32"},
        "root_id": "net",
        "nodes": [
            {"id": "net", "kind": "module", "class_name": "Net", "params": {"hidden": 8},
             "parent_id": None, "module_path": "", "output_shape": [1, 8]},
            {"id": "fc", "kind": "leaf", "class_name": "nn.Linear", "module_path": "fc",
             "params": {"in_features": 8, "out_features": 8},
             "parent_id": "net", "input_shape": [1, 8], "output_shape": [1, 8]},
        ],
        "edges": [],
    }


def _reports(project_id: str) -> Path:
    ws = Path(project_manager.get_project(project_id)["workspace_path"])
    d = ws / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_ir(project_id: str, ir: dict) -> None:
    (_reports(project_id) / "ir.json").write_text(
        json.dumps(ir, ensure_ascii=False), encoding="utf-8")


def _write_verification(project_id: str, ir: dict, overall: str = "passed") -> None:
    (_reports(project_id) / "verification.json").write_text(json.dumps({
        "schema_version": "1.0",
        "overall": overall,
        "ir_hash": ir_hash(ir),
        "structure": {"passed": overall == "passed"},
        "numeric": {"passed": overall == "passed", "per_seed": []},
    }, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- 验证新鲜度

def test_verification_status_none_when_never_verified(ir_env):
    project_id = ir_env["project_id"]
    _write_ir(project_id, _ir(project_id))

    assert decompose_service.verification_status(project_id) == ("none", None)


def test_verification_status_valid_then_stale_after_param_edit_then_valid_again(ir_env):
    """调参必须让旧验证变 stale；重新验证后回到 valid（这是入库闸门的核心判据）。"""
    project_id = ir_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir)

    status, verification = decompose_service.verification_status(project_id)
    assert status == "valid" and verification["overall"] == "passed"

    # 调参：把 fc 的输出维度改掉 → ir_hash 变化
    decompose_service.update_node_params(
        project_id, "fc", {"in_features": 8, "out_features": 16})

    status, _ = decompose_service.verification_status(project_id)
    assert status == "stale"

    # 重新验证（用新 IR 的 hash 写一条）→ 回到 valid
    fresh_ir = decompose_service.read_ir(project_id)
    _write_verification(project_id, fresh_ir)
    status, _ = decompose_service.verification_status(project_id)
    assert status == "valid"


def test_update_input_spec_also_marks_stale(ir_env):
    """补 input_spec 与调参同口径：写回即 stale。"""
    project_id = ir_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir)
    assert decompose_service.verification_status(project_id)[0] == "valid"

    decompose_service.update_input_spec(project_id, [1, 16], "float32")

    assert decompose_service.verification_status(project_id)[0] == "stale"


# ---------------------------------------------------------------- 调参 / 补规格的守卫

def test_update_node_params_rejects_unknown_node(ir_env):
    project_id = ir_env["project_id"]
    _write_ir(project_id, _ir(project_id))

    with pytest.raises(LookupError, match="node not found"):
        decompose_service.update_node_params(project_id, "nope", {"a": 1})


def test_update_node_params_requires_ir(ir_env):
    project_id = ir_env["project_id"]

    with pytest.raises(LookupError, match="ir not found"):
        decompose_service.update_node_params(project_id, "fc", {"out_features": 4})


def test_ir_edits_reject_non_original_project(ir_env):
    """只有原始项目能调参：结构化项目应被 require_type 挡下（API 层映射 400）。"""
    structured_id = project_manager.create_project("structured", name="net")

    with pytest.raises(PermissionError):
        decompose_service.update_node_params(structured_id, "fc", {})


@pytest.mark.parametrize("bad", [[], None, [0, 8], [1, None], "1,8", [-1]])
def test_update_input_spec_rejects_invalid_shape(ir_env, bad):
    project_id = ir_env["project_id"]
    _write_ir(project_id, _ir(project_id))

    with pytest.raises(ValueError, match="非空正整数数组"):
        decompose_service.update_input_spec(project_id, bad)  # type: ignore[arg-type]


def test_update_input_spec_writes_spec_and_keeps_dtype(ir_env):
    project_id = ir_env["project_id"]
    _write_ir(project_id, _ir(project_id))

    spec = decompose_service.update_input_spec(project_id, [1, 32], "float16")

    assert spec == {"shape": [1, 32], "dtype": "float16", "user_edited": True}
    assert decompose_service.read_ir(project_id)["input_spec"]["shape"] == [1, 32]


# ---------------------------------------------------------------- 入库前置 / 拆解前置

def test_ingest_precheck_requires_ir(ir_env):
    with pytest.raises(LookupError, match="尚未拆解"):
        decompose_service.ingest_precheck(ir_env["project_id"])


def test_ingest_precheck_requires_verification(ir_env):
    project_id = ir_env["project_id"]
    _write_ir(project_id, _ir(project_id))

    with pytest.raises(LookupError, match="尚未验证"):
        decompose_service.ingest_precheck(project_id)


def test_ingest_precheck_rejects_failed_verification(ir_env):
    project_id = ir_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir, overall="failed")

    with pytest.raises(PermissionError, match="验证未通过"):
        decompose_service.ingest_precheck(project_id)


def test_ingest_precheck_rejects_stale_verification(ir_env):
    """过期验证必须挡住：这是「调参后忘了重验就入库」的防线。"""
    project_id = ir_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir)
    decompose_service.update_node_params(project_id, "fc", {"in_features": 8, "out_features": 16})

    with pytest.raises(ValueError, match="已过期"):
        decompose_service.ingest_precheck(project_id)


def test_ingest_precheck_returns_module_ref_when_healthy(ir_env):
    project_id = ir_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir)

    info = decompose_service.ingest_precheck(project_id)

    assert info["module_id"].startswith("mod_")
    assert info["module_version"] == "v1"
    assert info["existing_versions"] == []


def test_decompose_precheck_requires_structure_report(ir_env):
    project_id = ir_env["project_id"]

    with pytest.raises(FileNotFoundError, match="结构分析"):
        decompose_service.decompose_precheck(project_id)

    (_reports(project_id) / "structure_report.json").write_text("{}", encoding="utf-8")
    decompose_service.decompose_precheck(project_id)  # 存在即通过，不抛错


def test_regenerate_requires_ir(ir_env):
    with pytest.raises(LookupError, match="ir not found"):
        decompose_service.regenerate(ir_env["project_id"])


def test_regenerate_returns_code_for_healthy_ir(ir_env):
    """再生成是「画布导出/训练」的代码来源之一，此处只锁契约：IR 完整即出码且含根类。

    根类名由再生成引擎按 ir_codegen 口径生成（`Decomp_<root_id>`），不是原始类名。
    """
    project_id = ir_env["project_id"]
    _write_ir(project_id, _ir(project_id))

    code = decompose_service.regenerate(project_id)

    assert "class Decomp_net(nn.Module)" in code
    assert "self.fc = nn.Linear(in_features=8, out_features=8)" in code


def test_module_versions_isolated_from_real_library(ir_env):
    """守卫：本文件的用例只写临时 modules 目录，真实模块库不受影响。"""
    project_id = ir_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir)
    module_id = decompose_service.ingest_precheck(project_id)["module_id"]

    assert knowledge_service.list_module_versions(module_id) == []
