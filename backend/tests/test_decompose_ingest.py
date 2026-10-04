"""模块四 6.5「标准化模块入库」失败留痕回归（run_record）。

已知缺陷：`decompose_service._run_ingest` 只在成功路径写 run_record，
失败路径（IR 缺失/不完整、验证缺失/未过/过期、写盘失败）直接 raise，
而同章 decompose / trace / verify 的失败都会留痕 → 同一条链路口径不一致，
失败无法检索、agent 也没有改进依据。

修法：入库链路每一处 raise 都「先记一条 run_type=module_ingest、status=failed 的
run_record（error=失败原因），再原样抛出」，成功路径字段不变。本用例覆盖：
IR 缺失、验证缺失、验证未过、验证过期、IR 不完整（再生成拒绝）、包写盘失败
6 个失败分支 + 成功路径仍写 success。

只使用临时库（conftest.isolated_db）与临时工作区/模块目录（monkeypatch
project_manager.PROJECTS_DIR / decompose_service.MODULES_DIR），不碰真实 data/。
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from app.services import decompose_service, knowledge_service, project_manager
from app.services.ir_codegen import IrIncompleteError
from app.services.ir_schema import ir_hash

RUN_TYPE = "module_ingest"
TASK_ID = "t-ingest"


@pytest.fixture()
def ingest_env(isolated_db, tmp_path, monkeypatch):
    """原始项目 + 临时工作区/模块目录（绝不触碰 data/projects、data/modules）。"""
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    monkeypatch.setattr(decompose_service, "MODULES_DIR", modules_dir)
    project_id = project_manager.create_project("original", source="local-test")
    return {"project_id": project_id, "modules_dir": modules_dir}


def _ir(project_id: str, fc_params: dict | None = None) -> dict:
    """最小可再生成 IR：Net(root) → fc(nn.Linear)。"""
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
             "params": {"in_features": 8, "out_features": 8} if fc_params is None else fc_params,
             "parent_id": "net", "input_shape": [1, 8], "output_shape": [1, 8]},
        ],
        "edges": [],
    }


def _reports_dir(project_id: str) -> Path:
    ws = Path(project_manager.get_project(project_id)["workspace_path"])
    reports = ws / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports


def _write_ir(project_id: str, ir: dict) -> None:
    (_reports_dir(project_id) / "ir.json").write_text(
        json.dumps(ir, ensure_ascii=False), encoding="utf-8")


def _write_verification(project_id: str, ir: dict, overall: str = "passed",
                        ir_hash_value: str | None = None) -> None:
    passed = overall == "passed"
    verification = {
        "schema_version": "1.0",
        "overall": overall,
        "ir_hash": ir_hash(ir) if ir_hash_value is None else ir_hash_value,
        "structure": {"passed": passed, "param_count": {"original": 72, "regenerated": 72}},
        "numeric": {"passed": passed, "per_seed": []},
    }
    (_reports_dir(project_id) / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False), encoding="utf-8")


def _run_ingest(project_id: str, task_id: str = TASK_ID) -> None:
    asyncio.run(decompose_service._run_ingest({"project_id": project_id}, task_id))


def _failed_runs(project_id: str) -> list[dict]:
    return knowledge_service.list_runs(project_id, RUN_TYPE, status="failed")


def _assert_single_failure(project_id: str, needle: str) -> dict:
    """唯一一条 failed 记录，字段齐全且 error 含关键原因；同时无成功记录。"""
    runs = _failed_runs(project_id)
    assert len(runs) == 1, [r["error"] for r in runs]
    run = runs[0]
    assert run["run_type"] == RUN_TYPE
    assert run["status"] == "failed"
    assert run["project_id"] == project_id
    assert run["task_id"] == TASK_ID
    assert needle in run["error"]
    assert run["started_at"] and run["finished_at"]
    assert knowledge_service.list_runs(project_id, RUN_TYPE) == []
    return run


def test_ingest_without_ir_records_failed_run(ingest_env):
    """IR 不存在：抛错语义不变，且落一条 failed run_record。"""
    project_id = ingest_env["project_id"]

    with pytest.raises(RuntimeError, match="尚未拆解"):
        _run_ingest(project_id)

    _assert_single_failure(project_id, "尚未拆解")


def test_ingest_without_verification_records_failed_run(ingest_env):
    """IR 在但未验证：同样留痕。"""
    project_id = ingest_env["project_id"]
    _write_ir(project_id, _ir(project_id))

    with pytest.raises(RuntimeError, match="尚未验证"):
        _run_ingest(project_id)

    _assert_single_failure(project_id, "尚未验证")


def test_ingest_unpassed_verification_records_failed_run(ingest_env):
    """验证未通过（overall=failed）：留痕并带上原因。"""
    project_id = ingest_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir, overall="failed")

    with pytest.raises(RuntimeError, match="验证未通过"):
        _run_ingest(project_id)

    _assert_single_failure(project_id, "验证未通过")


def test_ingest_stale_verification_records_failed_run(ingest_env):
    """验证结果过期（ir_hash 不一致）：留痕并带上原因。"""
    project_id = ingest_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir, ir_hash_value="stale-ir-hash")

    with pytest.raises(RuntimeError, match="验证结果已过期"):
        _run_ingest(project_id)

    _assert_single_failure(project_id, "验证结果已过期")


def test_ingest_incomplete_ir_records_failed_run(ingest_env):
    """IR 不完整（nn.Linear 缺 out_features）→ 再生成拒绝：异常类型/消息不变，但先留痕。"""
    project_id = ingest_env["project_id"]
    ir = _ir(project_id, fc_params={"in_features": 8})
    _write_ir(project_id, ir)
    _write_verification(project_id, ir)

    with pytest.raises(IrIncompleteError) as excinfo:
        _run_ingest(project_id)

    assert "out_features" in str(excinfo.value)  # 调用方仍收到 IrIncompleteError 与缺项清单
    run = _assert_single_failure(project_id, "无法再生成代码")
    assert "out_features" in run["error"]


def test_ingest_package_failure_records_failed_run_and_compensates(ingest_env, monkeypatch):
    """包写盘/入库失败（版本号连续冲突）→ 留痕，且半成品结构化项目被补偿清掉。"""
    project_id = ingest_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir)

    def _boom(module_json: dict) -> dict:
        raise sqlite3.IntegrityError("UNIQUE constraint failed: module.module_id, module.module_version")

    monkeypatch.setattr(decompose_service.knowledge_service, "record_module", _boom)

    with pytest.raises(RuntimeError, match="版本号连续冲突"):
        _run_ingest(project_id)

    _assert_single_failure(project_id, "版本号连续冲突")
    assert project_manager.list_projects("structured") == []  # 半成品结构化项目不残留
    assert list(ingest_env["modules_dir"].iterdir()) == []     # 临时包目录不残留


def test_ingest_success_records_success_run(ingest_env):
    """成功路径行为与字段不变：写 success 记录 + 模块包落盘 + 无 failed 记录。"""
    project_id = ingest_env["project_id"]
    ir = _ir(project_id)
    _write_ir(project_id, ir)
    _write_verification(project_id, ir)

    _run_ingest(project_id)

    runs = knowledge_service.list_runs(project_id, RUN_TYPE)
    assert len(runs) == 1
    run = runs[0]
    assert run["status"] == "success"
    assert run["run_type"] == RUN_TYPE
    assert run["task_id"] == TASK_ID
    assert run["duration_s"] is not None
    metrics = json.loads(run["metrics"])
    assert metrics["module_version"] == "v1"
    assert metrics["structured_project_id"]

    package_dir = ingest_env["modules_dir"] / metrics["module_id"] / "v1"
    assert (package_dir / "module.py").exists()
    assert (package_dir / "module.json").exists()
    assert run["artifact_path"] == str(package_dir / "module.json")

    assert len(project_manager.list_projects("structured")) == 1
    assert _failed_runs(project_id) == []
