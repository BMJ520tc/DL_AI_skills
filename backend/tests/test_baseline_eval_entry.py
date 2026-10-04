"""缺口一：平台必须真正「加载原始模型权重」再评估（《模块详细设计》5.2）。

覆盖：
- eval 入口缺失/仍是模板 → agent 阅读项目代码后就地写出真实实现（此处 monkeypatch agent，
  写出的实现**真实加载 torch 权重**），随后用**项目独立环境解释器**执行并把指标落库；
- agent 无产出/无模型凭证 → 必须明确失败，不得回退规则桩、不得伪造指标；
- `model_dir` 为空（未定位到权重）→ 明确提示先跑通模块一/复现，不用规则桩顶替；
- fixture 规则桩（`"model": "fixture-rule"`）仅测试可用，真实运行一律拒绝。
"""
from __future__ import annotations

import asyncio
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.services import baseline_service, knowledge_service, task_manager

TEMPLATE_TEXT = (Path(__file__).resolve().parents[2] / "scripts" / "eval_entry_template.py").read_text(
    encoding="utf-8"
)

# 一份「真实加载权重」的 eval 实现：agent 生成结果的测试替身（缺口一 A6 的免凭证验证手段）
GENERATED_IMPL = '''"""agent 生成的 eval 入口：真实加载 model_dir 下的权重后推理。"""
import csv
import json
import sys
from pathlib import Path

import torch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def evaluate(data_dir: Path, model_dir: str | None, out_json: Path) -> dict:
    if not model_dir:
        raise SystemExit("缺少 model_dir，无法加载原始权重")
    model = Net()
    payload = torch.load(Path(model_dir) / "model.pth", map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()

    with (Path(data_dir) / "preprocessed.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    predictions = []
    correct = 0
    with torch.no_grad():
        for row in rows:
            y_true = int(row["label"])
            x = torch.zeros(4)
            x[y_true] = 1.0
            logits = model(x.unsqueeze(0))
            y_pred = int(torch.argmax(logits, dim=1).item())
            correct += int(y_pred == y_true)
            predictions.append({"id": row["id"], "y_true": y_true, "y_pred": y_pred,
                                "prob": 0.9, "path": row.get("input")})

    total = len(predictions) or 1
    return {
        "schema_version": "1.0",
        "task_type": "classification",
        "model": "linear-probe",
        "dataset": "self",
        "metrics": {"accuracy": round(correct / total, 4), "loss": 0.0},
        "primary_metric": "accuracy",
        "n_samples": len(predictions),
        "per_class": {},
        "predictions": predictions,
    }


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: python eval_entry.py <data_dir> <model_dir> <out_json>", file=sys.stderr)
        return 2
    result = evaluate(Path(sys.argv[1]), sys.argv[2] or None, Path(sys.argv[3]))
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

FIXTURE_STUB = '''"""fixture 规则桩（仅测试）：不加载任何权重，固定规则出预测。"""
import csv
import json
import sys
from pathlib import Path


def evaluate(data_dir, model_dir, out_json):
    with (data_dir / "preprocessed.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "schema_version": "1.0", "task_type": "classification", "model": "fixture-rule",
        "dataset": "self", "metrics": {"accuracy": 0.5, "loss": 0.5},
        "primary_metric": "accuracy", "n_samples": len(rows), "predictions": [],
    }


def main():
    result = evaluate(Path(sys.argv[1]), sys.argv[2] or None, Path(sys.argv[3]))
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _make_venv(ws: Path) -> Path:
    """项目独立环境（venv + 继承宿主 site-packages，使 torch 可用），不装任何包。"""
    env = ws / "env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(env)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    python = env / "Scripts" / "python.exe"
    assert python.exists()
    return python


def _make_workspace(tmp_path: Path, *, with_weights: bool = True, entry: str | None = TEMPLATE_TEXT) -> dict:
    ws = tmp_path / "proj"
    source = ws / "source"
    source.mkdir(parents=True)
    (ws / "data" / "self").mkdir(parents=True)
    with (ws / "data" / "self" / "preprocessed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "split", "label", "input"])
        for i, label in enumerate([0, 1, 2, 0]):
            writer.writerow([str(i), "train", str(label), f"x{i}"])

    if with_weights:
        import torch

        ckpt = source / "checkpoints"
        ckpt.mkdir(parents=True)
        model = torch.nn.Linear(4, 3)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
            for i in range(3):
                model.weight[i, i] = 2.0
        # 生成的入口用 Net.fc 包裹同一层，故 state_dict 键名带 fc. 前缀
        torch.save({"state_dict": {"fc.weight": model.weight, "fc.bias": model.bias}},
                   ckpt / "model.pth")

    if entry is not None:
        (source / "eval_entry.py").write_text(entry, encoding="utf-8")
    return {"ws": ws, "source": source, "data_dir": ws / "data" / "self",
            "python": _make_venv(ws)}


def _wire(monkeypatch, ws: Path, run_sync) -> None:
    monkeypatch.setattr(baseline_service.project_manager, "get_project",
                        lambda pid: {"project_id": pid, "workspace_path": str(ws), "project_type": "original"})
    monkeypatch.setattr(baseline_service.agent_service, "run_sync", run_sync)
    # 避免 proc_util 在真实 data/ 下建 numba 缓存（测试不碰真实数据目录）
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(ws / "numba_cache"))


def _agent_writes_impl(calls: list):
    async def _run_sync(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        (Path(kwargs["cwd"]) / "eval_entry.py").write_text(GENERATED_IMPL, encoding="utf-8")
        return {"structured_output": {"implemented": True, "uses_model_dir": True,
                                      "weights_source": "checkpoints/model.pth",
                                      "rationale": "torch.load state_dict 后前向"},
                "result": "done"}
    return _run_sync


# ---------- A1/A2/A4/A6：agent 就地写出真实实现 → 项目解释器执行 → 指标落库 ----------

def test_agent_generates_entry_that_really_loads_weights(isolated_db, tmp_path, monkeypatch):
    ws_info = _make_workspace(tmp_path)
    task_id = task_manager.create_task("baseline", project_id="p1", params={"project_id": "p1"})
    calls: list = []
    _wire(monkeypatch, ws_info["ws"], _agent_writes_impl(calls))

    run = asyncio.run(baseline_service.run_eval("p1", task_id, ws_info["data_dir"], run_type="baseline"))

    # 脚本已落盘且不再是模板
    entry = ws_info["source"] / "eval_entry.py"
    assert entry.read_text(encoding="utf-8") == GENERATED_IMPL
    assert "NotImplementedError" not in entry.read_text(encoding="utf-8")

    # agent 被要求阅读项目代码、并被告知权重位置
    assert calls and calls[0]["cwd"] == str(ws_info["source"])
    assert "structure_report.json" in calls[0]["prompt"]
    assert "checkpoints" in calls[0]["prompt"] and "model_dir" in calls[0]["prompt"]

    # 用项目独立环境解释器执行（命令里是 ws/env 下的 python）
    assert str(ws_info["python"]) in run["command"]
    assert run["metrics"]["accuracy"] == 1.0
    assert run["params"]["n_samples"] == 4

    # 指标落库（run_record 可查），并带上生成证据
    assert run["params"]["eval_entry_source"] == "agent_generated"
    assert run["params"]["model_dir"] == str(ws_info["source"] / "checkpoints")
    generation = run["params"]["eval_entry_generation"]
    assert generation["script_path"] == str(entry)
    assert generation["uses_model_dir"] is True
    assert Path(generation["evidence_path"]).exists()

    saved = json.loads(Path(run["artifact_path"]).read_text(encoding="utf-8"))
    assert saved["n_samples"] == 4 and len(saved["predictions"]) == 4

    runs = knowledge_service.list_runs("p1", "baseline", status="success")
    assert [r["run_id"] for r in runs] == [run["run_id"]]
    assert json.loads(runs[0]["metrics"])["accuracy"] == 1.0


def test_existing_real_entry_is_reused_without_agent(isolated_db, tmp_path, monkeypatch):
    ws_info = _make_workspace(tmp_path, entry=GENERATED_IMPL)
    task_id = task_manager.create_task("baseline", project_id="p1", params={"project_id": "p1"})

    async def _explode(*args, **kwargs):
        raise AssertionError("入口已是真实实现时不应再调用 agent")

    _wire(monkeypatch, ws_info["ws"], _explode)
    run = asyncio.run(baseline_service.run_eval("p1", task_id, ws_info["data_dir"], run_type="baseline"))
    assert run["params"]["eval_entry_source"] == "existing"
    assert run["params"]["eval_entry_generation"] is None


def test_cross_dataset_eval_artifacts_do_not_overwrite_each_other(isolated_db, tmp_path, monkeypatch):
    """同一 compare 任务里多个数据集各跑一次，指标文件必须分开（否则运行记录指向被覆盖的文件）。"""
    ws_info = _make_workspace(tmp_path, entry=GENERATED_IMPL)
    other = ws_info["ws"] / "data" / "public-x"
    other.mkdir(parents=True)
    import shutil

    shutil.copyfile(ws_info["data_dir"] / "preprocessed.csv", other / "preprocessed.csv")
    task_id = task_manager.create_task("compare", project_id="p1", params={"project_id": "p1"})

    async def _explode(*args, **kwargs):
        raise AssertionError("入口已是真实实现时不应再调用 agent")

    _wire(monkeypatch, ws_info["ws"], _explode)
    first = asyncio.run(baseline_service.run_eval(
        "p1", task_id, ws_info["data_dir"], run_type="eval", dataset_label="self",
    ))
    second = asyncio.run(baseline_service.run_eval(
        "p1", task_id, other, run_type="eval", dataset_label="public-x",
        extra_params={"alignment_used": True, "aligned_copy": str(other / "aligned" / "preprocessed.csv")},
    ))
    assert first["artifact_path"] != second["artifact_path"]
    assert Path(first["artifact_path"]).exists() and Path(second["artifact_path"]).exists()
    assert second["params"]["alignment_used"] is True


# ---------- A2：无模型凭证 → 明确失败，无伪造指标 ----------

def test_agent_without_credentials_fails_without_fake_metrics(isolated_db, tmp_path, monkeypatch):
    ws_info = _make_workspace(tmp_path)
    task_id = task_manager.create_task("baseline", project_id="p1", params={"project_id": "p1"})

    async def _no_credentials(prompt, **kwargs):
        return {"structured_output": None, "result": "Not logged in · Please run /login"}

    _wire(monkeypatch, ws_info["ws"], _no_credentials)

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(baseline_service.run_eval("p1", task_id, ws_info["data_dir"], run_type="baseline"))

    assert "eval 入口生成失败" in str(exc.value)
    assert "凭证" in str(exc.value)
    # 未写任何成功记录、无伪造指标
    assert knowledge_service.list_runs("p1", "baseline", status="success") == []
    failed = knowledge_service.list_runs("p1", "baseline", status="failed")
    assert len(failed) == 1 and failed[0]["metrics"] is None
    # 入口仍是模板，指标文件未产出
    assert "NotImplementedError" in (ws_info["source"] / "eval_entry.py").read_text(encoding="utf-8")
    assert not (ws_info["ws"] / "runs" / task_id / "baseline_metrics.json").exists()
    # 进度留下明确失败原因（不静默）
    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert progress["eval_entry_generation"]["status"] == "failed"


def test_agent_empty_structured_output_fails(isolated_db, tmp_path, monkeypatch):
    ws_info = _make_workspace(tmp_path)
    task_id = task_manager.create_task("baseline", project_id="p1", params={"project_id": "p1"})

    async def _empty(prompt, **kwargs):
        return {"structured_output": {}, "result": ""}

    _wire(monkeypatch, ws_info["ws"], _empty)
    with pytest.raises(RuntimeError, match="eval 入口生成失败"):
        asyncio.run(baseline_service.run_eval("p1", task_id, ws_info["data_dir"], run_type="baseline"))
    assert knowledge_service.list_runs("p1", "baseline", status="success") == []


def test_agent_call_exception_fails_explicitly(isolated_db, tmp_path, monkeypatch):
    """agent 调用本身抛异常（端点/CLI/网络）→ 也要明确失败并落 failed 记录。"""
    ws_info = _make_workspace(tmp_path)
    task_id = task_manager.create_task("baseline", project_id="p1", params={"project_id": "p1"})

    async def _boom(prompt, **kwargs):
        raise RuntimeError("claude CLI not found / endpoint unreachable")

    _wire(monkeypatch, ws_info["ws"], _boom)
    with pytest.raises(RuntimeError, match="agent 调用异常"):
        asyncio.run(baseline_service.run_eval("p1", task_id, ws_info["data_dir"], run_type="baseline"))
    assert knowledge_service.list_runs("p1", "baseline", status="success") == []
    failed = knowledge_service.list_runs("p1", "baseline", status="failed")
    assert len(failed) == 1 and failed[0]["metrics"] is None
    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert progress["eval_entry_generation"]["status"] == "failed"


def test_agent_claims_done_but_leaves_template_fails(isolated_db, tmp_path, monkeypatch):
    """agent 声称 implemented 但没改文件 → 拒绝把模板当真实实现。"""
    ws_info = _make_workspace(tmp_path)
    task_id = task_manager.create_task("baseline", project_id="p1", params={"project_id": "p1"})

    async def _claims_only(prompt, **kwargs):
        return {"structured_output": {"implemented": True, "uses_model_dir": True}, "result": "ok"}

    _wire(monkeypatch, ws_info["ws"], _claims_only)
    with pytest.raises(RuntimeError, match="仍不存在或仍是模板"):
        asyncio.run(baseline_service.run_eval("p1", task_id, ws_info["data_dir"], run_type="baseline"))
    assert knowledge_service.list_runs("p1", "baseline", status="success") == []


# ---------- A3：model_dir 为空 → 明确提示，不用规则桩 ----------

def test_missing_weights_prompts_module_one_and_never_runs_fixture(isolated_db, tmp_path, monkeypatch):
    ws_info = _make_workspace(tmp_path, with_weights=False, entry=FIXTURE_STUB)
    task_id = task_manager.create_task("baseline", project_id="p1", params={"project_id": "p1"})

    async def _explode(*args, **kwargs):
        raise AssertionError("权重缺失时不应调用 agent")

    _wire(monkeypatch, ws_info["ws"], _explode)
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(baseline_service.run_eval("p1", task_id, ws_info["data_dir"], run_type="baseline"))

    assert "未在项目中发现模型权重" in str(exc.value)
    assert "模块一" in str(exc.value)
    # 规则桩没有被执行：无成功记录、无指标文件
    assert knowledge_service.list_runs("p1", "baseline", status="success") == []
    assert not (ws_info["ws"] / "runs" / task_id / "baseline_metrics.json").exists()


# ---------- A5：fixture 仅测试可用，真实运行拒绝 ----------

def test_fixture_output_rejected_unless_explicitly_allowed(isolated_db, tmp_path, monkeypatch):
    ws_info = _make_workspace(tmp_path, entry=FIXTURE_STUB)
    task_id = task_manager.create_task("baseline", project_id="p1", params={"project_id": "p1"})

    async def _explode(*args, **kwargs):
        raise AssertionError("入口不是模板时不应调用 agent")

    _wire(monkeypatch, ws_info["ws"], _explode)

    with pytest.raises(RuntimeError, match="fixture"):
        asyncio.run(baseline_service.run_eval("p1", task_id, ws_info["data_dir"], run_type="baseline"))
    assert knowledge_service.list_runs("p1", "baseline", status="success") == []

    # 测试路径显式放行后才能跑（保留 fixture 仅供测试）
    run = asyncio.run(baseline_service.run_eval(
        "p1", task_id, ws_info["data_dir"], run_type="baseline", allow_fixture=True,
    ))
    assert run["metrics"]["accuracy"] == 0.5
    assert run["params"]["model"] == "fixture-rule"


def test_template_and_fixture_detection():
    assert baseline_service.is_template_entry(Path("no-such-file.py")) is True
    assert baseline_service.looks_like_fixture({"model": "fixture-rule"}) is True
    assert baseline_service.looks_like_fixture({"model": "resnet18"}) is False
