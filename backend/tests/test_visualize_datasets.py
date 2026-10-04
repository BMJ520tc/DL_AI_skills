"""5.5 可视化覆盖参与对比的全部数据集 + per_class 被消费（缺陷 B/C 的回归）。

背景（本次修复的缺陷）：
- 误差分布图与典型案例图原先只取 baseline，跨数据集评估结果根本没进图（需求三.2 要求
  「可视化呈现性能对比图、误差分布图、典型案例的预测结果」，跨数据集对比后应覆盖各数据集）；
- eval 入口契约里的 `per_class` 全仓库无人消费。

本用例用两份假评估结果（baseline + 一个跨数据集，其中只有 baseline 含 predictions/per_class）
跑真实的服务与固定脚本，断言：两个数据集的来源都出现在 HTML 里、缺数据的那个被如实标注为
不可绘、per_class 的内容落进产物、且 HTML 自包含（外部 http(s) 资源引用为 0）。
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from app.services import knowledge_service, visualize_service

# 只认「外部资源引用」：src/href 指向 http(s)、CSS url(http...)、@import http...
# 内联的 ECharts 源码里有 Apache 许可证链接与 SVG 命名空间常量，那些不是外部引用。
EXTERNAL_REF = re.compile(
    r"""(?:src|href)\s*=\s*["']https?://|url\(\s*["']?https?://|@import\s+["']?https?://""",
    re.IGNORECASE,
)


def _external_refs(html: str) -> list[str]:
    return EXTERNAL_REF.findall(html)


BASELINE_PAYLOAD = {
    "schema_version": "1.0",
    "task_type": "classification",
    "model": "resnet18",
    "dataset": "self",
    "metrics": {"accuracy": 0.5, "f1": 0.5},
    "primary_metric": "accuracy",
    "n_samples": 4,
    "per_class": {"cat": {"f1": 0.875, "recall": 0.8}, "dog": {"f1": 0.125, "recall": 0.0}},
    "predictions": [
        {"id": "1", "y_true": "cat", "y_pred": "cat", "prob": 0.95, "path": "img/1.png"},
        {"id": "2", "y_true": "cat", "y_pred": "dog", "prob": 0.55, "path": "img/2.png"},
        {"id": "3", "y_true": "dog", "y_pred": "cat", "prob": 0.40, "path": "img/3.png"},
        {"id": "4", "y_true": "dog", "y_pred": "dog", "prob": 0.99, "path": "img/4.png"},
    ],
}

# 老 eval 实现：只有聚合指标，没有 predictions / per_class
CROSS_PAYLOAD = {
    "schema_version": "1.0",
    "task_type": "classification",
    "model": "resnet18",
    "dataset": "public-pub",
    "metrics": {"accuracy": 0.25, "f1": 0.2},
    "primary_metric": "accuracy",
    "n_samples": 4,
}


def _record(project_id: str, run_type: str, label: str, artifact: Path, metrics: dict, started: str) -> None:
    knowledge_service.record_run({
        "project_id": project_id,
        "task_id": "task-1",
        "run_type": run_type,
        "params": {"dataset_label": label, "data_dir": str(artifact.parent)},
        "status": "success",
        "metrics": metrics,
        "artifact_path": str(artifact),
        "started_at": started,
        "finished_at": started,
    })


def _wire(monkeypatch, ws: Path) -> None:
    monkeypatch.setattr(visualize_service.project_manager, "require_type",
                        lambda pid, types: {"project_id": pid})
    monkeypatch.setattr(visualize_service.project_manager, "get_project",
                        lambda pid: {"project_id": pid, "workspace_path": str(ws),
                                     "project_type": "original"})
    # 别让 proc_util 在真实 data/ 下建 numba 缓存（测试不碰真实数据目录）
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(ws / "numba_cache"))


def _setup(isolated_db, tmp_path, monkeypatch, cross_payload=CROSS_PAYLOAD):
    ws = tmp_path / "proj"
    (ws / "reports").mkdir(parents=True)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "baseline.json").write_text(json.dumps(BASELINE_PAYLOAD, ensure_ascii=False), encoding="utf-8")
    (runs / "cross.json").write_text(json.dumps(cross_payload, ensure_ascii=False), encoding="utf-8")
    _record("p1", "baseline", "baseline", runs / "baseline.json",
            BASELINE_PAYLOAD["metrics"], "2026-10-01T00:00:00")
    _record("p1", "eval", "public-pub", runs / "cross.json",
            cross_payload["metrics"], "2026-10-01T00:01:00")
    _wire(monkeypatch, ws)
    return ws


def test_error_dist_covers_all_datasets_and_flags_undrawable(isolated_db, tmp_path, monkeypatch):
    _setup(isolated_db, tmp_path, monkeypatch)

    result = asyncio.run(visualize_service.run("p1", "error_dist"))

    assert result["degraded"] is False
    html = Path(result["html"]).read_text(encoding="utf-8")
    # 两个数据集的来源标识都在图里
    assert "baseline" in html and "public-pub" in html
    # 缺 predictions 的数据集被如实标注为不可绘（不留白、不用代理顶替）
    assert "不可绘" in html
    assert "缺少 predictions" in html
    # baseline 是分类任务 → 误差只能用代理，必须明确标注
    assert "1 - 预测置信度" in html and "代理" in html
    # per_class 被消费：按类别分组、指标值落进产物
    assert "按类别分组" in html
    assert "cat" in html and "dog" in html
    assert "0.875" in html and "0.125" in html
    # 图确实画了（echarts 内联 + 子图容器）
    assert 'id="chart0"' in html and len(html) > 500_000
    # 自包含：外部 http(s) 资源引用为 0
    assert _external_refs(html) == []


def test_cases_covers_all_datasets_and_flags_undrawable(isolated_db, tmp_path, monkeypatch):
    _setup(isolated_db, tmp_path, monkeypatch)

    result = asyncio.run(visualize_service.run("p1", "cases"))

    assert result["degraded"] is False
    html = Path(result["html"]).read_text(encoding="utf-8")
    assert "baseline" in html and "public-pub" in html
    assert "不可绘" in html and "缺少 predictions" in html
    # per_class 被消费：按类别的指标明细落进产物
    assert "per_class" in html
    assert "0.875" in html and "0.125" in html
    # 案例表带数据集列，行内也能区分来源
    assert "<th>数据集</th>" in html
    assert _external_refs(html) == []


def test_performance_chart_lists_all_datasets(isolated_db, tmp_path, monkeypatch):
    _setup(isolated_db, tmp_path, monkeypatch)

    result = asyncio.run(visualize_service.run("p1", "performance"))

    assert result["degraded"] is False
    html = Path(result["html"]).read_text(encoding="utf-8")
    assert "baseline" in html and "public-pub" in html
    assert _external_refs(html) == []


def test_all_datasets_undrawable_degrades_without_fake_values(isolated_db, tmp_path, monkeypatch):
    """两份结果都只有聚合指标时，两图整体降级并列出原因，而不是拿 1-prob/空值顶替。"""
    _setup(isolated_db, tmp_path, monkeypatch, cross_payload=CROSS_PAYLOAD)

    for chart_type in ("error_dist", "cases"):
        result = asyncio.run(visualize_service.run("p1", chart_type))
        html = Path(result["html"]).read_text(encoding="utf-8")
        assert result["degraded"] is False  # baseline 仍有 predictions，可绘
        assert "baseline" in html

    # 把 baseline 也换成「只有聚合指标」，则两图都整体降级
    (tmp_path / "runs" / "baseline.json").write_text(
        json.dumps({k: v for k, v in BASELINE_PAYLOAD.items() if k not in ("predictions", "per_class")},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    for chart_type in ("error_dist", "cases"):
        result = asyncio.run(visualize_service.run("p1", chart_type))
        html = Path(result["html"]).read_text(encoding="utf-8")
        assert result["degraded"] is True
        assert "降级" in html
        assert "baseline" in html and "public-pub" in html
        assert _external_refs(html) == []
