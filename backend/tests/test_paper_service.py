"""模块二论文服务回归：保真核对、抽取取样、指标口径、索引重建与论文四表索引。

背景（需求二.1 复核发现的缺陷）：① _fix_markdown 的 prompt 不引用 paper.pdf、无对照产物；
② 长论文抽取只取 markdown 头部，结果表（报告值来源）被整段切掉；③ 指标名未按
CANONICAL_METRICS 归一、复现输出契约的 metric_name/n_samples/log_tail 被丢弃；
④ 论文三张子表未进 unified_index；⑤ 索引重建后 page/n_pages 全空。

本文件只用临时库（isolated_db）与临时目录；大模型、PDF 解析脚本、子进程一律 monkeypatch。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.services import knowledge_service as ks
from app.services import paper_service, task_manager

# --------------------------- 4.4 分档：四档 verdict ---------------------------


def test_classify_four_verdicts():
    """一致/近似/不一致/无法复现四档语义保持（阈值 5% / 20% 可配置）。"""
    deviation, passed, verdict = paper_service._classify(0.912, 0.915)
    assert verdict == paper_service.VERDICT_CONSISTENT and passed == 1
    assert deviation is not None and deviation <= paper_service.DEVIATION_CONSISTENT

    deviation, passed, verdict = paper_service._classify(0.80, 0.90)
    assert verdict == paper_service.VERDICT_APPROX and passed == 0
    assert paper_service.DEVIATION_CONSISTENT < deviation <= paper_service.DEVIATION_APPROX

    deviation, passed, verdict = paper_service._classify(0.50, 0.90)
    assert verdict == paper_service.VERDICT_INCONSISTENT and passed == 0
    assert deviation > paper_service.DEVIATION_APPROX

    assert paper_service._classify(None, 0.90)[2] == paper_service.VERDICT_UNREPRODUCIBLE


def test_classify_non_numeric_reported_is_unreproducible():
    """报告值非数值 / 为 0：无从比较 → 无法复现，不算「不一致」。"""
    for reported in ("未报告", ">50% improvement", "85-90", "", None, 0, "0"):
        deviation, passed, verdict = paper_service._classify(0.91, reported)
        assert verdict == paper_service.VERDICT_UNREPRODUCIBLE, reported
        assert deviation is None and passed == 0


def test_classify_percent_and_fraction_interchange():
    """百分比与小数互换：带单位或值里自带 % 都能判一致，不被算成 99% 偏差。"""
    assert paper_service._classify(0.915, 91.5, "%")[2] == paper_service.VERDICT_CONSISTENT
    assert paper_service._classify(91.5, 0.915, "百分比")[2] == paper_service.VERDICT_CONSISTENT
    # 单位缺失但报告值自带 % 时按方向判定
    assert paper_service._classify(0.915, "91.5%")[2] == paper_service.VERDICT_CONSISTENT
    assert paper_service._classify("0.915", "91.5 %")[2] == paper_service.VERDICT_CONSISTENT


def test_classify_does_not_guess_ambiguous_scale():
    """无任何单位线索、只有 ×100 才能吻合 → 方向不明，如实判「无法复现」而不是猜。"""
    deviation, passed, verdict = paper_service._classify(91.5, 0.915)
    assert verdict == paper_service.VERDICT_UNREPRODUCIBLE
    assert deviation is None and passed == 0


# --------------------------- _align_unit 量纲归一 ---------------------------


def test_align_unit_percent_ratio_and_per_myriad():
    assert paper_service._align_unit(0.915, 91.5, "%") == pytest.approx((91.5, 91.5))
    assert paper_service._align_unit(91.5, 0.915, "ratio") == pytest.approx((0.915, 0.915))
    assert paper_service._align_unit(0.000915, 9.15, "万分比") == pytest.approx((9.15, 9.15))


def test_align_unit_time_and_length_families():
    assert paper_service._align_unit(3_600_000.0, 3600.0, "s") == pytest.approx((3600.0, 3600.0))
    assert paper_service._align_unit(1.5, 90.0, "s") == pytest.approx((90.0, 90.0))
    assert paper_service._align_unit(50.0, 0.5, "m") == pytest.approx((0.5, 0.5))
    assert paper_service._align_unit(2500.0, 2.5, "km") == pytest.approx((2.5, 2.5))


def test_align_unit_leaves_unresolvable_dimension_untouched():
    """换算后仍不吻合（或不认识的量纲）原样返回，不硬猜单位。"""
    assert paper_service._align_unit(1.0, 1_000_000_000.0, "s") == (1.0, 1_000_000_000.0)
    assert paper_service._align_unit(1.0, 2.0, "dB") == (1.0, 2.0)
    # 方向不明时不猜：保持原值，由 _classify 判「无法复现」
    assert paper_service._align_unit(91.5, 0.915, None) == (91.5, 0.915)


# --------------------------- 4.2 抽取取样：章节 + 结果表 ---------------------------


def _build_markdown_and_index(scale: int = 1) -> tuple[str, dict]:
    """构造带章节索引的长 markdown（结果表在文末，模拟真实论文排布）。"""
    lines = [
        "# 1 Introduction",
        *["intro filler line"] * (20 * scale),
        "",
        "## 2 Related Work",
        *["related work filler"] * (20 * scale),
        "",
        "## 3 Method",
        *["method detail line"] * (200 * scale),
        "",
        "## 4 Implementation",
        *["implementation detail"] * (20 * scale),
        "",
        "## 5 Experiments",
        *["experiment setup line"] * (100 * scale),
        "",
        "## 6 Results",
        *["result discussion line"] * (50 * scale),
        "",
        "| Model | Accuracy |",
        "| --- | --- |",
        "| Ours | 91.5 |",
        "| Baseline | 88.0 |",
    ]
    markdown = "\n".join(lines)
    index: dict = {"schema_version": "1.0", "n_pages": 3, "quality": "text",
                   "sections": [], "tables": [], "equations": [], "captions": []}
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip("#").strip()
        if line.startswith("#"):
            index["sections"].append({
                "level": len(line) - len(line.lstrip("#")), "title": stripped, "page": 1, "line": i,
            })
    return markdown, index


def test_extract_sampling_keeps_late_result_table_within_full_budget():
    """长 markdown：结果表在文末也必须进取样（不再头部截断），并说明取样范围。"""
    markdown, index = _build_markdown_and_index(scale=16)
    assert len(markdown) > 100_000  # 与真实超限论文（131k~289k）同量级

    body, meta = paper_service.select_extract_context(
        markdown, index, tables_text="[第9页 表1]\nOurs | 91.5\nBaseline | 88.0"
    )

    assert "[第9页 表1]" in body and "Ours | 91.5" in body  # tables.json 网格片段在内
    assert "| Model | Accuracy |" in body and "| Ours | 91.5 |" in body  # markdown 结果表在内
    assert "method detail line" in body and "experiment setup line" in body
    assert not body.startswith("# 1 Introduction")  # 不是头部截断
    assert "3 Method" in meta["sections_included"] and "6 Results" in meta["sections_included"]
    assert meta["table_fragments_included"] >= 2
    assert len(body) <= paper_service.MAX_EXTRACT_CHARS


def test_extract_sampling_reports_omissions_under_small_budget():
    """预算不足时：结果表仍优先保留，被截断/省略的部分如实写明。"""
    markdown, index = _build_markdown_and_index(scale=16)
    body, meta = paper_service.select_extract_context(markdown, index, tables_text="Ours | 91.5", budget=3000)

    assert len(body) <= 3000
    assert "Ours | 91.5" in body
    assert "method detail line" in body
    notes = "；".join(meta["omitted_notes"])
    assert "截断" in notes
    assert meta["truncated"] is True
    assert meta["full_chars"] > meta["used_chars"]
    assert meta["used_chars"] <= 3000


def test_extract_sampling_falls_back_without_section_index():
    """section_index 缺失时按空行分段兜底，不编造标题也不丢内容。"""
    markdown = "\n\n".join(f"paragraph {i} " + "content " * 50 for i in range(5))
    body, meta = paper_service.select_extract_context(markdown, {})
    assert "paragraph 4" in body
    assert meta["sections_included"] == [] and meta["abstract_included"] == []


# --------------------------- 4.1 保真核对 ---------------------------


def _make_pdf(path: Path, page_texts: list[str]) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 720), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def test_fidelity_report_fails_honestly_without_pdf(tmp_path):
    """无 PDF / PDF 不存在：ok=false 且写明原因，不伪造成通过。"""
    md = tmp_path / "paper.md"
    md.write_text("# 1 Introduction\n\nsome text\n", encoding="utf-8")
    index = {"sections": [{"level": 1, "title": "1 Introduction", "page": 1, "line": 1}],
             "tables": [], "equations": [], "captions": []}

    report = paper_service._check_fidelity("p1", None, md, index, tmp_path / "tables.json")
    assert report["ok"] is False and "pdf_path" in report["error"]
    assert report["warnings"] and "未通过" in report["warnings"][0]

    report = paper_service._check_fidelity("p1", str(tmp_path / "missing.pdf"), md, index)
    assert report["ok"] is False and "不存在" in report["error"]

    saved = json.loads((tmp_path / "fidelity_report.json").read_text(encoding="utf-8"))
    assert saved["ok"] is False and saved["paper_id"] == "p1"


def test_fidelity_report_coverage_and_gaps_with_real_pdf(tmp_path, monkeypatch):
    """有 PDF：给出页数/章节/表格/图注计数、按页覆盖率与明显缺口，并明确告警。"""
    monkeypatch.setattr(paper_service, "FIDELITY_PAGE_MIN_CHARS", 10)
    monkeypatch.setattr(paper_service, "FIDELITY_MIN_COVERAGE", 0.9)
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf, [
        "Title\nAbstract\n" + "\n".join(f"intro line {i}" for i in range(20)),
        "1 Method\n" + "\n".join(f"method line {i}" for i in range(20)) + "\nTable 1: results",
    ])
    md_lines = ["# Title", "Abstract", *[f"intro line {i}" for i in range(20)], "", "## 1 Method"]
    md = tmp_path / "paper.md"
    md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    index = {"n_pages": 2, "quality": "text",
             "sections": [{"level": 1, "title": "Title", "page": 1, "line": 1},
                          {"level": 2, "title": "1 Method", "page": 2,
                           "line": md_lines.index("## 1 Method") + 1}],
             "tables": [], "equations": [], "captions": []}

    report = paper_service._check_fidelity("p1", str(pdf), md, index, tmp_path / "tables.json")

    assert report["ok"] is True and report["error"] is None
    assert report["pdf"]["pages"] == 2 and report["pdf"]["chars"] > 0
    assert report["markdown"]["sections"] == 2
    assert report["coverage"]["available"] is True
    assert report["coverage"]["mean_ratio"] is not None
    assert 2 in report["coverage"]["pages_below_threshold"]
    assert any("Table" in gap for gap in report["gaps"])
    assert any("覆盖率均值" in w for w in report["warnings"])
    assert Path(report["report_path"]).exists()


def test_fidelity_report_marks_missing_page_anchors_as_unavailable(tmp_path, monkeypatch):
    """索引无分页信息（--index-only 重建后 page 为空）：如实记为「无法按页比对」。"""
    monkeypatch.setattr(paper_service, "FIDELITY_PAGE_MIN_CHARS", 10)
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf, ["1 Introduction\n" + "\n".join(f"line {i}" for i in range(20))])
    md = tmp_path / "paper.md"
    md.write_text("# 1 Introduction\nsome text\n", encoding="utf-8")
    index = {"n_pages": None, "quality": "text",
             "sections": [{"level": 1, "title": "1 Introduction", "page": None, "line": 1}],
             "tables": [], "equations": [], "captions": []}

    report = paper_service._check_fidelity("p1", str(pdf), md, index)
    assert report["coverage"]["available"] is False and "无分页信息" in report["coverage"]["reason"]
    assert any("无法按页比对" in w for w in report["warnings"])


# --------------------------- 4.1 索引重建条件（只有真改了才算） ---------------------------

_MD_TEXT = "# 1 Introduction\n\nA short paper body.\n"
_INDEX_DOC = {
    "schema_version": "1.0", "engine": "pymupdf4llm", "n_pages": 2, "quality": "text",
    "sections": [{"level": 1, "title": "1 Introduction", "page": 1, "line": 1}],
    "tables": [], "equations": [], "captions": [],
}


def _fake_fidelity(paper_id, pdf_path, md_path, index, tables_path=None):
    return {"ok": True, "report_path": None, "error": None,
            "coverage": {"mean_ratio": 1.0, "pages_below_threshold": []},
            "gaps": [], "warnings": []}


def _fake_script(calls: list, index_doc: dict, rebuilt_doc: dict | None = None):
    async def _run(cmd, **kwargs):
        calls.append([str(c) for c in cmd])
        if "--index-only" in cmd:
            Path(cmd[4]).write_text(json.dumps(rebuilt_doc or index_doc, ensure_ascii=False), encoding="utf-8")
            return 0, ""
        Path(cmd[3]).write_text(_MD_TEXT, encoding="utf-8")
        Path(cmd[4]).write_text(json.dumps(index_doc, ensure_ascii=False), encoding="utf-8")
        Path(cmd[5]).write_text(json.dumps({"tables": []}), encoding="utf-8")
        return 0, ""
    return _run


def _setup_parse(monkeypatch, tmp_path) -> str:
    monkeypatch.setattr(paper_service, "PAPERS_DIR", tmp_path / "papers")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 placeholder")
    ks.record_paper({"paper_id": "p-parse", "title": "T", "pdf_path": str(pdf)})
    monkeypatch.setattr(paper_service, "_check_fidelity", _fake_fidelity)
    return task_manager.create_task("pdf_parse", params={"paper_id": "p-parse"})


def test_parse_keeps_index_when_agent_makes_no_change(isolated_db, tmp_path, monkeypatch):
    """agent 未改动 markdown → 保留原索引（含页码），不调 --index-only。"""
    task_id = _setup_parse(monkeypatch, tmp_path)
    calls: list = []
    monkeypatch.setattr(paper_service.proc_util, "run_command", _fake_script(calls, _INDEX_DOC))
    seen: dict = {}

    async def _no_change(md_path, index, pdf_path=None):
        seen["pdf_path"] = pdf_path

    monkeypatch.setattr(paper_service, "_fix_markdown", _no_change)

    asyncio.run(paper_service._run_parse({"paper_id": "p-parse"}, task_id))

    assert not any("--index-only" in call for call in calls)
    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert progress["index_rebuilt"] is False and progress["markdown_changed"] is False
    assert progress["agent_fix"] == "applied"
    assert progress["fidelity"]["ok"] is True
    stored = json.loads(ks.get_item("paper", "p-parse")["section_index"])
    assert stored["sections"][0]["page"] == 1  # 原索引的分页信息被保留
    assert seen["pdf_path"].endswith("paper.pdf")  # 对照用的 PDF 传给了 agent 步骤


def test_parse_rebuilds_index_only_when_markdown_changed(isolated_db, tmp_path, monkeypatch):
    """agent 真的改了 markdown → 调一次 --index-only 重建并采用新索引。"""
    task_id = _setup_parse(monkeypatch, tmp_path)
    calls: list = []
    rebuilt = {**_INDEX_DOC,
               "sections": _INDEX_DOC["sections"] + [{"level": 2, "title": "2 Method", "page": 2, "line": 5}]}
    monkeypatch.setattr(paper_service.proc_util, "run_command", _fake_script(calls, _INDEX_DOC, rebuilt))

    async def _change(md_path, index, pdf_path=None):
        md_path.write_text(md_path.read_text(encoding="utf-8") + "\n## 2 Method\n\nfixed table\n",
                           encoding="utf-8")

    monkeypatch.setattr(paper_service, "_fix_markdown", _change)

    asyncio.run(paper_service._run_parse({"paper_id": "p-parse"}, task_id))

    assert sum(1 for call in calls if "--index-only" in call) == 1
    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert progress["index_rebuilt"] is True and progress["markdown_changed"] is True
    assert progress["sections"] == 2


def test_parse_survives_agent_failure_and_keeps_index(isolated_db, tmp_path, monkeypatch):
    """agent 修正失败：容错为 agent_fix=failed，markdown 未改则仍保留原索引。"""
    task_id = _setup_parse(monkeypatch, tmp_path)
    calls: list = []
    monkeypatch.setattr(paper_service.proc_util, "run_command", _fake_script(calls, _INDEX_DOC))

    async def _boom(md_path, index, pdf_path=None):
        raise RuntimeError("agent down")

    monkeypatch.setattr(paper_service, "_fix_markdown", _boom)

    asyncio.run(paper_service._run_parse({"paper_id": "p-parse"}, task_id))

    progress = json.loads(task_manager.get_task(task_id)["progress"])
    assert progress["agent_fix"].startswith("failed")
    assert progress["index_rebuilt"] is False
    assert not any("--index-only" in call for call in calls)


# --------------------------- 论文子表进统一索引 ---------------------------


def test_experiment_item_indexed_and_searchable(isolated_db):
    """experiment_item 写入后可按主键取用，并能在统一检索里命中。"""
    ks.record_paper({"paper_id": "p-idx", "title": "T"})
    item_ids = ks.record_experiment_items("p-idx", [
        {"metric_name": "accuracy", "metric_value_reported": "91.5", "dataset_name": "cifar10"},
    ])

    assert ks.get_item("experiment_item", item_ids[0])["metric_name"] == "accuracy"
    hits = ks.search(types=["experiment_item"], q="accuracy")
    assert [h["ref_id"] for h in hits] == item_ids
    assert [h["ref_id"] for h in ks.search(types=["experiment_item"], dataset="cifar10")] == item_ids
    assert ks.delete_item("experiment_item", item_ids[0]) is True
    assert ks.search(types=["experiment_item"]) == []


def test_subtable_dimensions_not_invented_when_absent(isolated_db):
    """缺上下文时不编造维度（对齐 test_dimensions_not_invented_when_absent 口径）。"""
    ks.record_paper({"paper_id": "p-dim", "title": "T"})
    ks.record_experiment_items("p-dim", [{"metric_name": "rmse", "metric_value_reported": "1.2"}])

    row = ks.search(types=["experiment_item"])[0]
    assert row["task_type"] is None and row["model_name"] is None and row["dataset_name"] is None
    assert ks.search(model="resnet50") == [] and ks.search(task_type="classification") == []


def test_reproduction_result_and_conclusion_indexed(isolated_db):
    """reproduction_result / credibility_conclusion 进索引，判定更新后摘要同步。"""
    ks.record_paper({"paper_id": "p-rep", "title": "T"})
    (item_id,) = ks.record_experiment_items(
        "p-rep", [{"metric_name": "accuracy", "metric_value_reported": "0.915", "dataset_name": "cifar10"}]
    )
    result_id = ks.record_reproduction_result(
        {"item_id": item_id, "run_id": "run-1", "metric_value_actual": 0.91}
    )
    assert [h["ref_id"] for h in ks.search(types=["reproduction_result"], q="accuracy")] == [result_id]

    assert ks.update_reproduction_result(result_id, deviation=0.005, passed_threshold=1, verdict="一致") is True
    row = ks.search(types=["reproduction_result"])[0]
    assert row["ref_id"] == result_id and "一致" in (row["summary"] or "")
    assert ks.get_item("reproduction_result", result_id)["verdict"] == "一致"
    # 维度取自条目上下文（数据集来自 experiment_item，而不是编造）
    assert row["dataset_name"] == "cifar10"

    conclusion_id = ks.record_credibility_conclusion(
        "p-rep", {"overall_verdict": "一致", "summary": "复现良好", "item_results": []}
    )
    assert [h["ref_id"] for h in ks.search(types=["credibility_conclusion"], q="复现良好")] == [conclusion_id]
    assert ks.get_item("credibility_conclusion", conclusion_id)["overall_verdict"] == "一致"

    # 重算结论：旧结论索引条目一并清掉，不留孤儿
    second_id = ks.record_credibility_conclusion("p-rep", {"overall_verdict": "近似", "summary": "略有偏差"})
    assert [h["ref_id"] for h in ks.search(types=["credibility_conclusion"])] == [second_id]
    assert ks.get_item("credibility_conclusion", conclusion_id) is None


def test_clear_reproduction_results_removes_index(isolated_db):
    """重跑复现前清对照记录时，同步清掉它的索引条目。"""
    ks.record_paper({"paper_id": "p-clr", "title": "T"})
    (item_id,) = ks.record_experiment_items("p-clr", [{"metric_name": "f1", "metric_value_reported": "0.8"}])
    ks.record_reproduction_result({"item_id": item_id, "metric_value_actual": 0.79})

    assert ks.clear_reproduction_results("p-clr") == 1
    assert ks.search(types=["reproduction_result"]) == []


def test_delete_paper_still_cascades_with_subtable_index(isolated_db):
    """子表进索引后，删论文的级联仍可用（子条目不得把自己算成「外部引用」）。"""
    ks.record_paper({"paper_id": "p-del", "title": "T"})
    (item_id,) = ks.record_experiment_items(
        "p-del", [{"metric_name": "f1", "metric_value_reported": "0.8", "dataset_name": "cifar"}]
    )
    ks.record_reproduction_result({"item_id": item_id, "metric_value_actual": 0.79})
    ks.record_credibility_conclusion("p-del", {"overall_verdict": "近似", "summary": "s"})

    assert ks.delete_item("paper", "p-del") is True
    assert ks.search() == []


def test_experiment_item_edit_refreshes_index(isolated_db):
    """用户改条目指标名后，检索命中的是改后的口径。"""
    ks.record_paper({"paper_id": "p-edit", "title": "T"})
    (item_id,) = ks.record_experiment_items("p-edit", [{"metric_name": "acc", "metric_value_reported": "0.9"}])
    assert [h["ref_id"] for h in ks.search(types=["experiment_item"], q="acc")] == [item_id]

    assert ks.update_experiment_item(item_id, metric_name="accuracy") is True
    assert [h["ref_id"] for h in ks.search(types=["experiment_item"], q="accuracy")] == [item_id]
    assert ks.search(types=["experiment_item"], q="acc") == []


def test_subtable_search_via_api(app_client, isolated_db):
    """GET /api/knowledge/search 能命中论文子表条目（前端检索面板口径）。"""
    ks.record_paper({"paper_id": "p-api", "title": "T"})
    item_ids = ks.record_experiment_items(
        "p-api", [{"metric_name": "accuracy", "metric_value_reported": "91.5", "dataset_name": "cifar10"}]
    )
    response = app_client.get("/api/knowledge/search", params={"types": "experiment_item", "q": "accuracy"})
    assert response.status_code == 200
    assert [h["ref_id"] for h in response.json()] == item_ids

    detail = app_client.get(f"/api/knowledge/items/experiment_item/{item_ids[0]}")
    assert detail.status_code == 200 and detail.json()["metric_name"] == "accuracy"


# --------------------------- 指标名归一 ---------------------------


def test_normalize_metric_name_reuses_canonical_metrics():
    assert paper_service.normalize_metric_name("Acc") == "accuracy"
    assert paper_service.normalize_metric_name("Macro-F1") == "f1"
    assert paper_service.normalize_metric_name("ROC-AUC") == "auc"
    assert paper_service.normalize_metric_name("Top-1") == "top1"
    assert paper_service.normalize_metric_name("  ") is None
    # 未收录的指标名原样规范化保留，不被丢弃
    assert paper_service.normalize_metric_name("BLEU-4") == "bleu_4"
