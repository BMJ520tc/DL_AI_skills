"""知识库读写与统一索引（《模块详细设计》2.6、数据设计二/三）。"""
from __future__ import annotations

from app.services import knowledge_service as ks


def test_paper_roundtrip_and_fts(isolated_db):
    paper_id = ks.record_paper({
        "paper_id": "p1",
        "title": "A Tiny Paper on Widgets",
        "abstract": "We study widget robustness.",
        "source": "local",
    })
    assert paper_id == "p1"

    item = ks.get_item("paper", "p1")
    assert item["title"] == "A Tiny Paper on Widgets"
    assert item["status"] == "downloaded"

    hits = ks.search(types=["paper"], q="widgets")
    assert [h["ref_id"] for h in hits] == ["p1"]
    assert ks.search(types=["paper"], q="definitely-absent-term") == []


def test_run_record_written_and_indexed(isolated_db):
    run_id = ks.record_run({
        "project_id": "proj-1",
        "run_type": "env_install",
        "status": "success",
        "command": "pip install -r requirements.txt",
        "started_at": "2026-10-01T00:00:00+00:00",
        "finished_at": "2026-10-01T00:00:10+00:00",
    })
    runs = ks.list_runs("proj-1", "env_install")
    assert [r["run_id"] for r in runs] == [run_id]
    assert runs[0]["duration_s"] == 10
    assert [h["ref_id"] for h in ks.search(types=["run"], q="requirements")] == [run_id]


def test_experiment_items_confirm_gate_and_rebuild_idempotent(isolated_db):
    ks.record_paper({"paper_id": "p2", "title": "T", "source": "local"})
    ids = ks.record_experiment_items("p2", [
        {"metric_name": "accuracy", "metric_value_reported": 0.91, "dataset_name": "cifar"},
    ])
    assert len(ids) == 1
    assert ks.get_experiment_item(ids[0])["status"] == "extracted"
    assert ks.list_experiment_items("p2")[0]["metric_value_reported"] == "0.91"

    # 确认闸门（4.2）：extracted → confirmed
    assert ks.confirm_experiment_item(ids[0]) is True
    assert ks.list_experiment_items("p2")[0]["status"] == "confirmed"

    # 重新抽取幂等：旧条目被清理
    ks.record_experiment_items("p2", [{"metric_name": "f1", "metric_value_reported": 0.8}])
    items = ks.list_experiment_items("p2")
    assert len(items) == 1
    assert items[0]["metric_name"] == "f1"


def test_knowledge_draft_then_confirm(isolated_db):
    knowledge_id = ks.record_knowledge({
        "type": "usage_guidance",
        "title": "Widget guidance",
        "content": "Use widgets carefully on cross-domain data.",
        "confidence": "medium",
    })
    assert ks.get_item("knowledge", knowledge_id)["status"] == "draft"
    assert ks.confirm_knowledge(knowledge_id) is True
    assert ks.get_item("knowledge", knowledge_id)["status"] == "confirmed"
    # 仅 draft → confirmed，重复确认返回 False
    assert ks.confirm_knowledge(knowledge_id) is False
    assert [h["ref_id"] for h in ks.search(types=["knowledge"], q="widgets")] == [knowledge_id]


def test_unknown_data_type_rejected(isolated_db):
    try:
        ks.get_item("nope", "x")
    except ValueError as e:
        assert "unknown data_type" in str(e)
    else:  # pragma: no cover
        raise AssertionError("未知 data_type 应抛 ValueError")


def test_test_fixture_never_touches_real_db(isolated_db):
    """隔离自检：夹具生效时索引表应为空，且不指向仓库 data/index.db。"""
    from app.db import connection

    assert str(connection.DB_PATH).endswith("index.db")
    assert str(connection.DB_PATH) != str((connection.DB_PATH.parent.parent / "data" / "index.db"))
    assert ks.search() == []
