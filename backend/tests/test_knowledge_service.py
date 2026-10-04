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


def test_bring_advice_summary_only_actionable_and_empty_when_none(isolated_db):
    """带入紧凑视图（8.2）：只保留参数建议与冲突预警，其余类型不进；无命中返回 {}。"""
    assert ks.bring_advice_summary() == {}          # 空库 → 静默跳过

    # 未确认的使用建议不进带入；已确认的参数建议进
    ks.record_knowledge({"type": "usage_guidance", "title": "U", "content": "u", "status": "confirmed"})
    assert ks.bring_advice_summary() == {}

    kid = ks.record_knowledge({"type": "param_advice", "title": "P", "content": "use lr=1e-3",
                               "scope": {"task_type": "classification"}, "status": "confirmed"})
    summary = ks.bring_advice_summary(task_type="classification")
    assert [i["knowledge_id"] for i in summary["param_advice"]] == [kid]
    assert summary["dependency_conflict"] == []
    # 维度不匹配 → 静默跳过
    assert ks.bring_advice_summary(task_type="regression") == {}


def test_conflict_detection_and_supersede_on_confirm(isolated_db):
    """冲突检测与 superseded 状态机（数据设计六.3）：同类型 + scope 相容才冲突；确认时推翻旧结论。"""
    old = ks.record_knowledge({"type": "param_advice", "title": "old", "content": "lr=1e-2",
                               "scope": {"task_type": "classification"}, "status": "confirmed"})
    new = ks.record_knowledge({"type": "param_advice", "title": "new", "content": "lr=1e-3",
                               "scope": {"task_type": "classification"}})
    assert [c["knowledge_id"] for c in ks.find_conflicts(new)] == [old]

    # 不相容 scope（regression）不算冲突
    other = ks.record_knowledge({"type": "param_advice", "title": "r", "content": "lr=1e-4",
                                 "scope": {"task_type": "regression"}})
    assert ks.find_conflicts(other) == []
    # 不同类型不算冲突
    diff_type = ks.record_knowledge({"type": "usage_guidance", "title": "u", "content": "x",
                                     "scope": {"task_type": "classification"}})
    assert ks.find_conflicts(diff_type) == []

    assert ks.confirm_knowledge(new, supersede_conflicts=True) is True
    assert ks.get_item("knowledge", new)["status"] == "confirmed"
    assert ks.get_item("knowledge", old)["status"] == "superseded"   # 被推翻


def test_supersede_and_list_by_status(isolated_db):
    kid = ks.record_knowledge({"type": "usage_guidance", "title": "t", "content": "c",
                               "status": "confirmed"})
    assert ks.supersede_knowledge(kid) is True
    assert ks.get_item("knowledge", kid)["status"] == "superseded"
    assert ks.supersede_knowledge(kid) is False          # 仅 confirmed → superseded

    ks.record_knowledge({"type": "param_advice", "title": "d", "content": "x"})  # draft
    assert [k["status"] for k in ks.list_knowledge(status="draft")] == ["draft"]
    assert len(ks.list_knowledge(status="superseded")) == 1


def test_knowledge_for_task_exists(isolated_db):
    assert ks.knowledge_for_task_exists("t1") is False
    ks.record_knowledge({"type": "dependency_conflict", "title": "x", "content": "c",
                         "structured": {"source_task_id": "t1"}})
    assert ks.knowledge_for_task_exists("t1") is True
    assert ks.knowledge_for_task_exists("t2") is False


def test_scope_normalization_and_matching(isolated_db):
    """scope 归一（别名/无关键/列表）+ 匹配容错（列表任一命中、互为子串）。

    背景：蒸馏草稿的 scope 由大模型给，实测会出现别名键 `task`、无关键 `conditioning`、
    列表值 `["BioSNAP"]`、描述句 `跨数据集通用（Norman…）`。用严格字符串相等时，只要该维度
    被查询就整条被排除（知识反而带不出来）。
    """
    import json
    kid = ks.record_knowledge({
        "type": "usage_guidance", "title": "t", "content": "c", "status": "confirmed",
        "scope": {"task": "classification", "conditioning": ["MD"],
                  "dataset": ["BioSNAP", "BindingDB"], "model": "GEARS (SGC)"},
    })
    stored = json.loads(ks.get_item("knowledge", kid)["scope"])
    # 别名 task→task_type；无关键 conditioning 剔除；列表保留；模型名（含后缀）保留
    assert stored == {"task_type": "classification", "dataset": ["BioSNAP", "BindingDB"], "model": "GEARS (SGC)"}

    def others(**kw):
        return [x["knowledge_id"] for x in ks.bring_knowledge(**kw)["others"]]

    assert kid in others(dataset="BioSNAP")          # 列表任一命中
    assert kid in others(model="GEARS")              # 互为子串（"GEARS" ⊂ "GEARS (SGC)"）
    assert kid in others(task_type="classification")  # 别名键已归一
    assert kid not in others(dataset="OtherDS")      # 不相容 → 排除
    assert kid not in others(task_type="regression")

    # 旧式未归一行的兼容：裸 "task" 键也能匹配
    assert ks._match_scope({"scope": json.dumps({"task": "classification"})}, "classification", None, None) is True
    assert ks._match_scope({"scope": json.dumps({"task": "classification"})}, "regression", None, None) is False
    # 列表值 + 字符串查询的兼容
    assert ks._match_scope({"scope": json.dumps({"dataset": ["A", "B"]})}, None, None, "B") is True


def test_test_fixture_never_touches_real_db(isolated_db):
    """隔离自检：夹具生效时索引表应为空，且不指向仓库 data/index.db。"""
    from app.db import connection

    assert str(connection.DB_PATH).endswith("index.db")
    assert str(connection.DB_PATH) != str((connection.DB_PATH.parent.parent / "data" / "index.db"))
    assert ks.search() == []
