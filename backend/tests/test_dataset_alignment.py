"""数据集对齐确认闸门（《模块详细设计》5.3、GB-3 相关链路）。"""
from __future__ import annotations

from app.services import dataset_service, knowledge_service as ks


def _register(name: str = "public-x", url: str = "https://zenodo.org/records/1") -> str:
    return ks.register_dataset({
        "name": name,
        "url": url,
        "source": "zenodo",
        "task_type": "classification",
        "format": "csv",
    })


def test_alignment_gate_draft_then_confirmed(isolated_db):
    dataset_id = _register()
    # 未对齐时不得确认
    assert dataset_service.get_alignment(dataset_id) is None
    assert dataset_service.confirm_alignment(dataset_id) is False

    assert ks.update_alignment(dataset_id, {
        "field_mapping": {"label": "label", "input": "input"},
        "label_merge": {"DOG": "dog"},
        "version": "1.0",
    }) is True

    draft = dataset_service.get_alignment(dataset_id)
    assert draft["field_mapping"]["label"] == "label"
    assert draft.get("status") != "confirmed"          # 起草态：对齐规则不得直接参与评估

    assert dataset_service.confirm_alignment(dataset_id) is True
    confirmed = dataset_service.get_alignment(dataset_id)
    assert confirmed["status"] == "confirmed"
    assert "confirmed_at" in confirmed
    # 已确认时幂等
    assert dataset_service.confirm_alignment(dataset_id) is True


def test_alignment_unknown_dataset(isolated_db):
    assert dataset_service.get_alignment("no-such-dataset") is None
    assert dataset_service.confirm_alignment("no-such-dataset") is False


def test_dataset_registered_with_url_is_searchable(isolated_db):
    dataset_id = _register(name="public-real", url="https://zenodo.org/records/23080173")
    item = ks.get_item("dataset", dataset_id)
    assert item["url"] == "https://zenodo.org/records/23080173"
    assert [h["ref_id"] for h in ks.search(types=["dataset"], q="public-real")] == [dataset_id]
    found = ks.find_datasets(task_type="classification")
    assert dataset_id in [d["dataset_id"] for d in found]


# ---------- 规则兜底必须解析 JSON 文本列（E1 真实数据集暴露） ----------

def test_as_json_parses_db_text_and_tolerates_garbage():
    assert dataset_service._as_json('["id","label"]') == ["id", "label"]
    assert dataset_service._as_json(["id"]) == ["id"]
    assert dataset_service._as_json(None) is None
    assert dataset_service._as_json("[broken") is None


def test_rule_alignment_parses_json_text_columns():
    """get_item 返回的 fields/labels 是 JSON 文本；直接遍历字符串会得到空映射。"""
    source = {"dataset_id": "s1", "labels": '["Cat","dog"]'}
    target = {"dataset_id": "t1",
              "fields": '["id","split","label","input","meta_width"]',
              "labels": '["cat","dog"]'}
    draft = dataset_service._rule_alignment(source, target)
    assert draft["field_mapping"] == {"id": "id", "split": "split", "label": "label", "input": "input"}
    assert draft["label_merge"] == {"cat": "Cat"}


def test_rule_alignment_accepts_already_parsed_lists():
    draft = dataset_service._rule_alignment(
        {"labels": ["Cat", "dog"]},
        {"fields": ["label", "meta_x"], "labels": ["dog"]},
    )
    assert draft["field_mapping"] == {"label": "label"}
    assert draft["label_merge"] == {}


def test_find_self_dataset_excludes_external_datasets(isolated_db, tmp_path):
    """外部数据集预处理后也落在项目 data/ 下，不能因此被当成「自带数据」。"""
    ws = tmp_path / "proj"
    self_csv = ws / "data" / "self" / "preprocessed.csv"
    ext_csv = ws / "data" / "public-real" / "preprocessed.csv"
    for p in (self_csv, ext_csv):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("id,split,label,input\n", encoding="utf-8")

    self_id = ks.register_dataset({
        "name": "self", "source": "自带", "task_type": "classification",
        "fields": ["id", "split", "label", "input"], "labels": ["Cat", "dog"],
        "local_path": str(self_csv),
    })
    # 外部数据集后登记：按 updated_at 倒序取首条的实现会误选它
    ks.register_dataset({
        "name": "public-real", "source": "zenodo", "url": "https://zenodo.org/records/1",
        "fields": ["id", "split", "label", "input"], "labels": ["cat", "dog"],
        "local_path": str(ext_csv),
    })

    found = dataset_service.find_self_dataset(ws)
    assert found is not None
    assert found["dataset_id"] == self_id
    assert found["name"] == "self"


def test_find_self_dataset_returns_none_without_self_data(isolated_db, tmp_path):
    ws = tmp_path / "proj"
    ext_csv = ws / "data" / "only-external" / "preprocessed.csv"
    ext_csv.parent.mkdir(parents=True, exist_ok=True)
    ext_csv.write_text("id,split,label,input\n", encoding="utf-8")
    ks.register_dataset({"name": "ext", "source": "figshare", "local_path": str(ext_csv)})
    assert dataset_service.find_self_dataset(ws) is None


def test_align_task_falls_back_to_rule_when_agent_unavailable(isolated_db, tmp_path, monkeypatch):
    """对齐任务全链路：agent 无产出时必须由规则兜底产出可用对齐（而不是报「缺少字段信息」）。"""
    import asyncio

    ws = tmp_path / "proj"
    (ws / "data" / "self").mkdir(parents=True)
    (ws / "source").mkdir(parents=True)
    self_csv = ws / "data" / "self" / "preprocessed.csv"
    self_csv.write_text("id,split,label,input\n0,train,Cat,a.png\n", encoding="utf-8")

    ks.register_dataset({
        "name": "self", "source": "自带", "task_type": "classification",
        "fields": ["id", "split", "label", "input"], "labels": ["Cat", "dog"],
        "local_path": str(self_csv),
    })
    target_id = ks.register_dataset({
        "name": "public-real", "source": "zenodo", "url": "https://zenodo.org/records/1",
        "task_type": "classification", "format": "image_dir",
        "fields": ["id", "split", "label", "input", "meta_width"], "labels": ["cat", "dog"],
        "local_path": str(tmp_path / "public" / "preprocessed.csv"),
    })

    async def _empty_agent(*args, **kwargs):
        return {"structured_output": None, "result": "Not logged in · Please run /login"}

    monkeypatch.setattr(dataset_service.agent_service, "run_sync", _empty_agent)
    monkeypatch.setattr(dataset_service.project_manager, "get_project",
                        lambda pid: {"project_id": pid, "workspace_path": str(ws), "project_type": "original"})

    asyncio.run(dataset_service._run({"project_id": "p1", "dataset_id": target_id}, "task-align"))

    alignment = dataset_service.get_alignment(target_id)
    assert alignment is not None
    assert alignment["drafted_by"] == "rule_fallback"
    assert alignment["field_mapping"] == {"id": "id", "split": "split", "label": "label", "input": "input"}
    assert alignment["label_merge"] == {"cat": "Cat"}
    assert alignment["status"] == "draft"
    assert alignment["aligned_projects"] == ["p1"]
