"""使用建议起草的失败闸门（《模块详细设计》5.4、2.5 异常边界）。

背景（2026-10-01 审核发现）：agent 端点不可用时 `run_sync` 返回空 structured_output，
compare 此前会继续落库，写入 content=None 的「使用建议」并标记为待确认，
把「起草失败」伪装成「已产出待确认」，问题要到用户确认时才暴露。
"""
from __future__ import annotations

import pytest

from app.services import compare_service, task_manager

TABLE = {"metrics": ["accuracy"], "columns": ["baseline", "public-x"], "rows": [{"baseline": 0.79, "public-x": 0.20}]}


def test_empty_draft_raises_and_does_not_write_knowledge(isolated_db):
    task_id = task_manager.create_task("compare", params={"project_id": "p"})

    with pytest.raises(RuntimeError) as exc:
        compare_service._require_guidance({}, TABLE, task_id)
    assert "使用建议起草失败" in str(exc.value)

    # 失败时把已算好的对比表写进任务进度（便于排查），且不写任何知识记录
    progress = task_manager.get_task(task_id)["progress"]
    assert progress is not None and "draft_failed" in progress and "accuracy" in progress


def test_draft_missing_content_field_is_rejected(isolated_db):
    task_id = task_manager.create_task("compare")
    with pytest.raises(RuntimeError):
        compare_service._require_guidance({"guidance_title": "标题但没有正文"}, TABLE, task_id)


def test_valid_draft_passes_through(isolated_db):
    task_id = task_manager.create_task("compare")
    draft = {"guidance_title": "使用建议", "guidance_content": "正文", "summary": "摘要"}
    assert compare_service._require_guidance(draft, TABLE, task_id) is draft
