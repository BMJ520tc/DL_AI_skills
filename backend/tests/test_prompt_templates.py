"""prompt 模板接线（《模块详细设计》2.5、阶段1实施方案「落位」、需求一.2）。

背景（2026-10-04 复核发现）：`agents/prompts/*.md` 三份模板从未被任何代码读取，
prompt 全部内联在 service 里，且内联内容与模板已经漂移。本文件锁住三件事：

① 加载器：白名单（不接受任意路径/名字）、进程内缓存、模板缺失时明确报错；
② 三处 prompt 组装 = 模板文本 + 运行期上下文 + **代码提供的 schema**；
③ 模板与实现契约对齐（`dependency_fix` 的「修正后的完整依赖清单」、
   `address_extract` 的「正文之外也看补充材料」、`dynamic_analysis` 的必需字段）。

不联网、不碰真实 `data/`；`agent_service.run_sync` 一律用假实现替换。
"""
from __future__ import annotations

import asyncio

import pytest

from app.services import analysis_service, download_service, env_manager, prompts


def _capture_run_sync(monkeypatch, module, structured: dict) -> dict:
    """把 `module.agent_service.run_sync` 换成捕获 prompt 的假实现，返回捕获容器。"""
    captured: dict = {}

    async def fake_run_sync(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {"structured_output": structured, "result": None}

    monkeypatch.setattr(module.agent_service, "run_sync", fake_run_sync)
    return captured


# ---------------------------------------------------------------- ① 加载器

def test_load_template_returns_file_text():
    text = prompts.load_template("address_extract")
    assert text.strip()
    assert text == prompts.template_path("address_extract").read_text(encoding="utf-8")


def test_all_whitelisted_templates_loadable():
    for name in prompts.TEMPLATE_NAMES:
        assert prompts.load_template(name).strip(), name


def test_load_template_missing_file_raises(tmp_path, monkeypatch):
    prompts.clear_cache()
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)   # 空目录 → 模板缺失

    with pytest.raises(FileNotFoundError) as exc:
        prompts.load_template("address_extract")

    assert "prompt 模板缺失" in str(exc.value)


def test_load_template_rejects_name_outside_whitelist(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)
    # 文件确实存在，但仍必须被白名单拒绝（安全性不靠「文件不存在」兜底）
    (tmp_path / "secret.md").write_text("不该被读到", encoding="utf-8")

    for bad in ("secret", "address_extract.md", "../../etc/passwd", "system", ""):
        with pytest.raises(ValueError):
            prompts.load_template(bad)


def test_load_template_is_cached_in_process(tmp_path, monkeypatch):
    prompts.clear_cache()
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)
    target = tmp_path / "dynamic_analysis.md"
    target.write_text("第一版", encoding="utf-8")
    assert prompts.load_template("dynamic_analysis") == "第一版"

    target.write_text("第二版", encoding="utf-8")
    assert prompts.load_template("dynamic_analysis") == "第一版"   # 进程内缓存

    prompts.clear_cache()
    assert prompts.load_template("dynamic_analysis") == "第二版"


def test_render_prompt_appends_context_and_schema():
    schema = {"type": "object", "properties": {"repositories": {"type": "array"}}, "required": ["repositories"]}
    text = prompts.render_prompt("address_extract", context="### 论文正文\n\nCONTEXT-MARK", schema=schema)

    assert prompts.load_template("address_extract").rstrip("\n") in text
    assert "CONTEXT-MARK" in text
    assert prompts.RUNTIME_HEADING in text
    assert prompts.SCHEMA_HEADING in text
    assert '"repositories"' in text and '"required"' in text


# ---------------------------------------------------------------- ② 三处组装

def test_run_extract_prompt_contains_template_context_and_schema(monkeypatch, isolated_db):
    captured = _capture_run_sync(
        monkeypatch, download_service, {"repositories": [], "datasets": []}
    )

    asyncio.run(download_service._run_extract({"paper_text": "PAPER-BODY-MARK", "cwd": None}, "task-x"))

    prompt = captured["prompt"]
    assert prompts.load_template("address_extract").rstrip("\n") in prompt
    assert "PAPER-BODY-MARK" in prompt
    assert '"repositories"' in prompt and '"datasets"' in prompt
    assert captured["kwargs"]["output_schema"] is download_service.ADDRESS_SCHEMA


def test_run_extract_prompt_marks_supplementary_material(monkeypatch, tmp_path, isolated_db):
    """模板声明「正文之外也看补充材料」，运行期上下文里补充材料要真的被带上。"""
    supp = tmp_path / "supplementary_notes.txt"
    supp.write_text("代码地址见 SUPP-MARK", encoding="utf-8")
    captured = _capture_run_sync(
        monkeypatch, download_service, {"repositories": [], "datasets": []}
    )

    asyncio.run(download_service._run_extract(
        {"paper_text": "BODY", "cwd": None, "supplementary_paths": [str(supp)]}, "task-x"
    ))

    prompt = captured["prompt"]
    assert "SUPP-MARK" in prompt
    assert "补充材料" in prompt


def test_dynamic_supplement_prompt_contains_template_context_and_schema(monkeypatch, tmp_path):
    captured = _capture_run_sync(monkeypatch, analysis_service, {"supplements": []})

    out = asyncio.run(analysis_service._dynamic_supplement(
        tmp_path, [{"file": "model.py", "reason": "getattr 动态取模块"}]
    ))

    assert out == {"supplements": []}
    prompt = captured["prompt"]
    assert prompts.load_template("dynamic_analysis").rstrip("\n") in prompt
    assert "model.py" in prompt and str(tmp_path) in prompt
    assert '"supplements"' in prompt
    assert captured["kwargs"]["output_schema"] is analysis_service.DYNAMIC_SCHEMA


def test_agent_fix_advice_prompt_contains_template_context_and_schema(monkeypatch, tmp_path):
    advice = {"requirements": ["numpy"], "reason": "保留 numpy"}
    captured = _capture_run_sync(monkeypatch, env_manager, advice)

    out = asyncio.run(env_manager._agent_fix_advice(tmp_path, "ERROR-MARK: No matching distribution"))

    assert out == advice
    prompt = captured["prompt"]
    assert prompts.load_template("dependency_fix").rstrip("\n") in prompt
    assert "ERROR-MARK" in prompt and str(tmp_path) in prompt
    assert '"requirements"' in prompt and '"reason"' in prompt
    assert captured["kwargs"]["output_schema"] is env_manager.FIX_SCHEMA


# ---------------------------------------------------------------- ③ 模板与实现对齐

def test_dependency_fix_template_matches_complete_requirements_contract():
    """契约是「返回修正后的完整依赖清单」，不是旧模板写的单点 action 修复。"""
    text = prompts.load_template("dependency_fix")
    assert "修正后的完整依赖清单" in text
    assert "requirements" in text and "reason" in text
    assert "target_version" not in text
    assert '"action"' not in text


def test_address_extract_template_covers_supplementary_material():
    text = prompts.load_template("address_extract")
    assert "补充材料" in text
    assert "正文" in text


def test_dynamic_schema_requires_supplements_to_match_template():
    assert analysis_service.DYNAMIC_SCHEMA["required"] == ["supplements"]
    assert "为必需字段" in prompts.load_template("dynamic_analysis")
