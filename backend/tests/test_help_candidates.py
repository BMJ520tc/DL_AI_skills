"""`--help` 输出解析以定位最小可运行命令（需求一.2 第三种来源、《模块详细设计》3.5）。

背景（2026-10-04 复核发现）：原实现只把 `--help` 当作候选命令的「快速验证变体」，
从不解析其输出，因此无法从 `usage: train.py [-h] --data DIR [--epochs N]` 推出带必填参数的候选。
本文件锁住：解析口径（固定代码 + 正则，不调大模型）、候选来源标注，以及
「解析不到 → 退回既有 README/脚本目录口径且不抛错」的退化路径。

假脚本用临时目录 + 本仓库解释器（`sys.executable`）跑 `--help`，不联网、不碰真实 `data/`。
"""
from __future__ import annotations

import asyncio
import sys

from app.services import analysis_service

FAKE_TRAIN = '''\
import argparse

p = argparse.ArgumentParser(description="fake trainer")
p.add_argument("--data", required=True, metavar="DIR", help="dataset dir")
p.add_argument("--epochs", type=int, default=1, metavar="N")
p.add_argument("--lr", type=float, default=0.1, metavar="LR")
p.parse_args()
'''


def _write_project(tmp_path, files: dict):
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return source


# ---------------------------------------------------------------- 解析口径

def test_required_option_tokens_takes_required_and_skips_optional():
    """需求原文的示例：`usage: train.py [-h] --data DIR [--epochs N]`。"""
    block = analysis_service._usage_block(
        "usage: train.py [-h] --data DIR [--epochs N] [--lr LR]\n\noptions:\n  -h, --help\n"
    )

    assert block is not None
    assert analysis_service._required_option_tokens(block, None) == ["--data", "<DIR>"]


def test_required_option_tokens_uses_real_path_when_project_has_one(tmp_path):
    source = _write_project(tmp_path, {})
    (source / "data").mkdir()

    block = analysis_service._usage_block("usage: train.py --data DIR\n")

    assert analysis_service._required_option_tokens(block, source) == ["--data", "data"]


def test_required_group_takes_first_alternative_only():
    block = analysis_service._usage_block("usage: run.py [-h] (--a A | --b B)\n")
    assert analysis_service._required_option_tokens(block, None) == ["--a", "<A>"]


def test_usage_block_follows_wrapped_lines_and_stops_at_options_section():
    block = analysis_service._usage_block(
        "usage: train.py [-h] --data DIR [--epochs N]\n"
        "                [--batch-size B]\n"
        "\n"
        "options:\n  -h, --help  show\n"
    )
    assert block == "train.py [-h] --data DIR [--epochs N] [--batch-size B]"


def test_usage_block_returns_none_without_usage():
    assert analysis_service._usage_block("no usage line here") is None
    assert analysis_service._usage_block("") is None


# ---------------------------------------------------------------- 候选生成与来源

def test_help_output_yields_candidate_with_required_args(tmp_path):
    source = _write_project(tmp_path, {"train.py": FAKE_TRAIN})
    (source / "data").mkdir()

    cands = asyncio.run(analysis_service._candidate_commands(source, sys.executable))
    help_cmds = [c["command"] for c in cands if c["source"] == "help"]

    assert ["train.py", "--data", "data"] in help_cmds
    # 可选参数不进候选（只推断必填项）
    assert all("--epochs" not in cmd and "--lr" not in cmd for cmd in help_cmds)


def test_three_sources_are_distinguishable(tmp_path):
    """需求一.2 的三种来源（README / 脚本目录 / --help 输出）在候选里要能区分。"""
    source = _write_project(tmp_path, {
        "README.md": "训练：python main.py\n",
        "main.py": FAKE_TRAIN,
        "scripts/run.sh": "echo run\n",
    })
    (source / "data").mkdir()

    cands = asyncio.run(analysis_service._candidate_commands(source, sys.executable))
    sources = {c["source"] for c in cands}

    assert {"readme", "script_dir", "help"} <= sources
    assert ["main.py", "--data", "data"] in [c["command"] for c in cands if c["source"] == "help"]


def test_candidate_order_follows_readme_script_dir_then_help(tmp_path):
    """《模块详细设计》3.5：候选来源优先级 README → 脚本目录 → `--help` 输出 → 库型兜底。"""
    source = _write_project(tmp_path, {
        "README.md": "训练：python main.py\n",
        "main.py": FAKE_TRAIN,
        "scripts/run.sh": "echo run\n",
        "pkg/__init__.py": "",
    })
    (source / "data").mkdir()

    sources = [c["source"] for c in asyncio.run(
        analysis_service._candidate_commands(source, sys.executable)
    )]

    assert sources.index("readme") < sources.index("script_dir") < sources.index("help")
    assert sources.index("help") < sources.index("package")


# ---------------------------------------------------------------- 退化路径（不报错、退回既有口径）

def test_no_help_output_falls_back_to_existing_sources(tmp_path, monkeypatch):
    source = _write_project(tmp_path, {
        "README.md": "run: python train.py\n",
        "train.py": FAKE_TRAIN,
        "scripts/run.sh": "echo run\n",
    })
    monkeypatch.setattr(analysis_service, "_probe_help", lambda *a, **k: "")   # 输出为空

    cands = asyncio.run(analysis_service._candidate_commands(source, sys.executable))
    sources = {c["source"] for c in cands}

    assert "help" not in sources                      # 解析不到就不生成 help 候选
    assert {"readme", "script_dir"} <= sources        # 退回既有口径
    assert ["train.py"] in [c["command"] for c in cands]


def test_script_without_help_support_does_not_raise(tmp_path):
    """脚本没有 argparse（`--help` 无 usage 输出）→ 不报错，退回既有口径。"""
    source = _write_project(tmp_path, {"train.py": "print('plain script')\n"})

    cands = asyncio.run(analysis_service._candidate_commands(source, sys.executable))

    assert all(c["source"] != "help" for c in cands)
    assert ["train.py"] in [c["command"] for c in cands]


def test_nonzero_exit_with_usage_is_parsed_and_never_raises(tmp_path):
    """非零退出不算错误：打印了 usage 就照常解析（信息更全），没 usage 就退回既有口径。"""
    source = _write_project(tmp_path, {
        "train.py": "import sys\nprint('usage: train.py [-h] --data DIR')\nsys.exit(1)\n",
    })

    cands = asyncio.run(analysis_service._candidate_commands(source, sys.executable))
    help_cmds = [c["command"] for c in cands if c["source"] == "help"]

    assert help_cmds and help_cmds[0][:2] == ["train.py", "--data"]


def test_missing_project_interpreter_does_not_probe_host(tmp_path, monkeypatch):
    """环境未就绪（python=None）不回退宿主解释器：一次探测都不发。"""
    source = _write_project(tmp_path, {"train.py": FAKE_TRAIN})
    calls: list = []
    monkeypatch.setattr(analysis_service, "_probe_help", lambda *a, **k: calls.append(a) or "")

    cands = asyncio.run(analysis_service._candidate_commands(source, None))

    assert calls == []
    assert all(c["source"] != "help" for c in cands)


def test_unusable_interpreter_path_does_not_raise(tmp_path):
    source = _write_project(tmp_path, {"train.py": FAKE_TRAIN})

    cands = asyncio.run(
        analysis_service._candidate_commands(source, str(tmp_path / "no_such_python.exe"))
    )

    assert all(c["source"] != "help" for c in cands)


# ---------------------------------------------------------------- 来源落 run_record

def test_record_smoke_marks_candidate_source(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        analysis_service.knowledge_service, "record_run", lambda run: captured.update(run) or "run-x"
    )

    analysis_service._record_smoke("pid", "tid", {
        "command": "train.py --data data", "source": "help", "mode": "exited", "ok": False,
        "error": "boom", "started_at": "s", "finished_at": "f",
    })

    assert captured["params"] == {"mode": "exited", "source": "help"}
    assert captured["status"] == "failed"
