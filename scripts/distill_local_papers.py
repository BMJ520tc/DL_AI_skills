# scripts/distill_local_papers.py — 对**本地已有论文**做知识蒸馏（模块六 8.3、需求六.1）。
#
# 从每篇论文的结构化素材（实验条目 + 复现/可信度结论）提炼可复用结论，**以 draft 落库**
# （不自动确认；在界面「知识库 → 草稿」页签确认）。幂等：已蒸馏过的论文（structured.source_paper_id
# 命中）会跳过。默认跳过明显是测试样例的论文，用 --all 或显式 id 覆盖。
#
# 用法:
#     D:\python.exe scripts/distill_local_papers.py                 # 全部真实论文（跳过测试样例）
#     D:\python.exe scripts/distill_local_papers.py --all           # 含测试样例
#     D:\python.exe scripts/distill_local_papers.py gears esm3      # 指定论文 id
#     D:\python.exe scripts/distill_local_papers.py --dry-run       # 只看会蒸馏哪些论文、不调用模型

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.services import distill_service, knowledge_service  # noqa: E402

# 测试/样例论文（默认跳过；它们不是真实研究论文，蒸馏只会产出噪声）
FIXTURE_PAPERS = {"test-paper-1", "e2e-paper", "gate-demo", "scan-paper", "noexp-paper"}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    dry = "--dry-run" in flags

    all_ids = knowledge_service.list_paper_ids()
    if args:
        paper_ids = [p for p in args if p in all_ids]
        missing = [p for p in args if p not in all_ids]
        if missing:
            print(f"[warn] 库中不存在这些论文 id：{missing}")
    elif "--all" in flags:
        paper_ids = all_ids
    else:
        paper_ids = [p for p in all_ids if p not in FIXTURE_PAPERS]

    print(f"论文库共 {len(all_ids)} 篇；本次蒸馏 {len(paper_ids)} 篇：{paper_ids}")
    if dry:
        return 0

    result = asyncio.run(distill_service.distill_papers(paper_ids))
    print("\n=== 蒸馏结果 ===")
    for pid, ids in result["papers"].items():
        if isinstance(ids, dict):
            print(f"  {pid}: 失败 — {ids.get('error')}")
        else:
            print(f"  {pid}: {len(ids)} 条草稿")
    print(f"合计新增草稿：{result['total']} 条（在「知识库 → 草稿」确认后生效）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
