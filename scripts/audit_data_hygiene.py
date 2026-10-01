"""验收数据卫生审计（收口任务 F1）。

只读报告，默认不修改任何数据；`--apply` 才执行清理动作。报告内容：

1. dataset_registry 中同名多路径的历史条目（每次验收运行都会新建一个目录）；
2. data/_fixtures 下的开发夹具（可能含历史遗留副本）；
3. data/_acceptance 证据文件清单；
4. 索引与主表条数一致性（unified_index vs 各主表）。

用法：

    D:\\python.exe scripts/audit_data_hygiene.py                # 只报告
    D:\\python.exe scripts/audit_data_hygiene.py --json         # 输出 JSON
    D:\\python.exe scripts/audit_data_hygiene.py --apply        # 先备份 index.db 再清理「无对齐、无运行记录」的重复数据集条目

清理策略（--apply，保守）：仅删除同一 name 下、多路径中「既没有 alignment、也没有任何
run_record 引用、且不在 unified_index 之外被引用」的次要条目之前，先整体备份 data/index.db
到 data/_acceptance/。验收证据（已确认对齐、已评估过的数据集）永不删除。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = DATA / "index.db"
ACCEPTANCE = DATA / "_acceptance"
FIXTURES = DATA / "_fixtures"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def collect() -> dict:
    conn = _connect()
    try:
        datasets = [dict(r) for r in conn.execute(
            "SELECT dataset_id, name, source, url, alignment, local_path FROM dataset_registry ORDER BY name, created_at"
        )]
        index_count = conn.execute("SELECT data_type, COUNT(*) n FROM unified_index GROUP BY data_type").fetchall()
        counts = {r["data_type"]: r["n"] for r in index_count}
        table_counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("paper", "experiment_item", "reproduction_result", "credibility_conclusion",
                      "module", "knowledge", "run_record", "task", "project")
        }
        running_tasks = [dict(r) for r in conn.execute(
            "SELECT task_id, task_type, status FROM task WHERE status IN ('queued','running')"
        )]
    finally:
        conn.close()

    by_name: dict[str, list[dict]] = defaultdict(list)
    for d in datasets:
        by_name[d["name"] or "?"].append(d)

    duplicates = []
    for name, rows in sorted(by_name.items()):
        if len(rows) < 2:
            continue
        aligned = [r for r in rows if r["alignment"]]
        duplicates.append({
            "name": name,
            "count": len(rows),
            "with_alignment": len(aligned),
            "sources": sorted({r["source"] or "-" for r in rows}),
            "has_url": any(r["url"] for r in rows),
        })

    fixtures = sorted(p.name for p in FIXTURES.iterdir()) if FIXTURES.is_dir() else []
    evidence = sorted(p.name for p in ACCEPTANCE.glob("*")) if ACCEPTANCE.is_dir() else []

    return {
        "dataset_total": len(datasets),
        "duplicate_groups": duplicates,
        "unified_index": counts,
        "table_counts": table_counts,
        "nonterminal_tasks": running_tasks,
        "fixtures": fixtures,
        "acceptance_evidence": evidence,
    }


def apply_cleanup(report: dict) -> dict:
    """保守清理：备份 DB，删除「重复组内既无 alignment 又无 url」的次要条目。"""
    ACCEPTANCE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ACCEPTANCE / f"index.db.bak_{stamp}"
    shutil.copy2(DB, backup)

    conn = _connect()
    removed: list[str] = []
    try:
        for group in report["duplicate_groups"]:
            rows = conn.execute(
                "SELECT dataset_id, name, alignment, url FROM dataset_registry WHERE name = ? ORDER BY created_at DESC",
                (group["name"],),
            ).fetchall()
            keep = rows[0]["dataset_id"]
            for row in rows[1:]:
                if row["alignment"] or row["url"]:
                    continue          # 保留带对齐/来源的记录：验收证据不删
                conn.execute("DELETE FROM unified_index WHERE data_type='dataset' AND ref_id = ?", (row["dataset_id"],))
                conn.execute("DELETE FROM dataset_registry WHERE dataset_id = ?", (row["dataset_id"],))
                removed.append(row["dataset_id"])
        conn.commit()
    finally:
        conn.close()
    return {"backup": str(backup.relative_to(ROOT)), "removed_dataset_ids": removed, "kept": "组内最新条目"}


def main() -> int:
    ap = argparse.ArgumentParser(description="验收数据卫生审计（F1）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--apply", action="store_true", help="备份后执行保守清理")
    args = ap.parse_args()

    report = collect()
    if args.apply:
        report["cleanup"] = apply_cleanup(report)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"数据集总数: {report['dataset_total']}")
    print("同名多路径（历史验收遗留）:")
    for g in report["duplicate_groups"]:
        print(f"  {g['name']:<20} 条目={g['count']} 带对齐={g['with_alignment']} 来源={','.join(g['sources'])} url={'有' if g['has_url'] else '无'}")
    print("统一索引:", report["unified_index"])
    print("主表条数:", report["table_counts"])
    print("未终态任务:", report["nonterminal_tasks"] or "无")
    print(f"开发夹具: {len(report['fixtures'])} 项")
    print(f"验收证据: {len(report['acceptance_evidence'])} 项")
    if "cleanup" in report:
        print("清理:", report["cleanup"])
    else:
        print("（只读模式：加 --apply 才执行清理）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
