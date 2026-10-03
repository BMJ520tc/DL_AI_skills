"""数据交接包的导入器（自包含，可脱离本仓库运行——会被复制进交接包里）。

把交接包里的数据库/模块/项目报告放进本机仓库的 `data/`，并把库中记录的**绝对路径**
从占位符还原为本机的仓库根路径。

单独用法（在交接包目录里）：
    python import_data.py --repo-root E:\\work\\DL-AI-skills
    python import_data.py --repo-root . --force

被 `scripts/export_data.py --import` 复用（避免同一套逻辑写两遍）。
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

_TOK_ESC, _TOK, _TOK_FWD = "{{ROOT_ESC}}", "{{ROOT}}", "{{ROOT_FWD}}"
_TEXT_SUFFIXES = {".json", ".md", ".txt", ".py", ".csv", ".yaml", ".yml", ".cfg", ".ini"}


def _pairs(root: str) -> list[tuple[str, str]]:
    return [
        (_TOK_ESC, root.replace("\\", "\\\\")),   # JSON 转义形式
        (_TOK, root),
        (_TOK_FWD, root.replace("\\", "/")),
    ]


def rewrite_db(db_path: Path, pairs: list[tuple[str, str]]) -> int:
    """改写库中所有文本列里的占位符。

    跳过 `unified_index_fts*`：外部内容型 FTS5 的影子表由触发器维护，直接改会破坏检索；
    改主表 `unified_index` 时触发器会同步索引。
    """
    conn = sqlite3.connect(str(db_path))
    changed = 0
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            if t.startswith("sqlite_") or t.startswith("unified_index_fts"):
                continue
            for col in [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]:
                for old, new in pairs:
                    cur = conn.execute(
                        f'UPDATE "{t}" SET "{col}" = REPLACE("{col}", ?, ?) WHERE "{col}" LIKE ?',
                        (old, new, f"%{old}%"),
                    )
                    changed += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return changed


def copy_tree(src: Path, dst: Path, pairs: list[tuple[str, str]]) -> int:
    n = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in _TEXT_SUFFIXES:
            try:
                target.write_text(
                    _apply(p.read_text(encoding="utf-8"), pairs), encoding="utf-8")
            except UnicodeDecodeError:
                shutil.copy2(p, target)
            else:
                n += 1
        else:
            shutil.copy2(p, target)
    return n


def _apply(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        if old in text:
            text = text.replace(old, new)
    return text


def import_package(package: Path, repo_root: Path, force: bool = False) -> int:
    """把交接包导入 `repo_root/data/`，返回进程退出码。"""
    if not package.is_dir():
        print(f"未找到交接包目录：{package}", file=sys.stderr)
        return 2
    data_dir = repo_root / "data"
    pairs = _pairs(str(repo_root))

    src_db = package / "index.db"
    if src_db.exists():
        dst_db = data_dir / "index.db"
        if dst_db.exists() and not force:
            print(f"{dst_db} 已存在（加 --force 覆盖）", file=sys.stderr)
            return 2
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_db, dst_db)
        n = rewrite_db(dst_db, pairs)
        print(f"[1/5] index.db 已导入并改写 {n} 处路径 → 仓库根 = {repo_root}")

    for name, label in (("modules", "modules/"), ("papers", "papers/"), ("datasets", "datasets/")):
        if (package / name).is_dir():
            n = copy_tree(package / name, data_dir / name, pairs)
            print(f"[·] {label} 已导入（{n} 个文本文件已改写）")

    if (package / "projects").is_dir():
        n = 0
        for proj in sorted((package / "projects").iterdir()):
            if (proj / "reports").is_dir():
                copy_tree(proj / "reports", data_dir / "projects" / proj.name / "reports", pairs)
                n += 1
        print(f"[·] projects/*/reports/ 已导入（{n} 个项目）")

    print("\n导入完成。启动后端即可查看已有知识与模块。")
    print("提示：只带了「系统产出」（论文解析结果与图片、数据集注册信息），"
          "不含 PDF 原件与数据压缩包，也没有项目源码/虚拟环境——"
          "浏览可用；「打开项目继续跑」不可用。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="导入数据交接包并还原绝对路径")
    ap.add_argument("--package", default=str(Path(__file__).resolve().parent),
                    help="交接包目录（默认为本脚本所在目录）")
    ap.add_argument("--repo-root", required=True, help="你的仓库根目录（数据将放到其 data/ 下）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 data/index.db")
    args = ap.parse_args()
    return import_package(Path(args.package).resolve(), Path(args.repo_root).resolve(), args.force)


if __name__ == "__main__":
    sys.exit(main())
