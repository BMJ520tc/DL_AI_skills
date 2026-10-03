"""数据交接：导出/导入 data/ 中「可交接」的部分，并改写其中的绝对路径。

背景：`data/` 整体约 8.8 GB，但其中绝大部分是可重建或纯缓存——
  - `projects/*/env`（约 4.65 GB）项目虚拟环境：内部是绝对路径，换机器直接用不了，可重建
  - `projects/*/source`（约 3.94 GB）克隆的源码：可按 URL 重新拉取
  - `papers/`（122 MB）论文全文、`datasets/`（32 MB）数据集：体积大，且登记信息已在库里
  - `agent_tasks/`、`numba_cache/`：任务中间产物与编译缓存，可丢弃
真正小而关键的是 `index.db`（约 2 MB，全部表都在这个文件里）。

本脚本只带「看已有知识/模块」所需的最小集合，并解决**换机器后数据库里的绝对路径指空**的问题：
  - 导出：`index.db`（用 SQLite 的 VACUUM INTO 导出，**后端在跑也能安全导出**，不必停服务）
          + `modules/`（标准化模块包）+ `projects/*/reports/`（IR / 结构报告 / 验证记录）
          + 导入说明；导出时把本机仓库根路径替换为占位符
  - 导入：把占位符还原为目标机器的仓库根路径

用法：
    # 导出（本机）
    D:\\python.exe scripts/export_data.py --out data_export --zip
    # 导入（对方机器，在本仓库内执行；默认根 = 本脚本所在仓库的根）
    D:\\python.exe scripts/export_data.py --import data_export
    D:\\python.exe scripts/export_data.py --import data_export --root E:\\work\\DL-AI-skills
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = DATA / "index.db"

# 占位符：区分 JSON 里的转义形式（双反斜杠）与普通形式，导入时按目标根分别还原
_TOK_ESC, _TOK, _TOK_FWD = "{{ROOT_ESC}}", "{{ROOT}}", "{{ROOT_FWD}}"
_TEXT_SUFFIXES = {".json", ".md", ".txt", ".py", ".csv", ".yaml", ".yml", ".cfg", ".ini"}

README = """\
# 数据交接包（DL-AI-skills）

包含「查看已有知识与模块」所需的系统数据：

| 项 | 说明 |
|---|---|
| `index.db` | 数据库本体（项目 / 论文 / 数据集 / 运行记录 / 知识 / 模块 / 统一检索索引） |
| `modules/` | 标准化模块包（`module.py` + `module.json` + `assets/`） |
| `projects/*/reports/` | 各项目的 IR / 结构报告 / 验证记录（供查看器展示） |
| `papers/` | 论文解析结果（markdown / 表格 / 图片，**不含 PDF 原件**） |
| `datasets/` | 数据集注册信息（**不含数据压缩包**） |
| `import_data.py` | 导入脚本（自包含，无需本仓库的其它文件） |

**不含**（体积大或与本机绑定，属可重建/缓存）：项目虚拟环境 `env/`、克隆源码 `source/`、
PDF 原件、数据集压缩包、任务中间产物、编译缓存。
因此**浏览知识库、模块、论文解析内容、项目列表与报告都可用**；但「打开项目继续跑」不可用。

## 导入步骤

1. 把本仓库放到你的机器上（任意目录），并装好后端依赖。
2. 解压本交接包，在**解压出来的目录**里执行：

   ```bash
   python import_data.py --repo-root <你的仓库根>
   ```

   脚本会把数据放到 `<你的仓库>/data/`，并把库里记录的**绝对路径改写为你本机的仓库根**。
3. 启动后端即可在界面上看到已有知识、模块与项目列表。

> 如果你手上有本仓库的 `scripts/export_data.py`，也可用它导入（效果相同）：
> `python scripts/export_data.py --import <本包目录>`
"""


def _pairs_to_tokens() -> list[tuple[str, str]]:
    root = str(ROOT)
    return [
        (root.replace("\\", "\\\\"), _TOK_ESC),  # JSON 转义形式（先长后短）
        (root, _TOK),
        (root.replace("\\", "/"), _TOK_FWD),
    ]


def _apply(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        if old in text:
            text = text.replace(old, new)
    return text


def _rewrite_db(db_path: Path, pairs: list[tuple[str, str]]) -> int:
    """把库中所有文本列里的路径整体替换。

    跳过 `unified_index_fts*`：它是外部内容型 FTS5 的影子表，由触发器维护，
    直接改会破坏检索；改主表 `unified_index` 时触发器会同步索引。
    """
    conn = sqlite3.connect(str(db_path))
    changed = 0
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            if t.startswith("sqlite_") or t.startswith("unified_index_fts"):
                continue
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
            for col in cols:
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


def _copy_tree(src: Path, dst: Path, pairs: list[tuple[str, str]], skip=None) -> int:
    """复制目录；文本文件顺带改写路径，其余按二进制原样复制。`skip` 可排除部分文件。"""
    n = 0
    for p in sorted(src.rglob("*")):
        if skip and p.is_file() and skip(p):
            continue
        rel = p.relative_to(src)
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in _TEXT_SUFFIXES:
            try:
                target.write_text(_apply(p.read_text(encoding="utf-8"), pairs), encoding="utf-8")
            except UnicodeDecodeError:
                shutil.copy2(p, target)
            else:
                n += 1
        else:
            shutil.copy2(p, target)
    return n


def _collect_reports(data_dir: Path) -> list[Path]:
    """有报告内容的项目（`reports/` 为空的项目没有可交接的内容，不计入）。"""
    out = []
    if not (data_dir / "projects").is_dir():
        return out
    for proj in sorted((data_dir / "projects").iterdir()):
        rep = proj / "reports"
        if proj.is_dir() and rep.is_dir() and any(rep.iterdir()):
            out.append(proj)
    return out


def do_export(out: Path, with_zip: bool, force: bool, with_papers: bool,
              with_pdfs: bool, with_datasets: bool, with_dataset_files: bool) -> int:
    if not DB.exists():
        print(f"未找到数据库：{DB}", file=sys.stderr)
        return 2
    if out.exists() and any(out.iterdir()) and not force:
        print(f"输出目录非空：{out}（加 --force 覆盖已存在文件）", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    # 1) 数据库：VACUUM INTO 生成一致副本（对运行中的库也安全）
    dst_db = out / "index.db"
    if dst_db.exists():
        dst_db.unlink()
    src = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        src.execute("VACUUM INTO '%s'" % str(dst_db).replace("'", "''"))
    finally:
        src.close()
    pairs = _pairs_to_tokens()
    changed = _rewrite_db(dst_db, pairs)
    print(f"[1/5] 数据库导出 {dst_db.name}（{dst_db.stat().st_size / 1024:.0f} KB，改写 {changed} 处路径）")

    # 2) 模块包
    if (DATA / "modules").is_dir():
        n = _copy_tree(DATA / "modules", out / "modules", pairs)
        print(f"[2/5] modules/ 已导出（{n} 个文本文件已改写）")

    # 3) 各项目的报告
    projs = _collect_reports(DATA)
    for proj in projs:
        _copy_tree(proj / "reports", out / "projects" / proj.name / "reports", pairs)
    print(f"[3/5] projects/*/reports/ 已导出（{len(projs)} 个项目）")

    # 4) 论文：默认带「系统产出」（解析出的 markdown / 表格 / 图片）；PDF 原件体积大，另开关
    if with_papers and (DATA / "papers").is_dir():
        skip = None if with_pdfs else (lambda p: p.suffix.lower() == ".pdf")
        n = _copy_tree(DATA / "papers", out / "papers", pairs, skip=skip)
        print(f"[4/5] papers/ 已导出（{'含 PDF 原件' if with_pdfs else '不含 PDF 原件'}，改写 {n} 个文本文件）")
    else:
        print("[4/5] 跳过 papers/（未指定 --with-papers）")

    # 5) 数据集：默认只带注册信息；数据压缩包体积大，另开关
    if with_datasets and (DATA / "datasets").is_dir():
        skip = None if with_dataset_files else (
            lambda p: p.suffix.lower() in (".zip", ".tar", ".gz", ".7z", ".rar"))
        n = _copy_tree(DATA / "datasets", out / "datasets", pairs, skip=skip)
        print(f"[5/5] datasets/ 已导出（{'含数据压缩包' if with_dataset_files else '仅注册信息'}，改写 {n} 个文本文件）")
    else:
        print("[5/5] 跳过 datasets/（未指定 --with-datasets）")

    # 自带导入器：让交接包不依赖本仓库（对方无需本脚本也能导入）
    shutil.copy2(Path(__file__).resolve().parent / "_data_import.py", out / "import_data.py")
    (out / "README_导入说明.md").write_text(README, encoding="utf-8")

    if with_zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=str(out.parent), base_dir=out.name)
        print(f"\n已打包：{archive}（{Path(archive).stat().st_size / 1024 / 1024:.1f} MB）")
    print(f"交接包目录：{out}")
    return 0


def do_import(src: Path, target_root: Path, force: bool) -> int:
    """委托给自带导入器（`_data_import.py`，同一份逻辑也会被复制进交接包）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _data_import import import_package
    return import_package(src, target_root, force)


def main() -> int:
    ap = argparse.ArgumentParser(description="数据交接：导出/导入可交接部分并改写绝对路径")
    ap.add_argument("--out", default="data_export", help="导出目录（默认 data_export）")
    ap.add_argument("--import", dest="import_dir", default=None, metavar="DIR",
                    help="导入模式：指定交接包目录")
    ap.add_argument("--root", default=None, help="导入时的目标仓库根（默认取本脚本所在仓库的根）")
    ap.add_argument("--zip", action="store_true", help="导出后打成 zip")
    ap.add_argument("--with-papers", action="store_true",
                    help="附带论文（解析出的 markdown/表格/图片；默认不含 PDF 原件）")
    ap.add_argument("--with-pdfs", action="store_true", help="论文附带 PDF 原件（体积大，隐含 --with-papers）")
    ap.add_argument("--with-datasets", action="store_true",
                    help="附带数据集（默认只含注册信息）")
    ap.add_argument("--with-dataset-files", action="store_true",
                    help="数据集附带数据压缩包（体积大，隐含 --with-datasets）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出/数据")
    args = ap.parse_args()

    if args.import_dir:
        root = Path(args.root).resolve() if args.root else ROOT
        return do_import(Path(args.import_dir).resolve(), root, args.force)
    return do_export(Path(args.out).resolve(), args.zip, args.force,
                     args.with_papers or args.with_pdfs, args.with_pdfs,
                     args.with_datasets or args.with_dataset_files, args.with_dataset_files)


if __name__ == "__main__":
    sys.exit(main())
