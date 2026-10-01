"""图片/压缩包预处理：类别与划分推导、产物路径可持续（真实数据集暴露的缺陷回归）。

背景（2026-10-01 E1 用真实 Zenodo 数据集暴露）：
1. `<root>/train/<class>/*.jpg` 结构下，标签被取成目录第一层 `train`/`test`（1400 行全错）；
2. 压缩包解压到临时目录，运行结束即删除，统一 CSV 里的 `input` 全部指向已失效路径。
"""
from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import preprocess_dataset as pp  # noqa: E402


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(path)


def _read_rows(out_dir: Path) -> list[dict]:
    with (out_dir / "preprocessed.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture()
def cat_dog_zip(tmp_path) -> Path:
    src = tmp_path / "src"
    for split in ("train", "test"):
        for cls in ("cat", "dog"):
            for i in range(2):
                _png(src / split / cls / f"{cls}.{i}.jpg")
    zip_path = tmp_path / "cats_dogs.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for p in sorted(src.rglob("*.jpg")):
            z.write(p, p.relative_to(src))
    return zip_path


def test_archive_split_dirs_are_not_labels(cat_dog_zip, tmp_path):
    out_dir = tmp_path / "out"
    summary = pp.run(cat_dog_zip, out_dir, "cats-dogs", "classification")

    assert summary["status"] == "ok"
    assert summary["format"] == "image_dir"
    assert summary["labels"] == ["cat", "dog"]          # 不是 ["test", "train"]
    assert summary["n_rows"] == 8

    rows = _read_rows(out_dir)
    assert {r["label"] for r in rows} == {"cat", "dog"}
    assert {r["split"] for r in rows} == {"train", "test"}   # 目录划分被还原成 split


def test_archive_images_are_kept_under_out_dir(cat_dog_zip, tmp_path):
    """产物路径必须在本次运行结束后仍可访问（不能指向已删除的临时目录）。"""
    out_dir = tmp_path / "out"
    pp.run(cat_dog_zip, out_dir, "cats-dogs", "classification")

    for row in _read_rows(out_dir):
        path = Path(row["input"])
        assert path.is_file(), f"input 指向不存在的文件: {path}"
        assert out_dir in path.parents, f"图片未保存在产出目录内: {path}"


def test_flat_image_dir_falls_back_to_filename(tmp_path):
    src = tmp_path / "flat"
    _png(src / "dog.1.jpg")
    _png(src / "cat.2.jpg")
    summary = pp.run(src, tmp_path / "out", "flat", "classification")
    assert summary["labels"] == ["cat", "dog"]


def test_class_dir_before_split_dir(tmp_path):
    """<root>/<class>/<split>/*.jpg 顺序也应正确解析。"""
    src = tmp_path / "ordered"
    _png(src / "cat" / "train" / "a.jpg")
    _png(src / "dog" / "test" / "b.jpg")
    summary = pp.run(src, tmp_path / "out", "ordered", "classification")
    assert summary["labels"] == ["cat", "dog"]
    rows = _read_rows(tmp_path / "out")
    assert {r["split"] for r in rows} == {"train", "test"}


def test_nested_single_class_dir_without_split(tmp_path):
    """只有类别目录、没有划分目录时，split 保持默认（train）。"""
    src = tmp_path / "nested"
    _png(src / "wolf" / "a.jpg")
    summary = pp.run(src, tmp_path / "out", "nested", "classification")
    assert summary["labels"] == ["wolf"]
    assert {r["split"] for r in _read_rows(tmp_path / "out")} == {"train"}
