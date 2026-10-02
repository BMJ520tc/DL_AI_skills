"""预处理脚本：FASTA/PDB 读取与图片归一（P6 回归）。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import preprocess_dataset as pp  # noqa: E402


def _read_rows(out_dir: Path) -> list[dict]:
    with (out_dir / "preprocessed.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_fasta_is_parsed_and_normalized(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">seq1 description\nacgtacgtacgt\nacgt\n>seq2\ntttttttttttt\n", encoding="utf-8")

    summary = pp.run(fasta, tmp_path / "out", "prot", "classification")
    assert summary["status"] == "ok" and summary["format"] == "fasta"
    assert summary["n_rows"] == 2
    # 序列列被识别并统一转大写（_looks_like_sequence 命中）
    assert summary["alignment"]["sequence"]["case"] == "upper"
    assert {r["input"] for r in _read_rows(tmp_path / "out")} == {"ACGTACGTACGTACGT", "TTTTTTTTTTTT"}


def test_pdb_atom_records_become_table(tmp_path):
    pdb = tmp_path / "atoms.pdb"
    pdb.write_text(
        "HEADER    TEST\n"
        "ATOM      1  N   ALA A   1      11.104  13.207   9.984  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      12.560  12.900  10.900  1.00 20.00           C\n",
        encoding="utf-8",
    )
    summary = pp.run(pdb, tmp_path / "out", "pdb", "classification")
    assert summary["status"] == "ok" and summary["format"] == "pdb"
    assert summary["n_rows"] == 2
    assert any(c.startswith("meta_") for c in summary["fields"])


def test_image_normalization_stats_land_in_alignment(tmp_path):
    src = tmp_path / "src" / "cat"
    src.mkdir(parents=True)
    Image.new("L", (8, 8), 5).save(src / "a.png")  # 灰度图 → 需转 RGB

    summary = pp.run(tmp_path / "src", tmp_path / "out", "gray", "classification", (4, 4))
    images = summary["alignment"]["images"]
    assert images["color_space"] == "RGB"
    assert images["converted_to_rgb"] == 1
    assert images["resized"] == 1 and images["target_size"] == [4, 4]

    out_img = Path(_read_rows(tmp_path / "out")[0]["input"])
    assert out_img.is_file()
    with Image.open(out_img) as im:
        assert im.mode == "RGB" and im.size == (4, 4)


def test_image_dir_default_does_not_resize(tmp_path):
    src = tmp_path / "src" / "dog"
    src.mkdir(parents=True)
    Image.new("RGB", (6, 5), (1, 2, 3)).save(src / "b.png")

    summary = pp.run(tmp_path / "src", tmp_path / "out", "rgb", "classification")
    images = summary["alignment"]["images"]
    assert images["resized"] == 0 and images["target_size"] is None
    # 本来就 RGB → 无需归一副本，input 仍指向原图
    assert summary["n_rows"] == 1


def test_legacy_xls_gives_clear_hint(tmp_path):
    legacy = tmp_path / "old.xls"
    legacy.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    summary = pp.run(legacy, tmp_path / "out", "old", "classification")
    assert summary["status"] == "unsupported"
    assert ".xlsx" in summary["message"]
