"""典型案例预测结果（表格 + 缩略图）：模块三 5.5 固定脚本。

用法: python visualize_cases.py <input.json> <out.html> <echarts.min.js>

输入 JSON:
{
  "title": "典型案例",
  "cases": [{"id": "3", "y_true": "cat", "y_pred": "dog", "prob": 0.62,
             "path": "img/3.png", "correct": false}]
}
缩略图：path 指向本地存在的图片时内嵌为 data-uri；否则只显示路径文本。
"""
import base64
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _viz_common import render  # noqa: E402

THUMB_MAX = 96


def _thumbnail(path_value, base_dir: Path) -> str | None:
    if not path_value:
        return None
    candidate = Path(str(path_value))
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        from PIL import Image

        with Image.open(candidate) as im:
            im.thumbnail((THUMB_MAX, THUMB_MAX))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _table(cases: list[dict], base_dir: Path) -> str:
    rows = []
    for case in cases:
        thumb = _thumbnail(case.get("path"), base_dir)
        preview = (f'<img class="thumb" src="{thumb}">' if thumb
                   else html.escape(str(case.get("path") or "-")))
        cls = "" if case.get("correct") else ' class="bad"'
        rows.append(
            f"<tr{cls}><td>{html.escape(str(case.get('id')))}</td>"
            f"<td>{html.escape(str(case.get('y_true')))}</td>"
            f"<td>{html.escape(str(case.get('y_pred')))}</td>"
            f"<td>{case.get('prob')}</td><td>{preview}</td></tr>"
        )
    return (
        "<table><thead><tr><th>id</th><th>真值</th><th>预测</th><th>置信度</th><th>输入</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: python visualize_cases.py <input.json> <out.html> <echarts.min.js>", file=sys.stderr)
        return 2
    src, out, echarts = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    payload = json.loads(src.read_text(encoding="utf-8"))

    title = payload.get("title") or "典型案例预测结果"
    cases = payload.get("cases") or []
    wrong = sum(1 for c in cases if not c.get("correct"))
    meta = f"案例数：{len(cases)}　|　其中误判：{wrong}"

    if not cases:
        render(out, title=title, echarts_path=echarts, meta=meta, extra_html="",
               degrade="缺少 cases 字段（需逐样本预测结果）")
    else:
        render(out, title=title, echarts_path=echarts, meta=meta, extra_html=_table(cases, src.parent))
    print(json.dumps({"status": "ok", "html": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
