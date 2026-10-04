"""典型案例预测结果（按数据集分组的表格 + 缩略图）：模块三 5.5 固定脚本。

用法: python visualize_cases.py <input.json> <out.html> <echarts.min.js>

输入 JSON（分组形态，覆盖本次对比参与评估的全部数据集）：
{
  "title": "典型案例预测结果（按数据集分组）",
  "datasets": [
    {"name": "baseline", "drawable": true,
     "per_class": {"cat": {"f1": 0.9, "support": 12}},
     "cases": [{"id": "3", "y_true": "cat", "y_pred": "dog", "prob": 0.62,
                "path": "img/3.png", "correct": false}]},
    {"name": "public-pub", "drawable": false, "reason": "缺 predictions，只有聚合指标"}
  ]
}

- 每个可绘数据集一张案例表（表头带数据集名，行内也有「数据集」列），
  `per_class` 有值时附按类别的指标汇总（案例挑选已按类别分组）；
- 不可绘的数据集在总览表与提示条里**如实标注**，不静默留白。
兼容旧单数据集载荷（顶层 `cases`）。
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


def _datasets(payload: dict) -> list[dict]:
    """取数据集列表；兼容旧单数据集载荷（顶层 cases）。"""
    datasets = payload.get("datasets")
    if isinstance(datasets, list):
        return [d for d in datasets if isinstance(d, dict)]
    if payload.get("cases"):
        return [{
            "name": payload.get("dataset") or payload.get("name") or "baseline",
            "drawable": True,
            "cases": payload["cases"],
            "per_class": payload.get("per_class") or {},
        }]
    return []


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


def _table(cases: list[dict], base_dir: Path, dataset_name: str) -> str:
    rows = []
    for case in cases:
        thumb = _thumbnail(case.get("path"), base_dir)
        preview = (f'<img class="thumb" src="{thumb}">' if thumb
                   else html.escape(str(case.get("path") or "-")))
        cls = "" if case.get("correct") else ' class="bad"'
        rows.append(
            f"<tr{cls}><td>{html.escape(dataset_name)}</td>"
            f"<td>{html.escape(str(case.get('id')))}</td>"
            f"<td>{html.escape(str(case.get('y_true')))}</td>"
            f"<td>{html.escape(str(case.get('y_pred')))}</td>"
            f"<td>{html.escape(str(case.get('prob')))}</td><td>{preview}</td></tr>"
        )
    return (
        "<table><thead><tr><th>数据集</th><th>id</th><th>真值</th><th>预测</th>"
        "<th>置信度</th><th>输入</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _per_class_table(per_class: dict) -> str:
    """按类别的指标汇总（per_class 的用法：案例挑选的分组依据与可复核明细）。"""
    rows = []
    metric_keys = sorted({k for v in per_class.values() if isinstance(v, dict) for k in v})
    for label in sorted(per_class, key=str):
        metrics = per_class[label]
        metrics = metrics if isinstance(metrics, dict) else {"value": metrics}
        cells = "".join(f"<td>{html.escape(str(metrics.get(k, '-')))}</td>" for k in metric_keys)
        rows.append(f"<tr><td>{html.escape(str(label))}</td>{cells}</tr>")
    header = "".join(f"<th>{html.escape(k)}</th>" for k in metric_keys)
    return ("<table><thead><tr><th>类别</th>" + header + "</tr></thead><tbody>" +
            "".join(rows) + "</tbody></table>")


def _overview_table(datasets: list[dict], drawable: list[dict]) -> str:
    rows = []
    for d in datasets:
        ok = d in drawable
        note = d.get("reason") if not ok else f"{len(d.get('cases') or [])} 条"
        rows.append(
            f"<tr><td>{html.escape(str(d.get('name')))}</td>"
            f"<td>{'可绘' if ok else '<b>不可绘</b>'}</td>"
            f"<td>{html.escape(str(note or '-'))}</td></tr>"
        )
    return ("<h3>数据集一览</h3><table><thead><tr><th>数据集</th><th>是否可绘</th>"
            "<th>说明</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: python visualize_cases.py <input.json> <out.html> <echarts.min.js>", file=sys.stderr)
        return 2
    src, out, echarts = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    payload = json.loads(src.read_text(encoding="utf-8"))

    title = payload.get("title") or "典型案例预测结果"
    datasets = _datasets(payload)
    drawable = [d for d in datasets if d.get("drawable") and d.get("cases")]
    undrawable = [d for d in datasets if d not in drawable]
    total = sum(len(d.get("cases") or []) for d in drawable)
    wrong = sum(1 for d in drawable for c in (d.get("cases") or []) if not c.get("correct"))
    if datasets:
        parts = [
            (f"{d.get('name')}（{len(d.get('cases') or [])} 条）" if d in drawable
             else f"{d.get('name')}（不可绘）")
            for d in datasets
        ]
        meta = ("对比数据集：" + "　|　".join(parts)
                + f"　|　案例数：{total}　|　其中误判：{wrong}")
    else:
        meta = ""

    if not datasets:
        render(out, title=title, echarts_path=echarts, meta=meta, extra_html="",
               degrade="缺少 datasets 字段（需各数据集的逐样本预测结果）")
    elif not drawable:
        reasons = "；".join(f"{d.get('name')}：{d.get('reason') or '缺少逐样本预测'}" for d in datasets)
        render(out, title=title, echarts_path=echarts, meta=meta, extra_html=_overview_table(datasets, drawable),
               degrade=f"本次对比的数据集都不可绘（{reasons}）")
    else:
        body = _overview_table(datasets, drawable)
        if undrawable:
            items = "".join(
                f"<li>{html.escape(str(d.get('name')))}：{html.escape(str(d.get('reason') or '缺少逐样本预测'))}</li>"
                for d in undrawable
            )
            body = ('<div class="banner">以下数据集缺逐样本预测，已如实标注为不可绘'
                    f"（不留白、也不用代理值顶替）：<ul>{items}</ul></div>") + body
        for d in drawable:
            name = html.escape(str(d.get("name")))
            body += f"<h3>数据集：{name}</h3>"
            per_class = d.get("per_class") or {}
            if per_class:
                body += f"<h3>数据集：{name} —— 按类别指标（per_class）</h3>"
                body += _per_class_table(per_class)
            body += _table(d.get("cases") or [], src.parent, name)
        render(out, title=title, echarts_path=echarts, meta=meta, extra_html=body)
    print(json.dumps({"status": "ok", "html": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
