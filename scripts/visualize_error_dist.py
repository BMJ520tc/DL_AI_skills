"""误差分布图（按数据集分组的直方图 + 共用散点）：模块三 5.5 固定脚本。

用法: python visualize_error_dist.py <input.json> <out.html> <echarts.min.js>

输入 JSON（分组形态，覆盖本次对比参与评估的全部数据集）：
{
  "title": "误差分布图（按数据集分组）",
  "datasets": [
    {"name": "baseline", "drawable": true, "unit": "绝对误差", "proxy": false,
     "values": [0.1, 0.4, ...], "bins": 10,
     "scatter": [{"x": 1.0, "y": 1.2}, ...],
     "class_breakdown": [{"label": "cat", "n": 3, "mean_error": 0.2,
                          "metrics": {"f1": 0.9}}]},
    {"name": "public-pub", "drawable": false, "reason": "缺 predictions，只有聚合指标"}
  ]
}

- 每个可绘数据集一张独立直方图（标题带数据集名），散点按数据集分系列并带图例；
- 不可绘的数据集在总览表与提示条里**如实标注**，不静默留白、也不用代理值冒充；
- `unit` 为「1 - 预测置信度」时是分类任务的**误差代理**，脚本在标题与信息条上明确标注。
兼容旧单数据集载荷（顶层 `values` / `bins` / `scatter`）。
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _viz_common import render  # noqa: E402

INIT = """
if (typeof echarts !== 'undefined') {
  const PALETTE = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4'];
  (DATA.hists || []).forEach(function (h, i) {
    const el = document.getElementById('chart' + i);
    if (!el) return;
    const chart = echarts.init(el);
    chart.setOption({
      tooltip: { trigger: 'axis' },
      title: { text: h.name + '（' + h.unit + '）', left: 'center', textStyle: { fontSize: 13 } },
      grid: { left: 60, right: 24, top: 56, bottom: 48 },
      xAxis: { type: 'category', data: h.hist.map(function (b) { return b.label; }), name: '误差区间' },
      yAxis: { type: 'value', name: '样本数' },
      series: [{ type: 'bar', name: h.name, itemStyle: { color: PALETTE[i % PALETTE.length] },
                 data: h.hist.map(function (b) { return b.count; }) }]
    });
    window.addEventListener('resize', function () { chart.resize(); });
  });
  if (DATA.scatter && DATA.scatter.length) {
    const c2 = echarts.init(document.getElementById('scatter'));
    c2.setOption({
      tooltip: { formatter: function (p) { return p.seriesName + ' 真值 ' + p.value[0] + ' / 预测 ' + p.value[1]; } },
      legend: { data: DATA.scatter.map(function (s) { return s.name; }) },
      grid: { left: 60, right: 24, top: 48, bottom: 48 },
      xAxis: { type: 'value', name: '真值' },
      yAxis: { type: 'value', name: '预测' },
      series: DATA.scatter.map(function (s, i) {
        return { name: s.name, type: 'scatter', symbolSize: 8, data: s.points,
                 itemStyle: { color: PALETTE[i % PALETTE.length] } };
      })
    });
    window.addEventListener('resize', function () { c2.resize(); });
  }
}
"""


def _datasets(payload: dict) -> list[dict]:
    """取数据集列表；兼容旧单数据集载荷（顶层 values/bins/scatter）。"""
    datasets = payload.get("datasets")
    if isinstance(datasets, list):
        return [d for d in datasets if isinstance(d, dict)]
    if payload.get("values"):
        return [{
            "name": payload.get("dataset") or payload.get("name") or "baseline",
            "drawable": True,
            "unit": payload.get("unit") or "绝对误差",
            "values": payload["values"],
            "bins": payload.get("bins") or 10,
            "scatter": payload.get("scatter") or [],
            "per_class": payload.get("per_class") or {},
            "proxy": bool(payload.get("proxy")),
        }]
    return []


def _histogram(values: list[float], bins: int) -> list[dict]:
    lo, hi = min(values), max(values)
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    buckets = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        buckets[idx] += 1
    return [
        {"label": f"{lo + i * width:.3g}~{lo + (i + 1) * width:.3g}", "count": buckets[i]}
        for i in range(bins)
    ]


def _overview_table(datasets: list[dict], drawable: list[dict]) -> str:
    rows = []
    for d in datasets:
        ok = d in drawable
        unit = html.escape(str(d.get("unit") or "-"))
        if ok and d.get("proxy"):
            unit += "（代理，非真实误差）"
        note = d.get("reason") if not ok else (d.get("note") or "-")
        rows.append(
            f"<tr><td>{html.escape(str(d.get('name')))}</td>"
            f"<td>{'可绘' if ok else '<b>不可绘</b>'}</td>"
            f"<td>{unit}</td><td>{html.escape(str(note or '-'))}</td></tr>"
        )
    return ("<h3>数据集一览</h3><table><thead><tr><th>数据集</th><th>是否可绘</th>"
            "<th>误差口径</th><th>说明</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _class_table(rows: list[dict]) -> str:
    metric_keys = sorted({k for r in rows for k in (r.get("metrics") or {})})
    header = "".join(f"<th>{html.escape(k)}</th>" for k in metric_keys)
    body = []
    for r in rows:
        metrics = r.get("metrics") or {}
        cells = "".join(f"<td>{html.escape(str(metrics.get(k, '-')))}</td>" for k in metric_keys)
        mean = "—" if r.get("mean_error") is None else f"{r['mean_error']:g}"
        body.append(f"<tr><td>{html.escape(str(r.get('label')))}</td><td>{r.get('n')}</td>"
                    f"<td>{mean}</td>{cells}</tr>")
    return ("<table><thead><tr><th>类别</th><th>样本数</th><th>平均误差</th>" + header +
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: python visualize_error_dist.py <input.json> <out.html> <echarts.min.js>", file=sys.stderr)
        return 2
    src, out, echarts = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    payload = json.loads(src.read_text(encoding="utf-8"))

    title = payload.get("title") or "误差分布图"
    datasets = _datasets(payload)
    drawable = [d for d in datasets if d.get("drawable") and d.get("values")]
    undrawable = [d for d in datasets if d not in drawable]

    if datasets:
        meta = "对比数据集：" + "　|　".join(
            f"{d.get('name')}（{d.get('unit') if d in drawable else '不可绘'}）" for d in datasets
        )
    else:
        meta = ""

    if not datasets:
        render(out, title=title, echarts_path=echarts, meta=meta, extra_html="",
               degrade="缺少 datasets 字段（需各数据集的逐样本误差或预测置信度）")
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
        data_hists = []
        for i, d in enumerate(drawable):
            unit = html.escape(str(d.get("unit") or "误差"))
            tag = "（代理：1 - 预测置信度，非真实误差）" if d.get("proxy") else ""
            body += (f"<h3>数据集：{html.escape(str(d.get('name')))} —— {unit}{tag}</h3>"
                     f'<div id="chart{i}" style="width:100%;height:340px"></div>')
            if d.get("note"):
                body += f'<div class="meta">{html.escape(str(d["note"]))}</div>'
            breakdown = d.get("class_breakdown") or []
            if breakdown:
                body += f"<h3>数据集：{html.escape(str(d.get('name')))} —— 按类别分组</h3>"
                body += _class_table(breakdown)
            numeric_values = [float(v) for v in d["values"] if isinstance(v, (int, float))]
            data_hists.append({
                "name": d.get("name"), "unit": d.get("unit") or "误差",
                "proxy": bool(d.get("proxy")),
                "hist": _histogram(numeric_values, int(d.get("bins") or 10)),
            })
        data_scatter = [{"name": d.get("name"),
                         "points": [[p.get("x"), p.get("y")] for p in (d.get("scatter") or [])]}
                        for d in drawable if d.get("scatter")]
        if data_scatter:
            body += '<h3>真值-预测散点（按数据集区分）</h3><div id="scatter" style="width:100%;height:420px"></div>'
        render(out, title=title, echarts_path=echarts, meta=meta, body_html=body,
               data={"hists": data_hists, "scatter": data_scatter}, init_js=INIT)
    print(json.dumps({"status": "ok", "html": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
