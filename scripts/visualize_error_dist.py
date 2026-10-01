"""误差分布图（直方图 + 散点）：模块三 5.5 固定脚本。

用法: python visualize_error_dist.py <input.json> <out.html> <echarts.min.js>

输入 JSON:
{
  "title": "误差分布",
  "values": [0.1, 0.4, ...],                       # 每样本误差，用于直方图
  "bins": 10,
  "scatter": [{"x": 1.0, "y": 1.2}, ...]           # 可选：真值-预测散点（回归类）
}
散点缺失时只出直方图（不作为降级）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _viz_common import render  # noqa: E402

INIT = """
if (typeof echarts !== 'undefined') {
  const chart = echarts.init(document.getElementById('chart'));
  chart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { left: 60, right: 24, top: 32, bottom: 48 },
    xAxis: { type: 'category', data: DATA.hist.map(b => b.label), name: '误差区间' },
    yAxis: { type: 'value', name: '样本数' },
    series: [{ type: 'bar', data: DATA.hist.map(b => b.count) }]
  });
  window.addEventListener('resize', () => chart.resize());

  if (DATA.scatter && DATA.scatter.length) {
    const c2 = echarts.init(document.getElementById('chart2'));
    c2.setOption({
      tooltip: { formatter: p => '真值 ' + p.value[0] + ' / 预测 ' + p.value[1] },
      grid: { left: 60, right: 24, top: 32, bottom: 48 },
      xAxis: { type: 'value', name: '真值' },
      yAxis: { type: 'value', name: '预测' },
      series: [{ type: 'scatter', symbolSize: 8, data: DATA.scatter.map(p => [p.x, p.y]) }]
    });
    window.addEventListener('resize', () => c2.resize());
  }
}
"""

EXTRA_SCATTER = '<h3>真值-预测散点</h3><div id="chart2"></div>'


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


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: python visualize_error_dist.py <input.json> <out.html> <echarts.min.js>", file=sys.stderr)
        return 2
    src, out, echarts = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    payload = json.loads(src.read_text(encoding="utf-8"))

    title = payload.get("title") or "误差分布图"
    values = [float(v) for v in (payload.get("values") or []) if isinstance(v, (int, float))]
    scatter = payload.get("scatter") or []
    meta = f"样本数：{len(values)}　|　直方图分箱：{payload.get('bins') or 10}"

    if not values:
        render(out, title=title, echarts_path=echarts, meta=meta, extra_html="",
               degrade="缺少 values 字段（需逐样本误差或预测置信度）")
    else:
        hist = _histogram(values, int(payload.get("bins") or 10))
        extra = EXTRA_SCATTER if scatter else ""
        render(out, title=title, echarts_path=echarts, meta=meta, extra_html=extra,
               data={"hist": hist, "scatter": scatter}, init_js=INIT)
    print(json.dumps({"status": "ok", "html": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
