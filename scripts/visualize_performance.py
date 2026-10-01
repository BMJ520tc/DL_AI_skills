"""性能对比图（分组柱状）：模块三 5.5 固定脚本。

用法: python visualize_performance.py <input.json> <out.html> <echarts.min.js>

输入 JSON:
{
  "title": "性能对比",
  "metrics": ["accuracy", "f1"],
  "series": [{"name": "baseline", "values": {"accuracy": 0.79, "f1": 0.79}},
             {"name": "public-pub", "values": {"accuracy": 0.20, "f1": 0.20}}]
}
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
    legend: { data: DATA.series.map(s => s.name) },
    grid: { left: 60, right: 24, top: 48, bottom: 48 },
    xAxis: { type: 'category', data: DATA.metrics },
    yAxis: { type: 'value' },
    series: DATA.series.map(s => ({
      name: s.name,
      type: 'bar',
      label: { show: true, position: 'top' },
      data: DATA.metrics.map(m => s.values[m] !== undefined ? s.values[m] : null)
    }))
  });
  window.addEventListener('resize', () => chart.resize());
}
"""


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: python visualize_performance.py <input.json> <out.html> <echarts.min.js>", file=sys.stderr)
        return 2
    src, out, echarts = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    payload = json.loads(src.read_text(encoding="utf-8"))

    title = payload.get("title") or "性能对比图"
    metrics = payload.get("metrics") or []
    series = payload.get("series") or []
    meta = f"指标：{', '.join(metrics)}　|　对比对象：{', '.join(s.get('name','') for s in series)}"

    if not series or not metrics:
        render(out, title=title, echarts_path=echarts, extra_html="", meta=meta,
               degrade="缺少 metrics 或 series 字段（需基准与评估的指标结果）")
    else:
        render(out, title=title, echarts_path=echarts, meta=meta,
               data={"metrics": metrics, "series": series}, init_js=INIT)
    print(json.dumps({"status": "ok", "html": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
