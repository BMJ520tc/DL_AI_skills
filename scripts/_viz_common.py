"""三个可视化脚本共用的自包含 HTML 渲染壳（模块详细设计 5.5，D4/D7）。

产出的 HTML 内嵌 ECharts 离线单文件与数据，无任何外链，浏览器双击即可打开。
"""
import json
from pathlib import Path

_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2328; }}
  h2 {{ margin: 0 0 4px; }}
  .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 16px; }}
  .banner {{ background: #fff7e6; border: 1px solid #ffd591; color: #874d00;
             padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; }}
  #chart {{ width: 100%; height: 560px; }}
  #chart2 {{ width: 100%; height: 420px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 13px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; }}
  th {{ background: #f6f8fa; }}
  tr.bad td {{ background: #fff1f0; }}
  img.thumb {{ max-width: 96px; max-height: 96px; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div class="meta">{meta}</div>
{banner}
{body}
<script>const DATA = {data};</script>
<script>{echarts}</script>
<script>{init}</script>
</body>
</html>
"""


def _json_for_script(value) -> str:
    # 防止数据中出现 </script> 提前闭合脚本块
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render(
    out_html: Path,
    *,
    title: str,
    echarts_path: str,
    data=None,
    init_js: str = "",
    extra_html: str = "",
    degrade: str | None = None,
    meta: str = "",
) -> None:
    """写出自包含 HTML。degrade 非空时只输出降级提示与已有内容，不加载图表。"""
    # 仅在有图表脚本时才内联 ECharts（纯表格类页面无需 1MB 的图表库）
    echarts_src = ""
    if degrade is not None:
        init_js = ""
    elif init_js:
        ec = Path(echarts_path)
        if not ec.exists():
            raise RuntimeError(
                f"未找到 ECharts 离线文件: {ec}（应放置于 backend/app/vendor/echarts.min.js）"
            )
        echarts_src = ec.read_text(encoding="utf-8")

    banner = f'<div class="banner">数据缺失，该图降级：{degrade}</div>' if degrade else ""
    body = ('<div id="chart"></div>' if init_js else "") + extra_html

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        _PAGE.format(
            title=title, meta=meta, banner=banner, body=body,
            data=_json_for_script(data if data is not None else {}),
            echarts=echarts_src, init=init_js,
        ),
        encoding="utf-8",
    )
