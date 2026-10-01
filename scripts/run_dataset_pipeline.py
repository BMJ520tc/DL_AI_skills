"""模块三全链路驱动：把「原始模型使用」流程串成一次无人值守运行。

模块三各步骤本为后台任务，但 5.3 的对齐确认是**设计上的人工闸门**
（未确认的对齐不参与 5.4 结果对比）。本脚本以「调用方」身份驱动后端 HTTP API，
按策略代为确认对齐，把（预处理 →）基准 → 对齐 →（确认对齐）→ 对比 → 可视化
串成一次端到端运行。

用法：

    python scripts/run_dataset_pipeline.py --project <project_id>
    python scripts/run_dataset_pipeline.py --project <id> --preprocess <原始数据路径>
    python scripts/run_dataset_pipeline.py --project <id> --dataset <dataset_id> [--dataset <id2>]
    python scripts/run_dataset_pipeline.py --project <id> --no-auto-confirm   # 停在闸门

    --no-auto-confirm : 停在对齐闸门，只列出待确认数据集（保持人工把关）
    --skip-visualize  : 不生成三张图
    --chart           : 指定图表，可重复（默认 performance/error_dist/cases 全出）

退出码：0 全成功；1 = 失败（含网络/接口错误）；2 = 参数错误（argparse）；3 = 停在人工闸门（--no-auto-confirm 且有待确认对齐）。依赖后端已启动。
"""
import argparse
import json
import sys

from _pipeline_common import ApiError, get as _get, request as _req, run_step as _run_step

DEFAULT_CHARTS = ("performance", "error_dist", "cases")


def _aligned_unconfirmed(base: str, project_id: str) -> list[dict]:
    """本项目中已对齐但未确认的数据集（5.3 闸门）。"""
    datasets = _get(base, "/api/knowledge/list?data_type=dataset&limit=200") or []
    out = []
    for ds in datasets:
        raw = ds.get("alignment")
        if not raw:
            continue
        try:
            alignment = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if project_id in (alignment.get("aligned_projects") or []) and alignment.get("status") != "confirmed":
            out.append(ds)
    return out


def run(args: argparse.Namespace) -> int:
    base = args.base_url.rstrip("/")
    project_id = args.project

    step = 1
    total = 4 + (1 if args.preprocess else 0)

    if args.preprocess:
        print(f"[{step}/{total}] preprocess（数据预处理）", flush=True)
        _run_step(base, "preprocess", "/api/preprocess", {
            "input_path": args.preprocess, "task_type": args.task_type, "project_id": project_id,
        }, args.timeout, args.interval)
        step += 1

    print(f"[{step}/{total}] baseline（自带数据基准运行）", flush=True)
    _run_step(base, "baseline", f"/api/projects/{project_id}/baseline", None, args.timeout, args.interval)
    step += 1

    print(f"[{step}/{total}] align（跨数据集对齐）", flush=True)
    for dataset_id in args.dataset:
        _run_step(base, "align", f"/api/projects/{project_id}/datasets/align", {"dataset_id": dataset_id},
                  args.timeout, args.interval)
    step += 1

    # 5.3 对齐确认闸门
    print(f"[{step}/{total}] 对齐确认闸门", flush=True)
    pending = _aligned_unconfirmed(base, project_id)
    if not pending:
        print("  无待确认对齐（若尚未对齐，请用 --dataset 指定数据集）")
    elif args.no_auto_confirm:
        for ds in pending:
            print(f"  待确认: {ds['dataset_id']}  [{ds.get('name')}]")
        print("--no-auto-confirm：停在闸门，未对比。")
        return 3
    else:
        for ds in pending:
            _req(base, "POST", f"/api/datasets/{ds['dataset_id']}/alignment/confirm")
        print(f"  已确认 {len(pending)} 个对齐")

    step += 1

    print(f"[{step}/{total}] compare（结果对比与使用建议）", flush=True)
    _run_step(base, "compare", f"/api/projects/{project_id}/compare", None, args.timeout, args.interval)

    if args.skip_visualize:
        return 0

    print("visualize（三张可视化图）", flush=True)
    for chart in args.chart:
        result = _req(base, "POST", f"/api/projects/{project_id}/visualize/{chart}")
        flag = "降级" if result.get("degraded") else "正常"
        print(f"  [{chart}] {flag} -> {result.get('html')}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="模块三全链路驱动（无人值守运行）")
    p.add_argument("--project", required=True, help="项目 id")
    p.add_argument("--preprocess", help="原始数据路径；提供则先执行 5.1 预处理")
    p.add_argument("--task-type", default="classification", help="预处理的任务类型（仅 --preprocess 时）")
    p.add_argument("--dataset", action="append", default=[], help="要对齐的数据集 id，可重复")
    p.add_argument("--chart", action="append", default=[], help=f"图表类型，可重复（默认 {DEFAULT_CHARTS}）")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--no-auto-confirm", action="store_true", help="停在对齐闸门（保持人工把关）")
    p.add_argument("--skip-visualize", action="store_true", help="不生成图表")
    p.add_argument("--timeout", type=int, default=7200, help="单步任务超时（秒，默认 ≥ 后端最长任务超时）")
    p.add_argument("--interval", type=float, default=3.0, help="轮询间隔（秒）")
    args = p.parse_args()
    if not args.chart:
        args.chart = list(DEFAULT_CHARTS)

    try:
        raise SystemExit(run(args))
    except ApiError as e:
        print(f"错误: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
