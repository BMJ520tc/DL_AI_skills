"""模块二全链路驱动：把论文复现流程串成一次无人值守运行（阶段2' 自动化）。

模块二各步骤本是后台任务（触发后自动跑完），但流程里有**设计上的人工确认闸门**
（4.2 条目确认）。本脚本以「调用方」身份驱动后端 HTTP API，按策略代为确认，
从而把 parse → extract →（确认条目）→ reproduce → conclusion 串成一次端到端运行。

用法：

    python scripts/run_paper_pipeline.py --paper <paper_id> --project <project_id>
    python scripts/run_paper_pipeline.py --paper <paper_id> --repo <仓库路径或URL>   # 顺带建项目+环境
    python scripts/run_paper_pipeline.py --paper <paper_id> --project <id> --no-auto-confirm

    --no-auto-confirm : 停在闸门，只列出待确认条目（保持人工把关）
    --skip-reproduce  : 抽取（并确认）后即止，不建环境/不复现

退出码：0 全成功；1 = 失败（含网络/接口错误）；2 = 参数错误（argparse）；3 = 停在人工闸门（--no-auto-confirm 且有未确认条目）。

依赖后端已启动（默认 http://127.0.0.1:8000，--base-url 覆盖）。
"""
import argparse
import sys

from _pipeline_common import ApiError, request as _req, run_step as _run_step


def run(args: argparse.Namespace) -> int:
    base = args.base_url.rstrip("/")

    # 可选：建项目 + 载入源码 + 建环境（提供 --repo 时）
    project_id = args.project
    if args.repo:
        print("[1/6] 建项目并载入源码", flush=True)
        created = _req(base, "POST", "/api/projects", {
            "project_type": "original", "source": "local",
            "name": args.name or f"pipeline-{args.paper}", "source_url": args.repo,
        })
        project_id = created["project_id"]
        print(f"  project_id={project_id}")
        _run_step(base, "env", f"/api/projects/{project_id}/env", None, args.timeout, args.interval)
    elif project_id:
        print(f"[1/6] 使用既有项目 {project_id}")

    # 4.1 PDF → markdown
    print("[2/6] parse（PDF → markdown）", flush=True)
    _run_step(base, "parse", f"/api/papers/{args.paper}/parse", None, args.timeout, args.interval)

    # 4.2 实验条目抽取
    print("[3/6] extract（实验条目抽取）", flush=True)
    _run_step(base, "extract", f"/api/papers/{args.paper}/extract", None, args.timeout, args.interval)

    items = _req(base, "GET", f"/api/papers/{args.paper}/items") or []
    print(f"  抽出条目 {len(items)} 条")
    if not items:
        print("无实验条目（4.2 异常与边界：流程正常结束）")
        return 0

    # 确认闸门（4.2）
    print("[4/6] 条目确认闸门", flush=True)
    if args.no_auto_confirm:
        for it in items:
            if it.get("status") != "confirmed":
                print(f"  待确认: {it['item_id']}  [{it.get('metric_name')}]")
        print("--no-auto-confirm：停在闸门，未复现。")
        return 3
    for it in items:
        if it.get("status") != "confirmed":
            _req(base, "POST", f"/api/papers/{args.paper}/items/{it['item_id']}/confirm")
    confirmed = [it for it in (_req(base, "GET", f"/api/papers/{args.paper}/items") or []) if it.get("status") == "confirmed"]
    print(f"  已确认 {len(confirmed)}/{len(items)} 条")

    if args.skip_reproduce or not project_id:
        if not project_id:
            print("未提供 --project/--repo，跳过复现与结论。")
        return 0

    # 4.3 自动复现 + 4.4 逐条对照与可信度结论
    print("[5/6] reproduce（自动复现）", flush=True)
    _run_step(base, "reproduce", f"/api/papers/{args.paper}/reproduce", {"project_id": project_id},
              args.timeout, args.interval)

    print("[6/6] conclusion（逐条对照与可信度结论）", flush=True)
    _run_step(base, "conclusion", f"/api/papers/{args.paper}/conclusion", None, args.timeout, args.interval)

    conclusion = _req(base, "GET", f"/api/papers/{args.paper}/conclusion") or {}
    results = _req(base, "GET", f"/api/papers/{args.paper}/reproduce") or []
    print("\n=== 结果 ===")
    for r in results:
        print(f"  {r.get('metric_name')}: 报告={r.get('metric_value_reported')} "
              f"实际={r.get('metric_value_actual')} -> {r.get('verdict')}")
    print(f"  总体可信度: {conclusion.get('overall_verdict')}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="模块二全链路驱动（无人值守运行）")
    p.add_argument("--paper", required=True, help="论文 id")
    p.add_argument("--project", help="既有项目 id（提供模块一环境）")
    p.add_argument("--repo", help="仓库路径或 URL；提供则新建项目并载入源码、建环境")
    p.add_argument("--name", help="新建项目名")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--no-auto-confirm", action="store_true", help="停在确认闸门（保持人工把关）")
    p.add_argument("--skip-reproduce", action="store_true", help="抽取并确认后即止")
    p.add_argument("--timeout", type=int, default=7200, help="单步任务超时（秒，默认 ≥ 后端最长任务超时）")
    p.add_argument("--interval", type=float, default=3.0, help="轮询间隔（秒）")
    args = p.parse_args()

    if not args.project and not args.repo and not args.skip_reproduce:
        print("提示：未提供 --project/--repo，将在抽取（并确认）后停止，不复现。", file=sys.stderr)

    try:
        raise SystemExit(run(args))
    except ApiError as e:
        print(f"错误: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
