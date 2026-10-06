"""把某个项目独立环境里的 torch **就地换成 CUDA 版**（不重建环境）。

为什么需要它：项目环境由 `env_manager` 创建。早期版本在「依赖清单没声明 CUDA」时按默认索引安装，
而 **Windows 上默认索引（PyPI/镜像）给的 `torch` 就是 CPU 版** → 环境永远拿不到 GPU 轮子
（实测 scGPT 环境装出 `torch 2.14.1+cpu`）。重建环境会走新的 `env_manager.plan_cuda`（有 NVIDIA
驱动就装 CUDA 轮子）自动修好，但重建要把整份依赖清单重装一遍；本脚本只**换 torch 这一个 build**。

选索引/版本的逻辑**与建环境共用一份**（`env_manager.plan_cuda` / `preinstall_torch_cmd`），
所以这里做什么、建环境时就做什么。

用法（仓库根，用后端解释器跑）:
    python scripts/install_cuda_torch.py --project-id <原始项目id>
    python scripts/install_cuda_torch.py --project-id <id> --find-links https://mirror.sjtu.edu.cn/pytorch-wheels/cu126
    python scripts/install_cuda_torch.py --project-id <id> --dry-run      # 只看要跑什么命令

镜像（国内推荐）：官方 CDN 下 2.5GB 的 wheel 极易 `ReadTimeoutError`（2026-10-06 实测失败），
可传 `--find-links`（扁平目录，不能当 `--index-url`）或设 `ENV_PYTORCH_FIND_LINKS`。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", required=True, help="原始项目 id（用它的独立环境）")
    ap.add_argument("--find-links", action="append", default=[],
                    help="CUDA 轮子镜像（扁平目录 URL，可重复）；覆盖 plan 里的镜像列表")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不真装")
    args = ap.parse_args()

    from app.services import analysis_service, env_manager, project_manager

    project = project_manager.get_project(args.project_id)
    ws = Path(project["workspace_path"])
    source = ws / "source"
    import os

    python = analysis_service._project_python(ws)
    if python is None:
        print(f"[FAIL] 项目 {args.project_id} 没有独立环境解释器（先建环境）", file=sys.stderr)
        return 2
    pip_exe = str(Path(python).parent / "pip.exe")
    if not Path(pip_exe).exists():
        pip_exe = python                      # 退而用 `python -m pip` 之外的兜底（少见）
    print(f"项目环境解释器: {python}")

    driver = env_manager._detect_cuda()
    plan = env_manager.plan_cuda(source, {"cuda": driver})
    if args.find_links:
        plan = {**plan, "find_links": [u.rstrip("/") + "/" for u in args.find_links]}
        if plan.get("action") != "cuda_wheel":
            # 显式给了镜像就按 CUDA 走（用户明确要求），索引取候选里驱动支持的最高档
            index, why = env_manager._pick_cuda_index(driver)
            if index is None:
                print(f"[FAIL] {why}", file=sys.stderr)
                return 1
            plan = {**plan, "action": "cuda_wheel", "index": index}
    print("判断:", plan.get("action"), "|", plan.get("reason"))
    if plan.get("action") != "cuda_wheel":
        print("[warn] 当前判定不是「装 CUDA 版 torch」，无事可做（可用 --find-links 强制走镜像）")
        return 0

    main_index = env_manager.PIP_INDEX_URL
    # 配了镜像就必须钉住镜像里的版本号，否则主索引上版本号更高的 CPU 版会赢
    find_links = [str(u) for u in (plan.get("find_links") or [])]
    torch_pin = torch_link = None
    if find_links:
        resolved = env_manager.resolve_mirror_torch_pin(python, find_links)
        if resolved:
            torch_pin, torch_link = resolved
        print("镜像版本钉:", torch_pin or "（镜像里没找到匹配本解释器/平台的 cu 轮子）")
        if torch_link:
            print("镜像 wheel 页:", torch_link)
    cmd = env_manager.preinstall_torch_cmd(pip_exe, plan, main_index,
                                           torch_pin=torch_pin, torch_link=torch_link)
    print("命令:", " ".join(cmd))
    if args.dry_run:
        return 0
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[FAIL] pip 安装失败（rc={rc}）——若为下载超时，换国内镜像重试："
              f"  --find-links https://mirror.sjtu.edu.cn/pytorch-wheels/<cuTag>", file=sys.stderr)
        return rc

    check = subprocess.run(
        [python, "-c",
         "import torch;print('torch', torch.__version__, '| cuda 可用:', torch.cuda.is_available());"
         "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '（不可用）')"],
        capture_output=True, text=True)
    print(check.stdout.strip() or check.stderr.strip())
    if "cuda 可用: True" not in check.stdout:
        print("[FAIL] 装完仍不可用 CUDA：检查驱动/轮子是否匹配", file=sys.stderr)
        return 1
    print("[OK] 该环境的 torch 已换成 CUDA 版")
    return 0


if __name__ == "__main__":
    sys.exit(main())
