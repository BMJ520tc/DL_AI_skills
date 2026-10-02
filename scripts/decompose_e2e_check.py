"""模块四「不调用大模型」的端到端自检：真实项目环境跑 trace → 回填 → 再生成 → 两步验证。

为什么需要它：拆解里的 agent 步骤需要模型凭证，很多环境（CI、别人的机器、接手者刚上手时）
跑不了完整验收；但「IR 本身是否自洽」这一侧完全可以离线验证——用已经存在的 IR，
在项目独立环境里跑形状追踪、把结果回填、再生成代码、跑结构+数值两步比对。
R27（模块结构签名对 agent 可选参数敏感）就是用它复现并验收的；阶段4 改动补参/再生成/签名时
同样应跑它做回归。

用法（在仓库根目录）:
    python scripts/decompose_e2e_check.py --project-id <原始项目 id>
    python scripts/decompose_e2e_check.py --project-id <id> --ab     # 附带签名 A/B 回归
    python scripts/decompose_e2e_check.py --project-id <id> --out-dir D:/tmp/chk

判据:
    - 两步验证 overall=passed（结构比对 + 数值比对都过）→ 退出码 0
    - `--ab`：把「模型实际值与构造默认值不等价」的参数全部从 IR 里删掉（模拟 agent 漏写），
      重新回填后结构签名必须与原始 IR 相同（否则就是 R27 类问题复发）；无法按 module_path
      回填的节点会如实列出（这是已知残留局限，见《模块详细设计》六.6-5）。

产物写在 --out-dir（默认 data/_acceptance/decompose_e2e/<project_id>/），
**不修改**项目目录里的 reports/ir.json 与 reports/verification.json。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services import decompose_service as ds  # noqa: E402  (需在 sys.path 之后导入)
from app.services.ir_schema import ir_hash  # noqa: E402


def _env_with_numba_cache() -> dict:
    """与 proc_util.run_command 同口径：给项目子进程注入短路径 JIT 缓存（Windows 长路径坑）。"""
    env = {**os.environ}
    if os.name == "nt" and not env.get("NUMBA_CACHE_DIR"):
        cache = REPO_ROOT / "data" / "numba_cache"
        cache.mkdir(parents=True, exist_ok=True)
        env["NUMBA_CACHE_DIR"] = str(cache)
    return env


def _run(cmd: list[str], cwd: Path, timeout: int):
    return subprocess.run(
        [str(c) for c in cmd], cwd=str(cwd), env=_env_with_numba_cache(),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )


def _drop_optional_params(ir: dict, keys_by_path: dict) -> tuple[dict, list[str]]:
    """把 trace 里「与默认值不等价」的键从 IR 的 params 删掉（模拟 agent 漏写）。"""
    variant = copy.deepcopy(ir)
    touched: list[str] = []
    for n in variant["nodes"]:
        params = n.get("params")
        if not isinstance(params, dict):
            continue
        for k in (keys_by_path.get(n.get("module_path")) or ()):
            if k in params:
                params.pop(k)
                touched.append(f"{n['id']}.{k}")
    return variant, touched


def main() -> int:
    ap = argparse.ArgumentParser(description="模块四无 agent 端到端自检（trace → 回填 → 再生成 → 两步验证）")
    ap.add_argument("--project-id", required=True, help="原始项目 id（data/projects/<id>）")
    ap.add_argument("--out-dir", default=None, help="产物目录（默认 data/_acceptance/decompose_e2e/<id>）")
    ap.add_argument("--ab", action="store_true", help="附带签名 A/B 回归（删掉非默认参数后签名必须不变）")
    ap.add_argument("--timeout", type=int, default=1800, help="单步子进程超时秒数（默认 1800）")
    args = ap.parse_args()

    ws = REPO_ROOT / "data" / "projects" / args.project_id
    ir_path = ws / "reports" / "ir.json"
    if not ir_path.exists():
        print(f"[失败] 找不到 IR：{ir_path}（该原始项目还没拆解过）")
        return 1
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "_acceptance" / "decompose_e2e" / args.project_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    python = ds.analysis_service._project_python(ws)
    if python is None:
        print(f"[失败] 项目环境未就绪：{ws / 'env'} 下找不到解释器（先跑模块一 env 创建）")
        return 1

    # 1) 形状/参数追踪（项目独立环境）
    ir_src = out_dir / "ir_source.json"
    ir_src.write_text(json.dumps(ir, ensure_ascii=False), encoding="utf-8")
    trace_out = out_dir / "trace.json"
    proc = _run([python, REPO_ROOT / "scripts" / "trace_shapes.py", ws / "source", ir_src, trace_out],
                out_dir, args.timeout)
    if proc.returncode != 0:
        print(f"[失败] trace 退出码 {proc.returncode}\n{(proc.stderr or proc.stdout)[-1500:]}")
        return 1
    trace = json.loads(trace_out.read_text(encoding="utf-8"))
    print(f"[1/3] trace 成功：{len(trace.get('shapes', {}))} 个模块路径，"
          f"{len(trace.get('params_delta', {}))} 个含非默认参数")

    # 2) 回填 + 再生成（用工作副本，不动项目里的 ir.json）
    merged = copy.deepcopy(ir)
    before_hash = ir_hash(merged)
    ds._merge_shapes(merged, copy.deepcopy(trace))
    merged_path = out_dir / "ir_merged.json"
    merged_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    code = ds.ir_codegen.generate(merged)
    regen = out_dir / "regenerated.py"
    regen.write_text(code, encoding="utf-8")
    sig = ds._module_signature(merged)
    print(f"[2/3] 回填完成：签名 mod_{sig}；代码 {len(code.splitlines())} 行；"
          f"ir_hash {'变化' if ir_hash(merged) != before_hash else '不变'}")

    # 3) 两步验证（结构 + 数值）
    verification = out_dir / "verification.json"
    proc = _run([python, REPO_ROOT / "scripts" / "verify_decompose.py", ws / "source", merged_path,
                 regen, verification, ds.DECOMPOSE_NUM_SEEDS, str(ds.DECOMPOSE_NUM_RTOL),
                 str(ds.DECOMPOSE_NUM_ATOL)], out_dir, args.timeout)
    if proc.returncode != 0:
        print(f"[失败] verify 退出码 {proc.returncode}\n{(proc.stderr or proc.stdout)[-1500:]}")
        return 1
    result = json.loads(verification.read_text(encoding="utf-8"))
    st, num = result["structure"], result["numeric"]
    print(f"[3/3] 两步验证 overall={result['overall']}：结构 {'通过' if st['passed'] else '未通过'}"
          f"（带参数层 {st['layer_count']['original']}/{st['layer_count']['regenerated']}、"
          f"参数 {st['param_count']['original']}/{st['param_count']['regenerated']}）、"
          f"数值 {'通过' if num['passed'] else '未通过'}"
          f"（max_rel_err={max((p['max_rel_err'] for p in num['per_seed']), default=None)}）")
    if result.get("failure_reason"):
        print(f"      失败原因：{str(result['failure_reason'])[:400]}")

    exit_code = 0 if result["overall"] == "passed" else 1

    # 4) 可选：签名 A/B 回归（R27）
    if args.ab:
        keys_by_path = {p: tuple(d.keys()) for p, d in (trace.get("params_delta") or {}).items()}
        variant, touched = _drop_optional_params(ir, keys_by_path)
        ds._merge_shapes(variant, copy.deepcopy(trace))
        sig_b = ds._module_signature(variant)
        unmatched = sorted({n["id"] for n in variant["nodes"]
                            if n.get("kind") == "leaf" and not n.get("params_model")})
        same = sig_b == sig
        print(f"[A/B] 删掉 {len(touched)} 个非默认参数后签名 mod_{sig_b} → "
              f"{'相同（通过）' if same else '不同（R27 类问题）'}")
        if unmatched:
            print(f"      按 module_path 未能回填模型参数的 leaf 节点（已知残留局限）：{unmatched[:10]}")
        if not same:
            exit_code = 1

    print(f"产物目录：{out_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
