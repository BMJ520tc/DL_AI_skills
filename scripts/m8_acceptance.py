"""M8 阶段6 联调验收驱动脚本：6.1 推荐路径（论文 → 画布训练成功）+ 6.2 多数据集与综合分析 + 6.3 自迭代闭环。

口径（2026-10-05，用户裁定 A：真实库 + 复用既有真实资产）：

  推荐路径（方案七）：真实论文 → 代码仓库加载 → 建真实独立环境 → 结构分析报告 →
  复现实验（agent）→ 可信度结论 → 拆解入库 → 画布拼装 → 真实 CPU 训练 → 版本留存。

  本脚本连**真实后端**（默认 :8000），逐环断言：
    - 论文/条目/项目/环境/结构报告/复现/结论/拆解/入库：断言**既有真实证据**
      （打印 run_record / reproduction_result / conclusion 等 id 指针）；
    - **画布拼装 + 训练 + 版本**：本脚本**执行**——用论文拆解入库得到的模块
      （读其 input/output_spec）在一个真实数据集上拼装并训练，得到第一条
      「论文 → … → 画布训练成功」的完整链。

  不重跑 agent 步骤（复现 49min、拆解需凭证）；`--rewalk-agent` 预留（需凭证，未实现）。

真实资产默认 = GEARS 链：论文 `gears` → 项目 `36f864f7…`（GEARS 仓库，含缓存）→
模块 `mod_f03f6c28d99bcec5`（论文模型拆解产物）。可用参数覆盖。

    D:\\python.exe scripts/m8_acceptance.py
    D:\\python.exe scripts/m8_acceptance.py --base http://127.0.0.1:8000 --paper gears
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "index.db"

# 默认真实资产（GEARS 链）
DEFAULT_PAPER = "gears"
DEFAULT_PROJECT = "36f864f72afc407e83ab824baca190cc"
DEFAULT_DATASET = "b210db13fe8f4db7aa02f42c480a8f03"   # realcase-chembl-freq-balanced（数值，1 维输入）
DEFAULT_MM_PROJECT = "8b3d6f0b236a477a96b8d923d4b65a8f"  # module3-fixture（baseline/eval/对齐/三图 真实资产）
DEFAULT_LOOP_PROJECT = "ed5870e8321649a8ba26dffcefdc1b4f"  # 真实训练用例（Linear 1→2），自迭代闭环项目
DEFAULT_LOOP_ENV = "36f864f72afc407e83ab824baca190cc"      # 带 torch 的真建环境
DEFAULT_LOOP_DATASET = "b210db13fe8f4db7aa02f42c480a8f03"  # realcase-chembl-freq-balanced

RESULTS: list[tuple[str, bool, str]] = []
KEEP_MM = False   # 6.2 多模型产物是否保留（默认自清，避免验收反复跑堆积）


def _cleanup_multi_model(knowledge_id: str, analysis_id: str) -> None:
    """删除本轮验收 6.2 多模型产生的 knowledge 行 + 索引 + 任务 + 报告文件。"""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("DELETE FROM knowledge WHERE knowledge_id=?", (knowledge_id,))
        conn.execute("DELETE FROM unified_index WHERE data_type='knowledge' AND ref_id=?", (knowledge_id,))
        conn.execute("DELETE FROM task WHERE task_id=?", (analysis_id,))
        conn.commit()
    finally:
        conn.close()
    rep = REPO_ROOT / "data" / "multi_model" / f"{analysis_id}.json"
    if rep.is_file():
        rep.unlink()


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 60):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                try:
                    return resp.status, (json.loads(raw) if raw else None)
                except json.JSONDecodeError:
                    return resp.status, raw.decode(errors="replace")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except Exception:
                return e.code, raw.decode(errors="replace")

    def ok(self, method: str, path: str, body: dict | None = None):
        st, payload = self.call(method, path, body)
        if st != 200 or isinstance(payload, str):
            raise RuntimeError(f"{method} {path} → HTTP {st}: {payload}")
        return payload

    def poll(self, task_id: str, label: str, timeout_s: int = 600) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            task = self.ok("GET", f"/api/tasks/{task_id}")
            if task.get("status") in ("success", "failed", "cancelled"):
                if task["status"] != "success":
                    raise RuntimeError(f"{label} 任务失败: {task.get('error')}")
                return task
            time.sleep(2)
        raise RuntimeError(f"{label} 任务超时（>{timeout_s}s）")


def _db():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _spec_dim(spec) -> int | None:
    """模块 input_spec/output_spec 取末维；API 可能回 JSON 字符串或已解析对象。"""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except Exception:
            return None
    if isinstance(spec, dict):
        shape = spec.get("shape")
        if isinstance(shape, list) and shape:
            try:
                return int(shape[-1])
            except (TypeError, ValueError):
                return None
    return None


def _dataset_input_dim(path: str) -> int | None:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh), None)
            if not row:
                return None
            col = "input" if "input" in row else next((k for k in row if "input" in k.lower()), None)
            if not col:
                return None
            try:
                v = json.loads(row[col])
                return len(v) if isinstance(v, list) else 1
            except Exception:
                return len([x for x in str(row[col]).strip("[]").split(",") if x.strip()])
    except Exception:
        return None


def link_paper(cli: Client, paper: str) -> None:
    print("\n-- 1. 论文 --")
    p = (cli.ok("GET", f"/api/papers/{paper}") or {}).get("paper") or {}
    md = p.get("markdown_path")
    check("论文已入库且有 markdown",
          p.get("paper_id") == paper and bool(md) and Path(md).exists(),
          f"{paper} status={p.get('status')} source={p.get('source')}")

    items = cli.ok("GET", f"/api/papers/{paper}/items") or []
    conf = [i for i in items if i.get("status") == "confirmed"]
    check("实验条目已抽取且有人工确认条目", bool(items) and bool(conf),
          f"条目 {len(items)} 确认 {len(conf)}，确认项 metric={[c.get('metric_name') for c in conf][:2]}")


def link_project_env_report(cli: Client, project: str) -> None:
    print("\n-- 2. 代码仓库加载 / 3. 真实独立环境 / 4. 结构分析报告 --")
    proj = cli.ok("GET", f"/api/projects/{project}")
    ws = Path(proj.get("workspace_path") or "")
    src = ws / "source"
    check("项目已加载源码（真实仓库）", proj.get("project_type") == "original" and src.exists(),
          f"{proj.get('name')} status={proj.get('status')} src={src.name}")

    env_dirs = [ws / "env" / "Scripts" / "python.exe", ws / "env" / "python.exe", ws / "env" / "bin" / "python"]
    env_py = next((e for e in env_dirs if e.exists()), None)
    check("真实独立环境已就绪（env_manager 建）", env_py is not None,
          str(env_py) if env_py else f"未见 env 解释器于 {ws / 'env'}")

    report = cli.ok("GET", f"/api/projects/{project}/report")
    hier = report.get("module_hierarchy") or []
    check("结构分析报告含模块层级与入口", bool(hier) and bool(report.get("entry_points") or report.get("model_files")),
          f"module_hierarchy={len(hier)} keys={list(report)[:6]}")


def link_reproduce_conclusion(cli: Client, paper: str, project: str) -> None:
    print("\n-- 5. 复现实验 / 6. 可信度结论 --")
    conn = _db()
    try:
        rr = conn.execute(
            """select ru.run_id, ru.duration_s, ru.command, rr.metric_value_actual, rr.verdict,
                      rr.result_id, ei.metric_name
               from reproduction_result rr
               join experiment_item ei on rr.item_id = ei.item_id
               left join run_record ru on rr.run_id = ru.run_id
               where ei.paper_id = ? """, (paper,)).fetchall()
    finally:
        conn.close()
    check("复现记录落库（reproduction_result + run_record）", bool(rr),
          "; ".join(f"{r[6]} actual={r[3]} verdict={r[4]}" for r in rr[:3]) if rr else "无复现记录")

    concl = cli.ok("GET", f"/api/papers/{paper}/conclusion") or {}
    check("可信度结论落库（overall_verdict）", bool(concl.get("overall_verdict")),
          f"overall={concl.get('overall_verdict')}")


def link_decompose_ingest(cli: Client, project: str) -> tuple[dict | None, str | None]:
    print("\n-- 7. 拆解入库 --")
    ir = cli.ok("GET", f"/api/projects/{project}/ir")
    body = ir.get("ir") or {}
    check("拆解 IR 生成且双验证通过",
          len(body.get("nodes", [])) > 0 and len(body.get("edges", [])) > 0
          and ir.get("verification_status") == "valid"
          and (ir.get("verification") or {}).get("overall") == "passed",
          f"nodes={len(body.get('nodes', []))} edges={len(body.get('edges', []))} "
          f"verif={ir.get('verification_status')}")

    modules = cli.ok("GET", "/api/modules") or []
    mine = [m for m in modules if m.get("source_project_id") == project]
    check("模块入库（source_project_id = 本项目）", bool(mine),
          f"{mine[0]['module_id']}:{mine[0]['module_version']} {mine[0]['name']}" if mine else "无模块")

    structured = cli.ok("GET", "/api/projects?project_type=structured") or []
    children = [p for p in structured if p.get("parent_project_id") == project]
    # 入库生成的那个以模块名为名（如 MLP）；排除验收脚本自建的同父项目
    mname = mine[0].get("name") if mine else None
    ingested = next((c for c in children if c.get("name") == mname), children[0] if children else None)
    check("结构化项目（画布）已由入库生成（parent = 本项目）", ingested is not None,
          f"{ingested['project_id']} {ingested.get('name')}" if ingested else "无子结构化项目")

    return (mine[0] if mine else None), (children[0]["project_id"] if children else None)


def link_canvas_train_version(cli: Client, paper: str, project: str, module: dict, dataset_id: str,
                              epochs: int, batch: int, lr: float) -> bool:
    print("\n-- 8. 画布拼装 + 9. 真实 CPU 训练 + 10. 版本留存 --")
    in_dim = _spec_dim(module.get("input_spec"))
    out_dim = _spec_dim(module.get("output_spec"))
    if not in_dim or not out_dim:
        check("模块 input/output 规格可读", False, f"input={module.get('input_spec')!r} output={module.get('output_spec')!r}")
        return False
    mref = f"{module['module_id']}:{module['module_version']}"

    # 数据集输入维度（用于前置 Linear 适配）
    conn = _db()
    try:
        row = conn.execute("select name, local_path from dataset_registry where dataset_id = ?", (dataset_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        check("目标数据集存在", False, dataset_id)
        return False
    ds_name, ds_path = row
    d_in = _dataset_input_dim(ds_path) or 1
    check("目标数据集（真实注册条目）就绪", True, f"{ds_name} dim={d_in} ← {ds_path}")

    # 画布拼装：Linear(d_in → 模块输入维) → 论文模块 → Linear(模块输出维 → 2)
    # 复用同名验收项目（脚本可重跑不堆积；无删除接口）
    name = f"M8 验收：{paper} 模块拼装"
    existing = [p for p in (cli.ok("GET", "/api/projects?project_type=structured") or [])
                if p.get("parent_project_id") == project and p.get("name") == name]
    if existing:
        pid = existing[0]["project_id"]
        print(f"  复用既有验收画布项目 {pid}")
    else:
        created = cli.ok("POST", "/api/projects", {
            "project_type": "structured", "name": name, "parent_project_id": project})
        pid = created["project_id"]
    graph = {
        "nodes": [
            {"id": "l1", "type": "linear_layer", "position": {"x": 0, "y": 0},
             "data": {"in_features": d_in, "out_features": int(in_dim), "bias": True}},
            {"id": "m1", "type": "module_ref", "position": {"x": 260, "y": 0},
             "data": {"moduleId": mref, "handles": {"inputs": ["in"], "outputs": ["out"]}}},
            {"id": "l2", "type": "linear_layer", "position": {"x": 520, "y": 0},
             "data": {"in_features": int(out_dim), "out_features": 2, "bias": True}},
        ],
        "edges": [
            {"id": "e1", "source": "l1", "sourceHandle": "out-0", "target": "m1",
             "targetHandle": "in", "data": {"label": "out_l1_out-0"}},
            {"id": "e2", "source": "m1", "sourceHandle": "out", "target": "l2",
             "targetHandle": "in-0", "data": {"label": "out_m1_out"}},
        ],
    }
    put = cli.ok("PUT", f"/api/projects/{pid}/graph", graph)
    code = (cli.ok("GET", f"/api/networks/{pid}/export") or {}).get("code", "")
    check("画布拼装保存即提交 + 导出可编译（内联论文模块）",
          put.get("version", {}).get("commit") is not None and "nn.Module" in code
          and "def forward" in code and len(code) > 200,
          f"project={pid} commit={put.get('version', {}).get('commit')} code={len(code)}字符")

    opts = cli.ok("GET", f"/api/networks/{pid}/run-options")
    check("运行面板数据源（真实数据集 + 真建环境）",
          any(d["dataset_id"] == dataset_id for d in opts.get("datasets", []))
          and any(e["project_id"] == project for e in opts.get("environments", [])),
          f"datasets={len(opts.get('datasets', []))} envs={len(opts.get('environments', []))}")

    run = cli.ok("POST", f"/api/networks/{pid}/run", {
        "dataset_id": dataset_id, "environment_project_id": project,
        "epochs": epochs, "batch_size": batch, "learning_rate": lr})
    task = cli.poll(run["task_id"], "train", timeout_s=900)
    progress = task.get("progress")
    if isinstance(progress, str):
        try:
            progress = json.loads(progress)
        except Exception:
            progress = {}
    metrics = (progress or {}).get("metrics") or {}
    check("真实 CPU 训练成功（论文模块 + 真实数据集 + 真建环境）", task["status"] == "success",
          f"task={run['task_id']} metrics={metrics}")

    runs = cli.ok("GET", f"/api/networks/{pid}/runs") or []
    train_ok = [r for r in runs if r.get("run_type") == "train" and r.get("status") == "success"]
    check("训练落 run_record（run_type=train 成功 + 指标）", bool(train_ok),
          f"run_id={train_ok[0]['run_id']} metrics={train_ok[0]['metrics']}" if train_ok else "无成功训练记录")

    tree = cli.ok("GET", f"/api/versions/{pid}/tree")
    versions = tree.get("versions", [])
    ws = Path(cli.ok("GET", f"/api/projects/{pid}").get("workspace_path") or "")
    glog = subprocess.run(["git", "-C", str(ws), "log", "--oneline"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    check("版本留存（画布项目 git 提交 ≥ 3：新建/保存/训练运行）",
          len(versions) >= 3 and glog.returncode == 0 and len(glog.stdout.strip().splitlines()) >= 3,
          f"versions={len(versions)} commits={len(glog.stdout.strip().splitlines())} "
          f"top={glog.stdout.strip().splitlines()[0] if glog.stdout.strip() else ''}")
    print(f"\n  ▶ 完整链落点：论文 {DEFAULT_PAPER} → … → 画布项目 {pid}（训练成功、版本留存）")
    return True


def link_multi_dataset_analysis(cli: Client, project: str) -> None:
    """6.2 多数据集与综合分析（用户裁定 A：不依赖真实权重的部分）。

    口径 A（2026-10-05）：现存项目 eval 入口全为 fixture 规则桩、被 fixture 闸门拒绝，
    「跨数据集评估需真实权重」维持**已登记边界**——本段断言其**既有运行证据**，
    并**执行** 对比 → 三图 → 多模型综合分析（不依赖权重的部分）。
    """
    print("\n\n==== 6.2 多数据集与综合分析 ====")
    proj = cli.ok("GET", f"/api/projects/{project}")
    ws = Path(proj.get("workspace_path") or "")

    # 1. 数据预处理产物（预处理 = preprocessed.csv + dataset.schema.json）
    data_root = ws / "data"
    pre = [d.name for d in sorted(data_root.iterdir())
           if d.is_dir() and (d / "preprocessed.csv").exists() and (d / "dataset.schema.json").exists()] \
        if data_root.exists() else []
    check("数据预处理产物（preprocessed.csv + dataset.schema.json）", bool(pre),
          f"数据集 {pre}" if pre else f"未见预处理产物于 {data_root}")

    # 2. 跨数据集评估（证据：eval 运行记录）——真实权重依赖为已登记边界
    conn = _db()
    try:
        evals = [r[0] for r in conn.execute(
            "select run_id from run_record where project_id=? and run_type='eval' and status='success'"
            " order by started_at desc", (project,))]
    finally:
        conn.close()
    check("跨数据集评估运行记录（eval success；真实权重依赖=已登记边界）", len(evals) >= 1,
          f"eval 运行 {len(evals)} 条；最新 {evals[0] if evals else '—'}")

    # 3. 数据集对齐已确认（5.3 人工闸门）
    datasets = cli.ok("GET", "/api/knowledge/list?data_type=dataset&limit=200") or []
    confirmed = []
    for ds in datasets:
        raw = ds.get("alignment")
        if not raw:
            continue
        try:
            al = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if al.get("status") == "confirmed" and project in (al.get("aligned_projects") or []):
            confirmed.append(ds.get("dataset_id") or ds.get("ref_id"))
    check("数据集对齐已确认（人工闸门已过）", bool(confirmed),
          f"本项目已确认对齐 {len(confirmed)} 个：{confirmed[:3]}")

    # 4. 结果对比：compare 会**重跑 eval** → 受 fixture 闸门限制（真实权重依赖=已登记边界）
    st, body = cli.call("POST", f"/api/projects/{project}/compare")
    if st == 200 and isinstance(body, dict) and body.get("task_id"):
        try:
            cli.poll(body["task_id"], "compare", timeout_s=900)
            check("结果对比（compare 任务成功）", True, f"task={body['task_id']}")
        except RuntimeError as e:
            msg = str(e)
            check("结果对比（compare）—— 真实权重依赖=已登记边界（fixture 闸门按设计拒绝规则桩）",
                  "fixture" in msg, msg[:140])
    else:
        check("结果对比（compare）", False, f"HTTP {st}: {body}")

    # 5. 三张可视化图（执行；用既有评估结果，不重跑 eval）
    chart_ok = {}
    for ch in ("performance", "error_dist", "cases"):
        r = cli.ok("POST", f"/api/projects/{project}/visualize/{ch}")
        html = r.get("html")
        chart_ok[ch] = bool(html) and Path(html).exists()
    check("三张可视化图产出（performance / error_dist / cases）", all(chart_ok.values()), str(chart_ok))

    # 6. 多模型综合分析（执行；agent 归因需凭证，缺失时按 attribution_error 如实记，不伪造）
    if len(evals) >= 2:
        ds_id = confirmed[0] if confirmed else None
        mm = cli.ok("POST", "/api/multi-model",
                    {"run_ids": evals[:2], "task_type": "classification", "dataset_id": ds_id})
        tid = mm.get("task_id") or mm.get("analysis_id")
        cli.poll(tid, "multi-model", timeout_s=900)
        report = cli.ok("GET", f"/api/multi-model/{tid}")
        rep = report.get("report") if (isinstance(report, dict) and "report" in report) else report
        keys = set(rep.keys()) if isinstance(rep, dict) else set()
        need = {"alignment", "metrics_before", "metrics_after", "consistent", "disagreements",
                "fusion", "n_common_samples"}
        check("多模型报告结构完整（同口径对齐 / 一致分歧 / 融合前后指标）",
              need <= keys, f"missing={sorted(need - keys)} keys={sorted(keys)[:12]}")
        if isinstance(rep, dict):
            check("标签归并按 dataset_registry.alignment 生效", bool((rep.get("alignment") or {}).get("used")),
                  str(rep.get("alignment")))
            check("融合结论入库（报告 knowledge_id → knowledge fusion_insight）",
                  bool(rep.get("knowledge_id")), str(rep.get("knowledge_id")))
            if rep.get("knowledge_id") and not KEEP_MM:
                _cleanup_multi_model(rep["knowledge_id"], tid)
                print(f"  (已自清本轮多模型产物 {rep['knowledge_id'][:8]}；--keep-mm 可保留)")
    else:
        check("多模型综合分析（需两个真实 eval 运行）", False, f"eval 运行不足 2（{len(evals)}）")

    # 6. 对比图界面可达
    st, _ = cli.call("GET", f"/api/projects/{project}/figures/performance")
    check("对比图经接口可达（figures 端点）", st == 200, f"HTTP {st}")


def link_self_iteration_loop(cli: Client, net: str, env: str, dataset_id: str,
                             epochs: int, batch: int, lr: float) -> None:
    """6.3 自迭代闭环：检索 → 带入 → 训练(自动调参) → 蒸馏 → 确认 → 再带入命中 → 指标复现。

    蒸馏（agent 起草）需凭证：无凭证时断言**既有蒸馏产物**（带 source_paper_id/source_task_id 的条目）。
    """
    print("\n\n==== 6.3 自迭代闭环 ====")

    # 1. 知识库检索
    hits = cli.ok("GET", "/api/knowledge/search?types=knowledge&limit=200") or []
    check("知识库检索命中", len(hits) >= 1, f"命中 {len(hits)} 条")

    # 2/6. 任务前带入（命中已确认知识）
    ds_name = "realcase-chembl-freq-balanced"

    def bring():
        return cli.ok("POST", f"/api/knowledge/bring?task_type=classification&dataset={ds_name}") or {}

    brought = bring()
    pa = brought.get("param_advice") or []
    check("任务前带入命中知识（bring → param_advice）", len(pa) >= 1,
          f"param_advice {len(pa)} 条，命中 {[k.get('knowledge_id') for k in pa][:2]}")

    # 3. 训练：自动调参（带入知识 → 候选超参 → 选优）
    r = cli.ok("POST", f"/api/networks/{net}/autotune", {
        "dataset_id": dataset_id, "environment_project_id": env,
        "epochs": epochs, "batch_size": batch, "learning_rate": lr})
    task = cli.poll(r["task_id"], "autotune", timeout_s=1800)
    prog = task.get("progress")
    if isinstance(prog, str):
        try:
            prog = json.loads(prog)
        except Exception:
            prog = {}
    winner = (prog or {}).get("winner") or {}
    primary = (prog or {}).get("primary_value")
    check("自动调参成功（带入知识 → 候选逐个训练 → 选优）", task["status"] == "success" and bool(winner),
          f"winner={winner} primary={prog.get('primary_metric')}={primary}")

    # 4. 蒸馏产物入库（agent 起草需凭证；此处断言既有真实产物）
    conn = _db()
    try:
        n_distill = 0
        for (s,) in conn.execute("select structured from knowledge"):
            if not s:
                continue
            try:
                st = json.loads(s)
            except Exception:
                continue
            if isinstance(st, dict) and (st.get("source_paper_id") or st.get("source_task_id")):
                n_distill += 1
        n_conf = conn.execute("select count(*) from knowledge where status='confirmed'").fetchone()[0]
    finally:
        conn.close()
    check("蒸馏产物入库（带 source_paper_id/source_task_id 的知识条目）", n_distill >= 1,
          f"{n_distill} 条；全库 confirmed {n_conf} 条")

    # 5. 用户确认：带入命中的知识须为 confirmed
    if pa:
        kid = pa[0].get("knowledge_id")
        item = cli.ok("GET", f"/api/knowledge/items/knowledge/{kid}") or {}
        check("闭环用知识已人工确认（confirmed）", item.get("status") == "confirmed",
              f"{kid} status={item.get('status')}")

    # 6. 再带入命中（确认后仍在带入结果中）
    again = bring()
    pa2 = again.get("param_advice") or []
    check("再带入命中该知识（闭环回收）", len(pa2) >= 1,
          f"param_advice {len(pa2)} 条")

    # 7. 指标复现：采纳 winner 超参重训，指标复现
    run = cli.ok("POST", f"/api/networks/{net}/run", {
        "dataset_id": dataset_id, "environment_project_id": env,
        "epochs": int(winner.get("epochs") or epochs),
        "batch_size": int(winner.get("batch_size") or batch),
        "learning_rate": float(winner.get("learning_rate") or lr)})
    t2 = cli.poll(run["task_id"], "train(采纳建议)", timeout_s=900)
    p2 = t2.get("progress")
    if isinstance(p2, str):
        try:
            p2 = json.loads(p2)
        except Exception:
            p2 = {}
    got = (p2 or {}).get("metrics") or {}
    check("采纳建议重训 → 指标复现（≥ 调参 winner）",
          t2["status"] == "success" and got and (primary is not None)
          and float(got.get((prog or {}).get("primary_metric") or "accuracy") or 0) >= float(primary) - 1e-9,
          f"重训 {got} vs winner {primary}")

    # 版本留存：自动调参 + 训练各留版本节点
    tree = cli.ok("GET", f"/api/versions/{net}/tree")
    check("闭环留痕（版本树含自动调参/训练节点）", len((tree or {}).get("versions", [])) >= 2,
          f"versions={len((tree or {}).get('versions', []))}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="M8 6.1 推荐路径联调验收")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--paper", default=DEFAULT_PAPER)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--mm-project", default=DEFAULT_MM_PROJECT, help="6.2 多数据集与综合分析所用的项目")
    ap.add_argument("--loop-project", default=DEFAULT_LOOP_PROJECT, help="6.3 自迭代闭环所用的画布项目")
    ap.add_argument("--loop-env", default=DEFAULT_LOOP_ENV, help="6.3 训练所用环境项目")
    ap.add_argument("--loop-dataset", default=DEFAULT_LOOP_DATASET, help="6.3 训练所用数据集")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--keep-mm", action="store_true",
                    help="保留 6.2 多模型产物（默认跑完自清本轮 knowledge/报告，避免反复跑堆积）")
    args = ap.parse_args()
    global KEEP_MM
    KEEP_MM = args.keep_mm

    cli = Client(args.base)
    print(f"M8 6.1/6.2/6.3 联调 — base={args.base} paper={args.paper} project={args.project} mm_project={args.mm_project}")
    try:
        link_paper(cli, args.paper)
        link_project_env_report(cli, args.project)
        link_reproduce_conclusion(cli, args.paper, args.project)
        module, child = link_decompose_ingest(cli, args.project)
        if module:
            link_canvas_train_version(cli, args.paper, args.project, module, args.dataset,
                                      args.epochs, args.batch_size, args.learning_rate)
        else:
            check("画布拼装 + 训练 + 版本", False, "无入库模块，无法拼装")
        link_multi_dataset_analysis(cli, args.mm_project)
        link_self_iteration_loop(cli, args.loop_project, args.loop_env, args.loop_dataset,
                                 args.epochs, args.batch_size, args.learning_rate)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("执行中断", False, str(e))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n==== M8 6.1/6.2/6.3 联调结果: {passed}/{total} 通过 ====")
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"  ✗ {name}")
    return 0 if total and passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
