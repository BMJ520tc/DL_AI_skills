"""架构级自迭代验收（需求六.1 延伸）：建议（agent 起草）→ 确认 → 落画布（新版本）→ 可导出/可训练。

口径（连**真实后端**，默认 :8000；agent 起草**需模型凭证**）：
  A. 真实 agent 段（用真实「标准 + 模块」画布 d7470ae9 = M8 验收 gears 拼装）：
     A1 ir 图项目 arch-suggest → 400（与导出/训练同一口径）
     A2 arch-suggest 入队 → 轮询至 success → GET arch-suggestions
     A3 报告形状断言：每条建议 op ∈ {replace_module,add_layer,rewire}、target_node_id 属于当前图、
        valid 为布尔、invalid 的建议必须带 invalid_reason
     A4 若有**有效**建议：arch-apply → 断言返回图（无 ir 节点、边端点存在）→
        PUT 到一个**临时结构化项目**（按名复用）→ 断言保存即版本（version 非空）→
        GET export → 200 且代码可编译（改动后的图可导出/可训练）
  B. 确定性段（不依赖 agent 产出；脚本内直接调后端纯函数，覆盖 apply 本体）：
     B1 在 d7470ae9 的真实图上造一条 add_layer（锚定唯一出边）→ apply_suggestion →
        断言新增节点/两条新边、无 ir 节点、原图未改（纯函数）
     B2 把 B1 结果 PUT 到临时项目 → export 可编译、版本树 ≥1
     B3 ir 图 apply → ValueError（守卫）

    D:\\python.exe scripts/arch_acceptance.py --base http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))   # 供 B 段直接调后端纯函数

DEFAULT_STD_PROJECT = "d7470ae9ec9e4bf8a35610fa549a8b26"   # M8 验收：gears 模块拼装（标准 + module_ref）
DEFAULT_IR_PROJECT = "43396bbb556b47e299aff1c68a2a6ae6"    # MLP（拆解 ir 图 → 应拒）
TMP_PROJECT_NAME = "架构建议验收（临时）"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def _req(base: str, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def _poll_task(base: str, task_id: str, timeout_s: int = 240) -> dict:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        _, task = _req(base, f"/api/tasks/{task_id}")
        if isinstance(task, dict):
            last = task
            if task.get("status") in ("success", "failed", "cancelled"):
                return task
        time.sleep(2)
    return last


def _reuse_or_create_tmp_project(base: str) -> str:
    _, projects = _req(base, "/api/projects")
    for p in projects or []:
        if p.get("name") == TMP_PROJECT_NAME:
            return p["project_id"]
    _, created = _req(base, "/api/projects", "POST",
                      {"project_type": "structured", "name": TMP_PROJECT_NAME})
    return created["project_id"]


def _put_and_export(base: str, project_id: str, graph: dict) -> tuple[bool, str]:
    """保存图（断言版本生成）+ 导出（断言可编译）。返回 (ok, detail)。"""
    code, resp = _req(base, f"/api/projects/{project_id}/graph", "PUT", graph)
    if code != 200:
        return False, f"PUT graph {code}: {resp}"
    version = (resp or {}).get("version")
    if not version:
        return False, f"保存未生成版本：{resp}"
    code2, exp = _req(base, f"/api/networks/{project_id}/export")
    if code2 != 200:
        return False, f"export {code2}: {exp}"
    src = (exp or {}).get("code") or ""
    try:
        compile(src, "<export>", "exec")
    except SyntaxError as e:
        return False, f"导出代码不可编译：{e}"
    return True, f"version={version}；导出 {len(src)} 字节、可编译"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--std-project", default=DEFAULT_STD_PROJECT)
    ap.add_argument("--ir-project", default=DEFAULT_IR_PROJECT)
    args = ap.parse_args()
    base = args.base.rstrip("/")

    # ---- A1 ir 图应拒 ----
    code, resp = _req(base, f"/api/networks/{args.ir_project}/arch-suggest", "POST", {})
    check("A1 ir 图 arch-suggest → 400", code == 400,
          f"HTTP {code}；{(resp or {}).get('detail') if isinstance(resp, dict) else resp}")

    # ---- A2 真实 agent 起草 ----
    _, graph = _req(base, f"/api/projects/{args.std_project}/graph")
    node_ids = {n.get("id") for n in (graph or {}).get("nodes", [])}
    # 带一条**明确授权探索**的 hint：无 hint 时模型会「无可靠依据就返回空数组」（诚实——本画布的已确认
    # 知识指向评测口径问题、非架构问题，模型据此拒绝改架构，属正确行为）；验收要走到 apply 路径，
    # 故显式声明「做一次架构实验」，让 agent 在真实图上产出可执行建议。
    code, started = _req(base, f"/api/networks/{args.std_project}/arch-suggest", "POST",
                         {"hint": "我们想在这张画布上做一次**架构实验**（与既有评测结论无关）："
                                  "请给出 1-2 条可执行的改动（换模块 / 加层 / 改连接）。"})
    ok = code == 200 and isinstance(started, dict) and started.get("task_id")
    check("A2 arch-suggest 入队", bool(ok), f"HTTP {code}")
    report = None
    if ok:
        task = _poll_task(base, started["task_id"])
        check("A2 任务 success", task.get("status") == "success",
              f"status={task.get('status')} err={task.get('error')}")
        _, report = _req(base, f"/api/networks/{args.std_project}/arch-suggestions")
        report = report if isinstance(report, dict) else None

    # ---- A3 报告形状 ----
    if report:
        sgs = report.get("suggestions") or []
        shape_ok, problems = True, []
        for s in sgs:
            if s.get("op") not in ("replace_module", "add_layer", "rewire"):
                shape_ok, _ = False, problems.append(f"op 越界 {s.get('op')}")
            if s.get("target_node_id") not in node_ids:
                shape_ok = False
                problems.append(f"target 不在图内 {s.get('target_node_id')}")
            if not isinstance(s.get("valid"), bool):
                shape_ok, _ = False, problems.append("valid 非布尔")
            if s.get("valid") is False and not s.get("invalid_reason"):
                shape_ok = False
                problems.append("无效建议缺 invalid_reason")
        check("A3 报告形状", shape_ok,
              f"{len(sgs)} 条建议；{'OK' if shape_ok else '；'.join(problems)}"
              + (f"；agent_error={report.get('agent_error')}" if report.get("agent_error") else ""))
        valid = [s for s in sgs if s.get("valid")]
    else:
        valid = []
        check("A3 报告形状", False, "未取到报告")

    # ---- A4 应用有效建议 → 临时项目保存 + 导出 ----
    if report and valid:
        s = valid[0]
        code, applied = _req(base, f"/api/networks/{args.std_project}/arch-apply", "POST",
                             {"task_id": report["task_id"], "suggestion_id": s["suggestion_id"]})
        g2 = (applied or {}).get("graph") if isinstance(applied, dict) else None
        applied_ok = code == 200 and isinstance(g2, dict)
        if applied_ok:
            ids = {n.get("id") for n in g2["nodes"]}
            applied_ok = (all(n.get("type") != "ir" for n in g2["nodes"])
                          and all(e.get("source") in ids and e.get("target") in ids for e in g2["edges"]))
        check("A4 arch-apply 返回合法新图", applied_ok,
              f"op={s.get('op')} target={s.get('target_node_id')}" if applied_ok else f"HTTP {code}: {applied}")
        if applied_ok:
            pid = _reuse_or_create_tmp_project(base)
            ok2, detail = _put_and_export(base, pid, g2)
            check("A4 落画布（新版本）+ 可导出", ok2, detail)
    else:
        print("[SKIP] A4 —— 本轮 agent 未产出有效建议（形状与确定性路径见 A3/B 段）", flush=True)

    # ---- B 段：确定性 apply（不依赖 agent） ----
    from app.services import arch_service  # noqa: E402 —— 脚本内直接调后端纯函数

    g = graph
    add_sug = {"op": "add_layer", "target_node_id": "", "payload": {
        "anchor_edge_id": (g["edges"][0]["id"] if g.get("edges") else ""),
        "new_node_type": "relu_layer", "params": {}}}
    try:
        before_nodes, before_edges = len(g["nodes"]), len(g["edges"])
        out = arch_service.apply_suggestion(g, add_sug)
        ok_b1 = (len(out["nodes"]) == before_nodes + 1 and len(out["edges"]) == before_edges + 1
                 and all(n.get("type") != "ir" for n in out["nodes"])
                 and len(g["nodes"]) == before_nodes)
        detail = f"{before_nodes}→{len(out['nodes'])} 节点、{before_edges}→{len(out['edges'])} 边"
    except Exception as exc:  # noqa: BLE001
        ok_b1, out, detail = False, None, repr(exc)
    check("B1 确定性 add_layer（纯函数）", ok_b1, detail)

    if ok_b1 and out is not None:
        pid = _reuse_or_create_tmp_project(base)
        ok_b2, detail2 = _put_and_export(base, pid, out)
        check("B2 确定性结果落画布 + 可导出", ok_b2, detail2)

    try:
        arch_service.apply_suggestion({"nodes": [{"id": "m", "type": "ir", "data": {}}], "edges": []},
                                      {"op": "rewire", "payload": {"edits": []}})
        ok_b3, detail3 = False, "未抛错"
    except ValueError as exc:
        ok_b3, detail3 = True, str(exc)[:60]
    except Exception as exc:  # noqa: BLE001
        ok_b3, detail3 = False, repr(exc)
    check("B3 ir 图 apply 守卫", ok_b3, detail3)

    passed = sum(1 for _, ok_, _ in RESULTS if ok_)
    total = len(RESULTS)
    print(f"\n=== arch acceptance: {passed}/{total} ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
