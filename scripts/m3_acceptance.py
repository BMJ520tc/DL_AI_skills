"""M3 验收驱动脚本（模块四，阶段3 C1；宿主侧运行，仅标准库）。

前置：本地后端已启动（uvicorn :8000）、claude CLI 可用（agent 拆解用）、
样例仓库环境可创建（pip 装 torch，或已建好用 --no-env 跳过）。

自动走完 M3 四条目验收的 API 链路：
  ① 拆解 → 再生成 → 结构+数值双验证通过 → 入库（module 表 + unified_index）
  ② module.json 元信息完整（source_project_id/task_type/input_spec/output_spec/
     tags/description/verification/saved_module_compat）
  ③ 生成结构化项目（parent_project_id + graph.json），画布快照保存往返
  ④ 调参破坏验证：overall=failed、failure_reason/diff_layers 记录、入库被拦、无新行

用法:
    python scripts/m3_acceptance.py --source examples/m3_sample_repo     # ①②③
    python scripts/m3_acceptance.py --mode fail --project-id <项目id>    # ④（需先通过 ①②③）
    python scripts/m3_acceptance.py --mode all --source ...              # ①②③④ 连跑
"""
import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "index.db"

# 任务级超时（秒），对齐 backend task_manager.TASK_TIMEOUTS 留余量
TIMEOUTS = {
    "env_create": 3700,
    "verify": 700,
    "analyze": 700,
    "decompose": 4000,
    "decompose_trace": 1000,
    "decompose_verify": 2500,
    "module_ingest": 700,
}

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    line = ("  ✓ " if ok else "  ✗ ") + name
    if detail:
        line += f" — {detail}"
    print(line)


def api(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, object | None]:
    """HTTP 调用；返回 (status, parsed_json|None)。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:  # noqa: BLE001
            return e.code, raw.decode("utf-8", errors="replace")


def api_ok(base: str, method: str, path: str, body: dict | None = None):
    """200 且响应体可解析为 JSON（dict 或 list）才算成功；否则抛错并回显响应。"""
    status, payload = api(base, method, path, body)
    if status != 200 or isinstance(payload, str):
        raise RuntimeError(f"{method} {path} → HTTP {status}: {payload}")
    return payload


def poll_task(base: str, task_id: str, label: str, quiet: bool = False) -> dict:
    """轮询任务至终态；失败抛 RuntimeError（附 task.error）。"""
    timeout = None
    started = time.time()
    last_progress = None
    while True:
        status, payload = api(base, "GET", f"/api/tasks/{task_id}")
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"任务查询失败 HTTP {status}: {payload}")
        task = payload
        if timeout is None:
            timeout = TIMEOUTS.get(str(task.get("task_type")), 700) + 120
        progress = task.get("progress")
        if progress and progress != last_progress and not quiet:
            print(f"    [{label}] {task.get('status')} {progress}")
            last_progress = progress
        if task.get("status") in ("success", "failed", "cancelled"):
            if task.get("status") != "success":
                raise RuntimeError(f"{label} 任务失败: {task.get('error')}")
            return task
        if time.time() - started > timeout:
            raise RuntimeError(f"{label} 任务超时（>{timeout}s）")
        time.sleep(5)


def wait_env(base: str, project_id: str, no_env: bool) -> None:
    status, payload = api(base, "GET", f"/api/projects/{project_id}/env")
    if isinstance(payload, dict) and payload.get("status") == "ready":
        print("  env 已就绪")
        return
    if no_env:
        print("  --no-env：跳过 env 创建（假定就绪）")
        return
    created = api_ok(base, "POST", f"/api/projects/{project_id}/env")
    print(f"  env 创建任务 {created.get('task_id')}（pip 安装 torch 可能较久）")
    poll_task(base, created["task_id"], "env")


def create_or_reuse_project(base: str, source: str | None, project_id: str | None) -> str:
    if project_id:
        print(f"  复用项目 {project_id}")
        return project_id
    if not source:
        raise RuntimeError("需提供 --source 或 --project-id")
    created = api_ok(
        base, "POST", "/api/projects",
        {"project_type": "original", "source_url": source, "name": "M3验收样例"},
    )
    pid = created["project_id"]
    print(f"  创建原始项目 {pid}")
    return pid


def get_ir(base: str, project_id: str) -> dict:
    status, payload = api(base, "GET", f"/api/projects/{project_id}/ir")
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(f"GET /ir → HTTP {status}: {payload}")
    return payload


def find_conv_node(ir: dict) -> dict | None:
    for n in ir.get("nodes", []):
        cn = str(n.get("class_name") or "")
        params = n.get("params") or {}
        if "conv" in cn.lower() and "out_channels" in params:
            return n
    return None


def _layout_violations(g_nodes: list[dict]) -> list[str]:
    """布局自检（纯算术，不依赖前端）：每个父容器的 layout_hint 尺寸须包住其全部后代。

    子节点坐标是相对父节点的（React Flow parentId 语义），逐层累加得到绝对矩形；
    越界说明布局算错了（历史上表现为「所有节点叠在一起」）。
    纯标准库实现，避免给验收脚本引入依赖。
    """
    by_id = {n.get("id"): n for n in g_nodes}
    hint_of = {n.get("id"): (n.get("data") or {}).get("layout_hint") or {} for n in g_nodes}

    def absolute(nid: str) -> tuple[float, float] | None:
        node = by_id.get(nid)
        if node is None:
            return None
        pos = node.get("position") or {}
        x, y = float(pos.get("x") or 0), float(pos.get("y") or 0)
        parent = node.get("parentId")
        while parent:
            parent_node = by_id.get(parent)
            if parent_node is None:
                break
            ppos = parent_node.get("position") or {}
            x += float(ppos.get("x") or 0)
            y += float(ppos.get("y") or 0)
            parent = parent_node.get("parentId")
        return x, y

    bad: list[str] = []
    for n in g_nodes:
        parent = n.get("parentId")
        if not parent:
            continue
        size = hint_of.get(parent) or {}
        w, h = size.get("width"), size.get("height")
        if not w or not h:
            continue
        p_abs = absolute(parent)
        c_abs = absolute(n.get("id"))
        if p_abs is None or c_abs is None:
            continue
        if not (p_abs[0] - 1 <= c_abs[0] and c_abs[0] <= p_abs[0] + float(w) + 1
                and p_abs[1] - 1 <= c_abs[1] and c_abs[1] <= p_abs[1] + float(h) + 1):
            bad.append(f"{n.get('id')} 越出 {parent}")
    return bad


def db_module_row(module_id: str, module_version: str) -> dict | None:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT module_id, module_version FROM module WHERE module_id = ? AND module_version = ?",
            (module_id, module_version),
        ).fetchone()
        idx = conn.execute(
            "SELECT ref_id, data_type FROM unified_index WHERE ref_id = ?",
            (f"{module_id}:{module_version}",),
        ).fetchone()
    finally:
        conn.close()
    return {"module": row, "index": idx}


# ---------------------------------------------------------------------------
# ①②③
# ---------------------------------------------------------------------------
def run_full(base: str, source: str | None, project_id: str | None, no_env: bool, entry: str | None,
             light: bool) -> str:
    print("== M3 验收 ①②③：拆解 → 双验证 → 入库 → 结构化项目 ==")
    pid = create_or_reuse_project(base, source, project_id)
    wait_env(base, pid, no_env)

    print("[1/8] 模块一验证")
    t = api_ok(base, "POST", f"/api/projects/{pid}/verify")
    poll_task(base, t["task_id"], "verify")

    print("[2/8] 结构分析")
    t = api_ok(base, "POST", f"/api/projects/{pid}/analyze")
    poll_task(base, t["task_id"], "analyze")
    report = api_ok(base, "GET", f"/api/projects/{pid}/report")
    hierarchy = report.get("module_hierarchy", []) if isinstance(report, dict) else []
    check("结构报告含 module_hierarchy", len(hierarchy) > 0, f"{len(hierarchy)} 项")

    print("[3/8] agent 拆解" + (f"（指定入口类 {entry}）" if entry else ""))
    t = api_ok(base, "POST", f"/api/projects/{pid}/decompose", {"entry_class": entry} if entry else None)
    poll_task(base, t["task_id"], "decompose")
    ir = get_ir(base, pid)["ir"]
    kinds: dict[str, int] = {}
    for n in ir.get("nodes", []):
        kinds[n.get("kind", "?")] = kinds.get(n.get("kind", "?"), 0) + 1
    check("IR 生成", len(ir.get("nodes", [])) > 0 and len(ir.get("edges", [])) > 0,
          f"nodes={len(ir.get('nodes', []))} edges={len(ir.get('edges', []))} kinds={kinds} entry={ir.get('entry_class')}")

    print("[4/8] 补形状")
    t = api_ok(base, "POST", f"/api/projects/{pid}/decompose/trace")
    poll_task(base, t["task_id"], "trace")
    ir = get_ir(base, pid)["ir"]
    missing = [n["id"] for n in ir.get("nodes", [])
               if n.get("kind") != "op" and (not n.get("input_shape") or not n.get("output_shape"))]
    check("形状回填完整", not missing, f"缺失 {missing}" if missing else "全部已知")

    print("[5/8] 再生成代码")
    regen = api_ok(base, "POST", f"/api/projects/{pid}/decompose/regenerate")
    code = regen.get("code") or ""
    check("代码再生成", "class" in code and "nn.Module" in code and len(code) > 500, f"{len(code)} 字符")

    print("[6/8] 两步验证")
    t = api_ok(base, "POST", f"/api/projects/{pid}/decompose/verify")
    poll_task(base, t["task_id"], "verify")
    resp = get_ir(base, pid)
    ver = resp.get("verification") or {}
    struct = ver.get("structure") or {}
    numeric = ver.get("numeric") or {}
    struct_ok = all(struct.get(k) for k in
                    ("layer_sequence_match", "param_shapes_match", "output_shape_match"))
    seeds_ok = all(s.get("passed") for s in numeric.get("per_seed", []))
    check("验收①：结构+数值双验证通过",
          ver.get("overall") == "passed" and struct_ok and seeds_ok and resp.get("verification_status") == "valid",
          f"overall={ver.get('overall')} status={resp.get('verification_status')} "
          f"seeds={[s.get('seed') for s in numeric.get('per_seed', []) if s.get('passed')]}")

    if light:
        print("\n--light：跳过入库/画布（C2 论文库二次验证只评估拆解→验证链路质量）")
        return pid

    print("[7/8] 入库")
    t = api_ok(base, "POST", "/api/modules", {"project_id": pid})
    poll_task(base, t["task_id"], "ingest")
    modules = api_ok(base, "GET", "/api/modules")
    mine = [m for m in modules if m.get("source_project_id") == pid]
    check("模块入库", bool(mine), "module 表无记录" if not mine else
          f"module_id={mine[0]['module_id']} version={mine[0]['module_version']}")
    if mine:
        m = mine[0]
        db = db_module_row(m["module_id"], m["module_version"])
        db_ok = bool(db and db["module"] and db["index"])
        check("DB：module 表 + unified_index", db_ok,
              f"ref_id={m['module_id']}:{m['module_version']}" if db_ok else "index.db 未查到记录")
        # ② module.json 元信息（path 记模块包目录；兼容早期把 module.json 文件路径写进 path 的记录）
        pkg_path = Path(m.get("path") or "")
        pkg = pkg_path if pkg_path.suffix == ".json" else pkg_path / "module.json"
        meta_ok = False
        detail = ""
        if pkg.exists():
            meta = json.loads(pkg.read_text(encoding="utf-8"))
            need = ("description", "source_project_id", "task_type", "input_spec", "output_spec",
                    "tags", "verification", "saved_module_compat")
            absent = [k for k in need if meta.get(k) in (None, "", [])]
            meta_ok = not absent and meta.get("source_project_id") == pid and meta.get("task_type")
            detail = f"{pkg}（缺 {absent}）" if absent else str(pkg)
        else:
            detail = f"module.json 不存在: {pkg}"
        check("验收②：module.json 元信息完整", meta_ok, detail)

    print("[8/8] 结构化项目 + 画布快照")
    structured = api_ok(base, "GET", "/api/projects?project_type=structured")
    children = [p for p in structured if p.get("parent_project_id") == pid]
    check("结构化项目生成", bool(children), "无子结构化项目" if not children else
          f"{children[0]['project_id']} status={children[0].get('status')}")
    if children:
        sid = children[0]["project_id"]
        graph = api_ok(base, "GET", f"/api/projects/{sid}/graph")
        g_nodes, g_edges = graph.get("nodes", []), graph.get("edges", [])
        nested = any(n.get("parentId") for n in g_nodes)
        skips = [e for e in g_edges if e.get("kind") == "skip"]
        check("graph.json：层级嵌套 + 跳跃边",
              nested and bool(skips),
              f"nodes={len(g_nodes)} nested={nested} skip边={len(skips)}")
        # 布局自检：容器的建议尺寸必须真的装下其全部后代（历史缺陷：固定网格 + 固定父宽 →
        # 兄弟容器重叠、子节点溢出父框，界面上表现为「所有节点叠在一起」）
        layout_bad = _layout_violations(g_nodes)
        check("graph.json：布局自检（子节点不越出父容器）", not layout_bad,
              f"越界 {len(layout_bad)} 处: {layout_bad[:3]}" if layout_bad else f"{len(g_nodes)} 节点全部在父框内")
        # 画布保存往返：改某 conv 的 kernel_size → PUT → GET 仍在
        target = next((n for n in g_nodes
                       if isinstance(n.get("data", {}).get("params"), dict)
                       and "kernel_size" in (n.get("data", {}).get("params") or {})), None)
        if target is None:
            target = next((n for n in g_nodes if (n.get("data", {}).get("params") or {})), None)
        if target:
            params = dict(target["data"].get("params") or {})
            key = "kernel_size" if "kernel_size" in params else next(iter(params), None)
            if key is not None:
                old = params[key]
                params[key] = 5 if old == 3 else 3  # 3↔5 往返
                target["data"]["params"] = params
                api_ok(base, "PUT", f"/api/projects/{sid}/graph", graph)
                again = api_ok(base, "GET", f"/api/projects/{sid}/graph")
                again_nodes = again.get("nodes", [])
                again_target = next((n for n in again_nodes if n.get("id") == target["id"]), {})
                persisted = (again_target.get("data", {}).get("params") or {}).get(key) == params[key]
                check("验收③：画布保存往返（改 kernel_size 重开仍在）", persisted,
                      f"{target['id']}.{key} = {old} → {params[key]}")
            else:
                check("验收③：画布保存往返", False, "无可编辑参数节点")
        else:
            check("验收③：画布保存往返", False, "无参数节点")
    return pid


# ---------------------------------------------------------------------------
# ④
# ---------------------------------------------------------------------------
def run_fail(base: str, project_id: str) -> None:
    print("== M3 验收④：调参破坏 → 验证失败被记录 → 入库被拦 ==")
    resp = get_ir(base, project_id)
    ir = resp["ir"]
    node = find_conv_node(ir)
    if node is None:
        check("验收④", False, "未找到带 out_channels 的 conv 节点")
        return
    original = dict(node.get("params") or {})
    broken = dict(original)
    # 破坏方式取「参数形状变化、通道数不变」的键：两模型前向都能跑完，由结构比对如实报出
    # 差异层；若改 out_channels 而不动其后的 BN，前向会直接抛错（无法定位差异层）。
    if "kernel_size" in broken:
        key, old = "kernel_size", broken["kernel_size"]
        broken[key] = 5 if old != 5 else 3
    elif "out_channels" in broken:
        key, old = "out_channels", broken["out_channels"]
        broken[key] = 48 if old != 48 else 64
    else:
        check("验收④", False, "无可破坏的参数键（kernel_size/out_channels）")
        return
    print(f"  调参节点 {node['id']}（{node.get('class_name')}）{key} {old} → {broken[key]}")

    try:
        modules_before = api_ok(base, "GET", "/api/modules")
        n_modules_before = sum(1 for m in modules_before if m.get("source_project_id") == project_id)
        structured_before = api_ok(base, "GET", "/api/projects?project_type=structured")
        n_structured_before = sum(1 for p in structured_before if p.get("parent_project_id") == project_id)

        api_ok(base, "PUT", f"/api/projects/{project_id}/ir/nodes/{node['id']}", {"params": broken})
        resp = get_ir(base, project_id)
        check("调参后验证变 stale", resp.get("verification_status") == "stale",
              f"status={resp.get('verification_status')}")

        regen = api_ok(base, "POST", f"/api/projects/{project_id}/decompose/regenerate")
        check("调参后再生成", bool(regen.get("code")), f"{len(regen.get('code') or '')} 字符")

        t = api_ok(base, "POST", f"/api/projects/{project_id}/decompose/verify")
        poll_task(base, t["task_id"], "verify")
        resp = get_ir(base, project_id)
        ver = resp.get("verification") or {}
        numeric = ver.get("numeric") or {}
        diff = list((ver.get("structure") or {}).get("diff_layers") or [])
        diff += [d for s in numeric.get("per_seed", []) for d in (s.get("diff_layers") or [])]
        check("验证失败被记录（overall=failed + failure_reason + diff_layers）",
              ver.get("overall") == "failed" and bool(ver.get("failure_reason")) and bool(diff),
              f"overall={ver.get('overall')} reason={ver.get('failure_reason')!r} diff={diff[:3]}")

        # 入库被拦：POST /api/modules 的前置校验应同步 409（3.5）；若仍入队，任务也应失败
        ingest_failed = False
        try:
            t = api_ok(base, "POST", "/api/modules", {"project_id": project_id})
        except RuntimeError as e:
            ingest_failed = True
            print(f"    ingest 被前置校验同步拦下（预期 409）: {e}")
        else:
            try:
                poll_task(base, t["task_id"], "ingest")
            except RuntimeError as e:
                ingest_failed = True
                print(f"    ingest 任务按预期失败: {e}")
        modules_after = api_ok(base, "GET", "/api/modules")
        n_modules_after = sum(1 for m in modules_after if m.get("source_project_id") == project_id)
        structured_after = api_ok(base, "GET", "/api/projects?project_type=structured")
        n_structured_after = sum(1 for p in structured_after if p.get("parent_project_id") == project_id)
        check("入库被拦且无新行/新结构化项目",
              ingest_failed and n_modules_after == n_modules_before and n_structured_after == n_structured_before,
              f"ingest_failed={ingest_failed} modules {n_modules_before}→{n_modules_after} "
              f"structured {n_structured_before}→{n_structured_after}")
    finally:
        # 无论成功、失败还是中途异常，都把参数还原，项目保持可用（否则项目 IR 会留在半坏状态）
        api_ok(base, "PUT", f"/api/projects/{project_id}/ir/nodes/{node['id']}", {"params": original})
        print(f"  已还原 {node['id']} 参数（{key}={old}）")
        # ④ 的产物是「故意失败的验证记录」，会让查看器显示「比对：未通过 / 验证：已失效」。
        # 这里重验一次把项目复原为通过状态，避免把负向用例的残留误读成真实失败。
        try:
            t = api_ok(base, "POST", f"/api/projects/{project_id}/decompose/verify")
            poll_task(base, t["task_id"], "verify（还原后重验）")
            resp = get_ir(base, project_id)
            ver = resp.get("verification") or {}
            print(f"  还原后重验：overall={ver.get('overall')} status={resp.get('verification_status')}"
                  "（查看器应显示「比对：通过」）")
        except RuntimeError as e:  # noqa: BLE001 —— 重验失败不影响 ④ 的结论
            print(f"  还原后重验未成功（不影响 ④ 结论）: {e}")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="M3 验收驱动（模块四）")
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--source", default=None, help="样例仓库本地路径（默认 examples/m3_sample_repo）")
    parser.add_argument("--project-id", default=None, help="复用已有原始项目")
    parser.add_argument("--mode", choices=["full", "fail", "all"], default="full")
    parser.add_argument("--no-env", action="store_true", help="跳过 env 创建")
    parser.add_argument("--entry", default="ResNet", help="拆解入口类（默认 ResNet；SkipCNN/MLP 等可覆盖）")
    parser.add_argument("--light", action="store_true", help="仅拆解→补形状→验证链路（C2 论文库二次验证）")
    args = parser.parse_args()

    # 后端 load_source 按自身的 CWD 解析相对路径，这里统一转绝对，避免「文档给的相对路径」失效
    source = str(Path(args.source).resolve()) if args.source else str(REPO_ROOT / "examples" / "m3_sample_repo")
    print(f"后端: {args.base}\n样例: {source}\n模式: {args.mode}")

    try:
        if args.mode in ("full", "all"):
            pid = run_full(args.base, source if not args.project_id else None, args.project_id, args.no_env,
                           args.entry, args.light)
        else:
            pid = args.project_id or ""
        if args.mode in ("fail", "all"):
            if not pid:
                raise RuntimeError("fail 模式需 --project-id（或 all 模式连跑）")
            run_fail(args.base, pid)
    except Exception as e:  # noqa: BLE001
        print(f"\n中断: {e}")
        RESULTS.append(("执行中断", False, str(e)))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n==== 验收结果: {passed}/{total} 通过 ====")
    for name, ok, detail in RESULTS:
        print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))
    return 0 if total and passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
