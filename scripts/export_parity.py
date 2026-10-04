"""导出两端逐字节比对（阶段4 4c「导出即所存即所训」的机器化复核）。

《模块详细设计》7.4 声明「导出统一走后端再生成引擎……实测与前端 codeCompile 输出逐字节
一致」；7.7-2 定「后端为唯一导出源」。本脚本把该声明变成**可复跑**的比对：用同一张存好的
GraphIR，前端走真实生成器（`frontend/scripts/gb1_export_harness.ts`，esbuild 打包 + node），
后端走 `app.services.network_export.generate`，再逐字节比较。

内置图是一条覆盖多种标准节点的链（Linear / 激活 / Dropout / BatchNorm / Conv / Pool /
Flatten / Softmax / 位置编码 / Linear），比单测里的 Linear→ReLU→Linear 覆盖更广。

前置：node / npx 可用、frontend/node_modules 已安装。
用法：D:\\python.exe scripts/export_parity.py            # 用内置图
      D:\\python.exe scripts/export_parity.py --graph g.json   # 用指定 GraphIR（{nodes, edges}）
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = REPO_ROOT / "frontend"
HARNESS = FRONTEND / "scripts" / "gb1_export_harness.ts"


def _builtin_graph() -> dict:
    """覆盖多种标准节点的链；Linear 是唯一参数必填的节点，其余吃默认值。"""
    chain = [
        ("n1", "input_layer", {}),
        ("n2", "linear_layer", {"in_features": 8, "out_features": 16, "bias": True}),
        ("n3", "relu_layer", {}),
        ("n4", "dropout_layer", {}),
        ("n5", "batchnorm2d_layer", {}),
        ("n6", "conv2d_layer", {}),
        ("n7", "maxpool2d_layer", {}),
        ("n8", "flatten_layer", {}),
        ("n9", "sigmoid_layer", {}),
        ("n10", "softmax_layer", {}),
        ("n11", "positional_encoding_layer", {}),
        ("n12", "linear_layer", {"in_features": 16, "out_features": 3, "bias": False}),
    ]
    nodes = [{"id": nid, "type": typ, "data": data} for nid, typ, data in chain]
    edges = [
        {"id": f"e{i}", "source": chain[i - 1][0], "target": chain[i][0],
         "targetHandle": "in-0", "data": {"label": f"out_{chain[i - 1][0]}"}}
        for i in range(1, len(chain))
    ]
    return {"nodes": nodes, "edges": edges}


def _control_flow_graph(kind: str) -> dict:
    """控制流容器图（`repeat_layer` / `module_list`），形状**照真实画布**：

    - 容器节点的 `data.internalNodes/internalEdges` 是内部图的副本（画布侧由
      `containerLogic.syncContainerData` 维护、`graphIR.buildGraphIR` 原样写进 graph.json）；
    - 子节点**同时**出现在顶层 `nodes`（带 `parentId`）——React Flow 平铺渲染，画布上本来如此；
    - 容器与子节点之间的 `in-internal` / `out-internal` 边界边也在顶层 `edges` 里。
    """
    child = {"id": "c1", "type": "linear_layer", "parentId": "ctr", "extent": "parent",
             "position": {"x": 20, "y": 40},
             "data": {"in_features": 4, "out_features": 4}}
    internal_edges = [
        {"id": "ie1", "source": "ctr", "sourceHandle": "in-internal", "target": "c1",
         "targetHandle": "in-0", "data": {"label": "c1_in"}},
        {"id": "ie2", "source": "c1", "target": "ctr", "targetHandle": "out-internal",
         "data": {"label": "out_c1"}},
    ]
    return {
        "nodes": [
            {"id": "in1", "type": "input_layer", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "ctr", "type": kind, "position": {"x": 200, "y": 0}, "data": {
                "repetitions": 2, "internalNodes": [child], "internalEdges": internal_edges}},
            child,
        ],
        "edges": [
            {"id": "e1", "source": "in1", "target": "ctr",
             "targetHandle": "in-external" if kind == "repeat_layer" else "in",
             "data": {"label": "out_in1"}},
            *internal_edges,
        ],
    }


def _fixtures() -> list[tuple[str, dict]]:
    """内置比对图：标准节点链 + 两种控制流容器（各覆盖一类此前被拒的图）。"""
    return [
        ("标准节点链（12 种节点）", _builtin_graph()),
        ("repeat_layer 容器（含嵌套子节点）", _control_flow_graph("repeat_layer")),
        ("module_list 容器（含嵌套子节点）", _control_flow_graph("module_list")),
    ]


def _frontend_code(graph: dict, work: Path) -> bytes:
    if shutil.which("npx") is None or shutil.which("node") is None:
        raise SystemExit("跳过：本机缺 node/npx，无法跑前端生成器")
    graph_path = work / "graph.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    bundle = work / "harness.cjs"
    out = work / "front.py"
    # Windows 下 npx 是 .cmd，须经 shell 解析（无用户输入，路径均由本脚本生成）
    subprocess.run(
        ["npx", "esbuild", str(HARNESS), "--bundle", "--platform=node", "--format=cjs",
         f"--outfile={bundle}", "--log-level=warning"],
        cwd=str(FRONTEND), check=True, shell=True,
    )
    subprocess.run(
        ["node", str(bundle), "--out", str(out), "--graph", str(graph_path)],
        cwd=str(FRONTEND), check=True, shell=True,
    )
    return out.read_bytes()


def _backend_code(graph: dict) -> bytes:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.services import network_export

    return network_export.generate(graph).encode("utf-8")


def _undefined_node_refs(code: str) -> list[str]:
    """代码里引用到、但没有被赋值过的节点输出变量（`out_*`）。

    逐字节一致只保证两端相同，**不保证代码可运行**——曾出现「有声明源句柄的节点遇到
    无 sourceHandle 的旧边时凭空造名」的缺陷：两端一字不差，却引用未定义变量。
    """
    assigned: set[str] = set()
    for line in code.splitlines():
        stripped = line.strip()
        if "=" not in stripped or stripped.startswith("#") or stripped.startswith("def "):
            continue
        lhs = stripped.split("=")[0].strip()
        if " " not in lhs and lhs:
            assigned.add(lhs)
    used = set(re.findall(r"\bout_[A-Za-z0-9_]+\b", code))
    return sorted(v for v in used - assigned if not v.startswith(("out_features", "out_channels")))


def main() -> int:
    argv = sys.argv[1:]
    if "--graph" in argv:
        path = Path(argv[argv.index("--graph") + 1])
        fixtures = [(path.name, json.loads(path.read_text(encoding="utf-8")))]
    else:
        fixtures = _fixtures()

    failed = 0
    for name, graph in fixtures:
        with tempfile.TemporaryDirectory() as td:
            front = _frontend_code(graph, Path(td))
        back = _backend_code(graph)
        if front != back:
            failed += 1
            print(f"[FAIL] {name}：导出两端不一致（前端 {len(front)} / 后端 {len(back)} 字节）")
            fl = front.decode("utf-8", "replace").split("\n")
            bl = back.decode("utf-8", "replace").split("\n")
            for i in range(max(len(fl), len(bl))):
                f = fl[i] if i < len(fl) else "<缺行>"
                b = bl[i] if i < len(bl) else "<缺行>"
                if f != b:
                    print(f"  首个差异 行{i + 1}:\n    前端: {f!r}\n    后端: {b!r}")
                    break
            continue
        undef = _undefined_node_refs(back.decode("utf-8", "replace"))
        if undef:
            failed += 1
            print(f"[FAIL] {name}：两端一致但引用了未定义变量 {undef}")
            continue
        print(f"[PASS] {name}：逐字节一致（{len(back)} 字节）且无未定义节点输出变量")

    if failed:
        print(f"\n{len(fixtures) - failed}/{len(fixtures)} 张图通过")
        return 1
    print(f"\n{len(fixtures)}/{len(fixtures)} 张图全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
