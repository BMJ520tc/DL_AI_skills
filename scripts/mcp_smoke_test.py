"""MCP 冒烟测试：知识库 MCP 的「服务端协议闭环」与「agent 侧 tool-use 闭环」。

对应《开发计划》阶段0 任务 0.5 / M0 验收项「MCP 工具调用冒烟测试（agent 经 MCP 查询知识库一次）」，
以及审核已知边界 GB-2（此前无留痕）。

两个模式：

1. ``--mode stdio``（默认，确定性、不需要模型）：
   以 stdio JSON-RPC 直连 ``app.mcp.knowledge_mcp``，走 initialize → tools/list → tools/call，
   并把工具返回结果与直接查库结果比对（同一查询的命中条数必须一致）。

2. ``--mode agent``（需要后端在跑 + 模型端点可用，会真实消耗一次模型调用）：
   经 ``POST /api/agents/tasks`` 提交一个「必须调用 knowledge_search 工具」的任务，
   轮询到终态后读 ``data/agent_tasks/<task_id>/log.json``，把 agent 报告的命中条数与
   直接查库结果比对，验证 tool use 闭环真的打通。

用法：

    D:\\python.exe scripts/mcp_smoke_test.py
    D:\\python.exe scripts/mcp_smoke_test.py --mode agent --base-url http://127.0.0.1:8199
    D:\\python.exe scripts/mcp_smoke_test.py --mode all --base-url http://127.0.0.1:8199

证据输出：``data/_acceptance/mcp_smoke_<mode>_<时间戳>.json``
退出码：0 通过；1 失败（失败原因写入证据文件与 stderr）。
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
ACCEPTANCE_DIR = ROOT / "data" / "_acceptance"
AGENT_TASKS_DIR = ROOT / "data" / "agent_tasks"

QUERY = {"types": ["knowledge"], "q": "public", "limit": 5}
TERMINAL = ("success", "failed", "cancelled")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _http(base: str, method: str, path: str, body: dict | None = None, timeout: float = 60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def _db_count(types: list[str], q: str, limit: int = 50) -> tuple[int, list[str]]:
    """直接查库（不经过 MCP），作为工具返回结果的对照。"""
    sys.path.insert(0, str(BACKEND))
    from app.services import knowledge_service  # noqa: PLC0415

    rows = knowledge_service.search(types=types, q=q, limit=limit)
    return len(rows), [r["title"] for r in rows]


class McpStdioClient:
    """行分隔 JSON-RPC 的 stdio 客户端（knowledge_mcp 的帧格式）。"""

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "app.mcp.knowledge_mcp"],
            cwd=str(BACKEND),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._queue.put(line)
        self._queue.put(None)

    def send(self, msg: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def request(self, msg: dict) -> dict:
        self.send(msg)
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"MCP server 未在 {self.timeout}s 内响应请求 {msg.get('method')}")
            try:
                line = self._queue.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"MCP server 未响应 {msg.get('method')}") from None
            if line is None:
                raise RuntimeError("MCP server 提前退出（stdin 关闭或导入失败）")
            line = line.strip()
            if not line:
                continue
            resp = json.loads(line)
            if resp.get("id") == msg.get("id"):
                return resp

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def run_stdio() -> dict:
    """服务端协议闭环：initialize → tools/list → tools/call → 未知方法。"""
    evidence: dict = {"mode": "stdio", "python": sys.version.split()[0], "cwd": str(BACKEND)}
    checks: list[dict] = []
    client = McpStdioClient()
    try:
        init = client.request({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                               "params": {"protocolVersion": "2024-11-05", "capabilities": {}}})
        client.send({"jsonrpc": "2.0", "method": "notifications/initialized"})  # 通知不应有响应
        info = init.get("result", {}).get("serverInfo", {})
        evidence["server_info"] = info
        checks.append({"check": "initialize", "expect": "serverInfo.name=knowledge-mcp",
                       "actual": info.get("name"), "pass": info.get("name") == "knowledge-mcp"})

        tools = client.request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [t["name"] for t in tools.get("result", {}).get("tools", [])]
        evidence["tools"] = names
        checks.append({"check": "tools/list", "expect": "包含 knowledge_search",
                       "actual": names, "pass": "knowledge_search" in names})

        call = client.request({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                               "params": {"name": "knowledge_search", "arguments": QUERY}})
        payload = json.loads(call["result"]["content"][0]["text"])
        db_total, db_titles = _db_count(QUERY["types"], QUERY["q"], limit=50)
        expected_hits = min(QUERY["limit"], db_total)
        tool_titles = [row.get("title") for row in payload]
        evidence["tool_call"] = {
            "arguments": QUERY,
            "hits": len(payload),
            "titles": tool_titles,
            "db_total": db_total,
            "db_titles_top": db_titles[:len(payload)],
            "sample": payload[0] if payload else None,
        }
        checks.append({"check": "tools/call knowledge_search 条数受 limit 约束且与查库一致",
                       "expect": expected_hits, "actual": len(payload),
                       "pass": len(payload) == expected_hits and len(payload) > 0})
        checks.append({"check": "tools/call 返回的标题与查库前 N 条一致",
                       "expect": db_titles[:len(payload)], "actual": tool_titles,
                       "pass": tool_titles == db_titles[:len(payload)]})

        unknown = client.request({"jsonrpc": "2.0", "id": 4, "method": "no/such/method"})
        code = unknown.get("error", {}).get("code")
        checks.append({"check": "未知方法返回 JSON-RPC 错误", "expect": -32601,
                       "actual": code, "pass": code == -32601})

        # 空关键词过滤：仍应返回合法响应（不抛异常）
        empty = client.request({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                "params": {"name": "knowledge_search",
                                           "arguments": {"types": ["knowledge"], "q": "definitely-absent-xyz"}}})
        empty_payload = json.loads(empty["result"]["content"][0]["text"])
        checks.append({"check": "无命中查询返回空列表", "expect": [], "actual": empty_payload,
                       "pass": empty_payload == []})
    finally:
        client.close()
    evidence["checks"] = checks
    evidence["pass"] = all(c["pass"] for c in checks)
    return evidence


def run_agent(base_url: str, timeout_s: int = 300) -> dict:
    """agent 侧 tool-use 闭环：提交必须调用 MCP 工具的任务，核对 agent 报告的命中数。"""
    db_total, db_titles = _db_count(QUERY["types"], QUERY["q"])
    # agent 按 prompt 指定的 limit 检索，返回条数应与 stdio 模式一致：min(limit, 命中总数)
    db_count = min(QUERY["limit"], db_total)
    prompt = (
        "你有权访问一个名为 knowledge 的 MCP 服务，其工具为 knowledge_search（参数：types、q、limit）。\n"
        f"请调用该工具执行一次检索：types=[\"knowledge\"]，q=\"public\"，limit=5。\n"
        "然后只回答下面两行，不要做其它事、不要使用 Read/Write/Bash 等工具、不要读取任何文件：\n"
        "命中条数: <工具返回的条数>\n"
        "第一条标题: <工具返回的第一条标题>"
    )
    evidence: dict = {"mode": "agent", "base_url": base_url, "prompt": prompt,
                      "db_total": db_total, "db_hits": db_count, "db_titles": db_titles}
    checks: list[dict] = []

    created = _http(base_url, "POST", "/api/agents/tasks",
                    {"prompt": prompt, "max_turns": 6, "timeout_s": timeout_s, "attach_knowledge": True})
    task_id = created["task_id"]
    evidence["task_id"] = task_id

    deadline = time.monotonic() + timeout_s + 60
    task = None
    while time.monotonic() < deadline:
        task = _http(base_url, "GET", f"/api/tasks/{task_id}")
        if task["status"] in TERMINAL:
            break
        time.sleep(3)
    evidence["task_status"] = task["status"] if task else "unknown"
    evidence["task_error"] = (task or {}).get("error")
    checks.append({"check": "agent 任务到达终态", "expect": "success",
                   "actual": evidence["task_status"], "pass": evidence["task_status"] == "success"})

    log_path = AGENT_TASKS_DIR / task_id / "log.json"
    text = ""
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
        evidence["log"] = log
        text = log.get("result") or ""
    else:
        evidence["log"] = None
    evidence["agent_answer"] = text
    evidence["agent_turns"] = (evidence.get("log") or {}).get("num_turns")

    m = re.search(r"命中条数\s*[:：]\s*(\d+)", text)
    reported = int(m.group(1)) if m else None
    evidence["reported_hits"] = reported
    checks.append({"check": "agent 报告的命中数与查库一致", "expect": db_count,
                   "actual": reported, "pass": reported == db_count})

    first_title_ok = bool(db_titles) and db_titles[0] in text
    checks.append({"check": "agent 复述的第一条标题来自知识库", "expect": db_titles[0] if db_titles else None,
                   "actual": None if not first_title_ok else db_titles[0], "pass": first_title_ok})
    checks.append({"check": "agent 会话为多轮（发生工具调用）", "expect": "num_turns >= 2",
                   "actual": evidence["agent_turns"], "pass": (evidence["agent_turns"] or 0) >= 2})

    evidence["checks"] = checks
    evidence["pass"] = all(c["pass"] for c in checks)
    return evidence


def main() -> int:
    p = argparse.ArgumentParser(description="知识库 MCP 冒烟测试（stdio 协议 / agent tool-use）")
    p.add_argument("--mode", choices=("stdio", "agent", "all"), default="stdio")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--timeout", type=int, default=300, help="agent 模式单任务超时（秒）")
    args = p.parse_args()

    ACCEPTANCE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    if args.mode in ("stdio", "all"):
        results.append(run_stdio())
    if args.mode in ("agent", "all"):
        try:
            results.append(run_agent(args.base_url.rstrip("/"), args.timeout))
        except (urllib.error.URLError, OSError) as e:
            results.append({"mode": "agent", "pass": False,
                            "error": f"后端不可达（{args.base_url}）：{e}"})

    overall = all(r.get("pass") for r in results)
    out = {"run_at": datetime.now(timezone.utc).isoformat(), "overall_pass": overall, "results": results}
    out_path = ACCEPTANCE_DIR / f"mcp_smoke_{args.mode}_{_now_stamp()}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in results:
        print(f"[{r['mode']}] {'PASS' if r.get('pass') else 'FAIL'}")
        for c in r.get("checks", []):
            print(f"   {'✓' if c['pass'] else '✗'} {c['check']} (expect={c['expect']}, actual={c['actual']})")
        if r.get("error"):
            print(f"   ! {r['error']}")
    print(f"证据: {out_path.relative_to(ROOT)}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
