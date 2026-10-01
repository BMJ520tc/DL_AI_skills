"""全链路驱动脚本共用工具：极简 HTTP 客户端与后台任务轮询。

供 scripts/run_*_pipeline.py 复用（沿用 scripts/_viz_common.py 的共用模块先例）。
仅依赖标准库，直接驱动后端 HTTP API。
"""
import json
import time
import urllib.error
import urllib.request

TERMINAL = ("success", "failed", "cancelled")


class ApiError(RuntimeError):
    """HTTP 或任务失败。"""


def request(base: str, method: str, path: str, body: dict | None = None, timeout: float = 60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise ApiError(f"{method} {path} -> HTTP {e.code}: {e.read().decode(errors='ignore')[:200]}")
    except OSError as e:
        raise ApiError(f"{method} {path} -> {e}")
    return json.loads(raw) if raw else None


def get(base: str, path: str):
    return request(base, "GET", path)


def wait_task(base: str, task_id: str, timeout_s: int, interval: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        task = get(base, f"/api/tasks/{task_id}")
        if task["status"] in TERMINAL:
            return task
        time.sleep(interval)
    raise ApiError(f"任务 {task_id} 超时（>{timeout_s}s）")


def run_step(base: str, label: str, path: str, body: dict | None, timeout_s: int, interval: float) -> dict:
    """提交一步任务并等到终态；失败抛 ApiError；成功打印进度。"""
    task_id = request(base, "POST", path, body or {})["task_id"]
    print(f"  [{label}] queued {task_id}", flush=True)
    task = wait_task(base, task_id, timeout_s, interval)
    if task["status"] != "success":
        raise ApiError(f"{label} 失败: {task.get('error') or task['status']}")
    progress = task.get("progress")
    print(f"  [{label}] {task['status']} {progress}" if progress else f"  [{label}] {task['status']}", flush=True)
    return task
