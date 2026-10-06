"""环境管理（模块详细设计 2.3）。

决策树: Dockerfile → 容器（可选扩展，需本机 docker）; environment.yml → conda; 否则 venv。
容器分支按 2.3 步骤 1「无 docker 时此分支不支持并明确提示」先探测本机 docker：
探测不到 → 明确提示用 conda/venv；探测到但本仓库没有容器环境创建实现 → 也是明确的可执行报错，
绝不静默改道、也不假装成功（见 _container_unavailable_reason）。
版本判断（2.3 步骤 2 与「异常与边界」）:
  ① 清单/元数据声明的 Python 要求驱动 venv 解释器选择（选不到即告警回退，见 resolve_env_python）；
  ② 驱动 CUDA 与依赖清单里的 torch/CUDA 要求不匹配 → 记录结论并选 CPU 版 wheel（见 plan_cuda）。
依赖修正循环: 安装失败 → agent 判断 → 降级/替换/移除 → 重试（上限 3 次）。
环境创建步骤与每次安装尝试写 run_record(env_install)，报错入 error 字段（供蒸馏知识提炼，需求六.1）。
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import CONDA_PATH, ENV_VENV_PYTHON, PIP_FALLBACK_INDEX, PIP_INDEX_URL, project_env_dir
from app.services import agent_service, knowledge_service, long_paths, proc_util, project_manager, prompts, task_manager

ENV_TASK_TYPE = "env_create"
# 单次 pip 安装上限（超时即杀进程树）。**可配**：真实项目里 torch(+cuXXX)/dgl 这类
# 大 wheel 动辄 2GB+，600s 会把「下载慢」误判成「装不上」——实测 ATBAN-DTI 就是这样
# 被拖进依赖修正循环的（它的修正建议里写着「本次失败是 pip install 硬超时（>600s），
# 不是版本冲突」）。装不动的判断应当由 pip 自己给（无匹配版本/编译失败），而不是墙钟。
INSTALL_TIMEOUT_S = int(os.getenv("ENV_INSTALL_TIMEOUT_S", "600"))

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "修正后的完整依赖清单，每行一条（pip requirements 语法），供直接替换安装",
        },
        "reason": {"type": "string", "description": "修正理由：改了哪些包、为什么"},
    },
    "required": ["requirements", "reason"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_env_type(workspace: Path) -> str:
    source = workspace / "source"
    if (source / "Dockerfile").exists():
        return "container"
    if (source / "environment.yml").exists() or (source / "environment.yaml").exists():
        return "conda"
    return "venv"


def _detect_docker() -> Optional[str]:
    """探测本机 docker CLI（DOCKER_PATH 可覆盖，口径同 config 的 CONDA_EXE）。

    只探 CLI 是否在 PATH，不连 daemon：设计 2.3 只要求「需本机 docker」这一可用性判据；
    daemon 未启动会在真正执行容器命令时露出真实报错，不在探测阶段臆断。
    """
    return shutil.which(os.getenv("DOCKER_PATH") or "docker")


def _container_unavailable_reason() -> str:
    """容器通道不可用的明确原因（据此给出可执行的下一步）。

    本仓库现状（已核实）：backend/runner.py 是**形状追踪**的独立服务，按预构建镜像
    `torchlens-worker:latest` 跑 backend/worker.py，并不存在「按项目 Dockerfile 建/运行项目环境」
    的容器环境创建实现——故探测到 docker 也不能接上，只能如实报错。
    """
    docker = _detect_docker()
    if docker is None:
        return (
            "项目带 Dockerfile，需容器环境；但本机未检测到 docker，容器通道不可用；"
            "请用 conda/venv（提供 environment.yml 或 requirements.txt，"
            "或设 DOCKER_PATH 指向 docker 可执行文件后重试）"
        )
    return (
        f"项目带 Dockerfile，本机已检测到 docker（{docker}），但本仓库尚未实现容器环境创建："
        "backend/runner.py 仅是形状追踪的独立服务（按预构建镜像 torchlens-worker:latest 运行 worker.py），"
        "不按项目 Dockerfile 建/运行项目环境 → 容器通道不可用；请用 conda/venv"
        "（提供 environment.yml 或 requirements.txt）"
    )


def _progress_reporter(task_id: str):
    """任务进度写手：`update_progress` 是**整体覆盖**，故这里维护一份累积 dict，
    每次写入都带上既有键（env_precheck / env_cuda / env_fix …），滚动 stage 时不冲掉它们。
    """
    state: dict = {}

    def report(stage: str, **extra) -> None:
        state.update(extra)
        task_manager.update_progress(task_id, {**state, "stage": stage})

    return report


def create_env(project_id: str) -> str:
    project_manager.require_type(project_id, {"original"})
    return task_manager.create_task(ENV_TASK_TYPE, project_id=project_id, params={"project_id": project_id})


def get_env_status(project_id: str) -> dict:
    """环境状态 + 路径：`env_dir` 为短路径根下的目录，`env_python` 为解释器（未就绪为 None）。

    供界面与验收脚本**问后端**取环境位置，不必各自猜 `ws/env`（环境已迁到短路径根，见
    config.project_env_dir）。
    """
    project = project_manager.get_project(project_id)
    if project is None:
        return {"project_id": project_id, "status": None}
    from app.services import analysis_service  # 延迟导入：避免 env_manager ↔ analysis_service 环
    ws = Path(project["workspace_path"])
    return {
        "project_id": project_id,
        "status": project["status"],
        "env_dir": str(project_env_dir(project_id)),
        "env_python": analysis_service._project_python(ws, project_id),
    }


async def _run_env_create(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = project_manager.get_project(project_id)
    if project is None:
        raise RuntimeError("project not found")
    ws = Path(project["workspace_path"])
    source = ws / "source"
    env_type = detect_env_type(ws)

    if env_type == "container":
        # 2.3 步骤 1：先探测本机 docker 再决定提示——有 docker 也不假装能建容器环境（如实报错）。
        raise RuntimeError(_container_unavailable_reason())

    # 环境落在**短路径根**（`<ENV_ROOT>/<项目id前8位>`），规避 Windows 260 上限（见 config.project_env_dir）。
    env_dir = project_env_dir(project_id)
    legacy_env = ws / "env"   # 迁移前的老位置；重建时一并清掉，避免新旧环境并存被误用
    created_at = _now()
    report = _progress_reporter(task_id)

    # 8 位 id 目录若已属别的项目 → 明确报错（碰撞概率极低，但绝不静默串环境）
    marker = env_dir / ".project_id"
    if env_dir.exists() and marker.exists():
        try:
            existing = marker.read_text(encoding="utf-8").strip()
        except OSError:
            existing = ""
        if existing and existing != project_id:
            raise RuntimeError(
                f"环境目录 {env_dir} 已属于项目 {existing}（8 位 id 冲突）；"
                f"请设置 DL_AI_ENV_ROOT 指向其它目录后重试"
            )

    # 重建环境前必须清空目标目录：venv/conda 创建都不会清理已存在的目录，
    # 旧解释器的 site-packages（例如 cp312 的 numpy）会残留并与新环境混装 → 导入即失败。
    cleaned: list[str] = []
    for target in (env_dir, legacy_env):
        if not target.exists():
            continue
        shutil.rmtree(target, ignore_errors=True)
        if target.exists():           # Windows 上可能因占用/杀软扫描短暂锁定 → 稍等重试一次
            time.sleep(2)
            shutil.rmtree(target, ignore_errors=True)
        if target.exists():
            # 删除不完整时 venv 覆写 python.exe 会报 Permission denied，必须报清楚而不是让它神秘失败
            raise RuntimeError(
                f"旧环境目录无法删除（可能被进程占用或杀软扫描中）: {target}；请关闭占用后重试"
            )
        cleaned.append(str(target))
    if cleaned:
        knowledge_service.record_run(
            {"project_id": project_id, "task_id": task_id, "run_type": "env_install",
             "environment": {"type": env_type}, "params": {"step": "env_clean"},
             "command": f"清理旧环境目录 {'；'.join(cleaned)}", "status": "success",
             "started_at": created_at, "finished_at": _now()}
        )
    env_dir.mkdir(parents=True, exist_ok=True)
    try:
        marker.write_text(project_id, encoding="utf-8")
    except OSError:
        pass
    interp = "python"
    if env_type == "conda":
        cmd = _conda_create_cmd(source, env_dir)
        python_note: dict = {}
        report("创建 conda 环境…（首次可能较慢）")
    else:
        # 语言版本生效（2.3 步骤 2）：清单声明的 Python 要求 → venv 解释器选择；
        # 选不到匹配解释器时告警回退（不静默），没有要求时行为与旧版一致（后端解释器 / ENV_VENV_PYTHON）。
        python_cmd, python_note = resolve_env_python(source)
        cmd = [*python_cmd, "-m", "venv", str(env_dir)]
        interp = Path(str(python_cmd[0])).name if python_cmd else "python"
        report(f"创建 venv 环境（解释器 {interp}）…")
    create_params: dict = {"step": "env_create"}
    if python_note.get("required"):
        create_params["python_selection"] = python_note
        if python_note.get("warning"):
            report(f"创建 venv 环境（解释器回退：{interp}）…", env_python=python_note)
    err = await _create_env_dir(env_type, source, env_dir, cmd)
    if err is not None:
        knowledge_service.record_run(
            {"project_id": project_id, "task_id": task_id, "run_type": "env_install",
             "environment": {"type": env_type}, "params": create_params,
             "command": " ".join(cmd), "status": "failed", "error": err,
             "started_at": created_at, "finished_at": _now()}
        )
        project_manager.update_status(project_id, "env_failed")
        _draft_dependency_conflict(project_id, task_id, f"创建({env_type})", err)
        raise RuntimeError(f"环境创建失败({env_type}): {err}")

    knowledge_service.record_run(
        {"project_id": project_id, "task_id": task_id, "run_type": "env_install",
         "environment": {"type": env_type}, "params": create_params,
         "command": " ".join(cmd), "status": "success",
         "started_at": created_at, "finished_at": _now()}
    )

    report("环境已创建，开始安装依赖…")
    ok = await _install_with_fix(source, env_dir, project_id, task_id, env_type, report, ws=ws)
    if ok:
        # 仓库自己若也发在 PyPI，依赖清单一引就会装成安装版 → 运行时是「仓库入口类 + PyPI 子模块」
        # 的缝合怪（boltz 实测）。这里换成 editable，从根上避免。
        await _install_repo_editable(source, env_dir, task_id, report)
        project_manager.update_status(project_id, "env_ready")
        report("环境就绪 ✓")
        # 建环境顺带把仓库引用的模型权重下下来（失败不连坐环境，如实登记）
        await _fetch_weights(source, ws, task_id, report)
    else:
        project_manager.update_status(project_id, "env_failed")
        _draft_dependency_conflict(project_id, task_id, "依赖安装（修正循环耗尽）", "")
        raise RuntimeError("环境安装失败（依赖修正循环耗尽）" + _long_path_hint(project_id))


# --------------------------- 建环境顺带取模型权重 ---------------------------

# 仓库里引用的权重地址（boltz 那种在 main.py 里写成常量列表）
_WEIGHT_URL_RE = re.compile(
    r"https?://[^\s\"'<>)\]}]+\.(?:ckpt|pt|pth|safetensors|bin)\b", re.IGNORECASE
)
# 模型权重下载范围：entry=只下与入口类同名的那几个（Boltz2 → boltz2_*）；all=仓库引用到的全部
ENV_WEIGHT_SCOPE = os.getenv("ENV_WEIGHT_SCOPE", "entry").strip().lower()
ENV_FETCH_WEIGHTS = os.getenv("ENV_FETCH_WEIGHTS", "1") != "0"


def _mirror_url(url: str) -> str:
    """HuggingFace 官方域名换镜像（本机实测 huggingface.co 不可达、hf-mirror.com 可达）。

    注意：镜像对**大文件**会 302 回官方 CDN，所以下载必须带断点续传 + 重试。
    """
    return url.replace("https://huggingface.co/", "https://hf-mirror.com/")


def plan_weight_downloads(source: Path, entry_class: Optional[str] = None) -> list[dict]:
    """扫仓库源码里引用的权重地址 → 按**文件名去重**（HF 换镜像）→ 可选按入口类名过滤。

    只扫 `.py`（引用都写在代码里）；同名地址只保留一条，优先镜像地址。
    """
    by_name: dict[str, str] = {}
    for py in source.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for raw in _WEIGHT_URL_RE.findall(text):
            name = raw.rsplit("/", 1)[-1]
            if not name:
                continue
            url = _mirror_url(raw)
            # 已有镜像地址就别被官方地址覆盖（反过来可以）
            if name not in by_name or "hf-mirror" in url:
                by_name[name] = url
    plans = [{"filename": n, "url": u} for n, u in sorted(by_name.items())]
    if entry_class and ENV_WEIGHT_SCOPE != "all":
        key = entry_class.strip().lower()
        filtered = [p for p in plans if key and key in p["filename"].lower()]
        if filtered:  # 过滤后为空则退回全部，避免"因为名字对不上就一个都不下"
            plans = filtered
    return plans


async def _fetch_weights(source: Path, ws: Path, task_id: str, report) -> None:
    """把仓库引用的权重下到 `<ws>/data/`（脚本按约定会自动发现）。

    - **断点续传**（`curl -C -`）+ 重试：镜像对大文件会 302 回官方 CDN，实测会中途停滞。
    - **不连坐**：某个权重下不下来只记日志，环境仍算就绪（权重不是每个仓库都需要）。
    """
    if not ENV_FETCH_WEIGHTS:
        return
    try:
        entry_class = _entry_class_of(ws)
        plans = plan_weight_downloads(source, entry_class)
    except Exception as e:  # noqa: BLE001 —— 扫描失败不该影响环境
        report("权重扫描失败（不影响环境就绪）", weights_error=str(e)[:200])
        return
    if not plans:
        return

    dest_dir = ws / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    report(f"取模型权重：{len(plans)} 个（{', '.join(p['filename'] for p in plans)}）",
           weights_planned=[p["filename"] for p in plans])
    done: list[str] = []
    failed: list[str] = []
    for i, p in enumerate(plans, start=1):
        dest = dest_dir / p["filename"]
        label = f"权重 {i}/{len(plans)} {p['filename']}"
        ok, note = await _download_weight(p["url"], dest, label, report)
        if ok:
            done.append(p["filename"])
        else:
            failed.append(f"{p['filename']}：{note}")
    fields = {"weights_done": done}
    if failed:
        fields["weights_failed"] = failed
    report(
        f"权重取回 {len(done)}/{len(plans)} 个" + (f"（失败 {len(failed)} 个）" if failed else ""),
        **fields,
    )


def _entry_class_of(ws: Path) -> Optional[str]:
    """从项目当前的 IR 里取入口类名（用于按名字挑该模型的权重）；取不到就算了。"""
    try:
        ir = json.loads((ws / "reports" / "ir.json").read_text(encoding="utf-8"))
        value = ir.get("entry_class")
        return str(value) if value else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


async def _download_weight(url: str, dest: Path, label: str, report) -> tuple[bool, str]:
    """下单个权重（断点续传 + 重试）。返回 (是否成功, 失败说明)。"""
    cmd = [
        "curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
        "-sS", "-o", str(dest), url,
    ]
    task = asyncio.create_task(proc_util.run_command(cmd))
    try:
        while not task.done():
            await asyncio.sleep(5)
            size = dest.stat().st_size if dest.exists() else 0
            report(f"{label}：已下 {size // 1048576} MB…", weights_current={
                "file": dest.name, "downloaded_mb": size // 1048576,
            })
        rc, out = await task
    except Exception as e:  # noqa: BLE001
        return False, f"下载异常 {e}"
    if rc != 0:
        return False, (out or f"curl 退出码 {rc}")[-300:]
    if not dest.exists() or dest.stat().st_size == 0:
        return False, "下载后文件为空"
    report(f"{label}：完成（{dest.stat().st_size // 1048576} MB）")
    return True, ""


def _repo_package_name(source: Path) -> Optional[str]:
    """仓库**自己发布**的包名（pyproject.toml / setup.cfg / setup.py）。"""
    try:
        text = (source / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', text, re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    for fn in ("setup.cfg", "setup.py"):
        try:
            text = (source / fn).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', text)
        if m:
            return m.group(1)
    return None


async def _install_repo_editable(source: Path, env_dir: Path, task_id: str, report) -> None:
    """把**仓库自身**按 editable 装上（`pip install -e . --no-deps`）。

    为什么需要：很多仓库把自己也发到 PyPI，依赖清单一引就装成了**安装版**——运行时变成
    「仓库的入口类 + PyPI 的子模块」这种**缝合怪**（boltz 实测：仓库的 `boltz2.py` 配
    site-packages 的 `boltz 2.2.1`，报 `AttentionPairBias.forward() got an unexpected
    keyword argument`，**追踪的根本不是仓库这份代码**）。editable 装上后 `import` 直接命中仓库源码。
    `--no-deps`：依赖已由主安装装过，这里只换来源、不动依赖图。失败**不连坐**（如实登记）——
    加载侧还有「优先仓库包」的兜底。
    """
    if not (source / "pyproject.toml").is_file() and not (source / "setup.py").is_file() \
            and not (source / "setup.cfg").is_file():
        return
    pip = _env_pip(env_dir)
    cmd = [pip, "install", "-e", str(source), "--no-deps",
           "--retries", str(PIP_RETRIES), "--timeout", str(PIP_TIMEOUT_S)]
    try:
        rc, out = await proc_util.run_command(cmd, timeout=INSTALL_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        report(f"仓库源码安装（editable）异常：{str(e)[:200]}")
        return
    name = _repo_package_name(source) or "（未识别到包名）"
    if rc == 0:
        report(f"已把仓库自身按 editable 安装（{name}）：运行时用仓库源码，不会混 PyPI 版")
    else:
        report(f"仓库自身 editable 安装失败（{name}）——加载侧会兜底优先仓库源码",
               repo_editable_error=(out or "")[-300:])


def _long_path_hint(project_id: str) -> str:
    """安装失败若属 Windows 260 字符路径上限，附上可操作指引（界面据「长路径」字样给出开启按钮）。"""
    try:
        latest = knowledge_service.get_latest_run(project_id, "env_install", "failed") or {}
        blocked = long_paths.is_long_path_error(latest.get("error") or "")
    except Exception:  # noqa: BLE001 —— 提示失败不影响原始报错
        return ""
    if blocked and not long_paths.is_enabled():
        return ("（原因：Windows 路径超过 260 字符上限——该依赖的深层路径过长；"
                "请在界面上点「开启长路径支持」授权一次，再重试建环境）")
    return ""


def _draft_dependency_conflict(project_id: str, task_id: str, stage: str, err: str) -> None:
    """环境装不上 → 起草一条 dependency_conflict 蒸馏知识（需求六.1「任务结束提炼入库」）。

    只写 draft（待确认），失败绝不影响环境创建本身的报错路径。
    """
    try:
        latest = knowledge_service.get_latest_run(project_id, "env_install", "failed") or {}
        detail = (latest.get("error") or err or "").strip()
        sources = []
        if latest.get("run_id"):
            sources.append({"type": "run_record", "ref": latest["run_id"]})
        knowledge_service.record_knowledge({
            "type": "dependency_conflict",
            "title": f"环境自建失败（{stage}）：项目 {project_id[:8]}",
            "content": (
                f"阶段：{stage}；失败详情（依赖修正循环耗尽后的原始报错）：{detail[:800]}。"
                "后续同类项目遇到相同报错时，优先按此调整依赖版本或换环境类型。"
            ),
            "structured": {"project_id": project_id, "stage": stage, "error": detail[:2000],
                           "source_task_id": task_id},
            "sources": sources,
            "confidence": "low",
            "scope": {"project_id": project_id},
            "status": "draft",
        })
    except Exception:  # noqa: BLE001 —— 蒸馏是旁路，绝不影响主流程报错
        pass


def _conda_create_cmd(source: Path, env_dir: Path) -> list[str]:
    """conda 环境创建命令：environment.yml/yaml 优先，否则建基础环境 + 后续 pip 装依赖。"""
    if CONDA_PATH is None:
        raise RuntimeError("未找到 conda 可执行文件")
    for name in ("environment.yml", "environment.yaml"):
        env_yml = source / name
        if env_yml.exists():
            return [CONDA_PATH, "env", "create", "-f", str(env_yml), "-p", str(env_dir), "-y"]
    return [CONDA_PATH, "create", "-p", str(env_dir), "python=3.11", "-y"]


def _conda_update_cmd(source: Path, env_dir: Path) -> list[str]:
    """conda 环境更新命令（environment.yml 的 pip: 段重跑；prefix 已存在时 update 可用）。

    注意：conda env update 不支持 -y（env create 才支持），故不加该参数。
    """
    if CONDA_PATH is None:
        raise RuntimeError("未找到 conda 可执行文件")
    for name in ("environment.yml", "environment.yaml"):
        env_yml = source / name
        if env_yml.exists():
            return [CONDA_PATH, "env", "update", "-f", str(env_yml), "-p", str(env_dir)]
    return [CONDA_PATH, "env", "update", "-p", str(env_dir)]


def _decode_err(e: subprocess.CalledProcessError) -> str:
    err = e.stderr or b""
    return err.decode(errors="ignore")[-2000:] if isinstance(err, bytes) else str(err)[-2000:]


async def _run_create(cmd: list[str], index_url: Optional[str]) -> None:
    """执行环境创建命令；失败抛 RuntimeError（输出尾部入错误信息）。

    走 proc_util：任务取消/超时即杀**进程树**，避免「任务已 cancelled 但 conda/pip 还在写环境」。
    """
    env = {**os.environ, "PIP_INDEX_URL": index_url} if index_url else None
    rc, out = await proc_util.run_command(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"环境创建命令失败（rc={rc}）: {out[-2000:]}")


async def _create_env_dir(env_type: str, source: Path, env_dir: Path, cmd: list[str]) -> Optional[str]:
    """执行环境创建命令，成功返回 None、失败返回错误文本。

    conda 的 environment.yml 常带 `pip:` 段（torch 等），其 pip 安装发生在 conda 内部、
    不经过 _install_with_fix，故索引不可达时同样切备源重试一次（用 env update 覆盖 pip 段）。
    """
    try:
        await _run_create(cmd, None)
        return None
    except RuntimeError as e:
        err = str(e)
    if env_type == "conda" and PIP_FALLBACK_INDEX and _is_index_error(err):
        try:
            await _run_create(_conda_update_cmd(source, env_dir), PIP_FALLBACK_INDEX)
            return None
        except RuntimeError as e2:
            return str(e2)
    return err


async def _install_with_fix(
    source: Path, env_dir: Path, project_id: str, task_id: str, env_type: str, report=None,
    ws: Optional[Path] = None,
) -> bool:
    def _report(stage: str, **extra) -> None:
        if report is not None:
            report(stage, **extra)
        else:   # 未传写手（直接调用本函数的场景）：仍写进度，保持既有契约
            task_manager.update_progress(task_id, {**extra, "stage": stage})
    pip = _env_pip(env_dir)
    req_file = _find_requirements(source)
    # 安装前预检（需求六.1、8.2「安装依赖前提示绕开方案」）：把已确认的 dependency_conflict
    # 预警出来，并把可操作的版本钉**预应用**到依赖清单副本——已知冲突在任何安装尝试之前被绕开，
    # 而不是等到装挂了再靠 agent 修正。预警与「已应用」合并成**一次**进度写入（update_progress 是整体覆盖）。
    conflicts = _dependency_precheck(project_id)
    applied = _apply_known_pins(req_file, _pins_from_conflicts(conflicts))
    precheck_progress: dict = {}
    if conflicts:
        precheck_progress["env_precheck"] = {
            "count": len(conflicts),
            "items": [{"knowledge_id": c.get("knowledge_id"), "title": c.get("title"),
                       "content": (c.get("content") or "")[:400]} for c in conflicts],
        }
    if applied:
        precheck_progress["env_precheck_applied"] = applied["applied"]
        req_file = Path(applied["path"])
    if precheck_progress:
        _report("依赖预检完成（已应用已知版本钉）", **precheck_progress)
    versions = detect_versions(source, _env_python(env_dir))
    cuda_plan = versions.get("cuda_plan") or {}
    extra_index_url = None
    if cuda_plan.get("action") == "cpu_wheel":
        # CUDA 降级结论必须可见：进任务进度，且随 environment.cuda_plan 进每次安装的 run_record
        extra_index_url = cuda_plan.get("index")
        _report("检测到 CUDA 要求，改用 CPU 版 wheel", env_cuda=cuda_plan)
    elif cuda_plan.get("action") == "cuda_wheel":
        # 有 NVIDIA 驱动 → **先把 torch 从 CUDA 轮子索引装上**，再做主安装。
        # 单独一步是**确定性**的做法：`--extra-index-url` 只在版本比较上让 `2.14.1+cu126` 赢过
        # `2.14.1`（本地版本标签更大），一旦清单把 torch 钉成精确版本就未必生效。
        extra_index_url = cuda_plan.get("index")
        _report("检测到 NVIDIA 驱动，装 CUDA 版 torch", env_cuda=cuda_plan)
        await _preinstall_torch(pip, cuda_plan, versions, PIP_INDEX_URL, project_id, task_id,
                                report=_report)
    index_url = PIP_INDEX_URL  # None → 用 pip 自身配置（用户 pip.ini）

    attempt = 0
    ok = False
    for attempt in range(1, 4):
        _report(f"安装依赖（第 {attempt}/3 次尝试）…（下载/安装可能数分钟）")
        result = await _try_install(pip, req_file, index_url, extra_index_url)
        _record_install(project_id, task_id, attempt, result, versions, env_type)
        if result["ok"]:
            ok = True
            break
        # 索引不可达（非依赖冲突）：切备源重试一次，不消耗依赖修正循环
        if index_url != PIP_FALLBACK_INDEX and _is_index_error(result["error"]):
            index_url = PIP_FALLBACK_INDEX
            _report("主索引不可达，切换备源重试…")
            result = await _try_install(pip, req_file, index_url, extra_index_url)
            _record_install(project_id, task_id, attempt, result, versions, env_type, step="pip_install_fallback")
            if result["ok"]:
                ok = True
                break
        # Windows 260 字符路径上限：同一路径必然再失败，重试只是白等几分钟 → 立刻中止，
        # 由界面给出「开启长路径支持」入口（见 services/long_paths）
        if long_paths.is_long_path_error(result["error"]) and not long_paths.is_enabled():
            _report("安装中断：Windows 路径超过 260 字符上限（请先开启长路径支持再重试）")
            break
        if attempt >= 3:
            break
        _report(f"第 {attempt} 次安装失败，正在生成依赖修正建议…")
        advice = await _agent_fix_advice(source, result["error"])
        new_req = _apply_advice(req_file, advice)
        # 记录建议与是否真的改动了依赖（旧实现静默无输出，问题难定位）
        _report(f"已生成修正建议（第 {attempt} 次），准备重试…", env_fix={
            "attempt": attempt,
            "reason": advice.get("reason") if isinstance(advice, dict) else None,
            # 修正建议 agent 自身失败时如实记录（不掩盖 pip 的真实报错）
            "advice_error": advice.get("_advice_error") if isinstance(advice, dict) else None,
            "requirements_updated": new_req is not req_file,
        })
        req_file = new_req
    if not ok:
        return False
    # 主清单装好后，再装「额外依赖」（清单漏声明、此前由缺包自愈补装过的包）：重建环境因此能复现。
    return await _install_extras(pip, ws or source.parent, attempt, index_url, extra_index_url,
                                 versions, project_id, task_id, env_type, _report)


def _env_python(env_dir: Path) -> str:
    for c in (env_dir / "Scripts" / "python.exe", env_dir / "python.exe", env_dir / "bin" / "python"):
        if c.exists():
            return str(c)
    return sys.executable


def _env_pip(env_dir: Path) -> str:
    for c in (env_dir / "Scripts" / "pip.exe", env_dir / "bin" / "pip"):
        if c.exists():
            return str(c)
    return str(env_dir / "Scripts" / "pip.exe")


def extra_deps_path(ws: Path) -> Path:
    """项目的「额外依赖」清单（pip requirements 语法，一行一个包）。

    存放位置是**工作区**（`<ws>/deps_extra.txt`）——工作区不随环境重建而清空，故重建环境能复现。
    """
    return ws / "deps_extra.txt"


def load_extra_deps(ws: Path) -> list[str]:
    p = extra_deps_path(ws)
    if not p.exists():
        return []
    try:
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
    except OSError:
        return []


def record_extra_dep(ws: Path, package: str) -> None:
    """把「清单漏声明、装完 `import` 才发现缺」的包记进项目额外依赖清单。

    背景（2026-10-06 scGPT/IPython 实测）：`install_missing_package` 能就地把缺包补装自愈，
    但它**不留痕**——用户重建独立环境时按仓库清单重装，补装的包又没了，于是「昨天补形状好好的、
    重建环境后补形状直接挂」，而且该包不在仓库 requirements 里，**每次重建都会再丢一次**。
    记进工作区即可让重建复现出真实可用的环境。
    """
    pkg = (package or "").strip()
    if not pkg:
        return
    def _base(name: str) -> str:
        return name.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].strip().lower()
    names = load_extra_deps(ws)
    if any(_base(n) == _base(pkg) for n in names):
        return
    names.append(pkg)
    ws.mkdir(parents=True, exist_ok=True)
    extra_deps_path(ws).write_text("\n".join(names) + "\n", encoding="utf-8")


async def _install_extras(pip: str, ws: Path, attempt: int, index_url: Optional[str],
                          extra_index_url: Optional[str], versions: dict,
                          project_id: str, task_id: str, env_type: str, report) -> bool:
    """安装项目「额外依赖」（清单漏声明、由缺包自愈补装过的包），使**重建环境**能复现可用环境。

    没有额外依赖时直接 True。额外依赖安装失败即判本次建环境失败（否则会留下一个「建成功但 import
    起不来」的环境，问题更隐蔽）。
    """
    extra = load_extra_deps(ws)
    if not extra:
        return True
    report(f"安装额外依赖（清单漏声明、此前自动补装的 {len(extra)} 个：{', '.join(extra)}）…")
    result = await _try_install(pip, extra_deps_path(ws), index_url, extra_index_url)
    _record_install(project_id, task_id, attempt, result, versions, env_type, step="pip_install_extra")
    if result["ok"]:
        return True
    report("额外依赖安装失败（这些包不在仓库依赖清单里，但项目运行需要它们）",
           env_extra_error=(result.get("error") or "")[-800:])
    return False


async def install_missing_package(env_dir: Path, package: str, timeout_s: int = INSTALL_TIMEOUT_S) -> dict:
    """补装一个「装完但 import 时报缺」的包（模块一 3.5 最小命令验证发现 ModuleNotFoundError 时调用）。

    背景：依赖清单**漏声明**是常见情况（如 scGPT 用了 IPython 却没写进 pyproject），
    pip 按清单装完不报错，直到 `import <pkg>` 才炸；此时按缺包名补装一次即可自愈。
    返回 {"ok", "error", "command"}；是否留痕由调用方决定（这里不写 run_record）。
    """
    def _cmd(index_url: Optional[str]) -> list[str]:
        c = [_env_pip(env_dir), "install", package,
             "--retries", str(PIP_RETRIES), "--timeout", str(PIP_TIMEOUT_S)]
        if index_url:
            c += ["--index-url", index_url]
        return c

    # 索引可达性会瞬时抽风（实测：同一镜像前一刻 "from versions: none"、后一刻正常）→
    # 与主安装同口径：索引类错误切备源重试一次，非索引错误（如包名不对）就不折腾了。
    indexes: list[Optional[str]] = [PIP_INDEX_URL] if PIP_INDEX_URL else [None]
    if PIP_FALLBACK_INDEX and PIP_FALLBACK_INDEX not in indexes:
        indexes.append(PIP_FALLBACK_INDEX)
    last: dict = {"ok": False, "error": "未执行", "command": ""}
    for index_url in indexes:
        cmd = _cmd(index_url)
        env = {**os.environ, "PIP_INDEX_URL": index_url} if index_url else None
        try:
            rc, out = await proc_util.run_command(cmd, env=env, timeout=timeout_s)
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"补装 {package} 超时（>{timeout_s}s）", "command": " ".join(cmd)}
        if rc == 0:
            return {"ok": True, "error": None, "command": " ".join(cmd)}
        last = {"ok": False, "error": (out or "")[-1500:], "command": " ".join(cmd)}
        if not _is_index_error(out or ""):
            break
    return last


def _find_requirements(source: Path) -> Optional[Path]:
    for name in ("requirements.txt", "pyproject.toml", "setup.py"):
        p = source / name
        if p.exists():
            return p
    return None


# 项目清单（非 requirements 风格）：安装用 `pip install <项目目录>` 而非 `-r`
_PROJECT_MANIFESTS = ("pyproject.toml", "setup.py")


# 大包（torch/scanpy 等）在本机网络下易 ReadTimeoutError：给 pip 加重试与读超时（可用 env 覆盖）
PIP_RETRIES = int(os.getenv("PIP_RETRIES", "5"))
PIP_TIMEOUT_S = int(os.getenv("PIP_TIMEOUT_S", "120"))


def _install_cmd(pip: str, req_file: Path, index_url: Optional[str] = None,
                 extra_index_url: Optional[str] = None) -> list[str]:
    if req_file.name.lower() in _PROJECT_MANIFESTS:
        cmd = [pip, "install", str(req_file.parent)]  # 安装项目及其声明依赖
    else:
        cmd = [pip, "install", "-r", str(req_file)]
    if index_url:
        cmd += ["--index-url", index_url]
    if extra_index_url:
        cmd += ["--extra-index-url", extra_index_url]  # CUDA 不匹配 → CPU 版 wheel 索引
    cmd += ["--retries", str(PIP_RETRIES), "--timeout", str(PIP_TIMEOUT_S)]
    return cmd


# ---- 版本判断（2.3 步骤 2）：清单声明的语言/框架/CUDA 版本 —— 纯文本解析，不执行清单 ----

# 框架包名前缀；conda 的 pytorch / pytorch-cuda 归一到 torch
_FRAMEWORK_PKGS = ("torch", "torchvision", "torchaudio", "tensorflow", "keras", "jax",
                   "paddlepaddle", "paddle")
_FRAMEWORK_ALIASES = {"pytorch": "torch", "pytorch-cuda": "torch", "pytorch-gpu": "torch"}

# CPU 版 torch wheel 的额外索引（PEP 440：同一公共版本下 `2.1.2+cpu` 高于 `2.1.2`，
# 故把 CPU 索引作为额外索引交给 pip，pip 会优先取 CPU 版 wheel；主索引仍可服务其余依赖）。
CPU_TORCH_INDEX = os.getenv("ENV_CPU_TORCH_INDEX", "https://download.pytorch.org/whl/cpu")

# **有 NVIDIA 驱动就装 CUDA 版 torch**（2026-10-06）：默认索引（Windows 上的 PyPI/镜像）给的 `torch`
# 是 **CPU 版**——清单没声明 CUDA 时原来的结论是「按默认 wheel 安装」，于是环境永远拿不到 GPU 轮子
# （实测 scGPT 环境装出 `torch 2.14.1+cpu`）。候选 tag 从高到低取**驱动支持的最高者**，并做一次
# 轻量可达性探测（索引 404/不可达就往下降一档）。`ENV_USE_GPU_TORCH=0` 关闭；`ENV_CUDA_TORCH_INDEX`
# 可指名索引；`ENV_PYTORCH_WHEEL_ROOT` 可换轮子根（走镜像时用）。
GPU_TORCH_ENABLED = os.getenv("ENV_USE_GPU_TORCH", "1") != "0"
CUDA_TORCH_INDEX_OVERRIDE = os.getenv("ENV_CUDA_TORCH_INDEX")
PYTORCH_WHEEL_ROOT = os.getenv("ENV_PYTORCH_WHEEL_ROOT", "https://download.pytorch.org/whl").rstrip("/")
_CUDA_TAGS = ("cu130", "cu128", "cu126", "cu124", "cu121", "cu118")
# **国内镜像（任一即可）**：官方 CDN 下 2.5GB 的 CUDA wheel 极易 `ReadTimeoutError`（2026-10-06 实测
# 直接失败）；镜像站是**扁平目录**（不是 PEP503 索引），所以要按 `--find-links` 传，不能当 `--index-url`。
# 形如 `ENV_PYTORCH_FIND_LINKS=https://mirror.sjtu.edu.cn/pytorch-wheels/cu126`（逗号分隔多个）。
CUDA_FIND_LINKS = [u.strip().rstrip("/") + "/" for u in
                   (os.getenv("ENV_PYTORCH_FIND_LINKS") or "").split(",") if u.strip()]

_CUDA_SUFFIX_RE = re.compile(r"\+(cu\d{2,3})\b")
_CUDA_INDEX_RE = re.compile(r"/(cu\d{2,3})(?=[/\s\"']|$)")
_CUDA_DEP_RE = re.compile(r"-cu(\d{2})(?=[=<>!~\s\"]|$)")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v or ""))


def _normalize_version_spec(raw: str) -> Optional[str]:
    """归一化版本写法为比较式（供 _constraint_ok 判定）:
    poetry caret `^3.9` → `>=3.9,<4.0`；裸版本/conda `=3.9` → `==3.9.*`；其余原样返回。
    """
    s = (raw or "").strip().strip("\"'")
    if not s:
        return None
    if s.startswith("^"):
        parts = [int(x) for x in re.findall(r"\d+", s[1:])]
        if not parts:
            return None
        if len(parts) == 1:
            upper = [parts[0] + 1]
        elif len(parts) == 2:
            upper = [parts[0] + 1, 0]
        else:
            upper = parts[:-1] + [parts[-1] + 1]
        low = ".".join(str(p) for p in parts)
        return f">={low},<{'.'.join(str(u) for u in upper)}"
    if re.fullmatch(r"=?[\d.]+(\.\*)?", s):        # 3.9 / =3.9 / 3.9.*
        v = s.lstrip("=")
        return v if v.endswith(".*") else f"=={v}.*"
    return s


def _constraint_ok(version: str, constraint: str) -> bool:
    """版本是否满足比较式约束（自实现，不新增依赖）。

    宽松口径：`==3.9` 与 `==3.9.*` 都按前缀匹配（清单里 3.9 这种浮点写法很常见）；
    `~=` 按 PEP 440 展开；解析不出的片段不阻塞判定。
    """
    v = _version_tuple(version)
    if not v:
        return False
    for raw in (constraint or "").split(","):
        c = raw.strip()
        if not c:
            continue
        m = re.match(r"^(==|>=|<=|!=|~=|>|<)\s*([\d.*]+)", c)
        if not m:
            continue
        op, target = m.group(1), m.group(2)
        if op == "~=":
            t = _version_tuple(target)
            if not t:
                continue
            upper = t[:-1] + (t[-1] + 1,) if len(t) > 1 else (t[0] + 1,)
            if not (v >= t and v < upper):
                return False
            continue
        if target.endswith(".*"):
            prefix = _version_tuple(target[:-2])
            if not prefix:
                continue
            matched = v[:len(prefix)] == prefix
            if (op == "!=") == matched:      # `==3.9.*` 要求命中前缀，`!=3.9.*` 要求不命中
                return False
            continue
        t = _version_tuple(target)
        if not t:
            continue
        if op == "==":
            if v[:len(t)] != t:
                return False
        elif op == "!=":
            if v[:len(t)] == t:
                return False
        elif op == ">=":
            if v < t:
                return False
        elif op == "<=":
            if v > t:
                return False
        elif op == ">":
            if v <= t:
                return False
        elif op == "<":
            if v >= t:
                return False
    return True


def _conda_python_requirement(text: str) -> Optional[str]:
    """environment.yml 里 `- python=3.9` / `- python>=3.9` / `- python 3.9` → 归一化约束串。"""
    for line in text.splitlines():
        m = re.match(r"^\s*-\s*python\s*([=<>!~]+)\s*([0-9][0-9.*]*)\s*$", line)
        if m:
            op, ver = m.group(1), m.group(2)
            if op == "=":
                return ver if (ver.endswith(".*") or ver.count(".") >= 2) else f"=={ver}.*"
            return f"{op}{ver}"
        m = re.match(r"^\s*-\s*python\s+([0-9][0-9.*]*)\s*$", line)
        if m:
            ver = m.group(1)
            return ver if (ver.endswith(".*") or ver.count(".") >= 2) else f"=={ver}.*"
    return None


def _pyproject_python_requirement(text: str) -> Optional[str]:
    """pyproject.toml 的 `requires-python`（PEP 621）或 poetry 的 `python = "^3.9"`。"""
    m = re.search(r"requires-python\s*=\s*[\"']([^\"']+)[\"']", text)
    if m:
        return m.group(1)
    m = re.search(r"^\s*python\s*=\s*\{[^}]*?version\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    if m:
        return m.group(1)
    m = re.search(r"^\s*python\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    return m.group(1) if m else None


def _python_requirement(source: Path) -> Optional[dict]:
    """清单/元数据声明的 Python 版本要求 → {"spec": 归一化约束串, "source": 文件名}。

    读取面（2.3 步骤 2，均纯文本解析、不执行）：pyproject.toml 的 requires-python、
    setup.py 的 python_requires、environment.yml 的 python=。优先级与决策树一致：venv 类清单在前。
    """
    pyp = source / "pyproject.toml"
    if pyp.exists():
        spec = _normalize_version_spec(_pyproject_python_requirement(_read_text(pyp)) or "")
        if spec:
            return {"spec": spec, "source": "pyproject.toml"}
    setup = source / "setup.py"
    if setup.exists():
        m = re.search(r"python_requires\s*=\s*[\"']([^\"']+)[\"']", _read_text(setup))
        spec = _normalize_version_spec(m.group(1)) if m else None
        if spec:
            return {"spec": spec, "source": "setup.py"}
    for name in ("environment.yml", "environment.yaml"):
        yml = source / name
        if yml.exists():
            spec = _normalize_version_spec(_conda_python_requirement(_read_text(yml)) or "")
            if spec:
                return {"spec": spec, "source": name}
    return None


def _iter_python_interpreter_cmds(spec: str) -> list[list[str]]:
    """候选解释器命令前缀：先按约束里出现的版本号，再按 3.13→3.7 由高到低（pythonX.Y 与 py -X.Y）。"""
    wanted: list[tuple[int, int]] = []
    for major, minor in re.findall(r"(\d+)\.(\d+)", spec or ""):
        pair = (int(major), int(minor))
        if pair not in wanted:
            wanted.append(pair)
    for minor in range(13, 6, -1):
        pair = (3, minor)
        if pair not in wanted:
            wanted.append(pair)
    cmds: list[list[str]] = []
    for major, minor in wanted:
        cmds.append([f"python{major}.{minor}"])
        cmds.append(["py", f"-{major}.{minor}"])
    return cmds


def _cmd_python_version(cmd: list[str]) -> Optional[str]:
    """执行候选解释器命令取版本；命令不存在/超时 → None。"""
    try:
        proc = subprocess.run(
            [*cmd, "-c", "import sys;print(sys.version.split()[0])"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def resolve_env_python(source: Path) -> tuple[list[str], dict]:
    """按清单声明的 Python 要求选 venv 解释器（2.3 步骤 2「语言版本」落地）。

    优先级：ENV_VENV_PYTHON（显式配置）> 满足要求且已在用的后端解释器 > 本机探测到的匹配解释器
    > 告警回退默认解释器（回退不静默：返回的结论写进任务进度与 run_record）。
    无 Python 要求时行为与旧版完全一致（后端解释器 / ENV_VENV_PYTHON）。
    """
    host = sys.version.split()[0]
    req = _python_requirement(source)
    note: dict = {"required": None, "requirement_source": None, "command": None,
                  "matched": None, "warning": None}
    if req:
        note["required"] = req["spec"]
        note["requirement_source"] = req["source"]
    if ENV_VENV_PYTHON:
        note.update(command=[ENV_VENV_PYTHON], origin="ENV_VENV_PYTHON", matched=True)
        if req:
            ver = _cmd_python_version([ENV_VENV_PYTHON])
            if ver and not _constraint_ok(ver, req["spec"]):
                # 显式配置优先，但不静默：不满足清单要求要如实告警
                note["matched"] = False
                note["warning"] = (
                    f"ENV_VENV_PYTHON 指向的解释器 {ENV_VENV_PYTHON}（{ver}）不满足清单要求 "
                    f"{req['spec']}（{req['source']}）；仍按显式配置使用，依赖可能装不上"
                )
        return [ENV_VENV_PYTHON], note
    if req is None:
        note.update(command=[sys.executable], origin="backend", matched=True)
        return [sys.executable], note
    if _constraint_ok(host, req["spec"]):
        note.update(command=[sys.executable], origin="backend", matched=True, version=host)
        return [sys.executable], note
    for candidate in _iter_python_interpreter_cmds(req["spec"]):
        exe = shutil.which(candidate[0])
        if exe is None:
            continue
        cmd = [exe, *candidate[1:]]
        ver = _cmd_python_version(cmd)
        if ver and _constraint_ok(ver, req["spec"]):
            note.update(command=cmd, origin="detected", matched=True, version=ver)
            return cmd, note
    note.update(command=[sys.executable], origin="fallback", matched=False,
                warning=(f"清单要求 Python {req['spec']}（{req['source']}），但本机未找到满足要求的解释器"
                         f"（候选 pythonX.Y / py -X.Y 均不可用）→ 回退默认解释器 {sys.executable}（{host}），"
                         "依赖可能装不上；请安装匹配解释器或设 ENV_VENV_PYTHON"))
    return [sys.executable], note


def _framework_from_line(line: str) -> Optional[str]:
    """单行依赖是否是框架包 → 归一后的包名（小写），否则 None。"""
    m = re.match(r"^\s*[-*]?\s*([A-Za-z0-9_.\-]+)", line or "")
    if not m:
        return None
    name = m.group(1).lower()
    name = _FRAMEWORK_ALIASES.get(name, name)
    return name if name in _FRAMEWORK_PKGS else None


def _framework_names_in(frameworks: list[str]) -> set[str]:
    names = set()
    for line in frameworks:
        fw = _framework_from_line(line)
        if fw:
            names.add(fw)
    return names


def _add_framework(info: dict, line: str) -> None:
    """把依赖行按框架包去重收集（同一框架只记第一次出现的清单行）。"""
    fw = _framework_from_line(line)
    if fw and fw not in _framework_names_in(info["frameworks"]):
        info["frameworks"].append(line.strip())


def _conda_dependency_lines(text: str) -> list[str]:
    """environment.yml 的 dependencies 段（含 pip: 子列表）→ 依赖行；跳过 python=/pip/频道行。"""
    out: list[str] = []
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^dependencies\s*:", stripped):
            in_deps = True
            continue
        if not in_deps:
            continue
        if not stripped:
            continue
        if not stripped.startswith("-"):
            if not line[:1].isspace():
                break          # 回到顶层键（channels: 等），dependencies 段结束
            continue
        item = stripped.lstrip("-").strip()
        if not item or item == "pip" or re.match(r"^python\s*[=<>!~]", item):
            continue
        out.append(item)
    return out


def _pyproject_dependency_lines(text: str) -> list[str]:
    """pyproject.toml 的 `dependencies = [...]`（PEP 621）与 `[tool.poetry.dependencies]` 条目。"""
    out: list[str] = []
    m = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    if m:
        out += [s.strip() for s in re.findall(r"[\"']([^\"']+)[\"']", m.group(1))]
    for m in re.finditer(
        r"^\s*([A-Za-z0-9_.\-]+)\s*=\s*(?:\{[^}]*?version\s*=\s*)?[\"']([^\"']+)[\"']",
        text, re.MULTILINE,
    ):
        name, ver = m.group(1), m.group(2)
        if name.lower() in ("python", "name", "version", "description", "readme", "repository",
                            "requires-python", "license"):
            continue
        spec = _normalize_version_spec(ver) or ver
        out.append(f"{name}{spec}" if spec[:1] in "<>=!~" else f"{name}=={spec}")
    return out


def _setup_install_requires(text: str) -> list[str]:
    """setup.py 的 install_requires 列表（只读文本解析，不执行）。"""
    m = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not m:
        return []
    return [s.strip() for s in re.findall(r"[\"']([^\"']+)[\"']", m.group(1))]


def _manifest_dependency_lines(source: Path) -> list[str]:
    """各依赖清单的依赖行（requirements.txt / environment.yml / pyproject.toml / setup.py）。"""
    lines: list[str] = []
    req = source / "requirements.txt"
    if req.exists():
        lines += [ln.strip() for ln in _read_text(req).splitlines() if ln.strip()]
    for name in ("environment.yml", "environment.yaml"):
        yml = source / name
        if yml.exists():
            lines += _conda_dependency_lines(_read_text(yml))
    pyp = source / "pyproject.toml"
    if pyp.exists():
        lines += _pyproject_dependency_lines(_read_text(pyp))
    setup = source / "setup.py"
    if setup.exists():
        lines += _setup_install_requires(_read_text(setup))
    return lines


def _cuda_tag_to_version(tag: str) -> Optional[str]:
    """`cu121` → `12.1`、`cu118` → `11.8`、`cu12` → `12.0`。"""
    digits = tag[2:]
    if len(digits) == 3:
        return f"{digits[:2]}.{digits[2]}"
    if len(digits) == 2:
        return f"{digits[0]}.{digits[1]}"
    return None


def _required_cuda(source: Path) -> Optional[dict]:
    """依赖清单声明的 CUDA 要求 → {"version": "12.1", "source": 文件名}。

    三类写法（文本解析）：torch wheel 后缀 `torch==2.1.2+cu121`、pip 源
    `--index-url .../whl/cu118`、依赖包名 `nvidia-cuda-runtime-cu12`（只到主版本 → 记 12.0）。
    """
    for name in ("requirements.txt", "environment.yml", "environment.yaml",
                 "pyproject.toml", "setup.py"):
        p = source / name
        if not p.exists():
            continue
        text = _read_text(p)
        for regex in (_CUDA_SUFFIX_RE, _CUDA_INDEX_RE, _CUDA_DEP_RE):
            m = regex.search(text)
            if m:
                ver = _cuda_tag_to_version(m.group(1))
                if ver:
                    return {"version": ver, "source": name}
    return None


_TORCH_LINE_RE = re.compile(r"^\s*[-'\"]?\s*(?:[\w.-]+::)?(torch|pytorch)\s*(?P<spec>[=<>!~][^\s'\";#]*)?",
                            re.IGNORECASE)


def _torch_wheel_spec(source: Path) -> str:
    """依赖清单里 torch 的**版本钉**（如 `torch==2.1.0` → 原样返回；清单没写 → `"torch"`）。

    预装 CUDA 版 torch 时**尊重清单的钉**（只为换 build，不为换版本）；conda 的 `=` 写法归一成 `==`。
    """
    for name in ("requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml", "setup.py"):
        text = _read_text(source / name)
        for line in text.splitlines():
            m = _TORCH_LINE_RE.match(line)
            if not m:
                continue
            spec = (m.group("spec") or "").strip()
            if spec.startswith("=") and not spec.startswith("=="):
                spec = "=" + spec            # conda `=2.1` → pip `==2.1`
            return f"torch{spec}" if spec else "torch"
    return "torch"


_INDEX_REACHABLE_CACHE: dict[str, bool] = {}


def _index_reachable(index: str, timeout: float = 8.0) -> bool:
    """wheel 索引根是否可达（进程内缓存，只探一次）。探测失败一律当不可达——**不抛异常**。"""
    if index in _INDEX_REACHABLE_CACHE:
        return _INDEX_REACHABLE_CACHE[index]
    import urllib.request

    ok = False
    try:
        req = urllib.request.Request(index.rstrip("/") + "/", method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ok = 200 <= int(getattr(r, "status", 200)) < 400
    except Exception:  # noqa: BLE001 —— 探测失败不该让建环境失败，按不可达处理
        ok = False
    _INDEX_REACHABLE_CACHE[index] = ok
    return ok


def _pick_cuda_index(driver: Optional[str]) -> tuple[Optional[str], str]:
    """驱动支持的**最高可用** CUDA 轮子索引 → `(index|None, 说明)`。

    配了 `ENV_PYTORCH_FIND_LINKS` 时**不探测官方索引**（轮子从镜像来，官方 CDN 慢/超时都无所谓）；
    配了 `ENV_CUDA_TORCH_INDEX` 时按它走、也不探测。
    """
    if CUDA_TORCH_INDEX_OVERRIDE:
        return CUDA_TORCH_INDEX_OVERRIDE, f"由 ENV_CUDA_TORCH_INDEX 指定：{CUDA_TORCH_INDEX_OVERRIDE}"
    dver = _version_tuple(driver or "")
    tried: list[str] = []
    for tag in _CUDA_TAGS:
        ver = _cuda_tag_to_version(tag)
        if not ver or dver < _version_tuple(ver):
            continue                                    # 驱动不支持这一档
        index = f"{PYTORCH_WHEEL_ROOT}/{tag}"
        if CUDA_FIND_LINKS:
            return index, (f"驱动支持 CUDA {driver}，轮子取镜像 {', '.join(CUDA_FIND_LINKS)}"
                           f"（官方索引 {index} 仅备用）")
        tried.append(tag)
        if _index_reachable(index):
            return index, f"驱动支持 CUDA {driver}，取最高可用轮子索引 {index}"
    if not tried:
        return None, f"驱动只到 CUDA {driver}，没有更低的候选（{', '.join(_CUDA_TAGS)}）"
    return None, f"候选轮子索引都不可达（{', '.join(tried)}，根 {PYTORCH_WHEEL_ROOT}）"


def plan_cuda(source: Path, versions: dict) -> dict:
    """驱动 CUDA 与依赖要求的判定（2.3「异常与边界」）。

    返回结论字典（进 run_record 的 environment.cuda_plan 与任务进度，**不静默**），三种动作：
      - `action=None`：按默认 wheel 安装（无 GPU / 明确关闭 / 索引不可达）；
      - `action="cuda_wheel"`：**有 NVIDIA 驱动** → 以 `index` 为索引装 **CUDA 版** torch
        （清单没声明 CUDA 也照样装：Windows 上默认索引的 torch 就是 CPU 版）；
      - `action="cpu_wheel"`：清单声明了 CUDA 但驱动不支持 → 降级 CPU 版 wheel。
    """
    required = _required_cuda(source)
    driver = versions.get("cuda")
    base: dict = {"driver": driver,
                  "required": required["version"] if required else None,
                  "requirement_source": required["source"] if required else None}
    if required is not None and not (driver and _version_tuple(driver) >= _version_tuple(required["version"])):
        return {**base, "match": False, "action": "cpu_wheel", "index": CPU_TORCH_INDEX,
                "reason": (f"依赖要求 CUDA {required['version']}（{required['source']}），本机"
                           + ("未检测到 NVIDIA 驱动（nvidia-smi 不可用）" if not driver
                              else f"驱动仅支持 CUDA {driver}")
                           + f" → 降级 CPU 运行：以额外索引 {CPU_TORCH_INDEX} 选 CPU 版 torch wheel")}
    if not GPU_TORCH_ENABLED:
        return {**base, "match": bool(required), "action": None,
                "reason": "ENV_USE_GPU_TORCH=0：按默认 wheel 安装（不装 CUDA 版）"}
    if not driver:
        return {**base, "match": None, "action": None,
                "reason": "未检测到 NVIDIA 驱动（nvidia-smi 不可用），按默认 wheel 安装"}
    index, why = _pick_cuda_index(driver)
    if index is None:
        return {**base, "match": None, "action": None, "reason": f"{why} → 按默认 wheel 安装"}
    return {**base, "match": True, "action": "cuda_wheel", "index": index,
            "torch_spec": _torch_wheel_spec(source),
            "find_links": list(CUDA_FIND_LINKS),
            "reason": (f"{why}；本机有 NVIDIA 驱动（CUDA {driver}）→ 装 **CUDA 版** torch"
                       + (f"（清单要求 CUDA {required['version']}）" if required
                          else "（清单未声明 CUDA；默认索引在 Windows 上只给 CPU 版，故这里显式取 CUDA 轮子）"))}


def _detect_cuda() -> Optional[str]:
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return None
        m = re.search(r"CUDA Version:\s*([\d.]+)", proc.stdout)
        return m.group(1) if m else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _python_version(python_exe: Optional[str]) -> str:
    """取目标环境解释器版本（非宿主），失败则回退宿主版本。"""
    if python_exe:
        ver = _cmd_python_version([python_exe])
        if ver:
            return ver
    return sys.version.split()[0]


def detect_versions(source: Path, python_exe: Optional[str] = None) -> dict:
    """版本判断（需求一.2、2.3）：解析依赖清单与代码 import 确定语言/框架/CUDA 版本。

    语言版本：python_required/python_requirement_source 为清单声明的 Python 要求（同时作为
    venv 解释器选择依据，见 resolve_env_python）。框架版本：读取 requirements.txt、
    environment.yml、pyproject.toml、setup.py（install_requires）。CUDA：cuda 为驱动支持版本，
    cuda_plan 为「驱动 vs 依赖要求」的判定结论与降级动作（不匹配 → CPU 版 wheel），
    结论随 environment 进 run_record，不静默。
    """
    info: dict = {"python": _python_version(python_exe), "frameworks": [], "cuda": _detect_cuda()}

    py_req = _python_requirement(source)
    info["python_required"] = py_req["spec"] if py_req else None
    info["python_requirement_source"] = py_req["source"] if py_req else None

    for line in _manifest_dependency_lines(source):
        _add_framework(info, line)

    for py in source.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for fw in ("torch", "tensorflow", "keras", "jax"):
            if f"import {fw}" in text or f"from {fw}" in text:
                if fw not in _framework_names_in(info["frameworks"]):
                    info["frameworks"].append(fw)

    info["cuda_plan"] = plan_cuda(source, info)
    return info


_TORCH_CU_WHEEL_RE = re.compile(r"torch-(\d+\.\d+\.\d+\+cu\d+)-(cp\d+)-[^-]+-([A-Za-z0-9_.]+)\.whl")


def _fetch_text(url: str) -> Optional[str]:
    """取一段文本（镜像的 PEP503 列表页）；失败返回 None，不抛。"""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 —— 地址来自平台配置
            return resp.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


def _python_tag(python: str) -> Optional[str]:
    """解释器的 wheel tag（如 cp312）；问不出来就返回 None。"""
    try:
        done = subprocess.run(
            [python, "-c", "import sys;print('cp%d%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=60,
        )
        tag = (done.stdout or "").strip()
        return tag if tag.startswith("cp") else None
    except Exception:  # noqa: BLE001
        return None


def _platform_tag() -> str:
    """本机 wheel 平台标签（子串匹配用）：Windows → win_amd64，其余按 manylinux。"""
    return "win_amd64" if os.name == "nt" else "manylinux"


def resolve_mirror_torch_pin(python: str, find_links: list) -> Optional[tuple]:
    """从镜像里挑一个**匹配当前解释器/平台**的最新 CUDA 版 torch。

    返回 `(钉, 可用链接)`，例如 `("torch==2.14.1+cu126", "…/cu126/torch/")`；找不到返回 None。

    为什么必须钉版本：pip 在「find-links 镜像 + 主索引」之间**只按版本号取高**，而 PyPI 上的
    torch 版本号常高于镜像里的 cuXXX 轮子（Windows 上 PyPI 给的就是 CPU 版）→ 镜像被完全忽略。
    实测：`2.14.1`（CPU）> `2.9.1+cu126`；就算基础版本相同，PEP 440 的**本地版本按段比较**，
    `cu126`→`["cu",126]` 而 `"cu"` 是 `"cpu"` 的前缀 → **`+cpu` 反而更大**。钉死才覆盖得了这套规则。

    **还要返回链接**：镜像根目录（PEP503）里只有**包子目录**、没有 wheel，`--find-links` 指根目录
    pip 会说 `from versions: none`（实测踩到）；必须指向真正列出 wheel 的那一层（如 `torch/`）。
    """
    tag = _python_tag(python)
    if not tag:
        return None
    plat = _platform_tag()
    for link in find_links:
        base = str(link).rstrip("/")
        for url in (f"{base}/torch/", base):  # PEP503 包子目录 → 扁平目录
            html = _fetch_text(url)
            if not html:
                continue
            best: Optional[tuple] = None
            for m in _TORCH_CU_WHEEL_RE.finditer(html.replace("%2B", "+")):
                full, wheel_tag, wheel_plat = m.group(1), m.group(2), m.group(3)
                if wheel_tag != tag or plat not in wheel_plat:
                    continue
                key = tuple(int(x) for x in full.split("+")[0].split("."))
                if best is None or key > best[0]:
                    best = (key, full)
            if best:
                return f"torch=={best[1]}", url
    return None


def preinstall_torch_cmd(pip: str, cuda_plan: dict, main_index: Optional[str],
                         torch_pin: Optional[str] = None,
                         torch_link: Optional[str] = None) -> list[str]:
    """CUDA 版 torch 的预装命令（`_preinstall_torch` 与 `scripts/install_cuda_torch.py` 共用一份）。

    配了镜像（`find_links`）就按 `--find-links` 传——且**必须钉住镜像里的版本号**（`torch_pin`）
    并把链接指到**真正列出 wheel 的那一层**（`torch_link`），否则：
    ① 主索引上版本号更高的 CPU 版会赢；② 指根目录时 pip 说 `from versions: none`。
    不能把镜像当 `--index-url`：官方 CDN 下 2.5GB 的 wheel 会 `ReadTimeoutError`（2026-10-06 实测）。
    """
    spec = str(torch_pin or cuda_plan.get("torch_spec") or "torch")
    index = str(cuda_plan.get("index") or "")
    find_links = [torch_link] if torch_link else [
        str(u) for u in (cuda_plan.get("find_links") or [])
    ]
    cmd = [pip, "install", "--upgrade", spec]
    if find_links:
        for link in find_links:
            cmd += ["--find-links", link]
        if main_index:
            cmd += ["--index-url", main_index]      # 其余依赖（torch 的兄弟包）走主索引
    else:
        cmd += ["--index-url", index]
        if main_index:
            cmd += ["--extra-index-url", main_index]
    cmd += ["--retries", str(PIP_RETRIES), "--timeout", str(PIP_TIMEOUT_S)]
    return cmd


async def _preinstall_torch(pip: str, cuda_plan: dict, versions: dict,
                            main_index: Optional[str], project_id: str, task_id: str,
                            report=None) -> bool:
    """先从 **CUDA 轮子索引**把 torch 装上（其余依赖仍走主索引）。

    失败**如实登记**（run_record `step="pip_install_torch_cuda"` + 任务进度）并返回 False；
    主安装仍会带 CUDA 额外索引再试一次，所以这一步失败不等于建环境失败——但绝不静默。
    `report` 是调用方的进度写手（`_install_with_fix` 里的闭包），不传就直写任务进度。
    """
    def _say(stage: str, **extra) -> None:
        if report is not None:
            report(stage, **extra)
        else:
            task_manager.update_progress(task_id, {**extra, "stage": stage})

    spec = str(cuda_plan.get("torch_spec") or "torch")
    index = str(cuda_plan.get("index") or "")
    find_links = [str(u) for u in (cuda_plan.get("find_links") or [])]
    # 钉住镜像里的版本号——否则主索引上版本号更高的 **CPU 版** torch 会赢（实测过）
    torch_pin = torch_link = None
    if find_links:
        try:
            python = str(Path(pip).with_name("python.exe" if os.name == "nt" else "python"))
            resolved = await asyncio.to_thread(resolve_mirror_torch_pin, python, find_links)
            if resolved:
                torch_pin, torch_link = resolved
        except Exception:  # noqa: BLE001 —— 钉不上就回退旧行为，不阻断建环境
            torch_pin = torch_link = None
    cmd = preinstall_torch_cmd(pip, cuda_plan, main_index, torch_pin=torch_pin, torch_link=torch_link)
    started = _now()
    try:
        rc, out = await proc_util.run_command(cmd, timeout=INSTALL_TIMEOUT_S)
        ok = rc == 0
        error = None if ok else out[-2000:]
    except asyncio.TimeoutError:
        ok, error = False, f"CUDA 版 torch 预装超时（>{INSTALL_TIMEOUT_S}s，已终止进程树）"
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        ok, error = False, str(e)
    result = {"ok": ok, "error": error, "command": " ".join(cmd), "index_url": index,
              "extra_index_url": main_index, "started_at": started, "finished_at": _now()}
    _record_install(project_id, task_id, 0, result, versions, "venv", step="pip_install_torch_cuda")
    _say("CUDA 版 torch 预装成功（后续主安装沿用该索引）" if ok
         else f"CUDA 版 torch 预装失败（{spec} @ {index}）——继续主安装再试",
         env_cuda={**cuda_plan, "preinstall_ok": ok, "torch_pin": torch_pin},
         **({"env_cuda_error": (error or "")[-300:]} if not ok else {}))
    return ok


async def _try_install(pip: str, req_file: Optional[Path], index_url: Optional[str] = None,
                      extra_index_url: Optional[str] = None) -> dict:
    if req_file is None or not req_file.exists():
        # 无依赖清单（如 conda 项目仅 environment.yml），跳过 pip 安装
        return {"ok": True, "error": None, "command": None, "index_url": index_url,
                "extra_index_url": extra_index_url,
                "started_at": _now(), "finished_at": _now()}
    cmd = _install_cmd(pip, req_file, index_url, extra_index_url)
    started = _now()
    try:
        rc, out = await proc_util.run_command(cmd, timeout=INSTALL_TIMEOUT_S)
        ok = rc == 0
        error = None if ok else out[-2000:]
    except asyncio.TimeoutError:
        ok, error = False, f"pip install 超时（>{INSTALL_TIMEOUT_S}s，已终止进程树）"
    except asyncio.CancelledError:
        raise  # 取消要真的停：proc_util 已杀进程树，继续向上抛让任务置 cancelled
    except Exception as e:  # noqa: BLE001
        ok, error = False, str(e)
    return {"ok": ok, "error": error, "command": " ".join(cmd), "index_url": index_url,
            "extra_index_url": extra_index_url,
            "started_at": started, "finished_at": _now()}


# pip 取不到版本（索引不可达/被拒）的特征；与「依赖冲突」区分：前者换源重试，后者走修正循环
_INDEX_ERROR_MARKERS = (
    "from versions: none",
    "connectionerror", "connection aborted", "connection reset", "newconnectionerror",
    "read timed out", "readtimeouterror", "temporary failure", "max retries exceeded",
    "sslerror", "certificate verify failed", "proxy",
)


def _is_index_error(error: Optional[str]) -> bool:
    if not error:
        return False
    low = error.lower()
    return any(marker in low for marker in _INDEX_ERROR_MARKERS)


def _record_install(
    project_id: str, task_id: str, attempt: int, result: dict, versions: dict, env_type: str,
    step: str = "pip_install",
) -> None:
    knowledge_service.record_run(
        {
            "project_id": project_id,
            "task_id": task_id,
            "run_type": "env_install",
            "environment": {"type": env_type, **versions},
            "params": {"attempt": attempt, "step": step, "index_url": result.get("index_url"),
                       "extra_index_url": result.get("extra_index_url")},
            "command": result.get("command"),
            "status": "success" if result["ok"] else "failed",
            "error": result.get("error"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        }
    )


async def _agent_fix_advice(source: Path, error: str) -> dict:
    """依赖修正建议：模板（`agents/prompts/dependency_fix.md`）+ 本次报错与项目目录。

    契约是「返回修正后的完整依赖清单」（`FIX_SCHEMA.requirements` + `reason`），
    模板里固定角色/任务/约束，运行期上下文（目录、pip 报错）与 schema 由代码追加。
    """
    prompt = prompts.render_prompt(
        "dependency_fix",
        context=f"### 项目代码目录\n\n{source}\n\n### pip install 失败报错（stderr/stdout 末尾）\n\n{error}",
        schema=FIX_SCHEMA,
    )
    try:
        result = await agent_service.run_sync(prompt, cwd=str(source), output_schema=FIX_SCHEMA)
    except Exception as e:  # noqa: BLE001 —— 修正建议 agent 失败不应掩盖 pip 的真实错误
        return {"_advice_error": str(e)[:500]}
    return result.get("structured_output") or {}


def _advice_requirements(advice) -> Optional[list[str]]:
    """从 agent 建议里取「修正后的完整依赖清单」，兼容多种形状。

    DeepSeek 下结构化输出走文件兜底、不经 schema 校验，键名/形状可能漂移
    （实测曾返回 task/root_cause/fix_recommendations 这类自有结构，导致旧逻辑静默不动作）。
    """
    if isinstance(advice, list):
        lines = [str(x).strip() for x in advice]
        return [x for x in lines if x] or None
    if isinstance(advice, dict):
        for key in ("requirements", "lines", "fixed_requirements", "dependencies", "packages"):
            val = advice.get(key)
            if isinstance(val, list):
                lines = [str(x).strip() for x in val]
                if any(lines):
                    return [x for x in lines if x]
    return None


def _legacy_advice_lines(req_file: Path, advice) -> Optional[list[str]]:
    """兼容旧的 {action, package, target_version} 形式，返回改写后的行；不可执行则 None。"""
    if not isinstance(advice, dict):
        return None
    action = advice.get("action")
    pkg = (advice.get("package") or "").strip().lower()
    if not pkg or action in (None, "none"):
        return None
    out_lines = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        if _pkg_name(line) == pkg:
            if action == "remove":
                continue
            if action == "downgrade" and advice.get("target_version"):
                out_lines.append(f"{advice['package']}=={advice['target_version']}")
                continue
            if action == "replace":
                continue
        out_lines.append(line)
    return out_lines


def _apply_advice(req_file: Optional[Path], advice) -> Optional[Path]:
    """按建议改写依赖清单，返回新文件；无可执行建议时原样返回（不产出坏文件）。

    主路径：agent 给出「修正后的完整清单」→ 直接写新文件（对任意修正都适用）。
    兼容路径：旧 {action, package, target_version} → 按行增删改。
    pyproject.toml/setup.py 类清单不按行改写（避免损坏）。
    """
    if req_file is None or not req_file.exists() or req_file.name.lower() in _PROJECT_MANIFESTS:
        return req_file
    lines = _advice_requirements(advice)
    if lines is None:
        lines = _legacy_advice_lines(req_file, advice)
    if lines is None:
        return req_file
    out = req_file.parent / f"requirements_fixed_{uuid.uuid4().hex[:8]}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _pkg_name(line: str) -> str:
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", line)
    return m.group(1).lower() if m else line.strip().lower()


# ---- 任务前知识带入：依赖冲突预检（需求六.1、模块详细设计 8.2「安装依赖前提示绕开方案」） ----


def _parse_conflict_structured(item: dict) -> dict:
    """取蒸馏知识的 structured（可能是 dict 或 JSON 文本），非 dict 一律当空。"""
    s = item.get("structured")
    if isinstance(s, str):
        try:
            s = json.loads(s)
        except (ValueError, TypeError):
            return {}
    return s if isinstance(s, dict) else {}


def _pins_from_conflicts(conflicts: list[dict]) -> dict:
    """从已确认冲突知识抽出可落地的版本钉 {包名: 完整 pip 行}。

    兼容两种 structured 形状：设计 schema 的 {pkg, resolution}（resolution 可为
    `torch==2.1.2`、`==2.1.2` 或 `2.1.2`）与 {pkg, version_b}（取 version_b 钉住）。
    非可操作条目（无 pkg / 无版本线索，例如本仓库 historically 起草的 {project_id, stage, error}）
    不产出钉——它们只作预警，不强行改写依赖。
    """
    pins: dict = {}
    for c in conflicts or []:
        s = _parse_conflict_structured(c)
        pkg = (s.get("pkg") or s.get("package") or "").strip().lower()
        if not pkg:
            continue
        res = s.get("resolution") or s.get("pin")
        line: Optional[str] = None
        if isinstance(res, str) and res.strip():
            res = res.strip()
            if _pkg_name(res) == pkg:                       # 已是 `torch==2.1.2` 形态
                line = res
            elif res.startswith(("=", ">", "<", "~", "!")):  # `==2.1.2` 形态
                line = f"{pkg}{res}"
            else:                                            # `2.1.2` 纯版本
                line = f"{pkg}=={res}"
        elif s.get("version_b"):
            line = f"{pkg}=={s['version_b']}"
        if line:
            pins[pkg] = line
    return pins


def _apply_known_pins(req_file: Optional[Path], pins: dict) -> Optional[dict]:
    """把已确认冲突的版本钉预应用到依赖清单，返回 {path, applied}；无可应用项返回 None。

    不改动原清单（另存副本），与 `_apply_advice` 同口径；pyproject/setup.py 不按行改写。
    """
    if not pins or req_file is None or not req_file.exists() or req_file.name.lower() in _PROJECT_MANIFESTS:
        return None
    applied: list[dict] = []
    out: list[str] = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and _pkg_name(line) in pins:
            newline = pins[_pkg_name(line)]
            out.append(newline)
            applied.append({"pkg": _pkg_name(line), "line": newline})
        else:
            out.append(line)
    if not applied:
        return None
    new_file = req_file.parent / f"requirements_precheck_{uuid.uuid4().hex[:8]}.txt"
    new_file.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {"path": str(new_file), "applied": applied}


def _dependency_precheck(project_id: str) -> list[dict]:
    """安装依赖前预检（8.2）：返回「与本次项目相关」的已确认 dependency_conflict 条目。

    只保留 struct 无 project_id（通用）或 project_id 恰为本项目者——其它项目的专属冲突不外泄
    到本项目的预警里（避免噪音）。预检失败一律静默（返回空），绝不阻断环境创建。
    进度写入由调用方合并（update_progress 是整体覆盖，避免冲掉同批写入的 env_precheck_applied）。
    """
    try:
        conflicts = (knowledge_service.bring_knowledge() or {}).get("dependency_conflict") or []
    except Exception:  # noqa: BLE001 —— 带入是旁路，失败不影响主流程
        return []
    return [c for c in conflicts if _parse_conflict_structured(c).get("project_id") in (None, "", project_id)]


def register() -> None:
    task_manager.register_handler(ENV_TASK_TYPE, _run_env_create)
