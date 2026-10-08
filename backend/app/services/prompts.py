"""prompt 模板加载（《系统架构设计》八.1、《模块详细设计》2.5、阶段1实施方案 O4）。

**单一事实来源**：交给大模型的提示词固定部分放在 `agents/prompts/<name>.md`，
本模块负责读取它，并按「模板 + 运行期上下文」组装出最终 prompt：

- 模板里只写角色、输入说明、任务与约束这类**可读的中文说明**；
- 需要调用方填入的内容（论文文本、uncertain 列表、pip 报错、项目目录）以占位符
  `{{...}}` 标注，由调用方在运行期作为「运行期上下文」章节拼接；
- **机器可读的 JSON Schema 始终由代码提供**并在运行期追加（`ADDRESS_SCHEMA`、
  `DYNAMIC_SCHEMA`、`FIX_SCHEMA`），模板不再复制一份，避免模板与 schema 漂移。

安全与可靠性约定：
- 只接受固定的模板名白名单（`TEMPLATE_NAMES`），不接受任意路径/名字（防路径穿越）；
- 带**进程内缓存**（按模板文件绝对路径缓存），避免每个任务重复读盘；
- 模板缺失、为空或不可读时**明确报错**，绝不静默返回空 prompt（空提示词会让 agent
  自由发挥，产出不可复核的结论）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.config import resource_path

PROMPTS_DIR = resource_path("agents/prompts")

# 模板名白名单：只允许这三份（《阶段1实施方案》第 197 行列出的落位文件）
TEMPLATE_NAMES = ("address_extract", "dynamic_analysis", "dependency_fix")

# 运行期上下文章节标题（模板末尾也有同名说明，保证模板单读时能看懂缺什么）
RUNTIME_HEADING = "## 运行期上下文（调用方拼接，模板不固定这些内容）"
SCHEMA_HEADING = "## 本次结构化输出 Schema（由代码提供，以此为准）"

# 进程内缓存：key 为模板文件绝对路径（同名模板在不同目录下不会互相串味）
_CACHE: dict[str, str] = {}


def template_path(name: str) -> Path:
    """模板名 → 仓库内路径；白名单外的名字直接拒绝。"""
    if name not in TEMPLATE_NAMES:
        raise ValueError(
            f"未知的 prompt 模板名: {name!r}（只允许 {', '.join(TEMPLATE_NAMES)}）"
        )
    return PROMPTS_DIR / f"{name}.md"


def load_template(name: str) -> str:
    """读取模板文本（带进程内缓存）；缺失/为空/不可读时明确报错。"""
    path = template_path(name)
    key = str(path.resolve())
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"prompt 模板缺失: {path}") from e
    except OSError as e:
        raise RuntimeError(f"prompt 模板不可读: {path}: {e}") from e
    if not text.strip():
        raise RuntimeError(f"prompt 模板为空: {path}")
    _CACHE[key] = text
    return text


def render_prompt(name: str, *, context: str = "", schema: Optional[dict] = None) -> str:
    """组装最终 prompt = 模板 +（可选）运行期上下文 +（可选）结构化输出 schema。

    - `context`：调用方填入的运行期事实（正文/补充材料、uncertain 列表、pip 报错、目录），
      原样追加在模板之后，带明确的章节标题，便于事后按 prompt 回溯；
    - `schema`：**由代码传入**（不写在模板里），以 JSON 形式追加，供 agent 与文件兜底对齐。
    """
    parts = [load_template(name).rstrip("\n")]
    if context and context.strip():
        parts.append(f"{RUNTIME_HEADING}\n\n{context.strip()}")
    if schema is not None:
        parts.append(f"{SCHEMA_HEADING}\n\n```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```")
    return "\n\n".join(parts) + "\n"


def clear_cache() -> None:
    """清空进程内缓存（测试用；正常运行不需要）。"""
    _CACHE.clear()
