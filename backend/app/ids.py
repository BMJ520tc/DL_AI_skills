"""外部输入标识的安全校验（防路径穿越）。

约定：一切会被拼进文件路径的「id」（paper_id/dataset_id/task_id 等）都必须先过
`safe_id`。允许字母数字与 `. _ - :`（arXiv 号、`zenodo:123`、`pubmed:456` 等现实 id 形态），
禁止路径分隔符、`..`、空串与超长值——否则 `PAPERS_DIR / paper_id` 可被 `..\\..\\x` 逃逸出数据目录。
"""
import re

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def safe_id(value: str, what: str = "id") -> str:
    """校验并返回安全 id；不合法抛 ValueError（调用方按 400 处理）。"""
    if not isinstance(value, str) or not _ID_RE.match(value or ""):
        raise ValueError(f"非法 {what}: {value!r}（仅允许字母/数字/._:-，且不得以点开头或含路径分隔符）")
    if ".." in value:
        raise ValueError(f"非法 {what}: {value!r}（不得包含 ..）")
    return value


def fs_name(value: str, what: str = "id") -> str:
    """把（已校验的）id 转成**文件系统安全**的目录名：Windows 上 `:` 是 ADS 分隔符，
    直接用 `zenodo:23080173` 当目录名会抛 NotADirectoryError；统一改写为 `_`。
    原始 id 仍按原样入库/检索，只有落盘目录名被改写。
    """
    return safe_id(value, what).replace(":", "_")
