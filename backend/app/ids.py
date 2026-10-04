"""外部输入标识的安全校验（防路径穿越）与落盘目录名映射。

约定：一切会被拼进文件路径的「id」（paper_id/dataset_id/task_id 等）都必须先过 `safe_id`；
**落盘目录名再统一经 `fs_name`**——写盘与读回用同一个函数，才能保证同一个 id 每次都落到同一个目录
（`PAPERS_DIR / fs_name(paper_id)`）。

允许两类 id 形态：

1. **单段 id**（既有口径，未放宽）：字母数字开头，可含 `. _ - :`
   （arXiv 号 `2301.12345`、`zenodo:23080173`、`pubmed:456`）。
2. **多段 id**（`/` 分隔）：每段仍须字母数字开头、只含 `. _ -`。真实 DOI 属此类——
   bioRxiv/medRxiv 检索返回的 `paper_id` 就是 DOI（`10.1101/2023.10.03.560734`），
   Kaggle 数据集 id（`owner/dataset`）同理。多段 id 里**不允许** `:`（挡掉 `C:/Windows` 这类盘符写法）、
   `\\`、NUL/控制字符、空段（`/a`、`a//b`、`a/`）与 `.`/`..` 段。

两类 id 都拒绝 `..`、反斜杠、控制字符，以及 `C:` / `C:foo` 这类**盘符路径**——
`pathlib` 的 `数据目录 / "C:foo"` 会直接变成 `C:foo`（实测），不挡掉就等于把文件写到数据目录之外。

`fs_name` 把（已校验的）id 映射成**文件系统安全**的目录名：

- `:` → `_`：Windows 上 `:` 是 ADS 分隔符，`zenodo:23080173` 直接当目录名会抛 NotADirectoryError；
  沿用既有改写，**已落库的目录名因此不变**；
- 其余不在 `[A-Za-z0-9._-]` 里的字符 → `%XX`（字符码点的两位十六进制）：DOI 里的 `/` 落成 `%2F`，
  于是 DOI 变成**单层**目录名 `10.1101%2F2023.10.03.560734`，不再含任何路径分隔符。
  id 本身不允许含 `%`（两套校验都不放行），所以 `%XX` 不会与其它 id 撞名：映射**稳定且唯一**。
- Windows 保留设备名（CON/PRN/AUX/NUL/COM1-9/LPT1-9，含 `CON.txt` 写法）前缀补 `_`：
  这类 id 的旧映射在 Windows 上根本建不出目录，安全化不损失任何已落库数据。
"""
import re

# 单段 id（既有口径，未放宽长度上限）
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# 多段 id 的每一段（DOI 形态；不放行 : \ 空白 % 与通配符）
_ID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# 多段 id 总长上限（DOI 比 arXiv 号长，单独给上限，不动单段 id 的既有 128 上限）
_MULTI_SEGMENT_MAX_LEN = 200
# 盘符路径（`C:` / `C:foo` / `C:/x`）：pathlib 会把数据目录整个替换掉，必须拒绝
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_CONTROL_CHARS = tuple(chr(code) for code in list(range(0x20)) + [0x7F])
# Windows 保留设备名（含带扩展名写法，如 CON.txt）
_RESERVED_RE = re.compile(r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$", re.IGNORECASE)
_UNSAFE_FS_CHAR_RE = re.compile(r"[^A-Za-z0-9._-]")


def safe_id(value: str, what: str = "id") -> str:
    """校验并返回安全 id；不合法抛 ValueError（调用方按 400 处理）。"""
    if isinstance(value, str) and _ID_RE.match(value):
        if ".." in value:
            raise ValueError(f"非法 {what}: {value!r}（不得包含 ..）")
        if _DRIVE_RE.match(value):
            raise ValueError(f"非法 {what}: {value!r}（不得是盘符路径）")
        return value
    if isinstance(value, str) and "/" in value:
        _check_multi_segment(value, what)
        return value
    raise ValueError(
        f"非法 {what}: {value!r}（仅允许字母/数字/._:-，或 `10.1101/xxx` 这类逐段安全的多段 id；"
        "不得以点开头、不得含反斜杠/控制字符/..，也不得是盘符路径）"
    )


def _check_multi_segment(value: str, what: str) -> None:
    """多段 id（DOI / Kaggle 数据集 id）的逐段校验：放行 `/`，但不放行任何逃逸写法。"""

    def _reject(reason: str) -> None:
        raise ValueError(f"非法 {what}: {value!r}（{reason}）")

    if len(value) > _MULTI_SEGMENT_MAX_LEN:
        _reject(f"长度超过 {_MULTI_SEGMENT_MAX_LEN}")
    if ".." in value:
        _reject("不得包含 ..")
    if _DRIVE_RE.match(value):
        _reject("不得是盘符路径")
    if ":" in value:
        _reject("多段 id 不得包含 :（挡掉 C:/... 这类盘符路径）")
    if "\\" in value:
        _reject("不得包含反斜杠")
    if any(ch in value for ch in _CONTROL_CHARS):
        _reject("不得包含控制字符（含 \\0）")
    for segment in value.split("/"):
        if not segment:
            _reject("存在空的路径段（/a、a//b、a/ 均不允许）")
        if not _ID_SEGMENT_RE.match(segment):
            _reject(f"路径段 {segment!r} 不是安全段（须字母数字开头，且只含字母/数字/._-）")


def fs_name(value: str, what: str = "id") -> str:
    """把（已校验的）id 转成**文件系统安全**的目录名；**写盘与读回必须都用它**。

    映射规则见模块 docstring：`:` → `_`（沿用既有改写）、其余非 `[A-Za-z0-9._-]` 字符 → `%XX`
    （`/` 因此变成 `%2F`）、Windows 保留设备名前缀补 `_`。同一个 id 永远映射到同一个目录名。
    """
    checked = safe_id(value, what)
    name = "".join(
        "_" if ch == ":" else (ch if not _UNSAFE_FS_CHAR_RE.match(ch) else f"%{ord(ch):02X}")
        for ch in checked
    )
    if _RESERVED_RE.match(name):
        name = "_" + name
    return name
