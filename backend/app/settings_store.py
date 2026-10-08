"""本地设置存取：模型接口凭证（一键封装 6.6-a，阶段6实施方案 K2 口径）。

凭证落数据目录下的 `credentials.json`（K2：本地配置文件）——不进版本库、不打日志、
接口只回掩码不回明文。无凭证时大模型步骤按规矩 7 如实失败，本模块不做静默兜底。
"""
import json
import os
from pathlib import Path

from app.config import DATA_DIR, DEFAULT_ANTHROPIC_BASE_URL

# 凭证文件里允许的字段；api_key 为空 = 未配置
_ALLOWED = ("api_key", "base_url", "model", "small_model")


def credentials_path() -> Path:
    """凭证文件路径（函数取用，便于测试用 monkeypatch 隔离到临时目录）。"""
    return DATA_DIR / "credentials.json"


def load_credentials() -> dict:
    """读取凭证文件；不存在/损坏一律返回空 dict（损坏不抛出，视为未配置）。"""
    p = credentials_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: data[k] for k in _ALLOWED if isinstance(data.get(k), str) and data[k]}


def save_credentials(data: dict) -> None:
    """写凭证文件（只收白名单字段）。"""
    cleaned = {k: data[k] for k in _ALLOWED if isinstance(data.get(k), str) and data[k]}
    p = credentials_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def clear_credentials() -> None:
    """清除凭证文件。"""
    p = credentials_path()
    if p.is_file():
        p.unlink()


def _env_has_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def status() -> dict:
    """配置状态（对外口径：只回掩码与端点/模型信息，绝不回明文密钥）。"""
    data = load_credentials()
    if data.get("api_key"):
        key = data["api_key"]
        return {
            "configured": True,
            "source": "file",
            "key_mask": "***" + key[-4:] if len(key) > 4 else "***",
            "base_url": data.get("base_url") or DEFAULT_ANTHROPIC_BASE_URL,
            "model": data.get("model") or None,
            "small_model": data.get("small_model") or None,
        }
    if _env_has_key():
        return {
            "configured": True,
            "source": "env",
            "key_mask": "***（环境变量，未存储）",
            "base_url": os.getenv("ANTHROPIC_BASE_URL") or None,
            "model": os.getenv("ANTHROPIC_DEFAULT_MODEL") or None,
            "small_model": os.getenv("ANTHROPIC_DEFAULT_SMALL_MODEL") or None,
        }
    return {
        "configured": False,
        "source": None,
        "key_mask": None,
        "base_url": None,
        "model": None,
        "small_model": None,
    }


def apply_credentials_env() -> dict:
    """把凭证文件写入进程环境（SDK 的 CLI 子进程继承进程环境读取端点/密钥）。

    仅在文件已配置（有 api_key）时生效并**覆盖**同名环境变量——设置页的显式保存
    是最新的用户意图；未配置时不动环境，返回 {}。
    """
    data = load_credentials()
    if not data.get("api_key"):
        return {}
    os.environ["ANTHROPIC_API_KEY"] = data["api_key"]
    # 留空即用缺省端点（DeepSeek 的 Anthropic 兼容端点），否则 CLI 会打 api.anthropic.com → DeepSeek key 401。
    os.environ["ANTHROPIC_BASE_URL"] = data.get("base_url") or DEFAULT_ANTHROPIC_BASE_URL
    if data.get("model"):
        os.environ["ANTHROPIC_DEFAULT_MODEL"] = data["model"]
    if data.get("small_model"):
        os.environ["ANTHROPIC_DEFAULT_SMALL_MODEL"] = data["small_model"]
    return data
