"""设置端点：模型接口凭证（一键封装 6.6-a）。

- GET  /api/settings/credentials  配置状态（只回掩码，不回明文密钥）
- PUT  /api/settings/credentials  保存凭证：
    api_key 非空 → 全量更新；api_key 留空且其余全空 → 清除；
    api_key 留空但其余字段有值 → **保留现有密钥**，只更新其余字段（表单「留空则保留现有密钥」）。

凭证落数据目录下 `credentials.json`（K2：本地配置文件），见 `app/settings_store.py`。
"""
from fastapi import APIRouter
from pydantic import BaseModel

from app import settings_store

router = APIRouter(prefix="/api/settings", tags=["settings"])


class CredentialsPut(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    small_model: str = ""


@router.get("/credentials")
def get_credentials() -> dict:
    return settings_store.status()


@router.put("/credentials")
def put_credentials(body: CredentialsPut) -> dict:
    data = body.model_dump()
    if data.get("api_key"):
        settings_store.save_credentials(data)
    elif data.get("base_url") or data.get("model") or data.get("small_model"):
        # 留空密钥 + 其余字段非空 = 保留现有密钥只更新其余（全空的清除走「清除凭证」）
        existing = settings_store.load_credentials()
        if existing.get("api_key"):
            data["api_key"] = existing["api_key"]
        settings_store.save_credentials(data)
    else:
        settings_store.clear_credentials()
    return settings_store.status()
