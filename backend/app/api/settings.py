"""设置端点：模型接口凭证（一键封装 6.6-a）。

- GET  /api/settings/credentials  配置状态（只回掩码，不回明文密钥）
- PUT  /api/settings/credentials  保存凭证（api_key 为空 = 清除）

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
    if not data.get("api_key"):
        settings_store.clear_credentials()
    else:
        settings_store.save_credentials(data)
    return settings_store.status()
