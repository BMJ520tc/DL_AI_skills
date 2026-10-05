"""一键封装 6.6-a：设置端点（凭证）与系统自检（env-check）+ 静态挂载。

凭证用例全部 monkeypatch 到临时文件并清掉环境变量密钥，绝不碰真实 data/ 与真实凭证。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _isolate_credentials(monkeypatch, tmp_path: Path) -> Path:
    from app import settings_store

    path = tmp_path / "credentials.json"
    monkeypatch.setattr(settings_store, "credentials_path", lambda: path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    return path


def test_credentials_roundtrip_masked(app_client, monkeypatch, tmp_path):
    path = _isolate_credentials(monkeypatch, tmp_path)

    assert app_client.get("/api/settings/credentials").json()["configured"] is False

    resp = app_client.put(
        "/api/settings/credentials",
        json={"api_key": "sk-secret-1234", "base_url": "https://api.example.com", "model": "test-model"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["configured"] is True
    assert body["source"] == "file"
    assert body["key_mask"] == "***1234"
    assert body["model"] == "test-model"

    # 安全口径（K2）：响应绝不回明文密钥；磁盘文件必须含密钥（否则后续任务没法用），
    # 且只落白名单字段、无多余内容
    assert "sk-secret-1234" not in resp.text
    saved = path.read_text(encoding="utf-8")
    assert "sk-secret-1234" in saved
    assert "small_model" not in saved  # 未提供的字段不落盘


def test_credentials_clear(app_client, monkeypatch, tmp_path):
    path = _isolate_credentials(monkeypatch, tmp_path)
    app_client.put("/api/settings/credentials", json={"api_key": "sk-abc"})
    assert path.is_file()

    resp = app_client.put("/api/settings/credentials", json={"api_key": ""})
    assert resp.json()["configured"] is False
    assert not path.is_file()


def test_credentials_blank_key_keeps_existing_and_updates_fields(app_client, monkeypatch, tmp_path):
    """表单口径：已配置时留空密钥 = 保留现有密钥，只更新其余字段（UI 文案承诺）。"""
    path = _isolate_credentials(monkeypatch, tmp_path)
    app_client.put("/api/settings/credentials",
                   json={"api_key": "sk-old-5678", "base_url": "https://old.example", "model": "old-model"})

    resp = app_client.put("/api/settings/credentials",
                          json={"api_key": "", "base_url": "https://new.example", "model": "new-model"})
    body = resp.json()
    assert body["configured"] is True
    assert body["key_mask"] == "***5678"  # 旧密钥保留
    assert body["base_url"] == "https://new.example"  # 其余字段已更新
    assert body["model"] == "new-model"

    saved = path.read_text(encoding="utf-8")
    assert "sk-old-5678" in saved and "https://new.example" in saved


def test_credentials_blank_key_without_existing_stays_unconfigured(app_client, monkeypatch, tmp_path):
    """无现有密钥时留空 key + 其余字段 → 落盘无 key，状态仍未配置（无密钥不得静默算已配置）。"""
    path = _isolate_credentials(monkeypatch, tmp_path)
    resp = app_client.put("/api/settings/credentials",
                          json={"api_key": "", "base_url": "https://x.example", "model": "m"})
    assert resp.json()["configured"] is False
    assert "api_key" not in path.read_text(encoding="utf-8")


def test_credentials_env_source_shows_configured(app_client, monkeypatch, tmp_path):
    _isolate_credentials(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")

    body = app_client.get("/api/settings/credentials").json()
    assert body["configured"] is True
    assert body["source"] == "env"
    assert "sk-from-env" not in str(body)


def test_apply_credentials_env(monkeypatch, tmp_path):
    from app import settings_store

    path = _isolate_credentials(monkeypatch, tmp_path)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_DEFAULT_MODEL", raising=False)

    assert settings_store.apply_credentials_env() == {}  # 未配置：不动环境

    settings_store.save_credentials(
        {"api_key": "sk-file", "base_url": "https://api.file.example", "model": "file-model"}
    )
    import os

    applied = settings_store.apply_credentials_env()
    assert applied["api_key"] == "sk-file"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-file"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.file.example"
    assert os.environ["ANTHROPIC_DEFAULT_MODEL"] == "file-model"


def test_env_check_shape(app_client, monkeypatch, tmp_path):
    _isolate_credentials(monkeypatch, tmp_path)

    body = app_client.get("/api/system/env-check").json()
    assert set(body) == {
        "git", "python", "conda", "claude_cli", "data_dir", "static_served",
        "credentials_configured", "pip_index", "long_paths_enabled",
    }
    assert set(body["git"]) == {"found", "path", "version"}
    assert set(body["python"]) == {"found", "python", "py_launcher"}
    assert set(body["conda"]) == {"found", "path"}
    # Claude Code CLI 自检（大模型步骤需要，学生自装）
    assert set(body["claude_cli"]) == {"found", "path", "version"}
    # 数据目录 = 当前配置值（开发态项目根 data/），可写性为真
    from app.config import DATA_DIR

    assert body["data_dir"]["path"] == str(DATA_DIR)
    assert body["data_dir"]["writable"] is True
    assert body["static_served"] is False  # 开发默认关
    assert body["credentials_configured"] is False


def test_root_not_mounted_in_dev(app_client):
    # 开发态（SERVE_STATIC 关）不挂静态产物：根路径不是界面，而是 404
    assert app_client.get("/").status_code == 404


def test_spa_static_fallback(tmp_path):
    from app.static_serve import SPAStaticFiles

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>index</html>", encoding="utf-8")
    (dist / "assets").mkdir()
    (dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")

    app = FastAPI()

    @app.get("/api/known")
    def known():
        return {"ok": True}

    app.mount("/", SPAStaticFiles(directory=dist, html=True), name="static")
    client = TestClient(app)

    assert client.get("/").status_code == 200 and "index" in client.get("/").text
    assert client.get("/assets/app.js").text == "console.log('ok')"
    # SPA fallback：未知前端路径回退 index.html
    assert "index" in client.get("/some/deep/link").text
    # 未知 API 路径保持 404 JSON，不回退到页面
    assert client.get("/api/unknown").status_code == 404
    assert client.get("/api/known").json() == {"ok": True}
