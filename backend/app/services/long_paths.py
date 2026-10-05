"""Windows 长路径支持（>260 字符）的检测与开启（2026-10-05）。

背景：pip 装深层包时（如 orbax 的测试夹具，在 site-packages 内的相对路径就有 228 字符），
完整路径会超过 MAX_PATH(260) → `WinError 3 / Errno 2`，安装直接中断（scGPT 建环境卡死即此因）。

Windows 的解法是「注册表 `LongPathsEnabled=1` **且** 应用清单声明 longPathAware」二者齐备：
后者 python.exe 自带，故本应用只需把前者置 1。该键在 HKLM，写入需管理员 → 这里经 **UAC 提权**写一次。
短路径根（`config.project_env_dir`）能缓解但不能根治（C 盘单盘机器的最短可写位置仍 >260）。
"""
import sys
import time

REG_SUBKEY = r"SYSTEM\CurrentControlSet\Control\FileSystem"
REG_NAME = "LongPathsEnabled"


def is_enabled() -> bool:
    """当前是否已开启长路径。非 Windows 恒为 True（无此限制）。"""
    if sys.platform != "win32":
        return True
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, REG_SUBKEY) as key:
            return int(winreg.QueryValueEx(key, REG_NAME)[0]) == 1
    except OSError:
        return False


def is_long_path_error(text) -> bool:
    """pip 报错是否属于 Windows 长路径上限（据 pip 的 HINT 与 WinError 3 判定）。"""
    low = str(text or "").lower()
    return "enable-long-paths" in low or "winerror 3" in low


def enable(timeout_s: float = 10.0) -> dict:
    """弹一次 UAC 把 `LongPathsEnabled` 置 1。返回 {"ok", "enabled", "cancelled", "detail"}。

    `ShellExecuteW("runas")` 会显示 UAC 确认框（用户在框上点「是/否」期间本调用阻塞）；
    用户点「否」返回 ERROR_CANCELLED(1223)。提权进程异步执行，故这里轮询注册表确认。
    """
    if sys.platform != "win32":
        return {"ok": True, "enabled": True, "cancelled": False, "detail": "非 Windows 平台无需开启"}
    if is_enabled():
        return {"ok": True, "enabled": True, "cancelled": False, "detail": "长路径支持已开启"}

    import ctypes

    args = f'/c reg add "HKLM\\{REG_SUBKEY}" /v {REG_NAME} /t REG_DWORD /d 1 /f'
    try:
        rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", "cmd.exe", args, None, 1)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "enabled": False, "cancelled": False, "detail": f"提权调用失败：{e}"}

    if rc <= 32:
        cancelled = rc == 1223
        return {"ok": False, "enabled": is_enabled(), "cancelled": cancelled,
                "detail": "已取消管理员授权" if cancelled else f"提权未完成（代码 {rc}）"}

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_enabled():
            return {"ok": True, "enabled": True, "cancelled": False,
                    "detail": "已开启长路径支持；若仍失败，请重启本程序后重试建环境"}
        time.sleep(0.4)
    return {"ok": False, "enabled": is_enabled(), "cancelled": False,
            "detail": "提权进程已启动但注册表暂未读到新值（可能仍在处理；稍后重试或重启本程序）"}
