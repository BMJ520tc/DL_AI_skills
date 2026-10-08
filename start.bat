@echo off
rem 本文件按 GBK 编码保存（中文 Windows 默认 936 码页）：cmd 无法可靠解析 chcp 65001 + UTF-8 的批处理，
rem 多字节行会被拦腰切断当命令执行（2026-10-05 实测）。
setlocal
cd /d "%~dp0"

rem ============================================================
rem  DL-AI-skills 一键启动（一键封装 6.6-a，评审定案：免安装便携形态）
rem  自检：Python（优先项目自带虚拟环境，外置依赖学生自装）→ 后端依赖 → 前端产物 → 端口 8000；
rem  然后以「后端同源服务前端产物」方式启动，并打开浏览器。
rem  注意：本脚本不捆绑 git / conda / Python / torch——缺什么，自检会明确提示。
rem ============================================================

set "PORT=8000"

rem --- 0. pip 默认国内镜像（可被用户环境变量覆盖；后端各建环境步骤也会继承） ---
if not defined PIP_INDEX_URL set "PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple"

rem --- 1. Python 自检：优先项目内虚拟环境；否则本机后端解释器 D:\python.exe；再否则 PATH 里的 python ---
set "PY=%~dp0backend\.venv\Scripts\python.exe"
if exist "%PY%" goto :python_ready
rem --- 1a. 本机后端解释器（D:\python.exe）：项目内无虚拟环境时用它 ---
if exist "D:\python.exe" set "PY=D:\python.exe"
if exist "%PY%" goto :python_ready
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Python（PATH 里找不到 python）。
    echo 本程序运行后端需要 Python 3.10 及以上版本，Python 不随本程序打包，请先自行安装：
    echo     https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"，装好后重新双击本脚本。
    pause
    exit /b 1
)
set "PY=python"
:python_ready

rem --- 1.1 后端依赖自检（python 在并不等于后端能起，实测曾踩坑） ---
"%PY%" -c "import uvicorn" >nul 2>nul
if errorlevel 1 (
    echo [错误] Python 里缺后端依赖（uvicorn 导入失败）。
    echo 请用国内镜像装依赖后重新双击本脚本：
    echo     "%PY%" -m pip install -r backend/requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple
    pause
    exit /b 1
)

rem --- 2. 前端产物自检（首次使用前需先构建，见 AGENTS.md 常用命令） ---
if not exist "frontend\dist\index.html" (
    echo [错误] 找不到前端构建产物 frontend\dist\index.html。
    echo 请先构建前端：  cd frontend ^&^& npm install ^&^& npm run build
    echo 若尚未安装前端依赖，请先安装 Node.js（https://nodejs.org/）再构建。
    pause
    exit /b 1
)

rem --- 3. 端口自检（被占即中止，不静默换端口——避免界面连错后端） ---
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
    echo [错误] 端口 %PORT% 已被占用。请先释放端口（例如关掉已开的本程序后端），再重新运行本脚本。
    pause
    exit /b 1
)

rem --- 4. 后端同源服务前端产物 ---
set "DL_AI_SERVE_STATIC=1"

echo 正在启动后端（端口 %PORT%，Ctrl+C 或关闭本窗口即停止）...
rem 延迟 2 秒后打开浏览器，避免后端未就绪页面空白
start "" /b cmd /c "ping -n 3 127.0.0.1 >nul & start http://127.0.0.1:%PORT%"

rem --- 4.1 启动就绪自检（等后端起来后打印 git/Python/conda/claude CLI/凭证 摘要，与开浏览器并行） ---
start "" /b "%PY%" "%~dp0scripts\startup_check.py" "http://127.0.0.1:%PORT%"

cd backend
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%

echo.
echo 后端已停止。按任意键关闭窗口。
pause >nul
