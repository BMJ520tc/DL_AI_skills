# 一键封装构建脚本（6.6-b）：前端同源构建 → PyInstaller onedir → 拷 claude.exe → 压缩 zip。
# 用法：  powershell -ExecutionPolicy Bypass -File packaging\build.ps1 [-Version 0.1.0]
param(
    [string]$Version = "0.1.0",
    [string]$Python = "D:\python.exe",
    [string]$OutDir = "D:\releases"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot      # 仓库根（packaging 的上一级）

# 1) 前端：同源构建（VITE_API_BASE_URL="/" → 归一为空基址、打相对 /api，后端选任意端口都能连；
#    用 "/" 而非 "" 是因为 PowerShell 对空字符串环境变量不保证传给子进程）
Write-Host "[1/5] 构建前端（同源）..."
$env:VITE_API_BASE_URL = "/"
Push-Location (Join-Path $Root "frontend")
try { npm run build; if ($LASTEXITCODE -ne 0) { throw "前端构建失败" } }
finally { Pop-Location }

# 2) PyInstaller onedir
Write-Host "[2/5] PyInstaller 打包（onedir）..."
$dist = Join-Path $Root "dist"
if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }
& $Python -m PyInstaller (Join-Path $PSScriptRoot "DL-AI-skills.spec") --noconfirm --clean `
    --distpath $dist --workpath (Join-Path $Root "build\pyinstaller")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 失败" }

# 3) 拷内置 claude.exe（约 225MB；不放进 spec 的 binaries，作为普通文件随包）
Write-Host "[3/5] 拷贝内置 claude.exe..."
$claudeSrc = Join-Path $env:APPDATA "npm\node_modules\@anthropic-ai\claude-code\bin\claude.exe"
if (-not (Test-Path $claudeSrc)) {
    $claudeSrc = Join-Path $env:APPDATA "npm\node_modules\@anthropic-ai\claude-code\node_modules\@anthropic-ai\claude-code-win32-x64\claude.exe"
}
if (Test-Path $claudeSrc) {
    $claudeDst = Join-Path $dist "DL-AI-skills\claude"
    New-Item -ItemType Directory -Force -Path $claudeDst | Out-Null
    Copy-Item $claudeSrc (Join-Path $claudeDst "claude.exe") -Force
} else {
    Write-Warning "未找到 claude.exe —— 目标机需自装 claude CLI 才能用 agent 功能"
}

# 4) 压缩为 zip（落仓库外，不进版本库，K3）
Write-Host "[4/5] 压缩..."
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$zip = Join-Path $OutDir ("DL-AI-skills-{0}-win64.zip" -f $Version)
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $dist "DL-AI-skills") -DestinationPath $zip

# 5) 报告
Write-Host "[5/5] 完成。"
$sizeMB = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "产物：$zip（$sizeMB MB）"
