# DL-AI-skills 前端（FlowEditor + 知识库检索）

本目录是基于基底项目 **DL-Playground** 的 React + TypeScript + Vite 应用（`package.json` 的 `name` 仍为
`dlplayground`），在保留原有画布能力（FlowEditor / 节点编辑 / 模块系统 / 代码导出）的前提下，
新增了**知识库检索**界面。

## 目录速览

| 路径 | 说明 |
| --- | --- |
| `src/FlowEditor.tsx` | 画布主界面（既有功能，未重构） |
| `src/components/KnowledgeSearchPanel.tsx` | 知识库检索面板（新增） |
| `src/api/knowledgeClient.ts` | 知识库 API 客户端：拼 URL / fetch / 解析（新增） |
| `src/components/HeaderUtils.tsx` | 顶部工具栏；新增一个可选 `onOpenKnowledge` 属性用于挂「知识库」按钮 |
| `vite.config.ts` | 新增 `/api` 开发/预览代理（见下文「跨域」） |

---

## 本地运行

### 1. 环境要求

- Node.js ≥ 20（实测 v24.16.0）
- npm ≥ 10（实测 11.13.0）
- 后端可运行（`D:\python.exe`，已装 fastapi/uvicorn 等）

### 2. 安装依赖

`package-lock.json` 已随仓库提供，优先使用 `npm ci` 做可复现安装：

```powershell
cd D:\vs_word\DL-AI-skills\frontend
npm ci          # 需要联网；实测 214 个包，约 16 秒
# 若 package-lock.json 与 package.json 不一致，改用：
# npm install
```

> 注意：`package.json` 里的 `vite` 被 override 成了 `npm:rolldown-vite@7.2.5`（Rolldown 版 Vite），
> 安装后 `vite -v` 显示为 `rolldown-vite`，属预期行为。

### 3. 生产构建

```powershell
npm run build   # 等价于 tsc -b && vite build
```

产物输出到 `frontend/dist/`（`index.html` + `assets/`）。
构建结束时可能出现 “Some chunks are larger than 500 kB” 的**警告**，不影响退出码与产物可用性。

### 4. 本地开发服务器

```powershell
npm run dev     # 默认 http://127.0.0.1:5173/
```

指定端口：

```powershell
npm run dev -- --port 5199 --strictPort
```

### 5. 预览生产构建产物

```powershell
npm run build
npm run preview -- --port 5199 --strictPort
```

---

## 知识库检索

### 入口

打开应用后，在顶部工具栏**右侧**点击「**知识库**」按钮，即从右侧滑出检索面板（覆盖层，`Esc` 或点击遮罩可关闭）。
该按钮只在传入 `onOpenKnowledge` 时渲染，不影响原有工具栏布局。

### 功能

- 关键词输入（回车或点「检索」触发；留空表示按类型浏览最近条目）
- 类型多选：知识 `knowledge` / 数据集 `dataset` / 论文 `paper` / 运行记录 `run` / 模块 `module`
- 结果列表显示：标题、类型徽标、摘要（截断）、时间、`data_type/ref_id`
- 点击结果 → 调用详情接口，展示「正文 + 结构化 JSON」；后端以 JSON 字符串返回的字段
  （`structured` / `scope` / `sources` / `section_index` 等）会被自动展开，可切换回原始字符串视图
- 空结果、请求失败、后端未启动均有明确提示（含实际请求 URL 与排查建议）

### 调用的后端接口

| 用途 | 接口 |
| --- | --- |
| 检索 | `GET /api/knowledge/search?types=<逗号分隔>&q=<关键词>&limit=<n>&offset=<n>` |
| 详情 | `GET /api/knowledge/items/{data_type}/{ref_id}` |
| 健康检查 | `GET /api/health` |

> 检索结果的 `types` 为**逗号分隔**字符串；`q` 为空时不报错，后端返回该类型的最近条目。

---

## 后端地址配置

默认后端地址为 **`http://127.0.0.1:8000`**，可通过 Vite 环境变量覆盖（代码中用
`import.meta.env.VITE_API_BASE_URL` 读取，见 `src/api/knowledgeClient.ts`）：

| `VITE_API_BASE_URL` | 行为 |
| --- | --- |
| 未设置 | 直连 `http://127.0.0.1:8000` |
| `http://127.0.0.1:8300` | 直连该绝对地址（末尾斜杠会被去掉） |
| `/`（或以 `/` 开头的路径） | **同源相对路径**模式，请求 `/api/...`，走下面的 Vite 代理 |

在 `frontend/` 下新建 `.env.local` 即可持久化配置（该文件不应提交）：

```dotenv
# 直连模式（后端已开启 CORS 时可用）
VITE_API_BASE_URL=http://127.0.0.1:8000

# 或：同源代理模式（推荐，见下节）
# VITE_API_BASE_URL=/
# VITE_DEV_API_TARGET=http://127.0.0.1:8000
```

启动后端（另开一个终端）：

```powershell
cd D:\vs_word\DL-AI-skills\backend
D:\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8300
curl.exe -s http://127.0.0.1:8300/api/health   # {"status":"ok"}
```

### 跨域（CORS）与开发代理

`backend/app/main.py`（`uvicorn app.main:app` 的入口）**已挂载 `CORSMiddleware`**：
`config.CORS_ORIGIN_REGEX` 默认放行本机来源（`http(s)://localhost | 127.0.0.1` 的任意端口），
也可用 `CORS_ORIGINS`（逗号分隔白名单）扩展。因此浏览器从 Vite dev server（如 5173）
**直连后端**不再被跨域拦截——把 `VITE_API_BASE_URL` 指向后端地址即可。

若不便开 CORS、或想避免在浏览器直接暴露后端地址，也可改用**同源代理**：`vite.config.ts`
已把 `/api` 代理到 `VITE_DEV_API_TARGET`（默认 `http://127.0.0.1:8000`），`dev` 与 `preview` 都生效。
两种方式二选一：

- **直连**：`VITE_API_BASE_URL=http://127.0.0.1:8300`（后端已开 CORS）
- **代理**：`VITE_API_BASE_URL=/` + `VITE_DEV_API_TARGET=http://127.0.0.1:8300`

```powershell
# PowerShell：同源代理模式启动前端
$env:VITE_API_BASE_URL="/"
$env:VITE_DEV_API_TARGET="http://127.0.0.1:8300"
npm run dev -- --port 5199 --strictPort
```

验证代理是否通：

```powershell
curl.exe -s "http://127.0.0.1:5199/api/health"                                 # {"status":"ok"}
curl.exe -s "http://127.0.0.1:5199/api/knowledge/search?types=knowledge&limit=2"
```

> 后端 CORS 与开发代理两者都可用，按需二选一；同源代理在后端未开 CORS 或需隐藏后端地址时仍适用。

---

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 面板提示「网络不可达、连接被拒绝或跨域(CORS)被浏览器拦截」 | ① 确认后端已启动且端口正确；② 若为跨域，按上节改用同源代理模式 |
| 面板提示 `接口返回 HTTP 404：item not found` | 条目已被删除或 `ref_id` 失效，重新检索即可 |
| 检索无结果 | 换更短/更通用的关键词；勾选更多类型；或清空关键词直接浏览（`run`/`module` 等类型可能本来为空） |
| `npm run build` 报 TS 错误 | 先确认 `node_modules` 已安装（`npm ci`），再查看完整报错；本项目当前 `tsc -b && vite build` 通过 |
| 端口被占用 | 用 `--port <端口> --strictPort` 指定其他端口，避免占用 8000/8199 |
