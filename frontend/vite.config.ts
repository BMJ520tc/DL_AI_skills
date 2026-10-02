import { defineConfig, loadEnv, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * 生产构建的 CJS 互操作补丁（临时措施，作用于依赖而非本项目源码）。
 *
 * 症状：`npm run build` 产物一打开就整页白屏，控制台报
 *   TypeError: Cannot destructure property 'useDebugValue' of 'v.default' as it is undefined
 * 根因：zustand（@xyflow/react 的依赖）在 ESM 里写 `import ReactExports from 'react'`，
 *   而 react 19 的入口 node_modules/react/index.js 是「按 NODE_ENV 条件 require」的 CJS，
 *   Rollup 的 commonjs 互操作对这种非静态可分析的出口不合成 `default`，于是 ReactExports 为 undefined。
 *   开发服务器（esbuild 预打包）走另一套互操作，所以 `npm run dev` 正常、只有 build 产物崩。
 * 处理：只把该依赖里这一句默认导入改写成命名空间导入（命名导出是完整的，语义等价）。
 * 待上游/依赖版本修好后应删除本插件。
 */
function reactDefaultInterop(): Plugin {
  return {
    name: 'react-default-import-interop',
    enforce: 'pre',
    transform(code, id) {
      if (!/[\\/]node_modules[\\/]zustand[\\/]/.test(id)) return null
      const patched = code.replace(
        /import\s+ReactExports\s+from\s*(['"])react\1/g,
        'import * as ReactExports from "react"'
      )
      return patched === code ? null : { code: patched, map: null }
    },
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // 开发/预览期的后端代理目标。默认与前端 API 默认地址保持一致。
  const env = loadEnv(mode, process.cwd(), '')
  const devApiTarget = env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8000'

  // 同源代理：backend/app/main.py 已挂 CORSMiddleware（默认放行本机来源），浏览器可直连后端；
  // 本代理是「不便开 CORS / 想隐藏后端地址」时的替代。把 VITE_API_BASE_URL 设为 "/"
  //（或以 / 开头的相对路径）即走这里的代理，详见 src/api/knowledgeClient.ts 与
  // README.md「跨域（CORS）与开发代理」。
  const apiProxy = {
    '/api': {
      target: devApiTarget,
      changeOrigin: true,
    },
  }

  return {
    plugins: [react(), reactDefaultInterop()],
    server: { proxy: apiProxy },
    preview: { proxy: apiProxy },
  }
})
