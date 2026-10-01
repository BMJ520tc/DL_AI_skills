import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // 开发/预览期的后端代理目标。默认与前端 API 默认地址保持一致。
  const env = loadEnv(mode, process.cwd(), '')
  const devApiTarget = env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8000'

  // 同源代理：backend/app/main.py 未挂 CORSMiddleware，浏览器直连后端会被跨域拦截。
  // 把 VITE_API_BASE_URL 设为 "/"（或以 / 开头的相对路径）即走这里的代理，
  // 详见 src/api/knowledgeClient.ts 与 README.md「跨域（CORS）与开发代理」。
  const apiProxy = {
    '/api': {
      target: devApiTarget,
      changeOrigin: true,
    },
  }

  return {
    plugins: [react()],
    server: { proxy: apiProxy },
    preview: { proxy: apiProxy },
  }
})
