import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

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
    plugins: [react()],
    server: { proxy: apiProxy },
    preview: { proxy: apiProxy },
  }
})
