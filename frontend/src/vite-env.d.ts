/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 知识库后端地址；未设置时前端回退到 http://127.0.0.1:8000。 */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
