/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 知识库后端地址；未设置时前端回退到 http://127.0.0.1:8000。 */
  readonly VITE_API_BASE_URL?: string;
  /** 后端地址的兼容变量名；VITE_API_BASE_URL 未设置时生效。 */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
