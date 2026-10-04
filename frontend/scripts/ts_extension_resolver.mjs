// scripts/ts_extension_resolver.mjs — Node 解析钩子：把 src/ 下无扩展名的相对导入补成
// .ts/.tsx（Vite/tsc 的 bundler 解析口径），这样 `node` 能直接跑仓库里的 TS 源码，
// 不必为自检脚本再装打包器。仅自检脚本使用，不参与前端构建产物。
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const EXTENSIONS = [".ts", ".tsx", "/index.ts", "/index.tsx"];

export async function resolve(specifier, context, nextResolve) {
    if (specifier.startsWith(".") && !/\.[cm]?[jt]sx?$/.test(specifier)) {
        for (const ext of EXTENSIONS) {
            const candidate = new URL(specifier + ext, context.parentURL);
            if (existsSync(fileURLToPath(candidate))) {
                return { url: candidate.href, shortCircuit: true };
            }
        }
    }
    return nextResolve(specifier, context);
}
