/**
 * Node 端运行前端代码生成器所需的最小浏览器环境桩。
 *
 * 仅用于 GB-1「画布导出代码」的机器化验证（headless 跑前端 codegen），
 * 不参与浏览器运行时；放在单独模块是为了保证在其它 import 之前执行。
 */
const g = globalThis as any;

if (!g.localStorage) {
    const store = new Map<string, string>();
    g.localStorage = {
        getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
        setItem: (k: string, v: string) => void store.set(k, String(v)),
        removeItem: (k: string) => void store.delete(k),
        clear: () => store.clear(),
        key: (i: number) => Array.from(store.keys())[i] ?? null,
        get length() {
            return store.size;
        },
    };
}
if (!g.window) {
    g.window = g;
}
if (!g.document) {
    g.document = {
        createElement: () => ({ style: {}, setAttribute() {}, appendChild() {} }),
        documentElement: { style: {} },
        addEventListener() {},
        removeEventListener() {},
    };
}

export {};
