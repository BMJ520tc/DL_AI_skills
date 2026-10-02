/**
 * Node 端运行前端代码生成器所需的最小浏览器环境桩。
 *
 * 仅用于 GB-1「画布导出代码」的机器化验证（headless 跑前端 codegen），
 * 不参与浏览器运行时；放在单独模块是为了保证在其它 import 之前执行。
 */
/** headless 环境下需要注入的浏览器全局（仅本桩使用，不参与浏览器运行时）。 */
type HeadlessGlobal = {
    localStorage?: {
        getItem(key: string): string | null;
        setItem(key: string, value: string): void;
        removeItem(key: string): void;
        clear(): void;
        key(index: number): string | null;
        readonly length: number;
    };
    window?: unknown;
    document?: {
        createElement(): {
            style: Record<string, unknown>;
            setAttribute(): void;
            appendChild(): void;
        };
        documentElement: { style: Record<string, unknown> };
        addEventListener(): void;
        removeEventListener(): void;
    };
};

const g = globalThis as HeadlessGlobal;

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
