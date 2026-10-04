// scripts/ts_harness_loader.mjs — 注册 TS 相对导入解析钩子。
// 用法：node --import ./scripts/ts_harness_loader.mjs scripts/<harness>.ts
import { register } from "node:module";

register("./ts_extension_resolver.mjs", import.meta.url);
