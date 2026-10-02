import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// —— 外部 DOM 改动防护（配合 ErrorBoundary 的兜底）——
// 浏览器翻译类扩展（沉浸式翻译 / Google 翻译）会拆开并重组文本节点，Monaco 等编辑器也会移动自身 DOM；
// 这会让 React 卸载时 removeChild 失败（NotFoundError: ... is not a child of this node），开发态表现为整页黑屏。
// 这里只做最小容忍：目标已不在该父节点下就跳过；insertBefore 的参照节点失效就改为追加。首次触发打印一次告警。
const GUARD_TAG = "[dom-guard]";
let warnedRemove = false;
let warnedInsert = false;

const origRemoveChild = Node.prototype.removeChild;
Node.prototype.removeChild = (function (this: Node, child: Node): Node {
    if (child.parentNode !== this) {
        if (!warnedRemove) {
            warnedRemove = true;
            console.warn(`${GUARD_TAG} removeChild 的目标已不在该父节点下（多为翻译扩展/编辑器移动 DOM），已跳过`, child);
        }
        return child;
    }
    return origRemoveChild.call(this, child);
} as unknown) as typeof Node.prototype.removeChild;

const origInsertBefore = Node.prototype.insertBefore;
Node.prototype.insertBefore = (function (this: Node, node: Node, ref: Node | null): Node {
    if (ref && ref.parentNode !== this) {
        if (!warnedInsert) {
            warnedInsert = true;
            console.warn(`${GUARD_TAG} insertBefore 的参照节点已不在该父节点下，已改为追加`, ref);
        }
        return this.appendChild(node);
    }
    return origInsertBefore.call(this, node, ref);
} as unknown) as typeof Node.prototype.insertBefore;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
