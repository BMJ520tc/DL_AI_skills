# M3 验收样例仓库

模块四（模型拆解 → 可视化 → 再生成 → 两步验证 → 标准化入库）的验收样例。

## 结构

- `train.py` — 训练入口（`python train.py --help` 可安全退出，满足模块一验证）
- `models/resnet.py` — 手写 ResNet18：BasicBlock 跳跃连接 + downsample shortcut、
  Sequential 容器（layer1-4 / shortcut）、BN、Dropout
- `models/skip_cnn.py` — 小跳跃 CNN（开发期快速链路，CPU 秒级）

覆盖 M3 验收所需的 IR 要素：多级嵌套（Sequential 容器 → BasicBlock 子模块 →
叶子层）、op 节点（`+=`/`+` 残差加、torch.flatten）、eval 模式必要性（BN/Dropout
在 train/eval 输出不同，数值比对必须 eval）、可调结构参数（conv 的 out_channels
等）。

## 用法

```bash
python train.py --help
python train.py --model skipcnn --dry-run   # CPU 秒级
python train.py --model resnet18 --dry-run  # 稍慢
```

## 模块四验收流程（前端操作链）

1. 后端创建原始项目，source_url 填本目录（或仓库地址）
2. 项目列表 → 模型查看器：① 结构分析 → ② 拆解 IR → ③ 补形状 → ④ 再生成 → ⑤ 两步验证 → ⑥ 入库
3. 验证通过后生成结构化项目 → 打开画布（层级 + 跳跃边）→ 改参数 → 保存 → 重开仍在
4. 失败场景：参数面板把某 conv 的 out_channels 改小 → 保存 → 重新验证 → overall=failed、
   diff_layers 定位差异层、入库按钮禁用

注：入口类 `ResNet`/`SkipCNN` 均可无参实例化（`ResNet` 构造参数带默认值 =
BasicBlock + [2,2,2,2] + num_classes=10），适配拆解/验证链路的模型加载约定。
