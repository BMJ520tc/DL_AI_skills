"""结构扫描的层级树/调用链用例（《模块详细设计》3.6 步骤 3「类嵌套、调用链」）。

造一个最小仓库（容器类 Net + 子模块 Block + 入口 train.py），断言：
- 新字段 `module_tree`（组合嵌套树，容器元素展开）与 `call_chain`（按求值顺序的调用链，
  含实例化）结构正确；
- 原字段 `module_hierarchy` 保持不变（向后兼容既有消费者：decompose_service / m3_acceptance）。

用临时目录，不碰真实 data/。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCAN_SCRIPT = PROJECT_ROOT / "scripts" / "scan_structure.py"


def _load_scan_module():
    spec = importlib.util.spec_from_file_location("scan_structure_under_test", SCAN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan_mod = _load_scan_module()

MODELS_PY = '''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        if c > 4:
            self.shortcut = nn.Conv2d(c, c, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class Net(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.blocks = nn.Sequential(Block(c), Block(c))
        self.head = nn.Linear(c, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
'''

TRAIN_PY = '''
import torch

from pkg.models import Net


def train(loader, epochs=1):
    model = Net(8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
    return model


def evaluate(model, loader):
    model.eval()
    return 0.0
'''


def _write_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "models.py").write_text(MODELS_PY, encoding="utf-8")
    (pkg / "train.py").write_text(TRAIN_PY, encoding="utf-8")
    return repo


def _rel(*parts: str) -> str:
    return str(Path(*parts))


def test_module_hierarchy_unchanged_and_tree_nests_container(tmp_path):
    report = scan_mod.scan(str(_write_repo(tmp_path)))
    rel_models = _rel("pkg", "models.py")

    # 原字段保持不变（{file, class, parent}，parent=第一个基类名）
    assert report["module_hierarchy"] == [
        {"file": rel_models, "class": "Block", "parent": "Module"},
        {"file": rel_models, "class": "Net", "parent": "Module"},
    ]

    # 新字段：根=未被其他仓库模块类实例化的类（Block 被 Net 用 → 只作子节点）
    assert [node["class"] for node in report["module_tree"]] == ["Net"]
    net = report["module_tree"][0]
    children = {child["attr"]: child for child in net["children"]}
    assert set(children) == {"stem", "blocks", "head"}

    seq = children["blocks"]
    assert seq["class"] == "nn.Sequential"
    assert seq["source"] == "builtin"
    assert [(e["class"], e["element_index"]) for e in seq["children"]] == [("Block", 0), ("Block", 1)]

    block = seq["children"][0]
    assert block["source"] == "repo"
    assert block["file"] == rel_models
    assert block["parent"] == "Module"
    # 类内 self.x = SomeClass(...) 嵌套关系；条件分支里的赋目标 conditional
    block_children = {c["attr"]: c for c in block["children"]}
    assert set(block_children) == {"conv1", "bn1", "shortcut"}
    assert block_children["conv1"]["class"] == "nn.Conv2d"
    assert block_children["conv1"]["source"] == "builtin"
    assert block_children["shortcut"]["conditional"] is True


def test_module_tree_does_not_fabricate_unknown_callees(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model.py").write_text(
        "import torch\n"
        "import torch.nn as nn\n\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.encoder = build_encoder()\n"
        "        self.act = getattr(nn, 'ReLU')()\n"
        "        self.recursive = Net()\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.encoder(x)\n",
        encoding="utf-8",
    )

    report = scan_mod.scan(str(repo))
    net = report["module_tree"][0]
    children = {c["attr"]: c for c in net["children"]}

    assert children["encoder"]["source"] == "unknown"       # 工厂函数：不猜类
    assert children["encoder"]["children"] == []
    assert children["act"]["source"] == "unknown"           # getattr(...) 调用：不猜
    assert children["recursive"]["class"] == "Net"
    assert children["recursive"].get("recursive") is True   # 自引用：标出而不是无限展开


def test_call_chain_call_order_and_instantiation(tmp_path):
    report = scan_mod.scan(str(_write_repo(tmp_path)))
    chains = {(c["kind"], c["class"], c["function"]): c for c in report["call_chain"]}

    # forward 链：按求值顺序（内层 self.conv1 先于外层 F.relu）
    block_fwd = chains[("forward", "Block", "forward")]
    assert block_fwd["basis"] == "ast_call_order"
    assert [c["callee"] for c in block_fwd["calls"]] == ["self.conv1", "self.bn1", "F.relu"]
    assert [c["kind"] for c in block_fwd["calls"]] == ["module_call", "module_call", "call"]
    assert block_fwd["calls"][0]["module_attr"] == "conv1"

    # 训练入口链：实例化 → 优化器 → 前向调用
    train_chain = chains[("train", None, "train")]
    kinds = {(c["kind"], c["callee"]) for c in train_chain["calls"]}
    assert ("instantiate", "Net") in kinds
    assert ("instance_call", "model") in kinds
    assert ("call", "torch.optim.Adam") in kinds
    inst = next(c for c in train_chain["calls"] if c["callee"] == "Net")
    assert inst["resolved_class"] == "Net"
    assert inst["assign_to"] == "model"
    # 裸名噪声调用（range）不入链
    assert all(c["callee"] != "range" for c in train_chain["calls"])

    # 推理入口链
    infer_chain = chains[("inference", None, "evaluate")]
    assert [c["callee"] for c in infer_chain["calls"]] == ["model.eval"]


def test_report_has_new_fields_even_without_module_classes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "utils.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    report = scan_mod.scan(str(repo))

    assert report["module_hierarchy"] == []
    assert report["module_tree"] == []
    assert report["call_chain"] == []
