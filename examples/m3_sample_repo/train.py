"""M3 验收样例：ResNet18 / SkipCNN 训练入口。

用法:
    python train.py --help
    python train.py --model skipcnn --dry-run
    python train.py --model resnet18 --epochs 2
"""
import argparse


def train() -> int:
    parser = argparse.ArgumentParser(description="M3 验收样例训练入口")
    parser.add_argument("--model", choices=["resnet18", "skipcnn"], default="skipcnn")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="仅前向一次，不训练")
    args = parser.parse_args()

    import torch
    from torch import nn

    from models import build_model

    model = build_model(args.model, num_classes=10)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    print("output shape:", tuple(out.shape))
    total = sum(p.numel() for p in model.parameters())
    print("params:", total)
    if args.dry_run:
        return 0

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(args.epochs):
        y = torch.randint(0, 10, (2,))
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")
    return 0


def infer(checkpoint: str | None = None) -> int:
    """推理入口（结构分析的 inference_flow 识别用：eval/infer/predict 命名）。"""
    import torch
    from models import build_model

    model = build_model("skipcnn", num_classes=10)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    print("inference output shape:", tuple(out.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
