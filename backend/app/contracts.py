"""模块三共享契约：规范指标名与指标归一化。

CANONICAL_METRICS 是「基准指标与模块二复现指标保持可比」（模块详细设计 5.2）
的落地点——阶段2'（模块二复现）计算指标时须复用同一套键名，否则 5.4 无法对比。
"""

CANONICAL_METRICS = (
    # 分类
    "accuracy", "precision", "recall", "f1", "auc", "specificity", "kappa",
    # 回归
    "mae", "mse", "rmse", "r2", "mape",
    # 通用
    "loss", "iou", "dice", "map", "top1", "top5",
)

_ALIASES = {
    "acc": "accuracy",
    "accuracy_score": "accuracy",
    "macro_f1": "f1",
    "micro_f1": "f1",
    "f1_score": "f1",
    "f1score": "f1",
    "auc_roc": "auc",
    "roc_auc": "auc",
    "mean_absolute_error": "mae",
    "mean_squared_error": "mse",
    "root_mean_squared_error": "rmse",
    "r2_score": "r2",
    "val_loss": "loss",
    "eval_loss": "loss",
    # 常见写法补充（漏项会让同一指标在两份结果里异名，5.4 取交集为空 → 误报「不可比」）
    "top-1": "top1",
    "top-5": "top5",
    "top_1": "top1",
    "top_5": "top5",
    "top1_acc": "top1",
    "top5_acc": "top5",
    "precision_score": "precision",
    "recall_score": "recall",
    "sensitivity": "recall",
    "mean_absolute_percentage_error": "mape",
    "average_precision": "ap",
    "iou_score": "iou",
    "dice_score": "dice",
    "mean_average_precision": "map",
}


def _canonical(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(key, key)


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_metrics(raw: dict) -> dict:
    """把 eval 入口输出的 metrics 归一到规范键名与浮点值。

    无法转成数值的项丢弃（如有需要保留的原值应写入 predictions/摘要，而非 metrics）。
    """
    out: dict = {}
    for key, value in (raw or {}).items():
        num = _to_float(value)
        if num is None:
            continue
        out[_canonical(key)] = num
    return out


def ordered_metrics(keys) -> list[str]:
    """按规范顺序排列指标名，未知指标追加在后（用于对比表与图表统一顺序）。"""
    known = [m for m in CANONICAL_METRICS if m in keys]
    extra = sorted(k for k in keys if k not in CANONICAL_METRICS)
    return known + extra
