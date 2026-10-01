"""模块三 5.2 基准 eval 入口模板（模块详细设计 5.2）。

平台以固定代码调用本入口，签名固定：

    python eval_entry.py <data_dir> <model_dir> <out_json>

    data_dir : 预处理后数据集目录，内含 preprocessed.csv
               （统一 schema: id / split / label / input，其余列为 meta_*）
    model_dir: 模型权重所在目录；平台未定位到权重时为空串
    out_json : 指标输出路径，入口须把指标 JSON 写入此处

指标 JSON 形状（metrics 的键名须取自 backend/app/contracts.py::CANONICAL_METRICS，
以保证与模块二复现指标同口径可比）：

    {
      "schema_version": "1.0",
      "task_type": "classification",
      "model": "resnet18",
      "dataset": "self",
      "metrics": {"accuracy": 0.91, "f1": 0.90, "loss": 0.21},
      "primary_metric": "accuracy",
      "n_samples": 1000,
      "per_class": {},
      "predictions": [
        {"id": "1", "y_true": 0, "y_pred": 1, "prob": 0.83, "path": "img/1.png"}
      ]
    }

per_class 与 predictions 可选，供 5.5 的误差分布图与典型案例图使用。

使用方式：把本文件复制到项目 source/ 下并命名为 eval_entry.py，实现 evaluate()。
"""
import json
import sys
from pathlib import Path


def evaluate(data_dir: Path, model_dir: str | None, out_json: Path) -> dict:
    """在此实现：读 data_dir/preprocessed.csv → 加载 model_dir 下的权重 → 计算指标。

    返回上述形状的指标字典。
    """
    raise NotImplementedError("请在项目 source/eval_entry.py 中实现 evaluate()")


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: python eval_entry.py <data_dir> <model_dir> <out_json>", file=sys.stderr)
        return 2
    data_dir = Path(sys.argv[1])
    model_dir = sys.argv[2] or None
    out_json = Path(sys.argv[3])

    result = evaluate(data_dir, model_dir, out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "out_json": str(out_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
