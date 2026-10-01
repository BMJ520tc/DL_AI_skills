"""模块二 4.3 复现脚本模板（模块详细设计 4.3，D5/八.3 运行执行机制）。

平台按条目逐条调用本脚本，在模块一项目独立环境（conda/venv）中以子进程执行：

    python reproduce.py <source_dir> <item_json> <out_json>

    source_dir : 模块一项目的 source/ 目录（论文对应代码仓库）
    item_json  : 单个实验条目 JSON（dataset_name/split_method/metric_name/
                 metric_value_reported/metric_unit/hyperparams/baselines）
    out_json   : 实际指标输出路径，脚本须把结果 JSON 写入此处

输出 JSON 形状（metric_name 的键名须取自 backend/app/contracts.py::CANONICAL_METRICS，
以保证与模块三基准指标同口径可比，见 5.2）：

    {
      "schema_version": "1.0",
      "item_id": "…",
      "metric_name": "accuracy",
      "metric_value_actual": 0.905,
      "n_samples": 1000,
      "log_tail": "最后若干行日志（可选）"
    }

【平台约定】复现失败时不要吞异常：非 0 退出即视为「无法复现」，平台据此写入
reproduction_result(verdict=无法复现) 并保留日志，不中断其他条目（4.3 异常与边界）。
"""
import json
import sys


def reproduce(source_dir: str, item: dict) -> dict:
    """按条目复现实验，返回实际指标。

    agent 填空处：在此依据 item 的 dataset/split/hyperparams/baselines 与项目代码
    执行评测，取回 metric_name 对应的实际数值。
    """
    raise NotImplementedError("请在 agent 步骤中依据项目代码实现 reproduce()")


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: reproduce.py <source_dir> <item_json> <out_json>", file=sys.stderr)
        raise SystemExit(2)

    source_dir, item_json, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    item = json.loads(open(item_json, encoding="utf-8").read())

    result = reproduce(source_dir, item)
    payload = {
        "schema_version": "1.0",
        "item_id": item.get("item_id"),
        "metric_name": result.get("metric_name") or item.get("metric_name"),
        "metric_value_actual": result.get("metric_value_actual"),
        "n_samples": result.get("n_samples"),
        "log_tail": result.get("log_tail"),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
