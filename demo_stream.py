"""端到端演示：回放采集记录，输出分行交接结果与冲突留痕。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from caseio import load_case
from streamdeconv.feed import replay
from streamdeconv.products import GAP, PREVIEW, QUANTITATIVE


def main() -> None:
    parser = argparse.ArgumentParser(description="流式转置卷积封行交接演示")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--capacity", type=Path,
                        help="capacity.json，用于读取 max_unsealed_rows")
    parser.add_argument("--max-unsealed", type=int, default=None)
    args = parser.parse_args()

    records, tensors = load_case(args.capture)
    max_unsealed = args.max_unsealed
    if max_unsealed is None and args.capacity is not None:
        max_unsealed = json.loads(args.capacity.read_text(encoding="utf-8")).get(
            "max_unsealed_rows")

    engine, delivered, log = replay(records, tensors, max_unsealed)
    desc = engine.desc
    print(f"画幅 {desc.canvas}  裁剪输出 {desc.out_hw}  "
          f"通道顺序 {tuple(range(desc.out_channels))}")
    print(f"裁剪框 top,bottom,left,right = {desc.crop}")
    print("-" * 72)
    for subscription in (QUANTITATIVE, PREVIEW):
        print(f"订阅 {subscription} 共交接 {len(delivered[subscription])} 行")
        for product in delivered[subscription]:
            view = product.as_dict()
            gaps = view["gap_columns"]
            gap_text = f" 缺口列{gaps}" if gaps else ""
            measured = ("可测量@事件" + str(view["measurable_event"])
                        if view["measurable"] else "仅预览，不可测量")
            print(f"  行{product.row:>2} 封行@事件{product.sealed_event:>2} "
                  f"{measured}{gap_text} 来源{view['provenance']}")
            symbols = "".join(
                {"covered": "#", "gap": "x", "structural_zero": "."}[s]
                for s in product.coverage)
            print(f"      覆盖 {symbols}")
    print("-" * 72)
    for entry in log:
        if entry["kind"] == "conflict":
            rec = entry["record"]
            print(f"事件{entry['event']} 冲突：窗口 {rec['window']} 修订"
                  f" {rec['rejected_rev']} 被拒（生效修订 {rec['active_rev']}）")
            print(f"      原因：{rec['reason']}")
            print(f"      影响行：{rec['affected_rows']}（已留痕，像素未修改）")
        else:
            print(f"事件{entry['event']} 拥塞：{entry['blockage']['message']}")
    print("-" * 72)
    peak = engine.account.snapshot()
    print(f"工作区峰值 {peak['peak_bytes']} 字节，"
          f"当前驻留 {peak['live_bytes']} 字节，前沿 {engine.sealed_front}")
    print(f"未封行槽位 {len(engine._slots)}，冲突记录 {len(engine.conflicts)} 条")


if __name__ == "__main__":
    main()
