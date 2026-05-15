# analysis_logs.py
# 用于分析 logs/ui_nl_exec.jsonl 的 grasp 成功率（整体 + 按物体）

import json
from collections import Counter
from pathlib import Path

LOG_PATH = Path("logs/ui_nl_exec.jsonl")

# 你关心的几个物体关键词（和你实验保持一致）
TARGET_QUERIES = [
    "campbell's soup can",
    "hammer",
    "scissors",
]

def load_rows(path=LOG_PATH):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def stat(rows, filter_fn, name):
    subset = [r for r in rows if filter_fn(r)]
    if not subset:
        print(f"{name}: no data")
        return
    succ = sum(1 for r in subset if r.get("success"))
    rate = succ / len(subset)
    print(f"{name:30s}: {succ:3d}/{len(subset):3d} = {rate:.2f}")

def main():
    print(f"[INFO] Loading logs from {LOG_PATH} ...")
    rows = load_rows()
    print(f"[INFO] Total records: {len(rows)}\n")

    # 1) 总体成功率
    stat(rows, lambda r: not r.get("use_lggsn", False), "Baseline - ALL")
    stat(rows, lambda r: r.get("use_lggsn", False),     "With LGGSN - ALL")
    print()

    # 2) 统计每个 query（精确匹配）
    for q in TARGET_QUERIES:
        stat(rows,
             lambda r, q=q: r["query"].strip().lower() == q
                           and not r.get("use_lggsn", False),
             f"Baseline - {q}")
        stat(rows,
             lambda r, q=q: r["query"].strip().lower() == q
                           and r.get("use_lggsn", False),
             f"With LGGSN - {q}")
        print()

    # 3)（可选）看看最常见的 query 是什么
    qs = [r["query"].strip().lower() for r in rows]
    cnt = Counter(qs)
    print("Top queries:")
    for q, c in cnt.most_common(10):
        print(f"  {c:3d}x  {q!r}")

if __name__ == "__main__":
    main()

