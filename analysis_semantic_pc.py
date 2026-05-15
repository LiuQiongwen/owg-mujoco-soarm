# analysis_semantic_pc.py
import numpy as np
import glob
from collections import Counter
import os

SEM_DIR = "semantic_pc"

def load_all_semantic_pc(sem_dir=SEM_DIR):
    paths = sorted(glob.glob(os.path.join(sem_dir, "*.npz")))
    print(f"[INFO] Found {len(paths)} semantic_pc samples in {sem_dir}")
    data = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        pc = d["pc"]                # (N,3)
        q = str(d["query"]).strip().lower()
        meta = d["metas"].item() if "metas" in d.files else {}
        data.append((p, pc, q, meta))
    return data

# 你之前的 11 个物体，用关键词粗略归类
KEYWORDS = {
    "tennis ball": ["tennis"],
    "hammer": ["hammer"],
    "blue container": ["blue can", "blue container"],
    "green clamp": ["clamp"],
    "eraser": ["eraser"],
    "pringles can": ["pringles", "chips"],
    "scissors": ["scissors"],
    "mustard bottle": ["mustard", "yellow bottle"],
    "soup can": ["campbell", "soup"],
    "cheez-it box": ["cheez", "cracker box", "cheez-it"],
    "rectangular tin": ["tin", "green box"],
}

def match_category(query: str) -> str:
    for name, kws in KEYWORDS.items():
        if any(kw in query for kw in kws):
            return name
    return "other"

def main():
    data = load_all_semantic_pc()
    if not data:
        print("[WARN] No semantic_pc found.")
        return

    # 1) 点数统计
    lens = [pc.shape[0] for _, pc, _, _ in data]
    print("\n[STATS] Points per semantic point cloud:")
    print("  N samples   :", len(lens))
    print("  min points  :", min(lens))
    print("  mean points :", sum(lens) / len(lens))
    print("  max points  :", max(lens))

    # 2) 按物体类别统计
    cats = [match_category(q) for _, _, q, _ in data]
    cnt = Counter(cats)
    print("\n[STATS] Per-object sample counts:")
    for name, num in cnt.items():
        print(f"  {name:15s}: {num}")

    # 3) 随机看几条示例（方便你手动 sanity check）
    print("\n[EXAMPLES]")
    for i, (path, pc, q, meta) in enumerate(data[:5]):
        print(f"  {i+1}. {os.path.basename(path)}")
        print(f"     query = {q}")
        print(f"     pc.shape = {pc.shape}")
        if meta:
            print(f"     meta keys = {list(meta.keys())}")

if __name__ == "__main__":
    main()

