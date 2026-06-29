#!/usr/bin/env python3
"""
Diagnostic: does object-frame-normalised (x_rel, y_rel) carry a grasp-quality signal?

Two data sources:
  A. v7 JSONL  — existing labels, random-CoM-perturbation candidates
  B. CFM live  — generate candidates on-the-fly via cfm_allobj_ot.pt, no labels
                 (spatial distribution analysis only)

Questions:
  1. After centroid subtraction: distribution shape (is it spatially structured)?
  2. Point-biserial r(x_rel, label), r(y_rel, label), r(dist_rel, label)
     in v7 data (where we have labels).
"""

import json, collections, os, sys, math
import numpy as np

# ── scipy only for point-biserial, graceful fallback ───────────────────────
try:
    from scipy.stats import pointbiserialr
    _SCIPY = True
except ImportError:
    _SCIPY = False
    def pointbiserialr(x, y):
        # manual implementation
        n = len(x); x = np.asarray(x, float); y = np.asarray(y, float)
        m1 = x[y==1].mean(); m0 = x[y==0].mean()
        sd = x.std()
        n1 = (y==1).sum(); n0 = (y==0).sum()
        r = ((m1 - m0) / sd) * math.sqrt(n1 * n0 / (n * (n-1)))
        # p-value approximation via t-dist: skip exact, return (r, nan)
        return r, float('nan')

# ════════════════════════════════════════════════════════════════════════════
# Part A: v7 JSONL analysis
# ════════════════════════════════════════════════════════════════════════════
JSONL = "grasp_6dof/dataset/lggsn_candidates_v7.jsonl"
rows = [json.loads(l) for l in open(JSONL)]

# Group by episode
ep_raw = collections.defaultdict(list)
for r in rows:
    ep_raw[(r["query"], r["scene_id"])].append(r)

# Per-episode centroid subtraction
all_x_rel, all_y_rel, all_dist, all_lbl = [], [], [], []
per_obj = collections.defaultdict(lambda: dict(x_rel=[], y_rel=[], dist=[], lbl=[]))

for (query, sid), cands in ep_raw.items():
    xs = np.array([c["x"] for c in cands])
    ys = np.array([c["y"] for c in cands])
    labels = np.array([c["label"] for c in cands])
    cx, cy = xs.mean(), ys.mean()
    x_rel = xs - cx
    y_rel = ys - cy
    dist  = np.sqrt(x_rel**2 + y_rel**2)

    all_x_rel.extend(x_rel.tolist())
    all_y_rel.extend(y_rel.tolist())
    all_dist.extend(dist.tolist())
    all_lbl.extend(labels.tolist())
    for k, v in zip(["x_rel","y_rel","dist","lbl"],
                    [x_rel, y_rel, dist, labels]):
        per_obj[query][k].extend(v.tolist())

all_x_rel = np.array(all_x_rel)
all_y_rel = np.array(all_y_rel)
all_dist  = np.array(all_dist)
all_lbl   = np.array(all_lbl)

print("=" * 68)
print("PART A: v7 JSONL — random CoM-perturbation candidates  (n=%d)" % len(all_lbl))
print("=" * 68)

print("\n[A1] Distribution of (x_rel, y_rel) after per-episode centroid sub:")
print(f"  x_rel : mean={all_x_rel.mean():.4f}  std={all_x_rel.std():.4f}  "
      f"range=[{all_x_rel.min():.4f}, {all_x_rel.max():.4f}]")
print(f"  y_rel : mean={all_y_rel.mean():.4f}  std={all_y_rel.std():.4f}  "
      f"range=[{all_y_rel.min():.4f}, {all_y_rel.max():.4f}]")
print(f"  dist  : mean={all_dist.mean():.4f}  std={all_dist.std():.4f}  "
      f"range=[{all_dist.min():.4f}, {all_dist.max():.4f}]")
print(f"  label : pos={int(all_lbl.sum())}  neg={int((all_lbl==0).sum())}  "
      f"SR={all_lbl.mean():.1%}")

print("\n[A2] Point-biserial r(feature, label) — ALL objects:")
for feat_name, feat_arr in [("x_rel", all_x_rel), ("y_rel", all_y_rel), ("dist", all_dist),
                              ("dist_to_centroid_orig", np.array([r["dist_to_centroid"] for r in rows]))]:
    r_val, p_val = pointbiserialr(feat_arr, all_lbl)
    sig = "**" if (not math.isnan(p_val) and p_val < 0.01) else \
          "*"  if (not math.isnan(p_val) and p_val < 0.05) else ""
    p_str = f"p={p_val:.3f}" if not math.isnan(p_val) else "p=n/a"
    print(f"  {feat_name:<28}  r={r_val:+.4f}  {p_str} {sig}")

print("\n[A3] Per-object point-biserial r(x_rel, label) and r(y_rel, label):")
print(f"  {'object':<12}  {'n':>5}  {'SR':>6}  {'r(x_rel)':>10}  {'r(y_rel)':>10}  {'r(dist)':>10}")
for q in sorted(per_obj):
    d = per_obj[q]
    xl = np.array(d["x_rel"]); yl = np.array(d["y_rel"])
    dl = np.array(d["dist"]);  lb = np.array(d["lbl"])
    rx, _ = pointbiserialr(xl, lb)
    ry, _ = pointbiserialr(yl, lb)
    rd, _ = pointbiserialr(dl, lb)
    print(f"  {q:<12}  {len(lb):>5}  {lb.mean():>6.1%}  "
          f"{rx:>+10.4f}  {ry:>+10.4f}  {rd:>+10.4f}")

# Compare: is dist_to_centroid from original JSONL same as recomputed?
orig_dist = np.array([r["dist_to_centroid"] for r in rows])
ep_list = []
for r in rows:
    ep_list.append((r["query"], r["scene_id"]))
# recomputed dist after centroid sub
recomp_dist = all_dist  # same order as rows
print(f"\n[A4] dist_to_centroid: orig vs recomputed after OFN")
print(f"  orig  mean={orig_dist.mean():.4f}  std={orig_dist.std():.4f}")
print(f"  recomp mean={recomp_dist.mean():.4f}  std={recomp_dist.std():.4f}")
print(f"  max abs diff = {np.abs(orig_dist - recomp_dist).max():.6f}")

# ════════════════════════════════════════════════════════════════════════════
# Part B: CFM candidate spatial analysis
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("PART B: CFM candidates — spatial distribution (no labels)")
print("=" * 68)

# Load CFM model and generate candidates for several objects × seeds
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

try:
    import torch
    from tango_robot.ui import _load_cfm_model, _cfm_sample_candidates

    CFM_CKPT = "grasp_6dof/models/cfm_allobj_ot.pt"
    cfm_model, cfm_stats = _load_cfm_model(CFM_CKPT)
    if cfm_model is None:
        print("  [CFM] Failed to load model — skipping Part B")
    else:
        objects_to_test = [
            ("Banana",       0.007, -0.370),   # typical spawn CoM from eval logs
            ("PowerDrill",   0.020, -0.380),
            ("CrackerBox",  -0.010, -0.360),
            ("MustardBottle",0.015, -0.375),
            ("Pear",        -0.005, -0.365),
        ]
        N_SEEDS = 10
        N_CANDS = 5
        GZ = 0.810   # approx table top z

        cfm_stats_summary = {}
        for obj_name, gx_base, gy_base in objects_to_test:
            all_xr, all_yr = [], []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed + 100)
                # vary CoM position slightly as in eval (±0.05 spawn range)
                gx = gx_base + rng.uniform(-0.05, 0.05)
                gy = gy_base + rng.uniform(-0.03, 0.03)
                cands = _cfm_sample_candidates(
                    cfm_model, cfm_stats, obj_name,
                    n=N_CANDS, gx=gx, gy=gy, gz=GZ, pe=0.003, rng=rng
                )
                if cands is None:
                    continue
                xs = np.array([c["position"][0] for c in cands])
                ys = np.array([c["position"][1] for c in cands])
                cx, cy = xs.mean(), ys.mean()
                all_xr.extend((xs - cx).tolist())
                all_yr.extend((ys - cy).tolist())

            if not all_xr:
                print(f"  {obj_name}: no CFM candidates generated (object not in training set?)")
                continue

            xr = np.array(all_xr); yr = np.array(all_yr)
            dist_r = np.sqrt(xr**2 + yr**2)

            # Uniformity test: if spatially structured, std should be large
            # and distribution should be bimodal or directional
            # For uniform CoM-perturbation: std ≈ uniform(0.06)/sqrt(12) ≈ 0.017
            # If CFM is structured: std could be different, distribution non-uniform

            # Compute spatial entropy proxy: std / max(std_uniform)
            std_uniform_expect = 0.06 / math.sqrt(3)  # uniform(-0.06, 0.06)
            struct_x = xr.std() / std_uniform_expect
            struct_y = yr.std() / std_uniform_expect

            # Directional bias: check if one quadrant is preferred
            q1 = ((xr>0) & (yr>0)).mean()
            q2 = ((xr<0) & (yr>0)).mean()
            q3 = ((xr<0) & (yr<0)).mean()
            q4 = ((xr>0) & (yr<0)).mean()
            max_q = max(q1, q2, q3, q4)

            cfm_stats_summary[obj_name] = {
                "std_x": xr.std(), "std_y": yr.std(),
                "struct_x": struct_x, "struct_y": struct_y,
                "dist_mean": dist_r.mean(), "dist_std": dist_r.std(),
                "max_quadrant": max_q,
            }
            print(f"\n  {obj_name} (n={len(xr)} cand×seed pairs):")
            print(f"    x_rel std={xr.std():.4f}  y_rel std={yr.std():.4f}  "
                  f"(uniform expect ≈{std_uniform_expect:.4f})")
            print(f"    struct_ratio x={struct_x:.2f}x  y={struct_y:.2f}x  "
                  f"(>1 = more spread than uniform, <1 = tighter clustering)")
            print(f"    dist_rel: mean={dist_r.mean():.4f}  std={dist_r.std():.4f}")
            print(f"    quadrant fractions: Q1={q1:.2f} Q2={q2:.2f} Q3={q3:.2f} Q4={q4:.2f}  "
                  f"max={max_q:.2f} (0.25=uniform, >0.40=biased)")

except Exception as e:
    print(f"  [CFM] Error: {e}")
    import traceback; traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
# Part C: 6-DoF pre-generated grasps from mesh — the "ideal" candidate pool
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("PART C: 6-DoF mesh-based grasps — spatial coverage analysis")
print("=" * 68)

OBJECT_GRASP_FILES = {
    "banana":   "grasp_6dof/dataset/gen_banana.json",
}
# Find all gen_*.json files
import glob
for f in sorted(glob.glob("grasp_6dof/dataset/gen_*.json")):
    obj = os.path.basename(f).replace("gen_","").replace(".json","").split("_")[0]
    OBJECT_GRASP_FILES[obj] = f

for obj_name, jpath in sorted(OBJECT_GRASP_FILES.items()):
    if not os.path.exists(jpath):
        continue
    try:
        data = json.load(open(jpath))
        grasps = data if isinstance(data, list) else data.get("grasps", [])
        if not grasps:
            continue
        xs = np.array([g["position"][0] for g in grasps if "position" in g])
        ys = np.array([g["position"][1] for g in grasps if "position" in g])
        if len(xs) < 5:
            continue
        cx, cy = xs.mean(), ys.mean()
        xr = xs - cx; yr = ys - cy
        dist_r = np.sqrt(xr**2 + yr**2)
        scores = np.array([g.get("score", 0) for g in grasps if "position" in g])
        r_x, _ = pointbiserialr(xr, (scores > scores.median()).astype(int))
        r_y, _ = pointbiserialr(yr, (scores > scores.median()).astype(int))
        r_d, _ = pointbiserialr(dist_r, (scores > scores.median()).astype(int))
        print(f"\n  {obj_name} ({len(xs)} grasps from {os.path.basename(jpath)}):")
        print(f"    x_rel std={xr.std():.4f}  y_rel std={yr.std():.4f}  "
              f"dist std={dist_r.std():.4f}")
        print(f"    r(x_rel, score>median)={r_x:+.4f}  "
              f"r(y_rel, score>median)={r_y:+.4f}  r(dist, score>median)={r_d:+.4f}")
    except Exception as e:
        print(f"  {obj_name}: {e}")

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("SUMMARY")
print("=" * 68)
rx_all, _ = pointbiserialr(all_x_rel, all_lbl)
ry_all, _ = pointbiserialr(all_y_rel, all_lbl)
rd_all, _ = pointbiserialr(all_dist,  all_lbl)
print(f"v7 random-cand: r(x_rel)={rx_all:+.4f}  r(y_rel)={ry_all:+.4f}  r(dist)={rd_all:+.4f}")
print(f"Interpretation:")
if max(abs(rx_all), abs(ry_all)) < 0.05:
    print("  → x_rel/y_rel ≈ 0 correlation  = NOISE in v7 data  (expected for random perturbations)")
    print("  → OFN on v7 data cannot help  (confirmed)")
else:
    print(f"  → |r| > 0.05 suggests SOME positional signal in v7 data")
if abs(rd_all) > 0.05:
    print(f"  → dist_rel r={rd_all:+.4f} has signal (expected: dist_to_centroid already captures this)")
