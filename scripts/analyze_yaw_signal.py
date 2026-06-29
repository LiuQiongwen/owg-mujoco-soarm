#!/usr/bin/env python3
"""
Diagnostic: does yaw_rel = wrap_to_pi(yaw - yaw_obj) carry useful grasp-quality signal?

Key finding already from code:
  training: load_obj(yaw=0.0)  → yaw_obj = 0  → yaw_rel = yaw
  eval:     spawn_obj()        → yaw = uniform(0,π)  → real mismatch!

Questions:
  A. r(yaw, label) per object (= r(yaw_rel, label) since yaw_obj=0 in training)
  B. PCA yaw_obj from candidate positions — is it stable?
  C. PCA yaw_obj from mesh-based grasps — shape analysis per object
  D. If we had correct yaw_obj, would yaw_rel signs be consistent across objects?
"""

import json, collections, os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scipy.stats import pointbiserialr
    _SCIPY = True
except ImportError:
    _SCIPY = False
    def pointbiserialr(x, y):
        x, y = np.asarray(x,float), np.asarray(y,float)
        m1=x[y==1].mean(); m0=x[y==0].mean(); sd=x.std()
        n1=(y==1).sum(); n0=(y==0).sum(); n=len(x)
        if sd < 1e-12: return 0.0, 1.0
        r = ((m1-m0)/sd)*math.sqrt(n1*n0/(n*(n-1)))
        return r, float('nan')

def wrap_to_pi(a):
    return ((np.asarray(a) + math.pi) % (2*math.pi)) - math.pi

# gripper symmetry: yaw and yaw+π are the same grasp → fold to [0, π/2)
def fold_yaw(yaw):
    """Map yaw to [0, π/2) by exploiting gripper 180° symmetry."""
    y = np.asarray(yaw) % math.pi
    return np.where(y > math.pi/2, math.pi - y, y)

JSONL = "grasp_6dof/dataset/lggsn_candidates_v7.jsonl"
rows = [json.loads(l) for l in open(JSONL)]

ep_raw = collections.defaultdict(list)
for r in rows:
    ep_raw[(r["query"], r["scene_id"])].append(r)

# ═══════════════════════════════════════════════════════════════════════════
print("═"*68)
print("PART A  —  v7 JSONL: r(yaw, label) at training time (yaw_obj=0)")
print("  Note: yaw_rel = yaw − 0 = yaw since collection used yaw=0.0")
print("═"*68)

all_yaw, all_sin, all_cos, all_fold, all_lbl = [], [], [], [], []
per_obj = collections.defaultdict(lambda: dict(yaw=[], sin_y=[], cos_y=[], fold=[], lbl=[]))

for (query, sid), cands in ep_raw.items():
    for c in cands:
        y = float(c["yaw"])
        lbl = int(c["label"])
        all_yaw.append(y)
        all_sin.append(math.sin(y))
        all_cos.append(math.cos(y))
        all_fold.append(float(fold_yaw(y)))
        all_lbl.append(lbl)
        per_obj[query]["yaw"].append(y)
        per_obj[query]["sin_y"].append(math.sin(y))
        per_obj[query]["cos_y"].append(math.cos(y))
        per_obj[query]["fold"].append(float(fold_yaw(y)))
        per_obj[query]["lbl"].append(lbl)

all_yaw  = np.array(all_yaw);  all_sin = np.array(all_sin)
all_cos  = np.array(all_cos);  all_fold = np.array(all_fold)
all_lbl  = np.array(all_lbl)

print(f"\n[A1] yaw distribution over all 3500 candidates:")
print(f"  yaw (world)  : mean={all_yaw.mean():.3f}  std={all_yaw.std():.3f}  "
      f"range=[{all_yaw.min():.3f}, {all_yaw.max():.3f}] rad")
print(f"  folded [0,π/2): mean={all_fold.mean():.3f}  std={all_fold.std():.3f}")

print(f"\n[A2] Global point-biserial correlations:")
for name, feat in [("yaw (raw)", all_yaw), ("sin(yaw)", all_sin),
                    ("cos(yaw)", all_cos), ("|yaw| folded", all_fold)]:
    r, p = pointbiserialr(feat, all_lbl)
    p_str = f"p={p:.3f}" if not math.isnan(p) else "p=n/a"
    sig = "**" if (not math.isnan(p) and p<0.01) else \
          "*"  if (not math.isnan(p) and p<0.05) else ""
    print(f"  {name:<20}  r={r:+.4f}  {p_str}{sig}")

print(f"\n[A3] Per-object r(yaw, label) — check for sign consistency across objects:")
print(f"  {'object':<12}  {'n':>5}  {'SR':>6}  {'r(yaw)':>9}  {'r(sin)':>9}  "
      f"{'r(cos)':>9}  {'r(fold)':>9}  {'σ(yaw)':>8}")
for q in sorted(per_obj):
    d = per_obj[q]
    yaw_a = np.array(d["yaw"]); lbl_a = np.array(d["lbl"])
    sin_a = np.array(d["sin_y"]); cos_a = np.array(d["cos_y"])
    fold_a = np.array(d["fold"])
    ry, _ = pointbiserialr(yaw_a, lbl_a)
    rs, _ = pointbiserialr(sin_a, lbl_a)
    rc, _ = pointbiserialr(cos_a, lbl_a)
    rf, _ = pointbiserialr(fold_a, lbl_a)
    print(f"  {q:<12}  {len(lbl_a):>5}  {lbl_a.mean():>6.1%}  "
          f"{ry:>+9.4f}  {rs:>+9.4f}  {rc:>+9.4f}  {rf:>+9.4f}  "
          f"{yaw_a.std():>8.4f}")

# Count sign flips for each representation
print(f"\n[A3b] Sign consistency across objects:")
for feat_name in ["yaw", "sin_y", "cos_y", "fold"]:
    rs = []
    for q in sorted(per_obj):
        d = per_obj[q]
        r, _ = pointbiserialr(np.array(d[feat_name]), np.array(d["lbl"]))
        rs.append(r)
    n_pos = sum(1 for r in rs if r > 0.02)
    n_neg = sum(1 for r in rs if r < -0.02)
    n_zero = len(rs) - n_pos - n_neg
    print(f"  {feat_name:<20}  pos={n_pos}  neg={n_neg}  ~0={n_zero}  "
          f"(consistent if all same sign, noisy if mixed)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print("PART B  —  PCA yaw_obj stability from candidate positions")
print("═"*68)
from grasp_6dof.grasp_sampler import compute_pca_axes

yaw_obj_per_ep = {}  # (query, sid) → (yaw_obj_pca, elongation, n_pts)
yaw_obj_by_query = collections.defaultdict(list)

for (query, sid), cands in ep_raw.items():
    xs = np.array([c["x"] for c in cands])
    ys = np.array([c["y"] for c in cands])
    pts = np.column_stack([xs, ys, np.zeros(len(xs))])  # add z=0 for PCA
    if len(pts) < 3:
        continue
    major, minor, elongation = compute_pca_axes(pts)
    # Sign convention: force major[0] >= 0
    if major[0] < 0:
        major = -major
    yaw_obj_pca = float(math.atan2(major[1], major[0]))
    yaw_obj_per_ep[(query, sid)] = (yaw_obj_pca, elongation, len(pts))
    yaw_obj_by_query[query].append((yaw_obj_pca, elongation))

print(f"\n[B1] Per-object yaw_obj_pca from candidate positions:")
print(f"     (5 random CoM-perturbation pts → PCA → major axis → yaw_obj)")
print(f"  {'object':<12}  {'n_ep':>5}  {'yaw_obj_mean':>13}  {'yaw_obj_std':>12}  "
      f"{'elongation_mean':>16}  {'elong_std':>10}")
print(f"  Note: if yaw_obj_std ≈ π/4 (0.785) → axis is random/unstable")
for q in sorted(yaw_obj_by_query):
    data = yaw_obj_by_query[q]
    yaw_vals = [d[0] for d in data]
    elong_vals = [d[1] for d in data]
    # Circular std for yaw
    sin_m = np.sin(yaw_vals).mean(); cos_m = np.cos(yaw_vals).mean()
    R_circ = math.sqrt(sin_m**2 + cos_m**2)
    circ_std = math.sqrt(-2*math.log(R_circ)) if R_circ > 0 else math.pi
    print(f"  {q:<12}  {len(yaw_vals):>5}  {np.mean(yaw_vals):>+13.4f}  "
          f"{circ_std:>12.4f}  {np.mean(elong_vals):>16.3f}  "
          f"{np.std(elong_vals):>10.3f}")

print(f"\n  Expected circular std if uniform: π/sqrt(3) ≈ {math.pi/math.sqrt(3):.3f} rad")
print(f"  Practical threshold: std < 0.3 = stable, > 0.5 = mostly noise")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print("PART B2  —  yaw_rel = yaw - yaw_obj_pca signal quality")
print("═"*68)

all_yr_pca, all_yr_lbl = [], []
per_obj_yr = collections.defaultdict(lambda: dict(yr=[], lbl=[]))

for (query, sid), cands in ep_raw.items():
    if (query, sid) not in yaw_obj_per_ep:
        continue
    yaw_obj_pca, _, _ = yaw_obj_per_ep[(query, sid)]
    for c in cands:
        yr = float(wrap_to_pi(float(c["yaw"]) - yaw_obj_pca))
        all_yr_pca.append(yr)
        all_yr_lbl.append(int(c["label"]))
        per_obj_yr[query]["yr"].append(yr)
        per_obj_yr[query]["lbl"].append(int(c["label"]))

all_yr_pca = np.array(all_yr_pca); all_yr_lbl = np.array(all_yr_lbl)

print(f"\n[B2a] Global r(yaw_rel_pca, label):")
r_global_yr, p_global = pointbiserialr(all_yr_pca, all_yr_lbl)
r_global_raw, _ = pointbiserialr(all_yaw, all_lbl)
print(f"  r(yaw_rel_pca) = {r_global_yr:+.4f}  (baseline r(yaw_raw) = {r_global_raw:+.4f})")

print(f"\n[B2b] Per-object r(yaw_rel_pca, label):")
print(f"  {'object':<12}  {'r(yaw_raw)':>12}  {'r(yaw_rel_pca)':>15}  Δr")
for q in sorted(per_obj_yr):
    d = per_obj_yr[q]
    yr = np.array(d["yr"]); lbl = np.array(d["lbl"])
    ry_raw, _ = pointbiserialr(np.array(per_obj[q]["yaw"]), np.array(per_obj[q]["lbl"]))
    ry_rel, _ = pointbiserialr(yr, lbl)
    delta = ry_rel - ry_raw
    marker = " ← improved" if abs(ry_rel) > abs(ry_raw) + 0.02 else ""
    marker = " ← WORSE" if abs(ry_rel) < abs(ry_raw) - 0.02 else marker
    print(f"  {q:<12}  {ry_raw:>+12.4f}  {ry_rel:>+15.4f}  {delta:>+7.4f}{marker}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print("PART C  —  Object shape analysis from mesh-based grasp files")
print("             (proxy for PCA stability of episode point cloud)")
print("═"*68)

import glob

OBJ_DISPLAY = {
    "banana": "Banana      (elongated)",
    "pear":   "Pear        (near-symmetric)",
    "mustard":"MustardBottle(elongated bottle)",
    "cracker":"CrackerBox  (box, near-symmetric)",
    "drill":  "PowerDrill  (irregular)",
    "can":    "TomatoSoupCan(cylinder, symmetric)",
    "cylinder":"Cylinder   (clamp)",
}

for gfile in sorted(glob.glob("grasp_6dof/dataset/gen_*.json")):
    obj_key = os.path.basename(gfile).replace("gen_","").replace(".json","").split("_")[0]
    try:
        data = json.load(open(gfile))
        grasps = data if isinstance(data, list) else data.get("grasps", [])
        if not grasps or not isinstance(grasps[0], dict):
            continue
        xs = np.array([g["position"][0] for g in grasps if "position" in g])
        ys = np.array([g["position"][1] for g in grasps if "position" in g])
        if len(xs) < 10:
            continue
        pts = np.column_stack([xs, ys, np.zeros(len(xs))])
        major, minor, elongation = compute_pca_axes(pts)

        # Simulate PCA stability: bootstrap 20 samples of N_SAMPLE points
        N_SAMPLE = 20
        yaw_boots = []
        rng = np.random.default_rng(42)
        for _ in range(200):
            idx = rng.choice(len(pts), size=min(N_SAMPLE, len(pts)), replace=False)
            m, _, _ = compute_pca_axes(pts[idx])
            if m[0] < 0: m = -m
            yaw_boots.append(math.atan2(m[1], m[0]))

        # Circular std of bootstrap
        sin_m = np.sin(yaw_boots).mean(); cos_m = np.cos(yaw_boots).mean()
        R = math.sqrt(sin_m**2 + cos_m**2)
        boot_std = math.sqrt(-2*math.log(R)) if R > 0.01 else math.pi

        stability = "STABLE" if boot_std < 0.3 else \
                    "MODERATE" if boot_std < 0.6 else "UNSTABLE"
        sym_gate_pass = elongation > (1/0.85)  # only normalise if elong > 1.18

        obj_label = OBJ_DISPLAY.get(obj_key, obj_key)
        print(f"\n  {obj_label}  ({len(xs)} mesh grasps):")
        print(f"    XY elongation = {elongation:.3f}  (>1.18 = not symmetric → normalise)")
        print(f"    PCA bootstrap std (N={N_SAMPLE}) = {boot_std:.3f} rad  → {stability}")
        print(f"    Symmetry gate would {'PASS (normalise)' if sym_gate_pass else 'BLOCK (skip, too symmetric)'}")
    except Exception as e:
        pass

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print("PART D  —  Train/Inference yaw_obj mismatch (code-based proof)")
print("═"*68)

print("""
  collect_lggsn_data.py  → env.load_obj(yaw=0.0)     → yaw_obj = 0   for ALL episodes
  demo.py / quick_eval   → env.spawn_obj()            → yaw = uniform(0, π) per episode

  Consequence: the world-frame yaw in v7 JSONL represents gripper yaw at a FIXED
  object orientation (yaw_obj=0).  At inference, object yaw = uniform(0, π), so
  the same physical approach angle (e.g., "gripper perpendicular to object long axis")
  maps to a different world-frame yaw value.  This is a REAL train/inference mismatch.

  However: whether this matters depends on whether grasp quality has a preferred
  gripper-yaw direction relative to the object's axis.
""")

# Check: does the v7 data show a preferred gripper yaw direction per object?
print("[D1] Preferred gripper yaw direction per object (at yaw_obj=0 in training):")
print(f"  {'object':<12}  {'yaw(pos) mean':>14}  {'yaw(neg) mean':>14}  "
      f"{'|Δmean|':>9}  Preferred approach")
for q in sorted(per_obj):
    d = per_obj[q]
    yaw_a = np.array(d["yaw"]); lbl_a = np.array(d["lbl"])
    pos_yaw = yaw_a[lbl_a==1]; neg_yaw = yaw_a[lbl_a==0]
    # Circular mean
    def circ_mean(a):
        return math.atan2(np.sin(a).mean(), np.cos(a).mean())
    pm = circ_mean(pos_yaw); nm = circ_mean(neg_yaw)
    delta = abs(wrap_to_pi(pm - nm))
    # Interpret direction
    if delta < 0.2:
        direction = "no preferred direction"
    elif 0.8 < abs(pm) < 1.0 or 0.8 < abs(pm - math.pi/2) < 1.0:
        direction = f"preferred yaw≈{pm:.2f} rad"
    else:
        direction = f"preferred yaw≈{pm:.2f} rad"
    print(f"  {q:<12}  {pm:>+14.4f}  {nm:>+14.4f}  {delta:>9.4f}  {direction}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print("PART E  —  yaw_rel with KNOWN yaw_obj (MuJoCo simulation, 5 seeds × 3 objects)")
print("═"*68)

try:
    import os as _os
    _os.environ.setdefault("MUJOCO_GL", "egl")
    from tango_robot.env_soarm import EnvironmentSoArm

    env = EnvironmentSoArm(render=False, width=224, height=224)

    OBJECTS = [
        ("YcbBanana",       "banana",   "Banana"),
        ("YcbPowerDrill",   "drill",    "PowerDrill"),
        ("YcbTomatoSoupCan","can",      "TomatoSoupCan"),
        ("YcbCrackerBox",   "cracker",  "CrackerBox"),
    ]
    N_SEEDS = 10

    print(f"\n  Object-yaw distribution from spawn_obj (same as eval):")
    print(f"  {'object':<14}  {'yaw_obj_mean':>14}  {'yaw_obj_std':>14}  "
          f"{'range':>20}  stability")
    for pool_name, key, display_name in OBJECTS:
        yaw_objs = []
        for seed in range(N_SEEDS):
            np.random.seed(seed + 200)
            env.remove_all_obj()
            obj_id = env.spawn_obj(pool_name, name=display_name)
            # Get the quaternion of the spawned object
            for _ in range(100):  # let it settle
                import mujoco
                mujoco.mj_step(env.model, env.data)
            # Read freejoint quat: w,x,y,z at data.qpos for the object joint
            # Object joints follow arm joints (5 joints = 5 qpos values for arm)
            # Free joint = 7 values: x,y,z,qw,qx,qy,qz
            try:
                # Find the object's freejoint qpos offset
                slot = env._pool_names.index(pool_name)
                jnt_name = f"obj{slot}_freejoint"
                jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                if jnt_id < 0:
                    # try alternative naming
                    for jname in [f"object{slot}_joint", f"freejoint{slot}"]:
                        jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                        if jnt_id >= 0:
                            break
                if jnt_id >= 0:
                    qa = env.data.qpos[env.model.jnt_qposadr[jnt_id]:]
                    # qpos for freejoint: xyz(3) + quat(4) = qw,qx,qy,qz
                    qx, qy, qz, qw = qa[3], qa[4], qa[5], qa[6]
                    # Yaw from quat: arctan2(2*(qw*qz+qx*qy), 1-2*(qy^2+qz^2))
                    yaw_obj = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))
                    yaw_objs.append(yaw_obj)
                else:
                    # Fallback: use stored obj_orientations
                    orn = env.obj_orientations[-1]  # [w,x,y,z]
                    qw, qx, qy, qz = orn[0], orn[1], orn[2], orn[3]
                    yaw_obj = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))
                    yaw_objs.append(yaw_obj)
            except Exception as e2:
                # Last fallback: use stored orientations
                orn = env.obj_orientations[-1]  # [w,x,y,z]
                qw, qx, qy, qz = orn[0], orn[1], orn[2], orn[3]
                yaw_obj = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))
                yaw_objs.append(yaw_obj)

        if yaw_objs:
            arr = np.array(yaw_objs)
            sin_m = np.sin(arr).mean(); cos_m = np.cos(arr).mean()
            R = math.sqrt(sin_m**2 + cos_m**2)
            circ_std = math.sqrt(-2*math.log(R)) if R > 0.01 else math.pi
            print(f"  {display_name:<14}  {np.mean(arr):>+14.4f}  "
                  f"{circ_std:>14.4f}  [{arr.min():.3f}, {arr.max():.3f}]  "
                  f"({'uniform' if circ_std > 0.8 else 'biased'})")

    env.close()

except Exception as e:
    print(f"  [MuJoCo] {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print("SUMMARY  —  Should we build yaw_rel normalization (v11)?")
print("═"*68)
r_yaw_global, _ = pointbiserialr(all_yaw, all_lbl)
r_fold_global, _ = pointbiserialr(all_fold, all_lbl)
print(f"""
Key findings:
  1. r(yaw_raw, label) globally    = {r_yaw_global:+.4f}
  2. r(yaw_folded, label) globally = {r_fold_global:+.4f}
  3. yaw_obj mismatch: training yaw_obj=0, inference yaw_obj~uniform(0,π) ← CONFIRMED
  4. See per-object sign consistency in A3b above

Decision criteria:
  → Route 1 (don't do v11):
       per-object r(yaw) has mixed signs (like x_rel)  AND
       PCA yaw_obj from grasp positions is unstable (std > 0.5)
  → Route 2 (do v11, at least for PowerDrill):
       per-object r(yaw) has consistent sign for asymmetric objects  AND
       PCA yaw_obj from mesh grasps is stable (std < 0.3) for those objects
""")
