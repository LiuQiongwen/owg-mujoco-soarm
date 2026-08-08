"""P1.2: trajectory-prefix separability analysis.

Question this exists to answer: how much of a rollout must be observed
before success and failure become separable? The answer determines the
architecture of the next system, so it is worth settling empirically
rather than assuming:

  separable BEFORE execution      -> candidate feasibility critic
  separable only AFTER contact    -> temporal execution monitor
  separable only near lift        -> failure detection + recovery

Method: for each prefix cutoff (both phase-based and fraction-of-rollout),
compute summary features using ONLY the observed portion, then score each
feature's success/failure AUC under the same promotion gates as P1.1
(bootstrap CI excluding 0.5, direction consistency across objects,
minimum within-object effect). Reports the EARLIEST cutoff at which any
feature is promoted.

Guards carried over, each matching a real error from this investigation:
  - outcome-derived variables refused;
  - no pooled-only claims -- within-object always computed;
  - trajectory LENGTH is checked for leakage before anything else, since
    if failures produce shorter rollouts then any "prefix" feature is
    partly reading the outcome.

Run:  conda run -n tango python scripts/analyze_piper_trajectory_prefix.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAJ = ROOT / "outputs" / "piper_outcome_trajectories.jsonl"
OUT = ROOT / "outputs" / "piper_prefix_report.json"

PHASE_ORDER = ["transit_high", "approach", "descend", "descend_refresh",
               "lift", "transit_above_tray", "lower_into_tray"]
FRACTIONS = [0.10, 0.25, 0.50, 0.75]
MIN_PER_CLASS = 5


def auc(pos, neg):
    if len(pos) < 1 or len(neg) < 1:
        return None
    w = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def auc_ci(pos, neg, n_boot=1500, seed=0):
    if len(pos) < 3 or len(neg) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)
    v = []
    for _ in range(n_boot):
        v.append(auc(list(rng.choice(pos, len(pos), replace=True)),
                     list(rng.choice(neg, len(neg), replace=True))))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def prefix_features(traj):
    """Summary features over an already-truncated prefix. Everything here is
    computable from what has been observed so far -- nothing peeks ahead."""
    if not traj:
        return {}
    obj0 = np.array(traj[0]["obj"])
    q0 = np.array(traj[0]["obj_quat"])
    pen = [s["pen_mm"] for s in traj]
    grip = [s["grip_q"] for s in traj]
    rel = [np.linalg.norm(s["rel"]) for s in traj]
    objd = [float(np.linalg.norm(np.array(s["obj"]) - obj0)) for s in traj]

    def quat_ang(q):
        d = float(np.clip(abs(np.dot(q0, np.array(q))), -1.0, 1.0))
        return float(np.degrees(2 * np.arccos(d)))

    rot = [quat_ang(s["obj_quat"]) for s in traj]
    any_l = any(s["l"] for s in traj)
    any_r = any(s["r"] for s in traj)
    return {
        "min_grip_q": float(min(grip)),
        "last_grip_q": float(grip[-1]),
        "max_pen_mm": float(min(pen)),           # pen is negative on contact
        "any_contact": float(any_l or any_r),
        "any_bilateral": float(any_l and any_r),
        "max_obj_disp_mm": float(max(objd)) * 1000,
        "max_obj_rot_deg": float(max(rot)),
        "min_rel_dist_mm": float(min(rel)) * 1000,
        "last_rel_dist_mm": float(rel[-1]) * 1000,
        "max_qvel": float(max(s["qvel_max"] for s in traj)),
    }


def evaluate(rows, cutoff_name, get_prefix):
    feats, ys, objs = [], [], []
    for r in rows:
        pre = get_prefix(r["traj"])
        f = prefix_features(pre)
        if not f:
            continue
        feats.append(f); ys.append(r["success"]); objs.append(r["object"])
    if not feats:
        return []
    keys = sorted(feats[0])
    objects = sorted(set(objs))
    out = []
    for k in keys:
        P = [f[k] for f, y in zip(feats, ys) if y]
        N = [f[k] for f, y in zip(feats, ys) if not y]
        if len(P) < MIN_PER_CLASS or len(N) < MIN_PER_CLASS:
            continue
        if len(set(P + N)) < 2:
            continue
        a = auc(P, N)
        lo, hi = auc_ci(P, N)
        per = {}
        for o in objects:
            p = [f[k] for f, y, oo in zip(feats, ys, objs) if y and oo == o]
            n = [f[k] for f, y, oo in zip(feats, ys, objs) if not y and oo == o]
            per[o] = auc(p, n) if (len(p) >= 3 and len(n) >= 3) else None
        usable = [v for v in per.values() if v is not None]
        ci_ok = lo is not None and (lo > 0.5 or hi < 0.5)
        consistent = len(usable) >= 2 and (all(v > 0.5 for v in usable) or all(v < 0.5 for v in usable))
        weakest = min((abs(v - 0.5) for v in usable), default=0.0)
        out.append(dict(cutoff=cutoff_name, key=k, auc=a, lo=lo, hi=hi,
                        per_obj=per, promoted=bool(ci_ok and consistent and weakest >= 0.10),
                        weakest=weakest))
    return out


def main():
    rows = [json.loads(l) for l in TRAJ.open()]
    print(f"{len(rows)} trajectories, {sum(r['success'] for r in rows)} successes")

    # LEAKAGE CHECK FIRST: if trajectory length itself separates outcomes,
    # every prefix feature is partly reading the outcome.
    Ls = [len(r["traj"]) for r in rows]
    lp = [len(r["traj"]) for r in rows if r["success"]]
    ln = [len(r["traj"]) for r in rows if not r["success"]]
    la = auc(lp, ln)
    print(f"trajectory length: success mean {np.mean(lp):.1f}, failure mean {np.mean(ln):.1f}, "
          f"AUC {la:.2f}  (range {min(Ls)}-{max(Ls)})")
    if la is not None and abs(la - 0.5) > 0.15:
        print("  WARNING: rollout length itself separates outcomes -- prefix features")
        print("  computed on a FRACTION of steps would inherit that. Using phase-based")
        print("  cutoffs as the primary reading.")

    all_res = []
    print("\n" + "=" * 100)
    print("phase-based prefixes (observe up to and including each phase)")
    print("=" * 100)
    for i, ph in enumerate(PHASE_ORDER):
        allowed = set(PHASE_ORDER[:i + 1])

        def gp(traj, allowed=allowed):
            return [s for s in traj if s["phase"] in allowed]

        res = evaluate(rows, f"through_{ph}", gp)
        all_res += res
        if not res:
            print(f"  through {ph:20s} (no usable samples)")
            continue
        best = max(res, key=lambda d: abs(d["auc"] - 0.5))
        prom = [d for d in res if d["promoted"]]
        tag = f"  PROMOTED: {', '.join(d['key'] for d in prom)}" if prom else ""
        print(f"  through {ph:20s} best |AUC-0.5| = {abs(best['auc']-0.5):.2f} "
              f"({best['key']}, AUC {best['auc']:.2f}){tag}")

    print("\n" + "=" * 100)
    print("fraction-of-rollout prefixes")
    print("=" * 100)
    for fr in FRACTIONS:
        def gp(traj, fr=fr):
            return traj[:max(1, int(len(traj) * fr))]
        res = evaluate(rows, f"first_{int(fr*100)}pct", gp)
        all_res += res
        best = max(res, key=lambda d: abs(d["auc"] - 0.5)) if res else None
        prom = [d for d in res if d["promoted"]]
        tag = f"  PROMOTED: {', '.join(d['key'] for d in prom)}" if prom else ""
        if best:
            print(f"  first {int(fr*100):3d}%           best |AUC-0.5| = {abs(best['auc']-0.5):.2f} "
                  f"({best['key']}, AUC {best['auc']:.2f}){tag}")

    print("\n" + "=" * 100)
    promoted = [d for d in all_res if d["promoted"]]
    if promoted:
        order = {f"through_{p}": i for i, p in enumerate(PHASE_ORDER)}
        ph_prom = [d for d in promoted if d["cutoff"] in order]
        if ph_prom:
            earliest = min(ph_prom, key=lambda d: order[d["cutoff"]])
            print(f"EARLIEST promoted separation: {earliest['cutoff']} "
                  f"via '{earliest['key']}' (AUC {earliest['auc']:.2f} "
                  f"[{earliest['lo']:.2f},{earliest['hi']:.2f}])")
            print("\nall promoted, by cutoff:")
            for d in sorted(ph_prom, key=lambda d: order[d["cutoff"]]):
                per = ", ".join(f"{o}:{v:.2f}" for o, v in d["per_obj"].items() if v is not None)
                print(f"  {d['cutoff']:26s} {d['key']:20s} AUC {d['auc']:.2f}  [{per}]")
    else:
        print("NO prefix feature achieves promoted separation at any cutoff.")

    OUT.write_text(json.dumps(all_res, indent=1, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
