"""
Final, authoritative aggregation of every method x object comparison
collected during the 2026-07-09/10 seeding-bug audit and method-search, for
rewriting paper_final.tex's Table I/II/III from scratch.

All data on the FIXED (post seeding-bug-merge) code, n=50/condition unless
noted. Scissors excluded from OT-CFM-family / EBM comparisons where its own
separate CFM-name-matching bug makes the "OT-CFM"/"EBM" condition identical
to Baseline by construction (falls back to the same random-CoM sampler);
Scissors' own Baseline number is still reported since it does not depend on
that bug.

Methods with FULL 7-object (or 6, excl. Scissors) coverage at n=50:
  - Baseline (random CoM + LGGSN)
  - OT-CFM (original, condition-agnostic minibatch OT coupling) -- THE
    generator the paper originally claimed beats baseline; now the negative
    result driving the whole rewrite.
  - EBM v2 (energy-based scoring model, CEM-sampled, InfoNCE + adversarial
    hard-negative-mined training) -- the new candidate core method.

Methods tested only on 3 objects (Pear/TomatoSoupCan/CrackerBox) at n=25, as
a targeted diagnostic, NOT full main-table coverage:
  - Remove-OT (plain CFM, no OT coupling, same architecture/data)
  - DDPM (same architecture/data, diffusion instead of flow matching)
  - Stratified-OT (C2OT discrete-class-limit fix: per-object OT coupling)
  - EBM v1 (naive BCE training -- catastrophic failure, kept for the
    methodology narrative about adversarial negative mining, not as a
    competitive method)

Output: paperA_data/formal_results/final_paper_all_methods.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

BASE = Path(__file__).resolve().parent.parent
DIAG = BASE / "phase0_diag_extended"

OBJECTS_7 = ["Banana", "TomatoSoupCan", "Pear", "MustardBottle", "Scissors", "CrackerBox", "PowerDrill"]
OBJECTS_6 = ["Banana", "TomatoSoupCan", "Pear", "MustardBottle", "CrackerBox", "PowerDrill"]
OBJECTS_3 = ["Pear", "TomatoSoupCan", "CrackerBox"]


def load_count(files):
    succ, n = 0, 0
    for fname in files:
        rows = [json.loads(l) for l in open(DIAG / fname)]
        succ += sum(1 for r in rows if r["success"] == "true")
        n += len(rows)
    return succ, n


def two_prop_z(s1, n1, s2, n2):
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def wilson_ci(s, n, z=1.96):
    p = s / n
    denom = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (center - margin) / denom * 100, (center + margin) / denom * 100


# ── Baseline, n=50, all 7 objects ────────────────────────────────────────────
BASELINE_FILES = {
    "Banana":        ["clean_seed1_25_Banana_baseline.jsonl", "seed26_50_Banana_baseline.jsonl"],
    "TomatoSoupCan":  ["clean_seed1_25_TomatoSoupCan_baseline.jsonl", "seed26_50_TomatoSoupCan_baseline.jsonl"],
    "Pear":           ["clean_seed1_25_Pear_baseline.jsonl", "seed26_50_Pear_baseline.jsonl"],
    "MustardBottle":  ["clean_seed1_25_MustardBottle_baseline.jsonl", "seed26_50_MustardBottle_baseline.jsonl"],
    "Scissors":       ["scissors_recheck_baseline.jsonl", "clean_scissors_seed26_50_baseline.jsonl"],
    "CrackerBox":     ["clean_seed1_25_CrackerBox_baseline.jsonl", "seed26_50_CrackerBox_baseline.jsonl"],
    "PowerDrill":     ["clean_seed1_25_PowerDrill_baseline.jsonl", "seed26_50_PowerDrill_baseline.jsonl"],
}

# ── OT-CFM (original, condition-agnostic), n=50, all 7 objects ──────────────
OTCFM_FILES = {
    "Banana":        ["clean_seed1_25_Banana_otcfm.jsonl", "seed26_50_Banana_otcfm.jsonl"],
    "TomatoSoupCan":  ["clean_seed1_25_TomatoSoupCan_otcfm.jsonl", "seed26_50_TomatoSoupCan_otcfm.jsonl"],
    "Pear":           ["clean_seed1_25_Pear_otcfm.jsonl", "seed26_50_Pear_otcfm.jsonl"],
    "MustardBottle":  ["clean_seed1_25_MustardBottle_otcfm.jsonl", "seed26_50_MustardBottle_otcfm.jsonl"],
    "Scissors":       ["scissors_recheck_otcfm.jsonl", "clean_scissors_seed26_50_otcfm.jsonl"],
    "CrackerBox":     ["clean_seed1_25_CrackerBox_otcfm.jsonl", "seed26_50_CrackerBox_otcfm.jsonl"],
    "PowerDrill":     ["clean_seed1_25_PowerDrill_otcfm.jsonl", "seed26_50_PowerDrill_otcfm.jsonl"],
}

# ── EBM v2, n=50, 6 objects (no Scissors -- same name-matching bug as CFM) ──
EBM_FILES = {
    "Banana":        ["ebm_v2_full_Banana.jsonl"],
    "MustardBottle":  ["ebm_v2_full_MustardBottle.jsonl"],
    "PowerDrill":     ["ebm_v2_full_PowerDrill.jsonl"],
    "Pear":           ["ebm_v2_check_Pear.jsonl", "ebm_v2_full_Pear.jsonl"],
    "TomatoSoupCan":  ["ebm_v2_check_TomatoSoupCan.jsonl", "ebm_v2_full_TomatoSoupCan.jsonl"],
    "CrackerBox":     ["ebm_v2_check_CrackerBox.jsonl", "ebm_v2_full_CrackerBox.jsonl"],
}

# ── 3-object diagnostic methods, n=25 ────────────────────────────────────────
DIAGNOSTIC_FILES = {
    "Remove-OT":     {obj: [f"removeOT_check_{obj}.jsonl"] for obj in OBJECTS_3},
    "DDPM":          {obj: [f"ddpm_check_{obj}.jsonl"] for obj in OBJECTS_3},
    "Stratified-OT": {obj: [f"stratifiedOT_check_{obj}.jsonl"] for obj in OBJECTS_3},
    "EBM-v1-naive":  {obj: [f"ebm_check_{obj}.jsonl"] for obj in OBJECTS_3},
}


def main():
    rows = []

    baseline = {obj: load_count(files) for obj, files in BASELINE_FILES.items()}
    otcfm    = {obj: load_count(files) for obj, files in OTCFM_FILES.items()}
    ebm      = {obj: load_count(files) for obj, files in EBM_FILES.items()}

    for obj in OBJECTS_7:
        b_s, b_n = baseline[obj]
        assert b_n == 50, f"baseline {obj} n={b_n}"
        row = {"object": obj, "baseline_succ": b_s, "baseline_n": b_n,
               "baseline_pct": round(b_s / b_n * 100, 1)}
        if obj in otcfm:
            o_s, o_n = otcfm[obj]
            z, p = two_prop_z(o_s, o_n, b_s, b_n)
            row.update({"otcfm_succ": o_s, "otcfm_n": o_n, "otcfm_pct": round(o_s / o_n * 100, 1),
                        "otcfm_delta_pp": round(o_s / o_n * 100 - b_s / b_n * 100, 1),
                        "otcfm_p": round(p, 4)})
        if obj in ebm:
            e_s, e_n = ebm[obj]
            z, p = two_prop_z(e_s, e_n, b_s, b_n)
            row.update({"ebm_succ": e_s, "ebm_n": e_n, "ebm_pct": round(e_s / e_n * 100, 1),
                        "ebm_delta_pp": round(e_s / e_n * 100 - b_s / b_n * 100, 1),
                        "ebm_p": round(p, 4)})
        rows.append(row)

    out_path = BASE / "formal_results/final_paper_all_methods.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = ["object", "baseline_succ", "baseline_n", "baseline_pct",
                      "otcfm_succ", "otcfm_n", "otcfm_pct", "otcfm_delta_pp", "otcfm_p",
                      "ebm_succ", "ebm_n", "ebm_pct", "ebm_delta_pp", "ebm_p"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}\n")
    for r in rows:
        print(" ", r)

    # ── Pooled comparisons ───────────────────────────────────────────────────
    print("\n=== Pooled: Baseline vs OT-CFM, all 7 objects, n=350 each ===")
    tb_s = sum(baseline[o][0] for o in OBJECTS_7); tb_n = sum(baseline[o][1] for o in OBJECTS_7)
    to_s = sum(otcfm[o][0] for o in OBJECTS_7);    to_n = sum(otcfm[o][1] for o in OBJECTS_7)
    z, p = two_prop_z(to_s, to_n, tb_s, tb_n)
    lo_b, hi_b = wilson_ci(tb_s, tb_n); lo_o, hi_o = wilson_ci(to_s, to_n)
    print(f"Baseline {tb_s}/{tb_n}={tb_s/tb_n*100:.1f}% [Wilson {lo_b:.1f},{hi_b:.1f}]  "
          f"OT-CFM {to_s}/{to_n}={to_s/to_n*100:.1f}% [Wilson {lo_o:.1f},{hi_o:.1f}]  "
          f"delta={to_s/to_n*100-tb_s/tb_n*100:+.1f}pp  z={z:.3f}  p={p:.6f}")

    print("\n=== Pooled: Baseline vs OT-CFM, 6 objects excl. Scissors, n=300 each ===")
    tb_s6 = sum(baseline[o][0] for o in OBJECTS_6); tb_n6 = sum(baseline[o][1] for o in OBJECTS_6)
    to_s6 = sum(otcfm[o][0] for o in OBJECTS_6);    to_n6 = sum(otcfm[o][1] for o in OBJECTS_6)
    z, p = two_prop_z(to_s6, to_n6, tb_s6, tb_n6)
    print(f"Baseline {tb_s6}/{tb_n6}={tb_s6/tb_n6*100:.1f}%  OT-CFM {to_s6}/{to_n6}={to_s6/to_n6*100:.1f}%  "
          f"delta={to_s6/to_n6*100-tb_s6/tb_n6*100:+.1f}pp  z={z:.3f}  p={p:.6f}")

    print("\n=== Pooled: Baseline vs EBM v2, 6 objects excl. Scissors, n=300 each ===")
    te_s = sum(ebm[o][0] for o in OBJECTS_6); te_n = sum(ebm[o][1] for o in OBJECTS_6)
    z, p = two_prop_z(te_s, te_n, tb_s6, tb_n6)
    lo_e, hi_e = wilson_ci(te_s, te_n)
    print(f"Baseline {tb_s6}/{tb_n6}={tb_s6/tb_n6*100:.1f}%  EBM v2 {te_s}/{te_n}={te_s/te_n*100:.1f}% "
          f"[Wilson {lo_e:.1f},{hi_e:.1f}]  delta={te_s/te_n*100-tb_s6/tb_n6*100:+.1f}pp  z={z:.3f}  p={p:.6f}")

    # ── 3-object diagnostic table ────────────────────────────────────────────
    print("\n=== 3-object diagnostic (n=25 each, except Baseline/OT-CFM/EBM-v2 shown at their n=50 for reference) ===")
    diag_rows = []
    for obj in OBJECTS_3:
        row = {"object": obj,
               "baseline_n50": f"{baseline[obj][0]}/{baseline[obj][1]}={baseline[obj][0]/baseline[obj][1]*100:.0f}%",
               "otcfm_n50": f"{otcfm[obj][0]}/{otcfm[obj][1]}={otcfm[obj][0]/otcfm[obj][1]*100:.0f}%",
               "ebm_v2_n50": f"{ebm[obj][0]}/{ebm[obj][1]}={ebm[obj][0]/ebm[obj][1]*100:.0f}%"}
        for name, files_map in DIAGNOSTIC_FILES.items():
            s, n = load_count(files_map[obj])
            row[name] = f"{s}/{n}={s/n*100:.0f}%"
        diag_rows.append(row)
        print(" ", row)

    diag_path = BASE / "formal_results/final_paper_3obj_diagnostic.csv"
    with open(diag_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        w.writerows(diag_rows)
    print(f"\nwrote {diag_path}")


if __name__ == "__main__":
    main()
