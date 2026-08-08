"""Migration v1, milestone 1: the first Piper static-critic benchmark.

Question, deliberately narrow: using ONLY pre-execution information and
under strict provenance rules, how much success/failure separation is
actually achievable?

Not a modelling exercise. Simple, auditable models only (logistic
regression, random forest) so every feature's provenance is traceable. A
stronger model on an untrustworthy benchmark is worth nothing.

Constraints enforced in code, per docs/MIGRATION_V1_CHARTER.md:
  - every feature is computable BEFORE the arm moves (asserted by
    construction: features come from env.reset() state only);
  - OUTCOME_DERIVED quantities are refused;
  - four mandated reports: overall AUC + calibration, per-object AUC,
    worst-object, and object-held-out (object-disjoint) evaluation;
  - rel_dist is deliberately EXCLUDED -- it is not pre-execution, and the
    audit showed it encodes object identity when pooled.

Run:  conda run -n tango python scripts/piper_static_critic_benchmark.py
"""
import json, os, sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import scene_objects_for
from scripts.piper_outcome_dataset import OUTCOME_DERIVED
from scripts.piper_contact_local_features import (
    finger_contact_zrange, local_normal, BAND_HALF_Y_M,
)
from scripts.piper_execution_trace import _mesh_world

DATA = ROOT / "outputs" / "piper_outcome_dataset.jsonl"
FEAT = ROOT / "outputs" / "piper_static_features.jsonl"
MAX_OPENING_M = 0.100


def extract(env, obj):
    """Pre-execution features: computed from reset state, before any motion."""
    m, d = env.sim.model._model, env.sim.data._data
    bid = env.object_body_ids[obj]
    quat = d.xquat[bid].copy()
    ref = ppp.true_centroid_xy(d.xpos[bid].copy(), quat, obj)
    gm = ppp.compute_grasp_orientation(env, obj)
    aim = ref + np.array([0.0, 0.0, ppp.GRASP_HEIGHT_OFFSET])
    zlo, zhi = finger_contact_zrange(env)

    pts = _mesh_world(m, d, bid)
    loc = (pts - aim) @ gm
    band = loc[(loc[:, 2] >= zlo) & (loc[:, 2] <= zhi) &
               (np.abs(loc[:, 1]) <= BAND_HALF_Y_M)]
    f = {}
    if len(band) >= 10:
        lx, rx = float(band[:, 0].min()), float(band[:, 0].max())
        f["support_width_mm"] = (rx - lx) * 1000
        f["opening_margin_mm"] = (MAX_OPENING_M - (rx - lx)) * 1000
        f["centring_error_mm"] = ((rx + lx) / 2) * 1000
        nl = local_normal(band, band[np.argmin(band[:, 0])])
        nr = local_normal(band, band[np.argmax(band[:, 0])])
        if nl is not None and nr is not None:
            ax = np.array([1.0, 0.0, 0.0])
            f["antipodal_score"] = float(min(
                abs(np.dot(nl / np.linalg.norm(nl), ax)),
                abs(np.dot(nr / np.linalg.norm(nr), ax))))
    rel = (d.xpos[bid] - aim) @ gm
    span = zhi - zlo
    f["envelope_fraction"] = float((rel[2] - zlo) / span) if span else 0.0
    f["obj_longitudinal_mm"] = float(rel[2]) * 1000
    f["spawn_radius_mm"] = float(np.hypot(ref[0], ref[1])) * 1000
    f["object_height_mm"] = float(pts[:, 2].max() - pts[:, 2].min()) * 1000
    return f


def build():
    rows = [json.loads(l) for l in DATA.open()]
    out = []
    for i, r in enumerate(rows):
        np.random.seed(r["seed"])
        env = PiperMultiObjectScene(
            robots="Piper", ycb_objects=scene_objects_for(r["object"]),
            has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, control_freq=20)
        try:
            env.reset()
            f = extract(env, r["object"])
        finally:
            env.close()
        f.update({"object": r["object"], "seed": r["seed"], "success": r["success"]})
        out.append(f)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(rows)}")
    with FEAT.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    return out


def auc(p, n):
    if not p or not n:
        return float("nan")
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in p for b in n) / (len(p) * len(n))


def main():
    rows = build() if not FEAT.exists() else [json.loads(l) for l in FEAT.open()]
    feat_keys = sorted({k for r in rows for k in r
                        if k not in ("object", "seed", "success")})
    leaked = set(feat_keys) & OUTCOME_DERIVED
    if leaked:
        raise ValueError(f"outcome-derived feature reached the critic: {leaked}")
    print(f"\n{len(rows)} samples, {sum(r['success'] for r in rows)} successes")
    print(f"features ({len(feat_keys)}): {feat_keys}")

    X = np.array([[r.get(k, np.nan) for k in feat_keys] for r in rows], dtype=float)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    y = np.array([float(r["success"]) for r in rows])
    objs = np.array([r["object"] for r in rows])

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss

    models = {
        "logreg": lambda: make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000, C=1.0)),
        "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                             random_state=0),
    }

    print("\n--- 1. overall (stratified 5-fold CV) ---")
    oof = {}
    for name, mk in models.items():
        pred = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            m = mk(); m.fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
        oof[name] = pred
        print(f"  {name:7s} AUC={roc_auc_score(y, pred):.3f}  "
              f"Brier={brier_score_loss(y, pred):.4f}  "
              f"(base rate {y.mean():.3f}, Brier of base-rate predictor "
              f"{brier_score_loss(y, np.full_like(y, y.mean())):.4f})")

    print("\n--- 2. per-object AUC (out-of-fold) ---  3. worst-object flagged")
    for name, pred in oof.items():
        per = {}
        for o in sorted(set(objs)):
            msk = objs == o
            yy = y[msk]
            if 0 < yy.sum() < len(yy):
                per[o] = roc_auc_score(yy, pred[msk])
            else:
                per[o] = float("nan")
        worst = min((v for v in per.values() if not np.isnan(v)), default=float("nan"))
        cells = "  ".join(f"{o}={v:.3f}" if not np.isnan(v) else f"{o}=n/a"
                          for o, v in per.items())
        print(f"  {name:7s} {cells}   WORST={worst:.3f}")

    print("\n--- 4. object-held-out (train on 3 objects, test on the 4th) ---")
    for name, mk in models.items():
        line = []
        for o in sorted(set(objs)):
            tr, te = objs != o, objs == o
            if not (0 < y[te].sum() < te.sum()):
                line.append(f"{o}=n/a(no split)")
                continue
            m = mk(); m.fit(X[tr], y[tr])
            line.append(f"{o}={roc_auc_score(y[te], m.predict_proba(X[te])[:,1]):.3f}")
        print(f"  {name:7s} " + "  ".join(line))

    print("\nnote: AUC 0.5 = chance. Per R1, a value landing suspiciously on")
    print("0.5, 1.0, or the base rate should be checked as instrumentation first.")


if __name__ == "__main__":
    main()
