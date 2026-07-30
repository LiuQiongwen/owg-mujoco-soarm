"""
EXPERIMENT_PLAN.md Stage 2: offline/retrospective validation using the
REAL EmbodimentLGGSN architecture (lggsn_model.py) trained with genuine
BPR pairwise loss, instead of the toy logistic-regression proxy used in
IDEA_REPORT.md's pilots 1-3. Data: SO-ARM101's full live_candidates.jsonl
(dense pairwise) pooled with the Stage 1 Piper collection (25 scenes x 10
candidates = 250 Cracker trials, piper_pairwise_collector.py output).

Leave-scene-out CV on the Piper side (25 folds now, vs pilot 4's
uninterpretable 3 folds) -- the go/no-go gate EXPERIMENT_PLAN.md specifies
before any live physical Stage 3 test: does "interaction" mode beat
"none"/"additive" with a properly-sized dataset and the real architecture?

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.stage2_train_embodiment_lggsn
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/lena/projects/OWG-main")
from lggsn_model import EmbodimentLGGSN
from causal_validity_audit.provenance import audit_feature_set

SOARM_PATH = "/lena/projects/OWG-main/logs/lggsn_live_candidates.jsonl"
PIPER_DIR = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_OBJECTS = {"Scissors"}
FEATURE_DIM = 2
DEVICE = "cpu"

# Enforced gate, not just a retrospective check (see CAUSAL_VALIDITY_METHOD.md):
# fails fast at import time if soarm_feat/piper_feat below are ever edited to
# pull in an EXECUTION_DERIVED field again, instead of only being caught after
# a full training run's numbers look suspiciously good.
audit_feature_set(["z", "H"], context="stage2_train_embodiment_lggsn.py soarm_feat")
audit_feature_set(["spawn_pos", "object_H"], context="stage2_train_embodiment_lggsn.py piper_feat")

# CAUSALLY-VALID features only (2026-07-16 correction, SECOND PASS): the
# original 5-dim version included `score`/`need_dz` (SO-ARM101) and
# `quality_score`/`correction_proxy` (Piper) -- removed in the first
# correction. `yaw`/`grasp_yaw` was kept at that point, believed admissible.
# auto_tagger.py's static analysis then caught that Piper's `grasp_yaw`
# specifically is NOT admissible -- piper_pick_and_place.py reassigns
# grasp_mat post-commit at the "pre-close refresh" step (a post-descend
# env.sim.data.xquat read), and grasp_yaw is computed from that reassigned
# value, not the original pre-execution candidate orientation (see
# causal_validity_audit/provenance.py's third correction). SO-ARM101's
# `yaw` has no equivalent reassignment issue and remains legitimately
# admissible on its own, but a fair pooled/shared feature space needs the
# same fields on both sides, so it is dropped here too. Restricting to
# [z, H]: both are known from the candidate pose + object geometry alone,
# with no execution, IK solve, or post-commit re-measurement involved.
def soarm_feat(r):
    return [r["z"], r["H"]]


def piper_feat(r):
    return [r["spawn_pos"][2], r["object_H"]]


def load_soarm_episodes():
    rows = [json.loads(l) for l in open(SOARM_PATH)]
    rows = [r for r in rows if r["query"] not in EXCLUDE_OBJECTS]
    episodes = {}
    for r in rows:
        key = ("soarm", r["query"], r["scene_id"])
        episodes.setdefault(key, {"pos": [], "neg": []})
        side = "pos" if r["label"] == 1 else "neg"
        episodes[key][side].append(soarm_feat(r))
    return {k: v for k, v in episodes.items() if v["pos"] and v["neg"]}


def load_piper_episodes():
    files = sorted(glob.glob(os.path.join(PIPER_DIR, "pairwise_results_cracker_*.json")))
    episodes = {}
    for f in files:
        rows = json.load(open(f))
        by_scene = {}
        for r in rows:
            by_scene.setdefault(r["scene_id"], []).append(r)
        for scene_id, rs in by_scene.items():
            key = ("piper", "cracker", scene_id)
            pos = [piper_feat(r) for r in rs if r["success"]]
            neg = [piper_feat(r) for r in rs if not r["success"]]
            if pos and neg:
                episodes[key] = {"pos": pos, "neg": neg}
    return episodes


def make_pairs(episodes, embodiment_flag):
    """embodiment_flag: 0.0 for SO-ARM101, 1.0 for Piper."""
    pairs = []
    for ep in episodes.values():
        for pos in ep["pos"]:
            for neg in ep["neg"]:
                pairs.append((np.array(pos, dtype=np.float32), np.array(neg, dtype=np.float32), embodiment_flag))
    return pairs


def fit_zscore(feats):
    X = np.array(feats)
    return X.mean(axis=0), X.std(axis=0) + 1e-8


def zscore_pairs(pairs, mu, sigma):
    return [((p - mu) / sigma, (n - mu) / sigma, e) for p, n, e in pairs]


def onehot(flag):
    return torch.tensor([1.0 - flag, flag], dtype=torch.float32)


def train_model(pairs, mode, iters=1500, lr=0.01, l2=1e-3, seed=0):
    torch.manual_seed(seed)
    model = EmbodimentLGGSN(n_queries=1, geom_dim=FEATURE_DIM, n_embodiments=2, mode=mode, hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    rng = np.random.default_rng(seed)
    pairs = list(pairs)
    for _ in range(iters):
        idx = rng.integers(len(pairs))
        p, n, e = pairs[idx]
        p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
        n_t = torch.tensor(n, dtype=torch.float32).unsqueeze(0)
        emb_t = onehot(e).unsqueeze(0)
        qid = torch.zeros(1, dtype=torch.long)
        s_pos = model(p_t, qid, emb_t)
        s_neg = model(n_t, qid, emb_t)
        loss = F.softplus(-(s_pos - s_neg)).mean()  # -log sigmoid(pos-neg), numerically stable
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def pairwise_accuracy(model, pairs):
    if not pairs:
        return float("nan")
    correct = 0
    with torch.no_grad():
        for p, n, e in pairs:
            p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
            n_t = torch.tensor(n, dtype=torch.float32).unsqueeze(0)
            emb_t = onehot(e).unsqueeze(0)
            qid = torch.zeros(1, dtype=torch.long)
            s_pos = model(p_t, qid, emb_t).item()
            s_neg = model(n_t, qid, emb_t).item()
            if s_pos > s_neg:
                correct += 1
    return correct / len(pairs)


def main():
    soarm_eps = load_soarm_episodes()
    piper_eps = load_piper_episodes()
    print(f"SO-ARM101 mixed-label episodes: {len(soarm_eps)}")
    print(f"Piper mixed-label episodes (scenes): {len(piper_eps)}")

    all_soarm_feats = [f for ep in soarm_eps.values() for f in ep["pos"] + ep["neg"]]
    mu, sigma = fit_zscore(all_soarm_feats)
    all_piper_feats = [f for ep in piper_eps.values() for f in ep["pos"] + ep["neg"]]
    mu_pi, sigma_pi = fit_zscore(all_piper_feats)

    soarm_pairs_raw = make_pairs(soarm_eps, 0.0)
    soarm_pairs = [((p - mu) / sigma, (n - mu) / sigma, e) for p, n, e in soarm_pairs_raw]
    print(f"SO-ARM101 pairs: {len(soarm_pairs)}")

    piper_keys = list(piper_eps.keys())
    print(f"Piper scenes (leave-one-out folds): {len(piper_keys)}")

    modes = ["none", "additive", "interaction"]
    results = {m: [] for m in modes}
    zero_shot_results = []

    # train the zero-shot (SO-ARM101-only) model ONCE, reused across all folds (it never sees Piper data)
    zero_shot_model = train_model(soarm_pairs, mode="none", seed=0)

    for fold_i, held_out_key in enumerate(piper_keys):
        train_piper_eps = {k: v for k, v in piper_eps.items() if k != held_out_key}
        test_piper_eps = {held_out_key: piper_eps[held_out_key]}

        train_piper_pairs_raw = make_pairs(train_piper_eps, 1.0)
        train_piper_pairs = zscore_pairs(train_piper_pairs_raw, mu_pi, sigma_pi)
        test_piper_pairs_raw = make_pairs(test_piper_eps, 1.0)
        test_piper_pairs = zscore_pairs(test_piper_pairs_raw, mu_pi, sigma_pi)

        # zero-shot eval (reuse the pre-trained model, standardized with SO-ARM101's own mu/sigma
        # for a fair "what would you actually deploy zero-shot" comparison)
        test_piper_pairs_soarm_scale = zscore_pairs(test_piper_pairs_raw, mu, sigma)
        zero_shot_results.append(pairwise_accuracy(zero_shot_model, test_piper_pairs_soarm_scale))

        for mode in modes:
            pool = soarm_pairs + train_piper_pairs
            model = train_model(pool, mode=mode, seed=fold_i)
            results[mode].append(pairwise_accuracy(model, test_piper_pairs))

        if fold_i % 5 == 0:
            print(f"  fold {fold_i+1}/{len(piper_keys)} done ({held_out_key})")

    print(f"\n{'condition':20s} | mean_pairwise_acc | std")
    z = np.array(zero_shot_results)
    print(f"{'zero_shot':20s} | {z.mean():.4f}            | {z.std():.4f}")
    for m in modes:
        v = np.array(results[m])
        print(f"{'pooled_' + m:20s} | {v.mean():.4f}            | {v.std():.4f}")

    from scipy import stats
    none_v = np.array(results["none"])
    add_v = np.array(results["additive"])
    inter_v = np.array(results["interaction"])
    print()
    for label, x1, x2 in [("pooled_none vs zero_shot", none_v, z),
                           ("additive vs none", add_v, none_v),
                           ("interaction vs none", inter_v, none_v),
                           ("interaction vs additive", inter_v, add_v)]:
        t, p = stats.ttest_rel(x1, x2)
        print(f"[{label}] diff={x1.mean()-x2.mean():+.4f}, paired t-test p={p:.4f}")

    out = {
        "n_soarm_pairs": len(soarm_pairs), "n_piper_scenes": len(piper_keys),
        "zero_shot": zero_shot_results,
        **{f"pooled_{m}": results[m] for m in modes},
    }
    out_path = os.path.join(PIPER_DIR, "stage2_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
