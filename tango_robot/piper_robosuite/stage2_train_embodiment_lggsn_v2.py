"""
Re-verification of Stage 2 (see stage2_train_embodiment_lggsn.py) after
discovering and fixing a real RNG bug (2026-07-16): PiperMultiObjectScene's
placement sampler was using an unseeded, OS-entropy-based generator
(robosuite's UniformRandomSampler default), completely independent of
np.random.seed(scene_id) -- meaning the "25 scenes x 10 candidates" grouping
this whole Stage 2 analysis depends on never actually held object placement
fixed within a scene as intended. Fixed in piper_multi_object_scene.py;
this script re-runs Stage 2 against freshly re-collected data
(pairwise_results_v2_cracker_*.json) where scene-level pairing is now
genuinely valid, and additionally tests the new `candidate_grasp_yaw` field
(the true pre-commit candidate orientation, added alongside the RNG fix) --
the one per-candidate-varying, causally valid feature the original [z, H]
set was missing.

Two feature configurations tested:
  A: [z, H]                      -- re-verifies the existing paper claim
  B: [z, H, candidate_grasp_yaw] -- tests the new admissible feature

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.stage2_train_embodiment_lggsn_v2
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
DEVICE = "cpu"

audit_feature_set(["z", "H"], context="stage2_v2.py soarm_feat (config A)")
audit_feature_set(["spawn_pos", "object_H"], context="stage2_v2.py piper_feat (config A)")
audit_feature_set(["spawn_pos", "object_H", "candidate_grasp_yaw"], context="stage2_v2.py piper_feat (config B)")
# NOTE: config B pools SO-ARM101 (2-dim: z,H) with Piper (3-dim: z,H,yaw) --
# not directly poolable with a shared scorer without a matching SO-ARM101
# yaw-equivalent. Config B is therefore evaluated Piper-only (does
# candidate_grasp_yaw correlate with / predict success at all, zero-shot
# from a Piper-only model with leave-one-scene-out CV), not as a pooled
# cross-embodiment comparison -- a different, narrower question than
# Config A's, and stated as such in the output.


def soarm_feat(r):
    return [r["z"], r["H"]]


def piper_feat_a(r):
    return [r["spawn_pos"][2], r["object_H"]]


def piper_feat_b(r):
    return [r["spawn_pos"][2], r["object_H"], r["candidate_grasp_yaw"]]


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


def load_piper_episodes(piper_feat_fn):
    files = sorted(glob.glob(os.path.join(PIPER_DIR, "pairwise_results_v2_cracker_*.json")))
    episodes = {}
    for f in files:
        rows = json.load(open(f))
        by_scene = {}
        for r in rows:
            by_scene.setdefault(r["scene_id"], []).append(r)
        for scene_id, rs in by_scene.items():
            key = ("piper", "cracker", scene_id)
            pos = [piper_feat_fn(r) for r in rs if r["success"]]
            neg = [piper_feat_fn(r) for r in rs if not r["success"]]
            if pos and neg:
                episodes[key] = {"pos": pos, "neg": neg}
    return episodes


def make_pairs(episodes, embodiment_flag):
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


def train_model(pairs, mode, feature_dim, iters=1500, lr=0.01, l2=1e-3, seed=0):
    torch.manual_seed(seed)
    model = EmbodimentLGGSN(n_queries=1, geom_dim=feature_dim, n_embodiments=2, mode=mode, hidden_dim=16)
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
        loss = F.softplus(-(s_pos - s_neg)).mean()
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


def run_config_a():
    print("=" * 70)
    print("CONFIG A: [z, H], pooled SO-ARM101 + Piper (re-verifies existing paper claim)")
    print("=" * 70)
    soarm_eps = load_soarm_episodes()
    piper_eps = load_piper_episodes(piper_feat_a)
    print(f"SO-ARM101 mixed-label episodes: {len(soarm_eps)}")
    print(f"Piper mixed-label episodes (scenes): {len(piper_eps)}")

    all_soarm_feats = [f for ep in soarm_eps.values() for f in ep["pos"] + ep["neg"]]
    mu, sigma = fit_zscore(all_soarm_feats)
    all_piper_feats = [f for ep in piper_eps.values() for f in ep["pos"] + ep["neg"]]
    mu_pi, sigma_pi = fit_zscore(all_piper_feats)

    soarm_pairs_raw = make_pairs(soarm_eps, 0.0)
    soarm_pairs = zscore_pairs(soarm_pairs_raw, mu, sigma)

    piper_keys = list(piper_eps.keys())
    print(f"Piper scenes (leave-one-out folds): {len(piper_keys)}")

    modes = ["none", "additive", "interaction"]
    results = {m: [] for m in modes}
    zero_shot_results = []

    zero_shot_model = train_model(soarm_pairs, mode="none", feature_dim=2, seed=0)

    for fold_i, held_out_key in enumerate(piper_keys):
        train_piper_eps = {k: v for k, v in piper_eps.items() if k != held_out_key}
        test_piper_eps = {held_out_key: piper_eps[held_out_key]}

        train_piper_pairs_raw = make_pairs(train_piper_eps, 1.0)
        train_piper_pairs = zscore_pairs(train_piper_pairs_raw, mu_pi, sigma_pi)
        test_piper_pairs_raw = make_pairs(test_piper_eps, 1.0)
        test_piper_pairs = zscore_pairs(test_piper_pairs_raw, mu_pi, sigma_pi)

        test_piper_pairs_soarm_scale = zscore_pairs(test_piper_pairs_raw, mu, sigma)
        zero_shot_results.append(pairwise_accuracy(zero_shot_model, test_piper_pairs_soarm_scale))

        for mode in modes:
            pool = soarm_pairs + train_piper_pairs
            model = train_model(pool, mode=mode, feature_dim=2, seed=fold_i)
            results[mode].append(pairwise_accuracy(model, test_piper_pairs))

    print(f"\n{'condition':20s} | mean_pairwise_acc | std")
    z = np.array(zero_shot_results)
    print(f"{'zero_shot':20s} | {z.mean():.4f}            | {z.std():.4f}")
    for m in modes:
        v = np.array(results[m])
        print(f"{'pooled_' + m:20s} | {v.mean():.4f}            | {v.std():.4f}")

    from scipy import stats
    none_v = np.array(results["none"])
    print(f"\n[pooled_none vs zero_shot] diff={none_v.mean()-z.mean():+.4f}, "
          f"paired t-test p={stats.ttest_rel(none_v, z).pvalue:.4f}")
    return {"zero_shot": zero_shot_results, **{f"pooled_{m}": results[m] for m in modes}}


def run_config_b():
    print("\n" + "=" * 70)
    print("CONFIG B: [z, H, candidate_grasp_yaw], PIPER-ONLY (does the new")
    print("admissible per-candidate feature carry real signal for Cracker?)")
    print("=" * 70)
    piper_eps = load_piper_episodes(piper_feat_b)
    print(f"Piper mixed-label episodes (scenes): {len(piper_eps)}")

    all_piper_feats = [f for ep in piper_eps.values() for f in ep["pos"] + ep["neg"]]
    mu_pi, sigma_pi = fit_zscore(all_piper_feats)
    piper_keys = list(piper_eps.keys())

    accs = []
    for fold_i, held_out_key in enumerate(piper_keys):
        train_piper_eps = {k: v for k, v in piper_eps.items() if k != held_out_key}
        test_piper_eps = {held_out_key: piper_eps[held_out_key]}

        train_pairs = zscore_pairs(make_pairs(train_piper_eps, 1.0), mu_pi, sigma_pi)
        test_pairs = zscore_pairs(make_pairs(test_piper_eps, 1.0), mu_pi, sigma_pi)

        model = train_model(train_pairs, mode="none", feature_dim=3, seed=fold_i)
        accs.append(pairwise_accuracy(model, test_pairs))

    accs = np.array(accs)
    print(f"\nPiper-only [z, H, candidate_grasp_yaw], leave-one-scene-out: "
          f"mean_pairwise_acc={accs.mean():.4f} std={accs.std():.4f} (n_folds={len(accs)})")
    print(f"(majority baseline = 0.50)")

    # direct correlation check, matching the score_candidate_ik precedent
    from scipy.stats import pointbiserialr
    yaws, labels = [], []
    for f in sorted(glob.glob(os.path.join(PIPER_DIR, "pairwise_results_v2_cracker_*.json"))):
        for r in json.load(open(f)):
            yaws.append(r["candidate_grasp_yaw"])
            labels.append(1 if r["success"] else 0)
    corr, p = pointbiserialr(labels, yaws)
    print(f"point-biserial correlation(candidate_grasp_yaw, success) = {corr:.4f}, p={p:.4f}, n={len(yaws)}")
    return {"piper_only_z_H_yaw": accs.tolist(), "yaw_success_correlation": corr, "yaw_success_p": p}


def main():
    out_a = run_config_a()
    out_b = run_config_b()
    out_path = os.path.join(PIPER_DIR, "stage2_v2_results.json")
    with open(out_path, "w") as f:
        json.dump({"config_a": out_a, "config_b": out_b}, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
