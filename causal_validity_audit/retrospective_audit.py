"""
Retrospective audit: run the provenance registry (provenance.py) against
every feature set actually used across this project's cross-embodiment
reranking pilots, in the order they were tried. Demonstrates the tool would
have caught the causal-validity violation immediately, instead of only
being discovered after building Stage 2's full EmbodimentLGGSN architecture
and a 22-fold CV run.

Field names taken verbatim from the pilot scripts (xembod_pilot.py,
xembod_pilot_v2.py, stage2_train_embodiment_lggsn.py).
"""

from provenance import CausalValidityViolation, audit_feature_set

# NOTE (2026-07-16, fourth correction): the Piper-side entries below
# previously used the generic string "yaw" (matching the SO-ARM101 side's
# naming) instead of "grasp_yaw" -- the actual raw field name
# stage2_train_embodiment_lggsn.py's piper_feat() reads. Because "yaw" IS a
# separately-registered, correctly-PRE_EXECUTION field on the SO-ARM101 side
# (a raw candidate-pose component, no post-commit reassignment issue there),
# every Piper-side lookup of "yaw" silently resolved against the WRONG
# platform's entry instead of failing as UNREGISTERED -- masking
# grasp_yaw's contamination (see provenance.py's third correction) in every
# row below, including "Stage 2 CORRECTED, Piper side", the one row this
# whole retrospective demonstration exists to vouch for as clean. Fixed by
# using the real field name.
HISTORICAL_FEATURE_SETS = {
    "Pilot 1-2, SO-ARM101 side (3-feat: [z, score, need_dz])": ["z", "score", "need_dz"],
    "Pilot 1-2, Piper side (3-feat: [z, quality_score, correction_proxy])": ["z", "quality_score", "correction_proxy"],
    "Pilot 3, SO-ARM101 side (5-feat: [z, yaw, H, score, need_dz])": ["z", "yaw", "H", "score", "need_dz"],
    "Pilot 3, Piper side (5-feat: [z, grasp_yaw, H, quality_score, correction_proxy])": [
        "z", "grasp_yaw", "H", "quality_score", "correction_proxy",
    ],
    "Pilot 4 / Stage 2 pre-correction, SO-ARM101 side (reused Pilot 3's 5-feat)": [
        "z", "yaw", "H", "score", "need_dz",
    ],
    "Pilot 4 / Stage 2 pre-correction, Piper side (reused Pilot 3's 5-feat)": [
        "z", "grasp_yaw", "H", "quality_score", "correction_proxy",
    ],
    "Stage 2 CORRECTED, SO-ARM101 side (3-feat: [z, yaw, H])": ["z", "yaw", "H"],
    "Stage 2 CORRECTED, Piper side (3-feat: [z, grasp_yaw, H])": ["z", "grasp_yaw", "H"],
    "Follow-up check, Piper side (score_candidate_ik alone)": ["score_candidate_ik"],
}


def main():
    n_pass, n_fail = 0, 0
    for name, feats in HISTORICAL_FEATURE_SETS.items():
        try:
            audit_feature_set(feats, context=name)
            print(f"PASS  {name}")
            n_pass += 1
        except CausalValidityViolation as e:
            print(f"FAIL  {name}\n      -> {e}")
            n_fail += 1
    print(f"\n{n_pass} admissible, {n_fail} would have been flagged before ever training a model.")
    print(
        "\nThe leakage was concentrated entirely on the PIPER side "
        "(quality_score, correction_proxy, and -- caught later by "
        "auto_tagger.py's static analysis, not by hand -- grasp_yaw) -- all "
        "verified directly against piper_pick_and_place.py's "
        "run_pick_and_place. The SO-ARM101 side's score/need_dz, once traced "
        "against the ACTUAL live inference path (grasp_ranker_lggsn.py, "
        "policy.py, batch_s3s4.py) rather than a comment describing a "
        "different, inactive legacy dataset, turned out to be a harmless "
        "pre-execution proxy plus two dead constant-zero features -- not "
        "leakage. Every pooled pilot (SO-ARM101 side + Piper side combined) "
        "that showed the 'pooling beats zero-shot' effect is still flagged "
        "FAIL overall, because a pooled model is only as valid as its most-"
        "contaminated input.\n\n"
        "IMPORTANT: 'Stage 2 CORRECTED, Piper side' -- the one row this "
        "entire demonstration exists to vouch for as clean -- ALSO used to "
        "read 'yaw' instead of 'grasp_yaw', which silently resolved against "
        "the wrong (SO-ARM101) registry entry and masked grasp_yaw's "
        "contamination in every earlier version of this table. Fixed. "
        "Re-ran Stage 2 with genuinely clean [z, H] features (grasp_yaw "
        "dropped entirely, from both platforms for a fair shared feature "
        "space): the null finding survives -- zero_shot, pooled_none, "
        "pooled_additive, and pooled_interaction are STILL exactly "
        "identical (diff=0.0000) -- but the absolute pairwise accuracy "
        "dropped from 0.8236 (with the contaminated grasp_yaw included) to "
        "0.1327 (now below the 0.50 majority baseline) -- VERIFIED (not "
        "assumed) why: H is an exact dataset-wide constant (sigma=0 over "
        "n=250 rows), and z is a near-constant PER SCENE (14/25 scenes "
        "show within-scene spread under 2e-4, consistent with floating-"
        "point/physics-settling noise, because it is the object's own "
        "spawn height, read once before any candidate-specific action -- "
        "identical regardless of which pooled candidate is later "
        "attempted). Together [z, H] has almost no capacity to "
        "discriminate between different candidates drawn from the SAME "
        "scene, exactly the comparison the pairwise objective is trained "
        "and evaluated on. Reported plainly, not smoothed over -- this is "
        "a fourth correction to this project's own audit tool, caught by "
        "the tool's own static analysis rather than by hand, and it "
        "changed a number that had already been published in this "
        "project's documentation as 'clean.'"
    )


if __name__ == "__main__":
    main()
