#!/bin/bash
# Step 3D: recovery regression set across jaw contact models.
# Same protocol as the frozen seed-1041 confirmation run, at reduced n.
set -u
CKPT=paperA_data/lerobot_datasets/owg_3obj_act_clean_n50_deployable/checkpoints/010000/pretrained_model
N=${N:-15}
SEED=${SEED:-1041}
OUTDIR=outputs/step3d
mkdir -p "$OUTDIR"

declare -A TPL=(
  [cracker]=results/risk_gated_vla/act_recovery/recovery_success_templates_cracker_seed961_n15.json
  [mustard]=results/risk_gated_vla/act_recovery/recovery_success_templates_mustard_seed1001.json
  [drill]=results/risk_gated_vla/act_recovery/recovery_success_templates_drill_seed1001.json
)
declare -A RNK=(
  [cracker]=results/risk_gated_vla/act_recovery/recovery_candidate_oracle_cracker_templates15_safehome_train961_n30.json
  [mustard]=results/risk_gated_vla/act_recovery/recovery_candidate_oracle_mustard_templates15_safehome_train1001_n30.json
  [drill]=results/risk_gated_vla/act_recovery/recovery_candidate_oracle_drill_templates15_safehome_train1001_n30.json
)

for MODEL in proxy_spheres measured_pads_aimed; do
  for OBJ in cracker mustard drill; do
    OUT="$OUTDIR/recovery_${OBJ}_${MODEL}_seed${SEED}_n${N}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    echo "=== $OBJ / $MODEL ==="
    conda run -n tango python scripts/eval_attached_recovery_mujoco.py \
      --checkpoint "$CKPT" --object "$OBJ" --scenes "$N" --base-seed "$SEED" \
      --templates "${TPL[$OBJ]}" --ranker-train-results "${RNK[$OBJ]}" \
      --jaw-contact-model "$MODEL" --quiet --out "$OUT" 2>&1 \
      | grep -vE "OpenGL|Exception ignored|egl/|glCheckError" | tail -4
  done
done
echo "DONE"
