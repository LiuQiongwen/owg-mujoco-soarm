# TANGO — Transport-Aligned Next-Grasp Optimizer

**TANGO** is a 6-DoF robotic grasping system that combines optimal-transport conditional flow matching (OT-CFM) with language-grounded VLM planning and a pairwise geometric reranker (LGGSN).

> **Key finding:** Optimal-transport coupling is the *load-bearing* component of the generator.
> Removing it causes the flow model to collapse below random-candidate selection (78.9% vs 80.6%)
> with complete mode failure on geometrically challenging objects.
> OT-CFM achieves **94.3%** success across 7 YCB objects (+14.3 pp over unranked, p < 0.001).

---

## Pipeline

```
RGB Image ──► SAM Encoder ──► object descriptor c ∈ R²⁵⁶
                                        │ (conditioning)
Training Poses ──► OT-CFM Generator ──► 5 × 6-DoF candidates
                                        │
                              LGGSN Reranker (BPR, 17 geo. feats)
                                        │
                                  top-1 grasp ──► Execute
```

| Stage | Description | Flag |
|-------|-------------|------|
| 1 | GR-ConvNet baseline (top-1 only) | `--stage 1` |
| 2 | 6-DoF random sampling, no ranking | `--stage 2` |
| 3 | VLM semantic grounding + sampling | `--stage 3` |
| **4** | **Stage 3 + OT-CFM + LGGSN reranker (best)** | `--stage 4` |

---

## Results (175 trials, 7 YCB objects, MuJoCo)

| Method | Success Rate |
|--------|-------------|
| Random CoM | 80.6% |
| GR-ConvNet 6-DoF | 82.3% |
| CFM (no OT) | 78.9% |
| DDPM DDIM-50 | 81.7% |
| **OT-CFM + LGGSN (ours)** | **94.3%** |

Inference: OT-CFM 3.47 ms/batch (20-step Euler) vs DDIM-50 14.32 ms — 4× faster.

---

## Installation

```bash
conda create -n owg-mujoco python=3.9
conda activate owg-mujoco
pip install -r requirements.txt
```

Set your OpenAI key for VLM grounding:
```bash
export OPENAI_API_KEY=your_key
```

---

## Quick Start

```bash
# Single demo (Stage 4, Banana, seed 1)
conda run -n owg-mujoco python demo.py \
  --stage 4 --prompt Banana --seed 1 --once --verbose 1

# Full evaluation (7 objects × 25 seeds = 175 trials)
bash scripts/quick_eval.sh full

# Stage 3 baseline comparison
bash scripts/quick_eval.sh full 3
```

---

## Training

```bash
# Train OT-CFM grasp generator
conda run -n owg-mujoco python train_cfm_grasp.py

# Train LGGSN reranker
conda run -n owg-mujoco python train_lggsn_pairwise.py
```

---

## Project Structure

```
tango/              VLM policy and visual prompting
tango_robot/        MuJoCo SO-ARM101 environment
robots/             Sim-to-real backend abstraction
grasp_6dof/         6-DoF grasp generation and LGGSN models
benchmark/          BenchmarkRunner and trial logging
scripts/            Evaluation, data collection, analysis
figures/            Publication figures (fig_results.pdf, pipeline.pdf)
```

---

## Citation

```bibtex
@article{tango2026,
  title   = {Optimal Transport is the Load-Bearing Ingredient:
             Conditional Flow Matching for 6-DoF Grasp Generation},
  author  = {Qiongwen},
  year    = {2026},
}
```
