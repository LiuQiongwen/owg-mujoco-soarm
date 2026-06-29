#!/usr/bin/env python3
"""
Inference latency: OT-CFM vs DDPM (DDIM).

Measures wall-clock time to generate N_CANDS=5 grasp candidates per batch,
averaged over N_BATCHES=200 iterations (after WARMUP=30 warmup).

Usage:
  conda run -n owg-mujoco python scripts/benchmark_latency.py
  conda run -n owg-mujoco python scripts/benchmark_latency.py --device cpu
"""

import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_cfm_grasp      import VelocityNet, sample_poses
from train_diffusion_grasp import NoiseNet, sample_poses_ddpm

CFM_CKPT  = "grasp_6dof/models/cfm_allobj_ot.pt"
DDPM_CKPT = "grasp_6dof/models/ddpm_allobj.pt"

N_CANDS   = 5
N_BATCHES = 200
WARMUP    = 30

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="", help="cuda or cpu (default: auto)")
    return p.parse_args()

def load(ckpt, cls):
    m = cls(hidden=512)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd)
    return m

def benchmark(fn, dev, n_batches=N_BATCHES, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_batches):
        t0 = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))

def main():
    args = parse_args()
    if args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {dev}  |  Batches: {N_BATCHES}  |  Candidates/batch: {N_CANDS}  |  Warmup: {WARMUP}")

    results = []

    # ── OT-CFM 20 steps ───────────────────────────────────────────────────────
    if os.path.isfile(CFM_CKPT):
        cfm = load(CFM_CKPT, VelocityNet).eval().to(dev)
        cond_cfm = torch.randn(256, device=dev); cond_cfm /= cond_cfm.norm()
        def cfm20():
            return sample_poses(cfm, cond_cfm, n=N_CANDS, steps=20)
        mu, sd = benchmark(cfm20, dev)
        results.append(("OT-CFM", 20, mu, sd))
        print(f"  OT-CFM  (20 steps) : {mu:8.3f} ± {sd:.3f} ms/batch")
    else:
        print(f"  [SKIP] CFM checkpoint not found: {CFM_CKPT}")

    # ── DDIM at multiple step counts ──────────────────────────────────────────
    if os.path.isfile(DDPM_CKPT):
        ddpm = load(DDPM_CKPT, NoiseNet).eval().to(dev)
        cond_dp = torch.randn(256, device=dev); cond_dp /= cond_dp.norm()
        for steps in [20, 50, 100, 200]:
            def _fn(s=steps):
                return sample_poses_ddpm(ddpm, cond_dp, n=N_CANDS, steps=s)
            mu, sd = benchmark(_fn, dev)
            label = f"DDIM    ({steps:3d} steps)"
            results.append(("DDIM", steps, mu, sd))
            print(f"  {label} : {mu:8.3f} ± {sd:.3f} ms/batch")
    else:
        print(f"  [SKIP] DDPM checkpoint not found: {DDPM_CKPT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if results:
        cfm_ref = next((r for r in results if r[0] == "OT-CFM"), None)
        print("Summary (ms per 5-grasp batch):")
        print(f"  {'Method':<22}  {'Steps':>5}  {'Mean ms':>10}  {'vs CFM-20':>12}")
        for name, steps, mu, sd in results:
            if cfm_ref:
                ratio = f"{mu / cfm_ref[2]:.2f}×"
            else:
                ratio = "—"
            print(f"  {name:<22}  {steps:>5}  {mu:>10.3f}  {ratio:>12}")

if __name__ == "__main__":
    main()
