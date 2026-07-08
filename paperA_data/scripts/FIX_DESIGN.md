# Fix design — NOT applied to production code

## Fix 1: per-trial independent sampling

**Root cause**: `train_cfm_grasp.py:45` and `train_diffusion_grasp.py:47` call
`torch.manual_seed(SEED)` / `np.random.seed(SEED)` at **module level** (executes
on import, unconditionally — not guarded by `if __name__ == "__main__":`).
`_load_cfm_model()` (`ui.py:29`) does `from train_cfm_grasp import VelocityNet,
sample_poses` — triggering this import-time reset — inside `RobotEnvUI.__init__`,
which runs once per `demo.py` process. Since each of the 175 eval trials is a
**separate subprocess**, this reset happens identically every trial, and
`sample_poses`'s only randomness (`torch.randn(n, POSE_DIM, device=device)` at
line 191) draws from the same post-reset RNG state every time.

**Design**: stop relying on global RNG state for inference-time sampling.
Thread an explicit seed through a local `torch.Generator` instead.

```python
# train_cfm_grasp.py — sample_poses signature change
def sample_poses(model: VelocityNet, cond: torch.Tensor,
                 n: int = N_SAMPLES, steps: int = ODE_STEPS,
                 seed: int | None = None) -> np.ndarray:
    model.eval()
    cond = cond.unsqueeze(0).expand(n, -1).to(device) if cond.dim() == 1 else cond.to(device)
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        x = torch.randn(n, POSE_DIM, device=device, generator=gen)
    else:
        x = torch.randn(n, POSE_DIM, device=device)   # legacy behavior, global RNG
    ...  # ODE integration unchanged (deterministic given x)
```

Same pattern for `train_diffusion_grasp.py:sample_poses_ddpm` — thread `seed`
into its `torch.randn(n, POSE_DIM, device=dev)` (line 177) and into the
`DDPM_STOCHASTIC` branch's posterior-noise draws if that path is ever used.

Also **move the module-level `torch.manual_seed(SEED); np.random.seed(SEED)`
behind `if __name__ == "__main__":`** in both scripts. Right now merely
*importing* either module for inference has the side effect of resetting the
process-global RNG — a classic import-side-effect anti-pattern that silently
affects any other code in the same process that later draws from `torch`'s
default generator (not just this bug's symptom, a latent footgun generally).

```python
# ui.py — _cfm_sample_candidates, pass the trial's seed through
poses_norm = sample_poses(model, cond_t, n=n, steps=20, seed=self.seed)
                                                         # ^ RobotEnvUI.seed, already
                                                         #   derived from --seed CLI arg
```

This makes candidate generation a pure function of `(checkpoint, object, trial
seed)` — reproducible per-trial, but genuinely different across trials, matching
what a "for i in range(25): sample" comparison should actually measure.

## Fix 2: log the attempted-but-failed pose

**Root cause**: `env_soarm.py:pick_obj_by_id` (line 1990-2052) tries up to
`N_GRASP_ATTEMPTS` candidates in ranked order; on the **first** success it
returns `(True, obj_id, g)`. If **all** attempts fail, it returns
`(False, None, None)` (line 2052) — discarding every candidate it just tried.
`_log_ui_grasp_exec` (line 2159) then hardcodes `x=y=z=yaw=None` whenever
`grasp is None` (line 2161-2162). Net effect: a trial that fails all 5
attempts leaves **zero pose record**, even though 5 real candidate poses were
generated and physically attempted.

**Design**: return the top-ranked attempted candidate on total failure instead
of `None`, so the logger has something real to write.

```python
# env_soarm.py:pick_obj_by_id — change the final return only
for j, g in enumerate(grasps_to_try):
    ...
    success, grasped_id = self._execute_grasp((x, y, z), yaw, opening, obj_height)
    if success:
        ...
        return True, obj_id, g
    print("Grasping failed. Retrying...")

- return False, None, None
+ # Return the top-ranked attempted candidate (not None) so failed trials
+ # still get a pose logged — previously every failed trial's candidate was
+ # silently discarded here, leaving CFM(no-OT)-style 0/N methods with zero
+ # recoverable pose data.
+ return False, obj_id, (grasps_to_try[0] if grasps_to_try else None)
```

Line 2002-2004's early return (`stored_grasps` empty — no candidates existed
at all) stays `return False, None, None` unchanged — that case genuinely has
no pose to report.

No change needed in `_log_ui_grasp_exec` itself — it already branches
correctly on `grasp is dict / tuple / None`; it'll now receive a real dict on
the "all attempts failed" path and log real x/y/z/yaw with `success_grasp=False`.

**Scope check**: this only changes what's *returned on total failure* — the
grasp-attempt/execution logic itself (`_execute_grasp` calls, retry loop,
success detection) is untouched. Doesn't change any existing success-path
behavior or any existing benchmark result; purely additive logging fidelity
for future eval runs.
