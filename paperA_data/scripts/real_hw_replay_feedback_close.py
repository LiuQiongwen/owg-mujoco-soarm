"""
Phase 3 replay v2 (2026-07-10): replay only through the recorded trajectory's
pre-close settled pose, then let the REAL gripper's own load feedback drive
closure -- instead of blindly replaying the recorded closing sequence.

Why (see paperA_data/README.md, "Phase 3 real-robot pilot, round 2"):
GRASP_MODE_PHYSICS_WELD's recorded gripper channel closes all the way to
~0.001 (near GRIP_CLOSED) during transport regardless of true object width,
because sim kinematically welds the object to the EEF once bilateral contact
is detected at the moment of closing -- the true jaw width during transport
is decoupled from the object's actual size. Replaying that "closed to 0.001"
command sequence on real hardware, which has no weld fallback and must hold
the object by genuine friction, is not the same physical event. This script
instead:

  1. Replays joint_pos + gripper faithfully through the trajectory's
     PRE-CLOSE settled pose. The pre-close index is auto-detected as the end
     of the longest near-constant run in the gripper channel before its
     global minimum -- this is where `_settle_at_pose` finishes holding the
     object-appropriate pre-close width, right before `auto_close_gripper`
     starts moving it (see find_pre_close_idx docstring for why this is
     more robust than assuming a clean monotonic decrease).
  2. From there, closes the real gripper in small steps (default 3 mm),
     reading back Present_Load (STS3215 raw load register -- not exposed by
     SOARMRealBackend's public API, read directly via backend.bus) after
     each step. Stops once the load magnitude exceeds --load-threshold
     (contact resistance) or the achieved position stalls -- stops tracking
     the commanded target by more than --stall-tol for --stall-steps
     consecutive steps (backup signal in case load readings are noisy).
  3. Continues replaying joint_pos ONLY (not gripper) for the remaining
     lift/transport/place portion, leaving the gripper holding whatever
     position the feedback-driven close achieved.

IMPORTANT -- load threshold is NOT empirically calibrated yet. Present_Load's
raw units depend on Max_Torque_Limit and per-servo characteristics that
haven't been measured on this specific gripper motor. Run --calibrate-only
first (reads and prints Present_Load at rest, gripper fully open, no motion)
to sanity-check the noise floor before trusting --load-threshold on a live
close. Use --verbose to watch every step printed live and be ready to hit
e-stop; do not run this unattended on the first attempt.

IMPORTANT -- approach speed (round 3, 2026-07-10): a first live attempt at
--approach-speed 0.5 swept the arm through the real object's actual position
fast enough to physically knock it away before closing ever started (see
paperA_data/README.md, "Phase 3 real-robot pilot, round 3" -- no damage, but
a real collision risk given the real object isn't guaranteed to be exactly
where the trajectory assumes). --approach-speed now defaults to 0.15 (much
slower than --speed, which only applies to the post-close segment) so any
incidental proximity is gentle enough to watch and abort.

Usage:
    # 1. Baseline check (no motion beyond existing reset -- read-only load sample)
    conda run -n tango python paperA_data/scripts/real_hw_replay_feedback_close.py \\
        --traj trajs/pear_consensus_orient3_gen1.json --calibrate-only

    # 2. Live replay + feedback close
    conda run -n tango python paperA_data/scripts/real_hw_replay_feedback_close.py \\
        --traj trajs/pear_consensus_orient3_gen1.json --approach-speed 0.15 --speed 0.5 \\
        --load-threshold 120 --step-m 0.003 --verbose
"""
import argparse
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, "/lena/projects/OWG-main/paperA_data/scripts")
from real_hw_connect import connect_backend, PROJECT_ROOT

GRIPPER_MOTOR = "gripper"
LOAD_MAG_MASK = 0x3FF   # bits 0-9: magnitude, 0-1023 (bit 10 is direction, unused here)


def decode_load_magnitude(raw: int) -> int:
    """Present_Load isn't in lerobot's sign-decoding table (see feetech/tables.py
    STS_SMS_SERIES_ENCODINGS_TABLE), so `.read(..., normalize=False)` returns
    the raw unsigned register value. Magnitude is the low 10 bits regardless
    of which bit encodes direction -- sufficient for a threshold-on-magnitude
    stop condition without needing the sign."""
    return int(raw) & LOAD_MAG_MASK


def find_pre_close_idx(points, step_tol: float = 0.0001) -> int:
    """Return the index of the last point BEFORE the trajectory's final
    gripper-closing motion begins.

    `_settle_at_pose` pre-sets the gripper to the object-appropriate
    "opening" width early (before any arm motion), then HOLDS it flat
    through the hover/descend/settle physics -- this flat hold is the
    longest near-constant run in the whole gripper channel (an earlier,
    shorter flat run also exists while the gripper sits fully open before
    that initial pre-set close). `auto_close_gripper` only runs after
    `_settle_at_pose` returns, so the true pre-close pose is the END of
    that long hold, right before the final closing motion (which may
    include a brief widen-then-close excursion, not necessarily a clean
    monotonic decrease -- see paperA_data/README.md's round-2 pilot entry).

    Heuristic: find the global-minimum gripper index, then find the
    longest run of near-constant (step-to-step change <= step_tol) points
    that ends before it -- that run's last index is the pre-close pose.
    """
    grip = np.array([p.gripper for p in points], dtype=np.float64)
    min_idx = int(np.argmin(grip))

    diffs = np.abs(np.diff(grip[:min_idx + 1]))  # diffs[i] = |grip[i+1]-grip[i]|
    flat = diffs <= step_tol

    best_len, best_end = 0, min_idx
    run_len, run_end = 0, -1
    for i, is_flat in enumerate(flat):
        if is_flat:
            run_len += 1
            run_end = i + 1  # index of the later point in this flat step
        else:
            if run_len > best_len:
                best_len, best_end = run_len, run_end
            run_len, run_end = 0, -1
    if run_len > best_len:
        best_len, best_end = run_len, run_end

    return max(0, best_end)


def feedback_close(backend, start_opening: float, load_threshold: int,
                    step_m: float, stall_tol: float, stall_steps: int,
                    min_opening: float, settle_s: float, verbose: bool,
                    early_warn_threshold: Optional[int] = None,
                    fine_step_m: Optional[float] = None) -> float:
    """Incrementally close the real gripper, stopping on load or stall.

    early_warn_threshold / fine_step_m (2026-07-10, round 6): parallel-jaw
    grasping requires both jaws to contact the object at close to the same
    time -- literature on this failure mode (force-closure grasping without
    wrist compliance or independently-actuated jaws) points to the object
    being shoved out of the closing path when contact is uneven or the
    closing step is too coarse right as contact begins. Once load first
    crosses early_warn_threshold (well above the empty-jaws noise ceiling
    ~68, well below load_threshold's contact-stop level), switch to the
    smaller fine_step_m for the rest of the close -- a "soft landing"
    through the actual contact-forming steps instead of continuing at the
    coarser approach step size. Disabled (step_m used throughout) if either
    argument is None.

    Returns the final achieved opening (metres).
    """
    opening = float(start_opening)
    stall_count = 0
    cur_step = step_m
    fine_mode = False

    while opening > min_opening:
        opening = max(min_opening, opening - cur_step)
        backend.set_gripper(opening, blocking=False)
        time.sleep(settle_s)

        actual = backend.get_gripper_opening()
        raw_load = backend.bus.read("Present_Load", GRIPPER_MOTOR, normalize=False)
        load_mag = decode_load_magnitude(raw_load)

        if verbose:
            print(f"  [feedback_close] target={opening:.4f}m actual={actual:.4f}m "
                  f"load_raw={raw_load} load_mag={load_mag} step={cur_step:.4f}m")

        if load_mag >= load_threshold:
            print(f"  [feedback_close] STOP: load_mag={load_mag} >= threshold={load_threshold} "
                  f"at opening={actual:.4f}m")
            return actual

        if (not fine_mode and early_warn_threshold is not None and fine_step_m is not None
                and load_mag >= early_warn_threshold):
            fine_mode = True
            cur_step = fine_step_m
            print(f"  [feedback_close] early contact warning (load_mag={load_mag} >= "
                  f"{early_warn_threshold}) -- switching to fine step={fine_step_m:.4f}m")

        if abs(actual - opening) > stall_tol:
            stall_count += 1
            if stall_count >= stall_steps:
                print(f"  [feedback_close] STOP: stalled (actual={actual:.4f}m vs "
                      f"target={opening:.4f}m) for {stall_count} steps")
                return actual
        else:
            stall_count = 0

    print(f"  [feedback_close] reached min_opening={min_opening:.4f}m without "
          f"triggering load/stall stop -- fully closed, likely missed the object")
    return backend.get_gripper_opening()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--approach-speed", type=float, default=0.15,
                     help="Playback speed for the pre-close (approach/hover/descend) segment. "
                          "Kept slow by default -- see module docstring, round 3 finding: fast "
                          "approach replay can physically collide with a not-quite-correctly-"
                          "placed real object.")
    ap.add_argument("--speed", type=float, default=0.5,
                     help="Playback speed for the post-close (lift/transport/place) segment.")
    ap.add_argument("--max-relative-target", type=float, default=30.0)
    ap.add_argument("--load-threshold", type=int, default=150,
                     help="Present_Load raw magnitude (0-1023) to stop closing. NOT calibrated -- "
                          "run --calibrate-only first.")
    ap.add_argument("--step-m", type=float, default=0.003, help="Gripper closing step, metres.")
    ap.add_argument("--early-warn-threshold", type=int, default=80,
                     help="Present_Load magnitude that triggers switching to --fine-step-m for the "
                          "rest of the close (soft landing through the actual contact-forming "
                          "steps -- round 6 fix for the gripper shoving objects away on contact). "
                          "Set above the empty-jaws noise ceiling (~68) and below --load-threshold. "
                          "Pass 0 or a negative value to disable.")
    ap.add_argument("--fine-step-m", type=float, default=0.001,
                     help="Closing step used once --early-warn-threshold is crossed.")
    ap.add_argument("--settle-s", type=float, default=0.15, help="Wait after each closing step.")
    ap.add_argument("--stall-tol", type=float, default=0.004, help="Position tracking tolerance, metres.")
    ap.add_argument("--stall-steps", type=int, default=3)
    ap.add_argument("--min-opening", type=float, default=0.0)
    ap.add_argument("--calibrate-only", action="store_true",
                     help="Connect, read Present_Load at rest (no trajectory replay), print, exit.")
    ap.add_argument("--goto-pre-close-only", action="store_true",
                     help="Replay only the approach segment, then exit (torque disabled) without "
                          "closing. Lets the object be placed exactly where the gripper ends up, "
                          "instead of guessing a physical position ahead of time from simulation "
                          "xy coordinates -- run this first, place the object, then run again "
                          "WITHOUT this flag (the redundant re-approach is harmless, the arm is "
                          "already near the start of that path). See paperA_data/README.md, "
                          "Phase 3 round 5. Not combinable with --calibrate-only.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    backend = connect_backend(max_relative_target=args.max_relative_target)

    if args.calibrate_only:
        try:
            raw = backend.bus.read("Present_Load", GRIPPER_MOTOR, normalize=False)
            print(f"[calibrate] Present_Load raw={raw}  magnitude={decode_load_magnitude(raw)} "
                  f"(gripper at rest, no object) -- sample a few times below")
            for _ in range(9):
                time.sleep(0.3)
                raw = backend.bus.read("Present_Load", GRIPPER_MOTOR, normalize=False)
                print(f"  raw={raw}  magnitude={decode_load_magnitude(raw)}")
        finally:
            backend.close()
            print("[calibrate] disconnected cleanly (torque disabled).")
        return

    sys.path.insert(0, PROJECT_ROOT)
    from robots.trajectory import Trajectory

    traj = Trajectory.load(args.traj)
    print(f"[replay-fc] loaded {args.traj}: {traj}")

    pre_close_idx = find_pre_close_idx(traj.points)
    print(f"[replay-fc] auto-detected pre-close index: {pre_close_idx} / {traj.n_points} "
          f"(gripper={traj.points[pre_close_idx].gripper:.4f}m there, "
          f"trajectory min gripper={min(p.gripper for p in traj.points):.4f}m)")

    try:
        print("[replay-fc] connected. Starting replay in 3 seconds -- watch the arm...")
        time.sleep(3)

        backend.reset()
        t_wall_start = time.time()
        t_traj_origin = traj.points[0].t

        # Segment 1: replay joint_pos + gripper through the pre-close pose.
        # Deliberately slow (--approach-speed, default 0.15) -- see module
        # docstring, round 3 finding: a fast approach can physically collide
        # with a not-quite-correctly-placed real object.
        for idx in range(0, pre_close_idx + 1):
            pt = traj.points[idx]
            t_target = t_wall_start + (pt.t - t_traj_origin) / args.approach_speed
            wait = t_target - time.time()
            if wait > 0:
                time.sleep(wait)
            backend.move_joints(pt.joint_pos, blocking=False)
            backend.set_gripper(pt.gripper, blocking=False)
            if args.verbose and idx % 200 == 0:
                print(f"[replay-fc] pre-close step {idx}/{pre_close_idx}")

        print(f"[replay-fc] reached pre-close pose (idx={pre_close_idx}). "
              f"Settling before feedback close...")
        time.sleep(0.5)

        if args.goto_pre_close_only:
            print("[replay-fc] --goto-pre-close-only: stopping here, gripper stays open. "
                  "Place the object exactly where the gripper currently is, then re-run this "
                  "script WITHOUT --goto-pre-close-only to replay the (short, harmless) "
                  "re-approach and continue into the feedback close.")
            return

        early_warn = args.early_warn_threshold if args.early_warn_threshold > 0 else None
        start_opening = backend.get_gripper_opening()
        print(f"[replay-fc] starting feedback close from opening={start_opening:.4f}m "
              f"(load_threshold={args.load_threshold}, early_warn_threshold={early_warn}, "
              f"fine_step_m={args.fine_step_m})")
        final_opening = feedback_close(
            backend, start_opening, args.load_threshold, args.step_m,
            args.stall_tol, args.stall_steps, args.min_opening, args.settle_s, args.verbose,
            early_warn_threshold=early_warn, fine_step_m=args.fine_step_m,
        )
        print(f"[replay-fc] feedback close done, final opening={final_opening:.4f}m")

        # Segment 2: replay joint_pos ONLY for the rest (lift/transport/place).
        # Re-anchor timing to now so we don't try to "catch up" the time spent closing.
        t_wall_start2 = time.time()
        t_traj_origin2 = traj.points[pre_close_idx].t
        for idx in range(pre_close_idx + 1, traj.n_points):
            pt = traj.points[idx]
            t_target = t_wall_start2 + (pt.t - t_traj_origin2) / args.speed
            wait = t_target - time.time()
            if wait > 0:
                time.sleep(wait)
            backend.move_joints(pt.joint_pos, blocking=False)
            # Deliberately NOT calling backend.set_gripper here -- see module
            # docstring: the recorded gripper channel post-close is a sim-only
            # weld artifact, not a valid real-hardware command.
            if args.verbose and idx % 200 == 0:
                print(f"[replay-fc] post-close step {idx}/{traj.n_points}")

        print("[replay-fc] done.")
    finally:
        backend.close()
        print("[replay-fc] disconnected cleanly (torque disabled).")


if __name__ == "__main__":
    main()
