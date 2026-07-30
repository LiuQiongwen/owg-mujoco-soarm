"""Audit trial 1007's anomalous "success" at drift=28.67cm under
cr_cfm_descend -- re-run with the SAME seed/config, but with full
trajectory recording + verbose phase logging + explicit contact-force
tracing through descend/close, to determine whether the object was
actually gripped and placed, or the tray-proximity success check fired on
a collision fluke (object swept/knocked into the tray zone without ever
being held).
"""
import numpy as np
import torch

from tango_robot.piper_robosuite.cr_cfm.data import ACTION_DIM, HORIZON, DescendDataset
from tango_robot.piper_robosuite.cr_cfm.model import CRFlowNet
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite.piper_pick_and_place import (
    _has_object_contact, _object_contact_force_magnitude, _object_contact_geoms, run_pick_and_place,
)
from tango_robot.piper_robosuite.piper_trajectory import PiperTrajectoryRecorder

TRIAL_ID = 1007
CKPT = "/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/cr_cfm_cracker.pt"


class ContactTracingHook:
    """step_hook that both feeds a PiperTrajectoryRecorder (so we get the
    full joint/eef trajectory for later replay/inspection) AND separately
    logs per-step contact state + force magnitude against the target
    object, tagged by whatever phase solve_and_move most recently set --
    this is the direct evidence for "was this a grip or a collision"."""

    def __init__(self, env, obj_name):
        self.recorder = PiperTrajectoryRecorder()
        self.recorder.begin(metadata={"obj_name": obj_name, "trial_id": TRIAL_ID, "audit": True})
        self.obj_geom_ids = _object_contact_geoms(env, obj_name)
        self.contact_log = []  # (phase, step, contact: bool, force: float, obj_z)
        self._step = 0

    def set_phase(self, name):
        self.recorder.set_phase(name)

    def __call__(self, env):
        self.recorder.snap(env)
        contact = _has_object_contact(env, self.obj_geom_ids)
        force = _object_contact_force_magnitude(env, self.obj_geom_ids)
        obj_pos = env.get_object_positions()[list(env.get_object_positions().keys())[0]]
        self.contact_log.append((self.recorder._phase, self._step, contact, force, float(obj_pos[2])))
        self._step += 1


def main():
    dataset = DescendDataset.load(obj_name="cracker", horizon=HORIZON)
    mean_start = dataset.mean_start()
    template = dataset.mean_template()
    model = CRFlowNet(action_dim=ACTION_DIM, horizon=HORIZON)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    np.random.seed(TRIAL_ID)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=["cracker"],
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    hook = ContactTracingHook(env, "cracker")

    result = run_pick_and_place(
        env, "cracker", use_oriented_grasp=True, verbose=True,
        cr_cfm_descend=True, cr_cfm_model=model, cr_cfm_mean_start=mean_start,
        cr_cfm_template=template, cr_cfm_horizon=HORIZON, cr_cfm_num_steps=6, cr_cfm_device="cpu",
        step_hook=hook,
    )

    print("\n=== RESULT ===")
    print("success:", result["success"], "dist_to_tray:", result["dist_to_tray"])
    print("pre_close_drift_cm:", result["phases"].get("pre_close_drift_cm"))

    print("\n=== CONTACT TRACE (descend/descend_refresh/close only) ===")
    relevant_phases = {"descend", "descend_refresh", ""}
    close_seen = False
    for phase, step, contact, force, obj_z in hook.contact_log:
        if phase in relevant_phases or (phase == "" and not close_seen):
            tag = "CONTACT" if contact else "       "
            if force > 5.0:
                tag += f"  FORCE={force:.1f}N"
            print(f"  step={step:4d} phase={phase or '(post-descend_refresh/close)':22s} {tag}  obj_z={obj_z:.4f}")
        if phase == "" :
            close_seen = True

    # Summary stats: was there ever SUSTAINED, moderate (gripping-plausible)
    # contact force during/after close, vs. a single large transient spike
    # (collision) followed by no sustained hold?
    forces = [f for _, _, c, f, _ in hook.contact_log if c]
    print(f"\ncontact force stats while touching: n={len(forces)}, "
          f"max={max(forces) if forces else 0:.1f}N, mean={np.mean(forces) if forces else 0:.1f}N")

    traj = hook.recorder.end(success=result["success"], dist_to_tray=result["dist_to_tray"])
    out = "/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/audit_1007_traj.json"
    traj.save(out)
    print(f"\nfull trajectory saved -> {out}")


if __name__ == "__main__":
    main()
