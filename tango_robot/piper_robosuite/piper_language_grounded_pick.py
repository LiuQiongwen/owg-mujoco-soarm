"""
Language-grounded pick for Piper: camera -> VLM semantic grounding (reusing
tango/policy.py's OwgPolicy, decoupled from LGGSN) -> Piper's existing
classical grasp execution (run_pick_and_place, unmodified).

STATUS (2026-08-02): full pipeline VERIFIED WORKING through segmentation-to-
object-name resolution. The segmentation mapping bug from the first pass
(two inconsistent cross-referencing strategies, both involving a second,
separate `env.sim.render(segmentation=True)` call) is FIXED -- root cause
was never a rendering-order artifact, it was using the wrong method
entirely. Read RoboSuite's own segmentation-sensor source
(`robot_env.py::_create_segmentation_sensor`) directly and found the exact,
deterministic, non-rendering-dependent formula it uses internally:
`name2id = {inst: i for i, inst in enumerate(model.instances_to_ids.keys())}`,
then each pixel's compact segmentation value is `name2id[instance] + 1`
(0 reserved for unmapped/background). This needs zero additional rendering
and cannot drift between calls -- confirmed empirically: pear/can/mustard
resolved to seg_ids 1/2/3 with clean, non-overlapping pixel counts
(519/495/166 px), vs. the noisy, inconsistent counts the old two-render
cross-referencing approach produced.

Verified facts this file depends on (2026-08-02 interactive session):
  - PiperMultiObjectScene(..., use_camera_obs=True, camera_names='tablecam',
    camera_segmentations='instance', camera_heights=256, camera_widths=256)
    returns obs['tablecam_image'] (256,256,3) and
    obs['tablecam_segmentation_instance'] (256,256,1). Confirmed by direct
    env.reset() call.
  - env.model.instances_to_ids maps instance name (e.g. 'pear', 'can',
    'mustard', but also robot/mount/gripper/tray names) -> {'geom': [...],
    'site': [...]}. Confirmed by direct inspection.
  - tango.policy.OwgPolicy(config_path, use_grasp_ranker=False) does NOT
    require LGGSN -- its __init__ already has a graceful fallback path
    ("Stage 4 falling back to Stage 3 behavior") when the ranker is
    disabled or fails to load. predict()'s obs/all_grasps/obj_names/
    env_obj_ids arguments are generic (image, segmentation, camera params,
    grasp-pose list, name mapping) -- nothing SO-ARM101-specific in the
    signature or the code path actually exercised when use_grasp_ranker
    is False. Confirmed by direct source reading, not assumed from the
    module's general framing.

Usage (once the segmentation mapping is fixed):
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_language_grounded_pick \\
      --instruction "pick up the mustard bottle"
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))

from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa: registers Piper/PiperGripper
from tango_robot.piper_robosuite.piper_pick_and_place import run_pick_and_place

CAMERA_NAME = "tablecam"
IMG_SIZE = 256
# Instance names that are NOT pickable objects -- exclude from the VLM's
# candidate set. Taken directly from env.model.instances_to_ids's own keys
# (verified interactively, 2026-08-02), not guessed.
NON_OBJECT_INSTANCES = {"Piper0", "RethinkMount0", "PiperGripper0_right", "placement_tray"}


def build_scene(ycb_objects):
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=ycb_objects,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERA_NAME, camera_segmentations="instance",
        camera_heights=IMG_SIZE, camera_widths=IMG_SIZE, control_freq=20,
    )
    obs = env.reset()
    return env, obs


def _map_segmentation_to_object(env, obs):
    """Deterministic, verified 2026-08-02: replicates RoboSuite's own
    internal instance-segmentation formula (robot_env.py's
    _create_segmentation_sensor) directly from env.model.instances_to_ids's
    dict order -- no second render call, no rendering-state dependency at
    all. seg pixel value for instance name `n` is
    list(instances_to_ids.keys()).index(n) + 1 (0 = unmapped/background).
    Confirmed empirically: pear/can/mustard resolve to clean, non-
    overlapping seg_ids with no cross-contamination, unlike an earlier,
    now-removed attempt that cross-referenced a second env.sim.render(...)
    call and produced noisy, inconsistent results."""
    seg = obs[f"{CAMERA_NAME}_segmentation_instance"][:, :, 0]
    name2id = {inst: i for i, inst in enumerate(env.model.instances_to_ids.keys())}

    id_to_name = {}
    for name, idx in name2id.items():
        if name in NON_OBJECT_INSTANCES:
            continue
        seg_id = idx + 1
        pixel_count = int(np.sum(seg == seg_id))
        if pixel_count > 0:
            id_to_name[seg_id] = name
    return id_to_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="pick up the mustard bottle",
                     help="Natural-language pick instruction.")
    ap.add_argument("--objects", default="pear,can,mustard")
    ap.add_argument("--config-path", default="./config/pyb/TANGO.yaml",
                     help="Same VLM-grounding config used by the SO-ARM101 pipeline (env.yaml's "
                          "policy.config_path) -- reused as-is, no Piper-specific config exists yet.")
    ap.add_argument("--use-real-gpt", action="store_true",
                     help="Without this flag, uses OWG_NO_SEMANTIC=1 (string-match "
                          "fast path, no API call) for wiring tests. Pass this flag "
                          "once the segmentation mapping is fixed and verified, to "
                          "exercise the real GPT grounding call.")
    args = ap.parse_args()

    ycb_objects = args.objects.split(",")
    env, obs = build_scene(ycb_objects)

    id_to_name = _map_segmentation_to_object(env, obs)
    print(f"[segmentation] resolved id_to_name = {id_to_name}  "
          f"(EMPTY or missing expected objects means the mapping bug above is still live)")
    if not id_to_name:
        print("[ABORT] segmentation->name mapping produced nothing usable. "
              "Fix _map_segmentation_to_object before proceeding -- do not fall back "
              "to guessing an object name here.")
        return

    if not args.use_real_gpt:
        os.environ["OWG_NO_SEMANTIC"] = "1"
        print("[info] OWG_NO_SEMANTIC=1 (fast-path string match, no real GPT call). "
              "Pass --use-real-gpt once the segmentation mapping above is verified correct.")

    from tango.policy import OwgPolicy
    policy = OwgPolicy(config_path=args.config_path, verbose=True, use_grasp_ranker=False)

    obs_for_policy = {
        "image": obs[f"{CAMERA_NAME}_image"],
        "seg": obs[f"{CAMERA_NAME}_segmentation_instance"][:, :, 0],
    }
    env_obj_ids = list(id_to_name.keys())
    obj_names = [id_to_name[i] for i in env_obj_ids]

    action = policy.predict(
        obs=obs_for_policy, user_input=args.instruction, all_grasps={},
        obj_names=obj_names, env_obj_ids=env_obj_ids,
    )
    print(f"[grounding] action = {action}")

    if action.get("action") != "pick" or action.get("target_id") not in id_to_name:
        print(f"[ABORT] grounding did not resolve to a known object. action={action}")
        return

    resolved_obj_name = id_to_name[action["target_id"]]
    print(f"[grounding] resolved object: {resolved_obj_name!r} -- "
          f"handing off to run_pick_and_place (unmodified Piper execution pipeline)")

    result = run_pick_and_place(env, resolved_obj_name, use_oriented_grasp=True, verbose=True)
    print(f"[execution] result = {result}")
    return result


if __name__ == "__main__":
    main()
