"""
Explicit JOINT_POSITION (absolute) controller config for the Piper arm.

Background: Piper.default_controller_config returned the string
"default_spot" (copied from the ARX5 template), intended to load
controllers/config/robots/default_spot.json. That file does not exist --
robosuite's load_composite_controller_config() silently falls back to
controllers/config/default/composite/basic.json whenever the named file is
missing (confirmed by the "Loading controller configuration from:
.../composite/basic.json" line printed on every env construction this whole
session). basic.json's default is OSC_POSE, which is what actually diverged
during the earlier closed-loop reach attempt (piper_reach_grasp_demo.py) --
not necessarily a fundamental flaw in OSC for this arm, but JOINT_POSITION
with input_type="absolute" sidesteps operational-space dynamics entirely
(just per-joint PD to a target qpos), which is simpler to reason about and
pairs directly with the DLS IK solver's output (a target qpos vector) with
no unit conversion.

Pass this explicitly via controller_configs=PIPER_JOINT_POSITION_CONFIG when
constructing an env -- default_controller_config on the robot class is not
used by robosuite's file-lookup path and editing it further would silently
do nothing (as it did before this fix).

BUG FOUND AND FIXED 2026-07-15 -- the gripper's "type": "GRIP" sub-config had
no input_min/input_max, so robosuite's SimpleGripController used its own
defaults (-1, 1). PiperGripper.format_action (piper_gripper.py) does NOT
emit a normalized [-1,1] value like other robosuite grippers -- it tracks
and returns an ABSOLUTE joint position in real units (-0.05 to -0.004
metres, incrementally, clipped to the gripper's actual ctrlrange).
SimpleGripController.run_controller() then does its OWN separate real-units
mapping (`bias + weight * desired_qvel`, using actuator_min/max) on top of
that, silently DOUBLE-scaling an already-real-units value as if it still
needed converting from normalized space. Traced empirically: even at a
verified, fully-converged current_action=-0.05 ("fully open"), the actual
data.ctrl written to the finger actuators was -0.02815 -- nowhere near
either range endpoint, roughly the actuator range's own midpoint with a
small perturbation -- and "fully closed" (current_action=-0.004) computed
to -0.0271, a difference of only ~0.0001m between open and close! This
means the gripper's REAL commanded travel through its entire history in
this project has been on the order of a tenth of a millimetre, not the
0.076-0.10m documented -- every finger-width/opening investigation and fix
this session (see this file's other README entries) was operating on top
of this bug without knowing it.

FIRST FIX ATTEMPT (didn't work): set input_min/input_max on the gripper
sub-config to match format_action's real output range. Traced further and
found robosuite/robots/robot.py (~lines 958-969) REBUILDS
part_controller_config[gripper_name] from scratch when wiring up the
composite controller, copying ONLY "type" and "use_action_scaling" from
this file's "gripper" sub-dict -- input_min/input_max are silently
dropped, so that attempt had no effect (confirmed empirically: ctrl stayed
at -0.02815 after adding them).

ACTUAL FIX: "use_action_scaling": False. SimpleGripController.set_goal()
and .run_controller() both gate their entire rescaling logic behind
`if self.use_action_scaling:` -- with it off, the action passes straight
through to data.ctrl unchanged. Since PiperGripper.format_action already
computes the correct real-units target itself, this is exactly the
"don't touch it, the unit conversion is already done" behaviour needed --
and "use_action_scaling" IS one of the few keys robot.py actually copies
from the gripper sub-config (unlike input_min/input_max above).
"""

PIPER_JOINT_POSITION_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "JOINT_POSITION",
            "input_type": "absolute",
            "input_max": 3.14,
            "input_min": -3.14,
            "output_max": 3.14,
            "output_min": -3.14,
            "kp": 50,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "qpos_limits": None,
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {
                "type": "GRIP",
                # False (not the default True): PiperGripper.format_action
                # already returns an absolute real-units joint position
                # (see piper_gripper.py) -- SimpleGripController's own
                # bias+weight rescaling on top of that silently mapped
                # "fully open" and "fully closed" to within ~0.1mm of each
                # other. See this file's header comment for the full trace.
                "use_action_scaling": False,
            },
        }
    },
}
