import os, sys, argparse
import numpy as np
from copy import deepcopy

sys.path.append(os.path.dirname(__file__))  # ensure tango_robot import works

from tango_robot.ui import RobotEnvUI
from tango.utils.config import load_config

# ---- FIX 1: grasp sampler real API ----
from grasp_6dof.grasp_sampler import sample_grasps_from_mesh, pack_for_json


def cfg_set(cfg, path, value):
    """Set nested config by dot path. Works for dict / attr objects."""
    keys = path.split(".")
    obj = cfg
    for k in keys[:-1]:
        if isinstance(obj, dict):
            obj = obj.setdefault(k, {})
        else:
            if not hasattr(obj, k):
                setattr(obj, k, type("CfgNode", (), {})())
            obj = getattr(obj, k)
    last = keys[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def apply_stage_overrides(cfg, stage: int):
    """
    Stage1: 对标原项目（不开 6DoF sampler，不开 LGGSN，不开语义）
    Stage2: 只跑 6DoF grasp sampling + validation（不启动 UI）
    Stage3: 语义（prompt）+ 6DoF sampler（不开 LGGSN）
    Stage4: 语义 + 6DoF sampler + LGGSN ranker
    """
    if stage == 1:
        cfg_set(cfg, "policy.enable_grasp_sampling", False)
        cfg_set(cfg, "policy.use_grasp_ranker", False)
        # 如果你代码里有这个开关就会生效；没有也不会报错
        cfg_set(cfg, "policy.enable_semantic", False)

    elif stage == 2:
        cfg_set(cfg, "policy.enable_grasp_sampling", True)
        cfg_set(cfg, "policy.grasp_samples", getattr(cfg.policy, "grasp_samples", 100) or 100)
        cfg_set(cfg, "policy.use_grasp_ranker", False)
        cfg_set(cfg, "policy.enable_semantic", False)

    elif stage == 3:
        cfg_set(cfg, "policy.enable_semantic", True)
        cfg_set(cfg, "policy.enable_grasp_sampling", True)
        cfg_set(cfg, "policy.grasp_samples", getattr(cfg.policy, "grasp_samples", 100) or 100)
        cfg_set(cfg, "policy.use_grasp_ranker", False)

    elif stage == 4:
        cfg_set(cfg, "policy.enable_semantic", True)
        cfg_set(cfg, "policy.enable_grasp_sampling", True)
        cfg_set(cfg, "policy.grasp_samples", getattr(cfg.policy, "grasp_samples", 100) or 100)
        cfg_set(cfg, "policy.use_grasp_ranker", True)
        # 你自己的 ckpt 路径（没有也没关系，后面 ranker 自己会报“找不到ckpt”）
        if not hasattr(cfg.policy, "lggsn_ckpt"):
            cfg_set(cfg, "policy.lggsn_ckpt", "grasp_6dof/models/lggsn_pairwise_live_v5d.pt")

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return cfg


def sample_grasps(mesh_path, n_samples=100, table_z=0.0, seed=19,
                 out_json="grasp_6dof/dataset/sample_grasps.json"):
    """Compatibility wrapper so demo can call sample_grasps()."""
    grasps = sample_grasps_from_mesh(
        mesh_path=mesh_path,
        n_samples=int(n_samples),
        table_z=float(table_z),
        seed=int(seed) if seed is not None else None,
    )
    data = pack_for_json(grasps, topk=None)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    import json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote {len(grasps)} grasps -> {out_json}")
    return out_json


def main():
    # ---- Parse args (ALL add_argument BEFORE parse_args) ----
    parser = argparse.ArgumentParser(description="TANGO demo with staged pipeline.")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--config", type=str, default="./config/mujoco/env.yaml")
    parser.add_argument("--backend", type=str, default="mujoco",
                        choices=["pybullet", "mujoco"],
                        help="Simulation backend. Use 'mujoco' for SO-ARM101.")
    parser.add_argument("--robot_type", type=str, default=None, choices=["ur5", "panda"])
    parser.add_argument("--prompt", type=str, default="grasp the red cup")

    parser.add_argument("--n_objects", "--n-objects", type=int, default=None)
    parser.add_argument("--object", type=str, default=None,
                        help="Pin a specific YCB object in the scene (e.g. 'Banana'). "
                             "Case-insensitive. Overrides random shuffle.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gen-seed", type=int, default=None, dest="gen_seed",
                        help="Independent seed for the CFM/DDPM generator's sampling "
                             "noise (torch.Generator in sample_poses/sample_poses_ddpm). "
                             "Decoupled from --seed, which drives object spawn "
                             "orientation (and everything else previously tied to it: "
                             "object shuffling, width jitter, Random/GRC-6DoF candidate "
                             "draws). Defaults to being derived from --seed if omitted, "
                             "matching prior behavior.")
    parser.add_argument("--consensus-n", type=int, default=None, dest="consensus_n",
                        help="Phase 1 consensus-selection algorithm: draw this many "
                             "independent CFM/DDPM candidates (seeds --gen-seed, "
                             "--gen-seed+1, ..., --gen-seed+N-1), take their median "
                             "pose, execute only the candidate nearest to that median. "
                             "Unset or <=1 preserves the existing single-draw behavior.")
    parser.add_argument("--ikmargin-n", type=int, default=None, dest="ikmargin_n",
                        help="Phase 1 v2 IK-margin-selection algorithm: draw this many "
                             "independent CFM/DDPM candidates (seeds --gen-seed, "
                             "--gen-seed+1, ..., --gen-seed+N-1), solve headless IK for "
                             "each, execute only the candidate with smallest IK error "
                             "(pe_ik). Takes precedence over --consensus-n if both are "
                             "set. Unset or <=1 preserves the existing single-draw "
                             "behavior.")
    parser.add_argument("--vis", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=None)

    # stage2 grasp sampling args
    parser.add_argument("--mesh", type=str, default=None,
                        help="Stage2: mesh path for sample_grasps_from_mesh()")
    parser.add_argument("--grasps_out", type=str, default="grasp_6dof/dataset/sample_grasps.json")
    parser.add_argument("--once", action="store_true", help="Run one query then exit (Stage3/4 demo).")
    parser.add_argument("--grasp-mode", type=str, default=None,
                        choices=["physics", "demo_attach"],
                        help="Override grasp_mode from config. "
                             "Use 'demo_attach' for visual demos (kinematic attach), "
                             "'physics' for benchmark-accurate results (default from config).")
    parser.add_argument("--gate-delta", type=float, default=0.15, dest="gate_delta",
                        help="Uncertainty gate threshold: if max(scores)-min(scores) < gate_delta, "
                             "skip LGGSN reranking and use GR-ConvNet top-1. 0.0 = disabled.")
    parser.add_argument("--mc-gate-delta", type=float, default=0.0, dest="mc_gate_delta",
                        help="MC Dropout uncertainty gate: if mean(std) > delta, skip reranking. "
                             "0.0 = disabled. Overrides --gate-delta when non-zero.")
    parser.add_argument("--no-ranker", action="store_true", dest="no_ranker",
                        help="Disable LGGSN grasp ranking; use CFM candidates in generation order.")
    parser.add_argument("--no-semantic", action="store_true", dest="no_semantic",
                        help="Skip GPT-4o grounding; match target by name directly. "
                             "Use when API quota is exhausted or for offline benchmarking.")
    parser.add_argument("--cfm-ckpt", type=str, default="", dest="cfm_ckpt",
                        help="Path to OT-CFM checkpoint (.pt).  When set, replaces random "
                             "CoM-based candidate sampling with CFM-generated poses.  "
                             "Matching _stats.json must exist alongside the checkpoint.")
    parser.add_argument("--grconvnet-6dof", action="store_true", dest="grconvnet_6dof",
                        help="Use GR-ConvNet 2D predictions lifted to 6-DoF instead of "
                             "uniform random CoM sampling.  Benchmark baseline for comparing "
                             "GR-ConvNet spatial quality against OT-CFM.")

    args = parser.parse_args()

    os.environ["OWG_GATE_DELTA"]    = str(args.gate_delta)
    os.environ["OWG_MC_GATE_DELTA"] = str(args.mc_gate_delta)
    if args.cfm_ckpt:
        os.environ["CFM_CKPT"] = args.cfm_ckpt
    if args.no_ranker:
        os.environ["OWG_NO_RANKER"] = "1"
    if args.no_semantic:
        os.environ["OWG_NO_SEMANTIC"] = "1"
    if args.grconvnet_6dof:
        os.environ["OWG_GRC6DOF"] = "1"

    # ---- Load config FIRST ----
    cfg = load_config(args.config)

    # basic overrides
    if args.robot_type is not None:
        cfg_set(cfg, "robot_type", args.robot_type)
    if args.n_objects is not None:
        cfg_set(cfg, "n_objects", args.n_objects)
    if args.seed is not None:
        cfg_set(cfg, "seed", args.seed)
        # Seeds numpy's GLOBAL legacy RNG (np.random.*), which is distinct
        # from both torch's RNG (fixed by the sample_poses seed= param) and
        # Python's stdlib random module (already seeded via
        # YcbObjects.set_seed -> random.seed(seed)). Without this,
        # env_soarm.py:load_isolated_obj's `yaw = np.random.uniform(0, pi)`
        # spawn-orientation draw was uncontrolled by --seed — meaning
        # different methods run with the same --seed did not necessarily
        # face the same object spawn orientation, confounding any
        # cross-method comparison at fixed seed. Must run before
        # RobotEnvUI(...) is constructed (which spawns objects in __init__).
        np.random.seed(args.seed)
    if args.gen_seed is not None:
        # Independent override for generator sampling noise — see
        # ui.py:_cfm_sample_candidates for where GEN_SEED is read. Decoupled
        # from --seed's spawn-orientation control above.
        os.environ["GEN_SEED"] = str(args.gen_seed)
    if args.consensus_n is not None:
        os.environ["CONSENSUS_N"] = str(args.consensus_n)
    if args.ikmargin_n is not None:
        os.environ["IKMARGIN_N"] = str(args.ikmargin_n)
    if args.object is not None:
        cfg_set(cfg, "object", args.object)
    if args.vis is not None:
        cfg_set(cfg, "policy.vis", args.vis)
    if args.verbose is not None:
        cfg_set(cfg, "policy.verbose", args.verbose)

    cfg_set(cfg, "prompt", args.prompt)

    # ---- Apply stage overrides ----
    cfg = apply_stage_overrides(cfg, args.stage)

    # ---- Backend override ----
    cfg_set(cfg, "backend", args.backend)

    # ---- grasp_mode override (after all config reloads) ----
    if args.grasp_mode is not None:
        cfg_set(cfg, "grasp_mode", args.grasp_mode)
        print(f"[INFO] grasp_mode overridden → {args.grasp_mode}")

    # ---- MuJoCo stage-3 semantic demo: deterministic scene selection ----
    if args.backend == "mujoco" and args.stage == 3:
        _p = args.prompt.lower()
        _SCENES = {
            "banana":  ["Banana", "Pear", "MustardBottle"],
            "mustard": ["MustardBottle", "Pear", "Banana"],
        }
        for keyword, objects in _SCENES.items():
            if keyword in _p:
                cfg_set(cfg, "scene_objects", objects)
                print(f"[INFO] Semantic demo: prompt contains '{keyword}' "
                      f"→ scene_objects={objects}")
                break

    print("\n=== Loaded Configuration ===")
    print(cfg)
    print("============================\n")

    # ---- Stage2: generate grasps + validate only ----
    if args.stage == 2:
        # choose mesh
        mesh_path = args.mesh or getattr(cfg.policy, "grasp_obj_path", None)
        if not mesh_path:
            raise ValueError("Stage2 needs --mesh or cfg.policy.grasp_obj_path")
        n_samples = getattr(cfg.policy, "grasp_samples", 100)

        sample_file = sample_grasps(
            mesh_path,
            n_samples=n_samples,
            table_z=0.0,
            seed=getattr(cfg, "seed", 19),
            out_json=args.grasps_out,
        )

        # MuJoCo validation runs separately — use benchmark/replay_soarm_grasp.py:
        #   conda run -n owg-mujoco python benchmark/replay_soarm_grasp.py \
        #       --grasps <output_json>
        print("\n✅ Stage2 complete.")
        return

    # ---- Stage1/3/4: run UI ----
    demo = RobotEnvUI(cfg, backend=args.backend)

    # inject prompt to avoid Tkinter popup
    if not hasattr(demo, "user_input"):
        demo.user_input = cfg.prompt
    if hasattr(demo, "set_user_prompt"):
        demo.set_user_prompt(cfg.prompt)

    print(f'[INFO] Using prompt: "{cfg.prompt}"')
    demo.run(initial_query=cfg.prompt, once=args.once)


if __name__ == "__main__":
    main()

