import os, sys, argparse
from copy import deepcopy

sys.path.append(os.path.dirname(__file__))  # ensure owg_robot import works

from owg_robot.ui import RobotEnvUI
from owg.utils.config import load_config

# ---- FIX 1: grasp sampler real API ----
from grasp_6dof.grasp_sampler import sample_grasps_from_mesh, pack_for_json
from grasp_6dof.grasp_validator_panda import validate_grasps


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
            cfg_set(cfg, "policy.lggsn_ckpt", "grasp_6dof/models/lggsn_pairwise_live.pt")

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
    parser = argparse.ArgumentParser(description="OWG demo with staged pipeline.")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--config", type=str, default="./config/pyb/env.yaml")
    parser.add_argument("--backend", type=str, default="pybullet",
                        choices=["pybullet", "mujoco"],
                        help="Simulation backend. Use 'mujoco' for SO-ARM101.")
    parser.add_argument("--robot_type", type=str, default=None, choices=["ur5", "panda"])
    parser.add_argument("--prompt", type=str, default="grasp the red cup")

    parser.add_argument("--n_objects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vis", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=None)

    # stage2 grasp sampling args
    parser.add_argument("--mesh", type=str, default=None,
                        help="Stage2: mesh path for sample_grasps_from_mesh()")
    parser.add_argument("--grasps_out", type=str, default="grasp_6dof/dataset/sample_grasps.json")
    parser.add_argument("--once", action="store_true", help="Run one query then exit (Stage3/4 demo).")

    args = parser.parse_args()

    # ---- Load config FIRST ----
    cfg = load_config(args.config)

    # basic overrides
    if args.robot_type is not None:
        cfg_set(cfg, "robot_type", args.robot_type)
    if args.n_objects is not None:
        cfg_set(cfg, "n_objects", args.n_objects)
    if args.seed is not None:
        cfg_set(cfg, "seed", args.seed)
    if args.vis is not None:
        cfg_set(cfg, "policy.vis", args.vis)
    if args.verbose is not None:
        cfg_set(cfg, "policy.verbose", args.verbose)

    cfg_set(cfg, "prompt", args.prompt)

    # ---- Apply stage overrides ----
    cfg = apply_stage_overrides(cfg, args.stage)

    # ---- Backend override ----
    cfg_set(cfg, "backend", args.backend)
    if args.backend == "mujoco" and args.config == "./config/pyb/env.yaml":
        # auto-select MuJoCo config if user didn't specify one
        mj_cfg = "./config/mujoco/env.yaml"
        if os.path.exists(mj_cfg):
            cfg = load_config(mj_cfg)
            cfg = apply_stage_overrides(cfg, args.stage)
            cfg_set(cfg, "backend", "mujoco")
            cfg_set(cfg, "prompt", args.prompt)
            if args.n_objects is not None:
                cfg_set(cfg, "n_objects", args.n_objects)
            if args.seed is not None:
                cfg_set(cfg, "seed", args.seed)
            print(f"[INFO] Auto-loaded MuJoCo config: {mj_cfg}")

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

        # validate (uses panda routine)
        validate_grasps(cfg, sample_file)
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

