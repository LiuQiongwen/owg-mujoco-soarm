"""Scene generator for the diverse grasp benchmark.

Generates randomized scenes by difficulty level:
  easy   — centered spawn, no yaw variation, no clutter
  medium — random XY spawn, random object yaw, no clutter
  hard   — wide XY range, full yaw, 1-3 clutter objects, edge placement

SceneConfig is pure data — no MuJoCo interactions.  DiverseBenchmarkRunner
reads a SceneConfig and applies it during trial setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from owg_robot.env_soarm import TABLE_TOP_Z


# ── difficulty presets ────────────────────────────────────────────────────────

@dataclass
class DifficultyConfig:
    name: str
    spawn_x_range:   Tuple[float, float]   # (min, max) — target object x-spawn
    spawn_y_range:   Tuple[float, float]   # (min, max) — target object y-spawn
    spawn_yaw_range: Tuple[float, float]   # (min, max) radians — initial yaw
    n_clutter_min:   int   = 0
    n_clutter_max:   int   = 0
    edge_prob:       float = 0.0           # prob of pushing target to workspace edge


EASY = DifficultyConfig(
    name            = "easy",
    spawn_x_range   = (-0.04, 0.04),
    spawn_y_range   = (-0.44, -0.36),
    spawn_yaw_range = (0.0, 0.0),
    n_clutter_min   = 0, n_clutter_max = 0,
    edge_prob       = 0.0,
)

MEDIUM = DifficultyConfig(
    name            = "medium",
    spawn_x_range   = (-0.10, 0.10),
    spawn_y_range   = (-0.50, -0.30),
    spawn_yaw_range = (-math.pi / 2, math.pi / 2),
    n_clutter_min   = 0, n_clutter_max = 0,
    edge_prob       = 0.0,
)

HARD = DifficultyConfig(
    name            = "hard",
    spawn_x_range   = (-0.15, 0.15),
    spawn_y_range   = (-0.55, -0.25),
    spawn_yaw_range = (-math.pi, math.pi),
    n_clutter_min   = 1, n_clutter_max = 3,
    edge_prob       = 0.25,
)

DIFFICULTY_PRESETS: Dict[str, DifficultyConfig] = {
    "easy":   EASY,
    "medium": MEDIUM,
    "hard":   HARD,
}


# ── scene config ──────────────────────────────────────────────────────────────

@dataclass
class SceneConfig:
    """Complete scene description for one benchmark trial (pure data, no env)."""
    difficulty:  str
    obj_name:    str    # benchmark short name  (e.g. "banana")
    ycb_name:    str    # YCB pool name         (e.g. "YcbBanana")
    seed:        int

    spawn_x:     float  # m
    spawn_y:     float  # m
    spawn_z:     float  # m  (drop height, always TABLE_TOP_Z + offset)
    spawn_yaw:   float  # rad — initial object rotation around world Z

    clutter_ycb_names: List[str]          = field(default_factory=list)
    clutter_positions:  List[List[float]] = field(default_factory=list)  # [[x,y,z], ...]


# ── internal constants ────────────────────────────────────────────────────────

_CLUTTER_POOL = [
    "YcbBanana", "YcbPear", "YcbMustardBottle",
    "YcbTomatoSoupCan", "YcbCrackerBox", "YcbPowerDrill",
]

_DROP_Z_OFFSET   = 0.15   # m above TABLE_TOP_Z
_CLUTTER_SPREAD  = 0.12   # m — half-range for clutter offset from target
_MIN_OBJ_SEP     = 0.10   # m — minimum centre-to-centre distance between objects
_EDGE_INNER_R    = 0.30   # m — annulus inner radius for edge placement
_EDGE_OUTER_R    = 0.40   # m — annulus outer radius for edge placement
_WS_X_LIMIT      = 0.38   # m — workspace ±X limit
_WS_Y_MIN, _WS_Y_MAX = -0.58, -0.18   # workspace Y limits


# ── scene generation ──────────────────────────────────────────────────────────

def generate_scene(
    difficulty:    DifficultyConfig,
    obj_name:      str,
    ycb_name:      str,
    seed:          int,
    all_ycb_names: Optional[List[str]] = None,
) -> SceneConfig:
    """Return a deterministic SceneConfig from a difficulty preset and seed.

    Parameters
    ----------
    difficulty    : DifficultyConfig  — easy / medium / hard preset
    obj_name      : str               — benchmark short name (e.g. "banana")
    ycb_name      : str               — YCB pool name (e.g. "YcbBanana")
    seed          : int               — RNG seed
    all_ycb_names : list | None       — pool of YCB names to draw clutter from;
                                        defaults to _CLUTTER_POOL if None
    """
    rng = np.random.default_rng(seed)

    # ── target object position ────────────────────────────────────────────────
    x_lo, x_hi = difficulty.spawn_x_range
    y_lo, y_hi = difficulty.spawn_y_range

    if difficulty.edge_prob > 0 and rng.random() < difficulty.edge_prob:
        # place near workspace boundary — annulus sampling
        r  = rng.uniform(_EDGE_INNER_R, _EDGE_OUTER_R)
        th = rng.uniform(-math.pi, math.pi)
        cx = float(np.clip(r * math.cos(th), x_lo, x_hi))
        cy = float(np.clip(-0.40 + r * math.sin(th), y_lo, y_hi))
    else:
        cx = float(rng.uniform(x_lo, x_hi))
        cy = float(rng.uniform(y_lo, y_hi))

    cz = TABLE_TOP_Z + _DROP_Z_OFFSET

    yaw_lo, yaw_hi = difficulty.spawn_yaw_range
    yaw = float(rng.uniform(yaw_lo, yaw_hi)) if yaw_lo != yaw_hi else 0.0

    # ── clutter objects ───────────────────────────────────────────────────────
    n_clutter = int(rng.integers(difficulty.n_clutter_min, difficulty.n_clutter_max + 1))
    pool = [n for n in (all_ycb_names if all_ycb_names is not None else _CLUTTER_POOL)
            if n != ycb_name]

    clutter_ycb: List[str]          = []
    clutter_pos: List[List[float]]  = []
    placed_xy: List[Tuple[float, float]] = [(cx, cy)]  # include target for distance check

    for _ in range(n_clutter):
        if not pool:
            break
        cname = str(rng.choice(pool))
        placed = False
        for _attempt in range(40):
            dx = float(rng.uniform(-_CLUTTER_SPREAD, _CLUTTER_SPREAD))
            dy = float(rng.uniform(-_CLUTTER_SPREAD, _CLUTTER_SPREAD))
            px, py = cx + dx, cy + dy
            # workspace bounds
            if abs(px) > _WS_X_LIMIT or not (_WS_Y_MIN < py < _WS_Y_MAX):
                continue
            # separation from all placed objects
            if any(math.sqrt((px - ox)**2 + (py - oy)**2) < _MIN_OBJ_SEP
                   for ox, oy in placed_xy):
                continue
            clutter_ycb.append(cname)
            clutter_pos.append([px, py, TABLE_TOP_Z + _DROP_Z_OFFSET])
            placed_xy.append((px, py))
            placed = True
            break
        if not placed:
            break  # couldn't fit another object — stop early

    return SceneConfig(
        difficulty         = difficulty.name,
        obj_name           = obj_name,
        ycb_name           = ycb_name,
        seed               = seed,
        spawn_x            = cx,
        spawn_y            = cy,
        spawn_z            = cz,
        spawn_yaw          = yaw,
        clutter_ycb_names  = clutter_ycb,
        clutter_positions  = clutter_pos,
    )
