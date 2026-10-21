"""Round rules and arena geometry, one source for training, the CPU scripts and Unity.

Unity reads the same numbers from joint_map.json "arena" (written by koth/build_mjcf.py):
  {"plateau_r", "slope_k", "out_radius", "fall_height", "fall_up"}

  ground      z = -SLOPE_K * max(r - PLATEAU_R, 0)^2   (flat summit of radius PLATEAU_R, then the dome slope)
  out         radius > OUT_RADIUS, or pelvis up-axis z < FALL_UP, or pelvis height above the ground < FALL_HEIGHT
              (Unity MatchDirector, duel_stats.py, play.py).
  training    KothEnv ends an episode on tipped (up-axis z < TRAIN_TIP_UP, i.e. upside-down past horizontal),
              low (height < FALL_HEIGHT), radius > cfg["max_radius"] (OUT_RADIUS for the combat rungs) and, while
              cfg["terminate_on_leg_contact"], own foot-foot / foot-shin contact. The tipped threshold is looser
              than FALL_UP on purpose (a stumble that recovers is not a fall during training).
"""
import numpy as np

PLATEAU_R = 1.5
SLOPE_K = 1.0 / 9.0          # z = -k (r - 1.5)^2  -> 45 deg at r = 6
OUT_RADIUS = 1.7
FALL_HEIGHT = 0.3
FALL_UP = 0.3
TRAIN_TIP_UP = 0.0


def ground_z(r):
    """Ground height at distance r from the summit centre (scalar or numpy array)."""
    if np.ndim(r) == 0:
        return -SLOPE_K * max(r - PLATEAU_R, 0) ** 2
    return -SLOPE_K * np.maximum(np.asarray(r) - PLATEAU_R, 0) ** 2


def upright(quat) -> float:
    """World z of the body z axis for a wxyz quaternion (1 = upright)."""
    return 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)


def is_out(pos, quat, arena: bool = True) -> bool:
    """Unity's round rule for one fighter. pos = pelvis xyz, quat = pelvis wxyz (numpy / sequences).
    arena=False (flat floor): only the fall tests apply."""
    r = np.hypot(pos[0], pos[1])
    h = ground_z(r) if arena else 0.0
    return bool((arena and r > OUT_RADIUS) or upright(quat) < FALL_UP or pos[2] - h < FALL_HEIGHT)


def arena_json() -> dict:
    """The joint_map.json "arena" object (Unity mirror)."""
    return {"plateau_r": PLATEAU_R, "slope_k": SLOPE_K, "out_radius": OUT_RADIUS, "fall_height": FALL_HEIGHT, "fall_up": FALL_UP}
