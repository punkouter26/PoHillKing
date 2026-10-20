"""Vectorized G1 environment on mujoco_warp with a torch front-end. Implements the rsl_rl VecEnv contract.

One policy row per robot: N worlds x A robots = M rows, ordered [world0/a, world0/b, world1/a, ...].
Rungs:
  r0  flat, 1 robot, zero command, pushes + boxes            (stand and recover)
  r1  flat, 1 robot, random velocity commands                 (walk and turn)
  r2  arena, 1 robot, command = go to the plateau centre      (slope + plateau holding)
  r3  arena, 2 robots, command = go to the opponent           (approach)
  r5ffa  arena, 4 robots, learner a_ vs three frozen opponents (free-for-all, staged; see STAGED_TRAINING.md)
Observation layout is koth/obs.py (103 dims), mirrored by Unity's ObsBuilder.cs. For r2+ the command slots are
filled by obs.goal_command(), a fixed geometric law that Unity mirrors too, so the network input never changes.
With more than two robots "the opponent" of a row is its nearest opponent still in the round (as DemoDirector
retargets brains); cfg["combat2"] appends a second combat block for the second-nearest one.
"""
from __future__ import annotations
import json, math, os
import numpy as np, torch, mujoco, warp as wp, mujoco_warp as mjw
from tensordict import TensorDict
from koth.obs import build_obs, build_combat, quat_rotate_inverse, goal_command, down_like, tactic_goal, OBS_DIM, COMBAT_DIM, GAIT_FREQ_HZ, STYLE_POSES, SHOVE_FAR, SHOVE_NEAR, SHOVE_SMOOTH
from koth.action_map import spec_of as action_spec, ckpt_spec as ckpt_action_spec, is_default as action_is_default
from koth.actor import Actor
from koth.rules import PLATEAU_R, SLOPE_K, OUT_RADIUS, FALL_HEIGHT, TRAIN_TIP_UP

ROBOT = os.environ.get("KOTH_ROBOT", "g1")
ASSETS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", ROBOT))


def terrain_height(xy: torch.Tensor, arena: bool) -> torch.Tensor:
    """Ground height under world xy (..., 2). Flat floor = 0; arena = analytic dome profile."""
    if not arena:
        return torch.zeros(xy.shape[:-1], device=xy.device)
    return -SLOPE_K * (xy.norm(dim=-1) - PLATEAU_R).clamp(min=0) ** 2


def robot_mass(robot: str) -> float:
    """One robot's mass (kg) from its model dump (flat scene: one robot + the 4 x 2 kg box pool)."""
    d = json.load(open(os.path.join(os.path.dirname(ASSETS), robot, "model_dump_flat_1p.json")))
    return float(d["total_mass"]) - 4 * 2.0


def obs_dims(cfg: dict) -> int:
    """Policy observation width for a cfg: 103 locomotion, +9 per combat block, +29 fatigue state."""
    n = OBS_DIM
    if cfg["combat"]: n += COMBAT_DIM * (2 if cfg.get("combat2") else 1)
    if (cfg.get("fatigue") or {}).get("enabled") and cfg["fatigue"].get("obs"): n += 29
    return n


def default_cfg(rung: str = "r0") -> dict:
    cfg = dict(
        scene="scene_flat_1p_train.xml", robots=["a_"], arena=False, goal=None, episode_length_s=20.0,
        command=dict(lin_vel_x=[-1.0, 1.0], lin_vel_y=[-0.5, 0.5], ang_vel_yaw=[-1.0, 1.0], zero_prob=0.1, resample_s=10.0),
        goal_cmd=dict(stop_dist=0.3, vmax=0.8),
        spawn=dict(r=[0.0, 0.0], min_separation=0.9),
        # curriculum_iters > 0 (opt-in): push and box speeds ramp from curriculum_start x to 1 x over that many
        # PPO iterations (steps_per_iter env steps each).
        push=dict(interval_s=[5.0, 10.0], vel=[0.5, 2.0]),
        projectile=dict(enabled=True, interval_s=[3.0, 8.0], speed=[3.0, 8.0], range=3.0, life_s=2.5),
        dr=dict(mass=0.15, friction=[0.4, 0.9], kp=0.2, damping=0.2, latency_prob=0.5),
        noise=dict(joint_pos=0.03, joint_vel=1.5, gravity=0.05, linvel=0.1, gyro=0.2),
        reset_noise=dict(joint_pos=0.05, joint_vel=0.3, yaw=math.pi),
        # Positive terms dominate once the robot stands (tracking 1.75 + alive 0.5); penalties stay small enough
        # that the clipped sum is rarely zero. Run r0_b had the sum clipped to zero on every step (stand_still -0.15
        # and ang_vel_xy -0.10 per step vs +0.03), so PPO saw only the entropy bonus and action std grew to 2.5.
        reward=dict(tracking_lin_vel=1.0, tracking_ang_vel=0.75, alive=0.5, upright=0.5, orientation=-2.0, ang_vel_xy=-0.05,
                    lin_vel_z=-0.5, feet_air_time=2.0, feet_slip=-0.1, feet_phase=1.0, pose=-0.1, joint_deviation_hip=-0.25,
                    joint_deviation_knee=-0.1, dof_pos_limits=-1.0, action_rate=-0.01, joint_vel=-2e-4, stand_still=-0.1,
                    tracking_heading=1.0, plateau=0.0, off_rim=0.0, ring_advantage=0.0, opp_radius=0.0, self_edge=0.0, push_out=0.0, hand_push=0.0, win=0.0, draw=0.0, termination=-1.0,
                    stalemate=0.0, belly_push=0.0, arm_contact=0.0, getup_height=0.0, getup_upright=0.0, getup_pose=0.0, getup_still=0.0, getup_rise=0.0,
                    potential=0.0),
        tracking_sigma=0.25, swing_height=0.12, terminate_on_leg_contact=True, max_radius=8.0, combat=False,
        contact_free=False,        # True: within reach of the opponent, arms/waist are exempt from the pose penalty
        gait_free=False,           # True: within reach of the opponent, the gait-clock rewards are off (feet may plant)
        shove=False,               # True: the learner fights in the shove style (see SHOVE_*); frozen opponents opt in with ":shove"
        style_pose="shove",        # the styled rest pose: "shove" (arms forward, the G1's) or "bull" (Trump's charge), obs.STYLE_POSES
        style_ramp_iters=0,        # > 0 (training only): the styled pose and smoothing ease in from 0 to full over this many PPO iterations
        frozen_opponent=None,      # checkpoint path(s): robots b_.. are driven by these policies and only robot a_ learns
        frozen_stochastic=False,   # True while training: frozen opponents sample actions like the learner does
        # ---- opt-in staging options (all off by default; see STAGED_TRAINING.md) ----
        decimation=None,           # physics steps per policy step (None = joint_map's 10 -> 50 Hz; 15 = 33 Hz, 20 = 25 Hz)
        combat2=False,             # second combat block for the second-nearest opponent (121 inputs; needs > 2 robots to differ)
        stalemate_rule=dict(dist=0.6, speed=0.1, seconds=2.0),   # reward["stalemate"] fires after `seconds` this close and slow
        push_out_cap=0.0,          # > 0: at most this much push_out reward per episode
        action=None,               # koth/action_map.py: dict(map "clip"|"tanh", scale None|[29], window bool); None = the original path
        reward_clip=True,          # clip each step's shaped sum at zero (terminal terms and the potential are outside the clip)
        potential_gamma=0.99,      # reward["potential"]: F = gamma * phi(s') - phi(s), phi = (opponent radius - my radius) / OUT_RADIUS
        tactics=None,              # dict(probs=[5], hold_s=[lo, hi]): learner goal points from obs.TACTICS (Unity Tactics.Goal), held per segment
        fatigue=dict(enabled=False, tau_s=10.0, min_scale=0.6, gain=2.0, omega_ref=10.0, obs=False),
        getup=None,                # dict (rung "getup"): episodes start from drops in random orientations and never end on a fall
    )
    if rung == "r0":                       # stand only: zero command, disturbances on
        cfg["command"].update(zero_prob=1.0)
    elif rung == "getup":                  # get back up after a fall (Unity: PolicyRunner's get-up brain, GameSettings.GetUp)
        # drop_prob of the resets drop the robot from drop_z with a uniformly random orientation and joints at
        # default +- joint_noise; the rest start standing, so the brain also learns to stay up once it is up. Lying
        # down is the task, not a failure: only a NaN state ends an episode early. Rewards: pelvis height (share of
        # the standing height), uprightness, and once standing (height > 0.85 x standing, up > 0.9) the default pose
        # and no drift, so the fight brain can take over from a familiar stance.
        # Run getup1 (no assist) settled at iteration 400 into sitting up and staying there (pelvis at 26 % of the standing
        # height, up-axis 0.65): standing pays 3x more but was never found. As in HoST (humanoid standing-up control, 2025),
        # a vertical pull on the pelvis (assist x body weight, only while it is below the standing height) helps early on
        # and fades linearly to zero over assist_iters PPO iterations; getup_rise pays the pelvis' upward speed until standing.
        # getup2 (assist 0.8 only while low, warm from getup1) kept sitting: the help is now the whole body weight at any
        # height (assist_low_only False), from a fresh start, so a lying robot first hangs into a standing posture.
        cfg.update(getup=dict(drop_prob=0.85, drop_z=0.85, joint_noise=0.6, stand_h=0.85, stand_up=0.9, assist=1.0, assist_iters=1000, assist_low_only=False),
                   episode_length_s=8.0, terminate_on_leg_contact=False)
        cfg["command"].update(zero_prob=1.0)
        cfg["push"].update(interval_s=[1e6, 1e6]); cfg["projectile"].update(enabled=False)
        cfg["reward"] = {k: 0.0 for k in cfg["reward"]}
        cfg["reward"].update(getup_height=2.0, getup_upright=1.0, getup_pose=1.0, getup_still=-0.2, getup_rise=0.5, action_rate=-0.01, joint_vel=-2e-4,
                             dof_pos_limits=-1.0, termination=-1.0)
    elif rung == "r2":                     # arena, return to and hold the plateau
        cfg.update(scene="scene_koth_1p_train.xml", arena=True, goal="center")
        # 2026-10-08: every agent starts on the summit and nobody is asked to climb. Slope spawns ([0, 2.6]) were
        # used for run r2_a and are kept only as an eval option (eval.py --slope).
        cfg["spawn"].update(r=[0.0, 1.3])
        cfg["reward"].update(plateau=0.5, off_rim=-0.5)
    elif rung == "r3":                     # arena, two robots, walk up to the opponent
        cfg.update(scene="scene_koth_2p_train.xml", robots=["a_", "b_"], arena=True, goal="opponent")
        cfg["spawn"].update(r=[0.3, 1.3])
        cfg["goal_cmd"].update(stop_dist=0.5)
        cfg["reward"].update(off_rim=-0.5)
        cfg["projectile"].update(enabled=False)
    elif rung == "r4":                     # sumo: both robots walk into each other; leaving r < 1.7 m or falling loses
        cfg.update(scene="scene_koth_2p_train.xml", robots=["a_", "b_"], arena=True, goal="opponent", combat=True, max_radius=OUT_RADIUS)
        cfg["spawn"].update(r=[0.5, 1.3])
        cfg["goal_cmd"].update(stop_dist=0.0, vmax=1.0)
        cfg["projectile"].update(enabled=False); cfg["push"].update(interval_s=[1e6, 1e6])
        # A win must be worth more than surviving to the bell. Run r4_a (alive 0.5, upright 0.5, win 10) converged in
        # 130 iterations to both robots leaning on each other for the full 20 s: 40 reward for a draw vs 10 for a win.
        cfg["episode_length_s"] = 15.0
        cfg["reward"].update(tracking_lin_vel=0.3, tracking_ang_vel=0.1, tracking_heading=0.1, stand_still=0.0, alive=0.1, upright=0.2,
                             feet_air_time=0.5, feet_phase=0.3, ring_advantage=2.0, push_out=3.0, win=20.0, termination=-20.0, draw=-10.0)
    elif rung == "r4att":                  # attacker curriculum: a_ learns to drive a plateau-holding b_ off the summit
        cfg.update(default_cfg("r4"))
        # Defender = frozen policy with a zero command, standing near the rim. A defender that walks back to the
        # centre (goal "center") is the same robot with the same friction limit and survives 98 % of 2 m/s kicks:
        # runs r4att_a/b never produced a win against it.
        cfg.update(goal=["opponent", "stand"])
        # ring_advantage is flat while pushing from the inside (both radii grow together), so run r4att_a learned
        # only to stop falling. Reward the opponent's radius directly; a draw costs the same as a loss so that
        # engaging is never worse than stalling.
        cfg["reward"].update(ring_advantage=0.0, opp_radius=3.0, self_edge=-3.0, push_out=3.0, win=30.0, termination=-10.0, draw=-10.0)
        cfg["spawn"].update(r=[[0.0, 0.9], [0.9, 1.35]])
        cfg["goal_cmd"].update(stop_dist=[0.0, 0.3], vmax=[1.0, 0.8])
    elif rung == "r4sp":                   # mirror self-play: both robots attack and both learn (rows = 2 per world)
        cfg.update(default_cfg("r4"))
        cfg["spawn"].update(r=[0.3, 1.3])
        cfg["reward"].update(ring_advantage=0.0, opp_radius=3.0, self_edge=-3.0, push_out=3.0, win=30.0, termination=-10.0, draw=-10.0)
    elif rung == "r4league":              # learner a_ attacks; b_ is a frozen earlier attacker drawn from a pool, also attacking
        cfg.update(default_cfg("r4sp"))
    elif rung == "r4shove":               # r4league, but free to shove: arms up and feet planted while in contact
        cfg.update(default_cfg("r4league")); cfg["contact_free"] = True
        # careful1 held its arms at its sides (pose penalty on all 29 joints) and marched on the spot against the
        # opponent (gait clock), changing its targets by about 0.6 per joint per step (action_rate cost 0.002 a step).
        # shover1 (also gait_free, action_rate -0.05, hand_push = "a hand within 0.3 m of its torso"): no arm use, no
        # smoother, lost 13 % to the standing walker (careful1: 3 %). That hand test was already true half the time
        # with the arms hanging. hand_push is now forward reach toward the opponent; gait and action_rate are back.
        # shover2 (hand_push 0.5 on forward reach): hands stayed 5 cm behind the pelvis, 12 % losses to the walker.
        # Rewards could not get the arms forward or the jitter down (29 % of actions sat on the clamp, wrists, shoulder
        # yaw, elbows and ankles worst), so both are now built into the action path: see SHOVE_* and step().
        cfg.update(contact_free=False, shove=True)
        cfg["reward"].update(self_edge=-5.0, termination=-25.0)
    elif rung == "r4pbrs":                # r4league with the ring shaping replaced by one potential (2026-10-20, option 8)
        # ring_advantage, opp_radius, self_edge and push_out are shaped by hand and none is a potential difference: each can
        # be farmed without winning (r4att_a stood still for flat ring_advantage, push_out needed a per-episode cap). A
        # potential-based term F = gamma phi(s') - phi(s), phi = (opp radius - my radius) / OUT_RADIUS (Ng et al. 1999)
        # cannot change which policy is optimal; it only says "pushing outward is progress". It is added outside the zero
        # clip (a clipped potential is no longer a potential). Win, loss and draw stay as in r4league.
        cfg.update(default_cfg("r4league"))
        cfg["reward"].update(ring_advantage=0.0, opp_radius=0.0, self_edge=0.0, push_out=0.0, potential=10.0)
    elif rung == "r4tactic":              # r4league, but the learner's goal point follows a random tactic (option 6)
        # Unity's Tactician lost to plain Press in every variant (log 2026-10-19): the brains only ever walked straight at the
        # opponent, so a rollout that moves the goal point asks for behaviour they never practised. Here half the segments
        # are Press and the rest Flank left/right, Regroup or Lurk (Tactics.Goal), held 1-3 s; the command still comes from
        # the same goal law and the network inputs are unchanged (112), so the brain drops into the Tactician as it is.
        # Tracking pays more than in r4league so that the commanded goal point matters.
        cfg.update(default_cfg("r4league"))
        cfg.update(tactics=dict(probs=[0.5, 0.125, 0.125, 0.125, 0.125], hold_s=[1.0, 3.0]))
        cfg["reward"].update(tracking_lin_vel=0.6, tracking_heading=0.3)
    elif rung == "r4probe":               # a_ walks into b_ (no stop distance); b_ holds the plateau, starts at the rim
        cfg.update(scene="scene_koth_2p_train.xml", robots=["a_", "b_"], arena=True, goal=["opponent", "center"])
        cfg["spawn"].update(r=[[0.0, 0.5], [1.25, 1.35]])
        cfg["goal_cmd"].update(stop_dist=[0.0, 0.3], vmax=[1.0, 0.8])
        cfg["projectile"].update(enabled=False)
    elif rung == "r5ffa":                 # staged: four-way free-for-all, learner a_ vs frozen b_ c_ d_ (warm start from a 112 brain)
        cfg.update(default_cfg("r4league"))
        cfg.update(scene="scene_koth_4p_train.xml", robots=["a_", "b_", "c_", "d_"], combat2=True, episode_length_s=20.0,
                   terminate_on_leg_contact=False)          # Unity's rule (no own-leg-contact ending)
        cfg["spawn"].update(r=[0.7, 1.2])
    elif rung.split("_", 1)[0] in HUMANS and rung.split("_", 1)[1] in ("r0b", "r1b"):   # a human body's walking recipe (KOTH_ROBOT=<body>)
        cfg.update(default_cfg(rung.split("_", 1)[1][:2]))
        human_overrides(cfg, rung.split("_", 1)[0])
    elif "_att_" in rung:                 # cross-body attacker stage: push a standing frozen b_ of another body off (r4att, mixed scene)
        me, other = rung.split("_att_")
        cfg.update(default_cfg("r4att")); cfg.update(scene=mix_scene([me, other]), terminate_on_leg_contact=False)
        if me == "trump": cfg["reward"].update(push_out=1.0, belly_push=4.0, arm_contact=-1.0); cfg["goal_cmd"].update(vmax=[0.7, 0.8])
    elif "_vs_" in rung:                  # cross-body duel: learner a_ (KOTH_ROBOT) against frozen b_ of another body (build_mjcf mix)
        me, other = rung.split("_vs_")
        cfg.update(default_cfg("trump_r4bull" if me == "trump" else "r4league"))
        cfg.update(scene=mix_scene([me, other]), terminate_on_leg_contact=False)
    elif rung.endswith("_ffa") and rung[:-4] in HUMANS:   # a human's four-way free-for-all against three G1 (build_mjcf mix)
        me = rung[:-4]
        cfg.update(default_cfg("r5ffa")); cfg.update(scene=mix_scene([me, "g1", "g1", "g1"]))
        if me == "trump": cfg["reward"].update(push_out=1.0, belly_push=4.0, arm_contact=-1.0); cfg["goal_cmd"].update(vmax=0.7)
    elif rung == "trump_bull":            # Trump's bull charge: trump_r4bull with the styled action path (obs.BULL_POSE)
        # trump_bull1 switched the pose on at once and learned to stay out of contact (11 % wins vs a standing Trump):
        # ease it in over 300 iterations from fight1's plain stance instead.
        cfg.update(default_cfg("trump_r4bull"), shove=True, style_pose="bull", style_ramp_iters=300)
    elif rung == "trump_r4bull":          # Trump's fighting style (KOTH_ROBOT=trump), see the note below
        cfg.update(default_cfg("r4league"))
        # "Bulldozer": he wins by walking his 102 kg into the opponent and shoving with his chest and belly, arms
        # down. push_out is paid mostly through belly_push (the opponent's outward speed while one of his trunk
        # capsules touches it), arms touching the opponent cost (arm_contact), and he walks in slower than the G1
        # brains (vmax 0.7). The pose penalty on the arms stays on (no contact_free), so the arms hang.
        cfg["reward"].update(push_out=1.0, belly_push=4.0, arm_contact=-1.0, hand_push=0.0)
        cfg["goal_cmd"].update(vmax=0.7)
        cfg["terminate_on_leg_contact"] = False
    return cfg


HUMANS = ("kim", "trump", "grandpa")         # human bodies (koth/build_<body>.py) with the G1's 29 joints


def mix_scene(bodies: list) -> str:
    """Path, relative to this robot's assets, of the mixed-body training scene build_mjcf mix writes to assets/g1."""
    return os.path.relpath(os.path.join(os.path.dirname(ASSETS), "g1", f"scene_koth_{len(bodies)}p_{'_'.join(bodies)}_train.xml"), ASSETS)


def scene_bodies(scene_path: str, prefixes: list) -> list:
    """Body kind of each robot of a scene, from the "robots" list build_mjcf writes into model_dump_<tag>.json next to
    it (mixed scenes); every robot is this module's ROBOT when the dump has no list."""
    base = os.path.basename(scene_path); tag = base[len("scene_"):-len("_train.xml")] if base.endswith("_train.xml") else ""
    dump = os.path.join(os.path.dirname(scene_path), f"model_dump_{tag}.json")
    listed = {r["prefix"]: r["body"] for r in json.load(open(dump)).get("robots", [])} if tag and os.path.exists(dump) else {}
    return [listed.get(p, ROBOT) for p in prefixes]


def human_overrides(cfg: dict, body: str) -> dict:
    """A human body's walking recipe 'b' (staged), first written for Kim: her first r0 run used the G1's settings and
    failed (log 2026-10-09). Trump and Grandpa use the same recipe with their own mass ratio and swing height.
    - Kicks are a velocity change (qvel += dv), i.e. an impulse of m * dv: scale dv by m_G1 / m_Kim so a kick carries
      the same impulse as on the G1 (33.3 kg vs 56.0 kg -> x 0.60, 2 m/s -> 1.19 m/s). Boxes are real bodies and
      already transfer the right momentum, so their speed is left alone.
    - Disturbance curriculum: kicks and boxes ramp from 25 % to 100 % over the first 300 iterations, so she first
      learns to stand at all (run kim_r0 never kept a robot up for 20 s).
    - swing_height 0.09 (G1 0.12): her thigh is 0.27 m and her hips sit low; 0.12 m foot lift is a larger share of
      her leg and pushes the feet_phase optimum toward a high-stepping gait she cannot hold.
    - Pose penalties halved (pose -0.05, hip -0.15, knee -0.05): her torque per kilogram is about 2/3 of the G1's
      (knee 150 Nm / 56 kg vs 139 Nm / 33 kg), so recovering needs larger joint excursions than the G1-tuned
      weights allow.
    - terminate_on_leg_contact off: her short thighs and narrow stance make foot-shin touches common, and Unity's
      round rule does not end on them either."""
    try:
        ratio = robot_mass("g1") / robot_mass(body)
    except OSError:
        ratio = 33.341 / {"kim": 55.994, "trump": 102.0, "grandpa": 86.0}[body]
    cfg["push"].update(vel=[v * ratio for v in cfg["push"]["vel"]], curriculum_iters=300, curriculum_start=0.25, steps_per_iter=24)
    cfg["swing_height"] = {"kim": 0.09, "trump": 0.10, "grandpa": 0.09}[body]     # Trump, Grandpa: an older man's shuffle
    cfg["reward"].update(pose=-0.05, joint_deviation_hip=-0.15, joint_deviation_knee=-0.05)
    cfg["terminate_on_leg_contact"] = False
    cfg["mass_ratio"] = ratio
    return cfg


# Shove style, mirrored in Unity's PolicyRunner. Arms: as the opponent comes from SHOVE_FAR to SHOVE_NEAR metres the
# arm rest pose moves from the standing pose to hands-forward at chest height (the policy's action is still added on
# top, and the pose penalty follows the moved rest pose). Smoothing: the action is low-passed before it becomes a
# joint target, target = rest + scale * f, f = SHOVE_SMOOTH * f_prev + (1 - SHOVE_SMOOTH) * action.



def opp_spec(o: str):
    """'ckpt.pt[:stand][:shove]' -> (absolute checkpoint path, set of flags)."""
    path, _, flags = o.partition(".pt")
    return os.path.abspath(path + ".pt"), set(f for f in flags.split(":") if f)


class FrozenPolicy(torch.nn.Module):
    """Deterministic actor rebuilt from an rsl_rl checkpoint (koth.actor.Actor). Used for opponents that do not
    learn. Reads only as many observation columns as it was trained on."""
    def __init__(self, ckpt: str, device, stochastic: bool = False):
        super().__init__()
        sd = torch.load(ckpt, map_location=device, weights_only=False)["actor_state_dict"]
        # Training: sample actions with the policy's own exploration std. The learner explores with that noise, and a
        # noise-free opponent beat an identical noisy learner in about 95 % of rounds (league gen2, first attempt).
        self.noise_std = sd["distribution.std_param"].abs() if stochastic and "distribution.std_param" in sd else None
        self.actor = Actor(sd)
        self.obs_dim = self.actor.obs_dim

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.actor(obs[:, :self.obs_dim])
        if self.noise_std is not None: x = x + torch.randn_like(x) * self.noise_std
        return x


class FrozenPool:
    """Several frozen opponents; each opponent row is assigned one of them, re-drawn whenever its world resets.
    slots = frozen robots per world (1 in a duel: one row per world)."""
    def __init__(self, ckpts, device, num_rows, stochastic=False, slots=1):
        # an entry "path.pt:stand" is an opponent that gets a zero command (holds its ground) instead of attacking
        specs = [opp_spec(c) for c in ckpts]
        self.stands = torch.tensor(["stand" in f for _, f in specs], device=device)
        self.shoves = torch.tensor(["shove" in f for _, f in specs], device=device)
        self.any_stand = any("stand" in f for _, f in specs); self.any_shove = any("shove" in f for _, f in specs)
        loaded = {}                           # one module per checkpoint file, shared by all entries naming it
        for c, _ in specs:
            if c not in loaded: loaded[c] = FrozenPolicy(c, device, stochastic)
        self.nets = [loaded[c] for c, _ in specs]
        self.amaps = [ckpt_action_spec(c) for c, _ in specs]          # each opponent's own action path (koth/action_map.py)
        self.slots = slots
        self.choice = torch.randint(0, len(self.nets), (num_rows,), device=device)
        self.weights = None                   # draw weights per entry (koth/curriculum.py); None = uniform

    def set_weights(self, w):
        w = torch.tensor(w, dtype=torch.float32, device=self.choice.device); assert len(w) == len(self.nets) and w.sum() > 0
        self.weights = w / w.sum()

    def resample(self, world_ids):
        rows = world_ids if self.slots == 1 else (world_ids[:, None] * self.slots + torch.arange(self.slots, device=world_ids.device)).reshape(-1)
        if self.weights is None: self.choice[rows] = torch.randint(0, len(self.nets), (len(rows),), device=self.choice.device)
        else: self.choice[rows] = torch.multinomial(self.weights, len(rows), replacement=True)

    def __call__(self, obs):
        out = torch.empty(obs.shape[0], 29, device=obs.device)
        for i, net in enumerate(self.nets):
            sel = self.choice == i
            if sel.any(): out[sel] = net(obs[sel])
        return out


class KothEnv:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda", seed: int = 0):
        """num_envs = number of worlds. self.num_envs (what rsl_rl sees) = worlds x robots."""
        self.cfg, self.device = cfg, torch.device(device)
        torch.manual_seed(seed)
        wp.init()
        spec = json.load(open(os.path.join(ASSETS, "joint_map.json")))
        self.joints, self.decimation = spec["joints"], spec["decimation"]
        self.action_scale, self.sim_dt, self.ctrl_dt = spec["action_scale"], spec["sim_dt"], spec["ctrl_dt"]
        if cfg.get("decimation"):                        # decision-rate override (Unity must use the same decimation)
            self.decimation = int(cfg["decimation"]); self.ctrl_dt = self.decimation * self.sim_dt
        self.num_actions = len(self.joints)
        self.max_episode_length = int(round(cfg["episode_length_s"] / self.ctrl_dt))
        self.arena = bool(cfg["arena"])
        prefixes = cfg["robots"]
        N = self.N = num_envs; A = self.A = len(prefixes); M = self.M = N * A
        fo = cfg["frozen_opponent"]
        self.frozen = None if not fo else FrozenPool(fo if isinstance(fo, (list, tuple)) else [fo], self.device, N * max(A - 1, 1),
                                                     cfg["frozen_stochastic"], slots=max(A - 1, 1))
        assert self.frozen is None or A >= 2
        self.num_envs = N if self.frozen is not None else M      # rows rsl_rl sees (learner = robot a_ only when the others are frozen)
        # learner rows (robot a_ of every world) and frozen-opponent rows
        self.lrows = slice(0, None, A)
        self.orows = slice(1, None, 2) if A == 2 else torch.tensor([r for r in range(M) if r % A], dtype=torch.long, device=device)
        fz = cfg.get("fatigue") or {}; self.fatigue_on = bool(fz.get("enabled"))

        mjm = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, cfg["scene"]))
        mjd = mujoco.MjData(mjm); mujoco.mj_resetDataKeyframe(mjm, mjd, 0); mujoco.mj_forward(mjm, mjd)
        self.mjm = mjm
        batched = ("body_mass", "body_inertia", "body_subtreemass", "geom_friction", "dof_damping", "actuator_gainprm", "actuator_biasprm")
        if self.fatigue_on: batched += ("actuator_forcerange",)
        self.m = mjw.put_model(mjm, batch_sizes={k: N for k in batched})
        self.m.opt.warn_overflow = 0
        self.d = mjw.put_data(mjm, mjd, nworld=N, nconmax=48 * A, njmax=240 * A)

        # ---- indices (name based, same resolution rule as Unity's JointMap); leading dim = robot ----
        J, G, B, ACT = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_ACTUATOR
        jid = lambda n: mujoco.mj_name2id(mjm, J, n); gid = lambda n: mujoco.mj_name2id(mjm, G, n); bid = lambda n: mujoco.mj_name2id(mjm, B, n)
        T = lambda x, dt=torch.long: torch.tensor(np.array(x), device=device, dtype=dt)
        js = [[jid(p + j) for j in self.joints] for p in prefixes]; assert min(min(r) for r in js) >= 0, "joint names missing"
        self.qadr = T([[mjm.jnt_qposadr[j] for j in r] for r in js]); self.dadr = T([[mjm.jnt_dofadr[j] for j in r] for r in js])
        aid = [[mujoco.mj_name2id(mjm, ACT, p + j) for j in self.joints] for p in prefixes]; self.aid = T(aid)
        roots = [jid(p + "floating_base_joint") for p in prefixes]
        self.rq = T([mjm.jnt_qposadr[r] for r in roots]); self.rd = T([mjm.jnt_dofadr[r] for r in roots])
        self.rq7 = self.rq[:, None] + torch.arange(7, device=device); self.rd6 = self.rd[:, None] + torch.arange(6, device=device)
        self.feet_body = T([[bid(p + "left_ankle_roll_link"), bid(p + "right_ankle_roll_link")] for p in prefixes])
        bname = lambda b: mujoco.mj_id2name(mjm, B, b) or ""; gname = lambda g: mujoco.mj_id2name(mjm, G, g) or ""
        # Mixed bodies have different counts: pad each robot's list with its own last index (a repeated index in the
        # randomisation writes the same value twice).
        pad = lambda rows: [r + [r[-1]] * (max(map(len, rows)) - len(r)) for r in rows]
        self.robot_bodies = T(pad([[b for b in range(mjm.nbody) if bname(b).startswith(p)] for p in prefixes]))
        self.robot_geoms = T(pad([[g for g in range(mjm.ngeom) if gname(g).startswith(p) and mjm.geom_contype[g]] for p in prefixes]))
        # geom lookup: owner robot (-1 none) and kind (0 other, 1 ground, 2 left foot, 3 right foot, 4 shin, 5 trunk, 6 arm)
        owner = np.full(mjm.ngeom, -1); kind = np.zeros(mjm.ngeom, dtype=np.int64)
        for n in ("floor", "arena"):
            if gid(n) >= 0: kind[gid(n)] = 1
        for a, p in enumerate(prefixes):
            for g in range(mjm.ngeom):
                if gname(g).startswith(p): owner[g] = a
            for k, s in ((2, "left"), (3, "right")):
                for n in ("foot_box_collision", "foot1_collision", "foot2_collision", "foot3_collision"): kind[gid(f"{p}{s}_{n}")] = k
                for n in ("shin_collision", "linkage_brace_collision"): kind[gid(f"{p}{s}_{n}")] = 4
                for n in ("upper_arm_collision", "forearm_collision", "hand_collision"):      # human bodies' names
                    if gid(f"{p}{s}_{n}") >= 0: kind[gid(f"{p}{s}_{n}")] = 6
            for n in ("pelvis_collision", "abdomen_collision", "chest_collision"):
                if gid(p + n) >= 0: kind[gid(p + n)] = 5
        self.geom_owner = T(owner); self.geom_kind = T(kind)
        box_q = [int(mjm.jnt_qposadr[jid(f"{b}_free")]) for b in spec["pool_bodies"]]; box_d = [int(mjm.jnt_dofadr[jid(f"{b}_free")]) for b in spec["pool_bodies"]]
        self.box_q = T(box_q); self.box_d = T(box_d)
        self.box_park = T(spec["pool_park"], torch.float32)
        self.box_life = torch.zeros(N, len(spec["pool_bodies"]), device=device)   # seconds left in flight; <=0 = parked
        # every pool box's qpos (7 each) / qvel (6 each) columns, and the parked qpos row (park xyz, unit quaternion)
        qc = np.concatenate([np.arange(q, q + 7) for q in box_q]); dc = np.concatenate([np.arange(v, v + 6) for v in box_d])
        cols = lambda c: slice(int(c[0]), int(c[-1]) + 1) if np.array_equal(c, np.arange(c[0], c[-1] + 1)) else T(c)
        self.box_qcols, self.box_dcols = cols(qc), cols(dc)          # slices (views) when the pool is contiguous
        self.box_qpark = T([[*spec["pool_park"][i], 1.0, 0.0, 0.0, 0.0] for i in range(len(box_q))], torch.float32).reshape(-1)
        self.hip_idx = T([i for i, j in enumerate(self.joints) if "hip_roll" in j or "hip_yaw" in j])
        self.knee_idx = T([i for i, j in enumerate(self.joints) if "knee" in j])
        self.upper_idx = T([i for i, j in enumerate(self.joints) if any(k in j for k in ("waist", "shoulder", "elbow", "wrist"))])
        self.hand_body = T([[bid(p + "left_wrist_yaw_link"), bid(p + "right_wrist_yaw_link")] for p in prefixes])
        assert self.hand_body.min() >= 0, "hand bodies missing"
        # Per-robot body data, (A, 29) / (A,) and repeated per row (M, ...): a mixed scene (cross-body training) holds
        # robots of different bodies, each with its own standing pose, ranges and heights. Same-body scenes are unchanged.
        self.bodies = scene_bodies(os.path.join(ASSETS, cfg["scene"]), prefixes)
        specs = {b: spec if b == ROBOT else json.load(open(os.path.join(os.path.dirname(ASSETS), b, "joint_map.json"))) for b in set(self.bodies)}
        dp = np.array([specs[b]["default_pose"] for b in self.bodies]); rows = lambda x: x.repeat(N, *([1] * (x.dim() - 1)))
        style = STYLE_POSES[cfg.get("style_pose", "shove")]
        self.shove_offset = rows(T([[next((v - dp[a][i] for k, v in style.items() if k in j), 0.0) for i, j in enumerate(self.joints)] for a in range(A)], torch.float32))
        self.default_pose = rows(T(dp, torch.float32))
        self.key_qpos = T(mjm.key_qpos[0], torch.float32); self.key_ctrl = T(mjm.key_ctrl[0], torch.float32)
        self.key_root_z = T([float(specs[b]["key_root_z"]) for b in self.bodies], torch.float32)
        self.jnt_lo_a = T(mjm.jnt_range[js, 0], torch.float32); self.jnt_hi_a = T(mjm.jnt_range[js, 1], torch.float32)      # (A, 29), for resets
        self.jnt_lo, self.jnt_hi = rows(self.jnt_lo_a), rows(self.jnt_hi_a)
        self.ctrl_lo = rows(T(mjm.actuator_ctrlrange[aid, 0], torch.float32)); self.ctrl_hi = rows(T(mjm.actuator_ctrlrange[aid, 1], torch.float32))
        # Action path (koth/action_map.py), per row: the learner's from cfg "action", each frozen opponent's from its own
        # checkpoint (re-read whenever the pool draws). With every row on the original path the step is unchanged.
        self.amap = action_spec(cfg.get("action"))
        maps = [self.amap] + (self.frozen.amaps if self.frozen is not None else [])
        self.any_amap = not all(action_is_default(x) for x in maps)
        if self.any_amap:
            self.win_kp = rows(T(mjm.actuator_gainprm[aid, 0], torch.float32)); self.win_kv = rows(T(-mjm.actuator_biasprm[aid, 2], torch.float32))
            self.win_tau = rows(T(mjm.actuator_forcerange[aid, 1], torch.float32))
            to_t = lambda x: (torch.tensor(x["map"] == "tanh", device=device), torch.tensor(x["scale"] if x["scale"] is not None else [self.action_scale] * 29, dtype=torch.float32, device=device),
                              torch.tensor(bool(x["window"]), device=device))
            self._amap_t = [to_t(x) for x in maps]                     # 0 = learner, 1.. = pool entries
            lt, ls, lw = self._amap_t[0]
            self.row_tanh = lt.repeat(M).clone(); self.row_scale = ls.repeat(M, 1).clone(); self.row_window = lw.repeat(M).clone()
            if self.frozen is not None:
                self.pool_tanh = torch.stack([t for t, _, _ in self._amap_t[1:]]); self.pool_scale = torch.stack([s_ for _, s_, _ in self._amap_t[1:]])
                self.pool_window = torch.stack([w_ for _, _, w_ in self._amap_t[1:]])
        rest = lambda f: float(mjd.xpos[f][2] - (0.0 if not self.arena else float(terrain_height(torch.tensor(mjd.xpos[f][:2], dtype=torch.float32), True))))
        self.foot_rest_z = rows(T([rest(int(self.feet_body[a, 0])) for a in range(A)], torch.float32))[:, None]

        # ---- zero-copy torch views ----
        self.qpos = wp.to_torch(self.d.qpos); self.qvel = wp.to_torch(self.d.qvel); self.ctrl = wp.to_torch(self.d.ctrl)
        self.xpos = wp.to_torch(self.d.xpos); self.time = wp.to_torch(self.d.time)
        self.con_geom = wp.to_torch(self.d.contact.geom); self.con_world = wp.to_torch(self.d.contact.worldid)
        self.con_dist = wp.to_torch(self.d.contact.dist); self.nacon = wp.to_torch(self.d.nacon)
        self.mp = {k: wp.to_torch(getattr(self.m, k)) for k in batched}
        self.mp0 = {k: v.clone() for k, v in self.mp.items()}

        # ---- buffers (per policy row unless noted) ----
        z = lambda *s: torch.zeros(*s, device=device)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=device)           # per world
        self.command = z(M, 3); self.last_action = z(M, 29); self.action = z(M, 29); self.phase = z(M)
        self.feet_air_time = z(M, 2); self.feet_contact = torch.zeros(M, 2, dtype=torch.bool, device=device); self.prev_feet_xy = z(M, 2, 2)
        self.yaw_target = z(M)     # integral of the commanded yaw rate, leashed to the actual heading
        self.push_timer = z(M); self.proj_timer = z(N); self.next_box = torch.zeros(N, dtype=torch.long, device=device)
        self.dr_params = z(M, 3)   # mass scale, friction, kp scale  (critic obs)
        self.latency = torch.zeros(M, dtype=torch.bool, device=device); self.pending_ctrl = z(M, 29)
        self.style = torch.zeros(M, dtype=torch.bool, device=device)      # rows fighting in the shove style
        self.act_f = z(M, 29); self.pose_offset = z(M, 29)
        self.out = torch.zeros(M, dtype=torch.bool, device=device)        # > 2 robots: eliminated this round (latched)
        self.stale_t = z(M); self.push_out_sum = z(M)                    # anti-stalemate timer, push_out paid this episode
        self.fatigue = z(M, 29)                                          # per-joint fatigue state in [0, 1]
        self.phi_prev = z(M)                                             # reward["potential"]: phi of the state the step started from
        self.tactic = torch.zeros(M, dtype=torch.long, device=device); self.tactic_t = z(M)    # cfg tactics: current tactic, seconds left
        self.outcomes = torch.zeros(7, dtype=torch.float64, device=device)   # won, lost, drawn, rows, fallen, standing, steps (koth/curriculum.py)
        self._rows_const = torch.full((), float(M), dtype=torch.float64, device=device)
        self.push_scale_override = None; self.curriculum_log = None
        self.common_step = 0; self.curriculum_offset_iters = 0           # env steps taken (disturbance curriculum)
        if A > 2: self._nbr1 = self._nbr2 = torch.arange(M, device=device)
        if self.fatigue_on:
            self.aid_flat = self.aid.reshape(-1); self.frc0 = self.mp0["actuator_forcerange"][:, self.aid_flat].clone()   # (N, A*29, 2)
            self.act_force = wp.to_torch(self.d.actuator_force)
        self.extras: dict = {}
        self.reward_terms: dict[str, torch.Tensor] = {}
        per = lambda v: (v if isinstance(v, (list, tuple)) else [v] * A)          # scalar or one value per robot
        self.goal_is_opp = torch.tensor([g == "opponent" for g in per(cfg["goal"])], device=device).repeat(N)
        self.goal_is_none = torch.tensor([g == "stand" for g in per(cfg["goal"])], device=device).repeat(N)
        self.goal_stop = torch.tensor(per(cfg["goal_cmd"]["stop_dist"]), device=device, dtype=torch.float32).repeat(N)
        self.goal_vmax = torch.tensor(per(cfg["goal_cmd"]["vmax"]), device=device, dtype=torch.float32).repeat(N)

        mjw.step(self.m, self.d)                        # warm up kernels before capture
        with wp.ScopedCapture() as cap:
            for _ in range(self.decimation):
                mjw.step(self.m, self.d)
        self.graph = cap.graph
        self.reset()

    # ------------------------------------------------------------------ helpers
    def _rand(self, n, lo, hi):
        n = n if isinstance(n, tuple) else (n,)
        return torch.rand(*n, device=self.device) * (hi - lo) + lo

    def _flat(self, x): return x.reshape(self.M, *x.shape[2:])

    def _root(self):
        """Per row: pos (M,3), quat wxyz (M,4), world linvel (M,3), body angvel (M,3)."""
        q = self._flat(self.qpos[:, self.rq7]); v = self._flat(self.qvel[:, self.rd6])
        if self.A > 2: self._refresh_nbr(q[:, :3])
        return q[:, :3], q[:, 3:7], v[:, :3], v[:, 3:6]

    def _refresh_nbr(self, pos):
        """> 2 robots: flat row index of each row's nearest and second-nearest opponent still in the round (an
        eliminated robot is only chosen when nobody else is left; with one opponent left the second is the first)."""
        N, A = self.N, self.A
        p = pos[:, :2].reshape(N, A, 2); d = (p[:, :, None] - p[:, None]).norm(dim=3)
        d = d + torch.eye(A, device=self.device) * 1e6 + self.out.view(N, 1, A).float() * 1e3
        order = d.argsort(dim=2); base = (torch.arange(N, device=self.device) * A)[:, None]
        n1, n2 = order[:, :, 0], order[:, :, 1]
        n2 = torch.where(d.gather(2, n2[:, :, None])[:, :, 0] < 1e3, n2, n1)
        self._nbr1 = (n1 + base).reshape(-1); self._nbr2 = (n2 + base).reshape(-1)

    def _opp(self, x, k: int = 1):
        """Per-row tensor of the other robot in the same world (A == 2), or of the k-th nearest opponent (A > 2)."""
        if self.A > 2: return x[self._nbr1 if k == 1 else self._nbr2]
        return x.view(self.N, 2, *x.shape[1:]).flip(1).reshape(x.shape)

    def _joints(self):
        return self._flat(self.qpos[:, self.qadr]), self._flat(self.qvel[:, self.dadr])

    def _foot_contacts(self):
        """(M,2) bool feet on ground, (M,) bool own-leg contact (foot-foot or foot-shin of the same robot).
        Contacts that do not count are scattered into a dummy last slot instead of being masked out."""
        n = int(self.nacon[0].item())
        g = self.con_geom[:n].long(); w = self.con_world[:n].long(); active = self.con_dist[:n] < 0
        k1, k2 = self.geom_kind[g[:, 0]], self.geom_kind[g[:, 1]]; o1, o2 = self.geom_owner[g[:, 0]], self.geom_owner[g[:, 1]]
        f1, f2 = (k1 == 2) | (k1 == 3), (k2 == 2) | (k2 == 3)
        M2 = self.M * 2
        feet = torch.zeros(M2 + 1, dtype=torch.bool, device=self.device)
        hit = active & f1 & (k2 == 1); feet[torch.where(hit, (w * self.A + o1) * 2 + (k1 - 2), M2)] = True
        hit = active & f2 & (k1 == 1); feet[torch.where(hit, (w * self.A + o2) * 2 + (k2 - 2), M2)] = True
        leg = torch.zeros(self.M + 1, dtype=torch.bool, device=self.device)
        hit = active & (o1 == o2) & (o1 >= 0) & ((f1 & (f2 | (k2 == 4))) | (f2 & (f1 | (k1 == 4))))
        leg[torch.where(hit, w * self.A + o1, self.M)] = True
        # touching another robot with my trunk (5) or arms (6); read by belly_push / arm_contact
        other = active & (o1 >= 0) & (o2 >= 0) & (o1 != o2)
        trunk = torch.zeros(self.M + 1, dtype=torch.bool, device=self.device); arm = torch.zeros_like(trunk)
        for k, flag in ((5, trunk), (6, arm)):
            flag[torch.where(other & (k1 == k), w * self.A + o1, self.M)] = True
            flag[torch.where(other & (k2 == k), w * self.A + o2, self.M)] = True
        self._trunk_touch, self._arm_touch = trunk[:self.M], arm[:self.M]
        return feet[:M2].view(self.M, 2), leg[:self.M]

    # ------------------------------------------------------------------ commands
    def _resample_commands(self, rows):
        c = self.cfg["command"]; n = len(rows)
        cmd = torch.stack([self._rand(n, *c["lin_vel_x"]), self._rand(n, *c["lin_vel_y"]), self._rand(n, *c["ang_vel_yaw"])], 1)
        cmd[torch.rand(n, device=self.device) < c["zero_prob"]] = 0.0
        self.command[rows] = cmd

    def _goal_vec(self, pos):
        """World xy vector from each robot to its goal (M,2). cfg['goal'] is one mode or one per robot."""
        to_center = -pos[:, :2]
        if self.A == 1:
            return to_center
        if self.A > 2:
            to_opp = self._opp(pos)[:, :2] - pos[:, :2]
        else:
            p = pos[:, :2].view(self.N, self.A, 2); to_opp = (p.flip(1) - p).reshape(self.M, 2)
        return torch.where(self.goal_is_opp[:, None], to_opp, to_center)

    def _update_goal_commands(self):
        if self.cfg["goal"] is None: return
        pos, quat, _, _ = self._root()
        gvec, stop = self._goal_vec(pos), self.goal_stop
        if self.cfg.get("tactics"):                       # learner rows with a tactic other than Press walk at its goal point
            tg, ts = tactic_goal(self.tactic, pos[:, :2], pos[:, :2] + gvec, self.goal_stop)
            use = (self.tactic > 0) & self.goal_is_opp
            gvec = torch.where(use[:, None], tg, gvec); stop = torch.where(use, ts, stop)
        cmd = goal_command(quat, gvec, stop, self.goal_vmax, pos[:, :2].norm(dim=1), self.cfg["goal_cmd"].get("rim_brake"))
        if self.frozen is not None and self.frozen.any_stand:
            self.goal_is_none[self.orows] = self.frozen.stands[self.frozen.choice]     # per opponent: is it a stander?
        cmd[self.goal_is_none] = 0.0                       # "stand": zero command, the robot just holds its ground
        self.command[:] = cmd

    def _resample_tactics(self, rows):
        """cfg tactics: a new tactic (obs.TACTICS index) and hold time for these rows; frozen opponents always Press."""
        tc = self.cfg["tactics"]; n = len(rows)
        t = torch.multinomial(torch.tensor(tc["probs"], dtype=torch.float32, device=self.device), n, replacement=True)
        if self.frozen is not None: t = torch.where(rows % self.A == 0, t, 0)
        self.tactic[rows] = t; self.tactic_t[rows] = self._rand(n, *tc["hold_s"])

    def _phi(self, pos):
        """Potential of each row's state: (opponent radius - my radius) / OUT_RADIUS, 0 without an opponent."""
        if self.A < 2: return torch.zeros(self.M, device=self.device)
        r = pos[:, :2].norm(dim=1); return ((self._opp(pos)[:, :2].norm(dim=1) - r) / OUT_RADIUS).clamp(-1.0, 1.0)

    def _count_outcomes(self, fallen, decided, timeout, lost_w=None, won_w=None):
        """Accumulate, on the GPU, the learner's (robot a_) round outcomes: won, lost, drawn, plus rows, fallen rows,
        standing rows (getup) and steps. read_outcomes() returns and clears them (one sync per PPO iteration)."""
        N, A = self.N, self.A
        if A == 1: won = lost = drew = torch.zeros(N, device=self.device)
        else:
            f = fallen.view(N, A)
            lost = (lost_w if lost_w is not None else f[:, 0]).float()
            won = (won_w if won_w is not None else f[:, 1:].any(1) & ~f[:, 0]).float()
            drew = (timeout & ~decided).float()
        st = self._standing.sum() if self.cfg.get("getup") else torch.zeros((), device=self.device)
        c = self._rows_const
        self.outcomes += torch.stack([won.sum().double(), lost.sum().double(), drew.sum().double(), c, fallen.sum().double(), st.double(), c])

    def read_outcomes(self) -> list:
        out = self.outcomes.tolist(); self.outcomes.zero_(); return out

    # ------------------------------------------------------------------ reset / randomization
    def _randomize(self, ids):
        dr = self.cfg["dr"]; n = len(ids); A = self.A; w = ids[:, None, None]
        ms = 1 + self._rand((n, A), -dr["mass"], dr["mass"]); fr = self._rand((n, A), *dr["friction"])
        ks = 1 + self._rand((n, A), -dr["kp"], dr["kp"]); ds = 1 + self._rand((n, A), -dr["damping"], dr["damping"])
        rb = self.robot_bodies[None]
        for k in ("body_mass", "body_subtreemass"):
            self.mp[k][w, rb] = self.mp0[k][w, rb] * ms[:, :, None]
        self.mp["body_inertia"][w, rb] = self.mp0["body_inertia"][w, rb] * ms[:, :, None, None]
        self.mp["geom_friction"][w, self.robot_geoms[None], 0] = fr[:, :, None]
        self.mp["dof_damping"][w, self.dadr[None]] = self.mp0["dof_damping"][w, self.dadr[None]] * ds[:, :, None]
        self.mp["actuator_gainprm"][w, self.aid[None], 0] = self.mp0["actuator_gainprm"][w, self.aid[None], 0] * ks[:, :, None]
        self.mp["actuator_biasprm"][w, self.aid[None], 1] = self.mp0["actuator_biasprm"][w, self.aid[None], 1] * ks[:, :, None]
        rows = self._rows(ids)
        self.dr_params[rows] = torch.stack([ms, fr, ks], 2).reshape(-1, 3)
        self.latency[rows] = torch.rand(len(rows), device=self.device) < dr["latency_prob"]

    def _rows(self, world_ids):
        return (world_ids[:, None] * self.A + torch.arange(self.A, device=self.device)).reshape(-1)

    def _spawn_xy(self, n):
        """(n, A, 2) spawn positions. Flat scenes keep the keyframe position."""
        sp = self.cfg["spawn"]; rr = sp["r"]
        rr = rr if isinstance(rr[0], (list, tuple)) else [rr] * self.A
        if max(h for _, h in rr) <= 0:
            return None
        r = torch.stack([torch.sqrt(self._rand(n, lo * lo, hi * hi)) for lo, hi in rr], 1); ang = self._rand((n, self.A), 0, 2 * math.pi)
        if self.A == 2:                                   # opposite-ish sides, never overlapping
            ang[:, 1] = ang[:, 0] + math.pi + self._rand(n, -1.0, 1.0)
            r[:, 1] = torch.maximum(r[:, 1], (sp["min_separation"] - r[:, 0]).clamp(max=1.35))
        elif self.A > 2:                                  # evenly spread bearings (+-0.3 rad), like Unity's ring
            ang = ang[:, :1] + torch.arange(self.A, device=self.device) * (2 * math.pi / self.A) + self._rand((n, self.A), -0.3, 0.3)
        return torch.stack([r * torch.cos(ang), r * torch.sin(ang)], 2)

    def reset_idx(self, ids: torch.Tensor):
        if len(ids) == 0: return
        n = len(ids); A = self.A; rn = self.cfg["reset_noise"]; rows = self._rows(ids)
        q = self.key_qpos.expand(n, -1).clone(); v = torch.zeros(n, self.qvel.shape[1], device=self.device)
        jq = q[:, self.qadr] + (torch.rand(n, A, 29, device=self.device) * 2 - 1) * rn["joint_pos"]
        q[:, self.qadr] = torch.clamp(jq, self.jnt_lo_a, self.jnt_hi_a)
        v[:, self.dadr] = (torch.rand(n, A, 29, device=self.device) * 2 - 1) * rn["joint_vel"]
        yaw = self._rand((n, A), -rn["yaw"], rn["yaw"])
        xy = self._spawn_xy(n)
        if xy is not None:
            slope = 2 * SLOPE_K * (xy.norm(dim=2) - PLATEAU_R).clamp(min=0)
            q[:, self.rq + 0] = xy[:, :, 0]; q[:, self.rq + 1] = xy[:, :, 1]
            q[:, self.rq + 2] = self.key_root_z + terrain_height(xy, self.arena) + 0.02 + 0.12 * slope
        q[:, self.rq + 3] = torch.cos(yaw / 2); q[:, self.rq + 4] = 0; q[:, self.rq + 5] = 0; q[:, self.rq + 6] = torch.sin(yaw / 2)
        gu = self.cfg.get("getup")
        if gu:                                                  # drops: uniformly random orientation, loose joints, from drop_z
            drop = torch.rand(n, A, device=self.device) < gu["drop_prob"]
            rq = torch.randn(n, A, 4, device=self.device); rq = rq / rq.norm(dim=2, keepdim=True)
            for k in range(4): q[:, self.rq + 3 + k] = torch.where(drop, rq[:, :, k], q[:, self.rq + 3 + k])
            q[:, self.rq + 2] = torch.where(drop, gu["drop_z"] + terrain_height(q[:, self.rq[:, None] + torch.arange(2, device=self.device)], self.arena), q[:, self.rq + 2])
            loose = torch.clamp(self.key_qpos[self.qadr] + (torch.rand(n, A, 29, device=self.device) * 2 - 1) * gu["joint_noise"], self.jnt_lo_a, self.jnt_hi_a)
            q[:, self.qadr] = torch.where(drop[:, :, None], loose, q[:, self.qadr])
        q[:, self.box_qcols] = self.box_qpark                   # park the pool
        self.qpos[ids] = q; self.qvel[ids] = v; self.box_life[ids] = 0
        self.ctrl[ids] = self.key_ctrl; self.pending_ctrl[rows] = self.key_ctrl[self.aid].repeat(n, 1)
        self.last_action[rows] = 0; self.act_f[rows] = 0; self.pose_offset[rows] = 0; self.phase[rows] = 0; self.yaw_target[rows] = yaw.reshape(-1)
        self.feet_air_time[rows] = 0; self.feet_contact[rows] = True
        self.out[rows] = False; self.stale_t[rows] = 0; self.push_out_sum[rows] = 0; self.fatigue[rows] = 0
        fxy = q[:, self.rq + 0], q[:, self.rq + 1]   # feet start under the pelvis; exact value only matters for one step of slip
        self.prev_feet_xy[rows] = torch.stack(fxy, 2).reshape(-1, 1, 2).expand(-1, 2, -1)
        self.episode_length_buf[ids] = 0
        self.push_timer[rows] = self._rand(len(rows), *self.cfg["push"]["interval_s"])
        self.proj_timer[ids] = self._rand(n, *self.cfg["projectile"]["interval_s"])
        if self.cfg["goal"] is None: self._resample_commands(rows)
        if self.cfg.get("tactics"): self._resample_tactics(rows)
        if getattr(self, "frozen", None) is not None:
            self.frozen.resample(ids)
            if self.any_amap:                       # frozen rows take the action path of the opponent just drawn
                ch = self.frozen.choice
                self.row_tanh[self.orows] = self.pool_tanh[ch]; self.row_scale[self.orows] = self.pool_scale[ch]; self.row_window[self.orows] = self.pool_window[ch]
        self._randomize(ids)

    def reset(self):
        self.reset_idx(torch.arange(self.N, device=self.device))
        self._step_sim()                                        # one control step so xpos/contacts are valid
        self.prev_feet_xy = self._flat(self.xpos[:, self.feet_body])[:, :, :2].clone()
        self._update_goal_commands()
        return self.get_observations()

    # ------------------------------------------------------------------ disturbances
    def _disturb_scale(self):
        """Disturbance curriculum factor (None = off): curriculum_start -> 1 over cfg push curriculum_iters."""
        if self.push_scale_override is not None: return self.push_scale_override       # koth/curriculum.py stage
        p = self.cfg["push"]; iters = p.get("curriculum_iters", 0)
        if not iters: return None
        done_it = self.curriculum_offset_iters + self.common_step / p.get("steps_per_iter", 24)
        s0 = p.get("curriculum_start", 0.25)
        return s0 + (1 - s0) * min(1.0, done_it / iters)

    def _disturb(self):
        self.push_timer -= self.ctrl_dt; self.proj_timer -= self.ctrl_dt
        scale = self._disturb_scale()
        push = self.push_timer <= 0
        if push.any():
            rows = push.nonzero(as_tuple=True)[0]; n = len(rows); w = rows // self.A; a = rows % self.A
            ang = self._rand(n, 0, 2 * math.pi); mag = self._rand(n, *self.cfg["push"]["vel"])
            if scale is not None: mag = mag * scale
            self.qvel[w, self.rd[a]] += mag * torch.cos(ang); self.qvel[w, self.rd[a] + 1] += mag * torch.sin(ang)
            self.push_timer[rows] = self._rand(n, *self.cfg["push"]["interval_s"])
        if self.cfg["projectile"]["enabled"]:
            proj = self.proj_timer <= 0
            if proj.any():
                ids = proj.nonzero(as_tuple=True)[0]; n = len(ids)
                box = self.next_box[ids]; self.next_box[ids] = (box + 1) % len(self.box_q)
                victim = torch.randint(0, self.A, (n,), device=self.device)
                target = self.qpos[ids[:, None], self.rq7[victim][:, :3]].clone(); target[:, 2] += 0.1
                ang = self._rand(n, 0, 2 * math.pi); r = self.cfg["projectile"]["range"]
                start = target + torch.stack([r * torch.cos(ang), r * torch.sin(ang), torch.full((n,), 0.3, device=self.device)], 1)
                speed = self._rand(n, *self.cfg["projectile"]["speed"])
                if scale is not None: speed = speed * scale
                vel = (target - start); vel = vel / vel.norm(dim=1, keepdim=True) * speed[:, None]
                qa = self.box_q[box]; da = self.box_d[box]
                for k in range(3):
                    self.qpos[ids, qa + k] = start[:, k]; self.qvel[ids, da + k] = vel[:, k]
                self.qpos[ids, qa + 3] = 1; self.qpos[ids, qa + 4] = 0; self.qpos[ids, qa + 5] = 0; self.qpos[ids, qa + 6] = 0
                for k in range(3, 6): self.qvel[ids, da + k] = 0
                self.box_life[ids, box] = self.cfg["projectile"]["life_s"]
                self.proj_timer[ids] = self._rand(n, *self.cfg["projectile"]["interval_s"])
        # pin parked boxes (no shelf, no contacts): park pose, zero velocity, every control step
        self.box_life -= self.ctrl_dt
        parked = self.box_life <= 0                                                   # (N, boxes)
        self.qpos[:, self.box_qcols] = torch.where(parked.repeat_interleave(7, 1), self.box_qpark, self.qpos[:, self.box_qcols])
        self.qvel[:, self.box_dcols] = torch.where(parked.repeat_interleave(6, 1), 0.0, self.qvel[:, self.box_dcols])

    # ------------------------------------------------------------------ fatigue (opt-in)
    def _apply_fatigue_limits(self):
        """Scale every robot actuator's force range by 1 - (1 - min_scale) * fatigue."""
        fz = self.cfg["fatigue"]; scale = 1 - (1 - fz["min_scale"]) * self.fatigue
        self.mp["actuator_forcerange"][:, self.aid_flat] = self.frc0 * scale.view(self.N, self.A * 29, 1)

    def _update_fatigue(self, jvel):
        """Fatigue follows the normalized mechanical power |tau * qdot| / (tau_max * omega_ref) of the last physics
        step through a first-order lag: dF/dt = (gain * P - F) / tau_s, F clamped to [0, 1]."""
        fz = self.cfg["fatigue"]
        tau = self.act_force[:, self.aid_flat].reshape(self.M, 29); tmax = self.frc0[:, :, 1].reshape(self.M, 29)
        p = (tau * jvel).abs() / (tmax * fz["omega_ref"])
        self.fatigue = (self.fatigue + self.ctrl_dt * (fz["gain"] * p - self.fatigue) / fz["tau_s"]).clamp(0.0, 1.0)

    # ------------------------------------------------------------------ get-up assist (rung getup, training only)
    def _getup_assist(self):
        """Vertical pull on each pelvis (assist x subtree weight; with assist_low_only only while it is below stand_h x the
        standing height), fading linearly to 0 over assist_iters PPO iterations (24 env steps each). xfrc_applied is
        re-written every control step."""
        gu = self.cfg["getup"]; it = self.curriculum_offset_iters + self.common_step / 24
        a = gu["assist"] * max(0.0, 1.0 - it / gu["assist_iters"]); self._assist_now = a
        if not hasattr(self, "_xfrc"):
            self._xfrc = wp.to_torch(self.d.xfrc_applied)
            J = mujoco.mjtObj.mjOBJ_BODY
            self.pelvis_body = torch.tensor([mujoco.mj_name2id(self.mjm, J, p + "pelvis") for p in self.cfg["robots"]], device=self.device)
        pb = self.pelvis_body
        if a <= 0: self._xfrc[:, pb, 2] = 0.0; return
        weight = self.mp["body_subtreemass"][:, pb] * 9.81                                    # (N, A), mass randomised per world
        if gu.get("assist_low_only", True):
            pos = self._flat(self.qpos[:, self.rq7])[:, :3]
            low = (pos[:, 2] - terrain_height(pos[:, :2], self.arena)) < gu["stand_h"] * self.key_root_z.repeat(self.N)
            self._xfrc[:, pb, 2] = a * weight * low.view(self.N, self.A).float()
        else:
            self._xfrc[:, pb, 2] = a * weight

    # ------------------------------------------------------------------ step
    def _step_sim(self):
        # torch and warp use different CUDA streams: flush torch writes, run the graph, wait for it
        torch.cuda.synchronize(self.device)
        wp.capture_launch(self.graph)
        wp.synchronize_device()

    def step(self, actions: torch.Tensor):
        N, A, M = self.N, self.A, self.M
        if self.frozen is not None:
            full = torch.empty(M, 29, device=self.device); full[self.lrows] = actions.to(self.device)
            full[self.orows] = self.frozen(self._obs_full["policy"][self.orows]); actions = full
        raw = actions.to(self.device)
        if self.any_amap: self.action = torch.where(self.row_tanh[:, None], torch.tanh(raw), torch.clamp(raw, -1, 1))
        else: self.action = torch.clamp(raw, -1, 1)
        act = self.action
        if self.cfg["shove"] or (self.frozen is not None and self.frozen.any_shove):
            self.style[:] = self.cfg["shove"]
            if self.frozen is not None: self.style[self.orows] = self.frozen.shoves[self.frozen.choice]
            ramp_it = self.cfg.get("style_ramp_iters", 0)
            ramp = min(1.0, self.common_step / 24 / ramp_it) if ramp_it else 1.0      # 24 env steps per PPO iteration
            k = SHOVE_SMOOTH * ramp
            self.act_f = torch.where(self.style[:, None], k * self.act_f + (1 - k) * self.action, self.action)
            act = self.act_f
            p = self._root()[0]; dist = (self._opp(p)[:, :2] - p[:, :2]).norm(dim=1)
            self.pose_offset = (((SHOVE_FAR - dist) / (SHOVE_FAR - SHOVE_NEAR)).clamp(0, 1) * self.style.float() * ramp)[:, None] * self.shove_offset
        if not self.any_amap: target = torch.clamp(self.default_pose + self.pose_offset + self.action_scale * act, self.ctrl_lo, self.ctrl_hi)
        else:
            target = self.default_pose + self.pose_offset + self.row_scale * act
            jq, jv = self._joints(); kvq = self.win_kv * jv            # torque-feasible window at the current state (action_map.py)
            windowed = torch.minimum(torch.maximum(target, jq + (kvq - self.win_tau) / self.win_kp), jq + (kvq + self.win_tau) / self.win_kp)
            target = torch.clamp(torch.where(self.row_window[:, None], windowed, target), self.ctrl_lo, self.ctrl_hi)
        if A > 2: target = torch.where(self.out[:, None], 0.0, target)   # eliminated: limp, as Unity's paused runner (ctrl 0)
        # one-step actuation latency on a random subset of robots (DR)
        apply = torch.where(self.latency[:, None], self.pending_ctrl, target); self.pending_ctrl = target
        self.ctrl[:, self.aid.reshape(-1)] = apply.reshape(N, A * 29)
        self._disturb()
        if self.fatigue_on: self._apply_fatigue_limits()
        if self.cfg.get("getup") and self.cfg["getup"].get("assist", 0) > 0: self._getup_assist()
        self._step_sim()
        self.episode_length_buf += 1; self.common_step += 1
        self.phase = torch.remainder(self.phase + 2 * math.pi * GAIT_FREQ_HZ * self.ctrl_dt, 2 * math.pi)
        feet, leg = self._foot_contacts()
        pos, quat, linv, angv = self._root()
        grav = quat_rotate_inverse(quat, down_like(quat))                # gravity direction in the pelvis frame
        rew = self._rewards(feet, pos, quat, linv, angv, grav)           # uses the command the policy acted on
        if self.fatigue_on: self._update_fatigue(self._joints()[1])
        gz = grav[:, 2]
        nan = (torch.isnan(self.qpos).any(1) | torch.isnan(self.qvel).any(1)).repeat_interleave(A)
        # A world whose simulation blew up (rare; trump_bull1 hit one in its first 40 iterations) ends as a fall below,
        # but its rewards were computed from the NaN state: zero them so they never reach PPO.
        rew = torch.where(nan, torch.zeros_like(rew), rew)
        radius = pos[:, :2].norm(dim=1)
        tipped = gz > -TRAIN_TIP_UP; low = pos[:, 2] - terrain_height(pos[:, :2], self.arena) < FALL_HEIGHT
        fallen = tipped | low | nan
        if self.arena: fallen |= radius > self.cfg["max_radius"]
        if self.cfg["terminate_on_leg_contact"]: fallen |= leg
        if self.cfg.get("getup"): fallen = nan                  # getting up is the task: lying down never ends the episode
        timeout = self.episode_length_buf >= self.max_episode_length
        if A > 2:                                    # free-for-all: an eliminated robot stays in the world, limp
            fallen = fallen & ~self.out; self.out |= fallen               # fallen = eliminated this step
            o = self.out.view(N, A); lost = o[:, 0].clone(); won_w = o[:, 1:].all(1) & ~lost      # clone: the reset clears self.out before eval reads extras
            decided = lost | won_w; world_done = decided | timeout
        else:
            decided = fallen.view(N, A).any(1); world_done = decided | timeout
        done = world_done.repeat_interleave(A)
        term = self.cfg["reward"]["termination"] * fallen.float()      # one-off, outside the clip, not scaled by dt
        rew += term; self.reward_terms["termination"] = term
        if self.A == 2 and self.cfg["reward"]["win"] != 0.0:
            won = self._opp(fallen) & ~fallen; win = self.cfg["reward"]["win"] * won.float()
            rew += win; self.reward_terms["win"] = win
        elif self.A > 2 and self.cfg["reward"]["win"] != 0.0:       # last one in (the learner row of each world)
            won = torch.zeros(N, A, dtype=torch.bool, device=self.device); won[:, 0] = won_w
            win = self.cfg["reward"]["win"] * won.reshape(M).float(); rew += win; self.reward_terms["win"] = win
        draw_on = self.A >= 2 and self.cfg["reward"]["draw"] != 0.0
        if draw_on:                                  # the bell is a real terminal state in a duel, not a truncation
            drew = (timeout & ~decided).repeat_interleave(A); d_r = self.cfg["reward"]["draw"] * drew.float()
            rew += d_r; self.reward_terms["draw"] = d_r
        if self.cfg["reward"].get("potential", 0.0) != 0.0:     # potential-based shaping, outside the clip (r4pbrs)
            # phi(s') of a terminal state is 0 (Ng et al.): a round that ended pays -phi(s) for its last step. A truncation
            # (the bell without a draw reward) keeps gamma phi(s'), as the critic bootstraps through it.
            terminal = (fallen.view(N, A).any(1) | decided).repeat_interleave(A) | (drew if draw_on else torch.zeros_like(done))
            phi = self._phi(pos)
            pot = self.cfg["reward"]["potential"] * (self.cfg["potential_gamma"] * phi * (~terminal).float() - self.phi_prev)
            pot = torch.where(nan, torch.zeros_like(pot), pot)
            rew += pot; self.reward_terms["potential"] = pot
        self._count_outcomes(fallen, decided, timeout, lost if A > 2 else None, won_w if A > 2 else None)
        if A > 2: time_outs = (timeout & ~decided).repeat_interleave(A) & (~drew if draw_on else torch.ones_like(done))
        else: time_outs = (done & ~fallen) if not draw_on else (done & ~fallen & ~drew)
        lr = self.lrows if self.frozen is not None else slice(None)          # log the learner rows only
        self.extras = {"time_outs": time_outs, "log": {f"rew/{k}": v[lr].mean() for k, v in self.reward_terms.items()},
                       "fallen": fallen, "nan": nan, "radius": radius}
        pre = self._pre_clip[lr]                                              # option 7: how often the zero clip binds, and how much it removes
        self.extras["log"]["rew/clip_share"] = (pre < 0).float().mean(); self.extras["log"]["rew/clipped_away"] = (-pre).clamp(min=0).mean()
        if self.curriculum_log: self.extras["log"].update({f"curriculum/{k}": v for k, v in self.curriculum_log.items()})
        self.extras["log"]["ep/leg_contact_term"] = leg.float().mean(); self.extras["log"]["ep/fallen"] = fallen.float().mean()
        self.extras["cause"] = torch.stack([tipped, low, leg], 1)          # why a row counted as fallen (besides ring-out)
        if A >= 2: self.extras["sep"] = self._sep
        if A > 2: self.extras["out"] = self.out.clone(); self.extras["world_won"] = won_w; self.extras["world_lost"] = lost
        if self.arena: self.extras["log"]["ep/on_plateau"] = (radius < PLATEAU_R).float().mean()
        if self.cfg.get("getup"):
            self.extras["log"]["ep/standing"] = self._standing.mean(); self.extras["standing"] = self._standing
            self.extras["log"]["ep/assist"] = torch.tensor(getattr(self, "_assist_now", 0.0))
        if self.A >= 2:
            self.extras["log"]["ep/decided"] = (decided.float().sum() / world_done.float().sum().clamp(min=1))
            self.extras["log"]["ep/separation"] = self._sep.mean()
        self.last_action = self.action.clone()
        self.feet_contact = feet; self.prev_feet_xy = self._foot_xy.clone()
        if self.cfg["goal"] is None:
            resample = (self.episode_length_buf % int(self.cfg["command"]["resample_s"] / self.ctrl_dt) == 0).nonzero(as_tuple=True)[0]
            if len(resample): self._resample_commands(self._rows(resample))
        if self.cfg.get("tactics"):
            self.tactic_t -= self.ctrl_dt; due = (self.tactic_t <= 0).nonzero(as_tuple=True)[0]
            if len(due): self._resample_tactics(due)
        self.reset_idx(world_done.nonzero(as_tuple=True)[0])
        self._update_goal_commands()
        if self.cfg["reward"].get("potential", 0.0) != 0.0: self.phi_prev = self._phi(self._root()[0])   # reset rows start from their new state
        obs = self.get_observations()
        self.extras["observations"] = obs
        if self.frozen is not None:
            self.extras["time_outs"] = self.extras["time_outs"][self.lrows]
            return obs, rew[self.lrows], done[self.lrows], self.extras
        return obs, rew, done, self.extras

    # ------------------------------------------------------------------ rewards
    def _rewards(self, feet, pos, quat, linv_w, angv, grav):
        R = self.cfg["reward"]; dt = self.ctrl_dt
        jpos, jvel = self._joints()
        linv = quat_rotate_inverse(quat, linv_w)              # pelvis-frame linear velocity (obs[0:3]); grav = obs[6:9]
        cmd = self.command; cmd_on = (cmd.norm(dim=1) > 0.01).float()
        s = self.cfg["tracking_sigma"]
        t = {}
        t["tracking_lin_vel"] = torch.exp(-((cmd[:, :2] - linv[:, :2]) ** 2).sum(1) / s)
        t["tracking_ang_vel"] = torch.exp(-((cmd[:, 2] - angv[:, 2]) ** 2) / s)
        # Heading tracking: the gait makes the instantaneous yaw rate oscillate by 0.3-0.4 rad/s (std), which buries
        # the mean yaw-rate error in the term above; run r1_a tracked linear velocity but ignored yaw commands.
        # The integral of the commanded rate is a clean target. The leash keeps a shove from winding it up.
        w_, x_, y_, z_ = quat.unbind(1); yaw = torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
        err = self.yaw_target + cmd[:, 2] * dt - yaw; err = torch.atan2(torch.sin(err), torch.cos(err)).clamp(-0.6, 0.6)
        self.yaw_target = yaw + err
        t["tracking_heading"] = torch.exp(-(err ** 2) / 0.1)
        t["orientation"] = (grav[:, :2] ** 2).sum(1)
        t["ang_vel_xy"] = (angv[:, :2] ** 2).sum(1)
        t["lin_vel_z"] = linv[:, 2] ** 2
        # feet
        fpos = self._flat(self.xpos[:, self.feet_body]); foot_xy = fpos[:, :, :2]; self._foot_xy = foot_xy
        foot_z = fpos[:, :, 2] - terrain_height(foot_xy, self.arena) - self.foot_rest_z
        first_contact = feet & ~self.feet_contact
        near = torch.zeros_like(cmd_on)
        if self.A >= 2:
            opos = self._opp(pos); to_opp = opos[:, :2] - pos[:, :2]; dist = to_opp.norm(dim=1); self._sep = dist
            near = (dist < 0.9).float()
        free = near if self.cfg["contact_free"] else torch.zeros_like(near)      # 1 = within reach: arms exempt from the pose penalty
        plant = near if self.cfg["gait_free"] else torch.zeros_like(near)
        self.feet_air_time += dt
        t["feet_air_time"] = ((self.feet_air_time - 0.1) * first_contact.float()).sum(1) * cmd_on * (1 - plant)
        self.feet_air_time[feet] = 0
        slip = ((foot_xy - self.prev_feet_xy) / dt).norm(dim=2)
        t["feet_slip"] = (slip * feet.float()).sum(1)
        des = torch.stack([torch.sin(self.phase), torch.sin(self.phase + math.pi)], 1).clamp(min=0) * self.cfg["swing_height"]
        t["feet_phase"] = torch.exp(-((foot_z - des) ** 2).sum(1) / 0.01) * cmd_on * (1 - plant)
        # posture
        dev = jpos - self.default_pose - self.pose_offset
        t["pose"] = (dev ** 2).sum(1) - free * (dev[:, self.upper_idx] ** 2).sum(1)
        t["joint_deviation_hip"] = dev[:, self.hip_idx].abs().sum(1)
        t["joint_deviation_knee"] = dev[:, self.knee_idx].abs().sum(1)
        t["dof_pos_limits"] = ((self.jnt_lo + 0.05 - jpos).clamp(min=0) + (jpos - self.jnt_hi + 0.05).clamp(min=0)).sum(1)
        t["action_rate"] = ((self.action - self.last_action) ** 2).sum(1)
        t["stand_still"] = dev.abs().sum(1) * (1 - cmd_on)
        t["alive"] = torch.ones_like(cmd_on)
        t["upright"] = torch.exp(-(grav[:, :2] ** 2).sum(1) / 0.05)
        t["joint_vel"] = (jvel ** 2).sum(1)
        radius = pos[:, :2].norm(dim=1)
        if self.A >= 2:
            ovel = self._opp(linv_w); orad = opos[:, :2].norm(dim=1)
            t["ring_advantage"] = (orad - radius).clamp(-1.0, 1.0)                     # be more central than the opponent
            t["opp_radius"] = orad.clamp(max=OUT_RADIUS) / OUT_RADIUS                  # how far out the opponent is
            t["self_edge"] = (radius - 1.2).clamp(min=0.0)                             # my own margin to the rim
            t["push_out"] = ((ovel[:, :2] * opos[:, :2]).sum(1) / orad.clamp(min=0.1)).clamp(-1.0, 2.0) * near   # its outward speed while I am on it
            t["belly_push"] = t["push_out"] * self._trunk_touch.float()                 # ... while my trunk touches someone
            t["arm_contact"] = self._arm_touch.float()                                 # my arms touch someone
            # Arms: how far each hand reaches from my pelvis toward the opponent (0 with the arms hanging, about 0.35 m
            # with shoulders pitched forward), averaged over both hands, once the opponent is within 1.3 m.
            # ponytail: a pose-shaping term, not a contact force; kept small (0.5/s at most) so that standing in a clinch
            # with the arms out never beats winning (r4_a stalled that way). Use contact forces if it turns into posing.
            u = to_opp / dist.clamp(min=0.1)[:, None]
            hands = self._flat(self.xpos[:, self.hand_body])[:, :, :2] - pos[:, None, :2]
            t["hand_push"] = ((hands * u[:, None]).sum(2).clamp(0.0, 0.35) / 0.35).mean(1) * (dist < 1.3).float()
            if R.get("stalemate", 0.0) != 0.0:                # opt-in: both slow and close for too long
                st = self.cfg["stalemate_rule"]
                slow = torch.maximum(linv_w[:, :2].norm(dim=1), ovel[:, :2].norm(dim=1)) < st["speed"]
                self.stale_t = torch.where((dist < st["dist"]) & slow, self.stale_t + dt, torch.zeros_like(self.stale_t))
                t["stalemate"] = (self.stale_t > st["seconds"]).float()
        gu = self.cfg.get("getup")
        if gu:                                          # rung getup: height and uprightness, then a settled standing pose
            h = (pos[:, 2] - terrain_height(pos[:, :2], self.arena)) / self.key_root_z.repeat(self.N); up = -grav[:, 2]
            standing = ((h > gu["stand_h"]) & (up > gu["stand_up"])).float(); self._standing = standing
            t["getup_height"] = h.clamp(0.0, 1.0)
            t["getup_upright"] = ((up + 1) / 2) ** 2
            t["getup_pose"] = torch.exp(-(dev ** 2).sum(1)) * standing
            t["getup_still"] = ((linv_w[:, :2] ** 2).sum(1) + (angv ** 2).sum(1) * 0.1) * standing
            t["getup_rise"] = linv_w[:, 2].clamp(0.0, 1.0) * (1 - standing)
        t["plateau"] = (radius < 1.2).float()
        t["off_rim"] = (radius - PLATEAU_R).clamp(min=0)
        self.reward_terms = {k: R[k] * v * dt for k, v in t.items() if R.get(k, 0.0) != 0.0}
        cap = self.cfg.get("push_out_cap", 0.0)
        if cap and "push_out" in self.reward_terms:            # opt-in: at most `cap` push_out reward per episode
            r = torch.minimum(self.reward_terms["push_out"], (cap - self.push_out_sum).clamp(min=0.0))
            self.push_out_sum += r.clamp(min=0.0); self.reward_terms["push_out"] = r
        # Clip the per-step sum at zero (as mujoco_playground does). Without it the penalties outweigh the
        # tracking terms early on and the policy learns to end the episode fast: run r0_a went from episode
        # length 25 to 4.8 steps in 20 iterations while "reward" rose.
        total = torch.stack(list(self.reward_terms.values()), 0).sum(0); self._pre_clip = total
        return total.clamp(min=0.0) if self.cfg.get("reward_clip", True) else total

    # ------------------------------------------------------------------ observations
    def get_observations(self) -> TensorDict:
        """Observations for the rows rsl_rl trains. The full per-robot set is cached for the frozen opponents."""
        self._obs_full = self._observe()
        return self._obs_full[self.lrows] if self.frozen is not None else self._obs_full

    def _observe(self) -> TensorDict:
        pos, quat, linv_w, angv = self._root(); jpos, jvel = self._joints()
        clean = build_obs(quat, linv_w, angv, jpos, jvel, self.default_pose, self.last_action, self.command, self.phase)
        nz = self.cfg["noise"]; u = lambda n, sc: (torch.rand(self.M, n, device=self.device) * 2 - 1) * sc
        noisy = clean.clone()
        noisy[:, 0:3] += u(3, nz["linvel"]); noisy[:, 3:6] += u(3, nz["gyro"]); noisy[:, 6:9] += u(3, nz["gravity"])
        noisy[:, 12:41] += u(29, nz["joint_pos"]); noisy[:, 41:70] += u(29, nz["joint_vel"])
        height = pos[:, 2:3] - terrain_height(pos[:, :2], self.arena)[:, None]
        critic = torch.cat([clean, linv_w, height, self.feet_contact.float(), self.dr_params], 1)
        if self.cfg["combat"]:
            block = build_combat(pos, quat, linv_w, self._opp(pos), self._opp(quat), self._opp(linv_w))
            noisy = torch.cat([noisy, block], 1); critic = torch.cat([critic, block], 1)
            if self.cfg.get("combat2"):                  # second-nearest opponent (Unity: ObsBuilder.FillCombat at 112)
                block2 = build_combat(pos, quat, linv_w, self._opp(pos, 2), self._opp(quat, 2), self._opp(linv_w, 2))
                noisy = torch.cat([noisy, block2], 1); critic = torch.cat([critic, block2], 1)
        if self.fatigue_on and self.cfg["fatigue"].get("obs"):
            noisy = torch.cat([noisy, self.fatigue], 1); critic = torch.cat([critic, self.fatigue], 1)
        return TensorDict({"policy": noisy, "critic": critic}, batch_size=[self.M])


if __name__ == "__main__":
    import time, sys
    rung = sys.argv[1] if len(sys.argv) > 1 else "r0"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    env = KothEnv(default_cfg(rung), N)
    obs = env.get_observations(); assert obs["policy"].shape == (env.num_envs, obs_dims(env.cfg)), obs["policy"].shape
    for _ in range(3): env.step(torch.zeros(env.num_envs, 29, device=env.device))
    torch.cuda.synchronize(); t = time.perf_counter(); S = 100; falls = 0
    for i in range(S):
        obs, rew, done, extras = env.step(torch.zeros(env.num_envs, 29, device=env.device))
        falls += int(extras["fallen"].sum())
    torch.cuda.synchronize(); el = time.perf_counter() - t
    print(f"{rung}: {N} worlds x {env.A} robots: {env.num_envs * S / el:,.0f} policy steps/s ({N * S * env.decimation / el:,.0f} world sim-steps/s); "
          f"zero-action falls over {S * env.ctrl_dt:.0f}s: {falls}; mean rew {rew.mean():.3f}; nan {int(extras['nan'].sum())}; "
          f"cmd mean |v| {env.command[:, :2].norm(dim=1).mean():.2f}; radius mean {extras['radius'].mean():.2f}")
    assert not torch.isnan(obs["policy"]).any()
