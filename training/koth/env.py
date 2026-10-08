"""Vectorized G1 environment on mujoco_warp with a torch front-end. Implements the rsl_rl VecEnv contract.

One policy row per robot: N worlds x A robots = M rows, ordered [world0/a, world0/b, world1/a, ...].
Rungs:
  r0  flat, 1 robot, zero command, pushes + boxes            (stand and recover)
  r1  flat, 1 robot, random velocity commands                 (walk and turn)
  r2  arena, 1 robot, command = go to the plateau centre      (slope + plateau holding)
  r3  arena, 2 robots, command = go to the opponent           (approach)
Observation layout is koth/obs.py (103 dims), mirrored by Unity's ObsBuilder.cs. For r2+ the command slots are
filled by obs.goal_command(), a fixed geometric law that Unity mirrors too, so the network input never changes.
"""
from __future__ import annotations
import json, math, os
import numpy as np, torch, mujoco, warp as wp, mujoco_warp as mjw
from tensordict import TensorDict
from koth.obs import build_obs, build_combat, quat_rotate_inverse, goal_command, OBS_DIM, COMBAT_DIM, GAIT_FREQ_HZ

ASSETS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", "g1"))
PLATEAU_R, SLOPE_K = 1.5, 1.0 / 9.0          # must match build_mjcf.py


def terrain_height(xy: torch.Tensor, arena: bool) -> torch.Tensor:
    """Ground height under world xy (..., 2). Flat floor = 0; arena = analytic dome profile."""
    if not arena:
        return torch.zeros(xy.shape[:-1], device=xy.device)
    return -SLOPE_K * (xy.norm(dim=-1) - PLATEAU_R).clamp(min=0) ** 2


def default_cfg(rung: str = "r0") -> dict:
    cfg = dict(
        scene="scene_flat_1p_train.xml", robots=["a_"], arena=False, goal=None, episode_length_s=20.0,
        command=dict(lin_vel_x=[-1.0, 1.0], lin_vel_y=[-0.5, 0.5], ang_vel_yaw=[-1.0, 1.0], zero_prob=0.1, resample_s=10.0),
        goal_cmd=dict(stop_dist=0.3, vmax=0.8),
        spawn=dict(r=[0.0, 0.0], min_separation=0.9),
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
                    tracking_heading=1.0, plateau=0.0, off_rim=0.0, ring_advantage=0.0, push_out=0.0, win=0.0, termination=-1.0),
        tracking_sigma=0.25, swing_height=0.12, terminate_on_leg_contact=True, max_radius=8.0, combat=False,
    )
    if rung == "r0":                       # stand only: zero command, disturbances on
        cfg["command"].update(zero_prob=1.0)
    elif rung == "r2":                     # arena, return to and hold the plateau
        cfg.update(scene="scene_koth_1p_train.xml", arena=True, goal="center")
        cfg["spawn"].update(r=[0.0, 2.6])
        cfg["reward"].update(plateau=0.5, off_rim=-0.5)
    elif rung == "r3":                     # arena, two robots, walk up to the opponent
        cfg.update(scene="scene_koth_2p_train.xml", robots=["a_", "b_"], arena=True, goal="opponent")
        cfg["spawn"].update(r=[0.3, 1.3])
        cfg["goal_cmd"].update(stop_dist=0.5)
        cfg["reward"].update(off_rim=-0.5)
        cfg["projectile"].update(enabled=False)
    elif rung == "r4":                     # sumo: both robots walk into each other; leaving r < 1.7 m or falling loses
        cfg.update(scene="scene_koth_2p_train.xml", robots=["a_", "b_"], arena=True, goal="opponent", combat=True, max_radius=1.7)
        cfg["spawn"].update(r=[0.5, 1.3])
        cfg["goal_cmd"].update(stop_dist=0.0, vmax=1.0)
        cfg["projectile"].update(enabled=False); cfg["push"].update(interval_s=[1e6, 1e6])
        cfg["reward"].update(tracking_lin_vel=0.5, tracking_ang_vel=0.3, tracking_heading=0.3, stand_still=0.0,
                             ring_advantage=1.0, push_out=1.0, win=10.0, termination=-10.0)
    elif rung == "r4probe":               # a_ walks into b_ (no stop distance); b_ holds the plateau, starts at the rim
        cfg.update(scene="scene_koth_2p_train.xml", robots=["a_", "b_"], arena=True, goal=["opponent", "center"])
        cfg["spawn"].update(r=[[0.0, 0.5], [1.25, 1.35]])
        cfg["goal_cmd"].update(stop_dist=[0.0, 0.3], vmax=[1.0, 0.8])
        cfg["projectile"].update(enabled=False)
    return cfg


class KothEnv:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda", seed: int = 0):
        """num_envs = number of worlds. self.num_envs (what rsl_rl sees) = worlds x robots."""
        self.cfg, self.device = cfg, torch.device(device)
        torch.manual_seed(seed)
        wp.init()
        spec = json.load(open(os.path.join(ASSETS, "joint_map.json")))
        self.joints, self.decimation = spec["joints"], spec["decimation"]
        self.action_scale, self.sim_dt, self.ctrl_dt = spec["action_scale"], spec["sim_dt"], spec["ctrl_dt"]
        self.num_actions = len(self.joints)
        self.max_episode_length = int(round(cfg["episode_length_s"] / self.ctrl_dt))
        self.arena = bool(cfg["arena"])
        prefixes = cfg["robots"]
        N = self.N = num_envs; A = self.A = len(prefixes); M = self.M = N * A
        self.num_envs = M

        mjm = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, cfg["scene"]))
        mjd = mujoco.MjData(mjm); mujoco.mj_resetDataKeyframe(mjm, mjd, 0); mujoco.mj_forward(mjm, mjd)
        self.mjm = mjm
        self.m = mjw.put_model(mjm, batch_sizes={k: N for k in ("body_mass", "body_inertia", "body_subtreemass", "geom_friction",
                                                                   "dof_damping", "actuator_gainprm", "actuator_biasprm")})
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
        self.robot_bodies = T([[b for b in range(mjm.nbody) if bname(b).startswith(p)] for p in prefixes])
        self.robot_geoms = T([[g for g in range(mjm.ngeom) if gname(g).startswith(p) and mjm.geom_contype[g]] for p in prefixes])
        # geom lookup: owner robot (-1 none) and kind (0 other, 1 ground, 2 left foot, 3 right foot, 4 shin)
        owner = np.full(mjm.ngeom, -1); kind = np.zeros(mjm.ngeom, dtype=np.int64)
        for n in ("floor", "arena"):
            if gid(n) >= 0: kind[gid(n)] = 1
        for a, p in enumerate(prefixes):
            for g in range(mjm.ngeom):
                if gname(g).startswith(p): owner[g] = a
            for k, s in ((2, "left"), (3, "right")):
                for n in ("foot_box_collision", "foot1_collision", "foot2_collision", "foot3_collision"): kind[gid(f"{p}{s}_{n}")] = k
                for n in ("shin_collision", "linkage_brace_collision"): kind[gid(f"{p}{s}_{n}")] = 4
        self.geom_owner = T(owner); self.geom_kind = T(kind)
        self.box_q = T([mjm.jnt_qposadr[jid(f"{b}_free")] for b in spec["pool_bodies"]])
        self.box_d = T([mjm.jnt_dofadr[jid(f"{b}_free")] for b in spec["pool_bodies"]])
        self.box_park = T(spec["pool_park"], torch.float32)
        self.box_life = torch.zeros(N, len(spec["pool_bodies"]), device=device)   # seconds left in flight; <=0 = parked
        self.hip_idx = T([i for i, j in enumerate(self.joints) if "hip_roll" in j or "hip_yaw" in j])
        self.knee_idx = T([i for i, j in enumerate(self.joints) if "knee" in j])
        self.default_pose = T(spec["default_pose"], torch.float32)
        self.key_qpos = T(mjm.key_qpos[0], torch.float32); self.key_ctrl = T(mjm.key_ctrl[0], torch.float32)
        self.key_root_z = float(spec["key_root_z"])
        self.jnt_lo = T(mjm.jnt_range[js[0], 0], torch.float32); self.jnt_hi = T(mjm.jnt_range[js[0], 1], torch.float32)
        self.ctrl_lo = T(mjm.actuator_ctrlrange[aid[0], 0], torch.float32); self.ctrl_hi = T(mjm.actuator_ctrlrange[aid[0], 1], torch.float32)
        f0 = int(self.feet_body[0, 0]); self.foot_rest_z = float(mjd.xpos[f0][2] - (0.0 if not self.arena else float(terrain_height(torch.tensor(mjd.xpos[f0][:2], dtype=torch.float32), True))))

        # ---- zero-copy torch views ----
        self.qpos = wp.to_torch(self.d.qpos); self.qvel = wp.to_torch(self.d.qvel); self.ctrl = wp.to_torch(self.d.ctrl)
        self.xpos = wp.to_torch(self.d.xpos); self.time = wp.to_torch(self.d.time)
        self.con_geom = wp.to_torch(self.d.contact.geom); self.con_world = wp.to_torch(self.d.contact.worldid)
        self.con_dist = wp.to_torch(self.d.contact.dist); self.nacon = wp.to_torch(self.d.nacon)
        self.mp = {k: wp.to_torch(getattr(self.m, k)) for k in ("body_mass", "body_inertia", "body_subtreemass", "geom_friction", "dof_damping", "actuator_gainprm", "actuator_biasprm")}
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
        self.extras: dict = {}
        self.reward_terms: dict[str, torch.Tensor] = {}
        per = lambda v: (v if isinstance(v, (list, tuple)) else [v] * A)          # scalar or one value per robot
        self.goal_is_opp = torch.tensor([g == "opponent" for g in per(cfg["goal"])], device=device).repeat(N)
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
        return q[:, :3], q[:, 3:7], v[:, :3], v[:, 3:6]

    def _opp(self, x):
        """Per-row tensor of the other robot in the same world (A == 2)."""
        return x.view(self.N, 2, *x.shape[1:]).flip(1).reshape(x.shape)

    def _joints(self):
        return self._flat(self.qpos[:, self.qadr]), self._flat(self.qvel[:, self.dadr])

    def _foot_contacts(self):
        """(M,2) bool feet on ground, (M,) bool own-leg contact (foot-foot or foot-shin of the same robot)."""
        n = int(self.nacon[0].item())
        g = self.con_geom[:n].long(); w = self.con_world[:n].long(); active = self.con_dist[:n] < 0
        k1, k2 = self.geom_kind[g[:, 0]], self.geom_kind[g[:, 1]]; o1, o2 = self.geom_owner[g[:, 0]], self.geom_owner[g[:, 1]]
        f1, f2 = (k1 == 2) | (k1 == 3), (k2 == 2) | (k2 == 3)
        feet = torch.zeros(self.M * 2, dtype=torch.bool, device=self.device)
        hit = active & f1 & (k2 == 1); feet[((w[hit] * self.A + o1[hit]) * 2 + (k1[hit] - 2))] = True
        hit = active & f2 & (k1 == 1); feet[((w[hit] * self.A + o2[hit]) * 2 + (k2[hit] - 2))] = True
        leg = torch.zeros(self.M, dtype=torch.bool, device=self.device)
        hit = active & (o1 == o2) & (o1 >= 0) & ((f1 & (f2 | (k2 == 4))) | (f2 & (f1 | (k1 == 4))))
        leg[w[hit] * self.A + o1[hit]] = True
        return feet.view(self.M, 2), leg

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
        p = pos[:, :2].view(self.N, self.A, 2); to_opp = (p.flip(1) - p).reshape(self.M, 2)
        return torch.where(self.goal_is_opp[:, None], to_opp, to_center)

    def _update_goal_commands(self):
        if self.cfg["goal"] is None: return
        pos, quat, _, _ = self._root()
        self.command[:] = goal_command(quat, self._goal_vec(pos), self.goal_stop, self.goal_vmax)

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
        return torch.stack([r * torch.cos(ang), r * torch.sin(ang)], 2)

    def reset_idx(self, ids: torch.Tensor):
        if len(ids) == 0: return
        n = len(ids); A = self.A; rn = self.cfg["reset_noise"]; rows = self._rows(ids)
        q = self.key_qpos.expand(n, -1).clone(); v = torch.zeros(n, self.qvel.shape[1], device=self.device)
        jq = q[:, self.qadr] + (torch.rand(n, A, 29, device=self.device) * 2 - 1) * rn["joint_pos"]
        q[:, self.qadr] = torch.clamp(jq, self.jnt_lo, self.jnt_hi)
        v[:, self.dadr] = (torch.rand(n, A, 29, device=self.device) * 2 - 1) * rn["joint_vel"]
        yaw = self._rand((n, A), -rn["yaw"], rn["yaw"])
        xy = self._spawn_xy(n)
        if xy is not None:
            slope = 2 * SLOPE_K * (xy.norm(dim=2) - PLATEAU_R).clamp(min=0)
            q[:, self.rq + 0] = xy[:, :, 0]; q[:, self.rq + 1] = xy[:, :, 1]
            q[:, self.rq + 2] = self.key_root_z + terrain_height(xy, self.arena) + 0.02 + 0.12 * slope
        q[:, self.rq + 3] = torch.cos(yaw / 2); q[:, self.rq + 4] = 0; q[:, self.rq + 5] = 0; q[:, self.rq + 6] = torch.sin(yaw / 2)
        for i in range(len(self.box_q)):                       # park the pool
            b = int(self.box_q[i]); q[:, b:b + 3] = self.box_park[i]; q[:, b + 3] = 1; q[:, b + 4:b + 7] = 0
        self.qpos[ids] = q; self.qvel[ids] = v; self.box_life[ids] = 0
        self.ctrl[ids] = self.key_ctrl; self.pending_ctrl[rows] = self.key_ctrl[self.aid[0]]
        self.last_action[rows] = 0; self.phase[rows] = 0; self.yaw_target[rows] = yaw.reshape(-1)
        self.feet_air_time[rows] = 0; self.feet_contact[rows] = True
        fxy = q[:, self.rq + 0], q[:, self.rq + 1]   # feet start under the pelvis; exact value only matters for one step of slip
        self.prev_feet_xy[rows] = torch.stack(fxy, 2).reshape(-1, 1, 2).expand(-1, 2, -1)
        self.episode_length_buf[ids] = 0
        self.push_timer[rows] = self._rand(len(rows), *self.cfg["push"]["interval_s"])
        self.proj_timer[ids] = self._rand(n, *self.cfg["projectile"]["interval_s"])
        if self.cfg["goal"] is None: self._resample_commands(rows)
        self._randomize(ids)

    def reset(self):
        self.reset_idx(torch.arange(self.N, device=self.device))
        self._step_sim()                                        # one control step so xpos/contacts are valid
        self.prev_feet_xy = self._flat(self.xpos[:, self.feet_body])[:, :, :2].clone()
        self._update_goal_commands()
        return self.get_observations()

    # ------------------------------------------------------------------ disturbances
    def _disturb(self):
        self.push_timer -= self.ctrl_dt; self.proj_timer -= self.ctrl_dt
        push = self.push_timer <= 0
        if push.any():
            rows = push.nonzero(as_tuple=True)[0]; n = len(rows); w = rows // self.A; a = rows % self.A
            ang = self._rand(n, 0, 2 * math.pi); mag = self._rand(n, *self.cfg["push"]["vel"])
            self.qvel[w, self.rd[a]] += mag * torch.cos(ang); self.qvel[w, self.rd[a] + 1] += mag * torch.sin(ang)
            self.push_timer[rows] = self._rand(n, *self.cfg["push"]["interval_s"])
        proj = (self.proj_timer <= 0) & self.cfg["projectile"]["enabled"]
        if proj.any():
            ids = proj.nonzero(as_tuple=True)[0]; n = len(ids)
            box = self.next_box[ids]; self.next_box[ids] = (box + 1) % len(self.box_q)
            victim = torch.randint(0, self.A, (n,), device=self.device)
            target = self.qpos[ids[:, None], self.rq7[victim][:, :3]].clone(); target[:, 2] += 0.1
            ang = self._rand(n, 0, 2 * math.pi); r = self.cfg["projectile"]["range"]
            start = target + torch.stack([r * torch.cos(ang), r * torch.sin(ang), torch.full((n,), 0.3, device=self.device)], 1)
            vel = (target - start); vel = vel / vel.norm(dim=1, keepdim=True) * self._rand(n, *self.cfg["projectile"]["speed"])[:, None]
            qa = self.box_q[box]; da = self.box_d[box]
            for k in range(3):
                self.qpos[ids, qa + k] = start[:, k]; self.qvel[ids, da + k] = vel[:, k]
            self.qpos[ids, qa + 3] = 1; self.qpos[ids, qa + 4] = 0; self.qpos[ids, qa + 5] = 0; self.qpos[ids, qa + 6] = 0
            for k in range(3, 6): self.qvel[ids, da + k] = 0
            self.box_life[ids, box] = self.cfg["projectile"]["life_s"]
            self.proj_timer[ids] = self._rand(n, *self.cfg["projectile"]["interval_s"])
        # pin parked boxes (no shelf, no contacts): park pose, zero velocity, every control step
        self.box_life -= self.ctrl_dt
        for i in range(len(self.box_q)):
            parked = self.box_life[:, i] <= 0
            qa, da = int(self.box_q[i]), int(self.box_d[i])
            self.qpos[parked, qa:qa + 3] = self.box_park[i]; self.qpos[parked, qa + 3] = 1; self.qpos[parked, qa + 4:qa + 7] = 0
            self.qvel[parked, da:da + 6] = 0

    # ------------------------------------------------------------------ step
    def _step_sim(self):
        # torch and warp use different CUDA streams: flush torch writes, run the graph, wait for it
        torch.cuda.synchronize(self.device)
        wp.capture_launch(self.graph)
        wp.synchronize_device()

    def step(self, actions: torch.Tensor):
        N, A, M = self.N, self.A, self.M
        self.action = torch.clamp(actions.to(self.device), -1, 1)
        target = torch.clamp(self.default_pose + self.action_scale * self.action, self.ctrl_lo, self.ctrl_hi)
        # one-step actuation latency on a random subset of robots (DR)
        apply = torch.where(self.latency[:, None], self.pending_ctrl, target); self.pending_ctrl = target
        self.ctrl[:, self.aid.reshape(-1)] = apply.reshape(N, A * 29)
        self._disturb()
        self._step_sim()
        self.episode_length_buf += 1
        self.phase = torch.remainder(self.phase + 2 * math.pi * GAIT_FREQ_HZ * self.ctrl_dt, 2 * math.pi)
        feet, leg = self._foot_contacts()
        pos, quat, linv, angv = self._root()
        rew = self._rewards(feet, pos, quat, linv, angv)           # uses the command the policy acted on
        gz = quat_rotate_inverse(quat, torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(M, 3))[:, 2]
        nan = (torch.isnan(self.qpos).any(1) | torch.isnan(self.qvel).any(1)).repeat_interleave(A)
        radius = pos[:, :2].norm(dim=1)
        fallen = (gz > 0) | (pos[:, 2] - terrain_height(pos[:, :2], self.arena) < 0.3) | nan
        if self.arena: fallen |= radius > self.cfg["max_radius"]
        if self.cfg["terminate_on_leg_contact"]: fallen |= leg
        timeout = self.episode_length_buf >= self.max_episode_length
        world_done = fallen.view(N, A).any(1) | timeout
        done = world_done.repeat_interleave(A)
        term = self.cfg["reward"]["termination"] * fallen.float()      # one-off, outside the clip, not scaled by dt
        rew += term; self.reward_terms["termination"] = term
        if self.A == 2 and self.cfg["reward"]["win"] != 0.0:
            won = self._opp(fallen) & ~fallen; win = self.cfg["reward"]["win"] * won.float()
            rew += win; self.reward_terms["win"] = win
        self.extras = {"time_outs": done & ~fallen, "log": {f"rew/{k}": v.mean() for k, v in self.reward_terms.items()},
                       "fallen": fallen, "nan": nan, "radius": radius}
        self.extras["log"]["ep/leg_contact_term"] = leg.float().mean(); self.extras["log"]["ep/fallen"] = fallen.float().mean()
        if self.arena: self.extras["log"]["ep/on_plateau"] = (radius < PLATEAU_R).float().mean()
        if self.A == 2:
            self.extras["log"]["ep/decided"] = (fallen.view(N, A).any(1).float().sum() / world_done.float().sum().clamp(min=1))
            self.extras["log"]["ep/separation"] = (pos[:, :2] - self._opp(pos)[:, :2]).norm(dim=1).mean()
        self.last_action = self.action.clone()
        self.feet_contact = feet; self.prev_feet_xy = self._flat(self.xpos[:, self.feet_body])[:, :, :2].clone()
        if self.cfg["goal"] is None:
            resample = (self.episode_length_buf % int(self.cfg["command"]["resample_s"] / self.ctrl_dt) == 0).nonzero(as_tuple=True)[0]
            if len(resample): self._resample_commands(self._rows(resample))
        self.reset_idx(world_done.nonzero(as_tuple=True)[0])
        self._update_goal_commands()
        obs = self.get_observations()
        self.extras["observations"] = obs
        return obs, rew, done, self.extras

    # ------------------------------------------------------------------ rewards
    def _rewards(self, feet, pos, quat, linv_w, angv):
        R = self.cfg["reward"]; dt = self.ctrl_dt
        jpos, jvel = self._joints()
        obs = build_obs(quat, linv_w, angv, jpos, jvel, self.default_pose, self.last_action, self.command, self.phase)
        linv, grav = obs[:, 0:3], obs[:, 6:9]
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
        fpos = self._flat(self.xpos[:, self.feet_body]); foot_xy = fpos[:, :, :2]
        foot_z = fpos[:, :, 2] - terrain_height(foot_xy, self.arena) - self.foot_rest_z
        first_contact = feet & ~self.feet_contact
        self.feet_air_time += dt
        t["feet_air_time"] = ((self.feet_air_time - 0.1) * first_contact.float()).sum(1) * cmd_on
        self.feet_air_time[feet] = 0
        slip = ((foot_xy - self.prev_feet_xy) / dt).norm(dim=2)
        t["feet_slip"] = (slip * feet.float()).sum(1)
        des = torch.stack([torch.sin(self.phase), torch.sin(self.phase + math.pi)], 1).clamp(min=0) * self.cfg["swing_height"]
        t["feet_phase"] = torch.exp(-((foot_z - des) ** 2).sum(1) / 0.01) * cmd_on
        # posture
        dev = jpos - self.default_pose
        t["pose"] = (dev ** 2).sum(1)
        t["joint_deviation_hip"] = dev[:, self.hip_idx].abs().sum(1)
        t["joint_deviation_knee"] = dev[:, self.knee_idx].abs().sum(1)
        t["dof_pos_limits"] = ((self.jnt_lo + 0.05 - jpos).clamp(min=0) + (jpos - self.jnt_hi + 0.05).clamp(min=0)).sum(1)
        t["action_rate"] = ((self.action - self.last_action) ** 2).sum(1)
        t["stand_still"] = dev.abs().sum(1) * (1 - cmd_on)
        t["alive"] = torch.ones_like(cmd_on)
        t["upright"] = torch.exp(-(grav[:, :2] ** 2).sum(1) / 0.05)
        t["joint_vel"] = (jvel ** 2).sum(1)
        radius = pos[:, :2].norm(dim=1)
        if self.A == 2:
            opos = self._opp(pos); ovel = self._opp(linv_w); orad = opos[:, :2].norm(dim=1)
            near = ((opos[:, :2] - pos[:, :2]).norm(dim=1) < 0.9).float()
            t["ring_advantage"] = (orad - radius).clamp(-1.0, 1.0)                     # be more central than the opponent
            t["push_out"] = ((ovel[:, :2] * opos[:, :2]).sum(1) / orad.clamp(min=0.1)).clamp(-1.0, 2.0) * near   # its outward speed while I am on it
        t["plateau"] = (radius < 1.2).float()
        t["off_rim"] = (radius - PLATEAU_R).clamp(min=0)
        self.reward_terms = {k: R[k] * v * dt for k, v in t.items() if R.get(k, 0.0) != 0.0}
        # Clip the per-step sum at zero (as mujoco_playground does). Without it the penalties outweigh the
        # tracking terms early on and the policy learns to end the episode fast: run r0_a went from episode
        # length 25 to 4.8 steps in 20 iterations while "reward" rose.
        return torch.stack(list(self.reward_terms.values()), 0).sum(0).clamp(min=0.0)

    # ------------------------------------------------------------------ observations
    def get_observations(self) -> TensorDict:
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
        return TensorDict({"policy": noisy, "critic": critic}, batch_size=[self.M])


if __name__ == "__main__":
    import time, sys
    rung = sys.argv[1] if len(sys.argv) > 1 else "r0"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    env = KothEnv(default_cfg(rung), N)
    obs = env.get_observations(); assert obs["policy"].shape == (env.num_envs, OBS_DIM + (COMBAT_DIM if env.cfg["combat"] else 0)), obs["policy"].shape
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
