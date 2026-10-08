"""Vectorized G1 environment on mujoco_warp with a torch front-end. Implements the rsl_rl VecEnv contract.

Rungs R0 (stand + recover) and R1 (walk/turn) share this flat-ground env; the command distribution is the only
difference. Observation layout lives in koth/obs.py and is mirrored by Unity's ObsBuilder.cs.
"""
from __future__ import annotations
import json, math, os
import numpy as np, torch, mujoco, warp as wp, mujoco_warp as mjw
from tensordict import TensorDict
from koth.obs import build_obs, quat_rotate_inverse, OBS_DIM, GAIT_FREQ_HZ

ASSETS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", "g1"))


def default_cfg(rung: str = "r0") -> dict:
    cfg = dict(
        scene="scene_flat_1p_train.xml", robot="a_", episode_length_s=20.0,
        command=dict(lin_vel_x=[-1.0, 1.0], lin_vel_y=[-0.5, 0.5], ang_vel_yaw=[-1.0, 1.0], zero_prob=0.1, resample_s=10.0),
        push=dict(interval_s=[5.0, 10.0], vel=[0.5, 2.0]),
        projectile=dict(enabled=True, interval_s=[3.0, 8.0], speed=[3.0, 8.0], range=3.0),
        dr=dict(mass=0.15, friction=[0.4, 0.9], kp=0.2, damping=0.2, latency_prob=0.5),
        noise=dict(joint_pos=0.03, joint_vel=1.5, gravity=0.05, linvel=0.1, gyro=0.2),
        reset_noise=dict(joint_pos=0.05, joint_vel=0.3, yaw=math.pi),
        reward=dict(tracking_lin_vel=1.0, tracking_ang_vel=0.75, orientation=-2.0, ang_vel_xy=-0.15, lin_vel_z=-0.5,
                    feet_air_time=2.0, feet_slip=-0.25, feet_phase=1.0, pose=-0.1, joint_deviation_hip=-0.25,
                    joint_deviation_knee=-0.1, dof_pos_limits=-1.0, action_rate=-0.01, stand_still=-1.0,
                    alive=0.0, termination=-100.0),
        tracking_sigma=0.25, swing_height=0.12, terminate_on_leg_contact=True,
    )
    if rung == "r0":                       # stand only: zero command, disturbances on
        cfg["command"].update(zero_prob=1.0)
    return cfg


class KothEnv:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda", seed: int = 0):
        self.cfg, self.num_envs, self.device = cfg, num_envs, torch.device(device)
        torch.manual_seed(seed)
        wp.init()
        spec = json.load(open(os.path.join(ASSETS, "joint_map.json")))
        self.joints, self.decimation = spec["joints"], spec["decimation"]
        self.action_scale, self.sim_dt, self.ctrl_dt = spec["action_scale"], spec["sim_dt"], spec["ctrl_dt"]
        self.num_actions = len(self.joints)
        self.max_episode_length = int(round(cfg["episode_length_s"] / self.ctrl_dt))

        mjm = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, cfg["scene"]))
        mjd = mujoco.MjData(mjm); mujoco.mj_resetDataKeyframe(mjm, mjd, 0); mujoco.mj_forward(mjm, mjd)
        self.mjm = mjm
        N = num_envs
        self.m = mjw.put_model(mjm, batch_sizes={k: N for k in ("body_mass", "body_inertia", "body_subtreemass", "geom_friction",
                                                                   "dof_damping", "actuator_gainprm", "actuator_biasprm")})
        self.m.opt.warn_overflow = 0
        self.d = mjw.put_data(mjm, mjd, nworld=N, nconmax=64, njmax=400)

        # ---- indices (name based, same resolution rule as Unity's JointMap) ----
        p = cfg["robot"]
        jid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, n)
        gid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)
        bid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)
        js = [jid(p + j) for j in self.joints]; assert min(js) >= 0, "joint names missing"
        self.qadr = torch.tensor([mjm.jnt_qposadr[j] for j in js], device=device)
        self.dadr = torch.tensor([mjm.jnt_dofadr[j] for j in js], device=device)
        self.aid = torch.tensor([mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, p + j) for j in self.joints], device=device)
        root = jid(p + "floating_base_joint"); self.rq, self.rd = int(mjm.jnt_qposadr[root]), int(mjm.jnt_dofadr[root])
        self.pelvis = bid(p + "pelvis")
        self.feet_body = torch.tensor([bid(p + "left_ankle_roll_link"), bid(p + "right_ankle_roll_link")], device=device)
        self.robot_bodies = torch.tensor([b for b in range(mjm.nbody) if (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(p)], device=device)
        self.robot_geoms = torch.tensor([g for g in range(mjm.ngeom) if (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(p) and mjm.geom_contype[g]], device=device)
        ground = [gid(n) for n in ("floor", "arena") if gid(n) >= 0]
        foot_geoms = [[gid(f"{p}{s}_foot_box_collision"), gid(f"{p}{s}_foot1_collision"), gid(f"{p}{s}_foot2_collision"), gid(f"{p}{s}_foot3_collision")] for s in ("left", "right")]
        self.geom_class = torch.zeros(mjm.ngeom, dtype=torch.long, device=device)   # 0 other, 1 ground, 2 left foot, 3 right foot, 4 shin
        self.geom_class[ground] = 1
        self.geom_class[foot_geoms[0]] = 2; self.geom_class[foot_geoms[1]] = 3
        self.geom_class[[gid(f"{p}{s}_{n}") for s in ("left", "right") for n in ("shin_collision", "linkage_brace_collision")]] = 4
        self.box_q = torch.tensor([mjm.jnt_qposadr[jid(f"{b}_free")] for b in spec["pool_bodies"]], device=device)
        self.box_d = torch.tensor([mjm.jnt_dofadr[jid(f"{b}_free")] for b in spec["pool_bodies"]], device=device)
        self.box_park = torch.tensor([[-1.75 + 0.5 * i, 0.0, -50.0] for i in range(len(spec["pool_bodies"]))], device=device)
        hip_idx = [i for i, j in enumerate(self.joints) if "hip_roll" in j or "hip_yaw" in j]
        self.hip_idx = torch.tensor(hip_idx, device=device); self.knee_idx = torch.tensor([i for i, j in enumerate(self.joints) if "knee" in j], device=device)
        self.default_pose = torch.tensor(spec["default_pose"], device=device)
        key_q = torch.tensor(mjm.key_qpos[0], device=device, dtype=torch.float32)
        self.key_qpos = key_q; self.key_ctrl = torch.tensor(mjm.key_ctrl[0], device=device, dtype=torch.float32)
        self.jnt_lo = torch.tensor(mjm.jnt_range[js, 0], device=device, dtype=torch.float32); self.jnt_hi = torch.tensor(mjm.jnt_range[js, 1], device=device, dtype=torch.float32)
        self.ctrl_lo = torch.tensor(mjm.actuator_ctrlrange[self.aid.cpu().numpy(), 0], device=device, dtype=torch.float32)
        self.ctrl_hi = torch.tensor(mjm.actuator_ctrlrange[self.aid.cpu().numpy(), 1], device=device, dtype=torch.float32)
        self.foot_rest_z = float(mjd.xpos[int(self.feet_body[0])][2])

        # ---- zero-copy torch views ----
        self.qpos = wp.to_torch(self.d.qpos); self.qvel = wp.to_torch(self.d.qvel); self.ctrl = wp.to_torch(self.d.ctrl)
        self.xpos = wp.to_torch(self.d.xpos); self.time = wp.to_torch(self.d.time)
        self.con_geom = wp.to_torch(self.d.contact.geom); self.con_world = wp.to_torch(self.d.contact.worldid)
        self.con_dist = wp.to_torch(self.d.contact.dist); self.nacon = wp.to_torch(self.d.nacon)
        self.mp = {k: wp.to_torch(getattr(self.m, k)) for k in ("body_mass", "body_inertia", "body_subtreemass", "geom_friction", "dof_damping", "actuator_gainprm", "actuator_biasprm")}
        self.mp0 = {k: v.clone() for k, v in self.mp.items()}

        # ---- buffers ----
        z = lambda *s: torch.zeros(*s, device=device)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=device)
        self.command = z(N, 3); self.last_action = z(N, self.num_actions); self.prev_action = z(N, self.num_actions)
        self.action = z(N, self.num_actions); self.phase = z(N)
        self.feet_air_time = z(N, 2); self.feet_contact = torch.zeros(N, 2, dtype=torch.bool, device=device); self.prev_feet_xy = z(N, 2, 2)
        self.push_timer = z(N); self.proj_timer = z(N); self.next_box = torch.zeros(N, dtype=torch.long, device=device)
        self.dr_params = z(N, 3)   # mass scale, friction, kp scale  (critic obs)
        self.latency = torch.zeros(N, dtype=torch.bool, device=device); self.pending_ctrl = z(N, self.num_actions)
        self.extras: dict = {}
        self.reward_terms: dict[str, torch.Tensor] = {}

        mjw.step(self.m, self.d)                        # warm up kernels before capture
        with wp.ScopedCapture() as cap:
            for _ in range(self.decimation):
                mjw.step(self.m, self.d)
        self.graph = cap.graph
        self.reset()

    # ------------------------------------------------------------------ helpers
    def _rand(self, n, lo, hi): return torch.rand(n, device=self.device) * (hi - lo) + lo

    def _root(self):
        q = self.qpos[:, self.rq:self.rq + 7]; v = self.qvel[:, self.rd:self.rd + 6]
        return q[:, :3], q[:, 3:7], v[:, :3], v[:, 3:6]

    def _foot_contacts(self):
        """(N,2) bool feet-ground, (N,) bool leg self-contact (foot-foot or foot-shin)."""
        n = int(self.nacon[0].item())
        g = self.con_geom[:n].long(); w = self.con_world[:n].long(); active = self.con_dist[:n] < 0
        c1, c2 = self.geom_class[g[:, 0]], self.geom_class[g[:, 1]]
        feet = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        for foot, cls in ((0, 2), (1, 3)):
            hit = active & (((c1 == cls) & (c2 == 1)) | ((c2 == cls) & (c1 == 1)))
            feet[:, foot] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device).index_put_((w[hit],), torch.ones_like(w[hit], dtype=torch.bool))
        isfoot1, isfoot2 = (c1 == 2) | (c1 == 3), (c2 == 2) | (c2 == 3)
        legcon = active & ((isfoot1 & ((c2 == 4) | isfoot2)) | (isfoot2 & ((c1 == 4) | isfoot1)))
        leg = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device).index_put_((w[legcon],), torch.ones_like(w[legcon], dtype=torch.bool))
        return feet, leg

    # ------------------------------------------------------------------ reset / randomization
    def _resample_commands(self, ids):
        c = self.cfg["command"]
        n = len(ids)
        cmd = torch.stack([self._rand(n, *c["lin_vel_x"]), self._rand(n, *c["lin_vel_y"]), self._rand(n, *c["ang_vel_yaw"])], 1)
        cmd[torch.rand(n, device=self.device) < c["zero_prob"]] = 0.0
        self.command[ids] = cmd

    def _randomize(self, ids):
        dr = self.cfg["dr"]; n = len(ids)
        ms = 1 + self._rand(n, -dr["mass"], dr["mass"]); fr = self._rand(n, *dr["friction"]); ks = 1 + self._rand(n, -dr["kp"], dr["kp"])
        ds = 1 + self._rand(n, -dr["damping"], dr["damping"])
        rb = self.robot_bodies
        for k in ("body_mass", "body_subtreemass"):
            self.mp[k][ids[:, None], rb] = self.mp0[k][ids[:, None], rb] * ms[:, None]
        self.mp["body_inertia"][ids[:, None], rb] = self.mp0["body_inertia"][ids[:, None], rb] * ms[:, None, None]
        self.mp["geom_friction"][ids[:, None], self.robot_geoms, 0] = fr[:, None]
        self.mp["dof_damping"][ids[:, None], self.dadr] = self.mp0["dof_damping"][ids[:, None], self.dadr] * ds[:, None]
        self.mp["actuator_gainprm"][ids[:, None], self.aid, 0] = self.mp0["actuator_gainprm"][ids[:, None], self.aid, 0] * ks[:, None]
        self.mp["actuator_biasprm"][ids[:, None], self.aid, 1] = self.mp0["actuator_biasprm"][ids[:, None], self.aid, 1] * ks[:, None]
        self.dr_params[ids] = torch.stack([ms, fr, ks], 1)
        self.latency[ids] = torch.rand(n, device=self.device) < dr["latency_prob"]

    def reset_idx(self, ids: torch.Tensor):
        if len(ids) == 0: return
        n = len(ids); rn = self.cfg["reset_noise"]
        q = self.key_qpos.expand(n, -1).clone(); v = torch.zeros(n, self.qvel.shape[1], device=self.device)
        q[:, self.qadr] += (torch.rand(n, self.num_actions, device=self.device) * 2 - 1) * rn["joint_pos"]
        q[:, self.qadr] = torch.clamp(q[:, self.qadr], self.jnt_lo, self.jnt_hi)
        v[:, self.dadr] = (torch.rand(n, self.num_actions, device=self.device) * 2 - 1) * rn["joint_vel"]
        yaw = self._rand(n, -rn["yaw"], rn["yaw"])
        q[:, self.rq + 3] = torch.cos(yaw / 2); q[:, self.rq + 4:self.rq + 6] = 0; q[:, self.rq + 6] = torch.sin(yaw / 2)
        for i in range(len(self.box_q)):                       # park the pool
            q[:, self.box_q[i]:self.box_q[i] + 3] = self.box_park[i]; q[:, self.box_q[i] + 3] = 1; q[:, self.box_q[i] + 4:self.box_q[i] + 7] = 0
        self.qpos[ids] = q; self.qvel[ids] = v
        self.ctrl[ids] = self.key_ctrl; self.pending_ctrl[ids] = self.key_ctrl[self.aid]
        self.last_action[ids] = 0; self.prev_action[ids] = 0; self.phase[ids] = 0
        self.feet_air_time[ids] = 0; self.feet_contact[ids] = True
        self.prev_feet_xy[ids] = self.xpos[ids][:, self.feet_body, :2]
        self.episode_length_buf[ids] = 0
        self.push_timer[ids] = self._rand(n, *self.cfg["push"]["interval_s"])
        self.proj_timer[ids] = self._rand(n, *self.cfg["projectile"]["interval_s"])
        self._resample_commands(ids); self._randomize(ids)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._step_sim()                                        # one control step so xpos/contacts are valid
        return self.get_observations()

    # ------------------------------------------------------------------ disturbances
    def _disturb(self):
        self.push_timer -= self.ctrl_dt; self.proj_timer -= self.ctrl_dt
        push = self.push_timer <= 0
        if push.any():
            n = int(push.sum()); ang = self._rand(n, 0, 2 * math.pi); mag = self._rand(n, *self.cfg["push"]["vel"])
            self.qvel[push, self.rd] += mag * torch.cos(ang); self.qvel[push, self.rd + 1] += mag * torch.sin(ang)
            self.push_timer[push] = self._rand(n, *self.cfg["push"]["interval_s"])
        proj = (self.proj_timer <= 0) & self.cfg["projectile"]["enabled"]
        if proj.any():
            ids = proj.nonzero(as_tuple=True)[0]; n = len(ids)
            box = self.next_box[ids]; self.next_box[ids] = (box + 1) % len(self.box_q)
            pos, _, _, _ = self._root(); target = pos[ids].clone(); target[:, 2] += 0.1
            ang = self._rand(n, 0, 2 * math.pi); r = self.cfg["projectile"]["range"]
            start = target + torch.stack([r * torch.cos(ang), r * torch.sin(ang), torch.full((n,), 0.3, device=self.device)], 1)
            vel = (target - start); vel = vel / vel.norm(dim=1, keepdim=True) * self._rand(n, *self.cfg["projectile"]["speed"])[:, None]
            qa = self.box_q[box]; da = self.box_d[box]
            for k in range(3):
                self.qpos[ids, qa + k] = start[:, k]; self.qvel[ids, da + k] = vel[:, k]
            self.qpos[ids, qa + 3] = 1; self.qpos[ids, qa + 4] = 0; self.qpos[ids, qa + 5] = 0; self.qpos[ids, qa + 6] = 0
            for k in range(3, 6): self.qvel[ids, da + k] = 0
            self.proj_timer[ids] = self._rand(n, *self.cfg["projectile"]["interval_s"])

    # ------------------------------------------------------------------ step
    def _step_sim(self):
        # torch and warp use different CUDA streams: flush torch writes, run the graph, wait for it
        torch.cuda.synchronize(self.device)
        wp.capture_launch(self.graph)
        wp.synchronize_device()

    def step(self, actions: torch.Tensor):
        self.action = torch.clamp(actions.to(self.device), -1, 1)
        target = torch.clamp(self.default_pose + self.action_scale * self.action, self.ctrl_lo, self.ctrl_hi)
        # one-step actuation latency on a random subset of envs (DR)
        apply = torch.where(self.latency[:, None], self.pending_ctrl, target); self.pending_ctrl = target
        self.ctrl[:, self.aid] = apply
        self._disturb()
        self._step_sim()
        self.episode_length_buf += 1
        self.phase = torch.remainder(self.phase + 2 * math.pi * GAIT_FREQ_HZ * self.ctrl_dt, 2 * math.pi)
        feet, leg = self._foot_contacts()
        rew = self._rewards(feet)
        pos, quat, linv, angv = self._root()
        gz = quat_rotate_inverse(quat, torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3))[:, 2]
        nan = torch.isnan(self.qpos).any(1) | torch.isnan(self.qvel).any(1)
        fallen = (gz > 0) | (pos[:, 2] < 0.3) | nan
        if self.cfg["terminate_on_leg_contact"]: fallen |= leg
        timeout = self.episode_length_buf >= self.max_episode_length
        done = fallen | timeout
        rew += self.cfg["reward"]["termination"] * fallen.float() * self.ctrl_dt
        self.reward_terms["termination"] = self.cfg["reward"]["termination"] * fallen.float() * self.ctrl_dt
        self.extras = {"time_outs": timeout, "log": {f"rew/{k}": v.mean() for k, v in self.reward_terms.items()},
                       "fallen": fallen, "nan": nan}
        self.extras["log"]["ep/leg_contact_term"] = leg.float().mean(); self.extras["log"]["ep/fallen"] = fallen.float().mean()
        self.prev_action = self.last_action.clone(); self.last_action = self.action.clone()
        self.feet_contact = feet; self.prev_feet_xy = self.xpos[:, self.feet_body, :2].clone()
        resample = (self.episode_length_buf % int(self.cfg["command"]["resample_s"] / self.ctrl_dt) == 0).nonzero(as_tuple=True)[0]
        if len(resample): self._resample_commands(resample)
        self.reset_idx(done.nonzero(as_tuple=True)[0])
        obs = self.get_observations()
        self.extras["observations"] = obs
        return obs, rew, done, self.extras

    # ------------------------------------------------------------------ rewards
    def _rewards(self, feet):
        R = self.cfg["reward"]; dt = self.ctrl_dt
        pos, quat, linv_w, angv = self._root()
        jpos = self.qpos[:, self.qadr]; jvel = self.qvel[:, self.dadr]
        obs = build_obs(quat, linv_w, angv, jpos, jvel, self.default_pose, self.last_action, self.command, self.phase)
        linv, grav = obs[:, 0:3], obs[:, 6:9]
        cmd = self.command; cmd_on = (cmd.norm(dim=1) > 0.01).float()
        s = self.cfg["tracking_sigma"]
        t = {}
        t["tracking_lin_vel"] = torch.exp(-((cmd[:, :2] - linv[:, :2]) ** 2).sum(1) / s)
        t["tracking_ang_vel"] = torch.exp(-((cmd[:, 2] - angv[:, 2]) ** 2) / s)
        t["orientation"] = (grav[:, :2] ** 2).sum(1)
        t["ang_vel_xy"] = (angv[:, :2] ** 2).sum(1)
        t["lin_vel_z"] = linv[:, 2] ** 2
        # feet
        foot_xy = self.xpos[:, self.feet_body, :2]; foot_z = self.xpos[:, self.feet_body, 2] - self.foot_rest_z
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
        self.reward_terms = {k: R[k] * v * dt for k, v in t.items()}
        return torch.stack(list(self.reward_terms.values()), 0).sum(0)

    # ------------------------------------------------------------------ observations
    def get_observations(self) -> TensorDict:
        pos, quat, linv_w, angv = self._root()
        jpos = self.qpos[:, self.qadr]; jvel = self.qvel[:, self.dadr]
        clean = build_obs(quat, linv_w, angv, jpos, jvel, self.default_pose, self.last_action, self.command, self.phase)
        nz = self.cfg["noise"]; u = lambda n, sc: (torch.rand(self.num_envs, n, device=self.device) * 2 - 1) * sc
        noisy = clean.clone()
        noisy[:, 0:3] += u(3, nz["linvel"]); noisy[:, 3:6] += u(3, nz["gyro"]); noisy[:, 6:9] += u(3, nz["gravity"])
        noisy[:, 12:41] += u(29, nz["joint_pos"]); noisy[:, 41:70] += u(29, nz["joint_vel"])
        critic = torch.cat([clean, linv_w, pos[:, 2:3], self.feet_contact.float(), self.dr_params], 1)
        return TensorDict({"policy": noisy, "critic": critic}, batch_size=[self.num_envs])

    # state access for recording / parity
    def robot_state(self):
        pos, quat, linv, angv = self._root()
        return dict(root_pos=pos, root_quat=quat, root_linvel=linv, root_angvel=angv, joint_pos=self.qpos[:, self.qadr], joint_vel=self.qvel[:, self.dadr], ctrl=self.ctrl[:, self.aid])


if __name__ == "__main__":
    import time, sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    env = KothEnv(default_cfg("r0"), N)
    obs = env.get_observations(); assert obs["policy"].shape == (N, OBS_DIM), obs["policy"].shape
    torch.cuda.synchronize(); t = time.perf_counter(); S = 200; falls = 0
    for i in range(S):
        obs, rew, done, extras = env.step(torch.zeros(N, env.num_actions, device=env.device))
        falls += int(extras["fallen"].sum())
    torch.cuda.synchronize(); el = time.perf_counter() - t
    print(f"{N} envs: {N * S / el:,.0f} ctrl-steps/s ({N * S * env.decimation / el:,.0f} sim-steps/s); zero-action falls over {S * env.ctrl_dt:.0f}s: {falls} "
          f"({falls / N:.2f}/env, pushes+boxes on); mean rew {rew.mean():.3f}; nan {int(extras['nan'].sum())}")
    assert not torch.isnan(obs["policy"]).any()
