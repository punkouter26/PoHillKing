"""Observation layout shared by training (torch) and Unity (ObsBuilder.cs). Keep both in sync by hand; the
replay parity gate (reference_trajectory.json) catches drift.

Locomotion state, 103 dims, in this order:
  [0:3]    pelvis linear velocity, pelvis frame
  [3:6]    pelvis angular velocity, pelvis frame   (free-joint qvel[3:6] is already body-frame in MuJoCo)
  [6:9]    gravity direction in pelvis frame        (R^T * (0,0,-1))
  [9:12]   command (vx, vy, yaw rate)
  [12:41]  joint pos - default pose                 (canonical 29 order)
  [41:70]  joint vel
  [70:99]  last action
  [99:103] gait phase: cos/sin for left leg, cos/sin for right leg (right = left + pi)
"""
import torch

NUM_JOINTS = 29
OBS_DIM = 103
GAIT_FREQ_HZ = 1.5
# Shove style constants (used by env.py, duel_stats.py and mirrored in Unity's PolicyRunner).
SHOVE_POSE = {"shoulder_pitch": -1.0, "elbow": 0.5}      # hand 0.34 m ahead of the torso, 0.24 m above its origin
SHOVE_FAR, SHOVE_NEAR, SHOVE_SMOOTH = 1.6, 1.0, 0.5
# Trump's bull charge: the same mechanism with another target pose. Torso pitched 0.3 rad forward, arms swept back with
# the elbows tucked, knees lower (hip and ankle keep the feet flat): he leads with his chest and belly. Held with no
# brain on his body it stands with the torso 21 degrees forward (the hips and ankles also sit him back over his heels;
# hip -0.22 / ankle -0.18 fell forward).
BULL_POSE = {"waist_pitch": 0.30, "shoulder_pitch": 0.35, "elbow": 0.60, "knee": 0.40, "hip_pitch": -0.30, "ankle_pitch": -0.10}
STYLE_POSES = {"shove": SHOVE_POSE, "bull": BULL_POSE}


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vector v into the frame of unit quaternion q = (w, x, y, z). Shapes (N,4), (N,3)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v - w * t + torch.cross(xyz, t, dim=1)


def down_like(root_quat: torch.Tensor) -> torch.Tensor:
    """(N,3) world down vector (0, 0, -1) on root_quat's device and dtype, made by fill kernels (no host copy)."""
    g = root_quat.new_zeros(root_quat.shape[0], 3); g[:, 2] = -1.0
    return g


def build_obs(root_quat, root_linvel_world, root_angvel_body, jpos, jvel, default_pose, last_action, command, phase):
    """All tensors batched (N, ...). phase in radians (N,)."""
    g = down_like(root_quat)
    return torch.cat([
        quat_rotate_inverse(root_quat, root_linvel_world),
        root_angvel_body,
        quat_rotate_inverse(root_quat, g),
        command,
        jpos - default_pose,
        jvel,
        last_action,
        torch.stack([torch.cos(phase), torch.sin(phase), torch.cos(phase + torch.pi), torch.sin(phase + torch.pi)], dim=1),
    ], dim=1)


COMBAT_DIM = 9


def build_combat(root_pos, root_quat, root_linvel_w, opp_pos, opp_quat, opp_linvel_w):
    """Opponent and ring awareness appended to the 103-dim state for the combat rungs (r4+). All (M, ...).
      [0:2] opponent position relative to me, in my heading frame (m)
      [2:4] opponent velocity relative to me, in my heading frame (m/s)
      [4:6] plateau centre relative to me, in my heading frame (m)
      [6]   my distance from the centre (m)      [7] opponent distance from the centre (m)
      [8]   opponent uprightness (world-z component of its pelvis z axis, 1 = upright)
    Heading frame = world frame rotated by my yaw only, so the block stays meaningful when I am tilted."""
    w, x, y, z = root_quat.unbind(1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)); c, s = torch.cos(yaw), torch.sin(yaw)
    def to_heading(v):
        return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], 1)
    opp_up = 1 - 2 * (opp_quat[:, 1] ** 2 + opp_quat[:, 2] ** 2)
    return torch.cat([to_heading(opp_pos[:, :2] - root_pos[:, :2]), to_heading(opp_linvel_w[:, :2] - root_linvel_w[:, :2]),
                      to_heading(-root_pos[:, :2]), root_pos[:, :2].norm(dim=1, keepdim=True), opp_pos[:, :2].norm(dim=1, keepdim=True),
                      opp_up[:, None]], 1)


def goal_command(root_quat, goal_vec_xy, stop_dist: float, vmax: float = 0.8, self_radius=None, rim_brake=None):
    """Velocity command (vx, vy, yaw rate) that walks toward a goal. Fixed law, mirrored by Unity's GoalCommand.cs.
    root_quat (M,4) wxyz, goal_vec_xy (M,2) world vector from the robot to the goal.
      speed  = clip(dist - stop_dist, 0, vmax)
      ang    = bearing of the goal in the robot's heading frame
      vx, vy = speed * cos(ang), speed * sin(ang)   (clipped to the trained ranges +-1, +-0.5)
      yaw    = clip(2 * ang, -1, 1) while moving, else 0
    Rim brake (optional, rim_brake = (rim, gain, vmin) with self_radius (M,) = my distance from the centre):
      speed  = min(speed, clip(gain * (rim - self_radius), vmin, inf))   so a robot near the edge does not charge off it."""
    w, x, y, z = root_quat.unbind(1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    dist = goal_vec_xy.norm(dim=1)
    ang = torch.atan2(goal_vec_xy[:, 1], goal_vec_xy[:, 0]) - yaw
    ang = torch.atan2(torch.sin(ang), torch.cos(ang))
    speed = torch.minimum((dist - stop_dist).clamp(min=0.0), torch.as_tensor(vmax, dtype=dist.dtype, device=dist.device))
    if rim_brake is not None and self_radius is not None:
        rim, gain, vmin = rim_brake
        speed = torch.minimum(speed, (gain * (rim - self_radius)).clamp(min=vmin))
    moving = (speed > 0.05).to(speed.dtype)
    return torch.stack([(speed * torch.cos(ang)).clamp(-1.0, 1.0), (speed * torch.sin(ang)).clamp(-0.5, 0.5),
                        (2.0 * ang).clamp(-1.0, 1.0) * moving], dim=1)


# Tactics: goal points other than "straight at the opponent", mirrored from Unity's Tactics.Goal (index = C# enum Tactic).
# They only move the goal point fed to goal_command, so the commands stay inside the trained ranges.
TACTICS = ("press", "flank_left", "flank_right", "regroup", "lurk")
FLANK_OFFSET, LURK_RADIUS = 0.55, 0.6


def tactic_goal(tactic, me_xy, opp_xy, base_stop):
    """World xy vector from me to the goal point of each row's tactic (M,) long, and its stop distance (M,).
    Press = (opp - me, base_stop), as trained; Flank = a point FLANK_OFFSET beside the opponent (stop 0); Regroup = the
    summit centre (stop 0.2); Lurk = a spot on my own bearing at LURK_RADIUS from the centre (stop 0.1)."""
    d = opp_xy - me_xy; n = d.norm(dim=1, keepdim=True) + 1e-6; perp = torch.stack([-d[:, 1], d[:, 0]], 1) / n
    side = torch.where(tactic == 1, 1.0, -1.0)[:, None]
    flank = d + side * perp * FLANK_OFFSET
    r = me_xy.norm(dim=1, keepdim=True)
    lurk = torch.where(r < 0.05, torch.zeros_like(me_xy), me_xy / r.clamp(min=1e-6) * LURK_RADIUS - me_xy)
    t = tactic[:, None]
    goal = torch.where(t == 0, d, torch.where(t <= 2, flank, torch.where(t == 3, -me_xy, lurk)))
    stop = torch.where(tactic == 0, base_stop, torch.where(tactic <= 2, 0.0, torch.where(tactic == 3, 0.2, 0.1)))
    return goal, stop


if __name__ == "__main__":
    tg, ts = tactic_goal(torch.tensor([0, 1, 2, 3, 4]), torch.tensor([[1.0, 0.0]] * 5), torch.tensor([[1.0, 1.0]] * 5), torch.full((5,), 0.3))
    assert torch.allclose(tg[0], torch.tensor([0.0, 1.0])) and torch.allclose(tg[1], torch.tensor([-0.55, 1.0]), atol=1e-5), tg
    assert torch.allclose(tg[2], torch.tensor([0.55, 1.0]), atol=1e-5) and torch.allclose(tg[3], torch.tensor([-1.0, 0.0])), tg
    assert torch.allclose(tg[4], torch.tensor([-0.4, 0.0]), atol=1e-6) and torch.allclose(ts, torch.tensor([0.3, 0.0, 0.0, 0.2, 0.1])), (tg, ts)
    c = goal_command(torch.tensor([[1.0, 0, 0, 0], [0.7071068, 0, 0, 0.7071068]]), torch.tensor([[2.0, 0.0], [2.0, 0.0]]), 0.3)
    assert torch.allclose(c[0], torch.tensor([0.8, 0.0, 0.0]), atol=1e-6), c      # goal straight ahead
    assert c[1, 0].abs() < 1e-6 and c[1, 1] < -0.49 and c[1, 2] == -1.0, c          # goal to the right: sidestep + turn right
    q = torch.tensor([[1.0, 0, 0, 0], [0.7071068, 0, 0, 0.7071068]])   # identity, 90 deg yaw
    v = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]])
    r = quat_rotate_inverse(q, v)
    assert torch.allclose(r[0], v[0]) and torch.allclose(r[1], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6), r
    o = build_obs(q, v, v, torch.zeros(2, 29), torch.zeros(2, 29), torch.zeros(29), torch.zeros(2, 29), torch.zeros(2, 3), torch.zeros(2))
    assert o.shape == (2, OBS_DIM) and torch.allclose(o[0, 6:9], torch.tensor([0.0, 0.0, -1.0]))
    print("obs ok", o.shape)
