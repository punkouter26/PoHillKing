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


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vector v into the frame of unit quaternion q = (w, x, y, z). Shapes (N,4), (N,3)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v - w * t + torch.cross(xyz, t, dim=1)


def build_obs(root_quat, root_linvel_world, root_angvel_body, jpos, jvel, default_pose, last_action, command, phase):
    """All tensors batched (N, ...). phase in radians (N,)."""
    g = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device, dtype=root_quat.dtype).expand(root_quat.shape[0], 3)
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


if __name__ == "__main__":
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
