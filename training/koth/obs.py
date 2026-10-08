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


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vector v into the frame of unit quaternion q = (w, x, y, z). Shapes (N,4), (N,3)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v - w * t + torch.cross(xyz, t, dim=1)


def build_obs(root_quat, root_linvel_world, root_angvel_body, jpos, jvel, default_pose, last_action, command, phase):
    """All tensors batched (N, ...). phase in radians (N,)."""
    g = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device).expand(root_quat.shape[0], 3)
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


if __name__ == "__main__":
    q = torch.tensor([[1.0, 0, 0, 0], [0.7071068, 0, 0, 0.7071068]])   # identity, 90 deg yaw
    v = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]])
    r = quat_rotate_inverse(q, v)
    assert torch.allclose(r[0], v[0]) and torch.allclose(r[1], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6), r
    o = build_obs(q, v, v, torch.zeros(2, 29), torch.zeros(2, 29), torch.zeros(29), torch.zeros(2, 29), torch.zeros(2, 3), torch.zeros(2))
    assert o.shape == (2, OBS_DIM) and torch.allclose(o[0, 6:9], torch.tensor([0.0, 0.0, -1.0]))
    print("obs ok", o.shape)
