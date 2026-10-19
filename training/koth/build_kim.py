"""Profile of the fighter "Kim" (a 1.60 m woman, from her scan via tools/make_kim.py): the numbers koth/build_human.py
turns into assets/kim/kim_mjx.xml. Then `KOTH_ROBOT=kim python -m koth.build_mjcf` builds her scenes exactly as for the G1.
  mass      56 kg split by de Leva (1996) segment fractions for women.
  strength  peak joint torques of an untrained adult woman, as the actuator force limits.
  range     human ranges of motion.
"""
from __future__ import annotations
import os, sys

ROBOT = "kim"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets", "kim"))
MASS_KG = 56.0                       # 1.60 m at a body-mass index of 22
SPAWN_Z = 0.78
FRACTION = dict(head=0.0668, chest=0.1545, abdomen=0.1465, pelvis=0.1247, upper_arm=0.0255, forearm=0.0138, hand=0.0056,
                thigh=0.1478, shank=0.0481, foot=0.0129)
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.15, "left_knee_joint": 0.30, "left_ankle_pitch_joint": -0.15,
    "right_hip_pitch_joint": -0.15, "right_knee_joint": 0.30, "right_ankle_pitch_joint": -0.15,
    "left_shoulder_roll_joint": 0.12, "right_shoulder_roll_joint": -0.12, "left_elbow_joint": 0.35, "right_elbow_joint": 0.35,
}
# (kp, kv) per default class. Scaled up from the G1's set (35 kg) for her mass; checked by the passive-hold test.
PD_GAINS = {"kim": (0, 0), "kim_hip": (300, 8), "kim_knee": (400, 8), "kim_ankle": (80, 3), "kim_ankle_pitch": (300, 6),
            "kim_waist_yaw": (250, 6), "kim_waist_pitch": (350, 8), "kim_waist_roll": (350, 8),
            "kim_shoulder": (80, 3), "kim_elbow": (60, 2), "kim_wrist": (15, 0.5)}
# class: (axis, range, peak torque Nm, armature)
JOINT_CLASSES = {
    "kim_hip": {"kim_hip_pitch": ("0 1 0", (-2.1, 0.5), 150, 0.02), "kim_hip_roll": ("1 0 0", None, 100, 0.02), "kim_hip_yaw": ("0 0 1", (-0.7, 0.7), 50, 0.01)},
    "kim_knee": ("0 1 0", (-0.05, 2.4), 150, 0.02),
    "kim_ankle": {"kim_ankle_pitch": ("0 1 0", (-0.5, 0.8), 110, 0.01), "kim_ankle_roll": ("1 0 0", (-0.4, 0.4), 30, 0.005)},
    "kim_waist_yaw": ("0 0 1", (-0.8, 0.8), 60, 0.02), "kim_waist_roll": ("1 0 0", (-0.45, 0.45), 100, 0.02), "kim_waist_pitch": ("0 1 0", (-0.4, 1.0), 150, 0.02),
    "kim_shoulder": {"kim_shoulder_pitch": ("0 1 0", (-3.0, 0.9), 55, 0.005), "kim_shoulder_roll": ("1 0 0", None, 50, 0.005), "kim_shoulder_yaw": ("0 0 1", (-1.4, 1.4), 30, 0.003)},
    "kim_elbow": ("0 -1 0", (0.0, 2.5), 45, 0.004),
    "kim_wrist": {"kim_wrist_roll": ("0 0 1", (-1.4, 1.4), 10, 0.001), "kim_wrist_pitch": ("0 1 0", (-1.0, 1.0), 8, 0.001), "kim_wrist_yaw": ("1 0 0", (-0.4, 0.4), 8, 0.001)},
}
SIDE_RANGE = {"hip_roll": (-0.4, 0.8), "shoulder_roll": (-0.3, 2.8)}     # left side; mirrored for the right
DUMMY = dict(hip=0.4, waist=0.5, shoulder=0.2, wrist=0.05, ankle=0.15)   # kg of each in-between link


def build() -> str:
    from koth import build_human
    return build_human.build(sys.modules[__name__])


if __name__ == "__main__":
    build()
