"""Profile of the fighter "Grandpa" (a heavy-set 1.74 m man in his eighties, from Assets/Art/Raw/Grandpa/Grandpa.glb via
tools/make_grandpa.py): the numbers koth/build_human.py turns into assets/grandpa/grandpa_mjx.xml. Then
`KOTH_ROBOT=grandpa python -m koth.build_mjcf` builds his scenes exactly as for the G1.
  mass      86 kg (a body-mass index of about 28 at 1.74 m) split by de Leva (1996) segment fractions for men.
  strength  Trump's torques and gains scaled by the gravity torque they work against, mass x height
            (86 kg x 1.74 m vs 102 kg x 1.90 m: x 0.77), so he has the same strength for his size as Trump, whose
            walker and fighter train on the shared human recipe (env.human_overrides).
  range     human ranges of motion, reduced a little more than Trump's for his age (hip flexion, knee, shoulder elevation).
  geometry  a big belly over a narrow pelvis: wide trunk capsules (CAPS) placed at their vertices' x (LOCAL_X); his
            trouser legs are baggy, so the thigh capsule is capped below the cloth's radius. The rest pose has his
            arms hanging 20 degrees out from the body (they clear the belly there), so the shoulders need no roll.
"""
from __future__ import annotations
import os, sys

ROBOT = "grandpa"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets", "grandpa"))
MASS_KG = 86.0
SPAWN_Z = 0.98
LOCAL_X = True
FRACTION = dict(head=0.0694, chest=0.1596, abdomen=0.1633, pelvis=0.1117, upper_arm=0.0271, forearm=0.0162, hand=0.0061,
                thigh=0.1416, shank=0.0433, foot=0.0137)
CAPS = dict(pelvis=0.16, abdomen=0.17, chest=0.15, head=0.11, thigh=0.09, upper_arm=0.05, forearm=0.045, hand=0.035)
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.12, "left_knee_joint": 0.24, "left_ankle_pitch_joint": -0.12,
    "right_hip_pitch_joint": -0.12, "right_knee_joint": 0.24, "right_ankle_pitch_joint": -0.12,
    "left_elbow_joint": 0.25, "right_elbow_joint": 0.25,
}
# (kp, kv) per default class: Trump's set x 0.77 (gravity torque, see above).
PD_GAINS = {"grandpa": (0, 0), "grandpa_hip": (510, 13), "grandpa_knee": (680, 13), "grandpa_ankle": (135, 4.6), "grandpa_ankle_pitch": (620, 10),
            "grandpa_waist_yaw": (425, 10), "grandpa_waist_pitch": (595, 13), "grandpa_waist_roll": (595, 13),
            "grandpa_shoulder": (108, 3.9), "grandpa_elbow": (77, 2.7), "grandpa_wrist": (19, 0.6)}
# class: (axis, range, peak torque Nm, armature)
JOINT_CLASSES = {
    "grandpa_hip": {"grandpa_hip_pitch": ("0 1 0", (-1.8, 0.35), 155, 0.03), "grandpa_hip_roll": ("1 0 0", None, 108, 0.03), "grandpa_hip_yaw": ("0 0 1", (-0.55, 0.55), 54, 0.016)},
    "grandpa_knee": ("0 1 0", (-0.05, 2.2), 155, 0.03),
    "grandpa_ankle": {"grandpa_ankle_pitch": ("0 1 0", (-0.4, 0.7), 116, 0.016), "grandpa_ankle_roll": ("1 0 0", (-0.33, 0.33), 35, 0.008)},
    "grandpa_waist_yaw": ("0 0 1", (-0.6, 0.6), 62, 0.03), "grandpa_waist_roll": ("1 0 0", (-0.35, 0.35), 108, 0.03), "grandpa_waist_pitch": ("0 1 0", (-0.25, 0.85), 155, 0.03),
    "grandpa_shoulder": {"grandpa_shoulder_pitch": ("0 1 0", (-2.5, 0.7), 54, 0.0065), "grandpa_shoulder_roll": ("1 0 0", None, 50, 0.0065), "grandpa_shoulder_yaw": ("0 0 1", (-1.2, 1.2), 31, 0.004)},
    "grandpa_elbow": ("0 -1 0", (0.0, 2.3), 42, 0.005),
    "grandpa_wrist": {"grandpa_wrist_roll": ("0 0 1", (-1.2, 1.2), 9, 0.0012), "grandpa_wrist_pitch": ("0 1 0", (-0.85, 0.85), 8, 0.0012), "grandpa_wrist_yaw": ("1 0 0", (-0.3, 0.3), 8, 0.0012)},
}
SIDE_RANGE = {"hip_roll": (-0.3, 0.6), "shoulder_roll": (-0.4, 2.2)}    # left side; mirrored for the right (-0.35 brings his arm to his side)
DUMMY = dict(hip=0.6, waist=0.75, shoulder=0.3, wrist=0.07, ankle=0.2)   # kg of each in-between link


def build() -> str:
    from koth import build_human
    return build_human.build(sys.modules[__name__])


if __name__ == "__main__":
    build()
