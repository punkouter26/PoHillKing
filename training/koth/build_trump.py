"""Profile of the fighter "Trump" (a heavy-set 1.90 m man in his late seventies, from Assets/Art/Raw/Trump/Trump.glb via
tools/make_glb_fighter.py): the numbers koth/build_human.py turns into assets/trump/trump_mjx.xml. Then
`KOTH_ROBOT=trump python -m koth.build_mjcf` builds his scenes exactly as for the G1.
  mass      102 kg (a body-mass index of about 28 at 1.90 m) split by de Leva (1996) segment fractions for men.
  strength  peak joint torques of a large untrained older man: legs and trunk about a third stronger than Kim's in
            absolute terms, but less torque per kilogram (knee 200 Nm / 102 kg vs Kim's 150 / 56), so he is
            slow to recover and moves a lot of mass once he leans in. His fighting style is built on that
            (env.py rung trump_r4bull: he shoves with his chest and belly, arms down).
  range     human ranges of motion, a little reduced for his age (hip flexion, shoulder elevation).
  geometry  his torso is wide: the trunk capsules are capped well above Kim's, and sit at their vertices' x so the
            belly is in front of the hips (LOCAL_X).
"""
from __future__ import annotations
import os, sys

ROBOT = "trump"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets", "trump"))
MASS_KG = 102.0
SPAWN_Z = 1.02
LOCAL_X = True
FRACTION = dict(head=0.0694, chest=0.1596, abdomen=0.1633, pelvis=0.1117, upper_arm=0.0271, forearm=0.0162, hand=0.0061,
                thigh=0.1416, shank=0.0433, foot=0.0137)
CAPS = dict(pelvis=0.17, abdomen=0.17, chest=0.15, head=0.12, thigh=0.10, upper_arm=0.055, forearm=0.05, hand=0.035)
# His rest pose already has the arms hanging about 17 degrees out from the body, so the shoulders need no roll.
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.12, "left_knee_joint": 0.24, "left_ankle_pitch_joint": -0.12,
    "right_hip_pitch_joint": -0.12, "right_knee_joint": 0.24, "right_ankle_pitch_joint": -0.12,
    "left_elbow_joint": 0.25, "right_elbow_joint": 0.25,
}
# (kp, kv) per default class: Kim's set scaled by his gravity torque, mass x height (102 kg x 1.90 m vs 56 x 1.60: x 2.2).
# Scaled by mass alone (x 1.8) the ankles were too soft and the passive hold fell backwards after 5 s.
PD_GAINS = {"trump": (0, 0), "trump_hip": (660, 17), "trump_knee": (880, 17), "trump_ankle": (175, 6), "trump_ankle_pitch": (800, 13),
            "trump_waist_yaw": (550, 13), "trump_waist_pitch": (770, 17), "trump_waist_roll": (770, 17),
            "trump_shoulder": (140, 5), "trump_elbow": (100, 3.5), "trump_wrist": (25, 0.8)}
# class: (axis, range, peak torque Nm, armature)
JOINT_CLASSES = {
    "trump_hip": {"trump_hip_pitch": ("0 1 0", (-1.9, 0.4), 200, 0.035), "trump_hip_roll": ("1 0 0", None, 140, 0.035), "trump_hip_yaw": ("0 0 1", (-0.6, 0.6), 70, 0.02)},
    "trump_knee": ("0 1 0", (-0.05, 2.3), 200, 0.035),
    "trump_ankle": {"trump_ankle_pitch": ("0 1 0", (-0.45, 0.75), 150, 0.02), "trump_ankle_roll": ("1 0 0", (-0.35, 0.35), 45, 0.01)},
    "trump_waist_yaw": ("0 0 1", (-0.7, 0.7), 80, 0.035), "trump_waist_roll": ("1 0 0", (-0.4, 0.4), 140, 0.035), "trump_waist_pitch": ("0 1 0", (-0.3, 0.9), 200, 0.035),
    "trump_shoulder": {"trump_shoulder_pitch": ("0 1 0", (-2.7, 0.8), 70, 0.008), "trump_shoulder_roll": ("1 0 0", None, 65, 0.008), "trump_shoulder_yaw": ("0 0 1", (-1.3, 1.3), 40, 0.005)},
    "trump_elbow": ("0 -1 0", (0.0, 2.4), 55, 0.006),
    "trump_wrist": {"trump_wrist_roll": ("0 0 1", (-1.3, 1.3), 12, 0.0015), "trump_wrist_pitch": ("0 1 0", (-0.9, 0.9), 10, 0.0015), "trump_wrist_yaw": ("1 0 0", (-0.35, 0.35), 10, 0.0015)},
}
SIDE_RANGE = {"hip_roll": (-0.35, 0.7), "shoulder_roll": (-0.35, 2.5)}   # left side; mirrored for the right (-0.3 brings his arm to his side)
DUMMY = dict(hip=0.7, waist=0.9, shoulder=0.35, wrist=0.08, ankle=0.25)  # kg of each in-between link


def build() -> str:
    from koth import build_human
    return build_human.build(sys.modules[__name__])


if __name__ == "__main__":
    build()
