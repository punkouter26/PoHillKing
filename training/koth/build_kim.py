"""Physics body for the fighter "Kim": writes assets/kim/kim_mjx.xml from assets/kim/kim_rig.json (measured from her
scan by tools/make_kim.py). Then `KOTH_ROBOT=kim python -m koth.build_mjcf` builds her scenes exactly as for the G1.

She has the G1's 29 joints with the same names and order, so observations, the policy network shape, the training
environment and the Unity runner are shared. Everything physical is hers:
  geometry  joint centres are her rig's bone heads (left/right averaged so the body is symmetric); zero pose =
            standing with straight legs and arms hanging down, every body frame aligned with the world.
  mass      MASS_KG split by de Leva (1996) segment fractions for women. The three joints of a hip, the waist, a
            shoulder or a wrist sit at one point (ball joint), so the links between them are small dummy bodies whose
            mass is taken out of the segment they belong to.
  strength  peak joint torques of an untrained adult woman, as the actuator force limits.
  range     human ranges of motion.
Sign conventions (x forward, y left, z up): hip/shoulder pitch negative = limb forward; knee and elbow positive =
flexion; ankle pitch positive = toes down; waist pitch positive = lean forward.
"""
from __future__ import annotations
import json, os, xml.etree.ElementTree as ET
import numpy as np

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
EDGE_KG = 0.02          # each rounded foot edge; the Unity importer ignores mass="0" and falls back to density
DUMMY = dict(hip=0.4, waist=0.5, shoulder=0.2, wrist=0.05, ankle=0.15)   # kg of each in-between link


def f(*v): return " ".join(f"{x:.5g}" for x in v)


def build() -> str:
    rig = json.load(open(os.path.join(ASSETS, "kim_rig.json"))); B = rig["bones"]; M = MASS_KG
    head = lambda n: np.array(B[n]["head"])
    def sym(n):                                   # left/right average, returned for the left side (+y)
        l, r = head(n + ".L"), head(n + ".R"); return np.array([(l[0] + r[0]) / 2, (l[1] - r[1]) / 2, (l[2] + r[2]) / 2])
    hips, waist, chest_z, head_z, top = head("Hips"), head("Spine"), head("Chest")[2], head("Head")[2], rig["hi"][2]
    hip, knee, ankle, toe = sym("UpperLeg"), sym("LowerLeg"), sym("Foot"), sym("Toe")
    shoulder, elbow, wrist = sym("UpperArm"), sym("LowerArm"), sym("Hand")
    hand_len = float(np.linalg.norm(np.array(B["Hand.L"]["tail"]) - head("Hand.L")))
    span = lambda n, k: (np.array(B[n]["hi"]) - np.array(B[n]["lo"]))[k] / 2      # half extent of the bone's vertices
    centre = lambda n, k: (np.array(B[n]["hi"]) + np.array(B[n]["lo"]))[k] / 2

    root = ET.Element("mujoco", model="kim 29dof")
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets")
    dflt = ET.SubElement(ET.SubElement(root, "default"), "default", {"class": "kim"})
    ET.SubElement(dflt, "geom", condim="1", contype="0", conaffinity="0")
    ET.SubElement(dflt, "joint", frictionloss="0.2", damping="0.5", solimplimit="0 0.99 0.01", solreflimit=".008 1")
    ET.SubElement(dflt, "position", inheritrange="1", kp="75", kv="2")
    col = ET.SubElement(dflt, "default", {"class": "kim_collision"})
    ET.SubElement(col, "geom", group="3", rgba=".8 .6 .5 1", type="capsule")
    ET.SubElement(ET.SubElement(col, "default", {"class": "kim_foot_box"}), "geom", type="box", group="4")
    ET.SubElement(ET.SubElement(col, "default", {"class": "kim_foot_capsule"}), "geom", type="capsule", group="3")
    def joint_class(parent, name, spec):
        axis, rng, torque, arm = spec; c = ET.SubElement(parent, "default", {"class": name})
        a = dict(axis=axis, actuatorfrcrange=f(-torque, torque), armature=f(arm))
        if rng: a["range"] = f(*rng)
        ET.SubElement(c, "joint", a)
    for name, spec in JOINT_CLASSES.items():
        if isinstance(spec, dict):
            group = ET.SubElement(dflt, "default", {"class": name})
            for sub, s in spec.items(): joint_class(group, sub, s)
        else: joint_class(dflt, name, spec)
    ET.SubElement(root, "asset")
    wb = ET.SubElement(root, "worldbody")

    def body(parent, name, pos, joint=None, cls=None, rng=None, dummy=None):
        b = ET.SubElement(parent, "body", name=name, pos=f(*pos))
        if dummy: ET.SubElement(b, "inertial", pos="0 0 0", mass=f(dummy), diaginertia=f(*[dummy * 0.03 ** 2] * 3))
        if joint:
            a = {"name": joint, "class": cls}
            if rng: a["range"] = f(*rng)
            ET.SubElement(b, "joint", a)
        return b
    def capsule(b, name, p0, p1, radius, mass, cls="kim_collision"):
        ET.SubElement(b, "geom", {"name": name, "class": cls, "size": f(radius), "fromto": f(*p0, *p1), "mass": f(mass)})

    pelvis = ET.SubElement(wb, "body", name="pelvis", pos=f(0, 0, SPAWN_Z), childclass="kim")
    ET.SubElement(pelvis, "freejoint", name="floating_base_joint")
    pr = min(span("Hips", 0), 0.12); pw = max(span("Hips", 1) - pr, 0.02); pz = (hip[2] - hips[2]) * 0.5
    capsule(pelvis, "pelvis_collision", (0, -pw, pz), (0, pw, pz), pr, M * FRACTION["pelvis"])

    for side, s in (("left", 1.0), ("right", -1.0)):
        m = lambda v: np.array([v[0], s * v[1], v[2]])            # mirror a left-side point
        side_rng = lambda k: SIDE_RANGE[k] if s > 0 else (-SIDE_RANGE[k][1], -SIDE_RANGE[k][0])
        # leg
        b = body(pelvis, f"{side}_hip_pitch_link", m(hip) - hips, f"{side}_hip_pitch_joint", "kim_hip_pitch", dummy=DUMMY["hip"])
        b = body(b, f"{side}_hip_roll_link", (0, 0, 0), f"{side}_hip_roll_joint", "kim_hip_roll", side_rng("hip_roll"), dummy=DUMMY["hip"])
        b = body(b, f"{side}_hip_yaw_link", (0, 0, 0), f"{side}_hip_yaw_joint", "kim_hip_yaw")
        k = m(knee) - m(hip)
        capsule(b, f"{side}_thigh_collision", k * 0.08, k * 0.92, min(B["UpperLeg.L"]["r50"], 0.085), M * FRACTION["thigh"] - 2 * DUMMY["hip"])
        b = body(b, f"{side}_knee_link", k, f"{side}_knee_joint", "kim_knee")
        a = m(ankle) - m(knee); shank = M * FRACTION["shank"]
        capsule(b, f"{side}_shin_collision", a * 0.05, a * 0.55, B["LowerLeg.L"]["r50"] * 1.1, shank * 0.65)
        capsule(b, f"{side}_linkage_brace_collision", a * 0.6, a * 0.95, B["LowerLeg.L"]["r50"] * 0.8, shank * 0.35)     # lower shin; the name is the G1's
        b = body(b, f"{side}_ankle_pitch_link", a, f"{side}_ankle_pitch_joint", "kim_ankle_pitch", dummy=DUMMY["ankle"])
        b = body(b, f"{side}_ankle_roll_link", (0, 0, 0), f"{side}_ankle_roll_joint", "kim_ankle_roll")
        heel = B["Foot.L"]["lo"][0] - ankle[0]; tip = B["Toe.L"]["hi"][0] - ankle[0]; half_w = max(span("Foot.L", 1), 0.04); sole = -ankle[2]
        ET.SubElement(b, "site", name=f"{side}_foot", pos=f((heel + tip) / 2, 0, sole))
        ET.SubElement(b, "geom", {"name": f"{side}_foot_box_collision", "class": "kim_foot_box", "pos": f((heel + tip) / 2, 0, sole + 0.012),
                                  "size": f((tip - heel) / 2, half_w, 0.012), "mass": f(M * FRACTION["foot"] - DUMMY["ankle"] - 3 * EDGE_KG)})
        for i, y in ((1, -half_w * 0.7), (3, half_w * 0.7)):      # rounded toe edges, as on the G1's foot
            ET.SubElement(b, "geom", {"name": f"{side}_foot{i}_collision", "class": "kim_foot_capsule", "size": "0.012", "mass": f(EDGE_KG),
                                      "fromto": f(tip - 0.07, y, sole + 0.014, tip - 0.015, y, sole + 0.014)})
        ET.SubElement(b, "geom", {"name": f"{side}_foot2_collision", "class": "kim_foot_capsule", "size": "0.022", "mass": f(EDGE_KG),
                                  "fromto": f(heel + 0.025, 0, sole + 0.024, tip - 0.025, 0, sole + 0.024)})

    b = body(pelvis, "waist_yaw_link", waist - hips, "waist_yaw_joint", "kim_waist_yaw", dummy=DUMMY["waist"])
    b = body(b, "waist_roll_link", (0, 0, 0), "waist_roll_joint", "kim_waist_roll", dummy=DUMMY["waist"])
    torso = body(b, "torso_link", (0, 0, 0), "waist_pitch_joint", "kim_waist_pitch")
    rel = lambda z: z - waist[2]
    ar = min(span("Spine", 0), 0.11); aw = max(span("Spine", 1) - ar, 0.02)
    capsule(torso, "abdomen_collision", (centre("Spine", 0), -aw, rel((waist[2] + chest_z) / 2)), (centre("Spine", 0), aw, rel((waist[2] + chest_z) / 2)), ar, M * FRACTION["abdomen"] - 2 * DUMMY["waist"])
    cr = min(span("Chest", 0), 0.11); cw = max(shoulder[1] - cr, 0.02); cz = rel((chest_z + shoulder[2]) / 2)
    capsule(torso, "chest_collision", (centre("Chest", 0), -cw, cz), (centre("Chest", 0), cw, cz), cr, M * FRACTION["chest"])
    hr = min((top - head_z) / 2, 0.11)
    ET.SubElement(torso, "geom", {"name": "head_collision", "class": "kim_collision", "type": "sphere", "size": f(hr),
                                  "pos": f(centre("Head", 0), 0, rel(top - hr)), "mass": f(M * FRACTION["head"])})

    for side, s in (("left", 1.0), ("right", -1.0)):
        m = lambda v: np.array([v[0], s * v[1], v[2]])
        side_rng = lambda k: SIDE_RANGE[k] if s > 0 else (-SIDE_RANGE[k][1], -SIDE_RANGE[k][0])
        b = body(torso, f"{side}_shoulder_pitch_link", m(shoulder) - waist, f"{side}_shoulder_pitch_joint", "kim_shoulder_pitch", dummy=DUMMY["shoulder"])
        b = body(b, f"{side}_shoulder_roll_link", (0, 0, 0), f"{side}_shoulder_roll_joint", "kim_shoulder_roll", side_rng("shoulder_roll"), dummy=DUMMY["shoulder"])
        b = body(b, f"{side}_shoulder_yaw_link", (0, 0, 0), f"{side}_shoulder_yaw_joint", "kim_shoulder_yaw")
        e = m(elbow) - m(shoulder)
        capsule(b, f"{side}_upper_arm_collision", e * 0.1, e * 0.9, min(B["UpperArm.L"]["r50"], 0.045), M * FRACTION["upper_arm"] - 2 * DUMMY["shoulder"])
        b = body(b, f"{side}_elbow_link", e, f"{side}_elbow_joint", "kim_elbow")
        w = m(wrist) - m(elbow)
        capsule(b, f"{side}_forearm_collision", w * 0.1, w * 0.95, B["LowerArm.L"]["r50"], M * FRACTION["forearm"] - 2 * DUMMY["wrist"])
        b = body(b, f"{side}_wrist_roll_link", w, f"{side}_wrist_roll_joint", "kim_wrist_roll", dummy=DUMMY["wrist"])
        b = body(b, f"{side}_wrist_pitch_link", (0, 0, 0), f"{side}_wrist_pitch_joint", "kim_wrist_pitch", dummy=DUMMY["wrist"])
        b = body(b, f"{side}_wrist_yaw_link", (0, 0, 0), f"{side}_wrist_yaw_joint", "kim_wrist_yaw")
        capsule(b, f"{side}_hand_collision", (0, 0, -0.02), (0, 0, -hand_len * 0.85), 0.03, M * FRACTION["hand"])

    from koth.build_mjcf import JOINTS
    act = ET.SubElement(root, "actuator")
    for j in JOINTS:
        cls = "kim_" + j.replace("left_", "").replace("right_", "").replace("_joint", "")
        ET.SubElement(act, "position", {"name": j, "joint": j, "class": cls})
    ET.indent(root)
    path = os.path.join(ASSETS, "kim_mjx.xml"); ET.ElementTree(root).write(path, encoding="unicode")
    # Where each body sits in the zero pose (world frame, pelvis at the rig's hips): the Unity skin follower needs it.
    import mujoco
    mm = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(mm); d.qpos[2] = hips[2]; mujoco.mj_forward(mm, d)
    zero = {mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, i): d.xpos[i].round(5).tolist() for i in range(1, mm.nbody)}
    # Which body each bone of the scan follows, and where the bone sits in the same pose (her rest pose).
    follow = {"Hips": "pelvis"}
    follow.update({n: "torso_link" for n in ("Spine", "Chest", "Neck", "Head", "Shoulder.L", "Shoulder.R")})
    for side, s in (("left", "L"), ("right", "R")):
        follow.update({f"UpperArm.{s}": f"{side}_shoulder_yaw_link", f"LowerArm.{s}": f"{side}_elbow_link", f"Hand.{s}": f"{side}_wrist_yaw_link",
                       f"UpperLeg.{s}": f"{side}_hip_yaw_link", f"LowerLeg.{s}": f"{side}_knee_link", f"Foot.{s}": f"{side}_ankle_roll_link", f"Toe.{s}": f"{side}_ankle_roll_link"})
    skin = {"bones": [{"bone": n, "body": follow[n], "bone_pos": B[n]["head"], "body_pos": zero[follow[n]]} for n in B]}
    for path_out in (os.path.join(ASSETS, "kim_skin.json"), os.path.normpath(os.path.join(HERE, "..", "..", "Assets", "PoKingHill", "Models", "kim", "kim_skin.json"))):
        os.makedirs(os.path.dirname(path_out), exist_ok=True); json.dump(skin, open(path_out, "w"), indent=1)
    print(f"wrote {path}: {mm.nbody - 1} bodies, {mm.nu} actuators, mass {sum(mm.body_mass):.2f} kg, "
          f"thigh {np.linalg.norm(knee - hip):.3f} m, shank {np.linalg.norm(ankle - knee):.3f} m, hip height {hip[2]:.3f} m")
    return path


if __name__ == "__main__":
    build()
