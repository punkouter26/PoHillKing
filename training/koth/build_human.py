"""Physics body of a human fighter (Kim, Trump): writes assets/<robot>/<robot>_mjx.xml from assets/<robot>/<robot>_rig.json
(measured from the character's mesh by tools/make_kim.py or tools/make_glb_fighter.py). Then
`KOTH_ROBOT=<robot> python -m koth.build_mjcf` builds the scenes exactly as for the G1.

Every human has the G1's 29 joints with the same names and order, so observations, the policy network shape, the
training environment and the Unity runner are shared. Everything physical comes from the body's profile module
(koth/build_kim.py, koth/build_trump.py):
  geometry  joint centres are the rig's bone heads (left/right averaged so the body is symmetric); zero pose =
            the character's rest pose with every body frame aligned with the world.
  mass      MASS_KG split by de Leva (1996) segment fractions (FRACTION). The three joints of a hip, the waist, a
            shoulder or a wrist sit at one point (ball joint), so the links between them are small dummy bodies whose
            mass is taken out of the segment they belong to.
  strength  peak joint torques (JOINT_CLASSES) as the actuator force limits.
  range     human ranges of motion.
Sign conventions (x forward, y left, z up): hip/shoulder pitch negative = limb forward; knee and elbow positive =
flexion; ankle pitch positive = toes down; waist pitch positive = lean forward.
Default class names are the robot's name (`kim_hip`, `trump_hip`, ...) so several humans can share one scene.
"""
from __future__ import annotations
import json, os, xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
EDGE_KG = 0.02          # each rounded foot edge; the Unity importer ignores mass="0" and falls back to density
FOOT_GAP = 0.004        # least gap between the two feet's contact geoms in the rest pose (collision_audit.py)
# capsule radius caps (m) when a profile does not set its own: a scan's soft tissue is wider than the body's frame
CAPS = dict(pelvis=0.12, abdomen=0.11, chest=0.11, head=0.11, thigh=0.085, upper_arm=0.045, hand=0.03)


def assets(P) -> str: return os.path.normpath(os.path.join(HERE, "..", "assets", P.ROBOT))


def f(*v): return " ".join(f"{x:.5g}" for x in v)


def build(P) -> str:
    R = P.ROBOT; A = assets(P); caps = {**CAPS, **getattr(P, "CAPS", {})}
    rig = json.load(open(os.path.join(A, f"{R}_rig.json"))); B = rig["bones"]; M = P.MASS_KG; FR = P.FRACTION; DUMMY = P.DUMMY
    head = lambda n: np.array(B[n]["head"])
    def sym(n):                                   # left/right average, returned for the left side (+y)
        l, r = head(n + ".L"), head(n + ".R"); return np.array([(l[0] + r[0]) / 2, (l[1] - r[1]) / 2, (l[2] + r[2]) / 2])
    hips, waist, chest_z, head_z, top = head("Hips"), head("Spine"), head("Chest")[2], head("Head")[2], rig["hi"][2]
    hips_c = np.array([hips[0], 0.0, hips[2]])    # on the symmetry plane
    hip, knee, ankle, toe = sym("UpperLeg"), sym("LowerLeg"), sym("Foot"), sym("Toe")
    shoulder, elbow, wrist = sym("UpperArm"), sym("LowerArm"), sym("Hand")
    waist = np.array([waist[0], 0.0, waist[2]])
    # LOCAL_X: torso and pelvis capsules sit at their vertices' x in the body frame. Kim's body (built before this
    # option) used the rig's absolute x there, a few mm off; she keeps that so her trained brains still match.
    local_x = getattr(P, "LOCAL_X", False)
    hand_len = float(np.linalg.norm(np.array(B["Hand.L"]["tail"]) - head("Hand.L")))
    span = lambda n, k: (np.array(B[n]["hi"]) - np.array(B[n]["lo"]))[k] / 2      # half extent of the bone's vertices
    centre = lambda n, k: (np.array(B[n]["hi"]) + np.array(B[n]["lo"]))[k] / 2

    root = ET.Element("mujoco", model=f"{R} 29dof")
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets")
    dflt = ET.SubElement(ET.SubElement(root, "default"), "default", {"class": R})
    ET.SubElement(dflt, "geom", condim="1", contype="0", conaffinity="0")
    ET.SubElement(dflt, "joint", frictionloss="0.2", damping="0.5", solimplimit="0 0.99 0.01", solreflimit=".008 1")
    ET.SubElement(dflt, "position", inheritrange="1", kp="75", kv="2")
    col = ET.SubElement(dflt, "default", {"class": f"{R}_collision"})
    ET.SubElement(col, "geom", group="3", rgba=".8 .6 .5 1", type="capsule")
    ET.SubElement(ET.SubElement(col, "default", {"class": f"{R}_foot_box"}), "geom", type="box", group="4")
    ET.SubElement(ET.SubElement(col, "default", {"class": f"{R}_foot_capsule"}), "geom", type="capsule", group="3")
    def joint_class(parent, name, spec):
        axis, rng, torque, arm = spec; c = ET.SubElement(parent, "default", {"class": name})
        a = dict(axis=axis, actuatorfrcrange=f(-torque, torque), armature=f(arm))
        if rng: a["range"] = f(*rng)
        ET.SubElement(c, "joint", a)
    for name, spec in P.JOINT_CLASSES.items():
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
            a = {"name": joint, "class": f"{R}_{cls}"}
            if rng: a["range"] = f(*rng)
            ET.SubElement(b, "joint", a)
        return b
    def capsule(b, name, p0, p1, radius, mass, cls="collision"):
        ET.SubElement(b, "geom", {"name": name, "class": f"{R}_{cls}", "size": f(radius), "fromto": f(*p0, *p1), "mass": f(mass)})

    pelvis = ET.SubElement(wb, "body", name="pelvis", pos=f(0, 0, P.SPAWN_Z), childclass=R)
    ET.SubElement(pelvis, "freejoint", name="floating_base_joint")
    pr = min(span("Hips", 0), caps["pelvis"]); pw = max(span("Hips", 1) - pr, 0.02); pz = (hip[2] - hips_c[2]) * 0.5
    px = centre("Hips", 0) - hips_c[0] if local_x else 0.0
    capsule(pelvis, "pelvis_collision", (px, -pw, pz), (px, pw, pz), pr, M * FR["pelvis"])

    for side, s in (("left", 1.0), ("right", -1.0)):
        m = lambda v: np.array([v[0], s * v[1], v[2]])            # mirror a left-side point
        side_rng = lambda k: P.SIDE_RANGE[k] if s > 0 else (-P.SIDE_RANGE[k][1], -P.SIDE_RANGE[k][0])
        # leg
        b = body(pelvis, f"{side}_hip_pitch_link", m(hip) - hips_c, f"{side}_hip_pitch_joint", "hip_pitch", dummy=DUMMY["hip"])
        b = body(b, f"{side}_hip_roll_link", (0, 0, 0), f"{side}_hip_roll_joint", "hip_roll", side_rng("hip_roll"), dummy=DUMMY["hip"])
        b = body(b, f"{side}_hip_yaw_link", (0, 0, 0), f"{side}_hip_yaw_joint", "hip_yaw")
        k = m(knee) - m(hip)
        capsule(b, f"{side}_thigh_collision", k * 0.08, k * 0.92, min(B["UpperLeg.L"]["r50"], caps["thigh"]), M * FR["thigh"] - 2 * DUMMY["hip"])
        b = body(b, f"{side}_knee_link", k, f"{side}_knee_joint", "knee")
        a = m(ankle) - m(knee); shank = M * FR["shank"]
        capsule(b, f"{side}_shin_collision", a * 0.05, a * 0.55, B["LowerLeg.L"]["r50"] * 1.1, shank * 0.65)
        capsule(b, f"{side}_linkage_brace_collision", a * 0.6, a * 0.95, B["LowerLeg.L"]["r50"] * 0.8, shank * 0.35)     # lower shin; the name is the G1's
        b = body(b, f"{side}_ankle_pitch_link", a, f"{side}_ankle_pitch_joint", "ankle_pitch", dummy=DUMMY["ankle"])
        b = body(b, f"{side}_ankle_roll_link", (0, 0, 0), f"{side}_ankle_roll_joint", "ankle_roll")
        heel = B["Foot.L"]["lo"][0] - ankle[0]; tip = B["Toe.L"]["hi"][0] - ankle[0]; half_w = max(span("Foot.L", 1), 0.04); sole = -ankle[2]
        # A narrow stance (Kim) put the two foot boxes 3.6 mm into each other in the rest pose, so every episode began with
        # her feet pressing apart (collision audit 2026-10-19). Move each foot's contact geoms outward until the gap is
        # FOOT_GAP; a body whose feet already clear (Trump) is unchanged.
        shift = max(0.0, half_w + FOOT_GAP / 2 - (ankle[1] - hips_c[1])); out = s * shift if shift > 0 else 0.0
        ET.SubElement(b, "site", name=f"{side}_foot", pos=f((heel + tip) / 2, out, sole))
        ET.SubElement(b, "geom", {"name": f"{side}_foot_box_collision", "class": f"{R}_foot_box", "pos": f((heel + tip) / 2, out, sole + 0.012),
                                  "size": f((tip - heel) / 2, half_w, 0.012), "mass": f(M * FR["foot"] - DUMMY["ankle"] - 3 * EDGE_KG)})
        for i, y in ((1, -half_w * 0.7), (3, half_w * 0.7)):      # rounded toe edges, as on the G1's foot
            ET.SubElement(b, "geom", {"name": f"{side}_foot{i}_collision", "class": f"{R}_foot_capsule", "size": "0.012", "mass": f(EDGE_KG),
                                      "fromto": f(tip - 0.07, y + out, sole + 0.014, tip - 0.015, y + out, sole + 0.014)})
        ET.SubElement(b, "geom", {"name": f"{side}_foot2_collision", "class": f"{R}_foot_capsule", "size": "0.022", "mass": f(EDGE_KG),
                                  "fromto": f(heel + 0.025, out, sole + 0.024, tip - 0.025, out, sole + 0.024)})

    b = body(pelvis, "waist_yaw_link", waist - hips_c, "waist_yaw_joint", "waist_yaw", dummy=DUMMY["waist"])
    b = body(b, "waist_roll_link", (0, 0, 0), "waist_roll_joint", "waist_roll", dummy=DUMMY["waist"])
    torso = body(b, "torso_link", (0, 0, 0), "waist_pitch_joint", "waist_pitch")
    rel = lambda z: z - waist[2]; rx = (lambda x: x - waist[0]) if local_x else (lambda x: x)
    ar = min(span("Spine", 0), caps["abdomen"]); aw = max(span("Spine", 1) - ar, 0.02); az = rel((waist[2] + chest_z) / 2)
    capsule(torso, "abdomen_collision", (rx(centre("Spine", 0)), -aw, az), (rx(centre("Spine", 0)), aw, az), ar, M * FR["abdomen"] - 2 * DUMMY["waist"])
    cr = min(span("Chest", 0), caps["chest"]); cw = max(shoulder[1] - cr, 0.02); cz = rel((chest_z + shoulder[2]) / 2)
    capsule(torso, "chest_collision", (rx(centre("Chest", 0)), -cw, cz), (rx(centre("Chest", 0)), cw, cz), cr, M * FR["chest"])
    hr = min((top - head_z) / 2, caps["head"])
    ET.SubElement(torso, "geom", {"name": "head_collision", "class": f"{R}_collision", "type": "sphere", "size": f(hr),
                                  "pos": f(rx(centre("Head", 0)), 0, rel(top - hr)), "mass": f(M * FR["head"])})

    for side, s in (("left", 1.0), ("right", -1.0)):
        m = lambda v: np.array([v[0], s * v[1], v[2]])
        side_rng = lambda k: P.SIDE_RANGE[k] if s > 0 else (-P.SIDE_RANGE[k][1], -P.SIDE_RANGE[k][0])
        b = body(torso, f"{side}_shoulder_pitch_link", m(shoulder) - waist, f"{side}_shoulder_pitch_joint", "shoulder_pitch", dummy=DUMMY["shoulder"])
        b = body(b, f"{side}_shoulder_roll_link", (0, 0, 0), f"{side}_shoulder_roll_joint", "shoulder_roll", side_rng("shoulder_roll"), dummy=DUMMY["shoulder"])
        b = body(b, f"{side}_shoulder_yaw_link", (0, 0, 0), f"{side}_shoulder_yaw_joint", "shoulder_yaw")
        e = m(elbow) - m(shoulder)
        capsule(b, f"{side}_upper_arm_collision", e * 0.1, e * 0.9, min(B["UpperArm.L"]["r50"], caps["upper_arm"]), M * FR["upper_arm"] - 2 * DUMMY["shoulder"])
        b = body(b, f"{side}_elbow_link", e, f"{side}_elbow_joint", "elbow")
        w = m(wrist) - m(elbow)
        capsule(b, f"{side}_forearm_collision", w * 0.1, w * 0.95, min(B["LowerArm.L"]["r50"], caps.get("forearm", 1.0)), M * FR["forearm"] - 2 * DUMMY["wrist"])
        b = body(b, f"{side}_wrist_roll_link", w, f"{side}_wrist_roll_joint", "wrist_roll", dummy=DUMMY["wrist"])
        b = body(b, f"{side}_wrist_pitch_link", (0, 0, 0), f"{side}_wrist_pitch_joint", "wrist_pitch", dummy=DUMMY["wrist"])
        b = body(b, f"{side}_wrist_yaw_link", (0, 0, 0), f"{side}_wrist_yaw_joint", "wrist_yaw")
        capsule(b, f"{side}_hand_collision", (0, 0, -0.02), (0, 0, -hand_len * 0.85), caps["hand"], M * FR["hand"])

    from koth.build_mjcf import JOINTS
    act = ET.SubElement(root, "actuator")
    for j in JOINTS:
        cls = f"{R}_" + j.replace("left_", "").replace("right_", "").replace("_joint", "")
        ET.SubElement(act, "position", {"name": j, "joint": j, "class": cls})
    ET.indent(root)
    path = os.path.join(A, f"{R}_mjx.xml"); ET.ElementTree(root).write(path, encoding="unicode")
    # Where each body sits in the zero pose (world frame, pelvis at the rig's hips): the Unity skin follower needs it.
    import mujoco
    mm = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(mm); d.qpos[:3] = hips_c; mujoco.mj_forward(mm, d)
    zero = {mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, i): d.xpos[i].round(5).tolist() for i in range(1, mm.nbody)}
    # Which body each bone of the character follows, and where the bone sits in the same pose (its rest pose).
    follow = {"Hips": "pelvis"}
    follow.update({n: "torso_link" for n in ("Spine", "Chest", "Neck", "Head", "Shoulder.L", "Shoulder.R")})
    for side, s in (("left", "L"), ("right", "R")):
        follow.update({f"UpperArm.{s}": f"{side}_shoulder_yaw_link", f"LowerArm.{s}": f"{side}_elbow_link", f"Hand.{s}": f"{side}_wrist_yaw_link",
                       f"UpperLeg.{s}": f"{side}_hip_yaw_link", f"LowerLeg.{s}": f"{side}_knee_link", f"Foot.{s}": f"{side}_ankle_roll_link", f"Toe.{s}": f"{side}_ankle_roll_link"})
    if "source_bones" in rig:      # a GLB rig with its own bone names (tools/make_glb_fighter.py): every bone follows its semantic bone's body
        rows = [{"bone": src, "body": follow[e["bone"]], "bone_pos": e["pos"], "body_pos": zero[follow[e["bone"]]]} for src, e in rig["source_bones"].items()]
        skin = {"scale": rig["scale_from_scan"], "bones": rows}
    else:
        skin = {"bones": [{"bone": n, "body": follow[n], "bone_pos": B[n]["head"], "body_pos": zero[follow[n]]} for n in B]}
    for path_out in (os.path.join(A, f"{R}_skin.json"), os.path.join(REPO, "Assets", "PoKingHill", "Models", R, f"{R}_skin.json")):
        os.makedirs(os.path.dirname(path_out), exist_ok=True); json.dump(skin, open(path_out, "w"), indent=1)
    print(f"wrote {path}: {mm.nbody - 1} bodies, {mm.nu} actuators, mass {sum(mm.body_mass):.2f} kg, "
          f"thigh {np.linalg.norm(knee - hip):.3f} m, shank {np.linalg.norm(ankle - knee):.3f} m, hip height {hip[2]:.3f} m")
    return path
