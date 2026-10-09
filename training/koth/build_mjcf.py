"""Phase A asset generator. Single source of truth for everything both sims load.

Outputs (all under training/assets/):
  g1/assets/arena.obj          convex dome arena mesh (flat 3 m disc on top, convex slope)
  g1/scene_flat_1p.xml         one G1 on a plane + projectile pool          (R0/R1 training)
  g1/scene_koth_2p.xml         two G1 on the hfield arena + projectile pool (R2+ training)
  g1/scene_*_train.xml         MuJoCo-resolved canonical model + keyframe (what training loads)
  g1/scene_*_unity.xml         same text minus keyframe/sensor (what the Unity importer loads)
  g1/model_dump.json           parity reference (nq/nv/nu, masses, ranges, gains, options, hfield meta)
  g1/joint_map.json            canonical 29 joint names + default pose + ctrl scale

Design constraints (from the org.mujoco 3.15 plugin audit): no <contact><pair>, no keyframes, no sensors,
timestep/gravity come from Unity settings, no ls_iterations/eulerdamp knobs -> implicitfast + MuJoCo defaults.
Collision bitmasks:  1=robot A  2=robot B  4=arena  8=box  16=foot/shin extra
"""
from __future__ import annotations
import copy, json, os, struct, xml.etree.ElementTree as ET
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets", "g1"))
SRC = os.path.join(ASSETS, "g1_mjx.xml")

SIM_DT = 0.002
CTRL_DT = 0.02
ACTION_SCALE = 0.5

# canonical joint order = menagerie actuator order
JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# mild knee bend (hip + knee + ankle pitch ~ 0 keeps the torso vertical), arms from menagerie "stand"
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.20, "left_knee_joint": 0.42, "left_ankle_pitch_joint": -0.23,
    "right_hip_pitch_joint": -0.20, "right_knee_joint": 0.42, "right_ankle_pitch_joint": -0.23,
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 1.28,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 1.28,
}

# PD gains (kp, kv). Measured on CPU with the gravity-compensated hold: Unitree/IsaacLab-level gains
# (hips 100-150, knee 150-200, ankle 40) cannot hold ANY static pose (whole-leg chain too compliant vs m*g*h),
# menagerie's 500 holds rigidly. 200/300/200 is the softest set that holds (pitch settles ~1.3 deg). Joint
# actuatorfrcrange (88/139/50/25 Nm) still caps torque, so large errors behave torque-limited. See rl_optimization_log.md.
# Base class g1 is (0, 0) on purpose and every joint group sets its own pair: MuJoCo's XML writer truncates a
# child default's biasprm when its tail equals the parent's (knee kv == g1 kv gave biasprm="0 -300"), and the
# Unity importer, which re-saves the model through MuJoCo before parsing, then reads the missing kv as 0.
PD_GAINS = {"g1": (0, 0), "hip": (200, 5), "knee": (300, 5), "ankle": (60, 3), "ankle_pitch": (200, 5),
            "waist_yaw": (200, 5), "waist_pitch": (200, 5), "waist_roll": (200, 5),
            "shoulder": (60, 3), "elbow": (60, 3), "wrist": (40, 2)}


def apply_pd_gains(default: ET.Element) -> ET.Element:
    for cls in default.iter("default"):
        if cls.get("class") in PD_GAINS:
            kp, kv = PD_GAINS[cls.get("class")]
            pos = cls.find("position")
            if pos is None:
                pos = ET.SubElement(cls, "position")
            pos.set("kp", f"{kp:g}"); pos.set("kv", f"{kv:g}")
    # Torque limits: the Unity plugin drops joint actuatorfrcrange, so express the same limit as the actuator's
    # forcerange (identical with one actuator per joint).
    for cls in default.iter("default"):
        j = cls.find("joint")
        if j is not None and "actuatorfrcrange" in j.attrib:
            pos = cls.find("position")
            if pos is None:
                pos = ET.SubElement(cls, "position")
            pos.set("forcerange", j.attrib.pop("actuatorfrcrange"))
    return default


# ---- arena ---------------------------------------------------------------------------------------
HF_DEPTH = 10.0       # dome depth below the plateau
PLATEAU_R = 1.5
SLOPE_K = 1.0 / 9.0   # z = -k (r-1.5)^2  -> 45 deg at r = 6


def write_arena() -> str:
    """Convex dome: flat disc of radius PLATEAU_R at z=0, then z = -SLOPE_K (r-1.5)^2 down to HF_DEPTH.
    Written as OBJ (MuJoCo + the Unity plugin both load it; MuJoCo uses the convex hull, which equals the
    solid itself because the profile is concave). mujoco_warp gives full multi-contact for mesh pairs but
    only one contact per height-field pair, which is why this is a mesh and not an hfield."""
    n_ang, n_rad = 48, 16
    r_max = PLATEAU_R + np.sqrt(HF_DEPTH / SLOPE_K)          # where the slope reaches -HF_DEPTH
    rs = PLATEAU_R + (np.linspace(0, 1, n_rad) ** 1.5) * (r_max - PLATEAU_R)
    verts = []
    for r in rs:
        z = -SLOPE_K * (r - PLATEAU_R) ** 2
        for k in range(n_ang):
            a = 2 * np.pi * k / n_ang
            verts.append((r * np.cos(a), r * np.sin(a), z))
    for k in range(n_ang):                                   # bottom ring closes the solid
        a = 2 * np.pi * k / n_ang
        verts.append((r_max * np.cos(a), r_max * np.sin(a), -HF_DEPTH - 0.5))
    verts = np.array(verts, dtype=np.float64)
    faces = []
    rings = n_rad + 1
    for i in range(rings - 1):
        for k in range(n_ang):
            a, b = i * n_ang + k, i * n_ang + (k + 1) % n_ang
            c, d = a + n_ang, b + n_ang
            faces += [(a, b, d), (a, d, c)]
    top = len(verts); verts = np.vstack([verts, [[0, 0, 0]]])
    bot = len(verts); verts = np.vstack([verts, [[0, 0, -HF_DEPTH - 0.5]]])
    for k in range(n_ang):
        faces.append((top, (k + 1) % n_ang, k))
        base = (rings - 1) * n_ang
        faces.append((bot, base + k, base + (k + 1) % n_ang))
    # Binary STL, not OBJ: the Unity plugin converts STL meshes to Unity axes correctly (the robot links), but its
    # OBJ path left the dome lying on its side, so the hill was missing from the first Unity screenshots.
    path = os.path.join(ASSETS, "assets", "arena.stl")
    centre = verts.mean(axis=0); tris = []
    for a_, b_, c_ in faces:
        p0, p1, p2 = verts[a_], verts[b_], verts[c_]; nrm = np.cross(p1 - p0, p2 - p0)
        if np.dot(nrm, (p0 + p1 + p2) / 3 - centre) < 0: p1, p2 = p2, p1; nrm = -nrm      # outward-facing triangles
        ln = np.linalg.norm(nrm); tris.append((nrm / ln if ln > 0 else nrm, p0, p1, p2))
    with open(path, "wb") as f:
        f.write(b"PoKingHill arena dome, binary stl, generated by koth/build_mjcf.py".ljust(80, b" ")); f.write(struct.pack("<I", len(tris)))
        for nrm, p0, p1, p2 in tris:
            f.write(struct.pack("<12fH", *nrm, *p0, *p1, *p2, 0))
    return path
    path = os.path.join(ASSETS, "assets", "arena.obj")
    with open(path, "w") as f:
        f.write("# PoKingHill arena dome, generated by koth/build_mjcf.py" + chr(10))
        for v in verts: f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}" + chr(10))
        for a, b, c in faces: f.write(f"f {a + 1} {b + 1} {c + 1}" + chr(10))
    return path


# ---- robot ---------------------------------------------------------------------------------------
def _prefix(elem: ET.Element, p: str):
    for e in elem.iter():
        for a in ("name", "joint", "body1", "body2", "site", "target"):
            if a in e.attrib and e.tag != "mesh":
                e.set(a, p + e.get(a))


def make_robot(src_root: ET.Element, prefix: str, mask_bit: int, pos, quat) -> tuple[ET.Element, list[ET.Element], list[ET.Element]]:
    """Returns (pelvis body, actuator elements, exclude elements) for one robot."""
    pelvis = copy.deepcopy(src_root.find("worldbody/body[@name='pelvis']"))
    for junk in pelvis.findall("camera"):
        pelvis.remove(junk)
    pelvis.set("pos", " ".join(f"{v:g}" for v in pos))
    pelvis.set("quat", " ".join(f"{v:g}" for v in quat))
    other = 3 - mask_bit                              # 1 <-> 2
    for g in pelvis.iter("geom"):
        cls = g.get("class", "")
        if cls == "visual":
            continue
        is_foot = cls.startswith("foot")
        is_shin = "shin" in g.get("name", "") or "linkage_brace" in g.get("name", "")
        ct = mask_bit | (16 if is_foot else 0)
        ca = other | 4 | (16 if (is_foot or is_shin) else 0)
        g.set("contype", str(ct)); g.set("conaffinity", str(ca)); g.set("condim", "3")
        g.set("friction", "0.6" if is_foot else "0.8")
    _prefix(pelvis, prefix)
    acts = []
    for a in src_root.find("actuator"):
        a = copy.deepcopy(a)
        a.set("name", prefix + a.get("name")); a.set("joint", prefix + a.get("joint"))
        acts.append(a)
    excl = []
    for side in ("left", "right"):   # same-leg foot/shin would otherwise collide through bit 16
        excl.append(ET.Element("exclude", body1=f"{prefix}{side}_ankle_roll_link", body2=f"{prefix}{side}_knee_link"))
    return pelvis, acts, excl


POOL_N = 4
POOL_MASS = 2.0


def pool_park(i: int) -> tuple[float, float, float]:
    """Parked boxes float far away in the sky (never under the infinite floor plane: a body inside the plane is
    ejected at ~950 m/s). A plane floor is ~25% cheaper in mujoco_warp than a box slab."""
    return (100.0 + 0.5 * i, 0.0, 50.0)


def make_pool(n: int = POOL_N) -> list[ET.Element]:
    """Pre-allocated projectile pool. Parked boxes float in the sky with no contacts: both sims re-pin parked boxes
    (qpos = park pose, qvel = 0) every control step instead of resting them on a shelf, which cost ~30 permanent
    contacts per world. Measured in mujoco_warp: 8 boxes on a shelf 1297 ms per control step, 2 pinned boxes 563 ms."""
    bodies = []
    for i in range(n):
        b = ET.Element("body", name=f"box{i}", pos=" ".join(f"{v:g}" for v in pool_park(i)))
        ET.SubElement(b, "freejoint", name=f"box{i}_free")
        ET.SubElement(b, "geom", name=f"box{i}_geom", type="box", size="0.1 0.1 0.1", mass=f"{POOL_MASS:g}",
                      contype="8", conaffinity="15", condim="3", friction="0.5", rgba="0.9 0.3 0.1 1")
        bodies.append(b)
    return bodies


def build_scene(src_root: ET.Element, two_player: bool, arena: bool | None = None) -> ET.Element:
    arena = two_player if arena is None else arena
    root = ET.Element("mujoco", model="g1_koth_2p" if two_player else "g1_koth_1p")
    ET.SubElement(root, "compiler", angle="radian", assetdir="assets", autolimits="true")
    # implicitfast: Unity-settable, and makes the eulerdamp flag irrelevant. iterations=5 like menagerie mjx.
    # ls_iterations: the Unity plugin cannot import it, so PolicyRunner writes model->opt.ls_iterations after init.
    ET.SubElement(root, "option", timestep=f"{SIM_DT}", integrator="implicitfast", iterations="5", ls_iterations="10")
    root.append(apply_pd_gains(copy.deepcopy(src_root.find("default"))))
    asset = copy.deepcopy(src_root.find("asset"))
    if arena:
        ET.SubElement(asset, "mesh", name="arena", file="arena.stl")
    root.append(asset)
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", pos="0 0 6", dir="0 0 -1", directional="true")
    if arena:
        ET.SubElement(wb, "geom", name="arena", type="mesh", mesh="arena",
                      contype="4", conaffinity="31", condim="3", friction="0.6", rgba="0.45 0.4 0.35 1")
    else:
        ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", contype="4", conaffinity="31",
                      condim="3", friction="0.6", rgba="0.4 0.4 0.4 1")
    if two_player:
        robots = [("a_", 1, (-1.4, 0, 0.793), (1, 0, 0, 0)), ("b_", 2, (1.4, 0, 0.793), (0, 0, 0, 1))]
    else:
        robots = [("a_", 1, (0, 0, 0.793), (1, 0, 0, 0))]
    act = ET.Element("actuator"); contact = ET.Element("contact")
    for prefix, bit, pos, quat in robots:
        pelvis, acts, excl = make_robot(src_root, prefix, bit, pos, quat)
        wb.append(pelvis); act.extend(acts); contact.extend(excl)
    wb.extend(make_pool())
    root.append(contact); root.append(act)
    return root


def default_qpos(m: mujoco.MjModel, prefixes: list[str]) -> np.ndarray:
    """Default pose with the pelvis lowered so the lowest foot geom just touches z=0."""
    d = mujoco.MjData(m)
    for p in prefixes:
        for j in JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + j)
            d.qpos[m.jnt_qposadr[jid]] = DEFAULT_POSE.get(j, 0.0)
    mujoco.mj_forward(m, d)
    for p in prefixes:
        feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, p + n) for n in
                ("left_foot_box_collision", "right_foot_box_collision")]
        low = min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in feet)
        root = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + "floating_base_joint")]
        d.qpos[root + 2] -= low
    return d.qpos.copy()


def hold_ctrl(m: mujoco.MjModel, q0: np.ndarray, rounds: int = 6, settle: float = 0.4) -> np.ndarray:
    """ctrl that holds q0 against gravity with the PD actuators: joint targets offset by the steady-state sag.
    Iterative: simulate briefly, add the remaining joint error to ctrl, repeat. Converges when the stance is stable."""
    d = mujoco.MjData(m)
    act_q = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    ctrl = q0[act_q].copy()
    for _ in range(rounds):
        d.qpos[:] = q0; d.qvel[:] = 0; d.ctrl[:] = ctrl; mujoco.mj_forward(m, d)
        for _ in range(int(settle / m.opt.timestep)):
            mujoco.mj_step(m, d)
        ctrl += q0[act_q] - d.qpos[act_q]
    return np.clip(ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


def dump_model(m: mujoco.MjModel) -> dict:
    name = lambda t, i: mujoco.mj_id2name(m, t, i)
    J, A, B, G = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
    return {
        "nq": m.nq, "nv": m.nv, "nu": m.nu, "nbody": m.nbody, "ngeom": m.ngeom,
        "timestep": m.opt.timestep, "integrator": int(m.opt.integrator), "iterations": m.opt.iterations,
        "ls_iterations": m.opt.ls_iterations, "solver": int(m.opt.solver), "cone": int(m.opt.cone),
        "gravity": m.opt.gravity.tolist(), "total_mass": float(sum(m.body_mass)),
        "joints": [{"name": name(J, i), "type": int(m.jnt_type[i]), "qposadr": int(m.jnt_qposadr[i]),
                    "dofadr": int(m.jnt_dofadr[i]), "range": m.jnt_range[i].tolist(),
                    "damping": float(m.dof_damping[m.jnt_dofadr[i]]), "armature": float(m.dof_armature[m.jnt_dofadr[i]]),
                    "frictionloss": float(m.dof_frictionloss[m.jnt_dofadr[i]]),
                    "actfrcrange": m.jnt_actfrcrange[i].tolist()} for i in range(m.njnt)],
        "actuators": [{"name": name(A, i), "joint": name(J, m.actuator_trnid[i][0]), "kp": float(m.actuator_gainprm[i][0]),
                       "kv": float(-m.actuator_biasprm[i][2]), "ctrlrange": m.actuator_ctrlrange[i].tolist(),
                       "forcerange": m.actuator_forcerange[i].tolist()}
                      for i in range(m.nu)],
        "bodies": [{"name": name(B, i), "mass": float(m.body_mass[i]), "ipos": m.body_ipos[i].tolist()} for i in range(m.nbody)],
        "geoms": [{"name": name(G, i), "type": int(m.geom_type[i]), "body": name(B, m.geom_bodyid[i]),
                   "size": m.geom_size[i].tolist(), "contype": int(m.geom_contype[i]), "conaffinity": int(m.geom_conaffinity[i]),
                   "friction": m.geom_friction[i].tolist(), "condim": int(m.geom_condim[i])}
                  for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]],
        "meshes": [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i), "nvert": int(m.mesh_vertnum[i]), "nface": int(m.mesh_facenum[i])}
                   for i in range(m.nmesh) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) == "arena"],
    }


def strip_for_unity(path_in: str, path_out: str):
    """Resolved XML minus elements the plugin cannot import."""
    tree = ET.parse(path_in); root = tree.getroot()
    for tag in ("keyframe", "sensor", "visual", "statistic"):
        for e in root.findall(tag):
            root.remove(e)
    tree.write(path_out, encoding="unicode", xml_declaration=False)


def build(two_player: bool, arena: bool | None = None):
    """Authoring XML -> MuJoCo-resolved XML. The resolved text is canonical:
    scene_<tag>_train.xml = resolved + keyframe (Python), scene_<tag>_unity.xml = resolved, stripped (Unity importer).
    Both compile to the identical model (asserted)."""
    src_root = ET.parse(SRC).getroot()
    arena = two_player if arena is None else arena
    tag = ("koth_" if arena else "flat_") + ("2p" if two_player else "1p")
    scene = build_scene(src_root, two_player, arena)
    ET.indent(scene)
    authoring = os.path.join(ASSETS, f"scene_{tag}.xml")
    ET.ElementTree(scene).write(authoring, encoding="unicode")
    resolved = os.path.join(ASSETS, f"scene_{tag}_train.xml")
    mujoco.mj_saveLastXML(resolved, mujoco.MjModel.from_xml_path(authoring))
    m = mujoco.MjModel.from_xml_path(resolved)
    prefixes = ["a_", "b_"] if two_player else ["a_"]
    q0 = default_qpos(m, prefixes)
    ctrl = hold_ctrl(m, q0)
    tree = ET.parse(resolved); root = tree.getroot()
    # MuJoCo's writer emits partial arrays in nested defaults (knee: biasprm="0 -300", kv inherited). The Unity
    # importer does not inherit array tails and read knee kv as 0. Write every actuator's numbers explicitly.
    for i, a in enumerate(root.find("actuator")):
        assert a.get("name") == mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        a.set("biastype", "affine")
        a.set("gainprm", f"{m.actuator_gainprm[i][0]:g}")
        a.set("biasprm", " ".join(f"{v:g}" for v in m.actuator_biasprm[i][:3]))
        a.set("ctrlrange", " ".join(f"{v:.9g}" for v in m.actuator_ctrlrange[i])); a.set("ctrllimited", "true")
        a.set("forcerange", " ".join(f"{v:g}" for v in m.actuator_forcerange[i])); a.set("forcelimited", "true")
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="default", qpos=" ".join(f"{v:.6g}" for v in q0), ctrl=" ".join(f"{v:.6g}" for v in ctrl))
    ET.indent(root); tree.write(resolved, encoding="unicode")
    unity = os.path.join(ASSETS, f"scene_{tag}_unity.xml")
    strip_for_unity(resolved, unity)
    m = mujoco.MjModel.from_xml_path(resolved)
    d1, d2 = dump_model(m), dump_model(mujoco.MjModel.from_xml_path(unity))
    assert d1 == d2, "train and unity XML compile to different models"
    json.dump(d1, open(os.path.join(ASSETS, f"model_dump_{tag}.json"), "w"), indent=1)
    aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_" + j) for j in JOINTS]
    build.hold_ctrl = [float(ctrl[i]) for i in aid]            # canonical order; identical for both robots
    build.key_root_z = float(q0[2])
    print(f"{tag}: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
          f"robot mass={(sum(m.body_mass) - POOL_N * POOL_MASS) / len(prefixes):.2f}kg  pelvis z0={q0[2]:.4f}")
    return m


if __name__ == "__main__":
    print("arena:", write_arena())
    build(two_player=False)
    build(two_player=False, arena=True)
    build(two_player=True)
    json.dump({"joints": JOINTS, "default_pose": [DEFAULT_POSE.get(j, 0.0) for j in JOINTS],
               "action_scale": ACTION_SCALE, "sim_dt": SIM_DT, "ctrl_dt": CTRL_DT, "decimation": round(CTRL_DT / SIM_DT),
               "robot_prefixes": ["a_", "b_"], "pool_bodies": [f"box{i}" for i in range(POOL_N)], "pool_park": [list(pool_park(i)) for i in range(POOL_N)],
               "ls_iterations": 10, "hold_ctrl": build.hold_ctrl, "key_root_z": build.key_root_z},
              open(os.path.join(ASSETS, "joint_map.json"), "w"), indent=1)
    print("wrote joint_map.json")
    # Unity reads these as TextAssets
    import shutil
    unity_models = os.path.normpath(os.path.join(HERE, "..", "..", "Assets", "PoKingHill", "Models"))
    os.makedirs(unity_models, exist_ok=True)
    for f in ("joint_map.json", "model_dump_flat_1p.json", "model_dump_koth_1p.json", "model_dump_koth_2p.json"):
        shutil.copyfile(os.path.join(ASSETS, f), os.path.join(unity_models, f))
    print("copied json to", unity_models)
