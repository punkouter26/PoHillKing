"""Phase A asset generator. Single source of truth for everything both sims load.

Outputs (all under training/assets/):
  g1/assets/arena.stl          convex dome arena mesh (flat 3 m disc on top, convex slope)
  g1/scene_flat_1p.xml         one G1 on a plane + projectile pool          (R0/R1 training)
  g1/scene_koth_2p.xml         two G1 on the dome arena + projectile pool   (R2+ training)
  g1/scene_koth_4p.xml         four G1 on the arena (Unity free-for-all only; `python -m koth.build_mjcf 4p`)
  g1/scene_koth_5p.xml         four G1 (a_..d_) and Kim (e_) on the arena (`python -m koth.build_mjcf 5p`)
  g1/scene_koth_6p.xml         four G1, Kim (e_) and Trump (f_) (`python -m koth.build_mjcf 6p`)
  g1/scene_koth_game.xml       the game scene: two Kim, two Trump, two Grandpa, no G1 (`python -m koth.build_mjcf game`)
  g1/scene_koth_7p.xml         four G1, Kim (e_), Trump (f_) and Grandpa (g_): the former game scene, kept for the free-for-all gate (`7p`)
  g1/scene_*_train.xml         MuJoCo-resolved canonical model + keyframe (what training loads)
  g1/scene_*_unity.xml         same text minus keyframe/sensor (what the Unity importer loads)
  g1/model_dump_<tag>.json     parity reference per scene (nq/nv/nu, masses, ranges, gains, options)
  g1/joint_map.json            canonical 29 joint names + default pose + ctrl scale

Design constraints (from the org.mujoco 3.15 plugin audit): no <contact><pair>, no keyframes, no sensors,
timestep/gravity come from Unity settings, no ls_iterations/eulerdamp knobs -> implicitfast + MuJoCo defaults.
Collision bitmasks:  1=robot A  2=robot B  4=arena  8=box  16=foot/shin extra  32=robot C  64=robot D  128=Kim (e_)  256=Trump (f_)  512=Grandpa (g_)
"""
from __future__ import annotations
import copy, json, os, struct, xml.etree.ElementTree as ET
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = os.environ.get("KOTH_ROBOT", "g1")          # which fighter's body to build: g1 (default), kim, trump or grandpa
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets", ROBOT))
SRC = os.path.join(ASSETS, f"{ROBOT}_mjx.xml")
UNITY_MODELS = os.path.normpath(os.path.join(HERE, "..", "..", "Assets", "PoKingHill", "Models", *([] if ROBOT == "g1" else [ROBOT])))
LS_ITERATIONS = 10                                  # the plugin cannot import it: PolicyRunner writes it after init
SPAWN_Z = 0.793                                     # pelvis height in the authoring scene (the keyframe refines it)

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


def human(robot: str):
    """Profile module of a human body (koth/build_kim.py, koth/build_trump.py): same 29 joints, own pose, gains, height."""
    import importlib
    return importlib.import_module(f"koth.build_{robot}")


if ROBOT != "g1":
    DEFAULT_POSE, PD_GAINS, SPAWN_Z = human(ROBOT).DEFAULT_POSE, human(ROBOT).PD_GAINS, human(ROBOT).SPAWN_Z


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
from koth.rules import PLATEAU_R, SLOPE_K, arena_json     # z = -k (r-1.5)^2 -> 45 deg at r = 6 (rules.py is the one source)


def write_arena() -> str:
    """Convex dome: flat disc of radius PLATEAU_R at z=0, then z = -SLOPE_K (r-1.5)^2 down to HF_DEPTH.
    Written as binary STL (MuJoCo + the Unity plugin both load it; MuJoCo uses the convex hull, which equals the
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


# ---- robot ---------------------------------------------------------------------------------------
def _prefix(elem: ET.Element, p: str):
    for e in elem.iter():
        for a in ("name", "joint", "body1", "body2", "site", "target"):
            if a in e.attrib and e.tag != "mesh":
                e.set(a, p + e.get(a))


def make_robot(src_root: ET.Element, prefix: str, mask_bit: int, pos, quat, others: int | None = None) -> tuple[ET.Element, list[ET.Element], list[ET.Element]]:
    """Returns (pelvis body, actuator elements, exclude elements) for one robot. others = bits of the other robots."""
    pelvis = copy.deepcopy(src_root.find("worldbody/body[@name='pelvis']"))
    for junk in pelvis.findall("camera"):
        pelvis.remove(junk)
    pelvis.set("pos", " ".join(f"{v:g}" for v in pos))
    pelvis.set("quat", " ".join(f"{v:g}" for v in quat))
    other = 3 - mask_bit if others is None else others      # two robots: 1 <-> 2
    for g in pelvis.iter("geom"):
        cls = g.get("class", "")
        if cls == "visual":
            continue
        is_foot = "foot" in cls
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


ROBOT_BITS = [1, 2, 32, 64, 128, 256, 512, 1024]      # a_ .. h_ (4 ground, 8 boxes, 16 feet are taken)
PREFIXES = ["a_", "b_", "c_", "d_", "e_", "f_", "g_", "h_"]


def make_pool(n: int = POOL_N, robots_mask: int = 3) -> list[ET.Element]:
    """Pre-allocated projectile pool. Parked boxes float in the sky with no contacts: both sims re-pin parked boxes
    (qpos = park pose, qvel = 0) every control step instead of resting them on a shelf, which cost ~30 permanent
    contacts per world. Measured in mujoco_warp: 8 boxes on a shelf 1297 ms per control step, 2 pinned boxes 563 ms."""
    bodies = []
    for i in range(n):
        b = ET.Element("body", name=f"box{i}", pos=" ".join(f"{v:g}" for v in pool_park(i)))
        ET.SubElement(b, "freejoint", name=f"box{i}_free")
        ET.SubElement(b, "geom", name=f"box{i}_geom", type="box", size="0.1 0.1 0.1", mass=f"{POOL_MASS:g}",
                      contype="8", conaffinity=str(12 | robots_mask), condim="3", friction="0.5", rgba="0.9 0.3 0.1 1")
        bodies.append(b)
    return bodies


def build_scene(src_root: ET.Element, two_player: bool, arena: bool | None = None, players: int | None = None) -> ET.Element:
    arena = two_player if arena is None else arena
    n = players or (2 if two_player else 1)
    root = ET.Element("mujoco", model=f"{ROBOT}_koth_{n}p")
    ET.SubElement(root, "compiler", angle="radian", assetdir="assets", autolimits="true")
    # implicitfast: Unity-settable, and makes the eulerdamp flag irrelevant. iterations=5 like menagerie mjx.
    # ls_iterations: the Unity plugin cannot import it, so PolicyRunner writes model->opt.ls_iterations after init.
    ET.SubElement(root, "option", timestep=f"{SIM_DT}", integrator="implicitfast", iterations="5", ls_iterations=str(LS_ITERATIONS))
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
    mask = sum(ROBOT_BITS[:n]); others = lambda bit: None
    if n > 2:        # a ring on the plateau, everyone facing the centre
        robots = [(PREFIXES[i], ROBOT_BITS[i], (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n), SPAWN_Z),
                   (np.cos((2 * np.pi * i / n + np.pi) / 2), 0, 0, np.sin((2 * np.pi * i / n + np.pi) / 2))) for i in range(n)]
        others = lambda bit: mask - bit
    elif two_player:
        robots = [("a_", 1, (-1.4, 0, SPAWN_Z), (1, 0, 0, 0)), ("b_", 2, (1.4, 0, SPAWN_Z), (0, 0, 0, 1))]
    else:
        robots = [("a_", 1, (0, 0, SPAWN_Z), (1, 0, 0, 0))]
    act = ET.Element("actuator"); contact = ET.Element("contact")
    for prefix, bit, pos, quat in robots:
        pelvis, acts, excl = make_robot(src_root, prefix, bit, pos, quat, others(bit))
        wb.append(pelvis); act.extend(acts); contact.extend(excl)
    wb.extend(make_pool(robots_mask=mask if n > 2 else 3))
    root.append(contact); root.append(act)
    return root


def default_qpos(m: mujoco.MjModel, prefixes: list[str], poses: dict | None = None) -> np.ndarray:
    """Default pose with the pelvis lowered so the lowest foot geom just touches z=0. poses: prefix -> default pose
    for a robot whose pose is not this module's (Kim in the mixed scene)."""
    d = mujoco.MjData(m)
    for p in prefixes:
        pose = (poses or {}).get(p, DEFAULT_POSE)
        for j in JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + j)
            d.qpos[m.jnt_qposadr[jid]] = pose.get(j, 0.0)
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
    return np.clip(symmetric_hold(m, q0[act_q], ctrl), m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


def symmetric_hold(m: mujoco.MjModel, q0a: np.ndarray, ctrl: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Mirror-average the hold offsets (ctrl - q0) of every left/right actuator pair when any pair differs by more than
    tol rad (roll and yaw mirror with a sign flip). A double stance is statically indeterminate: the settle can end in a
    lateral lean that splits the weight unevenly (Trump: ankle roll offsets -0.016 / -0.002 rad, scripts/step_response.py
    2026-10-20). The G1 and Kim settle symmetric to 4e-5 rad and are left alone."""
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(m.nu)]
    pairs = [(i, names.index(n.replace("left_", "right_")), -1.0 if any(k in n for k in ("roll", "yaw")) else 1.0)
             for i, n in enumerate(names) if "left_" in n and n.replace("left_", "right_") in names]
    off = ctrl - q0a
    if not pairs or max(abs(off[l] - sg * off[r]) for l, r, sg in pairs) <= tol: return ctrl
    out = ctrl.copy()
    for l, r, sg in pairs:
        o = (off[l] + sg * off[r]) / 2; out[l] = q0a[l] + o; out[r] = q0a[r] + sg * o
    return out


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


def build(two_player: bool, arena: bool | None = None, players: int | None = None):
    """Authoring XML -> MuJoCo-resolved XML. The resolved text is canonical:
    scene_<tag>_train.xml = resolved + keyframe (Python), scene_<tag>_unity.xml = resolved, stripped (Unity importer).
    Both compile to the identical model (asserted)."""
    src_root = ET.parse(SRC).getroot()
    arena = two_player if arena is None else arena
    n = players or (2 if two_player else 1)
    tag = ("koth_" if arena else "flat_") + f"{n}p"
    scene = build_scene(src_root, two_player, arena, players)
    return finish(scene, tag, PREFIXES[:n])


# Human bodies added to the four G1 for the game scenes: prefix, collision bit, body kind, authoring position (x, y).
EXTRAS = [("e_", 128, "kim", (0.0, 0.0)), ("f_", 256, "trump", (0.5, 0.5)), ("g_", 512, "grandpa", (-0.5, -0.5))]


def build_mixed(n_extra: int = 1):
    """A game scene: four G1 and the first n_extra humans of EXTRAS (koth_5p: Kim; koth_6p: Kim and Trump; koth_7p: and Grandpa). Each
    human's body comes from assets/<robot>/<robot>_mjx.xml with its own default classes and gains; nothing here is
    used for training. Every robot collides with every other one and with the thrown boxes."""
    global PD_GAINS
    scene = build_scene(ET.parse(SRC).getroot(), True, True, players=4)
    extras = EXTRAS[:n_extra]; bits = sum(ROBOT_BITS[:4]) + sum(e[1] for e in extras)
    wb = scene.find("worldbody")
    for prefix, bit, robot, (x, y) in extras:
        P = human(robot); src = ET.parse(os.path.join(P.ASSETS, f"{robot}_mjx.xml")).getroot()
        g1_gains, PD_GAINS = PD_GAINS, P.PD_GAINS
        for cls in apply_pd_gains(copy.deepcopy(src.find("default"))): scene.find("default").append(cls)
        PD_GAINS = g1_gains
        pelvis, acts, excl = make_robot(src, prefix, bit, (x, y, P.SPAWN_Z), (1, 0, 0, 0), others=bits - bit)
        for g in pelvis.iter("geom"):          # and the boxes (the pool's mask covers the four G1 only)
            if g.get("conaffinity"): g.set("conaffinity", str(int(g.get("conaffinity")) | 8))
        wb.insert(list(wb).index(wb.find("body[@name='box0']")), pelvis)
        scene.find("actuator").extend(acts); scene.find("contact").extend(excl)
    tag = f"koth_{4 + n_extra}p"
    return finish(scene, tag, PREFIXES[:4] + [e[0] for e in extras], {e[0]: human(e[2]).DEFAULT_POSE for e in extras},
                  bodies={e[0]: e[2] for e in extras})


# The game's fighters (Demo_duel.unity, ParityBatch.BuildDuelDemo): two bodies of every human kind, so each brain can
# meet its own body (the only opponent it was trained against) as well as the others. A new rigged mesh joins here.
GAME_BODIES = ["kim", "kim", "trump", "trump", "grandpa", "grandpa"]


def build_game(bodies: list[str] | None = None):
    """The game scene koth_game: only human bodies (the G1 left the game on 2026-10-20), a ring on the summit, every
    body colliding with every other one and the thrown boxes. Built as a mixed scene (build_mix) and then stripped of the
    G1's mesh assets, which no body uses. Unity-only: no training rung reads it."""
    bodies = bodies or GAME_BODIES
    assert "g1" not in bodies and 2 <= len(bodies) <= len(PREFIXES), bodies
    return build_mix(bodies, tag="koth_game")


def build_mix(bodies: list[str], tag: str | None = None):
    """A training scene with mixed bodies (cross-body training, env rungs <learner>_vs_<other> and <learner>_ffa):
    bodies[i] sits at PREFIXES[i] on the arena, two facing each other or a ring of more. Built on the G1 scene (so the
    G1's meshes are at hand) with every non-G1 robot swapped for that human's body at the same place and collision
    bit; written to assets/g1 as scene_koth_<n>p_<body>_<body>..._train.xml. Run with ROBOT g1."""
    global PD_GAINS
    assert ROBOT == "g1", "build mixed scenes with KOTH_ROBOT=g1 (the G1's meshes live in assets/g1)"
    n = len(bodies); scene = build_scene(ET.parse(SRC).getroot(), True, True, players=n if n > 2 else None)
    mask = sum(ROBOT_BITS[:n]); wb, acts, contact = scene.find("worldbody"), scene.find("actuator"), scene.find("contact")
    merged = set()
    for i, body in enumerate(bodies):
        if body == "g1": continue
        prefix, bit = PREFIXES[i], ROBOT_BITS[i]
        g1 = wb.find(f"body[@name='{prefix}pelvis']"); at = list(wb).index(g1)
        x, y, _ = (float(v) for v in g1.get("pos").split()); quat = tuple(float(v) for v in g1.get("quat").split()); wb.remove(g1)
        for a in [a for a in acts if a.get("name", "").startswith(prefix)]: acts.remove(a)
        for e in [e for e in contact if e.get("body1", "").startswith(prefix)]: contact.remove(e)
        P = human(body); src = ET.parse(os.path.join(P.ASSETS, f"{body}_mjx.xml")).getroot()
        if body not in merged:
            g1_gains, PD_GAINS = PD_GAINS, P.PD_GAINS
            for cls in apply_pd_gains(copy.deepcopy(src.find("default"))): scene.find("default").append(cls)
            PD_GAINS = g1_gains; merged.add(body)
        pelvis, h_acts, excl = make_robot(src, prefix, bit, (x, y, P.SPAWN_Z), quat, others=mask - bit)
        for g in pelvis.iter("geom"):
            if g.get("conaffinity"): g.set("conaffinity", str(int(g.get("conaffinity")) | 8))      # thrown boxes
        wb.insert(at, pelvis); acts.extend(h_acts); contact.extend(excl)
    tag = tag or f"koth_{n}p_" + "_".join(bodies)
    if "g1" not in bodies:                     # no G1 body left: drop its meshes (the arena mesh stays)
        used = {g.get("mesh") for g in scene.iter("geom") if g.get("mesh")}
        asset = scene.find("asset")
        for m in [m for m in asset.findall("mesh") if m.get("name") not in used]: asset.remove(m)
    return finish(scene, tag, PREFIXES[:n], {PREFIXES[i]: human(b).DEFAULT_POSE for i, b in enumerate(bodies) if b != "g1"},
                  bodies={PREFIXES[i]: b for i, b in enumerate(bodies)})


def finish(scene: ET.Element, tag: str, prefixes: list[str], poses: dict | None = None, bodies: dict | None = None):
    """bodies: prefix -> body kind ("g1" / "kim" / "trump" / "grandpa") for robots that are not this module's ROBOT (written to the dump)."""
    ET.indent(scene)
    authoring = os.path.join(ASSETS, f"scene_{tag}.xml")
    ET.ElementTree(scene).write(authoring, encoding="unicode")
    resolved = os.path.join(ASSETS, f"scene_{tag}_train.xml")
    mujoco.mj_saveLastXML(resolved, mujoco.MjModel.from_xml_path(authoring))
    m = mujoco.MjModel.from_xml_path(resolved)
    q0 = default_qpos(m, prefixes, poses)
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
    d1["robots"] = [{"prefix": p, "body": (bodies or {}).get(p, ROBOT)} for p in prefixes]     # Unity: which body kind sits where
    json.dump(d1, open(os.path.join(ASSETS, f"model_dump_{tag}.json"), "w"), indent=1)
    aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_" + j) for j in JOINTS]
    build.hold_ctrl = [float(ctrl[i]) for i in aid]            # canonical order; identical for both robots
    build.key_root_z = float(q0[2])
    print(f"{tag}: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
          f"robot mass={(sum(m.body_mass) - POOL_N * POOL_MASS) / len(prefixes):.2f}kg  pelvis z0={q0[2]:.4f}")
    return m


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["mix"]:              # cross-body training scenes: python -m koth.build_mjcf mix trump g1 [g1 g1]
        build_mix(sys.argv[2:]); sys.exit(0)
    if sys.argv[1:2] == ["game"]:             # the game scene: python -m koth.build_mjcf game [kim kim trump trump ...]
        import shutil
        build_game(sys.argv[2:] or None)
        os.makedirs(UNITY_MODELS, exist_ok=True)
        shutil.copyfile(os.path.join(ASSETS, "model_dump_koth_game.json"), os.path.join(UNITY_MODELS, "model_dump_koth_game.json"))
        sys.exit(0)
    if sys.argv[1:] in (["4p"], ["5p"], ["6p"], ["7p"]):      # a Unity-only scene; leaves every training asset untouched
        import shutil
        tag = "koth_" + sys.argv[1]
        if tag in ("koth_5p", "koth_6p", "koth_7p"): build_mixed(int(sys.argv[1][0]) - 4)
        else: build(two_player=True, players=4)
        os.makedirs(UNITY_MODELS, exist_ok=True)
        shutil.copyfile(os.path.join(ASSETS, f"model_dump_{tag}.json"), os.path.join(UNITY_MODELS, f"model_dump_{tag}.json"))
        sys.exit(0)
    if ROBOT != "g1":
        human(ROBOT).build()
    os.makedirs(os.path.join(ASSETS, "assets"), exist_ok=True)
    print("arena:", write_arena())
    build(two_player=False)
    build(two_player=False, arena=True)
    build(two_player=True)
    json.dump({"joints": JOINTS, "default_pose": [DEFAULT_POSE.get(j, 0.0) for j in JOINTS],
               "action_scale": ACTION_SCALE, "sim_dt": SIM_DT, "ctrl_dt": CTRL_DT, "decimation": round(CTRL_DT / SIM_DT),
               "pool_bodies": [f"box{i}" for i in range(POOL_N)], "pool_park": [list(pool_park(i)) for i in range(POOL_N)],
               "ls_iterations": LS_ITERATIONS, "hold_ctrl": build.hold_ctrl, "key_root_z": build.key_root_z, "arena": arena_json()},
              open(os.path.join(ASSETS, "joint_map.json"), "w"), indent=1)
    print("wrote joint_map.json")
    # Unity reads these as TextAssets
    import shutil
    os.makedirs(UNITY_MODELS, exist_ok=True)
    for f in ("joint_map.json", "model_dump_flat_1p.json", "model_dump_koth_1p.json", "model_dump_koth_2p.json"):
        shutil.copyfile(os.path.join(ASSETS, f), os.path.join(UNITY_MODELS, f))
    print("copied json to", UNITY_MODELS)
