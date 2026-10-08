"""Phase A asset generator. Single source of truth for everything both sims load.

Outputs (all under training/assets/):
  g1/assets/arena.bin          MuJoCo custom-binary hfield (int32 nrow, int32 ncol, float32 data in [0,1])
  g1/arena.npy                 same data as float32 array (Unity editor script reads this too)
  g1/scene_flat_1p.xml         one G1 on a plane + projectile pool          (R0/R1 training)
  g1/scene_koth_2p.xml         two G1 on the hfield arena + projectile pool (R2+ training)
  g1/scene_*_train.xml         MuJoCo-resolved canonical model + keyframe (what training loads)
  g1/scene_*_unity.xml         same text minus keyframe/sensor (what the Unity importer loads)
  g1/model_dump.json           parity reference (nq/nv/nu, masses, ranges, gains, options, hfield meta)
  g1/joint_map.json            canonical 29 joint names + default pose + ctrl scale

Design constraints (from the org.mujoco 3.15 plugin audit): no <contact><pair>, no keyframes, no sensors,
timestep/gravity come from Unity settings, no ls_iterations/eulerdamp knobs -> implicitfast + MuJoCo defaults.
Collision bitmasks:  1=robot A  2=robot B  4=arena  8=box  16=foot/shin extra  32=parking shelf
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

# ---- arena ---------------------------------------------------------------------------------------
HF_N = 257            # Unity Terrain heightmap resolution must be 2^n+1
HF_RADIUS = 10.0      # half-extent in metres (20 m x 20 m)
HF_DEPTH = 10.0       # elevation scale: plateau (data=1) sits at z=0, data=0 at z=-10
PLATEAU_R = 1.5
SLOPE_K = 1.0 / 9.0   # z = -k (r-1.5)^2  -> 45 deg at r = 6
ROUGH_AMP = 0.02
RIM_NOISE_FADE = 1.0  # roughness ramps from 0 at the rim to full at rim + 1 m


def arena_heights(seed: int = 0) -> np.ndarray:
    xs = np.linspace(-HF_RADIUS, HF_RADIUS, HF_N)
    x, y = np.meshgrid(xs, xs)                       # row -> y, col -> x (MuJoCo hfield convention)
    r = np.hypot(x, y)
    z = -SLOPE_K * np.clip(r - PLATEAU_R, 0, None) ** 2
    fade = np.clip((r - PLATEAU_R) / RIM_NOISE_FADE, 0, 1)
    z += ROUGH_AMP * np.random.default_rng(seed).uniform(-1, 1, z.shape) * fade
    z = np.clip(z, -HF_DEPTH, 0)
    data = (z + HF_DEPTH) / HF_DEPTH                 # [0,1], plateau = 1 exactly
    data[r <= PLATEAU_R] = 1.0
    return data.astype(np.float32)


def write_arena() -> np.ndarray:
    data = arena_heights()
    with open(os.path.join(ASSETS, "assets", "arena.bin"), "wb") as f:
        f.write(struct.pack("<ii", HF_N, HF_N)); f.write(data.tobytes())
    np.save(os.path.join(ASSETS, "arena.npy"), data)
    return data


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


def make_pool(n: int = 8) -> list[ET.Element]:
    bodies = []
    for i in range(n):
        b = ET.Element("body", name=f"box{i}", pos=f"{-1.75 + 0.5 * i:g} 0 -50")
        ET.SubElement(b, "freejoint", name=f"box{i}_free")
        ET.SubElement(b, "geom", name=f"box{i}_geom", type="box", size="0.1 0.1 0.1", mass="2",
                      contype="8", conaffinity="47", condim="3", friction="0.5", rgba="0.9 0.3 0.1 1")
        bodies.append(b)
    shelf = ET.Element("geom", name="parking_shelf", type="box", size="3 0.5 0.1", pos="0 0 -50.2",
                       contype="32", conaffinity="8", rgba="0 0 0 0")
    return bodies + [shelf]


def build_scene(src_root: ET.Element, two_player: bool) -> ET.Element:
    root = ET.Element("mujoco", model="g1_koth_2p" if two_player else "g1_koth_1p")
    ET.SubElement(root, "compiler", angle="radian", assetdir="assets", autolimits="true")
    # implicitfast: Unity-settable, and makes the eulerdamp flag irrelevant. iterations=5 like menagerie mjx.
    ET.SubElement(root, "option", timestep=f"{SIM_DT}", integrator="implicitfast", iterations="5")
    root.append(copy.deepcopy(src_root.find("default")))
    asset = copy.deepcopy(src_root.find("asset"))
    if two_player:
        ET.SubElement(asset, "hfield", name="arena", file="arena.bin", nrow=str(HF_N), ncol=str(HF_N),
                      size=f"{HF_RADIUS} {HF_RADIUS} {HF_DEPTH} 0.5")
    root.append(asset)
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", pos="0 0 6", dir="0 0 -1", directional="true")
    if two_player:
        ET.SubElement(wb, "geom", name="arena", type="hfield", hfield="arena", pos=f"0 0 {-HF_DEPTH}",
                      contype="4", conaffinity="31", condim="3", friction="0.6", rgba="0.45 0.4 0.35 1")
        robots = [("a_", 1, (-1.4, 0, 0.793), (1, 0, 0, 0)), ("b_", 2, (1.4, 0, 0.793), (0, 0, 0, 1))]
    else:
        ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", contype="4", conaffinity="31",
                      condim="3", friction="0.6", rgba="0.4 0.4 0.4 1")
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
                       "kv": float(-m.actuator_biasprm[i][2]), "ctrlrange": m.actuator_ctrlrange[i].tolist()}
                      for i in range(m.nu)],
        "bodies": [{"name": name(B, i), "mass": float(m.body_mass[i]), "ipos": m.body_ipos[i].tolist()} for i in range(m.nbody)],
        "geoms": [{"name": name(G, i), "type": int(m.geom_type[i]), "body": name(B, m.geom_bodyid[i]),
                   "size": m.geom_size[i].tolist(), "contype": int(m.geom_contype[i]), "conaffinity": int(m.geom_conaffinity[i]),
                   "friction": m.geom_friction[i].tolist(), "condim": int(m.geom_condim[i])}
                  for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]],
        "hfield": [{"nrow": int(m.hfield_nrow[i]), "ncol": int(m.hfield_ncol[i]), "size": m.hfield_size[i].tolist(),
                    "data_min": float(m.hfield_data[m.hfield_adr[i]:m.hfield_adr[i] + m.hfield_nrow[i] * m.hfield_ncol[i]].min()),
                    "data_max": float(m.hfield_data[m.hfield_adr[i]:m.hfield_adr[i] + m.hfield_nrow[i] * m.hfield_ncol[i]].max())}
                   for i in range(m.nhfield)],
    }


def strip_for_unity(path_in: str, path_out: str):
    """Resolved XML minus elements the plugin cannot import."""
    tree = ET.parse(path_in); root = tree.getroot()
    for tag in ("keyframe", "sensor", "visual", "statistic"):
        for e in root.findall(tag):
            root.remove(e)
    tree.write(path_out, encoding="unicode", xml_declaration=False)


def build(two_player: bool):
    """Authoring XML -> MuJoCo-resolved XML. The resolved text is canonical:
    scene_<tag>_train.xml = resolved + keyframe (Python), scene_<tag>_unity.xml = resolved, stripped (Unity importer).
    Both compile to the identical model (asserted)."""
    src_root = ET.parse(SRC).getroot()
    tag = "koth_2p" if two_player else "flat_1p"
    scene = build_scene(src_root, two_player)
    ET.indent(scene)
    authoring = os.path.join(ASSETS, f"scene_{tag}.xml")
    ET.ElementTree(scene).write(authoring, encoding="unicode")
    resolved = os.path.join(ASSETS, f"scene_{tag}_train.xml")
    mujoco.mj_saveLastXML(resolved, mujoco.MjModel.from_xml_path(authoring))
    m = mujoco.MjModel.from_xml_path(resolved)
    prefixes = ["a_", "b_"] if two_player else ["a_"]
    q0 = default_qpos(m, prefixes)
    ctrl = np.zeros(m.nu)
    for i in range(m.nu):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0])
        ctrl[i] = DEFAULT_POSE.get(jn[2:], 0.0)
    tree = ET.parse(resolved); root = tree.getroot()
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="default", qpos=" ".join(f"{v:.6g}" for v in q0), ctrl=" ".join(f"{v:.6g}" for v in ctrl))
    ET.indent(root); tree.write(resolved, encoding="unicode")
    unity = os.path.join(ASSETS, f"scene_{tag}_unity.xml")
    strip_for_unity(resolved, unity)
    m = mujoco.MjModel.from_xml_path(resolved)
    d1, d2 = dump_model(m), dump_model(mujoco.MjModel.from_xml_path(unity))
    assert d1 == d2, "train and unity XML compile to different models"
    json.dump(d1, open(os.path.join(ASSETS, f"model_dump_{tag}.json"), "w"), indent=1)
    print(f"{tag}: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
          f"robot mass={(sum(m.body_mass) - 16.0) / len(prefixes):.2f}kg  pelvis z0={q0[2]:.4f}")
    return m


if __name__ == "__main__":
    data = write_arena()
    print(f"arena: {HF_N}x{HF_N}, min {data.min():.3f} max {data.max():.3f}, plateau cells {(data == 1).sum()}")
    build(two_player=False)
    build(two_player=True)
    json.dump({"joints": JOINTS, "default_pose": [DEFAULT_POSE.get(j, 0.0) for j in JOINTS],
               "action_scale": ACTION_SCALE, "sim_dt": SIM_DT, "ctrl_dt": CTRL_DT, "decimation": round(CTRL_DT / SIM_DT),
               "robot_prefixes": ["a_", "b_"], "pool_bodies": [f"box{i}" for i in range(8)]},
              open(os.path.join(ASSETS, "joint_map.json"), "w"), indent=1)
    print("wrote joint_map.json")
