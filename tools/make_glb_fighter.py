"""Prepare a rigged GLB character as a fighter without Blender (numpy + Pillow; run with training/.venv's python):

    training/.venv/Scripts/python.exe tools/make_glb_fighter.py trump

Reads Assets/Art/Raw/<Name>/<Name>.glb (the editable original) and writes
  training/assets/<robot>/<robot>_rig.json   bone heads/tails and limb thickness, in MuJoCo coordinates and metres, in the
                                             format tools/make_kim.py writes (semantic bone names: Hips, UpperArm.L, ...),
                                             plus each bone's source name and the scale and turn that map the GLB onto it;
  Assets/PoKingHill/Fighters/<Name>/<robot>.glb  the same skinned mesh with 2k textures, imported by glTFast.

Unlike Kim's scan the character is used in its own rest pose (arms already hang down), so the GLB is not edited: the
physics zero pose is built from this rest pose (koth/build_human.py) and the Unity skin follower scales the character
to the physics body. glTF faces +Z with +X on the character's left, so MuJoCo (x forward, y left, z up) = (z, x, y).
"""
import io, json, os, struct, sys
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# robot -> (source folder name, height in metres, semantic bone -> GLB bones whose vertices it owns (first = its head),
#           bone whose head is this bone's tail for bones without a natural child)
FIGHTERS = {
    "trump": dict(name="Trump", height=1.90, bones={
        "Hips": ["bone_0"], "Spine": ["bone_1", "bone_2"], "Chest": ["bone_3"], "Neck": ["bone_4"], "Head": ["bone_5"],
        "Shoulder.L": ["bone_6"], "UpperArm.L": ["bone_7"], "LowerArm.L": ["bone_8"],
        "Hand.L": ["bone_9", "bone_10", "bone_11", "bone_12", "bone_13", "bone_14", "bone_15"],
        "Shoulder.R": ["bone_16"], "UpperArm.R": ["bone_17"], "LowerArm.R": ["bone_18"],
        "Hand.R": ["bone_19", "bone_20", "bone_21", "bone_22", "bone_23", "bone_24", "bone_25"],
        "UpperLeg.L": ["bone_26"], "LowerLeg.L": ["bone_27"], "Foot.L": ["bone_28"], "Toe.L": ["bone_29"],
        "UpperLeg.R": ["bone_30"], "LowerLeg.R": ["bone_31"], "Foot.R": ["bone_32"], "Toe.R": ["bone_33"]},
        parents={"Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck", "UpperLeg.L": "Hips", "UpperLeg.R": "Hips",
                 **{f"{b}.{s}": p for s in "LR" for b, p in (("Shoulder", "Chest"), ("UpperArm", f"Shoulder.{s}"), ("LowerArm", f"UpperArm.{s}"),
                                                              ("Hand", f"LowerArm.{s}"), ("LowerLeg", f"UpperLeg.{s}"), ("Foot", f"LowerLeg.{s}"), ("Toe", f"Foot.{s}"))}},
        tails={"Hand.L": "bone_15", "Hand.R": "bone_25"}),
}
TEXTURE_PX = 2048


def read_glb(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n]), b[20 + n + 8:]


def accessor(j, blob, i):
    a = j["accessors"][i]; v = j["bufferViews"][a["bufferView"]]
    dt = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a["componentType"]]
    k = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0); stride = v.get("byteStride", 0)
    if stride and stride != k * np.dtype(dt).itemsize: raise SystemExit("interleaved buffers are not supported")
    return np.frombuffer(blob, dt, a["count"] * k, off).reshape(a["count"], k)


def node_world(j):
    """World matrix of every node (glTF TRS)."""
    parent = {c: i for i, nd in enumerate(j["nodes"]) for c in nd.get("children", [])}
    def local(nd):
        if "matrix" in nd: return np.array(nd["matrix"]).reshape(4, 4).T
        x, y, z, w = nd.get("rotation", [0, 0, 0, 1])
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M = np.eye(4); M[:3, :3] = R @ np.diag(nd.get("scale", [1, 1, 1])); M[:3, 3] = nd.get("translation", [0, 0, 0]); return M
    out = {}
    def world(i):
        if i not in out: out[i] = (world(parent[i]) if i in parent else np.eye(4)) @ local(j["nodes"][i])
        return out[i]
    return [world(i) for i in range(len(j["nodes"]))]


def rig(robot, j, blob):
    spec = FIGHTERS[robot]; W = node_world(j)
    mesh_node = next(i for i, nd in enumerate(j["nodes"]) if "skin" in nd)
    prim = j["meshes"][j["nodes"][mesh_node]["mesh"]]["primitives"][0]; skin = j["skins"][j["nodes"][mesh_node]["skin"]]
    pos = accessor(j, blob, prim["attributes"]["POSITION"]).astype(float)
    pos = (W[mesh_node] @ np.c_[pos, np.ones(len(pos))].T).T[:, :3]
    jnt = accessor(j, blob, prim["attributes"]["JOINTS_0"]); wgt = accessor(j, blob, prim["attributes"]["WEIGHTS_0"])
    owner = jnt[np.arange(len(jnt)), wgt.argmax(1)]                       # skin joint index that weighs most on each vertex
    name_of = {j["nodes"][ni]["name"]: k for k, ni in enumerate(skin["joints"])}
    head_g = {nm: W[skin["joints"][k]][:3, 3] for nm, k in name_of.items()}
    scale = spec["height"] / (pos[:, 1].max() - pos[:, 1].min())
    to_mj = lambda p: np.array([p[..., 2], p[..., 0], p[..., 1] - pos[:, 1].min()]).T * scale if np.ndim(p) > 1 else \
        np.array([p[2], p[0], p[1] - pos[:, 1].min()]) * scale
    co = to_mj(pos)
    bones = {}
    for sem, src in spec["bones"].items():
        head = to_mj(head_g[src[0]])
        child = next((s for s, p in spec["parents"].items() if p == sem), None)
        tail_src = spec["tails"].get(sem) or (spec["bones"][child][0] if child else None)
        tail = to_mj(head_g[tail_src]) if tail_src else head + np.array([0, 0, 0.08])
        if sem == "Head": tail = np.array([head[0], head[1], co[:, 2].max()])
        if sem.startswith("Toe"): tail = head + np.array([0.04, 0, 0])
        if np.linalg.norm(tail - head) < 1e-6: tail = head + np.array([0, 0, 0.05])
        axis = (tail - head) / np.linalg.norm(tail - head)
        pts = co[np.isin(owner, [name_of[s] for s in src])]
        e = {"parent": spec["parents"].get(sem), "src": src[0], "head": head.round(4).tolist(), "tail": tail.round(4).tolist(), "verts": int(len(pts))}
        if len(pts) > 20:
            rel = pts - head; radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
            e.update(r50=round(float(np.percentile(radial, 50)), 4), r80=round(float(np.percentile(radial, 80)), 4),
                     lo=np.percentile(pts, 3, axis=0).round(4).tolist(), hi=np.percentile(pts, 97, axis=0).round(4).tolist())
        bones[sem] = e
    # every GLB bone the skin follower drives: the semantic bone that owns it (fingers follow the hand)
    follow = {s: sem for sem, src in spec["bones"].items() for s in src}
    src_pos = {s: to_mj(head_g[s]).round(4).tolist() for s in follow}
    return {"height": spec["height"], "scale_from_scan": round(float(scale), 5), "lo": co.min(0).round(4).tolist(), "hi": co.max(0).round(4).tolist(),
            "bones": bones, "source_bones": {s: {"bone": follow[s], "pos": src_pos[s]} for s in follow}}


def shrink_textures(j, blob):
    """Rebuild the binary chunk with every image at TEXTURE_PX (base colour as JPEG, data maps as PNG)."""
    tex = [m.get("pbrMetallicRoughness", {}).get("baseColorTexture") for m in j["materials"]]
    color = {j["textures"][t["index"]]["source"] for t in tex if t}
    replaced = {}
    for k, im in enumerate(j["images"]):
        v = j["bufferViews"][im["bufferView"]]
        img = Image.open(io.BytesIO(blob[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]))
        if max(img.size) > TEXTURE_PX: img = img.resize((TEXTURE_PX, TEXTURE_PX), Image.LANCZOS)
        out = io.BytesIO()
        if k in color: img.convert("RGB").save(out, "JPEG", quality=90); im["mimeType"] = "image/jpeg"
        else: img.save(out, "PNG", optimize=True); im["mimeType"] = "image/png"
        replaced[im["bufferView"]] = out.getvalue()
    parts, off = [], 0
    for i, v in enumerate(j["bufferViews"]):
        data = replaced.get(i, blob[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]])
        pad = (-off) % 4; parts.append(b"\0" * pad); off += pad
        v["byteOffset"] = off; v["byteLength"] = len(data); parts.append(data); off += len(data)
    new = b"".join(parts); new += b"\0" * ((-len(new)) % 4)
    j["buffers"] = [{"byteLength": len(new)}]
    return new


def write_glb(path, j, blob):
    js = json.dumps(j, separators=(",", ":")).encode(); js += b" " * ((-len(js)) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(blob), 0x004E4942)); f.write(blob)


def main(robot):
    spec = FIGHTERS[robot]
    src = os.path.join(REPO, "Assets", "Art", "Raw", spec["name"], spec["name"] + ".glb")
    j, blob = read_glb(src)
    r = rig(robot, j, blob)
    out_rig = os.path.join(REPO, "training", "assets", robot, f"{robot}_rig.json")
    os.makedirs(os.path.dirname(out_rig), exist_ok=True); json.dump(r, open(out_rig, "w"), indent=1)
    out_dir = os.path.join(REPO, "Assets", "PoKingHill", "Fighters", spec["name"]); os.makedirs(out_dir, exist_ok=True)
    blob2 = shrink_textures(j, blob); write_glb(os.path.join(out_dir, robot + ".glb"), j, blob2)
    print(f"{robot}: scale {r['scale_from_scan']} | bbox {r['lo']} {r['hi']} | unity glb {len(blob2) // 1024} KB")
    for n, e in r["bones"].items(): print(f"  {n:12s} head {e['head']}  r50 {e.get('r50')}  r80 {e.get('r80')}  verts {e['verts']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trump")
