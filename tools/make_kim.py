"""Prepare the fighter "Kim" from Kim.glb. Run inside Blender:

    exec(open(r"<repo>/tools/make_kim.py").read())

Works in its own Blender scene ("Kim"). Steps:
  1. import the scan, drop its test animation, scale it to HEIGHT_M, turn it to face +X and stand it on z = 0, so
     Blender coordinates are MuJoCo coordinates (x forward, y left, z up, metres);
  2. lower the arms from the T-pose to hang straight down and make that the new rest pose. The physics model
     (training/koth/build_kim.py) uses this rest pose as its zero pose, so in Unity a bone simply follows its body;
  3. measure each limb's thickness from the vertices skinned to it;
  4. write training/assets/kim/kim_rig.json and export Assets/PoKingHill/Fighters/Kim/kim.fbx + textures.
"""
import json, math, os
import bpy
import numpy as np
from mathutils import Matrix, Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "."))))
GLB = os.path.join(REPO, "Kim.glb")
OUT_UNITY = os.path.join(REPO, "Assets", "PoKingHill", "Fighters", "Kim")
OUT_RIG = os.path.join(REPO, "training", "assets", "kim", "kim_rig.json")
HEIGHT_M = 1.60
ARM_CHAIN = ["UpperArm", "LowerArm", "Hand"]


def ops(obj, mode="OBJECT"):
    """Make obj the only selected, active object in the given mode."""
    if bpy.context.object and bpy.context.object.mode != "OBJECT": bpy.ops.object.mode_set(mode="OBJECT")
    for o in bpy.context.selected_objects: o.select_set(False)
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    if mode != "OBJECT": bpy.ops.object.mode_set(mode=mode)


scene = bpy.data.scenes.get("Kim") or bpy.data.scenes.new("Kim")
bpy.context.window.scene = scene
for o in list(scene.objects): bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.import_scene.gltf(filepath=GLB)
rig = next(o for o in scene.objects if o.type == "ARMATURE")
body = next(o for o in scene.objects if o.type == "MESH" and o.parent == rig)
for o in list(scene.objects):
    if o not in (rig, body): bpy.data.objects.remove(o, do_unlink=True)
rig.animation_data_clear()
for pb in rig.pose.bones: pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update()

# 1. size, heading, ground. The scan faces -Y; a quarter turn about Z makes it face +X.
co = np.array([(body.matrix_world @ v.co)[:] for v in body.data.vertices])
scale = HEIGHT_M / (co[:, 2].max() - co[:, 2].min())
hips = rig.matrix_world @ rig.data.bones["Hips"].head_local
fix = Matrix.Rotation(math.radians(90), 4, "Z") @ Matrix.Scale(scale, 4) @ Matrix.Translation((-hips.x, -hips.y, -co[:, 2].min()))
rig.matrix_world = fix @ rig.matrix_world
bpy.context.view_layer.update()
ops(rig); body.select_set(True); bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# 2. arms down, then freeze that pose as the rest pose (mesh first, then the armature).
ops(rig, "POSE")
for side in ("L", "R"):
    for name in ARM_CHAIN:
        pb = rig.pose.bones[f"{name}.{side}"]
        turn = (pb.tail - pb.head).rotation_difference(Vector((0, 0, -1))).to_matrix().to_4x4()
        pb.matrix = Matrix.Translation(pb.head) @ turn @ Matrix.Translation(-pb.head) @ pb.matrix
        bpy.context.view_layer.update()
skin = next(m for m in body.modifiers if m.type == "ARMATURE")
ops(body); bpy.ops.object.modifier_copy(modifier=skin.name); bpy.ops.object.modifier_apply(modifier=skin.name)
ops(rig, "POSE"); bpy.ops.pose.armature_apply(selected=False); bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()

# 3. limb thickness: each vertex belongs to the bone that weighs most on it.
co = np.array([v.co[:] for v in body.data.vertices])
names = [g.name for g in body.vertex_groups]; owner = np.full(len(co), -1)
for i, v in enumerate(body.data.vertices):
    if v.groups: owner[i] = max(v.groups, key=lambda g: g.weight).group
bones = {}
for b in rig.data.bones:
    head, tail = np.array(b.head_local), np.array(b.tail_local); axis = (tail - head) / np.linalg.norm(tail - head)
    pts = co[owner == names.index(b.name)] if b.name in names else np.zeros((0, 3))
    entry = {"parent": b.parent.name if b.parent else None, "head": head.round(4).tolist(), "tail": tail.round(4).tolist(), "verts": int(len(pts))}
    if len(pts) > 20:
        rel = pts - head; radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
        entry.update(r50=round(float(np.percentile(radial, 50)), 4), r80=round(float(np.percentile(radial, 80)), 4),
                     lo=np.percentile(pts, 3, axis=0).round(4).tolist(), hi=np.percentile(pts, 97, axis=0).round(4).tolist())
    bones[b.name] = entry
os.makedirs(os.path.dirname(OUT_RIG), exist_ok=True)
json.dump({"height": HEIGHT_M, "scale_from_scan": round(scale, 5), "lo": co.min(0).round(4).tolist(), "hi": co.max(0).round(4).tolist(), "bones": bones},
          open(OUT_RIG, "w"), indent=1)

# 4. Unity assets: 2k textures (the scan's 4k maps are 30 MB) and the skinned FBX.
os.makedirs(OUT_UNITY, exist_ok=True)
nodes = body.data.materials[0].node_tree.nodes
bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")


def texture(socket, name, fmt):
    link = bsdf.inputs[socket].links[0].from_node if bsdf.inputs[socket].links else None
    while link is not None and link.type != "TEX_IMAGE":
        link = next((i.links[0].from_node for i in link.inputs if i.links), None)
    if link is None: return None
    img = link.image.copy(); img.scale(2048, 2048)
    img.filepath_raw = os.path.join(OUT_UNITY, name); img.file_format = fmt; img.save(); bpy.data.images.remove(img)
    return name


saved = [texture("Base Color", "kim_albedo.png", "PNG"), texture("Normal", "kim_normal.png", "PNG")]
ops(rig); body.select_set(True)
bpy.ops.export_scene.fbx(filepath=os.path.join(OUT_UNITY, "kim.fbx"), use_selection=True, object_types={"ARMATURE", "MESH"},
                         apply_scale_options="FBX_SCALE_ALL", add_leaf_bones=False, bake_anim=False, mesh_smooth_type="FACE",
                         use_tspace=True, use_armature_deform_only=True)
print("kim: scale", round(scale, 4), "| bbox", co.min(0).round(3), co.max(0).round(3), "| textures", saved)
for n, e in bones.items(): print(f"  {n:12s} head {e['head']}  r50 {e.get('r50')}  r80 {e.get('r80')}  verts {e['verts']}")
