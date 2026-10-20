"""Prepare the fighter "Grandpa" from Assets/Art/Raw/Grandpa/Grandpa.glb, an unrigged textured mesh (an AI-generated
character standing with one arm out to the side and one forward). Run in headless Blender:

    blender -b --factory-startup --python tools/make_grandpa.py

Works in its own Blender scene ("Grandpa"). Steps:
  1. import, clean (merge split vertices, drop loose parts, consistent normals, triangles to quads where UVs allow),
     scale to HEIGHT_M, turn to face +X and stand the feet on z = 0 with the hips over the origin, apply transforms,
     so Blender coordinates are MuJoCo coordinates (x forward, y left, z up, metres);
  2. build the armature with the bone names tools/make_kim.py uses (Hips, Spine, ..., UpperArm.L, Toe.R): joint heights
     from adult proportions (Drillis & Contini), horizontal positions from the mesh, each arm along the centre line of
     its sleeve; skin with automatic weights (bone heat), or through a voxel-remeshed proxy when bone heat leaves
     vertices unweighted;
  3. T-pose: arms straight out sideways, palms down, legs straight under the hips, feet forward; made the rest pose.
     Exported with a walk cycle as Assets/Art/Exports/FBX/Grandpa/Character_TPose.fbx (textures embedded) and
     Assets/Art/Exports/GLB/Grandpa.glb; the walk is checked for stretched edges and rendered;
  4. game rest pose: arms hanging ARM_OUT_DEG out from the body, made the rest pose (the physics zero pose,
     training/koth/build_grandpa.py), limb thickness measured from the vertices each bone owns; writes
     training/assets/grandpa/grandpa_rig.json and Assets/PoKingHill/Fighters/Grandpa/grandpa.fbx + 2k textures;
  5. archives the scene as Assets/Art/Raw/Grandpa~/Grandpa.blend (the ~ keeps Unity from importing it).
Blender's stdout does not come back through the Store build's launcher, so everything is logged to LOG.
"""
import json, math, os, sys, traceback
import bmesh, bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "."))))
GLB = os.path.join(REPO, "Assets", "Art", "Raw", "Grandpa", "Grandpa.glb")
OUT_TPOSE = os.path.join(REPO, "Assets", "Art", "Exports", "FBX", "Grandpa", "Character_TPose.fbx")
OUT_GLB = os.path.join(REPO, "Assets", "Art", "Exports", "GLB", "Grandpa.glb")
OUT_UNITY = os.path.join(REPO, "Assets", "PoKingHill", "Fighters", "Grandpa")
OUT_RIG = os.path.join(REPO, "training", "assets", "grandpa", "grandpa_rig.json")
OUT_BLEND = os.path.join(REPO, "Assets", "Art", "Raw", "Grandpa~", "Grandpa.blend")
SHOTS = os.path.join(REPO, "SCREENSHOTS", "grandpa-261020")
LOG = os.path.join(SHOTS, "make_grandpa.log")
HEIGHT_M = 1.72
ARM_OUT_DEG = 20.0          # hanging arms clear his belly from about 18 degrees out
WALK_FRAMES = 32            # one stride at 30 fps
PARENTS = {"Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck", "UpperLeg.L": "Hips", "UpperLeg.R": "Hips",
           **{f"{b}.{s}": p for s in "LR" for b, p in (("Shoulder", "Chest"), ("UpperArm", f"Shoulder.{s}"), ("LowerArm", f"UpperArm.{s}"),
                                                        ("Hand", f"LowerArm.{s}"), ("LowerLeg", f"UpperLeg.{s}"), ("Foot", f"LowerLeg.{s}"), ("Toe", f"Foot.{s}"))}}
CONNECTED = {"Chest", "Neck", "Head", "LowerArm.L", "LowerArm.R", "Hand.L", "Hand.R", "LowerLeg.L", "LowerLeg.R", "Foot.L", "Foot.R", "Toe.L", "Toe.R"}


def say(*a):
    print(*a, flush=True)


def ops(obj, mode="OBJECT"):
    """Make obj the only selected, active object in the given mode."""
    if bpy.context.object and bpy.context.object.mode != "OBJECT": bpy.ops.object.mode_set(mode="OBJECT")
    for o in bpy.context.selected_objects: o.select_set(False)
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    if mode != "OBJECT": bpy.ops.object.mode_set(mode=mode)


def verts(obj):
    return np.array([v.co[:] for v in obj.data.vertices])


def non_manifold(obj):
    bm = bmesh.new(); bm.from_mesh(obj.data)
    n = sum(1 for e in bm.edges if not e.is_manifold), sum(1 for v in bm.verts if not v.link_edges)
    bm.free(); return n


# ---------------------------------------------------------------- 1. import and clean
def import_clean():
    scene = bpy.data.scenes.get("Grandpa") or bpy.data.scenes.new("Grandpa")
    bpy.context.window.scene = scene
    for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)      # the factory scene's cube and light too
    bpy.ops.import_scene.gltf(filepath=GLB)
    body = next(o for o in scene.objects if o.type == "MESH")
    for o in list(scene.objects):
        if o is not body: bpy.data.objects.remove(o, do_unlink=True)
    body.name = body.data.name = "Grandpa"
    ops(body); bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    n0 = len(body.data.vertices); nm0 = non_manifold(body)
    ops(body, "EDIT"); bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.select_all(action="SELECT"); bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.select_all(action="SELECT"); bpy.ops.mesh.tris_convert_to_quads(uvs=True, seam=True, sharp=True, materials=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    # The glTF importer leaves the character facing -Y (glTF +Z): a quarter turn about Z faces it +X, then scale and ground.
    co = verts(body)
    scale = HEIGHT_M / (co[:, 2].max() - co[:, 2].min())
    body.matrix_world = Matrix.Rotation(math.radians(90), 4, "Z") @ Matrix.Scale(scale, 4) @ Matrix.Translation((0, 0, -co[:, 2].min()))
    ops(body); bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    co = verts(body)
    # hips over the origin: x of the trunk between belt and chest, y halfway between the legs
    legs = co[(co[:, 2] > 0.15) & (co[:, 2] < 0.5)]; my = np.median(legs[:, 1])
    mid_y = (np.median(legs[legs[:, 1] > my, 1]) + np.median(legs[legs[:, 1] <= my, 1])) / 2
    torso = co[(co[:, 2] > 0.95) & (co[:, 2] < 1.15)]
    body.matrix_world = Matrix.Translation((-np.median(torso[:, 0]), -mid_y, 0))
    ops(body); bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    q = sum(1 for p in body.data.polygons if len(p.vertices) == 4)
    nm1 = non_manifold(body)
    say(f"1. mesh: {n0} -> {len(body.data.vertices)} vertices, {len(body.data.polygons)} faces ({q} quads), "
        f"non-manifold edges {nm0[0]} -> {nm1[0]}, loose vertices {nm1[1]}, scale {scale:.4f}, origin at the feet")
    return body, scale, {"vertices_in": n0, "vertices": len(body.data.vertices), "faces": len(body.data.polygons), "quads": q,
                         "non_manifold_before": nm0[0], "non_manifold_after": nm1[0], "loose": nm1[1]}


# ---------------------------------------------------------------- 2. armature and weights
def landmarks(co, H):
    """Bone (head, tail) of the semantic rig, in the character's current pose (MuJoCo frame, metres)."""
    out = {}
    def ring(z, side=None, dz=0.012, xmax=None):
        s = co[np.abs(co[:, 2] - z) < dz]
        if side == "L": s = s[s[:, 1] > 0.0]
        if side == "R": s = s[s[:, 1] < 0.0]
        if xmax is not None: s = s[s[:, 0] < xmax]
        return (s.min(0) + s.max(0)) / 2
    z_hip, z_knee, z_ankle = 0.53 * H, 0.285 * H, 0.05 * H
    for side in "LR":
        sg = 1 if side == "L" else -1
        knee = ring(z_knee, side); ankle = ring(z_ankle + 0.04, side, xmax=0.05); ankle[2] = z_ankle
        hip = np.array([0.0, ring(0.36 * H, side)[1] * 0.9, z_hip])
        # toe joint 6 cm short of the toe tip, along the foot (his right foot is turned out and set back)
        feet = co[(co[:, 2] < 0.06) & (sg * co[:, 1] > 0)]
        tip = feet[(feet[:, :2] - ankle[:2]) @ np.array([1.0, 0.0]) > 0]; tip = tip[np.linalg.norm(tip[:, :2] - ankle[:2], axis=1).argmax()]
        fwd = (tip[:2] - ankle[:2]) / np.linalg.norm(tip[:2] - ankle[:2])
        toe = np.array([*(tip[:2] - 0.06 * fwd), 0.035])
        out[f"UpperLeg.{side}"] = (hip, knee); out[f"LowerLeg.{side}"] = (knee, ankle)
        out[f"Foot.{side}"] = (ankle, toe); out[f"Toe.{side}"] = (toe, toe + np.array([0.05, 0, 0]))
    head_c = ring(0.92 * H); neck_z = 0.835 * H
    out["Hips"] = (np.array([0, 0, z_hip + 0.02]), np.array([0, 0, 0.60 * H]))
    out["Spine"] = (np.array([0, 0, 0.60 * H]), np.array([0, 0, 0.72 * H]))
    out["Chest"] = (np.array([0, 0, 0.72 * H]), np.array([-0.02, 0, neck_z]))
    out["Neck"] = (np.array([-0.02, 0, neck_z]), np.array([head_c[0] - 0.02, 0, 0.88 * H]))
    out["Head"] = (np.array([head_c[0] - 0.02, 0, 0.88 * H]), np.array([head_c[0] - 0.02, 0, H]))
    z_sh = 0.80 * H
    for side in "LR":
        sg = 1 if side == "L" else -1
        sh = np.array([-0.02, sg * 0.185, z_sh]); clav = np.array([-0.01, sg * 0.03, z_sh + 0.01])
        arm = co[(co[:, 2] > 0.61 * H) & (sg * co[:, 1] > 0)]
        tip = arm[np.linalg.norm(arm - sh, axis=1).argmax()]
        # the sleeve's centre line: bounding-box centres of thin slices across the shoulder -> fingertip line
        u = (tip - sh) / np.linalg.norm(tip - sh); L = np.linalg.norm(tip - sh); pts = [sh]
        for t in np.arange(0.08, L - 0.02, 0.03):
            q = arm[np.abs((arm - sh) @ u - t) < 0.02]
            q = q[np.linalg.norm((q - sh) - np.outer((q - sh) @ u, u), axis=1) < 0.12]
            if len(q) > 5: pts.append((q.min(0) + q.max(0)) / 2)
        pts = np.array(pts + [tip]); seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        at = lambda s: np.array([np.interp(s, seg, pts[:, k]) for k in range(3)])
        elbow, wrist = at(0.186 * H), at(0.332 * H)
        out[f"Shoulder.{side}"] = (clav, sh); out[f"UpperArm.{side}"] = (sh, elbow)
        out[f"LowerArm.{side}"] = (elbow, wrist); out[f"Hand.{side}"] = (wrist, tip)
    return out


def build_armature(body):
    J = landmarks(verts(body), HEIGHT_M)
    arm = bpy.data.armatures.new("GrandpaRig"); rig = bpy.data.objects.new("GrandpaRig", arm)
    bpy.context.scene.collection.objects.link(rig)
    ops(rig, "EDIT")
    for name, (h, t) in J.items():
        eb = arm.edit_bones.new(name); eb.head = Vector(h); eb.tail = Vector(t)
    for name, p in PARENTS.items():
        eb = arm.edit_bones[name]; eb.parent = arm.edit_bones[p]
        if name in CONNECTED and (eb.head - eb.parent.tail).length < 1e-4: eb.use_connect = True
    for eb in arm.edit_bones:     # roll: bone Z toward the character's front (+X), up for the feet
        eb.align_roll(Vector((0, 0, 1)) if eb.name.startswith(("Foot", "Toe")) else Vector((1, 0, 0)))
    bpy.ops.object.mode_set(mode="OBJECT")
    say("2. armature: " + ", ".join(f"{b.name} {tuple(round(x, 3) for x in b.head_local)}" for b in arm.bones))
    return rig


def skin(body, rig):
    ops(body); rig.select_set(True); bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    unweighted = sum(1 for v in body.data.vertices if not any(g.weight > 1e-4 for g in v.groups))
    method = "bone heat"
    if unweighted > 0:
        # Bone heat fails on meshes that are not closed. Weigh a closed voxel remesh instead and copy its weights over.
        say(f"   bone heat left {unweighted} vertices unweighted: weighing a voxel proxy")
        for g in list(body.vertex_groups): body.vertex_groups.remove(g)
        proxy = body.copy(); proxy.data = body.data.copy(); bpy.context.scene.collection.objects.link(proxy)
        proxy.modifiers.clear(); proxy.parent = None
        ops(proxy); m = proxy.modifiers.new("remesh", "REMESH"); m.mode = "VOXEL"; m.voxel_size = 0.012
        bpy.ops.object.modifier_apply(modifier=m.name)
        ops(proxy); rig.select_set(True); bpy.context.view_layer.objects.active = rig
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        for b in rig.data.bones:
            if b.name not in body.vertex_groups: body.vertex_groups.new(name=b.name)
        dt = body.modifiers.new("weights", "DATA_TRANSFER"); dt.object = proxy
        dt.use_vert_data = True; dt.data_types_verts = {"VGROUP_WEIGHTS"}; dt.vert_mapping = "POLYINTERP_NEAREST"
        dt.layers_vgroup_select_src = "ALL"; dt.layers_vgroup_select_dst = "NAME"
        ops(body); bpy.ops.object.modifier_move_to_index(modifier=dt.name, index=0); bpy.ops.object.modifier_apply(modifier=dt.name)
        bpy.data.objects.remove(proxy, do_unlink=True); method = "voxel proxy (bone heat)"
    # No weight smoothing: it bleeds weight across the touching inner thighs (walk check: worst stretch x4 -> x23).
    ops(body); bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
    unweighted = sum(1 for v in body.data.vertices if not any(g.weight > 1e-4 for g in v.groups))
    owner = owners(body)
    counts = {b.name: int((owner == body.vertex_groups[b.name].index).sum()) if b.name in body.vertex_groups else 0 for b in rig.data.bones}
    say(f"   skin: {method}, unweighted vertices {unweighted}, vertices owned per bone {counts}")
    return {"method": method, "unweighted": unweighted, "owned": counts}


def owners(body):
    owner = np.full(len(body.data.vertices), -1)
    for i, v in enumerate(body.data.vertices):
        if v.groups: owner[i] = max(v.groups, key=lambda g: g.weight).group
    return owner


# ---------------------------------------------------------------- posing
def aim(rig, name, direction):
    """Turn a pose bone about its head so it points along direction (armature space); children follow."""
    pb = rig.pose.bones[name]
    turn = (pb.tail - pb.head).rotation_difference(Vector(direction).normalized()).to_matrix().to_4x4()
    pb.matrix = Matrix.Translation(pb.head) @ turn @ Matrix.Translation(-pb.head) @ pb.matrix
    bpy.context.view_layer.update()


def twist(rig, name, angle):
    pb = rig.pose.bones[name]; axis = (pb.tail - pb.head).normalized()
    pb.matrix = Matrix.Translation(pb.head) @ Matrix.Rotation(angle, 4, axis) @ Matrix.Translation(-pb.head) @ pb.matrix
    bpy.context.view_layer.update()


def deformed(body):
    dg = bpy.context.evaluated_depsgraph_get(); ev = body.evaluated_get(dg)
    return np.array([v.co[:] for v in ev.data.vertices])


def apply_rest(body, rig):
    """Freeze the current pose as the rest pose: mesh first (a copy of its armature modifier), then the armature."""
    mod = next(m for m in body.modifiers if m.type == "ARMATURE")
    ops(body); bpy.ops.object.modifier_copy(modifier=mod.name)
    copy = body.modifiers[-1]; bpy.ops.object.modifier_move_to_index(modifier=copy.name, index=0)
    bpy.ops.object.modifier_apply(modifier=copy.name)
    ops(rig, "POSE"); bpy.ops.pose.armature_apply(selected=False); bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()


def straighten_legs(rig):
    """Legs straight under the hips and feet pointing forward, the same on both sides (the physics body is symmetric)."""
    foot = np.mean([np.array(rig.data.bones[f"Foot.{s}"].tail_local - rig.data.bones[f"Foot.{s}"].head_local) for s in "LR"], axis=0)
    ops(rig, "POSE")
    for s in "LR":
        aim(rig, f"UpperLeg.{s}", (0, 0, -1)); aim(rig, f"LowerLeg.{s}", (0, 0, -1))
        aim(rig, f"Foot.{s}", (math.hypot(foot[0], foot[1]), 0, foot[2])); aim(rig, f"Toe.{s}", (1, 0, 0))
    bpy.ops.object.mode_set(mode="OBJECT")


def ground(body, rig):
    """Stand the rest pose's lowest vertex on z = 0 (straight legs reach lower than the bent stance did)."""
    dz = -float(verts(body)[:, 2].min()); T = Matrix.Translation((0, 0, dz))
    body.data.transform(T); body.data.update()
    ops(rig, "EDIT")
    for eb in rig.data.edit_bones: eb.head.z += dz; eb.tail.z += dz
    bpy.ops.object.mode_set(mode="OBJECT"); bpy.context.view_layer.update()
    return dz


def palm_turn(body, rig, side):
    """Twist that turns the hand's flat side to face down, within +-90 degrees: the palm's normal is the dominant axis
    of the hand's surface normals across the arm (palm and back point opposite ways along it; a curled hand's vertex
    spread is too round to tell). No twist when the hand is not clearly flat."""
    dg = bpy.context.evaluated_depsgraph_get(); ev = body.evaluated_get(dg).data
    owner = owners(body); idx = np.where(owner == body.vertex_groups[f"Hand.{side}"].index)[0]
    nrm = np.array([ev.vertices[i].normal[:] for i in idx])
    pb = rig.pose.bones[f"Hand.{side}"]; axis = np.array((pb.tail - pb.head).normalized())
    nrm -= np.outer(nrm @ axis, axis)
    w, V = np.linalg.eigh(nrm.T @ nrm); normal = V[:, 2]
    if w[2] < 1.5 * w[1]:
        say(f"   palm {side}: no clear flat side (normal spread {w[2]:.0f} vs {w[1]:.0f}), left as is"); return 0.0
    up = np.array([0, 0, 1.0]); up -= axis * (up @ axis); up /= np.linalg.norm(up)
    ang = math.atan2(np.cross(normal, up) @ axis, normal @ up)
    if ang > math.pi / 2: ang -= math.pi
    if ang < -math.pi / 2: ang += math.pi
    return ang


def t_pose(body, rig):
    straighten_legs(rig)
    ops(rig, "POSE")
    for s, sg in (("L", 1), ("R", -1)):
        for b in ("UpperArm", "LowerArm", "Hand"): aim(rig, f"{b}.{s}", (0, sg, 0))
    turns = {}
    for s in "LR":
        a = palm_turn(body, rig, s); turns[s] = round(math.degrees(a), 1)
        twist(rig, f"LowerArm.{s}", a * 0.5); twist(rig, f"Hand.{s}", a * 0.5)   # forearm and wrist share the pronation
    bpy.ops.object.mode_set(mode="OBJECT")
    apply_rest(body, rig); dz = ground(body, rig)
    say(f"   grounded: raised {dz * 1000:.1f} mm")
    say(f"3. T-pose: arms horizontal, legs straight, palms turned down by {turns} degrees (L/R), made the rest pose")
    return turns


# ---------------------------------------------------------------- walk cycle and its check
def set_world_rot(pb, axis, angle):
    """Pose rotation about a world axis, expressed in the bone's rest frame (its parent turns about the same axis)."""
    B = pb.bone.matrix_local.to_quaternion()
    pb.rotation_mode = "QUATERNION"; pb.rotation_quaternion = B.inverted() @ Quaternion(axis, angle) @ B


def walk_cycle(rig):
    ops(rig, "POSE")
    scene = bpy.context.scene; scene.render.fps = 30; scene.frame_start = 1; scene.frame_end = WALK_FRAMES + 1
    Y, Z = Vector((0, 1, 0)), Vector((0, 0, 1))
    for f in range(WALK_FRAMES + 1):
        ph = 2 * math.pi * f / WALK_FRAMES
        for s, sg in (("L", 1), ("R", -1)):
            p = ph if s == "L" else ph + math.pi
            hip = -0.38 * math.sin(p)                                 # negative about +Y swings the leg forward
            knee = 0.08 + 0.65 * max(0.0, math.cos(p)) ** 1.5         # bent while the leg swings through
            ankle = -0.6 * (hip + knee) - (0.12 if math.cos(p) > 0 else 0.0)
            set_world_rot(rig.pose.bones[f"UpperLeg.{s}"], Y, hip)
            set_world_rot(rig.pose.bones[f"LowerLeg.{s}"], Y, knee)
            set_world_rot(rig.pose.bones[f"Foot.{s}"], Y, ankle)
            set_world_rot(rig.pose.bones[f"UpperArm.{s}"], Z, sg * 0.30 * math.sin(p))     # arms swing against the legs
            set_world_rot(rig.pose.bones[f"LowerArm.{s}"], Z, sg * 0.12 * max(0.0, math.sin(p)))
            for b in ("UpperLeg", "LowerLeg", "Foot", "UpperArm", "LowerArm"):
                rig.pose.bones[f"{b}.{s}"].keyframe_insert("rotation_quaternion", frame=f + 1)
        hips = rig.pose.bones["Hips"]; set_world_rot(hips, Z, 0.08 * math.sin(ph))
        Bm = hips.bone.matrix_local.to_quaternion()
        hips.location = Bm.inverted() @ Vector((0, 0.015 * math.sin(ph + math.pi / 2), 0.02 * math.cos(2 * ph) - 0.02))
        hips.keyframe_insert("rotation_quaternion", frame=f + 1); hips.keyframe_insert("location", frame=f + 1)
        chest = rig.pose.bones["Chest"]; set_world_rot(chest, Z, -0.10 * math.sin(ph)); chest.keyframe_insert("rotation_quaternion", frame=f + 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    action = rig.animation_data.action; action.name = "Walk"
    return action


def check_walk(body, rig):
    """Edge stretch of the deformed mesh over the stride, against the rest pose."""
    edges = np.array([e.vertices[:] for e in body.data.edges]); rest = verts(body)
    L0 = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1); ok = L0 > 1e-5
    worst, p999, where = 0.0, 0.0, {}
    owner = owners(body); names = [g.name for g in body.vertex_groups]
    for f in range(1, WALK_FRAMES + 2):
        bpy.context.scene.frame_set(f); co = deformed(body)
        r = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)[ok] / L0[ok]
        s = np.maximum(r, 1 / np.maximum(r, 1e-6))
        worst = max(worst, float(s.max())); p999 = max(p999, float(np.percentile(s, 99.9)))
        for e in edges[ok][s > 1.5]:
            n = names[owner[e[0]]] if owner[e[0]] >= 0 else "-"; where[n] = where.get(n, 0) + 1
    bpy.context.scene.frame_set(1)
    where = dict(sorted(where.items(), key=lambda kv: -kv[1])[:6])
    say(f"   walk check: worst edge stretch x{worst:.2f}, 99.9th percentile x{p999:.3f} over {WALK_FRAMES} frames; "
        f"edges stretched over x1.5 (edge-frames by bone): {where} of {int(ok.sum())} edges")
    return {"worst_stretch": round(worst, 3), "p999_stretch": round(p999, 4), "over_1_5_by_bone": where, "edges": int(ok.sum())}


# ---------------------------------------------------------------- renders for the report
def render(name, frames=(1,), view="three_quarter"):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"; scene.display.shading.light = "STUDIO"; scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x, scene.render.resolution_y = 640, 800; scene.render.film_transparent = False
    scene.world = scene.world or bpy.data.worlds.new("World")
    cam = bpy.data.objects.get("ShotCam")
    if cam is None:
        cam = bpy.data.objects.new("ShotCam", bpy.data.cameras.new("ShotCam")); scene.collection.objects.link(cam)
    scene.camera = cam; cam.data.type = "ORTHO"; cam.data.ortho_scale = 2.3
    pos = {"front": (6, 0, 0.9), "three_quarter": (4.5, 3.6, 1.3), "side": (0, 6, 0.9)}[view]
    cam.location = pos; d = Vector((0, 0, 0.88)) - Vector(pos); cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    out = []
    for f in frames:
        scene.frame_set(f); path = os.path.join(SHOTS, f"{name}_{view}_f{f:02d}.png")
        scene.render.filepath = path; bpy.ops.render.render(write_still=True); out.append(os.path.basename(path))
    scene.frame_set(1)
    return out


# ---------------------------------------------------------------- 4. game rest pose and its measurements
def game_pose(body, rig):
    ops(rig, "POSE"); a = math.radians(ARM_OUT_DEG)
    for s, sg in (("L", 1), ("R", -1)):
        for b in ("UpperArm", "LowerArm", "Hand"): aim(rig, f"{b}.{s}", (0, sg * math.sin(a), -math.cos(a)))
    bpy.ops.object.mode_set(mode="OBJECT")
    apply_rest(body, rig)
    co = verts(body); names = [g.name for g in body.vertex_groups]; owner = owners(body)
    bones = {}
    for b in rig.data.bones:
        head, tail = np.array(b.head_local), np.array(b.tail_local); axis = (tail - head) / np.linalg.norm(tail - head)
        pts = co[owner == names.index(b.name)] if b.name in names else np.zeros((0, 3))
        e = {"parent": b.parent.name if b.parent else None, "head": head.round(4).tolist(), "tail": tail.round(4).tolist(), "verts": int(len(pts))}
        if len(pts) > 20:
            rel = pts - head; radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
            e.update(r50=round(float(np.percentile(radial, 50)), 4), r80=round(float(np.percentile(radial, 80)), 4),
                     lo=np.percentile(pts, 3, axis=0).round(4).tolist(), hi=np.percentile(pts, 97, axis=0).round(4).tolist())
        bones[b.name] = e
    say(f"4. game rest pose: arms {ARM_OUT_DEG} degrees out; bbox {co.min(0).round(3)} {co.max(0).round(3)}")
    for n, e in bones.items(): say(f"   {n:12s} head {e['head']}  r50 {e.get('r50')}  r80 {e.get('r80')}  verts {e['verts']}")
    return co, bones


# ---------------------------------------------------------------- material
def clean_material(body):
    ops(body); bpy.ops.object.material_slot_remove_unused()
    mat = body.data.materials[0]; mat.name = "Grandpa"
    nodes = mat.node_tree.nodes; bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    # KHR_materials_specular x2 made the skin and cloth glossy; the Principled defaults read as cloth and skin.
    if "Specular Tint" in bsdf.inputs: bsdf.inputs["Specular Tint"].default_value = (1, 1, 1, 1)
    if "Specular IOR Level" in bsdf.inputs: bsdf.inputs["Specular IOR Level"].default_value = 0.5
    maps = {s: bool(bsdf.inputs[s].links) for s in ("Base Color", "Normal", "Roughness", "Metallic")}
    say(f"   material: one Principled BSDF, mapped inputs {maps}")
    return bsdf, maps


def texture(bsdf, socket, path, px=2048):
    link = bsdf.inputs[socket].links[0].from_node if bsdf.inputs[socket].links else None
    while link is not None and link.type != "TEX_IMAGE":
        link = next((i.links[0].from_node for i in link.inputs if i.links), None)
    if link is None: return None
    img = link.image.copy(); img.scale(px, px)
    img.filepath_raw = path; img.file_format = "PNG"; img.save(); bpy.data.images.remove(img)
    return os.path.basename(path)


def main():
    for d in (SHOTS, OUT_UNITY, os.path.dirname(OUT_TPOSE), os.path.dirname(OUT_GLB), os.path.dirname(OUT_RIG), os.path.dirname(OUT_BLEND)):
        os.makedirs(d, exist_ok=True)
    report = {}
    body, scale, report["mesh"] = import_clean()
    bsdf, report["material"] = clean_material(body)
    report["source_pose"] = render("0_source", view="three_quarter") + render("0_source", view="front")
    rig = build_armature(body)
    report["skin"] = skin(body, rig)
    report["palm_turn_deg"] = t_pose(body, rig)
    report["tpose"] = render("1_tpose", view="front") + render("1_tpose", view="three_quarter")
    walk_cycle(rig)
    report["walk"] = check_walk(body, rig)
    report["walk_frames"] = render("2_walk", frames=(1, 9, 17, 25), view="side") + render("2_walk", frames=(5,), view="three_quarter")
    ops(rig); body.select_set(True)
    bpy.ops.export_scene.fbx(filepath=OUT_TPOSE, use_selection=True, object_types={"ARMATURE", "MESH"}, global_scale=1.0,
                             apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
                             add_leaf_bones=False, path_mode="COPY", embed_textures=True, bake_anim=True, bake_anim_use_all_actions=False,
                             bake_anim_use_nla_strips=False, mesh_smooth_type="FACE", use_armature_deform_only=True)
    bpy.ops.export_scene.gltf(filepath=OUT_GLB, use_selection=True, export_format="GLB", export_animations=True)
    say(f"   exported {OUT_TPOSE} ({os.path.getsize(OUT_TPOSE) // 1024} KB) and {OUT_GLB}")
    # the game asset has no animation: physics drives every bone
    rig.animation_data.action = None
    for pb in rig.pose.bones: pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    co, bones = game_pose(body, rig)
    report["game_pose"] = render("3_game_rest", view="front") + render("3_game_rest", view="side")
    json.dump({"height": HEIGHT_M, "scale_from_scan": round(scale, 5), "lo": co.min(0).round(4).tolist(), "hi": co.max(0).round(4).tolist(), "bones": bones},
              open(OUT_RIG, "w"), indent=1)
    saved = [texture(bsdf, "Base Color", os.path.join(OUT_UNITY, "grandpa_albedo.png")),
             texture(bsdf, "Normal", os.path.join(OUT_UNITY, "grandpa_normal.png"))]
    ops(rig); body.select_set(True)
    bpy.ops.export_scene.fbx(filepath=os.path.join(OUT_UNITY, "grandpa.fbx"), use_selection=True, object_types={"ARMATURE", "MESH"},
                             apply_scale_options="FBX_SCALE_ALL", add_leaf_bones=False, bake_anim=False, mesh_smooth_type="FACE",
                             use_tspace=True, use_armature_deform_only=True)
    say(f"   wrote {OUT_RIG}, {OUT_UNITY}/grandpa.fbx, textures {saved}")
    bpy.context.preferences.filepaths.save_version = 0          # no Grandpa.blend1 beside the archive
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND, compress=True)
    report["outputs"] = [OUT_TPOSE, OUT_GLB, OUT_RIG, os.path.join(OUT_UNITY, "grandpa.fbx"), OUT_BLEND]
    json.dump(report, open(os.path.join(SHOTS, "make_grandpa.json"), "w"), indent=1)
    say("5. archived", OUT_BLEND)


if __name__ == "__main__":
    os.makedirs(SHOTS, exist_ok=True)
    with open(LOG, "w", buffering=1) as log:
        sys.stdout = sys.stderr = log
        try:
            main(); say("DONE")
        except Exception:
            traceback.print_exc(); say("FAILED")
