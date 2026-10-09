"""Render-only "Mountain top" map for PoKingHill. Run inside Blender (Scripting tab, or exec() it):

    exec(open(r"<repo>/tools/make_mountain_map.py").read())

Builds three meshes and their textures and exports them to Assets/PoKingHill/Maps/Mountain:
  Mountain   the hill itself, radius 10.5 m. Physics is untouched: the collision shape stays the dome from
             koth/build_mjcf.py (flat disc r < 1.5 m at z = 0, then z = -(r - 1.5)^2 / 9). The visible surface
             equals that dome exactly for r < 1.9 m and departs from it by a rock relief that grows with radius.
  Rocks      loose boulders on the slopes, 3.2 m and further from the centre. Render only, like everything here:
             a robot that has already been thrown off can pass through one.
  Backdrop   distant peaks standing in the sea, 120-220 m out.
Rock colour, normals, roughness and occlusion come from two CC0 photo scans from polyhaven.com (aerial_rocks_02
for the whole hill, rock_face_03 for the close-up grain and the boulders). They are downloaded into
tools/polyhaven on first use; that folder is not in git.
Blender axes are the MuJoCo axes (z up, metres). The dome is symmetric about z, so the FBX axis conversion
(a half turn about the vertical compared with the project's MuJoCo-to-Unity mapping) changes nothing physical.
"""
import os
import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", bpy.data.filepath or "."))))
OUT = globals().get("OUT") or os.path.join(REPO, "Assets", "PoKingHill", "Maps", "Mountain")
PLATEAU_R, SLOPE_K = 1.5, 1.0 / 9.0          # must equal koth/build_mjcf.py
EXACT_R = 1.9                                # visible surface = collision dome inside this radius
R_MAX = 10.5                                 # mountain mesh and texture half-extent; the sea starts 6 m down (r = 8.85)
TEX = 2048
BACK_HALF, BACK_TEX = 300.0, 1024
DETAIL_TILE_M = 2.7                        # real size of the rock_face_03 scan; the Unity material repeats it
AERIAL_SPAN_M = 36.0                       # metres of hill covered by the whole aerial photo
PHOTOS = os.path.join(REPO, "tools", "polyhaven")
PHOTO_RES = {"aerial_rocks_02": "4k", "rock_face_03": "2k"}
LUMA = np.array([0.30, 0.59, 0.11], np.float32)
_photos = {}


def photo(asset, kind):
    """A Poly Haven map (kind: Diffuse, nor_gl, Rough, AO) as an (h, w, 3) array, rows bottom-up, values as stored."""
    if (asset, kind) in _photos: return _photos[asset, kind]
    path = os.path.join(PHOTOS, f"{asset}_{kind.lower()}.jpg")
    if not os.path.exists(path):
        import json, urllib.request
        get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "PoKingHill"}), timeout=180).read()
        url = json.loads(get(f"https://api.polyhaven.com/files/{asset}"))[kind][PHOTO_RES[asset]]["jpg"]["url"]
        os.makedirs(PHOTOS, exist_ok=True); open(path, "wb").write(get(url))
    img = bpy.data.images.load(path); w, h = img.size
    a = np.empty(w * h * 4, np.float32); img.pixels.foreach_get(a); bpy.data.images.remove(img)
    _photos[asset, kind] = a.reshape(h, w, 4)[..., :3]
    return _photos[asset, kind]


def sample(tex, u, v):
    """Nearest texel with wrap-around (the scans tile)."""
    h, w, _ = tex.shape
    return tex[np.floor(v * h).astype(np.int64) % h, np.floor(u * w).astype(np.int64) % w]
rng = np.random.default_rng(7)
_P = rng.permutation(256); _V = rng.random(256).astype(np.float32)


def vnoise(x, y, seed=0, period=256):
    """Value noise in [0, 1]; repeats every `period` lattice cells (a power of two up to 256)."""
    ix = np.floor(x).astype(np.int64); iy = np.floor(y).astype(np.int64)
    fx = (x - ix).astype(np.float32); fy = (y - iy).astype(np.float32)
    u = fx * fx * (3 - 2 * fx); v = fy * fy * (3 - 2 * fy)
    m = min(period, 256) - 1                    # longer periods are multiples of 256, so the wrap still tiles
    h = lambda a, b: _V[(_P[(_P[a & m] + (b & m)) & 255] + seed) & 255]
    top = h(ix, iy) * (1 - u) + h(ix + 1, iy) * u
    bot = h(ix, iy + 1) * (1 - u) + h(ix + 1, iy + 1) * u
    return top * (1 - v) + bot * v


def fbm(x, y, octaves, seed=0, ridged=False, period=None):
    """period: lattice cells per repeat at the first octave (tileable when set; no per-octave offset then)."""
    total = 0.0; amp = 0.5; norm = 0.0
    for o in range(octaves):
        if period: n = vnoise(x * 2 ** o, y * 2 ** o, seed + 31 * o, period * 2 ** o)
        else: n = vnoise(x * 2 ** o + 17.3 * o, y * 2 ** o - 9.1 * o, seed + 31 * o)
        if ridged: n = (1 - np.abs(2 * n - 1)) ** 2
        total = total + amp * n; norm += amp; amp *= 0.5
    return total / norm


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)


def dome(r):
    return np.where(r < PLATEAU_R, 0.0, -SLOPE_K * (r - PLATEAU_R) ** 2)


def relief(x, y):
    """Rock relief carried by the mesh (wavelengths of 0.4 m and longer). Zero inside EXACT_R."""
    r = np.hypot(x, y)
    amp = 0.10 * ss(EXACT_R, 3.2, r) + 0.5 * ss(3.0, 6.0, r) + 1.1 * ss(5.0, 9.0, r)
    # Mostly carved into the dome (gullies between buttresses) so the rock rarely stands above the collision
    # surface: a robot sliding down may hang over a gully, it should not sink into a ridge.
    ridges = fbm(x * 0.28, y * 0.28, 5, seed=3, ridged=True)
    carved = amp * (1.9 * ridges - 1.0) + 0.25 * amp * (fbm(x * 1.1, y * 1.1, 3, seed=11) - 0.5)
    return carved


def detail(x, y):
    """Fine relief that only the normal map and the colours carry. Subtle gravel on the plateau."""
    r = np.hypot(x, y)
    cracks = ss(0.035, 0.0, np.abs(fbm(x * 0.8, y * 0.8, 4, seed=6) - 0.5))                # thin fracture lines
    d = 0.03 * fbm(x * 2.2, y * 2.2, 6, seed=5) + 0.008 * fbm(x * 21, y * 21, 3, seed=8) - 0.02 * cracks + 0.02
    return d * (0.3 + 0.7 * ss(PLATEAU_R, 2.6, r))


def colours(x, y, z, slope, det, pu, pv):
    """sRGB albedo and the Unity mask. Photo-scanned rock with muted moss over the whole hill, darker steep faces
    and gullies, water streaks down the slopes, a few snow drifts below the rim, wet rock towards the waterline.
    pu, pv: where each texel samples the aerial photo."""
    r = np.hypot(x, y); n1 = fbm(x * 0.5, y * 0.5, 4, seed=21); n2 = fbm(x * 4, y * 4, 3, seed=22)
    dark = np.array([0.20, 0.19, 0.19])
    scan = sample(photo("aerial_rocks_02", "Diffuse"), pu, pv); lum = scan @ LUMA
    rock = 0.6 * scan + 0.4 * lum[..., None] * np.array([0.98, 1.0, 1.05])       # mute the moss, cool the stone
    ang = np.arctan2(y, x); streaks = ss(0.56, 0.7, fbm(ang * 30, r * 0.25, 3, seed=29)) * ss(2.2, 3.5, r)
    rock = rock * (1.0 - 0.22 * streaks)[..., None]                               # water runs downhill
    # Tilted, wobbling strata instead of level rings (level bands on a dome read as contour lines).
    strata = 0.5 + 0.5 * np.sin((z + 0.22 * x - 0.13 * y) * 9 + 9 * n1 + 3 * n2)
    rock = rock * (0.86 + 0.14 * strata * ss(0.6, 1.4, slope))[..., None]
    rock = rock * (0.86 + 0.16 * np.clip(det / 0.03, 0, 1) + 0.10 * n2)[..., None]            # crevices darker
    rock = rock * (1.0 - 0.25 * ss(1.2, 3.0, slope))[..., None]                               # steep faces darker
    rock = rock + (dark - rock) * ss(-4.8, -6.4, z)[..., None]                                # wet band
    drifts = ss(0.54, 0.62, fbm(x * 0.7, y * 0.7, 4, seed=27))                                  # coherent patches, not speckle
    snow_fit = ss(1.25, 0.7, slope) * ss(-5.6, -3.6, z + 1.5 * (n1 - 0.5)) * drifts * ss(-0.3, -1.0, z)
    snow_fit = snow_fit * (0.10 + 0.9 * ss(PLATEAU_R + 0.2, 3.2, r))        # mostly bare rock where the fight happens
    snow = np.array([0.90, 0.93, 0.97]) * (0.93 + 0.07 * n2)[..., None]
    snow_fit = np.clip(snow_fit, 0, 1); wet = ss(-4.8, -6.4, z)
    # Unity URP mask: R metallic (none), G ambient occlusion (crevices and steep faces), A smoothness
    # (dry rock dull, snow and wet rock glossier). One texture feeds both the metallic and the occlusion slot.
    gully = np.clip(-relief(x, y) / 0.9, 0, 1)                                    # deep in a gully = less sky
    ao = sample(photo("aerial_rocks_02", "AO"), pu, pv)[..., 0] * np.clip(0.75 + 0.25 * np.clip(det / 0.03, 0, 1) - 0.12 * ss(1.2, 3.0, slope) - 0.3 * gully, 0, 1)
    ao = ao + (1 - ao) * snow_fit
    smooth = 0.5 * (1 - sample(photo("aerial_rocks_02", "Rough"), pu, pv)[..., 0])
    smooth = smooth * (1 - snow_fit) + 0.42 * snow_fit + (0.45 * wet + 0.15 * streaks) * (1 - snow_fit)
    mask = np.dstack([np.zeros_like(ao), ao, np.zeros_like(ao), smooth])
    return np.clip(rock + (snow - rock) * snow_fit[..., None], 0, 1), mask


def save_png(name, rgb, colorspace):
    h, w, _ = rgb.shape
    img = bpy.data.images.get(name) or bpy.data.images.new(name, w, h, alpha=rgb.shape[2] == 4)
    img.colorspace_settings.name = colorspace; img.alpha_mode = "CHANNEL_PACKED"
    px = np.ones((h, w, 4), np.float32); px[..., :rgb.shape[2]] = rgb
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = os.path.join(OUT, name + ".png"); img.file_format = "PNG"; img.save()
    return img


def normal_map(height, metres_per_px):
    gy, gx = np.gradient(height, metres_per_px)
    n = np.dstack([-gx, -gy, np.ones_like(height)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    return n * 0.5 + 0.5


def material(name, albedo, normal):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True; nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.85
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    ta = nt.nodes.new("ShaderNodeTexImage"); ta.image = albedo
    nt.links.new(ta.outputs["Color"], bsdf.inputs["Base Color"])
    if normal is not None:
        tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = normal
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_object(name, verts, faces, uvs, mat):
    old = bpy.data.objects.get(name)
    if old: bpy.data.objects.remove(old, do_unlink=True)
    mesh = bpy.data.meshes.new(name); mesh.from_pydata(verts.tolist(), [], faces)
    uv = mesh.uv_layers.new(name="UVMap")
    loop_verts = np.empty(len(mesh.loops), np.int32); mesh.loops.foreach_get("vertex_index", loop_verts)
    uv.data.foreach_set("uv", uvs[loop_verts].ravel())
    mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), bool)); mesh.materials.append(mat); mesh.update()
    if sum(p.normal.z for p in mesh.polygons) < 0: mesh.flip_normals()          # terrain faces look up
    assert max(len(p.vertices) for p in mesh.polygons) <= 4, "ngon in " + name
    coll = bpy.data.collections.get("PoKingHill_Map")
    if coll is None: coll = bpy.data.collections.new("PoKingHill_Map"); bpy.context.scene.collection.children.link(coll)
    obj = bpy.data.objects.new(name, mesh); coll.objects.link(obj)
    return obj


def polar_grid(radii, n_ang):
    """Centre vertex + rings; returns x, y arrays and quad/triangle faces."""
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    x = np.concatenate([[0.0], (radii[:, None] * np.cos(ang)).ravel()])
    y = np.concatenate([[0.0], (radii[:, None] * np.sin(ang)).ravel()])
    faces = [(0, 1 + k, 1 + (k + 1) % n_ang) for k in range(n_ang)]
    for i in range(len(radii) - 1):
        a = 1 + i * n_ang
        for k in range(n_ang):
            k2 = (k + 1) % n_ang
            faces.append((a + k, a + n_ang + k, a + n_ang + k2, a + k2))
    return x, y, faces


def build_mountain():
    # Mesh: rings are dense where the camera looks (plateau rim to about 6 m).
    radii = np.unique(np.concatenate([np.linspace(0.15, PLATEAU_R, 10), np.linspace(PLATEAU_R, 6.0, 110), np.linspace(6.0, R_MAX, 60)]))
    x, y, faces = polar_grid(radii, 288)
    z = dome(np.hypot(x, y)) + relief(x, y)
    # UVs: the dome unrolled along its own profile (distance walked down the slope, not horizontal radius), so
    # texels are not smeared down the steep lower faces as a plain top-down projection does.
    rr = np.linspace(0, R_MAX, 4000); arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(rr), np.diff(dome(rr))))]); s_max = arc[-1]
    r = np.hypot(x, y); k = np.interp(r, rr, arc) / np.maximum(r, 1e-9) / (2 * s_max)
    uvs = np.stack([x * k + 0.5, y * k + 0.5], axis=1).astype(np.float32)
    c = (np.arange(TEX) + 0.5) / TEX * 2 * s_max - s_max; px, py = np.meshgrid(c, c); ps = np.hypot(px, py)
    back = np.interp(ps, arc, rr) / np.maximum(ps, 1e-9); gx, gy = px * back, py * back         # texel -> world x, y
    surf = lambda u, v: dome(np.hypot(u, v)) + relief(u, v)
    det = detail(gx, gy); base = surf(gx, gy); full = base + det; mpp = 2 * s_max / TEX; e = 0.03
    slope = np.hypot(surf(gx + e, gy) - base, surf(gx, gy + e) - base) / e
    print("detail tiling for the Unity material:", round(2 * s_max / DETAIL_TILE_M, 2))
    pu, pv = px / AERIAL_SPAN_M + 0.5, py / AERIAL_SPAN_M + 0.5
    rgb, mask = colours(gx, gy, full, slope, det, pu, pv)
    albedo = save_png("mountain_albedo", rgb, "sRGB"); save_png("mountain_mask", mask, "Non-Color")
    # Normal map: the scan's own surface detail plus the generated cracks and grain.
    a = normal_map(det, mpp) * 2 - 1; b = sample(photo("aerial_rocks_02", "nor_gl"), pu, pv) * 2 - 1
    n = np.dstack([a[..., 0] + b[..., 0], a[..., 1] + b[..., 1], a[..., 2] * b[..., 2]]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    normal = save_png("mountain_normal", n * 0.5 + 0.5, "Non-Color")
    obj = make_object("Mountain", np.stack([x, y, z], axis=1), faces, uvs, material("MountainRock", albedo, normal))
    # The promise the game relies on: inside EXACT_R the visible surface is the collision dome.
    inner = np.hypot(x, y) < EXACT_R
    assert np.abs(z[inner] - dome(np.hypot(x, y))[inner]).max() < 1e-6, "visible summit differs from the collision dome"
    return obj


def build_backdrop():
    n = 180; c = np.linspace(-BACK_HALF, BACK_HALF, n); gx, gy = np.meshgrid(c, c)

    def height(px, py):
        h = np.full(px.shape, -14.0)
        prng = np.random.default_rng(11)
        for k in range(9):
            a = 2 * np.pi * (k + prng.uniform(-0.3, 0.3)) / 9; d = prng.uniform(120, 220)
            cx, cy = d * np.cos(a), d * np.sin(a); peak = prng.uniform(35, 90); width = peak * prng.uniform(1.0, 1.5)
            fall = np.clip(1 - np.hypot(px - cx, py - cy) / width, 0, 1) ** 1.25
            h = np.maximum(h, -14 + (peak + 14) * fall * (0.35 + 1.3 * fbm(px * 0.022 + k, py * 0.022, 6, seed=40 + k, ridged=True)))
        return np.where(np.hypot(px, py) < 60, -14.0, h)       # nothing near the arena

    z = height(gx, gy)
    verts = np.stack([gx.ravel(), gy.ravel(), z.ravel()], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    faces = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4).tolist()
    uvs = np.stack([gx.ravel() / (2 * BACK_HALF) + 0.5, gy.ravel() / (2 * BACK_HALF) + 0.5], axis=1).astype(np.float32)
    t = (np.arange(BACK_TEX) + 0.5) / BACK_TEX * 2 * BACK_HALF - BACK_HALF; tx, ty = np.meshgrid(t, t)
    tz = height(tx, ty); mpp = 2 * BACK_HALF / BACK_TEX; sy, sx = np.gradient(tz, mpp); slope = np.hypot(sx, sy)
    n1 = fbm(tx * 0.05, ty * 0.05, 4, seed=60)
    rock = np.array([0.24, 0.24, 0.27]) * (0.75 + 0.5 * n1)[..., None] * (1.0 - 0.3 * ss(0.8, 2.0, slope))[..., None]
    snow = ss(4.0, 14.0, tz + 12 * (n1 - 0.5)) * ss(2.2, 1.1, slope)
    rgb = rock + (np.array([0.92, 0.94, 0.98]) - rock) * snow[..., None]
    albedo = save_png("backdrop_albedo", np.clip(rgb, 0, 1), "sRGB")
    return make_object("Backdrop", verts, faces, uvs, material("BackdropRock", albedo, None))


def build_detail():
    """Close-up grain for the hill's detail slot (grey around 0.5, multiplied x2 in Unity) and the boulder
    material, all from the rock_face_03 scan (it tiles)."""
    scan = photo("rock_face_03", "Diffuse")[::2, ::2]; lum = scan @ LUMA
    grey = np.clip(0.5 + 0.9 * (lum - lum.mean()), 0.1, 0.9)
    save_png("rock_detail_albedo", np.dstack([grey] * 3), "Non-Color")
    normal = save_png("rock_detail_normal", photo("rock_face_03", "nor_gl")[::2, ::2], "Non-Color")
    stone = 0.45 * scan + 0.55 * lum[..., None] * np.array([0.98, 1.0, 1.05])     # grey it towards the hill's stone
    return material("BoulderRock", save_png("boulder_albedo", np.clip(stone * 1.05, 0, 1), "sRGB"), normal)


def build_rocks(mat):
    """Loose boulders resting on the visible slope, never inside 3.2 m (the fight happens inside 1.7 m)."""
    import bmesh
    prng = np.random.default_rng(23); bm = bmesh.new(); uv = bm.loops.layers.uv.new("UVMap")
    for i in range(70):
        ang, r = prng.uniform(0, 2 * np.pi), prng.uniform(3.2, 9.0); size = 0.12 + 0.5 * prng.random() ** 2.5
        cx, cy = r * np.cos(ang), r * np.sin(ang)
        cz = float(dome(np.array([r]))[0] + relief(np.array([cx]), np.array([cy]))[0]) - 0.25 * size      # a quarter buried
        verts = bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)["verts"]
        unit = np.array([v.co[:] for v in verts])
        lump = 0.65 + 0.7 * fbm(unit[:, 0] * 1.4 + 7.1 * i, unit[:, 1] * 1.4 + unit[:, 2] * 1.9, 3, seed=80)
        shape = unit * lump[:, None] * size * np.array([1.0, prng.uniform(0.6, 1.0), prng.uniform(0.5, 0.8)])
        turn = prng.uniform(0, 2 * np.pi); c, s_ = np.cos(turn), np.sin(turn)
        world = np.stack([cx + c * shape[:, 0] - s_ * shape[:, 1], cy + s_ * shape[:, 0] + c * shape[:, 1], cz + shape[:, 2]], axis=1)
        local = {v: unit[k] * size / DETAIL_TILE_M * 2 for k, v in enumerate(verts)}
        for f in {f for v in verts for f in v.link_faces}:   # box mapping: project each face along its dominant axis
            axis = int(np.argmax(np.abs(np.array(f.normal[:])))); keep = [k for k in range(3) if k != axis]
            for loop in f.loops: loop[uv].uv = (local[loop.vert][keep[0]], local[loop.vert][keep[1]])
            f.smooth = True
        for k, v in enumerate(verts): v.co = world[k]
    old = bpy.data.objects.get("Rocks")
    if old: bpy.data.objects.remove(old, do_unlink=True)
    mesh = bpy.data.meshes.new("Rocks"); bm.to_mesh(mesh); bm.free(); mesh.materials.append(mat)
    obj = bpy.data.objects.new("Rocks", mesh); bpy.data.collections["PoKingHill_Map"].objects.link(obj)
    return obj


def export(objs):
    for o in bpy.context.selected_objects: o.select_set(False)
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    path = os.path.join(OUT, "mountain.fbx")
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"MESH"}, apply_scale_options="FBX_SCALE_ALL",
                             bake_space_transform=True, axis_forward="-Z", axis_up="Y", bake_anim=False, add_leaf_bones=False,
                             mesh_smooth_type="FACE", use_tspace=True)
    return path


os.makedirs(OUT, exist_ok=True)
bpy.context.scene.unit_settings.system = "METRIC"; bpy.context.scene.unit_settings.scale_length = 1.0
mountain = build_mountain(); backdrop = build_backdrop(); rocks = build_rocks(build_detail())
print("exported", export([mountain, backdrop, rocks]), "| mountain verts", len(mountain.data.vertices), "| backdrop verts", len(backdrop.data.vertices))
bpy.data.orphans_purge(do_recursive=True)
