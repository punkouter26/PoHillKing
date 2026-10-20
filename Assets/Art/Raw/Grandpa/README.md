# Grandpa source art

`Grandpa.glb` is the original character: one textured mesh (4k PBR maps), no armature, standing with his right
arm out to the side and his left arm forward. Edit it, export it back here under the same name, then regenerate
everything derived from it:

1. `"$LOCALAPPDATA/Microsoft/WindowsApps/blender-launcher.exe" -b --factory-startup --python tools/make_grandpa.py`
   (log and renders: `SCREENSHOTS/grandpa-261020/`) cleans the mesh, rigs it (bone heat), and writes
   - `Assets/PoKingHill/Fighters/Grandpa/grandpa.fbx` + 2k textures: the game asset, arms hanging 20 degrees out;
   - `training/assets/grandpa/grandpa_rig.json`: joints and limb thickness for the physics body;
   - `Assets/Art/Exports/FBX/Grandpa/Character_TPose.fbx` (textures embedded) and `Assets/Art/Exports/GLB/Grandpa.glb`:
     the rig in a T-pose with a walk cycle, for use outside the game;
   - `Assets/Art/Raw/Grandpa~/Grandpa.blend`: the rigged scene.
2. `cd training && KOTH_ROBOT=grandpa .venv/Scripts/python.exe -m koth.build_mjcf`, then `python -m koth.build_mjcf 7p`
   rebuilds his physics body (profile: `koth/build_grandpa.py`) and scenes.
3. In Unity: `ParityBatch.Import("flat_1p"|"koth_1p"|"koth_2p", "grandpa")` and `Import("koth_7p")`, then `BuildDuelDemo`.

Joint positions come from adult proportions and the mesh (`landmarks` in `tools/make_grandpa.py`). Changing his
proportions changes his physics body, so his brains have to be retrained; textures alone need step 1 only.
