# Trump source art

`Trump.glb` is the original rigged character (34 bones named `bone_0` ... `bone_33`, arms hanging, 4k PBR textures).
Edit it, export it back here under the same name, then regenerate everything derived from it (no Blender needed):

1. `training/.venv/Scripts/python.exe tools/make_glb_fighter.py trump`
   writes `Assets/PoKingHill/Fighters/Trump/trump.glb` (2k textures) and `training/assets/trump/trump_rig.json`.
2. `cd training && KOTH_ROBOT=trump .venv/Scripts/python.exe -m koth.build_mjcf`, then `python -m koth.build_mjcf 6p`
   rebuilds his physics body (profile: `koth/build_trump.py`) and scenes.
3. In Unity: `ParityBatch.Import("flat_1p"|"koth_1p"|"koth_2p", "trump")` and `Import("koth_6p")`, then `GameScene.RefreshBodies` (game scene) and `GameScene.RefreshBrains`.

The bone-to-body table lives in `tools/make_glb_fighter.py` (FIGHTERS["trump"]). Changing his proportions or bone
positions changes his physics body, so his brains have to be retrained; textures alone need step 1 only.
