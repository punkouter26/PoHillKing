# Kim source art

`Kim.glb` is the original scan (rigged, T-pose, 4k textures). Edit this file in Blender, export it back here as
GLB under the same name, then regenerate everything derived from it:

1. In Blender: `exec(open(r"<repo>/tools/make_kim.py").read())`
   writes `Assets/PoKingHill/Fighters/Kim/` (mesh + textures) and `training/assets/kim/kim_rig.json`.
2. `cd training && KOTH_ROBOT=kim .venv/Scripts/python.exe -m koth.build_mjcf`
   rebuilds her physics body and scenes from the new measurements.
3. In Unity: `ParityBatch.ImportAll` with `-kothRobot kim` re-imports her testbed scenes.

Keep the bone names and the T-pose: the scripts find bones by name (`Hips`, `UpperArm.L`, ...). Changing her
proportions or bone positions changes her physics body, so any trained brain has to be retrained; changing only
textures or surface detail needs step 1 alone.
