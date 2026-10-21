# Mountain map source

`Mountain.blend` is the scene `tools/make_mountain_map.py` builds: the hill in three levels of detail
(`Mountain_LOD0-2`), loose rocks, crags, the ruined colonnade, the banners and the distant peaks, with the
textures it writes to `Assets/PoKingHill/Maps/Mountain` referenced (not packed). The folder name ends in `~`
so Unity does not try to import the `.blend` (that needs a Blender it can find, and the game only uses the FBX).
`Assets/Art/Exports/GLB/Mountain.glb` is the same scene as a self-contained GLB for reference.

The script is the source of truth; this file is its archived output. To change the map:

1. Edit `tools/make_mountain_map.py` (shapes, placement, textures), or edit `Mountain.blend` by hand and port the
   change back into the script so the next run does not lose it.
2. Run it headless: `blender -b --factory-startup --python tools/make_mountain_map.py` (with the Microsoft Store
   Blender: `%LOCALAPPDATA%/Microsoft/WindowsApps/blender-launcher.exe`). It rewrites `mountain.fbx` and the
   textures, re-bakes the ambient occlusion, and refreshes this archive and the GLB.
3. In Unity: the map is placed in `Scenes/Game.unity` by hand (the FBX import updates it in place).

Everything here is render only. The collision shape stays the dome from `koth/build_mjcf.py`; inside 1.9 m the
visible surface equals it in every level of detail, and no prop stands closer than 3.2 m to the centre
(banners 3.4 m). Props stay out of the broadcast camera's side of the summit (`CAMERA_SIDE` in the script).
