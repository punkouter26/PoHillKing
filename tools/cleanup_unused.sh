#!/usr/bin/env bash
# Removes tracked assets nothing references (found 2026-10-17 by a GUID scan of every .unity/.prefab/.mat/.asset
# outside Assets/Local/MjImports and Assets/_Recovery). Everything here is tracked, so `git checkout HEAD -- <path>`
# brings it back. Dry run by default; pass --apply to delete.   bash tools/cleanup_unused.sh [--apply]
cd "$(dirname "$0")/.." || exit 1

# MuJoCo import folders the current scenes still use (the importer appends a random number to each import).
KEEP="scene_flat_1p_unity660 scene_flat_1p_unity804 scene_koth_1p_unity323 scene_koth_1p_unity747
      scene_koth_2p_unity306 scene_koth_2p_unity370 scene_koth_2p_unity941 scene_koth_4p_unity966 scene_koth_5p_unity964"

paths=()
for d in Assets/Local/MjImports/*/; do
  b=$(basename "$d")
  [[ " $(echo $KEEP) " == *" $b "* ]] && continue
  paths+=("Assets/Local/MjImports/$b" "Assets/Local/MjImports/$b.meta")
done
# Unity template leftovers and a stale crash-recovery scene.
paths+=(Assets/_Recovery Assets/_Recovery.meta Assets/TutorialInfo Assets/TutorialInfo.meta
        Assets/Readme.asset Assets/Readme.asset.meta Assets/Scenes Assets/Scenes.meta)

# keep only paths git tracks (an import folder can exist on disk without ever having been committed)
tracked=(); for p in "${paths[@]}"; do [ -n "$(git ls-files -- "$p" | head -1)" ] && tracked+=("$p"); done; paths=("${tracked[@]}")
files=$(git ls-files -- "${paths[@]}" | wc -l)
mb=$(du -smc "${paths[@]}" 2>/dev/null | tail -1 | cut -f1)
echo "$files tracked files, about ${mb} MB:"
printf '  %s\n' "${paths[@]}" | grep -v '\.meta$'
if [ "$1" != "--apply" ]; then echo "dry run: pass --apply to delete"; exit 0; fi
git rm -r -q -- "${paths[@]}" && echo "removed; commit when the editor has reimported cleanly"
