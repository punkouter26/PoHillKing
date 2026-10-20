#!/usr/bin/env bash
# Deletes what the 2026-10-20 cleanup set aside, after a dry run. Everything tracked comes back with
# `git checkout HEAD -- <path>` (before the commit that removes it) or from history; untracked items do not.
#   bash tools/cleanup_unused.sh                    # dry run: list everything, with file counts and MB
#   bash tools/cleanup_unused.sh --apply            # delete the default set (and prune training runs)
#   add --old-gates   superseded Kim trajectory-gate brains (kimr1b, kimr1c, kimr3b, kimfight + reference trajectories)
#   add --builds      the git-ignored player build in Builds/ (BuildPlayer makes a new one)
# MuJoCo import folders are never listed by hand: a folder under Assets/Local/MjImports is unused only when no scene,
# prefab, material or asset outside it references one of the GUIDs that exist only in that folder.
cd "$(dirname "$0")/.." || exit 1
apply=0; old_gates=0; builds=0
for a in "$@"; do case $a in --apply) apply=1;; --old-gates) old_gates=1;; --builds) builds=1;; *) echo "unknown option $a"; exit 2;; esac; done

paths=(
  archive/cut-261020                                   # scripts and duplicate materials the cleanup moved out of Assets
  Assets/_Recovery Assets/_Recovery.meta               # Unity crash-recovery scene (untracked)
  training/scripts/check_standing.py training/scripts/diag_tracking.py training/scripts/smoke_warp.py   # phase 0/A one-offs, unreferenced
)
if [ $old_gates = 1 ]; then
  for n in kimr1b kimr1c kimr3b kimfight; do paths+=($(ls Assets/PoKingHill/Models/kim/${n}_* 2>/dev/null)); done
fi
[ $builds = 1 ] && paths+=(Builds)

# MuJoCo imports nothing references any more (computed, see above)
refs=$(find Assets -type f \( -name '*.unity' -o -name '*.prefab' -o -name '*.mat' -o -name '*.asset' \) -not -path 'Assets/Local/MjImports/*' -not -path 'Assets/_Recovery/*')
used=$(cat $refs 2>/dev/null | grep -oE 'guid: [0-9a-f]{32}' | cut -c7- | sort -u)
owned=$(for d in Assets/Local/MjImports/*/; do b=$(basename "$d"); find "$d" -name '*.meta' -exec grep -h '^guid:' {} + | cut -c7- | sed "s/\$/ $b/"; done)
unique=$(echo "$owned" | cut -d' ' -f1 | sort | uniq -u)
for d in Assets/Local/MjImports/*/; do
  b=$(basename "$d")
  mine=$(echo "$owned" | grep " $b\$" | cut -d' ' -f1 | grep -xFf <(echo "$unique"))
  [ -n "$(echo "$mine" | grep -xFf <(echo "$used"))" ] && continue
  paths+=("Assets/Local/MjImports/$b" "Assets/Local/MjImports/$b.meta")
done

existing=(); for p in "${paths[@]}"; do [ -e "$p" ] && existing+=("$p"); done
files=$( [ ${#existing[@]} -gt 0 ] && find "${existing[@]}" -type f | wc -l || echo 0)
mb=$( [ ${#existing[@]} -gt 0 ] && du -smc "${existing[@]}" | tail -1 | cut -f1 || echo 0)
echo "$files files, about ${mb} MB:"; printf '  %s\n' "${existing[@]}" | grep -v '\.meta$'

echo; echo "training runs (training/scripts/prune_runs.py):"
py=training/.venv/Scripts/python.exe; [ -x "$py" ] || py=python
(cd training && "../$py" scripts/prune_runs.py $([ $apply = 1 ] && echo --apply) | tail -3)

if [ $apply = 0 ]; then echo; echo "dry run: pass --apply to delete"; exit 0; fi
tracked=$(git ls-files -- "${existing[@]}"); [ -n "$tracked" ] && git rm -r -q --cached -- "${existing[@]}" 2>/dev/null
rm -rf -- "${existing[@]}" && echo "removed $files files (${mb} MB); commit once the editor has reimported cleanly"
