#!/usr/bin/env bash
# Body calibration fixes (2026-10-20, option 2), run after queue_options.sh so no run trains across a model change.
#   Kim    foot contact geoms moved 3.75 mm outward (build_human.py FOOT_GAP): her rest pose had them 3.6 mm into each other.
#   Trump  mirror-symmetric hold ctrl (build_mjcf.symmetric_hold): his stance settled in a lateral lean.
# Rebuilds every scene with a human in it, then re-checks the shipped human brains on the rebuilt bodies (no retrain
# unless they fall below their bars: the change is 4 mm of foot placement / the reset ctrl).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight; L=runs/league
note() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG/progress_opts.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
check() {    # check <suffix>: the shipped human brains on the current bodies -> runs/overnight/<body>_body_<suffix>_eval.txt
  for who in kim trump; do
    W=runs/${who}_r1b/${who}_r1b/model_final.pt
    if [ $who = trump ]; then F=runs/trump_r4bull/trump_fight1/model_final.pt; R=trump_r4bull; else F=runs/r4league/kim_fight1/model_final.pt; R=r4league; fi
    { echo "# ${who} walker on ${who}_r1b's bars"; KOTH_ROBOT=$who ev --rung ${who}_r1b --ckpt $W --seeds 3 --num-envs 256
      echo "# ${who} fight1 vs a standing ${who}"; KOTH_ROBOT=$who ev --rung r4att --ckpt $F --opponent $W --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity
      echo "# ${who} fight1 vs itself"; KOTH_ROBOT=$who ev --rung $R --ckpt $F --opponent $F --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity; } > $LOG/${who}_body_$1_eval.txt
  done
}
note "bodies: the human brains before the rebuild"; check before
note "bodies: rebuild Kim and Trump"
KOTH_ROBOT=kim $PY -m koth.build_mjcf > $LOG/build_kim.txt 2>&1 || { note "Kim build FAILED"; exit 1; }
KOTH_ROBOT=trump $PY -m koth.build_mjcf > $LOG/build_trump.txt 2>&1 || { note "Trump build FAILED"; exit 1; }
for tag in 5p 6p; do $PY -m koth.build_mjcf $tag >> $LOG/build_mixed.txt 2>&1 || note "koth_$tag build FAILED"; done
for mix in "kim g1" "trump g1" "kim g1 g1 g1" "trump g1 g1 g1"; do $PY -m koth.build_mjcf mix $mix >> $LOG/build_mixed.txt 2>&1 || note "mix $mix FAILED"; done
$PY scripts/collision_audit.py > $LOG/collision_audit.txt 2>&1 || note "collision audit FAILED"
note "bodies: rebuilt; re-checking the human brains"
check after
note "bodies: done"
