#!/usr/bin/env bash
# Calibration / action / reward / curriculum options (2026-10-20), one 2048-world run at a time.
# Every G1 run repeats duel_u1's recipe (400 iterations from careful1, the duel pool, critic warm-up 20, self_edge -5,
# termination -25, Unity's leg rule) and changes one thing, so duel_u1 is the control:
#   scale1   per-joint action scale from careful1's raw outputs (scripts/action_scale.py)             option 4
#   mask1    tanh action map + torque-feasible window                                                 option 5
#   pbrs1    rung r4pbrs: ring shaping replaced by one potential, outside the zero clip              option 8
#   tactic1  rung r4tactic: goal point follows random tactics (Tactics.Goal)                         option 6
# Then the humans against the G1 league with the success-gated opponent ladder (koth/curriculum.py)  options 9 + 10.
# Results: runs/overnight/<name>_eval.txt, progress in runs/overnight/progress_opts.txt.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight; L=runs/league; mkdir -p $LOG
note() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG/progress_opts.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
DPOOL="$L/walker.pt:stand $L/walker.pt:stand $L/gen7.pt $L/careful1.pt $L/gen1.pt"
LEAGUE="$L/careful1.pt $L/gen7.pt $L/gen1.pt"
suite() {   # suite <checkpoint> <label>: the duel bars every G1 candidate is judged on (3 seeds x 256 duels each)
  { echo "# $2 vs standing walker"; ev --rung r4att --ckpt $1 --opponent $L/walker.pt --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity
    echo "# $2 vs league (careful1, gen7, gen1)"; ev --rung r4league --ckpt $1 --opponent $LEAGUE --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
    echo "# $2 vs careful1"; ev --rung r4league --ckpt $1 --opponent $L/careful1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
    echo "# $2 vs duel_u1 (control)"; ev --rung r4league --ckpt $1 --opponent runs/r4league/duel_u1/model_final.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity; } > $LOG/$2_eval.txt
}
g1run() {   # g1run <name> <rung> [extra train args...]
  local name=$1 rung=$2; shift 2
  note "$name start ($rung, 400 iterations, 2048 worlds) $*"
  if $PY scripts/train.py --rung $rung --name $name --num-envs 2048 --iters 400 --warm $L/careful1.pt --opponent $DPOOL --critic-warmup 20 \
       --reward self_edge=-5.0 termination=-25.0 --leg-contact-rule unity "$@" > $LOG/$name.log 2>&1; then
    suite runs/$rung/$name/model_final.pt $name; note "$name done: $(grep -c '^{' $LOG/${name}_eval.txt) eval rows"
  else note "$name FAILED (see $LOG/$name.log)"; fi
}
if [ ! -f $LOG/careful1_eval.txt ]; then note "baselines: careful1, duel_u1"; suite $L/careful1.pt careful1; suite runs/r4league/duel_u1/model_final.pt duel_u1; fi
[ -f runs/r4league/scale1/model_final.pt ] || g1run scale1 r4league --action-scale runs/audits/scale_careful1.json
[ -f runs/r4league/mask1/model_final.pt ]  || g1run mask1 r4league --action-map tanh --torque-window
[ -f runs/r4pbrs/pbrs1/model_final.pt ]    || g1run pbrs1 r4pbrs
[ -f runs/r4tactic/tactic1/model_final.pt ] || g1run tactic1 r4tactic
# humans vs the G1 league: opponents weakest first (standing walker, gen1, gen7, careful1); the ladder starts on the
# walker and adds the next one when the win share over 20 iterations reaches 0.5, stepping back below 0.3.
for who in trump kim; do
  if [ $who = trump ]; then BASE=runs/trump_r4bull/trump_fight1/model_final.pt; else BASE=runs/r4league/kim_fight1/model_final.pt; fi
  CK=runs/${who}_vs_g1/${who}_xg4/model_final.pt
  [ -f $CK ] && continue
  note "${who}_xg4 start (${who}_vs_g1, ladder curriculum, 400 iterations)"
  if KOTH_ROBOT=$who $PY scripts/train.py --rung ${who}_vs_g1 --name ${who}_xg4 --num-envs 2048 --iters 400 --warm $BASE \
       --opponent $L/walker.pt:stand $L/gen1.pt $L/gen7.pt $L/careful1.pt --frozen-deterministic --critic-warmup 10 --reward draw=-20 \
       --curriculum ladder > $LOG/${who}_xg4.log 2>&1; then
    { echo "# ${who}_xg4 vs G1 league"; KOTH_ROBOT=$who ev --rung ${who}_vs_g1 --ckpt $CK --opponent $L/careful1.pt $L/careful1.pt $L/gen7.pt $L/gen1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
      echo "# ${who} fight1 vs G1 league (baseline)"; KOTH_ROBOT=$who ev --rung ${who}_vs_g1 --ckpt $BASE --opponent $L/careful1.pt $L/careful1.pt $L/gen7.pt $L/gen1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
      echo "# ${who}_xg4 vs standing G1"; KOTH_ROBOT=$who ev --rung ${who}_vs_g1 --ckpt $CK --opponent $L/walker.pt:stand --seeds 2 --num-envs 256 --seconds 15 --leg-contact-rule unity; } > $LOG/${who}_xg4_eval.txt
    note "${who}_xg4 done"
  else note "${who}_xg4 FAILED"; fi
done
note "options queue finished"
