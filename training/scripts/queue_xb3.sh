#!/usr/bin/env bash
# Cross-body recipe v3 (2026-10-19). xg2 (deterministic G1 league, critic warm-up) stalled: Trump 6 % wins / 65 %
# undecided, Kim 19 % wins (both from 32 %): with draw = loss (-10) a long stall against a stronger opponent paid.
# v3, as the humans' own fighters were trained: first push a standing G1 off (<body>_att_g1), then the G1 league
# (<body>_vs_g1, the standing walker kept in the pool); draws cost twice a loss; deterministic opponents; critic warm-up.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight; L=runs/league
note() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG/progress.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
G1DUEL="$L/careful1.pt $L/careful1.pt $L/gen7.pt $L/gen1.pt"
for who in trump kim; do
  if [ $who = trump ]; then BASE=runs/trump_r4bull/trump_fight1/model_final.pt; else BASE=runs/r4league/kim_fight1/model_final.pt; fi
  note "V ${who}_att3 start (250 iterations from fight1 vs a standing G1, draw -20)"
  KOTH_ROBOT=$who $PY scripts/train.py --rung ${who}_att_g1 --name ${who}_att3 --num-envs 2048 --iters 250 --warm $BASE --opponent $L/walker.pt:stand \
      --frozen-deterministic --critic-warmup 20 --reward draw=-20 > $LOG/${who}_att3.log 2>&1 || { note "V ${who}_att3 FAILED"; continue; }
  A=runs/r4att/${who}_att3/model_final.pt; [ -f $A ] || A=runs/${who}_att_g1/${who}_att3/model_final.pt
  note "V ${who}_xg3 start (300 iterations vs the G1 league + standing walker, draw -20)"
  KOTH_ROBOT=$who $PY scripts/train.py --rung ${who}_vs_g1 --name ${who}_xg3 --num-envs 2048 --iters 300 --warm $A --opponent $G1DUEL $L/walker.pt:stand \
      --frozen-deterministic --critic-warmup 10 --reward draw=-20 > $LOG/${who}_xg3.log 2>&1 || { note "V ${who}_xg3 FAILED"; continue; }
  CK=runs/${who}_vs_g1/${who}_xg3/model_final.pt
  { echo "# ${who}_att3 vs standing G1"; KOTH_ROBOT=$who ev --rung ${who}_att_g1 --ckpt $A --opponent $L/walker.pt --seeds 2 --num-envs 256 --seconds 15 --leg-contact-rule unity
    echo "# ${who}_xg3 vs G1 league"; KOTH_ROBOT=$who ev --rung ${who}_vs_g1 --ckpt $CK --opponent $G1DUEL --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
    echo "# ${who}_xg3 vs standing own body"; KOTH_ROBOT=$who ev --rung r4att --ckpt $CK --opponent runs/${who}_r1b/${who}_r1b/model_final.pt --seeds 2 --num-envs 256 --seconds 15 --leg-contact-rule unity; } > $LOG/${who}_xg3_eval.txt
  KOTH_ROBOT=$who $PY scripts/export_policy.py --ckpt $CK --name xg3 --value > $LOG/${who}_xg3_export.txt 2>&1
  note "V ${who}_xg3 done"
done
note "cross-body v3 finished"
