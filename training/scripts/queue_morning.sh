#!/usr/bin/env bash
# Cross-body duels, second try (2026-10-19). trump_xg1 / kim_xg1 trained against frozen G1 brains acting with the
# rebuilt actors' arbitrary std 0.3: those G1 mostly fell on their own (fight1 beats them 84 %, the deterministic ones
# 29 %), and both humans came out worse. Now: deterministic G1 opponents and a critic warm-up before the actor moves.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight; L=runs/league
note() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG/progress.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
G1DUEL="$L/careful1.pt $L/careful1.pt $L/gen7.pt $L/gen1.pt"
for who in trump kim; do
  if [ $who = trump ]; then BASE=runs/trump_r4bull/trump_fight1/model_final.pt; else BASE=runs/r4league/kim_fight1/model_final.pt; fi
  note "M ${who}_xg2 start (300 iterations from fight1, deterministic G1, critic warm-up 20)"
  if KOTH_ROBOT=$who $PY scripts/train.py --rung ${who}_vs_g1 --name ${who}_xg2 --num-envs 2048 --iters 300 --warm $BASE --opponent $G1DUEL \
        --frozen-deterministic --critic-warmup 20 > $LOG/${who}_xg2.log 2>&1; then
    CK=runs/${who}_vs_g1/${who}_xg2/model_final.pt
    { echo "# ${who}_xg2 vs G1 league"; KOTH_ROBOT=$who ev --rung ${who}_vs_g1 --ckpt $CK --opponent $G1DUEL --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
      echo "# ${who}_xg2 vs standing own body"; KOTH_ROBOT=$who ev --rung r4att --ckpt $CK --opponent runs/${who}_r1b/${who}_r1b/model_final.pt --seeds 2 --num-envs 256 --seconds 15 --leg-contact-rule unity; } > $LOG/${who}_xg2_eval.txt
    KOTH_ROBOT=$who $PY scripts/export_policy.py --ckpt $CK --name xg2 --value > $LOG/${who}_xg2_export.txt 2>&1
    note "M ${who}_xg2 done"
  else note "M ${who}_xg2 FAILED (see $LOG/${who}_xg2.log)"; fi
done
note "morning queue finished"
