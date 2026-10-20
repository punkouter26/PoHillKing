#!/usr/bin/env bash
# Reward audit (scripts/reward_audit.py) of every shipped brain on the rung and reward settings it was trained with.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe; OUT=runs/audits; L=runs/league; mkdir -p $OUT
DPOOL="$L/walker.pt:stand $L/gen7.pt $L/careful1.pt $L/gen1.pt"
run() { local name=$1; shift; echo "== $name"; "$@" --num-envs 64 --json $OUT/reward_$name.json 2>&1 | grep -v "^Module " | sed -n '/learner steps/,$p'; }
run careful1 $PY scripts/reward_audit.py --rung r4league --ckpt $L/careful1.pt --opponent $DPOOL --reward self_edge=-5 termination=-25 --leg-contact-rule unity
run ffa3     $PY scripts/reward_audit.py --rung r5ffa --ckpt runs/r5ffa/ffa3/model_final.pt --opponent $L/careful1.pt $L/careful1.pt $L/gen7.pt $L/gen1.pt $L/walker.pt:stand
run walker   $PY scripts/reward_audit.py --rung r1 --ckpt $L/walker.pt --seconds 20
run getup3   $PY scripts/reward_audit.py --rung getup --ckpt runs/getup/getup3/model_final.pt --seconds 8
KOTH_ROBOT=kim   run kim_fight1   $PY scripts/reward_audit.py --rung r4league --ckpt runs/r4league/kim_fight1/model_final.pt --opponent runs/r4league/kim_fight1/model_final.pt --leg-contact-rule unity
KOTH_ROBOT=trump run trump_fight1 $PY scripts/reward_audit.py --rung trump_r4bull --ckpt runs/trump_r4bull/trump_fight1/model_final.pt --opponent runs/trump_r4bull/trump_fight1/model_final.pt
KOTH_ROBOT=trump run trump_xg3    $PY scripts/reward_audit.py --rung trump_vs_g1 --ckpt runs/trump_vs_g1/trump_xg3/model_final.pt --opponent $L/careful1.pt $L/careful1.pt $L/gen7.pt $L/gen1.pt $L/walker.pt:stand --reward draw=-20
