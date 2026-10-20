#!/usr/bin/env bash
# Drive saturation audit (scripts/drive_audit.py --ckpt) of every shipped brain; JSON tables in runs/audits/.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe; OUT=runs/audits; mkdir -p $OUT
run() { local name=$1; shift; echo "== $name"; "$@" --json $OUT/drive_$name.json 2>&1 | grep -v -e "^Module " -e Warning | sed -n '/^rollout/,$p'; }
run careful1 $PY scripts/drive_audit.py --ckpt runs/league/careful1.pt --rung r4league --num-envs 128
run gen7     $PY scripts/drive_audit.py --ckpt runs/league/gen7.pt --rung r4league --num-envs 128
run ffa3     $PY scripts/drive_audit.py --ckpt runs/r5ffa/ffa3/model_final.pt --rung r5ffa --tag koth_4p --num-envs 64
run getup3   $PY scripts/drive_audit.py --ckpt runs/getup/getup3/model_final.pt --rung getup --tag flat_1p --num-envs 128 --seconds 8
KOTH_ROBOT=kim   run kim_fight1   $PY scripts/drive_audit.py --ckpt runs/r4league/kim_fight1/model_final.pt --rung r4league --num-envs 128
KOTH_ROBOT=trump run trump_fight1 $PY scripts/drive_audit.py --ckpt runs/trump_r4bull/trump_fight1/model_final.pt --rung trump_r4bull --num-envs 128
