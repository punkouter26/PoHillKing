#!/usr/bin/env bash
# Kim's first two rungs: stand under kicks and thrown boxes (r0), then walk to command (r1, continued from r0).
#   bash scripts/kim_ladder.sh [r0 iterations] [r1 iterations]
cd "$(dirname "$0")/.." || exit 1
export KOTH_ROBOT=kim; PY=.venv/Scripts/python.exe; R0=${1:-300}; R1=${2:-1500}
$PY scripts/train.py --rung r0 --name kim_r0 --iters $R0 > runs/kim_r0.log 2> runs/kim_r0.err || exit 1
$PY scripts/eval.py --rung r0 --ckpt runs/r0/kim_r0/model_final.pt --seeds 3 --num-envs 256 2>&1 | grep -E '^\{|VERDICT' > runs/kim_r0_eval.log
$PY scripts/train.py --rung r1 --name kim_r1 --iters $R1 --resume runs/r0/kim_r0/model_final.pt > runs/kim_r1.log 2> runs/kim_r1.err || exit 1
for r in r0 r1; do $PY scripts/eval.py --rung $r --ckpt runs/r1/kim_r1/model_final.pt --seeds 3 --num-envs 256 2>&1 | grep -E '^\{|VERDICT'; done > runs/kim_r1_eval.log
