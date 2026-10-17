#!/usr/bin/env bash
# Judge a fighter on the champion's bars and its contact style.   bash scripts/judge.sh <name> [--shove]
# <name> is a checkpoint in runs/league. Prints win / loss / tie averaged over 3 seeds x 256 duels per test.
cd "$(dirname "$0")/.." || exit 1
L=runs/league; N=$1; S=$2; PY=.venv/Scripts/python.exe; CH=$(cat $L/champion.txt)
ev() { local tag=$1; shift
  $PY scripts/eval.py "$@" $S --seeds 3 --num-envs 256 2>&1 | grep '^{' | $PY -m scripts.league_util --summarize "$tag"; }
ev "vs standing walker" --rung r4att --ckpt $L/$N.pt --opponent runs/r1/r1_b/model_final.pt --seconds 15
ev "vs gens 3-7       " --rung r4league --ckpt $L/$N.pt --opponent $L/gen3.pt $L/gen4.pt $L/gen5.pt $L/gen6.pt $L/gen7.pt --seconds 25
ev "vs champion $CH" --rung r4league --ckpt $L/$N.pt --opponent $L/$CH.pt --seconds 25
SELF=$L/$N.pt; [ -n "$S" ] && SELF=$SELF:shove
ev "vs itself         " --rung r4league --ckpt $L/$N.pt --opponent $SELF --seconds 25
$PY scripts/contact_style.py --ckpt $L/$N.pt --opponent $SELF --no-noise $S 2>&1 | grep '^{'
