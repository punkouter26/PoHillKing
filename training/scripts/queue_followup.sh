#!/usr/bin/env bash
# Follow-up of the reward audit (2026-10-20, option 7): careful1's rung clips the shaped reward at zero on 18 % of steps,
# erasing 13 % of the penalty mass. noclip1 repeats duel_u1's recipe with the clip off (--no-reward-clip).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight; L=runs/league
note() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG/progress_opts.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
DPOOL="$L/walker.pt:stand $L/walker.pt:stand $L/gen7.pt $L/careful1.pt $L/gen1.pt"
LEAGUE="$L/careful1.pt $L/gen7.pt $L/gen1.pt"
name=noclip1; CK=runs/r4league/$name/model_final.pt
if [ ! -f $CK ]; then
  note "$name start (r4league, --no-reward-clip, 400 iterations, 2048 worlds)"
  $PY scripts/train.py --rung r4league --name $name --num-envs 2048 --iters 400 --warm $L/careful1.pt --opponent $DPOOL --critic-warmup 20 \
      --reward self_edge=-5.0 termination=-25.0 --leg-contact-rule unity --no-reward-clip > $LOG/$name.log 2>&1 || { note "$name FAILED"; exit 1; }
fi
{ echo "# $name vs standing walker"; ev --rung r4att --ckpt $CK --opponent $L/walker.pt --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity
  echo "# $name vs league (careful1, gen7, gen1)"; ev --rung r4league --ckpt $CK --opponent $LEAGUE --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
  echo "# $name vs careful1"; ev --rung r4league --ckpt $CK --opponent $L/careful1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
  echo "# $name vs duel_u1 (control)"; ev --rung r4league --ckpt $CK --opponent runs/r4league/duel_u1/model_final.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity; } > $LOG/${name}_eval.txt
note "$name done; follow-up finished"
