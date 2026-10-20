#!/usr/bin/env bash
# Day-2 queue: Kim velocity-tracking fine-tune, then the second free-for-all generation. Exports go to new names.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight; L=runs/league; mkdir -p $LOG
note() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG/progress.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }

# ---------------------------------------------------------------- Kim: tracking (R1 linear error 0.20 -> < 0.15)
note "E kim_r1c start (300 iterations from kim_r1b, tracking_lin_vel 1.5)"
if KOTH_ROBOT=kim $PY scripts/train.py --rung kim_r1b --name kim_r1c --num-envs 4096 --iters 300 --resume runs/kim_r1b/kim_r1b/model_final.pt \
      --reward tracking_lin_vel=1.5 > $LOG/kim_r1c.log 2>&1; then
  CK=runs/kim_r1b/kim_r1c/model_final.pt
  { KOTH_ROBOT=kim ev --rung kim_r1b --ckpt $CK --seeds 3 --num-envs 256
    KOTH_ROBOT=kim ev --rung kim_r0b --ckpt $CK --seeds 3 --num-envs 256
    KOTH_ROBOT=kim ev --rung r3 --ckpt $CK --seeds 3 --num-envs 256 --leg-contact-rule unity; } > $LOG/kim_r1c_eval.txt
  KOTH_ROBOT=kim $PY scripts/export_policy.py --ckpt $CK --name walker_r1c --value > $LOG/kim_r1c_export.txt 2>&1
  note "E done"
else note "E FAILED"; fi

# ---------------------------------------------------------------- free-for-all, generation 2
OLD="$L/careful1.pt $L/careful1.pt $L/gen7.pt $L/gen1.pt $L/walker.pt:stand"
F1=runs/r5ffa/ffa1/model_final.pt
NEW="$F1 $F1 $L/careful1.pt $L/gen7.pt $L/gen1.pt $L/walker.pt:stand"
note "F r5ffa ffa2 start (600 iterations, warm from ffa1, pool with ffa1)"
if $PY scripts/train.py --rung r5ffa --name ffa2 --num-envs 1536 --iters 600 --warm $F1 --opponent $NEW > $LOG/ffa2.log 2>&1; then
  CK=runs/r5ffa/ffa2/model_final.pt
  { echo "# ffa2 vs old pool (ffa1's pool)"; ev --rung r5ffa --ckpt $CK --opponent $OLD --seeds 3 --num-envs 256 --seconds 20
    echo "# ffa2 vs new pool (with ffa1)"; ev --rung r5ffa --ckpt $CK --opponent $NEW --seeds 3 --num-envs 256 --seconds 20
    echo "# ffa1 vs new pool (with ffa1)"; ev --rung r5ffa --ckpt $F1 --opponent $NEW --seeds 3 --num-envs 256 --seconds 20
    echo "# ffa2 two-fighter vs pool"; ev --rung r4league --combat2 --ckpt $CK --opponent $L/careful1.pt $L/gen7.pt $L/gen1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity; } > $LOG/ffa2_eval.txt
  $PY scripts/export_policy.py --ckpt $CK --name ffa2 --value > $LOG/ffa2_export.txt 2>&1
  note "F done"
else note "F FAILED"; fi
note "day-2 queue finished"
