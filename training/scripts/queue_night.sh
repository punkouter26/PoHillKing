#!/usr/bin/env bash
# Night queue (2026-10-18), part 1: Trump's bull charge eased in (trump_bull, style_ramp_iters 300), then the third
# G1 free-for-all generation. Exports go to new names; nothing the game ships is replaced here.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight; L=runs/league; mkdir -p $LOG
note() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG/progress.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }

# ---------------------------------------------------------------- N1: Trump's bull charge, eased in
F1=runs/trump_r4bull/trump_fight1/model_final.pt; W=runs/trump_r1b/trump_r1b/model_final.pt
note "N1 trump_bull2 start (600 iterations from fight1, bull pose eased in over 300)"
if KOTH_ROBOT=trump $PY scripts/train.py --rung trump_bull --name trump_bull2 --num-envs 2048 --iters 600 --warm $F1 --opponent $F1 $F1 $W:stand > $LOG/trump_bull2.log 2>&1; then
  B=runs/trump_bull/trump_bull2/model_final.pt
  { echo "# bull2 vs standing Trump"; KOTH_ROBOT=trump ev --rung r4att --ckpt $B --opponent $W --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity --shove --style-pose bull
    echo "# bull2 vs fight1"; KOTH_ROBOT=trump ev --rung trump_bull --ckpt $B --opponent $F1 --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity --shove --style-pose bull
    echo "# style bull2 vs fight1"; KOTH_ROBOT=trump $PY scripts/contact_style.py --ckpt $B --opponent $F1 --shove --style-pose bull 2>&1 | tail -1
    echo "# style fight1 vs fight1"; KOTH_ROBOT=trump $PY scripts/contact_style.py --ckpt $F1 --opponent $F1 2>&1 | tail -1; } > $LOG/trump_bull2_eval.txt
  KOTH_ROBOT=trump $PY scripts/export_policy.py --ckpt $B --name bull2 --value > $LOG/trump_bull2_export.txt 2>&1
  note "N1 done"
else note "N1 FAILED (see $LOG/trump_bull2.log)"; fi

# ---------------------------------------------------------------- N4: G1 free-for-all, generation 3
F2=runs/r5ffa/ffa2/model_final.pt; F1G=runs/r5ffa/ffa1/model_final.pt
POOL2="$F1G $F1G $L/careful1.pt $L/gen7.pt $L/gen1.pt $L/walker.pt:stand"          # ffa2's training pool
POOL3="$F2 $F2 $F1G $L/careful1.pt $L/gen7.pt $L/walker.pt:stand"
note "N4 r5ffa ffa3 start (500 iterations, warm from ffa2, pool with ffa2)"
if $PY scripts/train.py --rung r5ffa --name ffa3 --num-envs 1536 --iters 500 --warm $F2 --opponent $POOL3 > $LOG/ffa3.log 2>&1; then
  CK=runs/r5ffa/ffa3/model_final.pt
  { echo "# ffa3 vs ffa2's pool"; ev --rung r5ffa --ckpt $CK --opponent $POOL2 --seeds 3 --num-envs 256 --seconds 20
    echo "# ffa3 vs pool3 (with ffa2)"; ev --rung r5ffa --ckpt $CK --opponent $POOL3 --seeds 3 --num-envs 256 --seconds 20
    echo "# ffa2 vs pool3 (with ffa2)"; ev --rung r5ffa --ckpt $F2 --opponent $POOL3 --seeds 3 --num-envs 256 --seconds 20
    echo "# ffa3 two-fighter vs pool"; ev --rung r4league --ckpt $CK --opponent $L/careful1.pt $L/gen7.pt $L/gen1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity --combat2; } > $LOG/ffa3_eval.txt
  $PY scripts/export_policy.py --ckpt $CK --name ffa3 --value > $LOG/ffa3_export.txt 2>&1
  note "N4 done"
else note "N4 FAILED (see $LOG/ffa3.log)"; fi
note "night queue part 1 finished"
