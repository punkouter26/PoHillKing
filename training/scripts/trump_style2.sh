#!/usr/bin/env bash
# Trump's style, second pass: trump_fight1 pushed with his trunk on 34 % of close-contact steps and his arms on 10 %,
# barely different from Kim's ordinary fighter (30 % / 8 %, scripts/style_stats.py). Pay only for trunk shoves
# (push_out 0, belly_push 8) and charge arm contact four times as much, warm from fight1.
cd "$(dirname "$0")/.." || exit 1
export KOTH_ROBOT=trump PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight
note() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG/progress.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
F1=runs/trump_r4bull/trump_fight1/model_final.pt; W=runs/trump_r1b/trump_r1b/model_final.pt
note "T2 trump_fight2 start (300 iterations from fight1, belly_push 8, arm_contact -4, push_out 0)"
$PY scripts/train.py --rung trump_r4bull --name trump_fight2 --num-envs 2048 --iters 300 --warm $F1 --opponent $F1 $F1 $W:stand \
    --reward belly_push=8.0 arm_contact=-4.0 push_out=0.0 > $LOG/trump_fight2.log 2>&1 || { note "T2 FAILED"; exit 1; }
F2=runs/trump_r4bull/trump_fight2/model_final.pt
{ echo "# fight2 vs standing Trump"; ev --rung r4att --ckpt $F2 --opponent $W --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity
  echo "# fight2 vs fight1"; ev --rung trump_r4bull --ckpt $F2 --opponent $F1 --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
  echo "# style fight2"; $PY scripts/contact_style.py --ckpt $F2 --opponent $W 2>&1 | tail -1; } > $LOG/trump_fight2_eval.txt
$PY scripts/export_policy.py --ckpt $F2 --name fighter2 --value > $LOG/trump_fight2_export.txt 2>&1
note "T2 done"
