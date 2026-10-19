#!/usr/bin/env bash
# Trump's bull charge (rung trump_bull: trump_r4bull + the styled action path with obs.BULL_POSE), warm from fight1.
# fight2 (reward weights alone: belly_push 8, arm_contact -4, push_out 0) lost to fight1 and touched more with its arms.
cd "$(dirname "$0")/.." || exit 1
export KOTH_ROBOT=trump PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight
note() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG/progress.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
F1=runs/trump_r4bull/trump_fight1/model_final.pt; W=runs/trump_r1b/trump_r1b/model_final.pt
note "T3 trump_bull1 start (400 iterations from fight1, bull pose)"
$PY scripts/train.py --rung trump_bull --name trump_bull1 --num-envs 2048 --iters 400 --warm $F1 --opponent $F1 $F1 $W:stand > $LOG/trump_bull1.log 2>&1 || { note "T3 FAILED"; exit 1; }
B=runs/trump_bull/trump_bull1/model_final.pt
{ echo "# bull1 vs standing Trump"; ev --rung r4att --ckpt $B --opponent $W --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity --shove --style-pose bull
  echo "# style bull1"; $PY scripts/contact_style.py --ckpt $B --opponent $W --shove --style-pose bull 2>&1 | tail -1
  echo "# style fight1"; $PY scripts/contact_style.py --ckpt $F1 --opponent $W 2>&1 | tail -1; } > $LOG/trump_bull1_eval.txt
$PY scripts/export_policy.py --ckpt $B --name bull --value > $LOG/trump_bull1_export.txt 2>&1
note "T3 done"
