#!/usr/bin/env bash
# Kim recipe b, unattended: stand (kim_r0b) -> walk (kim_r1b, resumed) -> evals. Exports go to a scratch name
# (walker_r1b) so the brain the game ships is only replaced after its bars are checked.
#   bash scripts/archive/overnight_kim.sh [r0 iterations] [r1 iterations]
cd "$(dirname "$0")/../.." || exit 1
export KOTH_ROBOT=kim PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; R0=${1:-600}; R1=${2:-1000}; LOG=runs/overnight; mkdir -p $LOG
stamp() { date '+%H:%M:%S'; }
echo "$(stamp) kim_r0b start ($R0 iterations)" | tee -a $LOG/progress.txt
$PY scripts/train.py --rung kim_r0b --name kim_r0b --num-envs 4096 --iters $R0 --tensorboard > $LOG/kim_r0b.log 2>&1 || { echo "$(stamp) kim_r0b FAILED" | tee -a $LOG/progress.txt; exit 1; }
CK0=runs/kim_r0b/kim_r0b/model_final.pt
for pv in "" "--push-vel 2.0"; do $PY scripts/eval.py --rung kim_r0b --ckpt $CK0 --seeds 3 --num-envs 256 $pv 2>&1 | grep -E '^\{|VERDICT'; done > $LOG/kim_r0b_eval.txt
echo "$(stamp) kim_r0b done: $(grep -c VERDICT $LOG/kim_r0b_eval.txt) verdict lines" | tee -a $LOG/progress.txt
echo "$(stamp) kim_r1b start ($R1 iterations)" | tee -a $LOG/progress.txt
$PY scripts/train.py --rung kim_r1b --name kim_r1b --num-envs 4096 --iters $R1 --resume $CK0 > $LOG/kim_r1b.log 2>&1 || { echo "$(stamp) kim_r1b FAILED" | tee -a $LOG/progress.txt; exit 1; }
CK1=runs/kim_r1b/kim_r1b/model_final.pt
{ $PY scripts/eval.py --rung kim_r1b --ckpt $CK1 --seeds 3 --num-envs 256 2>&1 | grep -E '^\{|VERDICT'
  $PY scripts/eval.py --rung kim_r0b --ckpt $CK1 --seeds 3 --num-envs 256 2>&1 | grep -E '^\{|VERDICT'
  $PY scripts/eval.py --rung r3 --ckpt $CK1 --seeds 3 --num-envs 256 --leg-contact-rule unity 2>&1 | grep -E '^\{|VERDICT'; } > $LOG/kim_r1b_eval.txt
$PY scripts/export_policy.py --ckpt $CK1 --name walker_r1b --value > $LOG/kim_export.txt 2>&1
echo "$(stamp) kim_r1b done" | tee -a $LOG/progress.txt
