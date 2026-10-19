#!/usr/bin/env bash
# Trump, unattended: stand (trump_r0b) -> walk (trump_r1b, resumed, Kim's tracking weight 1.5 from the start) ->
# attack a standing Trump in his style (r4att + belly_push) -> the style rung against a pool (trump_r4bull) -> evals
# and exports to Models/trump/ (walker_policy, fighter_policy, each with its critic).
#   bash scripts/trump_queue.sh [r0 iterations] [r1 iterations]
cd "$(dirname "$0")/.." || exit 1
export KOTH_ROBOT=trump PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; R0=${1:-600}; R1=${2:-1000}; LOG=runs/overnight; mkdir -p $LOG
note() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG/progress.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
STYLE="belly_push=4.0 arm_contact=-1.0 push_out=1.0"

note "T trump_r0b start ($R0 iterations)"
$PY scripts/train.py --rung trump_r0b --name trump_r0b --num-envs 4096 --iters $R0 --tensorboard > $LOG/trump_r0b.log 2>&1 || { note "T trump_r0b FAILED"; exit 1; }
CK0=runs/trump_r0b/trump_r0b/model_final.pt
for pv in "" "--push-vel 2.0"; do ev --rung trump_r0b --ckpt $CK0 --seeds 3 --num-envs 256 $pv; done > $LOG/trump_r0b_eval.txt
note "T trump_r0b done"

note "T trump_r1b start ($R1 iterations)"
$PY scripts/train.py --rung trump_r1b --name trump_r1b --num-envs 4096 --iters $R1 --resume $CK0 --reward tracking_lin_vel=1.5 > $LOG/trump_r1b.log 2>&1 || { note "T trump_r1b FAILED"; exit 1; }
CK1=runs/trump_r1b/trump_r1b/model_final.pt
{ ev --rung trump_r1b --ckpt $CK1 --seeds 3 --num-envs 256
  ev --rung trump_r0b --ckpt $CK1 --seeds 3 --num-envs 256
  ev --rung r3 --ckpt $CK1 --seeds 3 --num-envs 256 --leg-contact-rule unity; } > $LOG/trump_r1b_eval.txt
$PY scripts/export_policy.py --ckpt $CK1 --name walker --value > $LOG/trump_walker_export.txt 2>&1
note "T trump_r1b done"

note "T r4att trump_att1 start (warm from the walker, his style)"
$PY scripts/train.py --rung r4att --name trump_att1 --num-envs 2048 --iters 400 --warm $CK1 --opponent $CK1:stand --critic-warmup 10 \
    --leg-contact-rule unity --reward $STYLE > $LOG/trump_att1.log 2>&1 || { note "T trump_att1 FAILED"; exit 1; }
KA=runs/r4att/trump_att1/model_final.pt
note "T trump_r4bull trump_fight1 start"
$PY scripts/train.py --rung trump_r4bull --name trump_fight1 --num-envs 2048 --iters 300 --warm $KA --opponent $KA $KA $CK1:stand > $LOG/trump_fight1.log 2>&1
KF=runs/trump_r4bull/trump_fight1/model_final.pt; [ -f $KF ] || KF=$KA
{ echo "# trump fighter vs standing Trump"; ev --rung r4att --ckpt $KF --opponent $CK1 --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity
  echo "# trump fighter vs itself"; ev --rung trump_r4bull --ckpt $KF --opponent $KF --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity; } > $LOG/trump_fight_eval.txt
$PY scripts/export_policy.py --ckpt $KF --name fighter --value > $LOG/trump_fight_export.txt 2>&1
note "T done"
