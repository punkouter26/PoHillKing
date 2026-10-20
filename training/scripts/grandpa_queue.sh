#!/usr/bin/env bash
# Grandpa, unattended (Trump's queue with Kim's fighting recipe): waits until queue_options.sh has finished (one GPU
# run at a time), then stand (grandpa_r0b) -> walk (grandpa_r1b, resumed, tracking weight 1.5 from the start) ->
# push a standing Grandpa off (r4att) -> a three-way pool (r4league) -> evals and exports to Models/grandpa/
# (walker_policy, fighter_policy, each with its critic) and the Tactician's weights (export_mlp.py).
#   bash scripts/grandpa_queue.sh [r0 iterations] [r1 iterations]
cd "$(dirname "$0")/.." || exit 1
export KOTH_ROBOT=grandpa PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; R0=${1:-600}; R1=${2:-1000}; LOG=runs/overnight; mkdir -p $LOG
note() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG/progress_grandpa.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }

if [ -f $LOG/progress_opts.txt ] && ! grep -q "options queue finished" $LOG/progress_opts.txt; then
  note "waiting for queue_options.sh to finish"
  until grep -q "options queue finished" $LOG/progress_opts.txt; do sleep 60; done
fi

note "G grandpa_r0b start ($R0 iterations)"
$PY scripts/train.py --rung grandpa_r0b --name grandpa_r0b --num-envs 4096 --iters $R0 --tensorboard > $LOG/grandpa_r0b.log 2>&1 || { note "G grandpa_r0b FAILED"; exit 1; }
CK0=runs/grandpa_r0b/grandpa_r0b/model_final.pt
for pv in "" "--push-vel 2.0"; do ev --rung grandpa_r0b --ckpt $CK0 --seeds 3 --num-envs 256 $pv; done > $LOG/grandpa_r0b_eval.txt
note "G grandpa_r0b done"

note "G grandpa_r1b start ($R1 iterations)"
$PY scripts/train.py --rung grandpa_r1b --name grandpa_r1b --num-envs 4096 --iters $R1 --resume $CK0 --reward tracking_lin_vel=1.5 > $LOG/grandpa_r1b.log 2>&1 || { note "G grandpa_r1b FAILED"; exit 1; }
CK1=runs/grandpa_r1b/grandpa_r1b/model_final.pt
{ ev --rung grandpa_r1b --ckpt $CK1 --seeds 3 --num-envs 256
  ev --rung grandpa_r0b --ckpt $CK1 --seeds 3 --num-envs 256
  ev --rung r3 --ckpt $CK1 --seeds 3 --num-envs 256 --leg-contact-rule unity; } > $LOG/grandpa_r1b_eval.txt
$PY scripts/export_policy.py --ckpt $CK1 --name walker --value > $LOG/grandpa_walker_export.txt 2>&1
note "G grandpa_r1b done"

note "G r4att grandpa_att1 start (warm from the walker)"
$PY scripts/train.py --rung r4att --name grandpa_att1 --num-envs 2048 --iters 400 --warm $CK1 --opponent $CK1:stand --critic-warmup 10 \
    --leg-contact-rule unity > $LOG/grandpa_att1.log 2>&1 || { note "G grandpa_att1 FAILED"; exit 1; }
KA=runs/r4att/grandpa_att1/model_final.pt
note "G r4league grandpa_fight1 start"
$PY scripts/train.py --rung r4league --name grandpa_fight1 --num-envs 2048 --iters 300 --warm $KA --opponent $KA $KA $CK1:stand --leg-contact-rule unity > $LOG/grandpa_fight1.log 2>&1
KF=runs/r4league/grandpa_fight1/model_final.pt; [ -f $KF ] || KF=$KA
{ echo "# grandpa fighter vs a standing Grandpa"; ev --rung r4att --ckpt $KF --opponent $CK1 --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity
  echo "# grandpa fighter vs itself"; ev --rung r4league --ckpt $KF --opponent $KF --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity; } > $LOG/grandpa_fight_eval.txt
$PY scripts/export_policy.py --ckpt $KF --name fighter --value > $LOG/grandpa_fight_export.txt 2>&1
KOTH_ROBOT= $PY scripts/export_mlp.py > $LOG/grandpa_mlp_export.txt 2>&1
note "G done"
