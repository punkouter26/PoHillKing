#!/usr/bin/env bash
# Overnight queue after the Kim walk run (scripts/archive/overnight_kim.sh): waits for it, then trains
#   B  free-for-all brain r5ffa (121 inputs), warm from the champion (careful1, rebuilt from its ONNX)
#   C  duel fine-tune of the champion under the Unity rules (no own-leg-contact ending), plus a baseline eval
#   D  Kim's first fighting brain: push a standing Kim off (r4att), then a short league (r4league)
# Every export goes to a new name; the brains the game ships are replaced only after their numbers are checked.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight; mkdir -p $LOG
note() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG/progress.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
L=runs/league

until grep -qE "kim_r1b done|FAILED" $LOG/progress.txt 2>/dev/null; do sleep 60; done
note "queue start"

# ---------------------------------------------------------------- B: free-for-all
POOL="$L/careful1.pt $L/careful1.pt $L/gen7.pt $L/gen1.pt $L/walker.pt:stand"
note "B r5ffa ffa1 start (900 iterations, 1536 worlds)"
if $PY scripts/train.py --rung r5ffa --name ffa1 --num-envs 1536 --iters 900 --warm $L/careful1.pt --opponent $POOL --critic-warmup 20 > $LOG/ffa1.log 2>&1; then
  CK=runs/r5ffa/ffa1/model_final.pt
  { echo "# ffa1 vs pool, 4 fighters"; ev --rung r5ffa --ckpt $CK --opponent $POOL --seeds 3 --num-envs 256 --seconds 20
    echo "# careful1 (baseline) vs pool, 4 fighters"; ev --rung r5ffa --ckpt $L/careful1.pt --opponent $POOL --seeds 3 --num-envs 256 --seconds 20
    echo "# ffa1 two-fighter regression vs pool"; ev --rung r4league --combat2 --ckpt $CK --opponent $L/careful1.pt $L/gen7.pt $L/gen1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
    echo "# careful1 two-fighter baseline vs pool"; ev --rung r4league --ckpt $L/careful1.pt --opponent $L/careful1.pt $L/gen7.pt $L/gen1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity; } > $LOG/ffa1_eval.txt
  $PY scripts/export_policy.py --ckpt $CK --name ffa --value > $LOG/ffa_export.txt 2>&1
  note "B done"
else note "B FAILED (see $LOG/ffa1.log)"; fi

# ---------------------------------------------------------------- C: duel champion, Unity rules
DPOOL="$L/walker.pt:stand $L/walker.pt:stand $L/gen7.pt $L/careful1.pt $L/gen1.pt"
note "C r4league duel_u1 start (400 iterations, 2048 worlds)"
if $PY scripts/train.py --rung r4league --name duel_u1 --num-envs 2048 --iters 400 --warm $L/careful1.pt --opponent $DPOOL --critic-warmup 20 \
      --reward self_edge=-5.0 termination=-25.0 --leg-contact-rule unity > $LOG/duel_u1.log 2>&1; then
  CK=runs/r4league/duel_u1/model_final.pt
  for who in "$CK duel_u1" "$L/careful1.pt careful1"; do set -- $who
    { echo "# $2 vs standing walker at the rim"; ev --rung r4att --ckpt $1 --opponent $L/walker.pt --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity
      echo "# $2 vs gen7 + gen1"; ev --rung r4league --ckpt $1 --opponent $L/gen7.pt $L/gen1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
      echo "# $2 vs careful1"; ev --rung r4league --ckpt $1 --opponent $L/careful1.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
      echo "# $2 vs duel_u1"; ev --rung r4league --ckpt $1 --opponent $CK --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity; } >> $LOG/duel_u1_eval.txt
  done
  $PY scripts/export_policy.py --ckpt $CK --name duel_u1 --value > $LOG/duel_u1_export.txt 2>&1
  $PY scripts/export_policy.py --ckpt $L/careful1.pt --name careful1_check > /dev/null 2>&1    # round-trip check of the rebuilt champion
  note "C done"
else note "C FAILED (see $LOG/duel_u1.log)"; fi

# ---------------------------------------------------------------- D: Kim fights
export KOTH_ROBOT=kim
KW=runs/kim_r1b/kim_r1b/model_final.pt; [ -f $KW ] || KW=$L/kim_walker.pt
[ -f $KW ] || $PY scripts/onnx_to_ckpt.py --onnx ../Assets/PoKingHill/Models/kim/walker_policy.onnx --out $L/kim_walker.pt > /dev/null 2>&1
note "D kim r4att kim_att1 start (warm from $KW)"
if $PY scripts/train.py --rung r4att --name kim_att1 --num-envs 2048 --iters 400 --warm $KW --opponent $KW:stand --critic-warmup 10 --leg-contact-rule unity > $LOG/kim_att1.log 2>&1; then
  KA=runs/r4att/kim_att1/model_final.pt
  note "D kim r4league kim_fight1 start"
  $PY scripts/train.py --rung r4league --name kim_fight1 --num-envs 2048 --iters 300 --warm $KA --opponent $KA $KA $KW:stand --leg-contact-rule unity > $LOG/kim_fight1.log 2>&1
  KF=runs/r4league/kim_fight1/model_final.pt; [ -f $KF ] || KF=$KA
  { echo "# kim fighter vs standing Kim"; ev --rung r4att --ckpt $KF --opponent $KW --seeds 3 --num-envs 256 --seconds 15 --leg-contact-rule unity
    echo "# kim fighter vs itself"; ev --rung r4league --ckpt $KF --opponent $KF --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity; } > $LOG/kim_fight_eval.txt
  $PY scripts/export_policy.py --ckpt $KF --name fighter --value > $LOG/kim_fight_export.txt 2>&1
  note "D done"
else note "D FAILED (see $LOG/kim_att1.log)"; fi
note "queue finished"
