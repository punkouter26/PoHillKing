#!/usr/bin/env bash
# Night queue (2026-10-18), part 2, after part 1 (scripts/queue_night.sh): cross-body training. Every brain so far
# fought only its own body, while the game mixes them (Trump and Kim 0 of 8 six-way rounds). Mixed scenes come from
# `python -m koth.build_mjcf mix <body> ...` (assets/g1).
#   X: Trump and Kim duel the G1 league brains (rungs trump_vs_g1, kim_vs_g1, 112 inputs)
#   F: then a four-way free-for-all against three G1 (trump_ffa, kim_ffa, 121 inputs, warm from X)
# Trump continues from the bull charge (bull2) only if it passed part 1's bars; otherwise from fight1.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe; LOG=runs/overnight; L=runs/league
note() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG/progress.txt; }
ev() { $PY scripts/eval.py "$@" 2>&1 | grep -E '^\{|VERDICT'; }
until grep -q "night queue part 1 finished" $LOG/progress.txt; do sleep 60; done

G1DUEL="$L/careful1.pt $L/careful1.pt $L/gen7.pt $L/gen1.pt"
G1FFA="runs/r5ffa/ffa2/model_final.pt runs/r5ffa/ffa2/model_final.pt $L/careful1.pt $L/gen7.pt"
TF1=runs/trump_r4bull/trump_fight1/model_final.pt; TB2=runs/trump_bull/trump_bull2/model_final.pt; KF1=runs/r4league/kim_fight1/model_final.pt
# bull2 passes: >= 70 % vs a standing Trump (r4att PASS) and >= 40 % wins against fight1
BULL_OK=$($PY - <<'EOF'
import json, re
try: t = open("runs/overnight/trump_bull2_eval.txt").read()
except OSError: print(0); raise SystemExit
att = re.search(r"VERDICT r4att PASS", t) is not None
rows = [json.loads(l) for l in t.split("# bull2 vs fight1")[1].split("#")[0].splitlines() if l.startswith("{")] if "# bull2 vs fight1" in t else []
win = sum(r["attacker_ejects_defender"] for r in rows) / max(len(rows), 1)
print(1 if att and win >= 0.4 else 0)
EOF
)
if [ "$BULL_OK" = 1 ]; then TBASE=$TB2; TSTYLE="--style-pose bull"; note "X Trump base: bull2 (passed)"; else TBASE=$TF1; TSTYLE=""; note "X Trump base: fight1 (bull2 did not pass)"; fi

for who in trump kim; do
  if [ $who = trump ]; then BASE=$TBASE; STYLE=$TSTYLE; else BASE=$KF1; STYLE=""; fi
  note "X ${who}_vs_g1 start (400 iterations, warm from $BASE)"
  if KOTH_ROBOT=$who $PY scripts/train.py --rung ${who}_vs_g1 --name ${who}_xg1 --num-envs 2048 --iters 400 --warm $BASE --opponent $G1DUEL $STYLE > $LOG/${who}_xg1.log 2>&1; then
    CK=runs/${who}_vs_g1/${who}_xg1/model_final.pt
    { echo "# ${who}_xg1 vs G1 league"; KOTH_ROBOT=$who ev --rung ${who}_vs_g1 --ckpt $CK --opponent $G1DUEL --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity $STYLE ${STYLE:+--shove}
      echo "# base vs G1 league"; KOTH_ROBOT=$who ev --rung ${who}_vs_g1 --ckpt $BASE --opponent $G1DUEL --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity $STYLE ${STYLE:+--shove}; } > $LOG/${who}_xg1_eval.txt
    KOTH_ROBOT=$who $PY scripts/export_policy.py --ckpt $CK --name xg1 --value > $LOG/${who}_xg1_export.txt 2>&1
    # actor-only copies (scripts/onnx_to_ckpt.py) so the 112-input brains can be compared in the 121-input free-for-all
    $PY scripts/onnx_to_ckpt.py --onnx ../Assets/PoKingHill/Models/$who/xg1_policy.onnx --out $L/${who}_xg1.pt > /dev/null 2>&1
    note "X ${who}_xg1 done"
  else note "X ${who}_xg1 FAILED (see $LOG/${who}_xg1.log)"; CK=$BASE; fi
  if [ $who = trump ] && [ "$BULL_OK" = 1 ]; then BONNX=bull2; else BONNX=fighter; fi
  $PY scripts/onnx_to_ckpt.py --onnx ../Assets/PoKingHill/Models/$who/${BONNX}_policy.onnx --out $L/${who}_base.pt > /dev/null 2>&1

  note "F ${who}_ffa start (300 iterations, warm from ${who}_xg1)"
  if KOTH_ROBOT=$who $PY scripts/train.py --rung ${who}_ffa --name ${who}_ffa1 --num-envs 1536 --iters 300 --warm $CK --opponent $G1FFA $STYLE > $LOG/${who}_ffa1.log 2>&1; then
    FK=runs/${who}_ffa/${who}_ffa1/model_final.pt
    { echo "# ${who}_ffa1 vs three G1"; KOTH_ROBOT=$who ev --rung ${who}_ffa --ckpt $FK --opponent $G1FFA --seeds 3 --num-envs 256 --seconds 20 $STYLE ${STYLE:+--shove}
      echo "# ${who}_xg1 (112) vs three G1"; KOTH_ROBOT=$who ev --rung ${who}_ffa --ckpt $L/${who}_xg1.pt --opponent $G1FFA --seeds 3 --num-envs 256 --seconds 20 $STYLE ${STYLE:+--shove}
      echo "# base vs three G1"; KOTH_ROBOT=$who ev --rung ${who}_ffa --ckpt $L/${who}_base.pt --opponent $G1FFA --seeds 3 --num-envs 256 --seconds 20 $STYLE ${STYLE:+--shove}; } > $LOG/${who}_ffa1_eval.txt
    KOTH_ROBOT=$who $PY scripts/export_policy.py --ckpt $FK --name ffa --value > $LOG/${who}_ffa1_export.txt 2>&1
    note "F ${who}_ffa1 done"
  else note "F ${who}_ffa1 FAILED (see $LOG/${who}_ffa1.log)"; fi
done
note "night queue part 2 finished"
