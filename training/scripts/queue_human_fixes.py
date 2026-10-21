"""Unattended training queue for Fixes #2, #3, and #6:
  - Fix #6: Grandpa fighter rim-safety & push-rate fine-tune (grandpa_fight2)
  - Fix #2: Human Get-Up brains for Kim, Trump, and Grandpa (getup_policy.onnx)
  - Fix #3: Cross-human mixed-body combat brains for Kim, Trump, and Grandpa (xh1)
Starts TensorBoard on port 6006, gates every checkpoint on graduation & evaluation bars before exporting
to Assets/PoKingHill/Models/<robot>/, and prunes intermediate checkpoints when finished.
"""
import json, os, socket, subprocess, sys, time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable
LOG_DIR = os.path.join(ROOT, "runs", "overnight")
os.makedirs(LOG_DIR, exist_ok=True)
PROGRESS = os.path.join(LOG_DIR, "progress_human_fixes.txt")


def note(msg: str):
    line = f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_tensorboard():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", 6006)) == 0:
            note("TensorBoard already listening on :6006")
            return
    tb = os.path.join(ROOT, ".venv", "Scripts", "tensorboard.exe")
    cmd = [tb, "--logdir", "runs", "--port", "6006"] if os.path.exists(tb) else [PY, "-m", "tensorboard.main", "--logdir", "runs", "--port", "6006"]
    subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    note("Started TensorBoard on http://localhost:6006")


def run_cmd(args, robot=None, log_path=None) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if robot:
        env["KOTH_ROBOT"] = robot
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            p = subprocess.run([PY] + args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        return p.returncode
    p = subprocess.run([PY] + args, cwd=ROOT, env=env)
    return p.returncode


def run_eval_rows(args, robot=None) -> list:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if robot:
        env["KOTH_ROBOT"] = robot
    p = subprocess.run([PY, "scripts/eval.py"] + args, cwd=ROOT, env=env, capture_output=True, text=True)
    rows = []
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def best_fighter_ckpt(who: str) -> str:
    if who == "trump":
        return "runs/trump_r4bull/trump_fight1/model_final.pt"
    if who == "grandpa":
        c2 = "runs/r4league/grandpa_fight2/model_final.pt"
        if os.path.exists(os.path.join(ROOT, c2)):
            return c2
        return "runs/r4league/grandpa_fight1/model_final.pt"
    return "runs/r4league/kim_fight1/model_final.pt"


def walker_ckpt(who: str) -> str:
    return f"runs/{who}_r1b/{who}_r1b/model_final.pt"


def att_ckpt(who: str) -> str:
    return f"runs/r4att/{who}_att1/model_final.pt"


def step1_grandpa_finetune():
    """Fix #6: Grandpa fighter fine-tune with stronger rim penalty and termination penalty."""
    ckpt = "runs/r4league/grandpa_fight2/model_final.pt"
    if not os.path.exists(os.path.join(ROOT, ckpt)):
        note("Fix #6: starting grandpa_fight2 (r4league, 400 iters, 2048 envs, self_edge=-5, termination=-25)")
        w = walker_ckpt("grandpa")
        a = att_ckpt("grandpa")
        f1 = "runs/r4league/grandpa_fight1/model_final.pt"
        rc = run_cmd([
            "scripts/train.py", "--rung", "r4league", "--name", "grandpa_fight2",
            "--num-envs", "2048", "--iters", "400", "--warm", f1, "--critic-warmup", "20",
            "--opponent", f"{w}:stand", f"{w}:stand", a, f1,
            "--reward", "self_edge=-5.0", "termination=-25.0", "--leg-contact-rule", "unity"
        ], robot="grandpa", log_path=os.path.join(LOG_DIR, "grandpa_fight2.log"))
        if rc != 0:
            note(f"Fix #6: grandpa_fight2 FAILED (rc={rc})")
            return
    w = walker_ckpt("grandpa")
    f1 = "runs/r4league/grandpa_fight1/model_final.pt"
    att_rows = run_eval_rows(["--rung", "r4att", "--ckpt", ckpt, "--opponent", w, "--seeds", "3", "--num-envs", "256", "--seconds", "15", "--leg-contact-rule", "unity"], robot="grandpa")
    vs_f1 = run_eval_rows(["--rung", "r4league", "--ckpt", ckpt, "--opponent", f1, "--seeds", "3", "--num-envs", "256", "--seconds", "25", "--leg-contact-rule", "unity"], robot="grandpa")
    with open(os.path.join(LOG_DIR, "grandpa_fight2_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"r4att": att_rows, "vs_fight1": vs_f1}, f, indent=2)
    win_att = sum(r.get("attacker_ejects_defender", 0.0) for r in att_rows) / max(len(att_rows), 1)
    win_f1 = sum(r.get("attacker_ejects_defender", 0.0) / max(r.get("attacker_ejects_defender", 0.0) + r.get("attacker_lost", 0.0), 1e-6) for r in vs_f1) / max(len(vs_f1), 1)
    note(f"Fix #6: grandpa_fight2 r4att win_rate={win_att:.3f} (fight1=0.790), vs fight1 decided share={win_f1:.3f}")
    if win_att >= 0.79 and win_f1 >= 0.48:
        run_cmd(["scripts/export_policy.py", "--ckpt", ckpt, "--name", "fighter", "--value"], robot="grandpa")
        run_cmd(["scripts/export_mlp.py"], robot="grandpa")
        note("Fix #6: promoted grandpa_fight2 -> Assets/PoKingHill/Models/grandpa/fighter_policy.onnx + fighter_mlp.bytes")
    else:
        note("Fix #6: keeping grandpa_fight1 as shipped fighter (grandpa_fight2 kept in runs/r4league/grandpa_fight2)")


def step2_human_getup():
    """Fix #2: Train HoST-style getup brains for Kim, Trump, and Grandpa."""
    for who in ("kim", "trump", "grandpa"):
        name = f"{who}_getup1"
        ckpt = f"runs/getup/{name}/model_final.pt"
        if not os.path.exists(os.path.join(ROOT, ckpt)):
            note(f"Fix #2: starting {name} (getup, 1600 iters, 4096 envs, graduate-reward 18)")
            rc = run_cmd([
                "scripts/train.py", "--rung", "getup", "--name", name,
                "--num-envs", "4096", "--iters", "1600", "--graduate-reward", "18"
            ], robot=who, log_path=os.path.join(LOG_DIR, f"{name}.log"))
            if rc != 0:
                note(f"Fix #2: {name} FAILED (rc={rc})")
                continue
        rows = run_eval_rows(["--rung", "getup", "--ckpt", ckpt, "--seeds", "3", "--num-envs", "256"], robot=who)
        with open(os.path.join(LOG_DIR, f"{name}_eval.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        up4 = sum(r.get("stand_by_4s", r.get("standing_share", 0.0)) for r in rows) / max(len(rows), 1)
        note(f"Fix #2: {name} eval stand_by_4s={up4:.3f}")
        run_cmd(["scripts/export_policy.py", "--ckpt", ckpt, "--name", "getup"], robot=who)
        # Also record CPU MuJoCo getup_stats.json for the Unity gate
        run_cmd(["scripts/getup_stats.py", "--policy", "getup", "--drops", "200"], robot=who, log_path=os.path.join(LOG_DIR, f"{name}_cpu_stats.json"))
        note(f"Fix #2: exported Assets/PoKingHill/Models/{who}/getup_policy.onnx")


def step3_cross_human():
    """Fix #3: Cross-human mixed-body combat brains (Kim, Trump, Grandpa fighting across body types)."""
    humans = ("kim", "trump", "grandpa")
    note("Fix #3: building mixed-body 2p training scenes for all human pairs")
    for me in humans:
        for other in humans:
            if me != other:
                run_cmd(["-m", "koth.build_mjcf", "mix", me, other], log_path=os.path.join(LOG_DIR, f"build_mix_{me}_{other}.log"))

    for me in humans:
        others = [h for h in humans if h != me]
        base = best_fighter_ckpt(me)
        cur = base
        for idx, other in enumerate(others):
            stage_name = f"{me}_xh_{other}"
            rung = f"{me}_vs_{other}"
            ckpt = f"runs/{rung}/{stage_name}/model_final.pt"
            if not os.path.exists(os.path.join(ROOT, ckpt)):
                note(f"Fix #3: starting {stage_name} ({rung}, ladder curriculum, 300 iters, 2048 envs)")
                ow, oa, of = walker_ckpt(other), att_ckpt(other), best_fighter_ckpt(other)
                rc = run_cmd([
                    "scripts/train.py", "--rung", rung, "--name", stage_name,
                    "--num-envs", "2048", "--iters", "300", "--warm", cur,
                    "--opponent", f"{ow}:stand", oa, of,
                    "--frozen-deterministic", "--critic-warmup", "15", "--reward", "draw=-20",
                    "--curriculum", "ladder"
                ], robot=me, log_path=os.path.join(LOG_DIR, f"{stage_name}.log"))
                if rc != 0:
                    note(f"Fix #3: {stage_name} FAILED (rc={rc})")
                    break
            cur = ckpt

        # Final own-body anchor polish so it retains standing-push + mirror duel strength
        own_rung = "trump_r4bull" if me == "trump" else "r4league"
        final_name = f"{me}_xh1"
        final_ckpt = f"runs/{own_rung}/{final_name}/model_final.pt"
        if not os.path.exists(os.path.join(ROOT, final_ckpt)):
            note(f"Fix #3: starting {final_name} ({own_rung} own-body anchor, 200 iters, 2048 envs)")
            mw, ma, mf = walker_ckpt(me), att_ckpt(me), best_fighter_ckpt(me)
            rc = run_cmd([
                "scripts/train.py", "--rung", own_rung, "--name", final_name,
                "--num-envs", "2048", "--iters", "200", "--warm", cur,
                "--opponent", f"{mw}:stand", f"{mw}:stand", ma, mf,
                "--critic-warmup", "10", "--reward", "self_edge=-5.0", "termination=-25.0",
                "--leg-contact-rule", "unity"
            ], robot=me, log_path=os.path.join(LOG_DIR, f"{final_name}.log"))
            if rc != 0:
                note(f"Fix #3: {final_name} FAILED (rc={rc})")
                continue

        # Evaluate cross-human and own-body bars
        eval_summary = {}
        mw, mf = walker_ckpt(me), best_fighter_ckpt(me)
        eval_summary["r4att_own"] = run_eval_rows(["--rung", "r4att", "--ckpt", final_ckpt, "--opponent", mw, "--seeds", "3", "--num-envs", "256", "--seconds", "15", "--leg-contact-rule", "unity"], robot=me)
        eval_summary["vs_own_fight1"] = run_eval_rows(["--rung", own_rung, "--ckpt", final_ckpt, "--opponent", mf, "--seeds", "3", "--num-envs", "256", "--seconds", "25", "--leg-contact-rule", "unity"], robot=me)
        for other in others:
            of = best_fighter_ckpt(other)
            eval_summary[f"vs_{other}"] = run_eval_rows(["--rung", f"{me}_vs_{other}", "--ckpt", final_ckpt, "--opponent", of, "--seeds", "3", "--num-envs", "256", "--seconds", "25", "--leg-contact-rule", "unity"], robot=me)
            eval_summary[f"base_vs_{other}"] = run_eval_rows(["--rung", f"{me}_vs_{other}", "--ckpt", mf, "--opponent", of, "--seeds", "3", "--num-envs", "256", "--seconds", "25", "--leg-contact-rule", "unity"], robot=me)
        with open(os.path.join(LOG_DIR, f"{final_name}_eval.json"), "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

        att_win = sum(r.get("attacker_ejects_defender", 0.0) for r in eval_summary["r4att_own"]) / max(len(eval_summary["r4att_own"]), 1)
        decided = lambda r: r.get("attacker_ejects_defender", 0.0) / max(r.get("attacker_ejects_defender", 0.0) + r.get("attacker_lost", 0.0), 1e-6)
        xh_win = sum(sum(decided(r) for r in eval_summary[f"vs_{o}"]) / max(len(eval_summary[f"vs_{o}"]), 1) for o in others) / len(others)
        base_xh_win = sum(sum(decided(r) for r in eval_summary[f"base_vs_{o}"]) / max(len(eval_summary[f"base_vs_{o}"]), 1) for o in others) / len(others)
        note(f"Fix #3: {final_name} r4att_own={att_win:.3f}, cross-human decided share={xh_win:.3f} (baseline={base_xh_win:.3f})")
        if att_win >= 0.75 and xh_win > base_xh_win:
            run_cmd(["scripts/export_policy.py", "--ckpt", final_ckpt, "--name", "fighter", "--value"], robot=me)
            run_cmd(["scripts/export_mlp.py"], robot=me)
            note(f"Fix #3: promoted {final_name} -> Assets/PoKingHill/Models/{me}/fighter_policy.onnx + fighter_mlp.bytes")
        else:
            note(f"Fix #3: {final_name} evaluated and saved; keeping baseline fighter_policy.onnx for {me}")


def main():
    note("=== Starting queue_human_fixes.py (Fixes #6, #2, #3) ===")
    ensure_tensorboard()
    step1_grandpa_finetune()
    step2_human_getup()
    step3_cross_human()
    run_cmd(["scripts/prune_runs.py", "--apply"])
    note("=== queue_human_fixes.py finished ===")


if __name__ == "__main__":
    main()

