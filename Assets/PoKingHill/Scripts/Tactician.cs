using System;
using System.Collections.Generic;
using System.Threading;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Lookahead planner for the fighting brains (GameSettings.Tactician). Every decideEvery seconds of simulation, each
    /// planning fighter tries the candidate tactics (press, flank left, flank right, regroup to the centre): for each one
    /// the whole scene is copied (mj_copyData) and simulated horizon seconds ahead on a worker thread, with the planner
    /// following that tactic and its opponent its current one, both driven by their own networks (MlpNet, the same
    /// weights as the ONNX brains) through a copy of PolicyRunner's tick; everyone else holds their targets. The end
    /// state is scored by the planner's critic when it has one (Fighter.valueMlp), else by a physics estimate (margin
    /// to the rim, uprightness, height, for me minus my opponent); being knocked out or knocking out in the rollout is
    /// worth -/+ 20. The planner switches when another tactic beats the current one by switchMargin. A tactic only moves
    /// the goal point of the trained goal law (Tactics.Goal), so commands stay in their trained ranges.
    /// In play the rollouts run in the background and the choice lands a few frames later; in the statistics gate
    /// (gateSeat) the physics waits for them, so a run is the same at any time scale. Allocation-free per decision.
    /// Measured 2026-10-19 (ParityBatch.RunTacticianStats, champion vs champion, 200 rounds, A plans): it makes the
    /// fighter weaker. Press only 97-100 (the hook is harmless); 0.6 s 27-173; 1.2 s + Press prior 0.5 64-133; 2 s +
    /// 0.3 65-128; 1.2 s + 1.0 76-120 (the defaults); 3 averaged samples 74-125. The brains only know the Press law and
    /// contact fights are too chaotic for short rollouts to find a deviation that pays: an experimental show option,
    /// off by default (rl_optimization_log.md).
    /// </summary>
    public unsafe class Tactician : MonoBehaviour
    {
        public DemoDirector director;
        public float decideEvery = 0.75f, horizon = 1.2f, switchMargin = 0.15f;
        [Tooltip("Worker threads for the rollouts")] public int threads = 4;
        [Tooltip("Statistics gate: only this seat plans, whatever GameSettings.Tactician says; -1 = follow the setting")] public int gateSeat = -1;
        public static readonly Tactic[] Candidates = { Tactic.Press, Tactic.FlankLeft, Tactic.FlankRight, Tactic.Regroup };
        const int MaxPlanners = 6, MaxSamples = 4, MaxJobs = MaxPlanners * 4 * MaxSamples, NJ = 29;

        /// <summary>Decisions taken and the last scores per candidate, for the HUD and the log.</summary>
        public int Decisions { get; private set; }
        public double LastBatchMs { get; private set; }
        /// <summary>How often each candidate was chosen (Candidates order), since the component was enabled.</summary>
        public readonly int[] Chosen = new int[4];
        [Tooltip("Prior for Press, the trained law, added to its rollout score: leave it only when another tactic is clearly better")] public float pressBonus = 1.0f;
        [Tooltip("Candidates tried (Candidates order); a gate can narrow it to Press alone as a sanity check")] public int candidateCount = 4;
        [Tooltip("Rollouts per candidate, averaged. Contact fights are chaotic: one rollout is a noisy estimate, and the best of " +
                 "several noisy candidates beats Press by chance. Each extra sample starts with the pelvis velocities nudged by +-sampleNoise m/s.")]
        public int samples = 1; public float sampleNoise = 0.05f;

        // per planner (filled on the main thread at snapshot time)
        class Side
        {
            public PolicyRunner runner; public MlpNet net, value; public int obsDim; public JointMap map, opp, opp2;
            public GoalMode goal; public float stop, vmax; public bool shove; public float[] offsets = new float[NJ];
            public float[] last = new float[NJ], actF = new float[NJ]; public float phase; public Tactic tactic; public float keyZ;
        }
        class Plan { public readonly Side me = new(), them = new(); public int seat; public bool themActive; public readonly float[] score = new float[4]; public readonly float[,] sampled = new float[4, MaxSamples]; }
        class Job
        {
            public Plan plan; public int candidate, sample;
            public readonly float[] meLast = new float[NJ], meF = new float[NJ], themLast = new float[NJ], themF = new float[NJ];
            public readonly double[] meCtrl = new double[NJ], themCtrl = new double[NJ];
        }
        // Worker k runs jobs k, k + n, k + 2n, ... of a batch: it waits on its own event and signals the countdown when its
        // share is done, so a batch is finished exactly when the countdown is set and no worker touches the next one early.
        class Worker { public int index; public MujocoLib.mjData_* d; public readonly AutoResetEvent go = new(false); public float[] obs = new float[160], act = new float[NJ], scratch = new float[2048], cmd = new float[3], val = new float[1]; }

        readonly Plan[] _plans = new Plan[MaxPlanners]; readonly Job[] _jobs = new Job[MaxJobs]; Worker[] _workers = Array.Empty<Worker>();
        Thread[] _threads = Array.Empty<Thread>(); CountdownEvent _done;
        int _jobCount, _planCount; volatile bool _run, _busy;
        MujocoLib.mjModel_* _m; MujocoLib.mjData_* _snap; double[][] _held = Array.Empty<double[]>(); PolicyRunner[] _all = Array.Empty<PolicyRunner>();
        int _decimation; float _ctrlDt, _scale; double _nextDecision; ArenaSpec _arena = new(); float _outRadius;
        readonly Dictionary<TextAsset, MlpNet> _nets = new(); readonly System.Diagnostics.Stopwatch _sw = new();
        readonly int[] _footL = new int[16], _footR = new int[16];

        void OnEnable()
        {
            for (int i = 0; i < MaxPlanners; i++) _plans[i] = new Plan();
            for (int i = 0; i < MaxJobs; i++) _jobs[i] = new Job();
            MatchEvents.RoundStarted += OnRound;
            if (MjScene.InstanceExists) MjScene.Instance.preUpdateEvent += Pre;
        }

        void OnDisable()
        {
            MatchEvents.RoundStarted -= OnRound;
            if (MjScene.InstanceExists) MjScene.Instance.preUpdateEvent -= Pre;
            if (Decisions > 0) Debug.Log($"[Tactician] {Decisions} decisions; chosen press {Chosen[0]}, flank L {Chosen[1]}, flank R {Chosen[2]}, regroup {Chosen[3]}; horizon {horizon} s, press bonus {pressBonus}, samples {samples}");
            Shutdown();
        }

        void OnRound() { _nextDecision = 0; if (_busy) _done?.Wait(); _busy = false; }

        bool Enabled => director != null && (gateSeat >= 0 || (GameSettings.Tactician && director.statsRounds == 0));

        void Init()
        {
            _m = MjScene.Instance.Model; _snap = MujocoLib.mj_makeData(_m);
            _all = (PolicyRunner[])director.fighters.Clone();           // fixed body order: the director reorders its own array at every launch
            var spec = _all[0].Spec; _decimation = spec.decimation; _ctrlDt = spec.ctrl_dt; _scale = spec.action_scale; _arena = spec.arena ?? new ArenaSpec();
            _held = new double[_all.Length][]; for (int i = 0; i < _all.Length; i++) _held[i] = new double[NJ];
            int n = Mathf.Clamp(threads, 1, Mathf.Max(1, SystemInfo.processorCount - 1));
            _workers = new Worker[n]; _threads = new Thread[n]; _done = new CountdownEvent(0); _run = true;
            for (int i = 0; i < n; i++)
            {
                _workers[i] = new Worker { index = i, d = MujocoLib.mj_makeData(_m) }; int k = i;
                _threads[i] = new Thread(() => Loop(_workers[k])) { IsBackground = true, Name = "Tactician " + k }; _threads[i].Start();
            }
            for (int i = 0; i < _all.Length && i < _footL.Length; i++)
            {
                _footL[i] = MujocoLib.mj_name2id(_m, (int)MujocoLib.mjtObj.mjOBJ_BODY, _all[i].robotPrefix + "left_ankle_roll_link");
                _footR[i] = MujocoLib.mj_name2id(_m, (int)MujocoLib.mjtObj.mjOBJ_BODY, _all[i].robotPrefix + "right_ankle_roll_link");
            }
        }

        void Shutdown()
        {
            if (_threads.Length == 0) return;
            if (_busy) _done.Wait(3000);
            _run = false; foreach (var w in _workers) w.go.Set();
            foreach (var t in _threads) t.Join(2000);
            if (_m != null) { foreach (var w in _workers) if (w.d != null) MujocoLib.mj_deleteData(w.d); if (_snap != null) MujocoLib.mj_deleteData(_snap); }
            _threads = Array.Empty<Thread>(); _workers = Array.Empty<Worker>(); _snap = null; _m = null; _busy = false;
        }

        MlpNet Net(TextAsset t)
        {
            if (t == null) return null;
            if (!_nets.TryGetValue(t, out var net)) { try { net = MlpNet.Load(t.bytes); } catch (Exception e) { Debug.LogWarning($"[Tactician] {t.name}: {e.Message}"); net = null; } _nets[t] = net; }
            return net;
        }

        Fighter EntryOf(int seat) { int a = director.AgentOf(seat); return a >= 0 && a < director.roster.Length ? director.roster[a] : null; }

        void Pre(object sender, MjStepArgs args)
        {
            if (!Enabled || director.State != MatchState.Fight) return;
            if (_snap == null) { if (director.fighters.Length > 0 && director.fighters[0].Map != null) Init(); else return; }
            if (_busy && _done.IsSet) { Apply(); _busy = false; }
            var d = MjScene.Instance.Data;
            if (_busy || d->time < _nextDecision) return;
            _nextDecision = d->time + decideEvery;
            if (Schedule(d) && gateSeat >= 0) { _done.Wait(); Apply(); _busy = false; }
        }

        // Snapshot the scene and every planner's brain state, then release the jobs to the workers.
        bool Schedule(MujocoLib.mjData_* d)
        {
            _planCount = 0; _outRadius = director.OutRadiusNow;
            for (int i = 0; i < director.Playing && _planCount < MaxPlanners; i++)
            {
                var r = director.fighters[i];
                if (director.IsOut(i) || director.IsDown(i) || i == director.PuppetBody || r.goal != GoalMode.Opponent || r.opponent == null || r.tactic == Tactic.Lurk || r.GettingUp) continue;
                if (gateSeat >= 0 && i != gateSeat) continue;
                var e = EntryOf(i); var net = Net(e?.mlp); if (net == null || net.InputDim != r.policyObsDim) continue;
                var p = _plans[_planCount]; p.seat = i;
                Fill(p.me, r, net, Net(e.valueMlp));
                int oi = Array.IndexOf(director.fighters, r.opponent); var oe = oi >= 0 ? EntryOf(oi) : null; var onet = Net(oe?.mlp);
                p.themActive = onet != null && onet.InputDim == r.opponent.policyObsDim && !r.opponent.GettingUp && !r.opponent.paused && r.opponent.opponent != null;
                Fill(p.them, r.opponent, p.themActive ? onet : null, null);
                _planCount++;
            }
            if (_planCount == 0) return false;
            for (int i = 0; i < _all.Length; i++) { var r = _all[i]; for (int k = 0; k < NJ; k++) _held[i][k] = r.paused ? 0 : r.Ctrl[k]; }
            MujocoLib.mj_copyData(_snap, _m, d);
            _jobCount = 0;
            int nc = Mathf.Clamp(candidateCount, 1, Candidates.Length), ns = Mathf.Clamp(samples, 1, MaxSamples);
            for (int p = 0; p < _planCount; p++)
                for (int c = 0; c < nc; c++) for (int s = 0; s < ns; s++) { var j = _jobs[_jobCount++]; j.plan = _plans[p]; j.candidate = c; j.sample = s; }
            _done.Reset(_workers.Length); _busy = true; _sw.Restart();
            foreach (var w in _workers) w.go.Set();
            return true;
        }

        void Fill(Side s, PolicyRunner r, MlpNet net, MlpNet value)
        {
            s.runner = r; s.net = net; s.value = value != null && value.OutputDim == 1 ? value : null; s.obsDim = r.policyObsDim; s.map = r.Map;
            s.opp = r.opponent != null ? r.opponent.Map : null; s.opp2 = r.opponent2 != null ? r.opponent2.Map : s.opp;
            s.goal = r.goal; s.stop = r.goalStopDist; s.vmax = r.goalVmax; s.shove = r.shoveStyle; s.tactic = r.tactic; s.phase = r.GaitPhase;
            s.keyZ = r.Spec != null ? r.Spec.key_root_z : 0.75f;
            Array.Copy(r.StyleOffsets, s.offsets, NJ); Array.Copy(r.LastAction, s.last, NJ); Array.Copy(r.FilteredAction, s.actF, NJ);
        }

        void Apply()
        {
            LastBatchMs = _sw.Elapsed.TotalMilliseconds;
            for (int p = 0; p < _planCount; p++)
            {
                var plan = _plans[p]; var r = plan.me.runner;
                if (r == null || r.tactic == Tactic.Lurk || director.IsOut(plan.seat)) continue;
                int ns = Mathf.Clamp(samples, 1, MaxSamples);
                for (int c = 0; c < Candidates.Length; c++) { float sum = 0; for (int s = 0; s < ns; s++) sum += plan.sampled[c, s]; plan.score[c] = sum / ns; }
                plan.score[0] += pressBonus;
                int cur = Array.IndexOf(Candidates, r.tactic), best = 0, n = Mathf.Clamp(candidateCount, 1, Candidates.Length);
                for (int c = 1; c < n; c++) if (plan.score[c] > plan.score[best]) best = c;
                if (cur < 0 || cur >= n || plan.score[best] > plan.score[cur] + switchMargin) r.tactic = Candidates[best];
                Chosen[Array.IndexOf(Candidates, r.tactic)]++;
            }
            Decisions++;
        }

        /// <summary>The scores of the last decision for one fighter (Candidates order), or null.</summary>
        public float[] ScoresOf(PolicyRunner r) { for (int p = 0; p < _planCount; p++) if (_plans[p].me.runner == r) return _plans[p].score; return null; }

        // ------------------------------------------------------------------ worker side
        void Loop(Worker w)
        {
            while (true)
            {
                w.go.WaitOne(); if (!_run) return;
                for (int j = w.index; j < _jobCount; j += _workers.Length)
                {
                    try { Rollout(_jobs[j], w); } catch (Exception) { _jobs[j].plan.score[_jobs[j].candidate] = float.NegativeInfinity; }
                }
                _done.Signal();
            }
        }

        void Rollout(Job job, Worker w)
        {
            var plan = job.plan; var me = plan.me; var them = plan.them; var d = w.d;
            MujocoLib.mj_copyData(d, _m, _snap);
            if (job.sample > 0)
            {
                // a different, repeatable nudge per sample: both pelvises' horizontal velocity +- sampleNoise
                uint h = (uint)(job.sample * 73856093) ^ (uint)(job.candidate * 19349663) ^ (uint)(Decisions * 83492791);
                float U() { h ^= h << 13; h ^= h >> 17; h ^= h << 5; return (h & 0xFFFFFF) / (float)0xFFFFFF * 2f - 1f; }
                double* vm = d->qvel + me.map.RootDofAdr; vm[0] += sampleNoise * U(); vm[1] += sampleNoise * U();
                double* vt = d->qvel + them.map.RootDofAdr; vt[0] += sampleNoise * U(); vt[1] += sampleNoise * U();
            }
            Array.Copy(me.last, job.meLast, NJ); Array.Copy(me.actF, job.meF, NJ); Array.Copy(them.last, job.themLast, NJ); Array.Copy(them.actF, job.themF, NJ);
            for (int k = 0; k < NJ; k++) { job.meCtrl[k] = _held[Array.IndexOf(_all, me.runner)][k]; job.themCtrl[k] = _held[Array.IndexOf(_all, them.runner)][k]; }
            float mePhase = me.phase, themPhase = them.phase; var tactic = Candidates[job.candidate];
            int ticks = Mathf.Max(1, Mathf.RoundToInt(horizon / _ctrlDt)); float outcome = 0;
            for (int t = 0; t < ticks; t++)
            {
                Act(me, tactic, d, w, job.meLast, job.meF, ref mePhase, job.meCtrl);
                if (plan.themActive) Act(them, them.tactic, d, w, job.themLast, job.themF, ref themPhase, job.themCtrl);
                for (int s = 0; s < _decimation; s++)
                {
                    for (int i = 0; i < _all.Length; i++)
                    {
                        var map = _all[i].Map; var src = _all[i] == me.runner ? job.meCtrl : _all[i] == them.runner ? job.themCtrl : _held[i];
                        for (int k = 0; k < NJ; k++) d->ctrl[map.ActId[k]] = src[k];
                    }
                    MujocoLib.mj_step(_m, d);
                }
                bool meOut = IsOut(d, me.map), themOut = IsOut(d, them.map);
                if (meOut || themOut) { outcome = meOut && !themOut ? -20f : themOut && !meOut ? 20f : 0f; break; }
            }
            plan.sampled[job.candidate, job.sample] = outcome + (me.value != null ? Critic(me, d, w, job.meLast, mePhase) : Physics(d, me) - Physics(d, them));
        }

        // Mirror of PolicyRunner.PolicyTick for one fighter in a rollout (goal law with a tactic, observation, network,
        // clamp, shove style path, ctrl targets, gait clock).
        void Act(Side s, Tactic tactic, MujocoLib.mjData_* d, Worker w, float[] last, float[] actF, ref float phase, double[] ctrl)
        {
            var map = s.map; double* q = d->qpos + map.RootQposAdr;
            w.cmd[0] = w.cmd[1] = w.cmd[2] = 0;
            if (s.goal != GoalMode.None)
            {
                double gx = -q[0], gy = -q[1]; float stop = s.stop;
                if (s.goal == GoalMode.Opponent && s.opp != null) Tactics.Goal(tactic, q, d->qpos + s.opp.RootQposAdr, s.stop, out gx, out gy, out stop);
                GoalCommand.Compute(q + 3, gx, gy, stop, s.vmax, w.cmd);
            }
            ObsBuilder.Fill(w.obs, d, map, w.cmd, last, phase);
            if (s.obsDim > ObsBuilder.ObsDim && s.opp != null)
            {
                ObsBuilder.FillCombat(w.obs, ObsBuilder.ObsDim, d, map, s.opp);
                if (s.obsDim >= ObsBuilder.ObsDim + 18) ObsBuilder.FillCombat(w.obs, ObsBuilder.ObsDim + 9, d, map, s.opp2 ?? s.opp);
            }
            s.net.Forward(w.obs, w.act, w.scratch);
            float blend = 0f;
            if (s.shove && s.opp != null)
            {
                double* o = d->qpos + s.opp.RootQposAdr; double dist = Math.Sqrt((o[0] - q[0]) * (o[0] - q[0]) + (o[1] - q[1]) * (o[1] - q[1]));
                blend = Mathf.Clamp01((PolicyRunner.ShoveFar - (float)dist) / (PolicyRunner.ShoveFar - PolicyRunner.ShoveNear));
            }
            for (int i = 0; i < map.N; i++)
            {
                float a = Mathf.Clamp(w.act[i], -1f, 1f); last[i] = a;
                float f = s.shove ? PolicyRunner.ShoveSmooth * actF[i] + (1f - PolicyRunner.ShoveSmooth) * a : a; actF[i] = f;
                ctrl[i] = Math.Clamp(map.DefaultPose[i] + blend * s.offsets[i] + _scale * f, map.CtrlMin[i], map.CtrlMax[i]);
            }
            phase += 2f * Mathf.PI * ObsBuilder.GaitFreqHz * _ctrlDt; if (phase > 2f * Mathf.PI) phase -= 2f * Mathf.PI;
        }

        bool IsOut(MujocoLib.mjData_* d, JointMap map)
        {
            double* q = d->qpos + map.RootQposAdr; double rad = Math.Sqrt(q[0] * q[0] + q[1] * q[1]);
            return rad > _outRadius || ObsBuilder.Up(q + 3) < _arena.fall_up || q[2] - _arena.Ground(rad) < _arena.fall_height;
        }

        // the same estimate as WinOdds' physics source, with the body's own standing height
        float Physics(MujocoLib.mjData_* d, Side s)
        {
            double* q = d->qpos + s.map.RootQposAdr; double* v = d->qvel + s.map.RootDofAdr;
            double rad = Math.Sqrt(q[0] * q[0] + q[1] * q[1]), margin = _outRadius - rad, outward = rad > 1e-3 ? (q[0] * v[0] + q[1] * v[1]) / rad : 0;
            return (float)(3.0 * margin + 2.0 * (ObsBuilder.Up(q + 3) - 0.8) + 2.5 * (q[2] - _arena.Ground(rad) - 0.87 * s.keyZ) - 1.5 * Math.Max(0, outward));
        }

        // Critic input, as WinOdds.Critic / KothEnv._observe(): clean obs (103) | world linvel (3) | height (1) | feet (2) |
        // mass, friction, kp scale at nominal (3) | combat block(s).
        float Critic(Side s, MujocoLib.mjData_* d, Worker w, float[] last, float phase)
        {
            var map = s.map; int dim = s.value.InputDim; var b = w.obs;
            double* q = d->qpos + map.RootQposAdr; double* v = d->qvel + map.RootDofAdr;
            double gx = -q[0], gy = -q[1]; float stop = s.stop;
            if (s.opp != null) Tactics.Goal(Tactic.Press, q, d->qpos + s.opp.RootQposAdr, s.stop, out gx, out gy, out stop);
            GoalCommand.Compute(q + 3, gx, gy, stop, s.vmax, w.cmd);
            ObsBuilder.Fill(b, d, map, w.cmd, last, phase);
            int o = ObsBuilder.ObsDim; double rad = Math.Sqrt(q[0] * q[0] + q[1] * q[1]);
            b[o] = (float)v[0]; b[o + 1] = (float)v[1]; b[o + 2] = (float)v[2]; b[o + 3] = (float)(q[2] - _arena.Ground(rad));
            int idx = Array.IndexOf(_all, s.runner); b[o + 4] = b[o + 5] = 0;
            for (int c = 0; c < d->ncon; c++)
            {
                var con = d->contact + c; int b1 = _m->geom_bodyid[con->geom1], b2 = _m->geom_bodyid[con->geom2];
                if (b1 == _footL[idx] || b2 == _footL[idx]) b[o + 4] = 1; if (b1 == _footR[idx] || b2 == _footR[idx]) b[o + 5] = 1;
            }
            b[o + 6] = 1f; b[o + 7] = 0.6f; b[o + 8] = 1f;
            if (dim > o + 9 && s.opp != null) { ObsBuilder.FillCombat(b, o + 9, d, map, s.opp); if (dim > o + 18) ObsBuilder.FillCombat(b, o + 18, d, map, s.opp2 ?? s.opp); }
            s.value.Forward(b, w.val, w.scratch);
            return w.val[0] / 6f;          // WinOdds' critic temperature: comparable to the physics estimate's spread
        }
    }
}
