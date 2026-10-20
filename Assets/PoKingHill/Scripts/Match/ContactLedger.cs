using System;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Who pushed whom. Every physics step, every contact between two different fighters' geoms adds its normal impulse
    /// (mj_contactForce x dt) to the pusher's account against the other: the pusher is the one whose pelvis moves faster
    /// toward the other along the line between them. Accounts decay (1.5 s), so they say who pushed recently.
    /// When a fighter is knocked out, the one with the largest recent push on it (within creditSeconds, at least
    /// minImpulse) gets the knock-out and the runner-up the assist (MatchEvents.KnockoutCredited); with no such push
    /// it fell on its own. Grudge and Vulture targeting read it too. Read-only on mjData; allocation-free per step.
    /// </summary>
    public unsafe class ContactLedger : MonoBehaviour
    {
        public DemoDirector director;
        [Tooltip("A push older than this does not count for a knock-out")] public float creditSeconds = 3f;
        [Tooltip("Smallest decayed impulse (N s) that counts as a push")] public float minImpulse = 4f;
        public float decaySeconds = 1.5f;

        PolicyRunner[] _runners = Array.Empty<PolicyRunner>(); int[] _robotOfBody = Array.Empty<int>();
        double[,] _recent = new double[0, 0], _lastAt = new double[0, 0];      // [victim, pusher]
        double[] _lastContact = Array.Empty<double>(); float[] _pushed = Array.Empty<float>(); int[] _kos = Array.Empty<int>(), _assists = Array.Empty<int>();
        bool _ready;

        void OnEnable()
        {
            MatchEvents.RoundStarted += OnRound; MatchEvents.Knockout += OnKnockout;
            if (MjScene.InstanceExists) MjScene.Instance.postUpdateEvent += Post;
        }
        void OnDisable()
        {
            MatchEvents.RoundStarted -= OnRound; MatchEvents.Knockout -= OnKnockout;
            if (MjScene.InstanceExists) MjScene.Instance.postUpdateEvent -= Post;
        }

        void Init()
        {
            var m = MjScene.Instance.Model;
            _runners = (PolicyRunner[])director.fighters.Clone();       // fixed body order: the director reorders its own array
            int n = _runners.Length;
            _robotOfBody = new int[(int)m->nbody];
            for (int b = 0; b < _robotOfBody.Length; b++)
            {
                _robotOfBody[b] = -1;
                string name = System.Runtime.InteropServices.Marshal.PtrToStringAnsi((IntPtr)MujocoLib.mj_id2name(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, b)) ?? "";
                for (int k = 0; k < n; k++) if (name.StartsWith(_runners[k].robotPrefix)) { _robotOfBody[b] = k; break; }
            }
            _recent = new double[n, n]; _lastAt = new double[n, n]; _lastContact = new double[n]; _pushed = new float[n]; _kos = new int[n]; _assists = new int[n];
            _ready = true; OnRound();
        }

        int Index(PolicyRunner r) => r == null ? -1 : Array.IndexOf(_runners, r);

        void OnRound()
        {
            if (!_ready) return;
            Array.Clear(_recent, 0, _recent.Length); Array.Clear(_pushed, 0, _pushed.Length); Array.Clear(_kos, 0, _kos.Length); Array.Clear(_assists, 0, _assists.Length);
            for (int i = 0; i < _lastContact.Length; i++) _lastContact[i] = -1e9;
            for (int v = 0; v < _runners.Length; v++) for (int p = 0; p < _runners.Length; p++) _lastAt[v, p] = -1e9;
        }

        void Post(object s, MjStepArgs a)
        {
            if (director == null || director.statsRounds > 0) return;
            if (!_ready) { if (director.State >= MatchState.Menu && director.fighters.Length > 0 && director.fighters[0].Map != null) Init(); return; }
            if (director.State != MatchState.Fight) return;
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data; double dt = m->opt.timestep, now = d->time;
            double k = Math.Exp(-dt / decaySeconds); int n = _runners.Length;
            for (int v = 0; v < n; v++) for (int p = 0; p < n; p++) _recent[v, p] *= k;
            double* f = stackalloc double[6];
            for (int c = 0; c < d->ncon; c++)
            {
                var con = d->contact + c;
                int i = _robotOfBody[m->geom_bodyid[con->geom1]], j = _robotOfBody[m->geom_bodyid[con->geom2]];
                if (i < 0 || j < 0 || i == j || _runners[i].paused || _runners[j].paused) continue;
                MujocoLib.mj_contactForce(m, d, c, f);
                double imp = Math.Abs(f[0]) * dt; if (imp <= 0) continue;
                double* qi = d->qpos + _runners[i].Map.RootQposAdr; double* qj = d->qpos + _runners[j].Map.RootQposAdr;
                double* vi = d->qvel + _runners[i].Map.RootDofAdr; double* vj = d->qvel + _runners[j].Map.RootDofAdr;
                double ux = qj[0] - qi[0], uy = qj[1] - qi[1], len = Math.Sqrt(ux * ux + uy * uy) + 1e-6; ux /= len; uy /= len;
                double iToward = vi[0] * ux + vi[1] * uy, jToward = -(vj[0] * ux + vj[1] * uy);
                int pusher = iToward >= jToward ? i : j, victim = pusher == i ? j : i;
                _recent[victim, pusher] += imp; _lastAt[victim, pusher] = now; _pushed[pusher] += (float)imp;
                _lastContact[i] = _lastContact[j] = now;
            }
        }

        void OnKnockout(PolicyRunner victim)
        {
            if (!_ready || director == null || director.statsRounds > 0) return;
            int v = Index(victim); if (v < 0) return;
            double now = MjScene.Instance.Data->time; int best = -1, second = -1;
            for (int p = 0; p < _runners.Length; p++)
            {
                if (p == v || now - _lastAt[v, p] > creditSeconds || _recent[v, p] < minImpulse) continue;
                if (best < 0 || _recent[v, p] > _recent[v, best]) { second = best; best = p; }
                else if (second < 0 || _recent[v, p] > _recent[v, second]) second = p;
            }
            if (best >= 0) _kos[best]++;
            if (second >= 0) _assists[second]++;
            MatchEvents.RaiseKnockoutCredited(victim, best >= 0 ? _runners[best] : null, second >= 0 ? _runners[second] : null);
        }

        // ------------------------------------------------------------------ queries (fighters by runner, any seat order)
        /// <summary>Knock-outs credited to this fighter this round.</summary>
        public int Knockouts(PolicyRunner r) { int i = Index(r); return i >= 0 ? _kos[i] : 0; }
        public int Assists(PolicyRunner r) { int i = Index(r); return i >= 0 ? _assists[i] : 0; }
        /// <summary>Total pushing impulse this fighter delivered this round (N s).</summary>
        public float Pushed(PolicyRunner r) { int i = Index(r); return i >= 0 ? _pushed[i] : 0; }

        /// <summary>The fighter that pushed <paramref name="victim"/> most recently within <paramref name="seconds"/>, or null.</summary>
        public PolicyRunner LastPusher(PolicyRunner victim, double seconds)
        {
            int v = Index(victim); if (v < 0 || !_ready) return null;
            double now = MjScene.Instance.Data->time, latest = now - seconds; int who = -1;
            for (int p = 0; p < _runners.Length; p++) if (p != v && _lastAt[v, p] > latest && _recent[v, p] >= minImpulse * 0.25) { latest = _lastAt[v, p]; who = p; }
            return who >= 0 ? _runners[who] : null;
        }

        /// <summary>Touching any other fighter within the last <paramref name="seconds"/>.</summary>
        public bool Engaged(PolicyRunner r, double seconds) { int i = Index(r); return i >= 0 && _ready && MjScene.Instance.Data->time - _lastContact[i] < seconds; }

        /// <summary>a and b pushed each other within the last <paramref name="seconds"/>.</summary>
        public bool Fighting(PolicyRunner a, PolicyRunner b, double seconds)
        {
            int i = Index(a), j = Index(b); if (i < 0 || j < 0 || !_ready) return false;
            double since = MjScene.Instance.Data->time - seconds;
            return _lastAt[i, j] > since || _lastAt[j, i] > since;
        }
    }
}
