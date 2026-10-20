using System;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Live balance and strain for every fighter, 10 times a second, read-only on mjData:
    ///   stability  where the capture point (centre of mass + velocity / omega, omega = sqrt(g / CoM height)) lies
    ///              relative to the feet on the ground (a segment between two planted feet, or the one planted foot):
    ///              1 within a step's reach of it (stepReach), 0 at 0.35 m beyond, or no foot down. It drops before a fall shows.
    ///   rim        distance from the capture point to the "out" radius: below 0 while heading out, the fighter is
    ///              going over unless something stops it (Doomed).
    ///   strain     mean |actuator force| / force limit over its joints, with a 3 s history for the HUD sparkline.
    /// Vulture targeting reads stability to pick the weakest fighter.
    /// </summary>
    public unsafe class Biometrics : MonoBehaviour
    {
        public DemoDirector director;
        public float hz = 10f;
        [Tooltip("Capture-point distance from the support that one step can still catch (m): counts as full balance")] public float stepReach = 0.25f;
        public const int History = 30;

        public class Vitals
        {
            public float stability = 1f, rim = 1f, strain, upright = 1f;
            public bool doomed;
            public readonly float[] strainHistory = new float[History]; public int head;
        }

        PolicyRunner[] _runners = Array.Empty<PolicyRunner>(); Vitals[] _v = Array.Empty<Vitals>();
        int[][] _acts = Array.Empty<int[]>(); int[] _left = Array.Empty<int>(), _right = Array.Empty<int>();
        bool[] _lDown = Array.Empty<bool>(), _rDown = Array.Empty<bool>(); int[] _robotOfBody = Array.Empty<int>();
        float _next; bool _ready;

        public Vitals Of(PolicyRunner r) { int i = Array.IndexOf(_runners, r); return i >= 0 ? _v[i] : null; }

        void Init()
        {
            var m = MjScene.Instance.Model; _runners = (PolicyRunner[])director.fighters.Clone(); int n = _runners.Length;
            _v = new Vitals[n]; _acts = new int[n][]; _left = new int[n]; _right = new int[n]; _lDown = new bool[n]; _rDown = new bool[n];
            for (int k = 0; k < n; k++)
            {
                var r = _runners[k]; _v[k] = new Vitals(); _acts[k] = r.Map.ActId;
                _left[k] = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, r.robotPrefix + "left_ankle_roll_link");
                _right[k] = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, r.robotPrefix + "right_ankle_roll_link");
            }
            _robotOfBody = new int[(int)m->nbody];
            for (int b = 0; b < _robotOfBody.Length; b++) { _robotOfBody[b] = -1; for (int k = 0; k < n; k++) if (b == _left[k] || b == _right[k]) _robotOfBody[b] = k; }
            _ready = true;
        }

        void Update()
        {
            if (director == null || !MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            if (!_ready) { if (director.State >= MatchState.Menu && director.fighters.Length > 0 && director.fighters[0].Map != null) Init(); return; }
            if ((director.State != MatchState.Fight && director.State != MatchState.Intro) || Time.unscaledTime < _next) return;
            _next = Time.unscaledTime + 1f / hz;
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data; double g = -m->opt.gravity[2];
            Array.Clear(_lDown, 0, _lDown.Length); Array.Clear(_rDown, 0, _rDown.Length);
            for (int c = 0; c < d->ncon; c++)
            {
                var con = d->contact + c; int b1 = m->geom_bodyid[con->geom1], b2 = m->geom_bodyid[con->geom2];
                // a foot touching the world body (hill, floor) is planted
                if (b2 == 0 && _robotOfBody[b1] >= 0) Mark(b1); else if (b1 == 0 && _robotOfBody[b2] >= 0) Mark(b2);
            }
            var arena = director.fighters.Length > 0 ? director.fighters[0].Spec?.arena ?? new ArenaSpec() : new ArenaSpec();
            for (int k = 0; k < _runners.Length; k++)
            {
                var r = _runners[k]; var v = _v[k]; if (r.Map == null) continue;
                int pel = r.Map.PelvisBodyId; double* com = d->subtree_com + 3 * pel; double* qv = d->qvel + r.Map.RootDofAdr;
                double rad = Math.Sqrt(com[0] * com[0] + com[1] * com[1]), h = Math.Max(0.2, com[2] - arena.Ground(rad)), w = Math.Sqrt(g / h);
                double cx = com[0] + qv[0] / w, cy = com[1] + qv[1] / w;
                double dist;
                double* L = d->xpos + 3 * _left[k]; double* R = d->xpos + 3 * _right[k];
                if (_lDown[k] && _rDown[k]) dist = SegmentDistance(cx, cy, L[0], L[1], R[0], R[1]);
                else if (_lDown[k]) dist = Math.Sqrt((cx - L[0]) * (cx - L[0]) + (cy - L[1]) * (cy - L[1]));
                else if (_rDown[k]) dist = Math.Sqrt((cx - R[0]) * (cx - R[0]) + (cy - R[1]) * (cy - R[1]));
                else dist = stepReach + 0.35;                              // airborne: no support at all
                // walking is a controlled fall: at 1 m/s the capture point runs ~0.27 m ahead of the stance foot, which the
                // next step catches. So a step's reach (stepReach) is still full balance; the bar empties over the 0.35 m beyond.
                float stab = Mathf.Clamp01(1f - (float)Math.Max(0, dist - stepReach) / 0.35f);
                v.stability = Mathf.Lerp(v.stability, stab, 0.5f);
                double cr = Math.Sqrt(cx * cx + cy * cy), outward = rad > 1e-3 ? (com[0] * qv[0] + com[1] * qv[1]) / rad : 0;
                v.rim = (float)(director.OutRadiusNow - cr);
                v.doomed = v.rim < 0 && outward > 0.15 && !director.IsOut(Array.IndexOf(director.fighters, r));
                v.upright = (float)ObsBuilder.Up(d->qpos + r.Map.RootQposAdr + 3);
                double load = 0; int[] acts = _acts[k];
                for (int i = 0; i < acts.Length; i++)
                {
                    int a = acts[i]; double lim = Math.Max(Math.Abs(m->actuator_forcerange[2 * a]), Math.Abs(m->actuator_forcerange[2 * a + 1]));
                    if (m->actuator_forcelimited[a] == 0 || lim < 1e-6) lim = 100;
                    load += Math.Min(1.0, Math.Abs(d->actuator_force[a]) / lim);
                }
                v.strain = (float)(load / Math.Max(1, acts.Length));
                v.strainHistory[v.head] = v.strain; v.head = (v.head + 1) % History;
            }
        }

        void Mark(int body)
        {
            int k = _robotOfBody[body];
            if (body == _left[k]) _lDown[k] = true; else _rDown[k] = true;
        }

        /// <summary>Distance from point p to the segment ab (2D).</summary>
        public static double SegmentDistance(double px, double py, double ax, double ay, double bx, double by)
        {
            double ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey;
            double t = l2 > 1e-12 ? Math.Clamp(((px - ax) * ex + (py - ay) * ey) / l2, 0, 1) : 0;
            double dx = px - (ax + t * ex), dy = py - (ay + t * ey);
            return Math.Sqrt(dx * dx + dy * dy);
        }
    }
}
