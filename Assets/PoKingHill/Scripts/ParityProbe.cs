using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Zero-brain parity probe (gates B.4 / B.10 / B.12). Records each robot's pelvis height at fixed sim times,
    /// the model-dump result, and mj_step cost, then writes them to JSON next to the training reference trace.
    /// In batch mode it quits the editor when done. Optionally fires one box at each robot after the hold window
    /// to prove the impulse path (pool -> qpos/qvel -> contact) works.
    /// </summary>
    public unsafe class ParityProbe : MonoBehaviour
    {
        public string outputFile = "unity_hold_trace.json";   // written under training/assets/g1/
        public float[] sampleTimes = { 0.5f, 1f, 3f, 5f, 10f };
        public bool fireBoxesAfterHold = true;
        public float boxSpeed = 8f;

        PolicyRunner[] _runners; ProjectilePool _pool; ModelDumpCheck _check;
        readonly List<string> _rows = new();
        int _next; bool _fired; double _fireTime; bool _done;
        readonly System.Diagnostics.Stopwatch _sw = new(); double _stepMsSum; long _steps;

        void OnEnable()
        {
            _runners = FindObjectsByType<PolicyRunner>(FindObjectsInactive.Exclude);
            _pool = FindAnyObjectByType<ProjectilePool>(); _check = FindAnyObjectByType<ModelDumpCheck>();
            MjScene.Instance.preUpdateEvent += Pre; MjScene.Instance.postUpdateEvent += Post;
        }
        void OnDisable() { if (MjScene.InstanceExists) { MjScene.Instance.preUpdateEvent -= Pre; MjScene.Instance.postUpdateEvent -= Post; } }

        void Pre(object s, MjStepArgs a) => _sw.Restart();

        void Post(object s, MjStepArgs a)
        {
            _stepMsSum += _sw.Elapsed.TotalMilliseconds; _steps++;
            if (_done) return;
            var d = MjScene.Instance.Data; double t = d->time;
            if (_next < sampleTimes.Length && t >= sampleTimes[_next] - 1e-9)
            {
                var sb = new StringBuilder($"{{\"t\": {F(t)}");
                foreach (var r in _runners) sb.Append($", \"{r.robotPrefix}z\": {F(d->qpos[r.Map.RootQposAdr + 2])}, \"{r.robotPrefix}up\": {F(Up(d, r))}");
                sb.Append("}"); _rows.Add(sb.ToString()); _next++;
            }
            if (_next < sampleTimes.Length) return;
            if (fireBoxesAfterHold && _pool != null && !_fired)
            {
                foreach (var r in _runners)
                {
                    double* q = d->qpos + r.Map.RootQposAdr;
                    _pool.FireAt(new Vector3((float)q[0], (float)q[1], (float)q[2] + 0.1f), boxSpeed);
                }
                _fired = true; _fireTime = t; return;
            }
            if (fireBoxesAfterHold && _pool != null && t < _fireTime + 3.0) return;
            Finish(d, t);
        }

        static double Up(MujocoLib.mjData_* d, PolicyRunner r)
        {
            double* q = d->qpos + r.Map.RootQposAdr + 3;      // w x y z
            return 1 - 2 * (q[1] * q[1] + q[2] * q[2]);       // world-z component of the pelvis z axis
        }
        static string F(double v) => v.ToString("0.#####", CultureInfo.InvariantCulture);

        void Finish(MujocoLib.mjData_* d, double t)
        {
            _done = true;
            var m = MjScene.Instance.Model;
            var sb = new StringBuilder("{\n");
            sb.Append($" \"model_check_passed\": {(_check != null && _check.Passed ? "true" : "false")},\n");
            sb.Append($" \"model_check_mismatches\": {(_check != null ? _check.Mismatches : -1)},\n");
            sb.Append($" \"nq\": {m->nq}, \"nv\": {m->nv}, \"nu\": {m->nu}, \"nbody\": {m->nbody}, \"ngeom\": {m->ngeom},\n");
            sb.Append($" \"timestep\": {F(m->opt.timestep)}, \"fixed_dt\": {F(Time.fixedDeltaTime)}, \"integrator\": {m->opt.integrator}, \"iterations\": {m->opt.iterations}, \"ls_iterations\": {m->opt.ls_iterations},\n");
            sb.Append($" \"mean_step_ms\": {F(_stepMsSum / System.Math.Max(1, _steps))}, \"steps\": {_steps}, \"sim_time\": {F(t)},\n");
            sb.Append(" \"after_box\": {");
            for (int i = 0; i < _runners.Length; i++)
                sb.Append($"{(i > 0 ? ", " : "")}\"{_runners[i].robotPrefix}z\": {F(d->qpos[_runners[i].Map.RootQposAdr + 2])}, \"{_runners[i].robotPrefix}up\": {F(Up(d, _runners[i]))}");
            sb.Append("},\n \"trace\": [\n  " + string.Join(",\n  ", _rows) + "\n ]\n}\n");
            string path = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "training", "assets", "g1", outputFile));
            File.WriteAllText(path, sb.ToString());
            Debug.Log($"ParityProbe wrote {path}\n{sb}");
            if (_check != null && !_check.Passed) Debug.LogError(_check.Report);
#if UNITY_EDITOR
            if (Application.isBatchMode || System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "-kothExit") >= 0) UnityEditor.EditorApplication.Exit(0);
#endif
        }
    }
}
