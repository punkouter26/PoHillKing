using System;
using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Unity side of the get-up gate (ParityBatch.RunGetupStats on Testbed_flat_1p; the reference is
    /// training/scripts/getup_stats.py). Drops the robot with the training recipe (pelvis 0.85 m up, uniformly random
    /// orientation, joints at default +- 0.6 rad clamped to their ranges, zero velocity), lets the get-up brain drive for
    /// `seconds`, and records when it stands (up > 0.9 and pelvis above 85 % of the standing height for 0.4 s, the
    /// game's rule). After `drops` drops it writes training/assets/g1/getup_stats_unity.json and leaves play mode (or the
    /// editor with -kothExit). Inert unless added by the gate.
    /// </summary>
    public unsafe class GetupProbe : MonoBehaviour
    {
        public PolicyRunner runner;
        public ModelAsset policy;
        public int drops = 200, seed = 1;
        public float seconds = 8f, count = 5f;
        const double DropZ = 0.85, JointNoise = 0.6, StandH = 0.85, StandUp = 0.9, Hold = 0.4;

        System.Random _rng; int _done, _stoodEver; double _t0, _since = -1, _stand = -1; bool _started;
        readonly System.Collections.Generic.List<double> _times = new();

        void OnEnable()
        {
            _rng = new System.Random(seed);
            Time.timeScale = 50f; Time.maximumDeltaTime = 2f;      // as the statistics gates: one frame may carry 2 s of physics
            MjScene.Instance.preUpdateEvent += Pre; MjScene.Instance.postUpdateEvent += Post;
        }
        void OnDisable() { if (MjScene.InstanceExists) { MjScene.Instance.preUpdateEvent -= Pre; MjScene.Instance.postUpdateEvent -= Post; } }

        void Pre(object s, MjStepArgs a)
        {
            if (_started || runner.Map == null) return;
            _started = true; runner.SetBrain(policy, ObsBuilder.ObsDim, GoalMode.None, 0f, 0f); Drop();
        }

        void Drop()
        {
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data; var map = runner.Map;
            runner.ResetPose(); runner.SetBrain(policy, ObsBuilder.ObsDim, GoalMode.None, 0f, 0f);     // fresh clock and last action
            double* q = d->qpos + map.RootQposAdr;
            double w = Gauss(), x = Gauss(), y = Gauss(), z = Gauss(), n = Math.Sqrt(w * w + x * x + y * y + z * z);
            q[2] = DropZ; q[3] = w / n; q[4] = x / n; q[5] = y / n; q[6] = z / n;
            for (int i = 0; i < map.N; i++)
            {
                int j = m->dof_jntid[map.DofAdr[i]];
                double v = map.DefaultPose[i] + (_rng.NextDouble() * 2 - 1) * JointNoise;
                d->qpos[map.QposAdr[i]] = m->jnt_limited[j] != 0 ? Math.Clamp(v, m->jnt_range[2 * j], m->jnt_range[2 * j + 1]) : v;
            }
            for (int k = 0; k < (int)m->nv; k++) d->qvel[k] = 0;
            _t0 = d->time; _since = -1; _stand = -1;
        }

        double Gauss() { double u1 = 1.0 - _rng.NextDouble(), u2 = _rng.NextDouble(); return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2); }

        void Post(object s, MjStepArgs a)
        {
            if (!_started || _done >= drops) return;
            var d = MjScene.Instance.Data; double t = d->time - _t0; double* q = d->qpos + runner.Map.RootQposAdr;
            if (ObsBuilder.Up(q + 3) > StandUp && q[2] > StandH * runner.Spec.key_root_z)
            {
                if (_since < 0) _since = t;
                if (t - _since >= Hold && _stand < 0) _stand = t;
            }
            else _since = -1;
            if (t < seconds) return;
            if (_stand >= 0) _stoodEver++;
            if (_stand >= 0 && _stand <= count) _times.Add(_stand);
            if (++_done < drops) { Drop(); return; }
            Write();
        }

        void Write()
        {
            _times.Sort();
            double Q(double f) => _times.Count == 0 ? -1 : _times[Math.Min(_times.Count - 1, (int)(f * _times.Count))];
            string F(double v) => v.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture);
            string json = $"{{\n \"drops\": {drops}, \"success\": {F((double)_times.Count / drops)}, \"stood_within_seconds\": {F((double)_stoodEver / drops)}, \"count_s\": {F(count)}, \"seconds\": {F(seconds)},\n" +
                          $" \"median_time_s\": {F(Q(0.5))}, \"p25_time_s\": {F(Q(0.25))}, \"p75_time_s\": {F(Q(0.75))}\n}}\n";
            string path = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "training", "assets", runner.body, "getup_stats_unity.json"));
            System.IO.File.WriteAllText(path, json); Debug.Log("[GetupProbe] wrote " + path + "\n" + json);
            runner.paused = true; Time.timeScale = 1f; Time.maximumDeltaTime = 1f / 3f;
#if UNITY_EDITOR
            if (Cli.Exit) UnityEditor.EditorApplication.Exit(0); else UnityEditor.EditorApplication.ExitPlaymode();
#endif
        }
    }
}
