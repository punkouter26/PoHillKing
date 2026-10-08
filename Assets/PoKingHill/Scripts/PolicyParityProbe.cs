using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Early verification gate for a trained rung, against &lt;rung&gt;_reference_trajectory.json (recorded in CPU MuJoCo
    /// with the same ONNX by training/scripts/record_reference.py):
    ///   1. Replay: feed every recorded observation to the in-engine policy; max |action - recorded| must be &lt; 1e-4.
    ///   2. Closed loop: run the policy in Unity from the keyframe and compare, tick by tick, the observation, the
    ///      action, the pelvis position and the ctrl targets with the reference run.
    /// Writes unity_policy_parity_&lt;rung&gt;.json under training/assets/g1/ and exits the editor when -kothExit is passed.
    /// </summary>
    public unsafe class PolicyParityProbe : MonoBehaviour
    {
        public TextAsset referenceJson;
        public ModelAsset policy;
        public PolicyRunner runner;
        public string rung = "r0";

        List<object> _frames;
        float _replayMaxErr = -1;
        int _tick;
        float _obsErrMax, _actErrMax; double _posErrMax, _ctrlRelSum; double _minUp = 1, _minZ = 10; double _sumCtrlRef, _sumCtrlDiff;
        readonly StringBuilder _rows = new();
        bool _done;

        static float F(object o) => System.Convert.ToSingle(o, CultureInfo.InvariantCulture);
        static double D(object o) => System.Convert.ToDouble(o, CultureInfo.InvariantCulture);
        static List<object> L(object o) => (List<object>)o;
        static string S(double v) => v.ToString("0.#######", CultureInfo.InvariantCulture);

        void OnEnable()
        {
            var root = (Dictionary<string, object>)MiniJson.Parse(referenceJson.text);
            _frames = L(root["frames"]);
            var cmd = L(root["command"]);
            runner.command = new Vector3(F(cmd[0]), F(cmd[1]), F(cmd[2]));
            if (root.TryGetValue("goal", out var g) && g is string gs && gs != "none")
            {
                runner.goal = gs == "center" ? GoalMode.Center : GoalMode.Opponent;
                runner.goalStopDist = F(root["goal_stop_dist"]); runner.goalVmax = F(root["goal_vmax"]);
            }
            Replay();
            runner.OnPolicyStep += OnTick;
            MjScene.Instance.postInitEvent += ApplyInitialState;     // runs after the PolicyRunner handler (execution order -100)
        }
        void OnDisable()
        {
            if (runner != null) runner.OnPolicyStep -= OnTick;
            if (MjScene.InstanceExists) MjScene.Instance.postInitEvent -= ApplyInitialState;
        }

        // Start from the first frame of the reference run (root pose, joint angles, velocities) instead of the keyframe.
        void ApplyInitialState(object s, MjStepArgs a)
        {
            var d = MjScene.Instance.Data; var f = (Dictionary<string, object>)_frames[0]; var map = runner.Map;
            var p = L(f["root_pos"]); var q = L(f["root_quat"]); var jp = L(f["joint_pos"]); var jv = L(f["joint_vel"]);
            var lv = L(f["root_linvel"]); var av = L(f["root_angvel"]);
            for (int k = 0; k < 3; k++) { d->qpos[map.RootQposAdr + k] = D(p[k]); d->qvel[map.RootDofAdr + k] = D(lv[k]); d->qvel[map.RootDofAdr + 3 + k] = D(av[k]); }
            for (int k = 0; k < 4; k++) d->qpos[map.RootQposAdr + 3 + k] = D(q[k]);
            for (int i = 0; i < map.N; i++) { d->qpos[map.QposAdr[i]] = D(jp[i]); d->qvel[map.DofAdr[i]] = D(jv[i]); }
        }

        void Replay()
        {
            using var worker = new Worker(ModelLoader.Load(policy), BackendType.CPU);
            using var input = new Tensor<float>(new TensorShape(1, ObsBuilder.ObsDim));
            var obs = new float[ObsBuilder.ObsDim]; float max = 0;
            foreach (Dictionary<string, object> f in _frames)
            {
                var o = L(f["obs"]); var a = L(f["action"]);
                for (int i = 0; i < obs.Length; i++) obs[i] = F(o[i]);
                input.Upload(obs); worker.Schedule(input);
                using var outT = (worker.PeekOutput() as Tensor<float>).ReadbackAndClone();
                var act = outT.DownloadToArray();
                for (int i = 0; i < act.Length; i++) max = Mathf.Max(max, Mathf.Abs(act[i] - F(a[i])));
            }
            _replayMaxErr = max;
            Debug.Log($"[PolicyParity] replay over {_frames.Count} frames: max |action error| = {max:E3} ({(max < 1e-4f ? "PASS" : "FAIL")})");
        }

        // Called by PolicyRunner right after it computed obs/action for tick k (state is still the pre-step state).
        void OnTick(PolicyRunner r)
        {
            if (_done || _tick >= _frames.Count) return;
            var f = (Dictionary<string, object>)_frames[_tick];
            var d = MjScene.Instance.Data;
            var o = L(f["obs"]); var a = L(f["action"]); var p = L(f["root_pos"]); var c = L(f["ctrl"]);
            float oe = 0, ae = 0;
            for (int i = 0; i < ObsBuilder.ObsDim; i++) oe = Mathf.Max(oe, Mathf.Abs(r.Obs[i] - F(o[i])));
            for (int i = 0; i < r.Map.N; i++)
            {
                ae = Mathf.Max(ae, Mathf.Abs(r.RawAction[i] - F(a[i])));
                _sumCtrlRef += System.Math.Abs(D(c[i])); _sumCtrlDiff += System.Math.Abs(r.Ctrl[i] - D(c[i]));
            }
            double* q = d->qpos + r.Map.RootQposAdr;
            double pe = System.Math.Sqrt((q[0] - D(p[0])) * (q[0] - D(p[0])) + (q[1] - D(p[1])) * (q[1] - D(p[1])) + (q[2] - D(p[2])) * (q[2] - D(p[2])));
            double up = 1 - 2 * (q[4] * q[4] + q[5] * q[5]);
            _obsErrMax = Mathf.Max(_obsErrMax, oe); _actErrMax = Mathf.Max(_actErrMax, ae); _posErrMax = System.Math.Max(_posErrMax, pe);
            _minUp = System.Math.Min(_minUp, up); _minZ = System.Math.Min(_minZ, q[2]);
            if (_tick == 0 || _tick == 1 || _tick == 10 || _tick % 50 == 0 || _tick == _frames.Count - 1)
                _rows.Append($"{(_rows.Length > 0 ? ",\n  " : "")}{{\"tick\": {_tick}, \"obs_err\": {S(oe)}, \"action_err\": {S(ae)}, \"pelvis_pos_err_m\": {S(pe)}, \"pelvis_z\": {S(q[2])}, \"ref_z\": {S(D(p[2]))}, \"up\": {S(up)}}}");
            _tick++;
            if (_tick == _frames.Count) Finish(r);
        }

        void Finish(PolicyRunner r)
        {
            _done = true;
            double ctrlRel = _sumCtrlDiff / System.Math.Max(1e-9, _sumCtrlRef);
            bool replayOk = _replayMaxErr >= 0 && _replayMaxErr < 1e-4f;
            bool loopOk = _minUp > 0.8 && _minZ > 0.5 && ctrlRel < 0.10 && _posErrMax < 0.10;
            var sb = new StringBuilder("{\n");
            sb.Append($" \"rung\": \"{rung}\", \"frames\": {_frames.Count},\n");
            sb.Append($" \"replay_max_action_err\": {S(_replayMaxErr)}, \"replay_pass\": {(replayOk ? "true" : "false")},\n");
            sb.Append($" \"closed_loop\": {{\"max_obs_err\": {S(_obsErrMax)}, \"max_action_err\": {S(_actErrMax)}, \"max_pelvis_pos_err_m\": {S(_posErrMax)}, " +
                      $"\"mean_ctrl_rel_diff\": {S(ctrlRel)}, \"min_upright\": {S(_minUp)}, \"min_pelvis_z\": {S(_minZ)}, \"pass\": {(loopOk ? "true" : "false")}}},\n");
            sb.Append($" \"inference_ms_last\": {S(r.LastInferenceMs)},\n");
            sb.Append(" \"samples\": [\n  " + _rows + "\n ]\n}\n");
            string path = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "training", "assets", "g1", $"unity_policy_parity_{rung}.json"));
            File.WriteAllText(path, sb.ToString());
            Debug.Log($"[PolicyParity] wrote {path}\n{sb}");
#if UNITY_EDITOR
            if (Application.isBatchMode || System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "-kothExit") >= 0) UnityEditor.EditorApplication.Exit(replayOk && loopOk ? 0 : 4);
#endif
        }
    }
}
