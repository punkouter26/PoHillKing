using System;
using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Live chance of winning for every fighter still in, for the HUD bar. Two sources:
    ///   critic  when the seated agent's roster entry has a value model (scripts/export_policy.py --value): the
    ///           training critic's return estimate from the same inputs it saw in training (clean observation, world
    ///           velocity, height above the hill, foot contacts, nominal mass/friction/kp, combat block), turned into
    ///           odds with a softmax over the fighters still in;
    ///   physics otherwise: a softmax of how safe each one stands (distance to the rim, uprightness, pelvis height,
    ///           speed toward the rim). Labelled as an estimate in the HUD.
    /// Updated 10 times a second on the main thread, read-only access to mjData.
    /// </summary>
    public unsafe class WinOdds : MonoBehaviour
    {
        public DemoDirector director;
        public float criticTemperature = 6f, physicsTemperature = 2f, hz = 10f;
        public float[] Odds { get; private set; } = new float[8];
        public bool FromCritic { get; private set; }

        Worker[] _workers = new Worker[8]; ModelAsset[] _loaded = new ModelAsset[8];
        Tensor<float>[] _in = new Tensor<float>[8]; float[][] _buf = new float[8][];
        readonly float[] _score = new float[8]; float _next; int[] _footGeom; bool _footReady;

        void OnDisable() { for (int i = 0; i < _workers.Length; i++) { _workers[i]?.Dispose(); _in[i]?.Dispose(); _workers[i] = null; _in[i] = null; _loaded[i] = null; } }

        void Update()
        {
            if (director == null || director.State != MatchState.Fight || !GameSettings.WinOdds || Time.unscaledTime < _next) return;
            _next = Time.unscaledTime + 1f / hz;
            var d = MjScene.Instance.Data; int n = Mathf.Min(director.Playing, 8);
            bool allCritic = true;
            for (int i = 0; i < n; i++)
            {
                var r = director.fighters[i];
                if (director.IsOut(i)) { _score[i] = float.NegativeInfinity; continue; }
                int agent = director.AgentOf(i);
                var value = agent >= 0 && agent < director.roster.Length ? director.roster[agent].value : null;
                float v = value != null ? Critic(i, r, value, d) : float.NaN;
                if (float.IsNaN(v)) { allCritic = false; _score[i] = Physics(r, d) / physicsTemperature; }
                else _score[i] = v / criticTemperature;
            }
            FromCritic = allCritic && n > 0;
            if (!FromCritic) for (int i = 0; i < n; i++) if (!float.IsNegativeInfinity(_score[i])) _score[i] = Physics(director.fighters[i], d) / physicsTemperature;
            float max = float.NegativeInfinity; for (int i = 0; i < n; i++) max = Mathf.Max(max, _score[i]);
            float sum = 0; for (int i = 0; i < n; i++) { Odds[i] = float.IsNegativeInfinity(_score[i]) ? 0 : Mathf.Exp(_score[i] - max); sum += Odds[i]; }
            for (int i = 0; i < Odds.Length; i++) Odds[i] = i < n && sum > 0 ? Odds[i] / sum : 0;
        }

        float Physics(PolicyRunner r, MujocoLib.mjData_* d)
        {
            double* q = d->qpos + r.Map.RootQposAdr; double* v = d->qvel + r.Map.RootDofAdr;
            double rad = Math.Sqrt(q[0] * q[0] + q[1] * q[1]), margin = director.OutRadiusNow - rad;
            double outward = rad > 1e-3 ? (q[0] * v[0] + q[1] * v[1]) / rad : 0;
            return (float)(3.0 * margin + 2.0 * (ObsBuilder.Up(q + 3) - 0.8) + 2.5 * (q[2] - 0.65) - 1.5 * Math.Max(0, outward));
        }

        // Critic input, mirror of KothEnv._observe(): clean obs (103) | world linvel (3) | height (1) | feet contact (2)
        // | mass scale, friction, kp scale at their nominal values (3) | combat block (9) = 121.
        float Critic(int i, PolicyRunner r, ModelAsset value, MujocoLib.mjData_* d)
        {
            if (r.Obs == null || r.Obs.Length < ObsBuilder.ObsDim) return float.NaN;
            if (_loaded[i] != value)
            {
                _workers[i]?.Dispose(); _in[i]?.Dispose();
                var model = ModelLoader.Load(value); int dim = model.inputs[0].shape.Get(1);
                if (dim != ObsBuilder.ObsDim + 9 && dim != ObsBuilder.ObsDim + 18) { Debug.LogWarning($"[WinOdds] value model input {dim} is not a critic of this game"); _loaded[i] = value; _workers[i] = null; return float.NaN; }
                _workers[i] = new Worker(model, BackendType.CPU); _in[i] = new Tensor<float>(new TensorShape(1, dim)); _buf[i] = new float[dim]; _loaded[i] = value;
            }
            if (_workers[i] == null) return float.NaN;
            var b = _buf[i]; Array.Copy(r.Obs, b, ObsBuilder.ObsDim);           // the runner's last observation (Unity adds no noise)
            double* q = d->qpos + r.Map.RootQposAdr; double* v = d->qvel + r.Map.RootDofAdr; int o = ObsBuilder.ObsDim;
            b[o] = (float)v[0]; b[o + 1] = (float)v[1]; b[o + 2] = (float)v[2];
            var arena = r.Spec?.arena ?? new ArenaSpec(); double rad = Math.Sqrt(q[0] * q[0] + q[1] * q[1]);
            b[o + 3] = (float)(q[2] - arena.Ground(rad));
            FootContacts(r, d, out b[o + 4], out b[o + 5]);
            b[o + 6] = 1f; b[o + 7] = 0.6f; b[o + 8] = 1f;
            if (b.Length > o + 9 && r.opponent != null && r.opponent.Map != null) ObsBuilder.FillCombat(b, o + 9, d, r.Map, r.opponent.Map);
            _in[i].Upload(b); _workers[i].Schedule(_in[i]);
            var output = _workers[i].PeekOutput() as Tensor<float>; output.CompleteAllPendingOperations();
            return output.AsReadOnlySpan()[0];
        }

        void FootContacts(PolicyRunner r, MujocoLib.mjData_* d, out float left, out float right)
        {
            left = right = 0;
            var m = MjScene.Instance.Model;
            int lb = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, r.robotPrefix + "left_ankle_roll_link");
            int rb = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, r.robotPrefix + "right_ankle_roll_link");
            for (int c = 0; c < d->ncon; c++)
            {
                var con = d->contact[c]; int b1 = m->geom_bodyid[con.geom1], b2 = m->geom_bodyid[con.geom2];
                if (b1 == lb || b2 == lb) left = 1; if (b1 == rb || b2 == rb) right = 1;
            }
        }
    }
}
