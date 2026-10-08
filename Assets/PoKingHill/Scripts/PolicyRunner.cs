using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// One policy driving one robot. Runs inference once every <c>decimation</c> physics steps from
    /// MjScene.preUpdateEvent (never from Update), writes ctrl = default + scale * action into mjData.ctrl and
    /// holds it between policy ticks. With no model assigned it writes the gravity-compensated hold ctrl
    /// (HoldCtrl) so the robot stands passively: that is the zero-brain parity test.
    /// </summary>
    public unsafe class PolicyRunner : MonoBehaviour
    {
        public TextAsset jointMapJson;
        [Tooltip("a_ or b_")] public string robotPrefix = "a_";
        [Tooltip("ONNX policy. Leave empty for passive hold.")] public ModelAsset policy;
        [Tooltip("Hold ctrl from the training keyframe (29 values). Used when no policy is set.")] public float[] holdCtrl;
        public Vector3 command;   // vx, vy, yaw rate
        public bool paused;

        public JointMap Map { get; private set; }
        public float[] Obs { get; } = new float[ObsBuilder.ObsDim];
        public float[] LastAction { get; private set; }
        public int PolicySteps { get; private set; }
        public double LastInferenceMs { get; private set; }

        JointMapSpec _spec;
        Worker _worker;
        Tensor<float> _input;
        int _substep;
        float _phase;
        readonly float[] _cmd = new float[3];
        readonly System.Diagnostics.Stopwatch _sw = new();

        void OnEnable()
        {
            _spec = JointMap.LoadSpec(jointMapJson);
            LastAction = new float[_spec.joints.Length];
            if (policy != null)
            {
                _worker = new Worker(ModelLoader.Load(policy), BackendType.CPU);
                _input = new Tensor<float>(new TensorShape(1, ObsBuilder.ObsDim));
            }
            MjScene.Instance.postInitEvent += OnSceneInit;
            MjScene.Instance.preUpdateEvent += OnPreStep;
            if (MjScene.Instance.Model != null) OnSceneInit(null, null);
        }

        void OnDisable()
        {
            if (MjScene.InstanceExists)
            {
                MjScene.Instance.postInitEvent -= OnSceneInit;
                MjScene.Instance.preUpdateEvent -= OnPreStep;
            }
            _worker?.Dispose(); _input?.Dispose();
        }

        void OnSceneInit(object sender, MjStepArgs args)
        {
            Map = new JointMap(MjScene.Instance.Model, _spec, robotPrefix);
            if (Mathf.Abs(Time.fixedDeltaTime - _spec.sim_dt) > 1e-6f)
                Debug.LogError($"Fixed Timestep {Time.fixedDeltaTime} != training sim_dt {_spec.sim_dt}");
            _substep = 0; _phase = 0; PolicySteps = 0;
            System.Array.Clear(LastAction, 0, LastAction.Length);
        }

        void OnPreStep(object sender, MjStepArgs args)
        {
            if (Map == null || paused) return;
            if (_substep++ % _spec.decimation != 0) return;   // hold ctrl between policy ticks
            var d = MjScene.Instance.Data;
            if (_worker == null)
            {
                for (int i = 0; i < Map.N; i++) d->ctrl[Map.ActId[i]] = holdCtrl != null && holdCtrl.Length == Map.N ? holdCtrl[i] : Map.DefaultPose[i];
                return;
            }
            _cmd[0] = command.x; _cmd[1] = command.y; _cmd[2] = command.z;
            ObsBuilder.Fill(Obs, d, Map, _cmd, LastAction, _phase);
            _sw.Restart();
            _input.Upload(Obs);
            _worker.Schedule(_input);
            using var outCpu = (_worker.PeekOutput() as Tensor<float>).ReadbackAndClone();
            var act = outCpu.DownloadToArray();
            LastInferenceMs = _sw.Elapsed.TotalMilliseconds;
            for (int i = 0; i < Map.N; i++)
            {
                float a = Mathf.Clamp(act[i], -1f, 1f);
                LastAction[i] = a;
                double target = Map.DefaultPose[i] + _spec.action_scale * a;
                d->ctrl[Map.ActId[i]] = System.Math.Clamp(target, Map.CtrlMin[i], Map.CtrlMax[i]);
            }
            _phase += 2f * Mathf.PI * ObsBuilder.GaitFreqHz * _spec.ctrl_dt;
            if (_phase > 2f * Mathf.PI) _phase -= 2f * Mathf.PI;
            PolicySteps++;
        }

        /// <summary>Write the training keyframe pose for this robot (qpos for the 29 joints + root) and zero velocities.</summary>
        public void ResetPose(double[] rootQpos7, float[] jointQpos)
        {
            var d = MjScene.Instance.Data;
            for (int k = 0; k < 7; k++) d->qpos[Map.RootQposAdr + k] = rootQpos7[k];
            for (int k = 0; k < 6; k++) d->qvel[Map.RootDofAdr + k] = 0;
            for (int i = 0; i < Map.N; i++) { d->qpos[Map.QposAdr[i]] = jointQpos[i]; d->qvel[Map.DofAdr[i]] = 0; }
            _substep = 0; _phase = 0;
            System.Array.Clear(LastAction, 0, LastAction.Length);
        }
    }
}
