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
    [DefaultExecutionOrder(-100)]   // subscribes to MjScene.postInitEvent first: sets ls_iterations and the keyframe before other listeners read the model
    public unsafe class PolicyRunner : MonoBehaviour
    {
        public TextAsset jointMapJson;
        [Tooltip("a_ or b_")] public string robotPrefix = "a_";
        [Tooltip("ONNX policy. Leave empty for passive hold.")] public ModelAsset policy;
        public Vector3 command;   // vx, vy, yaw rate
        public bool paused;

        public JointMap Map { get; private set; }
        public float[] Obs { get; } = new float[ObsBuilder.ObsDim];
        public float[] LastAction { get; private set; }
        public float[] RawAction { get; private set; }   // network output before clamping
        public double[] Ctrl { get; private set; }        // held position targets, canonical order
        public event System.Action<PolicyRunner> OnPolicyStep;
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
            RawAction = new float[_spec.joints.Length]; Ctrl = new double[_spec.joints.Length];
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
            // The plugin's MjGlobalSettings has no ls_iterations field; mirror the training value directly.
            if (_spec.ls_iterations > 0) MjScene.Instance.Model->opt.ls_iterations = _spec.ls_iterations;
            if (Mathf.Abs(Time.fixedDeltaTime - _spec.sim_dt) > 1e-6f)
                Debug.LogError($"Fixed Timestep {Time.fixedDeltaTime} != training sim_dt {_spec.sim_dt}");
            _substep = 0; _phase = 0; PolicySteps = 0;
            System.Array.Clear(LastAction, 0, LastAction.Length);
            ResetPose();   // the plugin cannot import keyframes: start from the training keyframe explicitly
        }

        void OnPreStep(object sender, MjStepArgs args)
        {
            if (Map == null || paused) return;
            var d = MjScene.Instance.Data;
            if (_substep++ % _spec.decimation == 0) PolicyTick(d);
            // MjActuator.OnSyncState writes its own Control field (0) into mjData.ctrl after every mj_step, so the
            // held targets must be rewritten before every physics step, not only on policy ticks.
            for (int i = 0; i < Map.N; i++) d->ctrl[Map.ActId[i]] = Ctrl[i];
        }

        void PolicyTick(MujocoLib.mjData_* d)
        {
            if (_worker == null)
            {
                for (int i = 0; i < Map.N; i++) Ctrl[i] = _spec.hold_ctrl[i];
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
                RawAction[i] = act[i];
                float a = Mathf.Clamp(act[i], -1f, 1f);
                LastAction[i] = a;
                Ctrl[i] = System.Math.Clamp(Map.DefaultPose[i] + _spec.action_scale * a, Map.CtrlMin[i], Map.CtrlMax[i]);
            }
            _phase += 2f * Mathf.PI * ObsBuilder.GaitFreqHz * _spec.ctrl_dt;
            if (_phase > 2f * Mathf.PI) _phase -= 2f * Mathf.PI;
            PolicySteps++;
            OnPolicyStep?.Invoke(this);
        }

        /// <summary>Training keyframe for this robot: default joint pose, pelvis at its spawn x/y/yaw (qpos0) and the
        /// keyframe height, zero velocities, hold ctrl. Pure mjData writes; no scene recreation.</summary>
        public void ResetPose()
        {
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data;
            for (int k = 0; k < 7; k++) d->qpos[Map.RootQposAdr + k] = m->qpos0[Map.RootQposAdr + k];
            d->qpos[Map.RootQposAdr + 2] = _spec.key_root_z;
            for (int k = 0; k < 6; k++) d->qvel[Map.RootDofAdr + k] = 0;
            for (int i = 0; i < Map.N; i++)
            {
                d->qpos[Map.QposAdr[i]] = Map.DefaultPose[i]; d->qvel[Map.DofAdr[i]] = 0;
                Ctrl[i] = _spec.hold_ctrl[i]; d->ctrl[Map.ActId[i]] = Ctrl[i];
            }
            _substep = 0; _phase = 0;
            System.Array.Clear(LastAction, 0, LastAction.Length);
        }
    }
}
