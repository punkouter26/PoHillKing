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
        [Tooltip("None: use the command field. Center/Opponent: command is computed every tick by GoalCommand (same law as training).")]
        public GoalMode goal = GoalMode.None;
        public float goalStopDist = 0.3f, goalVmax = 0.8f;
        public PolicyRunner opponent;
        [Tooltip("Network input size: 103 = locomotion state, 112 = state + 9 opponent/ring values (combat policies).")]
        public int policyObsDim = ObsBuilder.ObsDim;
        public bool paused;
        [Tooltip("Shove style (mirror of SHOVE_* in training/koth/env.py): the arm rest pose moves to hands-forward as the opponent closes from 1.6 m to 1.0 m, and the action is low-passed before it becomes a joint target.")]
        public bool shoveStyle;
        const float ShoveFar = 1.6f, ShoveNear = 1.0f, ShoveSmooth = 0.5f, ShoveShoulderPitch = -1.0f, ShoveElbow = 0.5f;

        public JointMap Map { get; private set; }
        public float[] Obs { get; private set; } = new float[ObsBuilder.ObsDim];
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
        float[] _actF, _shoveOffset;
        readonly System.Diagnostics.Stopwatch _sw = new();

        void OnEnable()
        {
            _spec = JointMap.LoadSpec(jointMapJson);
            LastAction = new float[_spec.joints.Length];
            RawAction = new float[_spec.joints.Length]; Ctrl = new double[_spec.joints.Length];
            _actF = new float[_spec.joints.Length]; _shoveOffset = new float[_spec.joints.Length];
            for (int i = 0; i < _spec.joints.Length; i++)
                _shoveOffset[i] = _spec.joints[i].Contains("shoulder_pitch") ? ShoveShoulderPitch - _spec.default_pose[i]
                                : _spec.joints[i].Contains("elbow") ? ShoveElbow - _spec.default_pose[i] : 0f;
            if (policy != null)
            {
                _worker = new Worker(ModelLoader.Load(policy), BackendType.CPU);
                Obs = new float[policyObsDim];
                _input = new Tensor<float>(new TensorShape(1, policyObsDim));
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
            if (goal != GoalMode.None)
            {
                double* q = d->qpos + Map.RootQposAdr; double gx = -q[0], gy = -q[1];
                if (goal == GoalMode.Opponent && opponent != null && opponent.Map != null)
                {
                    double* o = d->qpos + opponent.Map.RootQposAdr; gx = o[0] - q[0]; gy = o[1] - q[1];
                }
                GoalCommand.Compute(q + 3, gx, gy, goalStopDist, goalVmax, _cmd);
                command = new Vector3(_cmd[0], _cmd[1], _cmd[2]);
            }
            ObsBuilder.Fill(Obs, d, Map, _cmd, LastAction, _phase);
            if (policyObsDim > ObsBuilder.ObsDim && opponent != null && opponent.Map != null)
                ObsBuilder.FillCombat(Obs, ObsBuilder.ObsDim, d, Map, opponent.Map);
            _sw.Restart();
            _input.Upload(Obs);
            _worker.Schedule(_input);
            // CPU backend: wait for the jobs, then read the worker's own output in place (no clone, no managed array).
            var output = _worker.PeekOutput() as Tensor<float>;
            output.CompleteAllPendingOperations();
            var act = output.AsReadOnlySpan();
            LastInferenceMs = _sw.Elapsed.TotalMilliseconds;
            float blend = 0f;
            if (shoveStyle && opponent != null && opponent.Map != null)
            {
                double* me = d->qpos + Map.RootQposAdr; double* op = d->qpos + opponent.Map.RootQposAdr;
                double dist = System.Math.Sqrt((op[0] - me[0]) * (op[0] - me[0]) + (op[1] - me[1]) * (op[1] - me[1]));
                blend = Mathf.Clamp01((ShoveFar - (float)dist) / (ShoveFar - ShoveNear));
            }
            for (int i = 0; i < Map.N; i++)
            {
                RawAction[i] = act[i];
                float a = Mathf.Clamp(act[i], -1f, 1f);
                LastAction[i] = a;
                float f = shoveStyle ? ShoveSmooth * _actF[i] + (1f - ShoveSmooth) * a : a; _actF[i] = f;
                Ctrl[i] = System.Math.Clamp(Map.DefaultPose[i] + blend * _shoveOffset[i] + _spec.action_scale * f, Map.CtrlMin[i], Map.CtrlMax[i]);
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
            System.Array.Clear(LastAction, 0, LastAction.Length); System.Array.Clear(_actF, 0, _actF.Length);
        }

        /// <summary>Swap the brain at runtime (menu fighter selection). policy = null gives the passive stance.</summary>
        public void SetBrain(ModelAsset newPolicy, int obsDim, GoalMode newGoal, float stopDist, float vmax, bool shove = false)
        {
            _worker?.Dispose(); _input?.Dispose(); _worker = null; _input = null;
            policy = newPolicy; policyObsDim = obsDim; goal = newGoal; goalStopDist = stopDist; goalVmax = vmax; command = Vector3.zero; shoveStyle = shove;
            if (policy != null)
            {
                _worker = new Worker(ModelLoader.Load(policy), BackendType.CPU);
                Obs = new float[policyObsDim]; _input = new Tensor<float>(new TensorShape(1, policyObsDim));
            }
            _substep = 0; _phase = 0; System.Array.Clear(LastAction, 0, LastAction.Length); System.Array.Clear(_actF, 0, _actF.Length);
        }

        /// <summary>Keyframe pose at a chosen spot on the summit (MuJoCo frame x, y in metres, yaw in radians).</summary>
        public void ResetPose(float x, float y, float yaw)
        {
            ResetPose();
            var d = MjScene.Instance.Data; double* q = d->qpos + Map.RootQposAdr;
            q[0] = x; q[1] = y; q[3] = System.Math.Cos(yaw / 2); q[4] = 0; q[5] = 0; q[6] = System.Math.Sin(yaw / 2);
        }
    }
}
