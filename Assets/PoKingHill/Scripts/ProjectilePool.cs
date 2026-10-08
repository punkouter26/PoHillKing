using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Pre-allocated pool of free-joint boxes imported from the MJCF (box0..boxN). Firing = writing qpos/qvel of
    /// the free joint. Parked boxes float at their MJCF pose with no contacts and are re-pinned (pose restored,
    /// velocity zeroed) before every physics step, the same rule training uses. No Instantiate/Destroy, no Rigidbody.
    /// </summary>
    public unsafe class ProjectilePool : MonoBehaviour
    {
        public TextAsset jointMapJson;
        [Tooltip("Seconds a fired box stays live before it is parked again. Training uses 2.5.")]
        public float lifeSeconds = 2.5f;

        int[] _qpos, _dof;
        double[][] _park;      // 7 qpos values per box, captured from the compiled model (qpos0)
        double[] _liveUntil;   // sim time; <= now means parked
        int _next;

        void OnEnable()
        {
            MjScene.Instance.postInitEvent += OnSceneInit;
            MjScene.Instance.preUpdateEvent += OnPreStep;
            if (MjScene.Instance.Model != null) OnSceneInit(null, null);
        }

        void OnDisable()
        {
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.postInitEvent -= OnSceneInit;
            MjScene.Instance.preUpdateEvent -= OnPreStep;
        }

        void OnSceneInit(object s, MjStepArgs a)
        {
            var m = MjScene.Instance.Model;
            var names = JointMap.LoadSpec(jointMapJson).pool_bodies;
            int n = names.Length;
            _qpos = new int[n]; _dof = new int[n]; _park = new double[n][]; _liveUntil = new double[n];
            for (int i = 0; i < n; i++)
            {
                int j = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, names[i] + "_free");
                if (j < 0) throw new System.InvalidOperationException($"pool joint {names[i]}_free missing");
                _qpos[i] = m->jnt_qposadr[j]; _dof[i] = m->jnt_dofadr[j];
                _park[i] = new double[7];
                for (int k = 0; k < 7; k++) _park[i][k] = m->qpos0[_qpos[i] + k];
            }
        }

        void OnPreStep(object s, MjStepArgs a)
        {
            if (_qpos == null) return;
            var d = MjScene.Instance.Data;
            for (int i = 0; i < _qpos.Length; i++)
            {
                if (d->time < _liveUntil[i]) continue;
                for (int k = 0; k < 7; k++) d->qpos[_qpos[i] + k] = _park[i][k];
                for (int k = 0; k < 6; k++) d->qvel[_dof[i] + k] = 0;
            }
        }

        /// <summary>MuJoCo-frame position and velocity (Z up).</summary>
        public void Fire(Vector3 mjPos, Vector3 mjVel)
        {
            var d = MjScene.Instance.Data;
            int i = _next++ % _qpos.Length;
            double* q = d->qpos + _qpos[i]; double* v = d->qvel + _dof[i];
            q[0] = mjPos.x; q[1] = mjPos.y; q[2] = mjPos.z; q[3] = 1; q[4] = q[5] = q[6] = 0;
            v[0] = mjVel.x; v[1] = mjVel.y; v[2] = mjVel.z; v[3] = v[4] = v[5] = 0;
            _liveUntil[i] = d->time + lifeSeconds;
        }

        /// <summary>Fire one box at a MuJoCo-frame target point from a random horizontal direction (same recipe as training).</summary>
        public void FireAt(Vector3 mjTarget, float speed, float range = 3f)
        {
            float ang = Random.Range(0f, 2f * Mathf.PI);
            var from = mjTarget + new Vector3(range * Mathf.Cos(ang), range * Mathf.Sin(ang), 0.3f);
            Fire(from, (mjTarget - from).normalized * speed);
        }

        public void ParkAll()
        {
            if (_liveUntil == null) return;
            for (int i = 0; i < _liveUntil.Length; i++) _liveUntil[i] = 0;
        }
    }
}
