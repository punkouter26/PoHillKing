using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// The 8 free-joint boxes imported from the MJCF (box0..box7), parked at z=-50 on a hidden shelf.
    /// Firing = writing qpos/qvel of the free joint. No Instantiate/Destroy, no Rigidbody.
    /// </summary>
    public unsafe class ProjectilePool : MonoBehaviour
    {
        public TextAsset jointMapJson;
        int[] _qpos, _dof;
        int _next;
        static readonly double[] ParkZ = { -50 };

        void OnEnable()
        {
            MjScene.Instance.postInitEvent += OnSceneInit;
            if (MjScene.Instance.Model != null) OnSceneInit(null, null);
        }
        void OnDisable() { if (MjScene.InstanceExists) MjScene.Instance.postInitEvent -= OnSceneInit; }

        void OnSceneInit(object s, MjStepArgs a)
        {
            var m = MjScene.Instance.Model;
            var names = JointMap.LoadSpec(jointMapJson).pool_bodies;
            _qpos = new int[names.Length]; _dof = new int[names.Length];
            for (int i = 0; i < names.Length; i++)
            {
                int j = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, names[i] + "_free");
                if (j < 0) throw new System.InvalidOperationException($"pool joint {names[i]}_free missing");
                _qpos[i] = m->jnt_qposadr[j]; _dof[i] = m->jnt_dofadr[j];
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
        }

        /// <summary>Fire one box at a MuJoCo-frame target point from a random horizontal direction.</summary>
        public void FireAt(Vector3 mjTarget, float speed, float range = 3f)
        {
            float ang = Random.Range(0f, 2f * Mathf.PI);
            var dir = new Vector3(Mathf.Cos(ang), Mathf.Sin(ang), 0f);
            var from = mjTarget + dir * range + new Vector3(0, 0, 0.3f);
            var vel = (mjTarget - from).normalized * speed;
            Fire(from, vel);
        }

        public void ParkAll()
        {
            var d = MjScene.Instance.Data;
            for (int i = 0; i < _qpos.Length; i++)
            {
                double* q = d->qpos + _qpos[i]; double* v = d->qvel + _dof[i];
                q[0] = -1.75 + 0.5 * i; q[1] = 0; q[2] = ParkZ[0]; q[3] = 1; q[4] = q[5] = q[6] = 0;
                for (int k = 0; k < 6; k++) v[k] = 0;
            }
        }
    }
}
