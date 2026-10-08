using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Rising sea. The water level climbs linearly from startDepth below the summit to the summit plateau (z = 0)
    /// in a duration drawn from [minSeconds, maxSeconds] each round; reaching the summit is the round deadline.
    /// Bodies below the surface get buoyancy and drag through mjData.xfrc_applied (native solver forces only, no
    /// PhysX). In training the sea is only the deadline: a robot that leaves the summit has already lost, so these
    /// forces act on robots that are out of the round and are presentation, not a policy input.
    ///   buoyancy_i = buoyancyRatio * m_i * g * f_i   (up),   drag_i = -dragPerKg * m_i * v_i * f_i
    ///   f_i = clamp((level - z_i) / band + 0.5, 0, 1) for the body's centre of mass height z_i
    /// </summary>
    public unsafe class Sea : MonoBehaviour
    {
        public Transform waterVisual;           // render-only disc, moved to the water level (Unity Y = MuJoCo Z)
        public float startDepth = 6f, minSeconds = 20f, maxSeconds = 30f;
        public float buoyancyRatio = 1.05f, dragPerKg = 4f, band = 0.2f;
        public string[] robotPrefixes = { "a_", "b_" };

        public float Level { get; private set; }          // MuJoCo z of the surface
        public float Duration { get; private set; }
        public bool ReachedSummit => Level >= 0f;

        int[] _bodies; double _t0;

        void OnEnable()
        {
            MjScene.Instance.postInitEvent += OnInit; MjScene.Instance.preUpdateEvent += Pre;
            if (MjScene.Instance.Model != null) OnInit(null, null);
        }
        void OnDisable()
        {
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.postInitEvent -= OnInit; MjScene.Instance.preUpdateEvent -= Pre;
        }

        void OnInit(object s, MjStepArgs a)
        {
            var m = MjScene.Instance.Model; var list = new System.Collections.Generic.List<int>();
            int nbody = (int)m->nbody; for (int b = 1; b < nbody; b++)
            {
                string name = System.Runtime.InteropServices.Marshal.PtrToStringAnsi((System.IntPtr)MujocoLib.mj_id2name(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, b)) ?? "";
                foreach (var p in robotPrefixes) if (name.StartsWith(p)) { list.Add(b); break; }
            }
            _bodies = list.ToArray();
            Restart();
        }

        /// <summary>Start a new flood: water back at the bottom, new random arrival time.</summary>
        public void Restart()
        {
            Duration = Random.Range(minSeconds, maxSeconds);
            _t0 = MjScene.Instance.Data != null ? MjScene.Instance.Data->time : 0;
            Level = -startDepth;
        }

        void Pre(object s, MjStepArgs a)
        {
            if (_bodies == null) return;
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data;
            Level = Mathf.Min(0.05f, -startDepth + startDepth * (float)((d->time - _t0) / Duration));
            double g = -m->opt.gravity[2];
            foreach (int b in _bodies)
            {
                double z = d->xipos[3 * b + 2], f = System.Math.Clamp((Level - z) / band + 0.5, 0.0, 1.0), mass = m->body_mass[b];
                double* F = d->xfrc_applied + 6 * b; double* v = d->cvel + 6 * b + 3;       // cvel = (angular, linear)
                F[0] = -dragPerKg * mass * v[0] * f; F[1] = -dragPerKg * mass * v[1] * f;
                F[2] = buoyancyRatio * mass * g * f - dragPerKg * mass * v[2] * f;
            }
        }

        void LateUpdate()
        {
            if (waterVisual != null) waterVisual.position = new Vector3(0f, Level, 0f);
        }
    }
}
