using System;
using System.Collections.Generic;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Resolves the canonical 29 joint names (training/assets/g1/joint_map.json) to indices in the
    /// Unity-compiled mjModel for one robot prefix ("a_" / "b_"). All obs/action code goes through this map;
    /// nothing is ever indexed by MJCF order assumptions.
    /// </summary>
    [Serializable]
    public class JointMapSpec
    {
        public string[] joints;
        public float[] default_pose;
        public float action_scale;
        public float sim_dt;
        public float ctrl_dt;
        public int decimation;
        public string[] pool_bodies;
        public int ls_iterations;
        public float[] hold_ctrl;     // gravity-compensated stance targets, canonical order
        public float key_root_z;       // pelvis height of the training keyframe
        public ArenaSpec arena;        // the hill and the "out" rule (training/koth/rules.py); defaults when absent
    }

    /// <summary>Hill shape and the game's elimination thresholds, written by build_mjcf.py from koth/rules.py. Doubles,
    /// so the C# rule is bit-identical to the Python one (a float 1/9 differs in the 8th digit).</summary>
    [Serializable]
    public class ArenaSpec
    {
        public double plateau_r = 1.5, slope_k = 1.0 / 9.0, out_radius = 1.7, fall_height = 0.3, fall_up = 0.3;
        /// <summary>Height of the dome surface at radius r (flat plateau, then z = -k (r - R)²).</summary>
        public double Ground(double r) => -slope_k * Math.Pow(Math.Max(r - plateau_r, 0), 2);
    }

    public unsafe class JointMap
    {
        public readonly int N;
        public readonly int[] QposAdr;     // per canonical joint
        public readonly int[] DofAdr;
        public readonly int[] ActId;       // actuator index per canonical joint
        public readonly double[] CtrlMin, CtrlMax;
        public readonly double[] Kp, Kv, Tau;  // nominal position-actuator gains and force limit (ActionMap's torque window)
        public readonly int RootQposAdr;   // free joint: 7 qpos (x y z qw qx qy qz)
        public readonly int RootDofAdr;    // 6 dof (vx vy vz wx wy wz), angular part is body-frame
        public readonly int PelvisBodyId;
        public readonly float[] DefaultPose;

        public JointMap(MujocoLib.mjModel_* m, JointMapSpec spec, string prefix)
        {
            N = spec.joints.Length;
            QposAdr = new int[N]; DofAdr = new int[N]; ActId = new int[N];
            CtrlMin = new double[N]; CtrlMax = new double[N]; Kp = new double[N]; Kv = new double[N]; Tau = new double[N];
            DefaultPose = spec.default_pose;
            var missing = new List<string>();
            for (int i = 0; i < N; i++)
            {
                string name = prefix + spec.joints[i];
                int j = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, name);
                int a = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_ACTUATOR, name);
                if (j < 0 || a < 0) { missing.Add(name); continue; }
                QposAdr[i] = m->jnt_qposadr[j];
                DofAdr[i] = m->jnt_dofadr[j];
                ActId[i] = a;
                CtrlMin[i] = m->actuator_ctrlrange[2 * a];
                CtrlMax[i] = m->actuator_ctrlrange[2 * a + 1];
                Kp[i] = m->actuator_gainprm[a * MujocoLib.mjNGAIN]; Kv[i] = -m->actuator_biasprm[a * MujocoLib.mjNBIAS + 2]; Tau[i] = m->actuator_forcerange[2 * a + 1];
            }
            int root = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint");
            PelvisBodyId = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, prefix + "pelvis");
            if (root < 0 || PelvisBodyId < 0) missing.Add(prefix + "floating_base_joint/pelvis");
            if (missing.Count > 0)
                throw new InvalidOperationException($"JointMap[{prefix}]: {missing.Count} names not found in mjModel: {string.Join(", ", missing)}");
            RootQposAdr = m->jnt_qposadr[root];
            RootDofAdr = m->jnt_dofadr[root];
            Debug.Log($"JointMap[{prefix}] resolved {N} joints. root qpos {RootQposAdr}, pelvis body {PelvisBodyId}, first act {ActId[0]}");
        }

        public static JointMapSpec LoadSpec(TextAsset json) => JsonUtility.FromJson<JointMapSpec>(json.text);
    }
}
