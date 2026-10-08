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
        public string[] robot_prefixes;
        public string[] pool_bodies;
        public int ls_iterations;
    }

    public unsafe class JointMap
    {
        public readonly string Prefix;
        public readonly int N;
        public readonly int[] QposAdr;     // per canonical joint
        public readonly int[] DofAdr;
        public readonly int[] ActId;       // actuator index per canonical joint
        public readonly double[] CtrlMin, CtrlMax;
        public readonly int RootQposAdr;   // free joint: 7 qpos (x y z qw qx qy qz)
        public readonly int RootDofAdr;    // 6 dof (vx vy vz wx wy wz), angular part is body-frame
        public readonly int PelvisBodyId;
        public readonly float[] DefaultPose;

        public JointMap(MujocoLib.mjModel_* m, JointMapSpec spec, string prefix)
        {
            Prefix = prefix;
            N = spec.joints.Length;
            QposAdr = new int[N]; DofAdr = new int[N]; ActId = new int[N];
            CtrlMin = new double[N]; CtrlMax = new double[N];
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
