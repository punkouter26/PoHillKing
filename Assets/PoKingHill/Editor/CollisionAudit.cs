using System;
using System.Collections.Generic;
using System.Text;
using Mujoco;

namespace PoKingHill.EditorTools
{
    /// <summary>
    /// The Unity side of training/scripts/collision_audit.py: loads an imported scene's MJCF (the _unity.xml the plugin
    /// reads) with the MuJoCo C API and lists the self-collision pairs of one fighter that interpenetrate in its zero
    /// pose (every hinge at 0). Only pairs the simulation collides count (contype/conaffinity allow it, no exclude), on
    /// bodies that are not the same or parent and child. The Python script covers the rest pose, T-pose, swing arcs and
    /// the pairs that would touch with every self-collision on, and writes the HTML report.
    /// </summary>
    public static unsafe class CollisionAudit
    {
        public readonly struct Overlap
        {
            public readonly string A, B; public readonly double DepthMm;
            public Overlap(string a, string b, double depthMm) { A = a; B = b; DepthMm = depthMm; }
            public override string ToString() => $"{A} / {B} {DepthMm:0.0} mm";
        }

        public static List<Overlap> ZeroPose(string xmlPath, string prefix = "a_", double tolerance = 1e-4)
        {
            var err = new StringBuilder(1000);
            var m = MujocoLib.mj_loadXML(xmlPath, null, err, err.Capacity);
            if (m == null) throw new Exception($"mj_loadXML {xmlPath}: {err}");
            var d = MujocoLib.mj_makeData(m);
            try
            {
                for (int j = 0; j < (int)m->njnt; j++)
                    if (m->jnt_type[j] == (int)MujocoLib.mjtJoint.mjJNT_HINGE && Name(m, MujocoLib.mjtObj.mjOBJ_JOINT, j).StartsWith(prefix)) d->qpos[m->jnt_qposadr[j]] = 0;
                MujocoLib.mj_kinematics(m, d); MujocoLib.mj_collision(m, d);
                var geoms = new List<int>();
                for (int g = 0; g < (int)m->ngeom; g++)
                {
                    int b = m->geom_bodyid[g];
                    if (b != 0 && (m->geom_contype[g] != 0 || m->geom_conaffinity[g] != 0) && Name(m, MujocoLib.mjtObj.mjOBJ_BODY, b).StartsWith(prefix)) geoms.Add(g);
                }
                var excluded = new HashSet<int>(); for (int e = 0; e < (int)m->nexclude; e++) excluded.Add(m->exclude_signature[e]);
                var found = new List<Overlap>(); double* fromto = stackalloc double[6];
                for (int i = 0; i < geoms.Count; i++)
                    for (int k = i + 1; k < geoms.Count; k++)
                    {
                        int a = geoms[i], b = geoms[k], ba = m->geom_bodyid[a], bb = m->geom_bodyid[b];
                        if (ba == bb || m->body_parentid[ba] == bb || m->body_parentid[bb] == ba) continue;
                        bool on = ((m->geom_contype[a] & m->geom_conaffinity[b]) | (m->geom_contype[b] & m->geom_conaffinity[a])) != 0
                                  && !excluded.Contains((ba << 16) + bb) && !excluded.Contains((bb << 16) + ba);
                        if (!on) continue;
                        double dist = MujocoLib.mj_geomDistance(m, d, a, b, 0.05, fromto);
                        if (dist < -tolerance) found.Add(new Overlap(Name(m, MujocoLib.mjtObj.mjOBJ_GEOM, a), Name(m, MujocoLib.mjtObj.mjOBJ_GEOM, b), -dist * 1000));
                    }
                return found;
            }
            finally { MujocoLib.mj_deleteData(d); MujocoLib.mj_deleteModel(m); }
        }

        static string Name(MujocoLib.mjModel_* m, MujocoLib.mjtObj type, int id) =>
            System.Runtime.InteropServices.Marshal.PtrToStringAnsi(MujocoLib.mj_id2name(m, (int)type, id)) ?? "";
    }
}
