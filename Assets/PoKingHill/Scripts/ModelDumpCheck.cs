using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Parity gate B.4: compare the Unity-compiled mjModel against training/assets/g1/model_dump_*.json
    /// (written by koth/build_mjcf.py). Logs every mismatch; Passed is false if any.
    /// Compared: nq nv nu nbody ngeom timestep integrator iterations solver cone gravity, per-joint
    /// range/damping/armature/frictionloss/actfrcrange, per-actuator kp/kv/ctrlrange, per-body mass.
    /// </summary>
    public unsafe class ModelDumpCheck : MonoBehaviour
    {
        public TextAsset modelDumpJson;
        public bool Passed { get; private set; }
        public string Report { get; private set; } = "";

        void OnEnable()
        {
            MjScene.Instance.postInitEvent += (s, a) => Run();
            if (MjScene.Instance.Model != null) Run();
        }

        public void Run()
        {
            var m = MjScene.Instance.Model;
            var dump = MiniJson.Parse(modelDumpJson.text) as Dictionary<string, object>;
            var sb = new StringBuilder(); int bad = 0;
            void Check(string what, double unity, double train, double tol)
            {
                if (System.Math.Abs(unity - train) > tol) { bad++; sb.AppendLine($"MISMATCH {what}: unity={unity} train={train}"); }
            }
            Check("nq", m->nq, (long)dump["nq"], 0); Check("nv", m->nv, (long)dump["nv"], 0); Check("nu", m->nu, (long)dump["nu"], 0);
            Check("nbody", m->nbody, (long)dump["nbody"], 0); Check("ngeom", m->ngeom, (long)dump["ngeom"], 0);
            Check("timestep", m->opt.timestep, (double)dump["timestep"], 1e-9);
            Check("integrator", m->opt.integrator, (long)dump["integrator"], 0);
            Check("iterations", m->opt.iterations, (long)dump["iterations"], 0);
            Check("ls_iterations", m->opt.ls_iterations, (long)dump["ls_iterations"], 0);
            Check("solver", m->opt.solver, (long)dump["solver"], 0); Check("cone", m->opt.cone, (long)dump["cone"], 0);
            var g = (List<object>)dump["gravity"]; Check("gravity.z", m->opt.gravity[2], (double)g[2], 1e-6);

            foreach (Dictionary<string, object> j in (List<object>)dump["joints"])
            {
                string name = (string)j["name"];
                int id = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, name);
                if (id < 0) { bad++; sb.AppendLine($"MISSING joint {name}"); continue; }
                var r = (List<object>)j["range"]; var afr = (List<object>)j["actfrcrange"];
                Check($"{name}.range0", m->jnt_range[2 * id], (double)r[0], 1e-6); Check($"{name}.range1", m->jnt_range[2 * id + 1], (double)r[1], 1e-6);
                int dof = m->jnt_dofadr[id];
                if ((long)j["type"] != 0) // not free
                {
                    Check($"{name}.damping", m->dof_damping[dof], (double)j["damping"], 1e-6);
                    Check($"{name}.armature", m->dof_armature[dof], (double)j["armature"], 1e-6);
                    Check($"{name}.frictionloss", m->dof_frictionloss[dof], (double)j["frictionloss"], 1e-6);
                    Check($"{name}.actfrc0", m->jnt_actfrcrange[2 * id], (double)afr[0], 1e-6);
                    Check($"{name}.actfrc1", m->jnt_actfrcrange[2 * id + 1], (double)afr[1], 1e-6);
                }
            }
            foreach (Dictionary<string, object> a in (List<object>)dump["actuators"])
            {
                string name = (string)a["name"];
                int id = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_ACTUATOR, name);
                if (id < 0) { bad++; sb.AppendLine($"MISSING actuator {name}"); continue; }
                Check($"{name}.kp", m->actuator_gainprm[id * MujocoLib.mjNGAIN], (double)a["kp"], 1e-6);
                Check($"{name}.kv", -m->actuator_biasprm[id * MujocoLib.mjNBIAS + 2], (double)a["kv"], 1e-6);
                var cr = (List<object>)a["ctrlrange"];
                Check($"{name}.ctrl0", m->actuator_ctrlrange[2 * id], (double)cr[0], 1e-6); Check($"{name}.ctrl1", m->actuator_ctrlrange[2 * id + 1], (double)cr[1], 1e-6);
            }
            foreach (Dictionary<string, object> b in (List<object>)dump["bodies"])
            {
                string name = (string)b["name"]; if (name == "world") continue;
                int id = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, name);
                if (id < 0) { bad++; sb.AppendLine($"MISSING body {name}"); continue; }
                Check($"{name}.mass", m->body_mass[id], (double)b["mass"], 1e-6);
            }
            Passed = bad == 0;
            Report = Passed ? $"ModelDumpCheck PASS (nq={m->nq} nv={m->nv} nu={m->nu} nbody={m->nbody})" : $"ModelDumpCheck FAIL: {bad} mismatches\n{sb}";
            if (Passed) Debug.Log(Report); else Debug.LogError(Report);
        }
    }

    /// <summary>Tiny JSON reader (objects, arrays, numbers as long/double, strings, bools, null). Enough for the dump file.</summary>
    public static class MiniJson
    {
        public static object Parse(string s) { int i = 0; return Value(s, ref i); }
        static void Ws(string s, ref int i) { while (i < s.Length && char.IsWhiteSpace(s[i])) i++; }
        static object Value(string s, ref int i)
        {
            Ws(s, ref i);
            char c = s[i];
            if (c == '{') { i++; var d = new Dictionary<string, object>(); Ws(s, ref i); if (s[i] == '}') { i++; return d; }
                while (true) { Ws(s, ref i); string k = Str(s, ref i); Ws(s, ref i); i++; d[k] = Value(s, ref i); Ws(s, ref i); if (s[i++] == '}') return d; } }
            if (c == '[') { i++; var l = new List<object>(); Ws(s, ref i); if (s[i] == ']') { i++; return l; }
                while (true) { l.Add(Value(s, ref i)); Ws(s, ref i); if (s[i++] == ']') return l; } }
            if (c == '"') return Str(s, ref i);
            if (s.Substring(i).StartsWith("true")) { i += 4; return true; }
            if (s.Substring(i).StartsWith("false")) { i += 5; return false; }
            if (s.Substring(i).StartsWith("null")) { i += 4; return null; }
            int st = i; while (i < s.Length && "+-0123456789.eE".IndexOf(s[i]) >= 0) i++;
            string num = s.Substring(st, i - st);
            if (num.IndexOfAny(new[] { '.', 'e', 'E' }) < 0 && long.TryParse(num, out long lv)) return lv;
            return double.Parse(num, CultureInfo.InvariantCulture);
        }
        static string Str(string s, ref int i)
        {
            var sb = new StringBuilder(); i++;
            while (s[i] != '"') { if (s[i] == '\\') { i++; } sb.Append(s[i++]); }
            i++; return sb.ToString();
        }
    }
}
