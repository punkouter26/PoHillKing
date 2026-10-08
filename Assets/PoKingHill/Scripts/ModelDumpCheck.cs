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
    /// </summary>
    public unsafe class ModelDumpCheck : MonoBehaviour
    {
        public TextAsset modelDumpJson;
        public bool Passed { get; private set; }
        public bool HasRun { get; private set; }
        public int Mismatches { get; private set; }
        public string Report { get; private set; } = "";

        void OnEnable()
        {
            MjScene.Instance.postInitEvent += OnInit;
            if (MjScene.Instance.Model != null) Run();
        }
        void OnDisable() { if (MjScene.InstanceExists) MjScene.Instance.postInitEvent -= OnInit; }
        void OnInit(object s, MjStepArgs a) => Run();

        static double D(object o) => System.Convert.ToDouble(o, CultureInfo.InvariantCulture);
        static List<object> L(object o) => (List<object>)o;

        public void Run()
        {
            var m = MjScene.Instance.Model;
            var dump = (Dictionary<string, object>)MiniJson.Parse(modelDumpJson.text);
            var sb = new StringBuilder(); int bad = 0;
            void Check(string what, double unity, object train, double tol)
            {
                double t = D(train);
                if (System.Math.Abs(unity - t) > tol) { bad++; if (bad <= 60) sb.AppendLine($"MISMATCH {what}: unity={unity} train={t}"); }
            }
            int Id(MujocoLib.mjtObj type, string name, string kind)
            {
                int id = MujocoLib.mj_name2id(m, (int)type, name);
                if (id < 0) { bad++; if (bad <= 60) sb.AppendLine($"MISSING {kind} {name}"); }
                return id;
            }
            Check("nq", m->nq, dump["nq"], 0); Check("nv", m->nv, dump["nv"], 0); Check("nu", m->nu, dump["nu"], 0);
            Check("nbody", m->nbody, dump["nbody"], 0);
            Check("timestep", m->opt.timestep, dump["timestep"], 1e-9);
            Check("integrator", m->opt.integrator, dump["integrator"], 0);
            Check("iterations", m->opt.iterations, dump["iterations"], 0);
            Check("ls_iterations", m->opt.ls_iterations, dump["ls_iterations"], 0);
            Check("solver", m->opt.solver, dump["solver"], 0); Check("cone", m->opt.cone, dump["cone"], 0);
            Check("gravity.z", m->opt.gravity[2], L(dump["gravity"])[2], 1e-6);

            foreach (Dictionary<string, object> j in L(dump["joints"]))
            {
                string name = (string)j["name"];
                int id = Id(MujocoLib.mjtObj.mjOBJ_JOINT, name, "joint"); if (id < 0) continue;
                if (D(j["type"]) == 0) continue;   // free joint: nothing to compare
                var r = L(j["range"]); int dof = m->jnt_dofadr[id];
                Check($"{name}.range0", m->jnt_range[2 * id], r[0], 1e-5); Check($"{name}.range1", m->jnt_range[2 * id + 1], r[1], 1e-5);
                Check($"{name}.damping", m->dof_damping[dof], j["damping"], 1e-6);
                Check($"{name}.armature", m->dof_armature[dof], j["armature"], 1e-6);
                Check($"{name}.frictionloss", m->dof_frictionloss[dof], j["frictionloss"], 1e-6);
            }
            foreach (Dictionary<string, object> a in L(dump["actuators"]))
            {
                string name = (string)a["name"];
                int id = Id(MujocoLib.mjtObj.mjOBJ_ACTUATOR, name, "actuator"); if (id < 0) continue;
                Check($"{name}.kp", m->actuator_gainprm[id * MujocoLib.mjNGAIN], a["kp"], 1e-6);
                Check($"{name}.kp_bias", -m->actuator_biasprm[id * MujocoLib.mjNBIAS + 1], a["kp"], 1e-6);
                Check($"{name}.kv", -m->actuator_biasprm[id * MujocoLib.mjNBIAS + 2], a["kv"], 1e-6);
                var cr = L(a["ctrlrange"]); var fr = L(a["forcerange"]);
                Check($"{name}.ctrl0", m->actuator_ctrlrange[2 * id], cr[0], 1e-5); Check($"{name}.ctrl1", m->actuator_ctrlrange[2 * id + 1], cr[1], 1e-5);
                Check($"{name}.force0", m->actuator_forcerange[2 * id], fr[0], 1e-6); Check($"{name}.force1", m->actuator_forcerange[2 * id + 1], fr[1], 1e-6);
            }
            foreach (Dictionary<string, object> b in L(dump["bodies"]))
            {
                string name = (string)b["name"]; if (name == "world") continue;
                int id = Id(MujocoLib.mjtObj.mjOBJ_BODY, name, "body"); if (id < 0) continue;
                Check($"{name}.mass", m->body_mass[id], b["mass"], 1e-5);
                var ip = L(b["ipos"]);
                for (int k = 0; k < 3; k++) Check($"{name}.ipos{k}", m->body_ipos[3 * id + k], ip[k], 1e-5);
            }
            foreach (Dictionary<string, object> g in L(dump["geoms"]))
            {
                string name = (string)g["name"];
                int id = Id(MujocoLib.mjtObj.mjOBJ_GEOM, name, "geom"); if (id < 0) continue;
                Check($"{name}.contype", m->geom_contype[id], g["contype"], 0);
                Check($"{name}.conaffinity", m->geom_conaffinity[id], g["conaffinity"], 0);
                Check($"{name}.condim", m->geom_condim[id], g["condim"], 0);
                Check($"{name}.friction", m->geom_friction[3 * id], L(g["friction"])[0], 1e-6);
                if (D(g["type"]) != 7 && D(g["type"]) != 0)   // skip mesh and plane sizes
                    for (int k = 0; k < 3; k++) Check($"{name}.size{k}", m->geom_size[3 * id + k], L(g["size"])[k], 1e-5);
            }
            Mismatches = bad; Passed = bad == 0; HasRun = true;
            Report = Passed ? $"ModelDumpCheck PASS (nq={m->nq} nv={m->nv} nu={m->nu} nbody={m->nbody} ngeom={m->ngeom})"
                            : $"ModelDumpCheck FAIL: {bad} mismatches (unity nq={m->nq} nv={m->nv} nu={m->nu} nbody={m->nbody})\n{sb}";
            if (Passed) Debug.Log(Report); else Debug.LogError(Report);
        }
    }

    /// <summary>Tiny JSON reader (objects, arrays, numbers as long/double, strings, bools, null). Enough for the dump file.</summary>
    public static class MiniJson
    {
        public static object Parse(string s) { int i = 0; return Value(s, ref i); }
        static void Ws(string s, ref int i) { while (i < s.Length && char.IsWhiteSpace(s[i])) i++; }
        static bool At(string s, int i, string lit) => string.CompareOrdinal(s, i, lit, 0, lit.Length) == 0;
        static object Value(string s, ref int i)
        {
            Ws(s, ref i);
            char c = s[i];
            if (c == '{')
            {
                i++; var d = new Dictionary<string, object>(); Ws(s, ref i);
                if (s[i] == '}') { i++; return d; }
                while (true) { Ws(s, ref i); string k = Str(s, ref i); Ws(s, ref i); i++; d[k] = Value(s, ref i); Ws(s, ref i); if (s[i++] == '}') return d; }
            }
            if (c == '[')
            {
                i++; var l = new List<object>(); Ws(s, ref i);
                if (s[i] == ']') { i++; return l; }
                while (true) { l.Add(Value(s, ref i)); Ws(s, ref i); if (s[i++] == ']') return l; }
            }
            if (c == '"') return Str(s, ref i);
            if (At(s, i, "true")) { i += 4; return true; }
            if (At(s, i, "false")) { i += 5; return false; }
            if (At(s, i, "null")) { i += 4; return null; }
            int st = i; while (i < s.Length && "+-0123456789.eE".IndexOf(s[i]) >= 0) i++;
            string num = s.Substring(st, i - st);
            if (num.IndexOfAny(new[] { '.', 'e', 'E' }) < 0 && long.TryParse(num, out long lv)) return lv;
            return double.Parse(num, CultureInfo.InvariantCulture);
        }
        static string Str(string s, ref int i)
        {
            var sb = new StringBuilder(); i++;
            while (s[i] != '"') { if (s[i] == '\\') i++; sb.Append(s[i++]); }
            i++; return sb.ToString();
        }
    }
}
