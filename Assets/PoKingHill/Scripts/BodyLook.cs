using System;
using System.Collections.Generic;
using Mujoco;
using UnityEngine;
using UnityEngine.Rendering;

namespace PoKingHill
{
    /// <summary>
    /// How the robots look, render side only:
    ///   paint      graphite metal with the seat colour on torso, pelvis and head, so a fighter matches its HUD badge
    ///   overlays   (GameSettings.Overlay) Strain = actuator force over its limit as a glow per limb; Fatigue = the work
    ///              each joint has done recently (|torque x speed|, 4 s decay) as heat; Colliders = the MuJoCo collision
    ///              primitives as wireframes plus contact points with force arrows, to check the collision model
    ///   intent     (GameSettings.IntentArrows) a line to the opponent each brain is pointed at, and its commanded velocity
    /// Reads mjModel/mjData only; per-geom material instances are made once at start.
    /// </summary>
    public unsafe class BodyLook : MonoBehaviour
    {
        public DemoDirector director;
        [Tooltip("Material for the intent lines (URP Unlit, FX/IntentLine.mat). A player build strips shaders no asset references, so Shader.Find alone returned null there and Init never finished: no team paint in builds.")]
        public Material lineMaterial;
        public float overlayHz = 15f, fatigueTau = 4f;

        class Part { public Renderer r; public int body, robot; public Color baseColor; public bool accent; }
        readonly List<Part> _parts = new(); MaterialPropertyBlock _mpb;
        float[] _bodyHeat; double[] _work; int[] _jointBody, _actJoint; int[] _robotOfBody;
        PolicyRunner[] _runners = Array.Empty<PolicyRunner>(); readonly List<int> _colliderGeoms = new();
        LineRenderer[] _toTarget = Array.Empty<LineRenderer>(), _command = Array.Empty<LineRenderer>();
        Material _lineMat; float _next; bool _ready;
        static readonly int BaseColor = Shader.PropertyToID("_BaseColor"), Emission = Shader.PropertyToID("_EmissionColor"),
                            Metallic = Shader.PropertyToID("_Metallic"), Smooth = Shader.PropertyToID("_Smoothness");

        void OnEnable()
        {
            MatchEvents.RoundStarted += Repaint; MatchEvents.StateChanged += OnState;
        }
        void OnDisable()
        {
            MatchEvents.RoundStarted -= Repaint; MatchEvents.StateChanged -= OnState;
        }
        void OnState(MatchState s) { if (s == MatchState.Menu || s == MatchState.Intro) Repaint(); }

        void Init()
        {
            var m = MjScene.Instance.Model; _mpb = new MaterialPropertyBlock();
            _runners = (PolicyRunner[])director.fighters.Clone();     // fixed body order: the director reorders its own array at every launch
            _robotOfBody = new int[(int)m->nbody]; for (int b = 0; b < (int)m->nbody; b++) _robotOfBody[b] = -1;
            for (int b = 1; b < (int)m->nbody; b++)
            {
                string name = Name(m, MujocoLib.mjtObj.mjOBJ_BODY, b);
                for (int k = 0; k < _runners.Length; k++) if (name.StartsWith(_runners[k].robotPrefix)) { _robotOfBody[b] = k; break; }
            }
            foreach (var g in FindObjectsByType<MjGeom>(FindObjectsInactive.Exclude, FindObjectsSortMode.None))
            {
                int id = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, g.gameObject.name); if (id < 0) continue;
                int body = m->geom_bodyid[id], robot = _robotOfBody[body]; if (robot < 0) continue;
                if (m->geom_contype[id] != 0 || m->geom_conaffinity[id] != 0) _colliderGeoms.Add(id);
                var r = g.GetComponent<Renderer>(); if (r == null || !r.enabled || r.sharedMaterial == null) continue;
                var mat = r.material; mat.EnableKeyword("_EMISSION"); mat.globalIlluminationFlags = MaterialGlobalIlluminationFlags.None;
                string bn = Name(m, MujocoLib.mjtObj.mjOBJ_BODY, body);
                _parts.Add(new Part { r = r, body = body, robot = robot, baseColor = mat.HasProperty(BaseColor) ? mat.GetColor(BaseColor) : Color.gray,
                                      accent = bn.Contains("torso") || bn.EndsWith("pelvis") || bn.Contains("head") });
            }
            _bodyHeat = new float[(int)m->nbody]; _work = new double[(int)m->njnt];
            _jointBody = new int[(int)m->njnt]; for (int j = 0; j < (int)m->njnt; j++) _jointBody[j] = m->jnt_bodyid[j];
            _actJoint = new int[(int)m->nu]; for (int a = 0; a < (int)m->nu; a++) _actJoint[a] = m->actuator_trntype[a] == (int)MujocoLib.mjtTrn.mjTRN_JOINT ? m->actuator_trnid[2 * a] : -1;
            _lineMat = new Material(Shader.Find("Hidden/Internal-Colored")) { hideFlags = HideFlags.HideAndDontSave };
            _lineMat.SetInt("_ZTest", (int)CompareFunction.Always); _lineMat.SetInt("_Cull", 0); _lineMat.SetInt("_ZWrite", 0);
            _toTarget = new LineRenderer[_runners.Length]; _command = new LineRenderer[_runners.Length];
            for (int k = 0; k < _runners.Length; k++) { _toTarget[k] = Line("intent target " + k, 0.02f); _command[k] = Line("intent command " + k, 0.05f); }
            _ready = true; Repaint();
        }

        LineRenderer Line(string name, float width)
        {
            var go = new GameObject(name); go.transform.SetParent(transform, false);
            var lr = go.AddComponent<LineRenderer>(); lr.positionCount = 2; lr.widthMultiplier = width; lr.useWorldSpace = true;
            lr.material = lineMaterial != null ? new Material(lineMaterial) : new Material(Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Hidden/Internal-Colored"));
            lr.shadowCastingMode = ShadowCastingMode.Off; lr.enabled = false;
            return lr;
        }

        static string Name(MujocoLib.mjModel_* m, MujocoLib.mjtObj type, int id) =>
            System.Runtime.InteropServices.Marshal.PtrToStringAnsi((IntPtr)MujocoLib.mj_id2name(m, (int)type, id)) ?? "";

        int SeatOf(int robot) => Array.IndexOf(director.fighters, _runners[robot]);

        // Seat colours follow the bodies: the director reorders fighters[] at every launch.
        void Repaint()
        {
            if (!_ready) return;
            foreach (var p in _parts)
            {
                int seat = SeatOf(p.robot); bool seated = seat >= 0 && seat < director.Playing && !director.InMenu;
                Color team = seated ? DemoDirector.SeatColors[seat % DemoDirector.SeatColors.Length] : new Color(0.75f, 0.75f, 0.78f);
                Color c = p.accent ? team : Color.Lerp(p.baseColor, new Color(0.16f, 0.17f, 0.19f), 0.55f);
                p.r.GetPropertyBlock(_mpb); _mpb.SetColor(BaseColor, c); _mpb.SetFloat(Metallic, p.accent ? 0.2f : 0.65f); _mpb.SetFloat(Smooth, p.accent ? 0.45f : 0.6f);
                _mpb.SetColor(Emission, Color.black); p.r.SetPropertyBlock(_mpb);
            }
        }

        void Update()
        {
            if (director == null || !MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            if (!_ready) { if (director.State >= MatchState.Menu) Init(); return; }
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data;
            // fatigue integrates every frame (cheap): work = |actuator force x joint speed|
            float dt = Time.deltaTime, decay = Mathf.Exp(-dt / fatigueTau);
            for (int j = 0; j < _work.Length; j++) _work[j] *= decay;
            for (int a = 0; a < (int)m->nu; a++) { int j = _actJoint[a]; if (j < 0) continue; _work[j] += Math.Abs(d->actuator_force[a] * d->qvel[m->jnt_dofadr[j]]) * dt; }
            Intent(d);
            if (Time.unscaledTime < _next) return; _next = Time.unscaledTime + 1f / overlayHz;
            var mode = GameSettings.Overlay;
            Array.Clear(_bodyHeat, 0, _bodyHeat.Length);
            if (mode == Overlay.Strain)
                for (int a = 0; a < (int)m->nu; a++)
                {
                    int j = _actJoint[a]; if (j < 0) continue; double lim = Math.Max(1e-6, Math.Max(Math.Abs(m->actuator_forcerange[2 * a]), Math.Abs(m->actuator_forcerange[2 * a + 1])));
                    if (m->actuator_forcelimited[a] == 0) lim = 100;
                    int b = _jointBody[j]; _bodyHeat[b] = Mathf.Max(_bodyHeat[b], (float)(Math.Abs(d->actuator_force[a]) / lim));
                }
            else if (mode == Overlay.Fatigue)
                for (int j = 0; j < _work.Length; j++) { int b = _jointBody[j]; _bodyHeat[b] = Mathf.Max(_bodyHeat[b], (float)(_work[j] / 400.0)); }   // 400 J of recent work = fully "hot"
            bool glow = mode == Overlay.Strain || mode == Overlay.Fatigue;
            Color hot = mode == Overlay.Strain ? new Color(1f, 0.15f, 0.1f) : new Color(1f, 0.55f, 0.05f);
            foreach (var p in _parts)
            {
                p.r.GetPropertyBlock(_mpb);
                float h = glow ? Mathf.Clamp01(_bodyHeat[p.body]) : 0f;
                _mpb.SetColor(Emission, hot * (h * h * 3f)); p.r.SetPropertyBlock(_mpb);
            }
        }

        void Intent(MujocoLib.mjData_* d)
        {
            bool on = GameSettings.IntentArrows && director.State == MatchState.Fight;
            for (int k = 0; k < _runners.Length; k++)
            {
                var r = _runners[k]; int seat = SeatOf(k); bool show = on && seat >= 0 && seat < director.Playing && !director.IsOut(seat) && r.PelvisTransform != null;
                _toTarget[k].enabled = show && r.opponent != null && r.opponent.PelvisTransform != null; _command[k].enabled = show;
                if (!show) continue;
                Color c = DemoDirector.SeatColors[seat % DemoDirector.SeatColors.Length];
                Vector3 p = r.PelvisTransform.position + Vector3.up * 0.35f;
                if (_toTarget[k].enabled) { _toTarget[k].SetPosition(0, p); _toTarget[k].SetPosition(1, r.opponent.PelvisTransform.position + Vector3.up * 0.35f); _toTarget[k].material.color = c * 0.8f; }
                double* q = d->qpos + r.Map.RootQposAdr; double yaw = Math.Atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
                float cy = (float)Math.Cos(yaw), sy = (float)Math.Sin(yaw);
                Vector2 v = new(cy * r.command.x - sy * r.command.y, sy * r.command.x + cy * r.command.y);     // heading frame -> world (MuJoCo x, y)
                _command[k].SetPosition(0, p); _command[k].SetPosition(1, p + new Vector3(v.x, 0f, v.y) * 0.8f); _command[k].material.color = c;
            }
        }

        // Collision primitives (sphere, capsule, box) of every robot plus contact force arrows: one line mesh rebuilt each
        // frame into reused buffers and drawn in the scene pass (so it stays under the HUD), always on top of the bodies.
        readonly List<Vector3> _lv = new(4096); readonly List<Color32> _lc = new(4096); readonly List<int> _li = new(4096); Mesh _lines; Color32 _col;

        void LateUpdate()
        {
            if (!_ready || GameSettings.Overlay != Overlay.Colliders || !MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            _lv.Clear(); _lc.Clear(); _li.Clear();
            BuildColliderLines();
            if (_lines == null) { _lines = new Mesh { name = "collider overlay" }; _lines.MarkDynamic(); _lines.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32; }
            _lines.Clear(); _lines.SetVertices(_lv); _lines.SetColors(_lc); _lines.SetIndices(_li, MeshTopology.Lines, 0, false);
            _lines.bounds = new Bounds(Vector3.zero, Vector3.one * 200f);
            Graphics.RenderMesh(new RenderParams(_lineMat) { layer = 0 }, _lines, 0, Matrix4x4.identity);
        }

        void BuildColliderLines()
        {
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data;
            _col = new Color32(80, 255, 128, 230);
            foreach (int g in _colliderGeoms)
            {
                Vector3 c = U(d->geom_xpos[3 * g], d->geom_xpos[3 * g + 1], d->geom_xpos[3 * g + 2]); double* R = d->geom_xmat + 9 * g;
                Vector3 ax = U(R[0], R[3], R[6]), ay = U(R[1], R[4], R[7]), az = U(R[2], R[5], R[8]); double* s = m->geom_size + 3 * g;
                switch (m->geom_type[g])
                {
                    case (int)MujocoLib.mjtGeom.mjGEOM_SPHERE: Ring(c, ax, ay, (float)s[0]); Ring(c, ax, az, (float)s[0]); Ring(c, ay, az, (float)s[0]); break;
                    case (int)MujocoLib.mjtGeom.mjGEOM_CAPSULE:
                    {
                        float rad = (float)s[0], half = (float)s[1]; Vector3 a = c + az * half, b = c - az * half;
                        Ring(a, ax, ay, rad); Ring(b, ax, ay, rad);
                        Seg(a + ax * rad, b + ax * rad); Seg(a - ax * rad, b - ax * rad); Seg(a + ay * rad, b + ay * rad); Seg(a - ay * rad, b - ay * rad);
                        break;
                    }
                    case (int)MujocoLib.mjtGeom.mjGEOM_BOX:
                    {
                        Vector3 x = ax * (float)s[0], y = ay * (float)s[1], z = az * (float)s[2];
                        for (int i = -1; i <= 1; i += 2) for (int j = -1; j <= 1; j += 2) { Seg(c + i * x + j * y - z, c + i * x + j * y + z); Seg(c + i * x - y + j * z, c + i * x + y + j * z); Seg(c - x + i * y + j * z, c + x + i * y + j * z); }
                        break;
                    }
                }
            }
            _col = new Color32(255, 216, 50, 255);
            double* f = stackalloc double[6];
            for (int i = 0; i < d->ncon; i++)
            {
                MujocoLib.mj_contactForce(m, d, i, f);
                var con = d->contact + i; Vector3 p = U(con->pos[0], con->pos[1], con->pos[2]);
                Vector3 n = U(con->frame[0], con->frame[1], con->frame[2]);
                Seg(p, p + n * Mathf.Clamp((float)f[0] / 400f, 0.02f, 0.6f));
            }
        }

        static Vector3 U(double x, double y, double z) => new((float)x, (float)z, (float)y);
        void Seg(Vector3 a, Vector3 b) { _li.Add(_lv.Count); _lv.Add(a); _lc.Add(_col); _li.Add(_lv.Count); _lv.Add(b); _lc.Add(_col); }
        void Ring(Vector3 c, Vector3 u, Vector3 v, float r)
        {
            const int N = 16; Vector3 prev = c + u * r;
            for (int i = 1; i <= N; i++) { float t = i * Mathf.PI * 2 / N; Vector3 p = c + (u * Mathf.Cos(t) + v * Mathf.Sin(t)) * r; Seg(prev, p); prev = p; }
        }
    }
}
