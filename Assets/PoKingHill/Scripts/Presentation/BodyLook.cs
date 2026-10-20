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
    ///   surface    (Effects) with fighterMaterial (FX/Fighter.mat, PoKingHill/Fighter) every opaque fighter material,
    ///              G1 parts and the humans' skins, is rebuilt on that shader, which adds:
    ///              rim     a fresnel rim in the seat colour, turning a pulsing red as the fighter's capture point comes within 12 cm of
    ///                      the edge (Biometrics.rim)
    ///              wear    scratches and grime on a G1 part from the pushing force it took this match (contacts with
    ///                      other fighters, and the ground for anything but the feet); cleared when a match starts
    ///              wet     darker and glossy after the sea, drying over wetSeconds
    ///              flash   a short hot glow on the parts in contact with another fighter when a hit lands
    ///                      (MatchEvents.Impact), not while they merely lean on each other
    ///              The humans' single skin takes the highest value of its bodies (no scratches on skin).
    /// Reads mjModel/mjData only; per-geom material instances are made once at start.
    /// </summary>
    public unsafe class BodyLook : MonoBehaviour
    {
        public DemoDirector director;
        [Tooltip("Material for the intent lines (URP Unlit, FX/IntentLine.mat). A player build strips shaders no asset references, so Shader.Find alone returned null there and Init never finished: no team paint in builds.")]
        public Material lineMaterial;
        public float overlayHz = 15f, fatigueTau = 4f;
        [Tooltip("Fighter surface (FX/Fighter.mat, shader PoKingHill/Fighter): rim, wear, wet, hit flash. Null keeps the imported materials.")]
        public Material fighterMaterial;
        public Biometrics bio;                    // optional: the rim turns red near the edge
        public float surfaceHz = 30f, wetSeconds = 20f, flashSeconds = 0.3f;
        [Tooltip("Pushing impulse (N s) that wears a part fully")] public float wearImpulse = 2500f;
        [Tooltip("Impact strength (MatchEvents.Impact, 0..1) below which a hit does not flash")] public float flashFrom = 0.35f;

        class Part { public Renderer r; public int body, robot; public Color baseColor, heat; public bool accent; }
        class Skin { public Renderer r; public int robot; }
        readonly List<Skin> _skins = new();
        float[] _wear = Array.Empty<float>(), _wet = Array.Empty<float>(), _flash = Array.Empty<float>(), _danger = Array.Empty<float>();
        float[] _robotWet = Array.Empty<float>(), _robotFlash = Array.Empty<float>(), _hit = Array.Empty<float>(); bool[] _isFoot = Array.Empty<bool>(); float _nextSurface;
        readonly List<Part> _parts = new(); MaterialPropertyBlock _mpb;
        float[] _bodyHeat; double[] _work; int[] _jointBody, _actJoint; int[] _robotOfBody;
        PolicyRunner[] _runners = Array.Empty<PolicyRunner>(); readonly List<int> _colliderGeoms = new();
        LineRenderer[] _toTarget = Array.Empty<LineRenderer>(), _command = Array.Empty<LineRenderer>();
        Material _lineMat; float _next; bool _ready;
        static readonly int BaseColor = Shader.PropertyToID("_BaseColor"), Emission = Shader.PropertyToID("_EmissionColor"),
                            Metallic = Shader.PropertyToID("_Metallic"), Smooth = Shader.PropertyToID("_Smoothness"),
                            RimColor = Shader.PropertyToID("_RimColor"), Wear = Shader.PropertyToID("_Wear"), Wet = Shader.PropertyToID("_Wet"), Flash = Shader.PropertyToID("_Flash");

        void OnEnable()
        {
            MatchEvents.RoundStarted += Repaint; MatchEvents.RoundStarted += Dry; MatchEvents.StateChanged += OnState; MatchEvents.Impact += OnImpact;
        }
        void OnDisable()
        {
            MatchEvents.RoundStarted -= Repaint; MatchEvents.RoundStarted -= Dry; MatchEvents.StateChanged -= OnState; MatchEvents.Impact -= OnImpact;
        }
        // a new match starts with clean bodies; every round places them afresh on the summit, dry
        void OnState(MatchState s) { if (s == MatchState.Menu || s == MatchState.Intro) { Array.Clear(_wear, 0, _wear.Length); Repaint(); } }
        void Dry() { Array.Clear(_wet, 0, _wet.Length); Array.Clear(_flash, 0, _flash.Length); }
        // a hit landed on this fighter: the parts touching another fighter flash at the next surface update
        void OnImpact(Vector3 p, float strength, PolicyRunner r)
        {
            int k = Array.IndexOf(_runners, r); if (k < 0 || k >= _hit.Length || strength < flashFrom) return;
            _hit[k] = Mathf.Max(_hit[k], strength);
        }

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
            foreach (var g in FindObjectsByType<MjGeom>(FindObjectsInactive.Exclude))
            {
                int id = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, g.gameObject.name); if (id < 0) continue;
                int body = m->geom_bodyid[id], robot = _robotOfBody[body]; if (robot < 0) continue;
                if (m->geom_contype[id] != 0 || m->geom_conaffinity[id] != 0) _colliderGeoms.Add(id);
                var r = g.GetComponent<Renderer>(); if (r == null || !r.enabled || r.sharedMaterial == null) continue;
                var mat = Surface(r.sharedMaterial); if (mat != null) r.sharedMaterial = mat; else mat = r.material;
                mat.EnableKeyword("_EMISSION"); mat.globalIlluminationFlags = MaterialGlobalIlluminationFlags.None;
                string bn = Name(m, MujocoLib.mjtObj.mjOBJ_BODY, body);
                _parts.Add(new Part { r = r, body = body, robot = robot, baseColor = mat.HasProperty(BaseColor) ? mat.GetColor(BaseColor) : Color.gray,
                                      accent = bn.Contains("torso") || bn.EndsWith("pelvis") || bn.Contains("head") });
            }
            // the humans' skins (SkinFollower): their opaque materials move to the fighter shader too
            foreach (var sf in FindObjectsByType<SkinFollower>(FindObjectsInactive.Exclude))
            {
                int robot = Array.FindIndex(_runners, x => x.robotPrefix == sf.robotPrefix); if (robot < 0 || sf.skinRoot == null) continue;
                foreach (var r in sf.skinRoot.GetComponentsInChildren<Renderer>())
                {
                    var mats = r.sharedMaterials; bool any = false;
                    for (int k = 0; k < mats.Length; k++) { var conv = Surface(mats[k]); if (conv != null) { mats[k] = conv; any = true; } }
                    if (any) { r.sharedMaterials = mats; _skins.Add(new Skin { r = r, robot = robot }); }
                }
            }
            int nb = (int)m->nbody;
            _wear = new float[nb]; _wet = new float[nb]; _flash = new float[nb]; _isFoot = new bool[nb];
            _danger = new float[_runners.Length]; _robotWet = new float[_runners.Length]; _robotFlash = new float[_runners.Length]; _hit = new float[_runners.Length];
            for (int b = 0; b < nb; b++) _isFoot[b] = Name(m, MujocoLib.mjtObj.mjOBJ_BODY, b).EndsWith("ankle_roll_link");
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

        /// <summary>The same look on the fighter shader; null when there is no fighter material, or the source is
        /// transparent or alpha-tested (hair keeps its own material). Reads URP Lit and glTF (glTFast) property names.</summary>
        Material Surface(Material src)
        {
            if (fighterMaterial == null || src == null) return null;
            if (src.shader == fighterMaterial.shader) return src;
            if (src.renderQueue >= 2450 || src.IsKeywordEnabled("_ALPHATEST_ON") || src.IsKeywordEnabled("_SURFACE_TYPE_TRANSPARENT")
                || (src.HasProperty("_Surface") && src.GetFloat("_Surface") > 0.5f)) return null;
            var m = new Material(fighterMaterial) { name = src.name + " (fighter)" };
            string tex = src.HasProperty("_BaseMap") ? "_BaseMap" : src.HasProperty("baseColorTexture") ? "baseColorTexture" : null;
            if (tex != null && src.GetTexture(tex) != null)
            { m.SetTexture("_BaseMap", src.GetTexture(tex)); m.SetTextureScale("_BaseMap", src.GetTextureScale(tex)); m.SetTextureOffset("_BaseMap", src.GetTextureOffset(tex)); }
            if (src.HasProperty("_BaseColor")) m.SetColor(BaseColor, src.GetColor("_BaseColor")); else if (src.HasProperty("baseColorFactor")) m.SetColor(BaseColor, src.GetColor("baseColorFactor"));
            string nrm = src.HasProperty("_BumpMap") ? "_BumpMap" : src.HasProperty("normalTexture") ? "normalTexture" : null;
            if (nrm != null && src.GetTexture(nrm) != null)
            {
                m.SetTexture("_BumpMap", src.GetTexture(nrm)); m.EnableKeyword("_NORMALMAP");
                if (src.HasProperty("_BumpScale")) m.SetFloat("_BumpScale", src.GetFloat("_BumpScale")); else if (src.HasProperty("normalTexture_scale")) m.SetFloat("_BumpScale", src.GetFloat("normalTexture_scale"));
            }
            if (src.HasProperty("_Metallic")) m.SetFloat(Metallic, src.GetFloat("_Metallic")); else if (src.HasProperty("metallicFactor")) m.SetFloat(Metallic, src.GetFloat("metallicFactor"));
            if (src.HasProperty("_Smoothness")) m.SetFloat(Smooth, src.GetFloat("_Smoothness")); else if (src.HasProperty("roughnessFactor")) m.SetFloat(Smooth, 1f - src.GetFloat("roughnessFactor"));
            m.SetFloat("_Cull", src.HasProperty("_Cull") ? src.GetFloat("_Cull") : src.HasProperty("_CullMode") ? src.GetFloat("_CullMode") : 2f);
            return m;
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
            if (Time.unscaledTime >= _nextSurface && director.statsRounds == 0)
            {
                float sdt = Mathf.Clamp(Time.unscaledTime - _nextSurface + 1f / surfaceHz, 0f, 0.2f); _nextSurface = Time.unscaledTime + 1f / surfaceHz;
                SurfaceLayers(m, d, sdt);
            }
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
                float h = glow ? Mathf.Clamp01(_bodyHeat[p.body]) : 0f;
                p.heat = hot * (h * h * 3f);
                if (director.statsRounds > 0) { p.r.GetPropertyBlock(_mpb); _mpb.SetColor(Emission, p.heat); p.r.SetPropertyBlock(_mpb); }   // SurfaceLayers is off in the gates
            }
        }

        // Rim, wear, wet and flash, written with the overlay glow into each renderer's property block.
        void SurfaceLayers(MujocoLib.mjModel_* m, MujocoLib.mjData_* d, float dt)
        {
            bool fx = GameSettings.Effects;
            // contacts: the force pushing on a part (from another fighter, or the ground for anything but a foot)
            double* f = stackalloc double[6];
            for (int i = 0; i < d->ncon; i++)
            {
                var con = d->contact + i; int b1 = m->geom_bodyid[con->geom1], b2 = m->geom_bodyid[con->geom2];
                int r1 = _robotOfBody[b1], r2 = _robotOfBody[b2];
                bool fighters = r1 >= 0 && r2 >= 0 && r1 != r2, ground = (r1 < 0) != (r2 < 0) && (b1 == 0 || b2 == 0);
                if (!fighters && !ground) continue;
                MujocoLib.mj_contactForce(m, d, i, f); float fn = (float)Math.Abs(f[0]);
                for (int side = 0; side < 2; side++)
                {
                    int b = side == 0 ? b1 : b2; if (_robotOfBody[b] < 0 || (ground && _isFoot[b])) continue;
                    _wear[b] = Mathf.Min(1f, _wear[b] + fn * dt / (ground ? 2f * wearImpulse : wearImpulse));
                    if (fighters && fn > 50f) _flash[b] = Mathf.Max(_flash[b], _hit[_robotOfBody[b]]);
                }
            }
            Array.Clear(_hit, 0, _hit.Length);
            var sea = director.sea; float dryRate = dt / Mathf.Max(0.1f, wetSeconds), fade = dt / Mathf.Max(0.02f, flashSeconds);
            Array.Clear(_robotWet, 0, _robotWet.Length); Array.Clear(_robotFlash, 0, _robotFlash.Length);
            for (int b = 1; b < _wet.Length; b++)
            {
                int k = _robotOfBody[b]; if (k < 0) continue;
                _wet[b] = sea != null && d->xipos[3 * b + 2] < sea.Level ? 1f : Mathf.Max(0f, _wet[b] - dryRate);
                _flash[b] = Mathf.Max(0f, _flash[b] - fade);
                _robotWet[k] = Mathf.Max(_robotWet[k], _wet[b]); _robotFlash[k] = Mathf.Max(_robotFlash[k], _flash[b]);
            }
            for (int k = 0; k < _runners.Length; k++)
            {
                int seat = SeatOf(k);
                var v = bio != null && seat >= 0 && seat < director.Playing && director.State == MatchState.Fight && !director.IsOut(seat) ? bio.Of(_runners[k]) : null;
                _danger[k] = v != null ? Mathf.Clamp01((0.12f - v.rim) / 0.25f) : 0f;      // from 12 cm inside the rim (as MatchFx's danger light) to 13 cm past it
            }
            float pulse = 0.65f + 0.35f * Mathf.Sin(Time.unscaledTime * 14f);
            foreach (var p in _parts)
            {
                p.r.GetPropertyBlock(_mpb);
                _mpb.SetColor(Emission, p.heat);
                _mpb.SetColor(RimColor, fx ? Rim(p.robot, 0.45f, 1.6f * pulse) : Color.clear);
                _mpb.SetFloat(Wear, fx ? _wear[p.body] : 0f); _mpb.SetFloat(Wet, fx ? _wet[p.body] : 0f); _mpb.SetFloat(Flash, fx ? _flash[p.body] : 0f);
                p.r.SetPropertyBlock(_mpb);
            }
            foreach (var s in _skins)
            {
                s.r.GetPropertyBlock(_mpb);
                _mpb.SetColor(RimColor, fx ? Rim(s.robot, 0.15f, 1.2f * pulse) : Color.clear);
                _mpb.SetFloat(Wet, fx ? _robotWet[s.robot] : 0f); _mpb.SetFloat(Flash, fx ? 0.35f * _robotFlash[s.robot] : 0f);
                s.r.SetPropertyBlock(_mpb);
            }
        }

        // seat colour at rest, pulsing red as the capture point nears the edge (alpha = rim strength)
        Color Rim(int robot, float calm, float alarm)
        {
            int seat = SeatOf(robot); bool seated = seat >= 0 && seat < director.Playing && !director.InMenu;
            Color team = seated ? DemoDirector.SeatColors[seat % DemoDirector.SeatColors.Length] : new Color(0.75f, 0.75f, 0.78f);
            return Color.Lerp(new Color(team.r, team.g, team.b, calm), new Color(1f, 0.12f, 0.08f, alarm), _danger[robot]);
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
