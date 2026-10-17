using System;
using System.IO;
using System.Linq;
using Unity.Cinemachine;
using Unity.Cinemachine.TargetTracking;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.UIElements;

namespace PoKingHill.EditorTools
{
    /// <summary>
    /// Dresses the game scene that ParityBatch.BuildDuelDemo imports: presentation components, the UI Toolkit HUD,
    /// the Cinemachine broadcast rig, particle systems, decal pool, lights and post-process volumes. Every object is a
    /// plain scene object afterwards, so it can be tuned by hand in the editor. Shared assets (panel settings, theme,
    /// FX textures and materials, volume profiles) are created once under Assets/PoKingHill/{UI,FX} and reused, so a
    /// rebuild does not churn their GUIDs. Nothing here adds a PhysX component.
    /// </summary>
    public static class GameSceneBuilder
    {
        const string UiDir = "Assets/PoKingHill/UI", FxDir = "Assets/PoKingHill/FX";

        public static void Dress(GameObject rig, DemoDirector director, Sea sea, ImpactSynth synth, Camera cam, MatchCamera classic, ProjectilePool pool)
        {
            Directory.CreateDirectory(UiDir); Directory.CreateDirectory(FxDir);
            var log = rig.AddComponent<MatchLog>(); log.director = director;
            var replay = rig.AddComponent<ReplaySystem>(); replay.director = director;
            rig.AddComponent<PuppetControl>().director = director;
            var odds = rig.AddComponent<WinOdds>(); odds.director = director;
            rig.AddComponent<Commentator>().director = director;
            var hazards = rig.AddComponent<Hazards>(); hazards.director = director; hazards.pool = pool;
            var chat = rig.AddComponent<ChatVotes>(); chat.hazards = hazards;
            rig.AddComponent<BodyLook>().director = director;

            BuildHud(director, odds, replay, log, hazards, chat);
            BuildCameraRig(rig, director, cam, classic);
            BuildFx(rig, director, sea);
            EnsureDecalFeature();
        }

        // ------------------------------------------------------------------ HUD
        static void BuildHud(DemoDirector director, WinOdds odds, ReplaySystem replay, MatchLog log, Hazards hazards, ChatVotes chat)
        {
            string themePath = UiDir + "/HudTheme.tss";
            if (!File.Exists(themePath)) { File.WriteAllText(themePath, "@import url(\"unity-theme://default\");\n"); AssetDatabase.ImportAsset(themePath); }
            string psPath = UiDir + "/HudPanel.asset";
            var ps = AssetDatabase.LoadAssetAtPath<PanelSettings>(psPath);
            if (ps == null) { ps = ScriptableObject.CreateInstance<PanelSettings>(); AssetDatabase.CreateAsset(ps, psPath); }
            ps.themeStyleSheet = AssetDatabase.LoadAssetAtPath<ThemeStyleSheet>(themePath);
            ps.scaleMode = PanelScaleMode.ScaleWithScreenSize; ps.referenceResolution = new Vector2Int(1080, 1920);
            ps.screenMatchMode = PanelScreenMatchMode.MatchWidthOrHeight; ps.match = 1f;    // portrait: scale by height
            EditorUtility.SetDirty(ps);

            var go = new GameObject("HUD (UI Toolkit)");
            var doc = go.AddComponent<UIDocument>(); doc.panelSettings = ps;
            var hud = go.AddComponent<HudView>();
            hud.director = director; hud.odds = odds; hud.replay = replay; hud.log = log; hud.hazards = hazards; hud.chat = chat;
            hud.style = AssetDatabase.LoadAssetAtPath<StyleSheet>(UiDir + "/Hud.uss");
            if (hud.style == null) throw new Exception(UiDir + "/Hud.uss missing");
        }

        // ------------------------------------------------------------------ cameras
        static void BuildCameraRig(GameObject rig, DemoDirector director, Camera cam, MatchCamera classic)
        {
            var brain = cam.gameObject.AddComponent<CinemachineBrain>();
            brain.IgnoreTimeScale = true; brain.UpdateMethod = CinemachineBrain.UpdateMethods.LateUpdate;
            cam.GetUniversalAdditionalCameraData().renderPostProcessing = true;

            var root = new GameObject("Broadcast cameras");
            var anchor = new GameObject("Summit anchor").transform; anchor.SetParent(root.transform); anchor.position = new Vector3(0f, 0.6f, 0f);
            var group = new GameObject("Fighters still in (target group)").AddComponent<CinemachineTargetGroup>(); group.transform.SetParent(root.transform);

            CinemachineCamera V(string name, float fov)
            {
                var go = new GameObject(name); go.transform.SetParent(root.transform);
                var c = go.AddComponent<CinemachineCamera>(); c.Priority = 10; c.Lens.FieldOfView = fov; c.Lens.FarClipPlane = 2000f;
                go.AddComponent<CinemachineImpulseListener>();
                return c;
            }
            var summit = V("Summit (group framing)", 38f);
            summit.Target.TrackingTarget = group.transform; summit.Target.LookAtTarget = group.transform;
            var sf = summit.gameObject.AddComponent<CinemachineFollow>(); sf.FollowOffset = new Vector3(0f, 2.6f, -6.2f); sf.TrackerSettings.BindingMode = Unity.Cinemachine.TargetTracking.BindingMode.WorldSpace;
            summit.gameObject.AddComponent<CinemachineRotationComposer>();
            var gf = summit.gameObject.AddComponent<CinemachineGroupFraming>(); gf.FramingSize = 0.7f;

            var faller = V("Faller at the water (cut)", 45f);
            var ff = faller.gameObject.AddComponent<CinemachineFollow>(); ff.FollowOffset = new Vector3(0f, 1.6f, -3.2f); ff.TrackerSettings.BindingMode = Unity.Cinemachine.TargetTracking.BindingMode.WorldSpace;
            faller.gameObject.AddComponent<CinemachineRotationComposer>();

            CinemachineCamera Orbit(string name, float fov, float radius, float pitch)
            {
                var c = V(name, fov);
                var o = c.gameObject.AddComponent<CinemachineOrbitalFollow>(); o.OrbitStyle = CinemachineOrbitalFollow.OrbitStyles.Sphere; o.Radius = radius;
                o.VerticalAxis.Value = pitch; o.TargetOffset = new Vector3(0f, 0.2f, 0f);
                c.gameObject.AddComponent<CinemachineRotationComposer>();
                return c;
            }
            var winner = Orbit("Winner orbit", 40f, 3.2f, 12f);
            var replay = Orbit("Replay orbit", 42f, 2.6f, 10f);
            var menu = Orbit("Menu orbit", 40f, 7.5f, 16f); menu.Target.TrackingTarget = anchor; menu.Target.LookAtTarget = anchor;

            var impulse = rig.AddComponent<CinemachineImpulseSource>();
            impulse.ImpulseDefinition.ImpulseShape = CinemachineImpulseDefinition.ImpulseShapes.Bump; impulse.ImpulseDefinition.ImpulseDuration = 0.22f;

            var bc = rig.AddComponent<BroadcastCamera>();
            bc.director = director; bc.brain = brain; bc.classic = classic; bc.summit = summit; bc.faller = faller; bc.winner = winner; bc.replay = replay; bc.menu = menu;
            bc.group = group; bc.impulse = impulse; bc.summitAnchor = anchor;
        }

        // ------------------------------------------------------------------ effects
        static void BuildFx(GameObject rig, DemoDirector director, Sea sea)
        {
            var fx = rig.AddComponent<MatchFx>(); fx.director = director;
            var root = new GameObject("Effects (render only)").transform;
            var soft = Tex("fx_soft", (x, y) => { float r = Mathf.Clamp01(1f - new Vector2(x - 0.5f, y - 0.5f).magnitude * 2f); return new Color(1, 1, 1, r * r); });
            var ring = Tex("fx_ring", (x, y) => { float r = new Vector2(x - 0.5f, y - 0.5f).magnitude * 2f; float a = Mathf.Clamp01(1f - Mathf.Abs(r - 0.82f) / 0.12f); return new Color(1, 1, 1, a * a); });
            var scuff = Tex("fx_scuff", (x, y) =>
            {
                float streak = Mathf.PerlinNoise(x * 3f, y * 26f) * Mathf.PerlinNoise(x * 11f + 7f, y * 4f);
                float mask = Mathf.Clamp01(1f - new Vector2((x - 0.5f) * 1.2f, (y - 0.5f) * 2.4f).magnitude * 1.6f);
                return new Color(0.11f, 0.09f, 0.08f, Mathf.Clamp01(streak * 2.2f * mask));
            });
            var alpha = Mat("fx_particles_alpha", "Universal Render Pipeline/Particles/Unlit", soft, false);
            var add = Mat("fx_particles_add", "Universal Render Pipeline/Particles/Unlit", soft, true);
            var ringMat = Mat("fx_ripple", "Universal Render Pipeline/Particles/Unlit", ring, false);

            fx.dust = Particles(root, "Dust", alpha, 1.2f, new Vector2(0.25f, 0.6f), new Color(0.55f, 0.50f, 0.45f, 0.55f), -0.05f, 600);
            fx.sparks = Particles(root, "Sparks", add, 0.35f, new Vector2(0.025f, 0.05f), new Color(1f, 0.75f, 0.35f, 1f), 1f, 300);
            fx.sparks.GetComponent<ParticleSystemRenderer>().renderMode = ParticleSystemRenderMode.Stretch;
            fx.splash = Particles(root, "Splash", alpha, 0.9f, new Vector2(0.08f, 0.22f), new Color(0.85f, 0.93f, 1f, 0.8f), 1f, 600);
            fx.ripple = Particles(root, "Ripple", ringMat, 1.6f, new Vector2(0.4f, 0.4f), new Color(0.9f, 0.97f, 1f, 0.7f), 0f, 40);
            var rr = fx.ripple.GetComponent<ParticleSystemRenderer>(); rr.alignment = ParticleSystemRenderSpace.World;
            var rmain = fx.ripple.main; rmain.startRotation3D = true; rmain.startRotationX = Mathf.PI / 2; rmain.startSpeed = 0;
            var rsize = fx.ripple.sizeOverLifetime; rsize.enabled = true; rsize.size = new ParticleSystem.MinMaxCurve(8f, AnimationCurve.EaseInOut(0, 0.1f, 1, 1));
            fx.confetti = Particles(root, "Confetti", alpha, 3f, new Vector2(0.05f, 0.1f), Color.white, 0.35f, 400);
            var cm = fx.confetti.main; cm.startColor = new ParticleSystem.MinMaxGradient(new Color(1f, 0.36f, 0.3f), new Color(0.3f, 0.65f, 1f)); cm.startSpeed = new ParticleSystem.MinMaxCurve(2f, 5f);
            var ce = fx.confetti.emission; ce.enabled = true; ce.SetBursts(new[] { new ParticleSystem.Burst(0f, 160) });
            var cs = fx.confetti.shape; cs.enabled = true; cs.shapeType = ParticleSystemShapeType.Cone; cs.angle = 35f; cs.radius = 0.2f; cs.rotation = new Vector3(-90f, 0, 0);

            // scuff decals: a pool of URP decal projectors, faded by MatchFx
            var decalShader = Shader.Find("Shader Graphs/Decal");
            if (decalShader != null)
            {
                var dm = AssetDatabase.LoadAssetAtPath<Material>(FxDir + "/fx_scuff_decal.mat");
                if (dm == null) { dm = new Material(decalShader); AssetDatabase.CreateAsset(dm, FxDir + "/fx_scuff_decal.mat"); }
                dm.SetTexture("Base_Map", scuff);
                var pool = new GameObject("Scuff decals").transform; pool.SetParent(root);
                fx.scuffs = Enumerable.Range(0, 24).Select(i =>
                {
                    var go = new GameObject("scuff " + i); go.transform.SetParent(pool);
                    var d = go.AddComponent<DecalProjector>(); d.material = dm; d.size = new Vector3(0.4f, 0.2f, 0.6f); d.fadeFactor = 0; d.pivot = Vector3.zero;
                    return d;
                }).ToArray();
            }
            else Debug.LogWarning("[GameSceneBuilder] URP decal shader not found: scuff decals skipped");

            // post-process: a calm global grade, plus a 'hit' volume whose weight MatchFx pulses on impacts
            var grade = Profile("PostGrade", p =>
            {
                p.Add<Bloom>(true).intensity.Override(0.35f); var v = p.Add<Vignette>(true); v.intensity.Override(0.22f); v.smoothness.Override(0.45f);
                var ca = p.Add<ColorAdjustments>(true); ca.contrast.Override(8f); ca.saturation.Override(6f);
            });
            var hit = Profile("PostHit", p => { p.Add<ChromaticAberration>(true).intensity.Override(0.65f); p.Add<LensDistortion>(true).intensity.Override(-0.22f); });
            var gv = new GameObject("Post-process (grade)").AddComponent<Volume>(); gv.isGlobal = true; gv.sharedProfile = grade; gv.priority = 0;
            var hv = new GameObject("Post-process (hit punch)").AddComponent<Volume>(); hv.isGlobal = true; hv.sharedProfile = hit; hv.priority = 1; hv.weight = 0;
            fx.hitVolume = hv;

            // lights: the imported directional light is the sun; a warm spot follows the king
            fx.sun = UnityEngine.Object.FindObjectsByType<Light>(FindObjectsInactive.Include).FirstOrDefault(l => l.type == LightType.Directional);
            if (fx.sun != null) fx.sun.shadows = LightShadows.Soft;
            var spot = new GameObject("King spotlight").AddComponent<Light>(); spot.type = LightType.Spot; spot.spotAngle = 30f; spot.range = 10f; spot.intensity = 8f;
            spot.color = new Color(1f, 0.88f, 0.62f); spot.shadows = LightShadows.None; spot.enabled = false;
            fx.kingSpot = spot;

            // water surface shader on the sea disc
            var waterShader = Shader.Find("PoKingHill/Water");
            if (waterShader != null && sea != null && sea.waterVisual != null)
            {
                var wm = AssetDatabase.LoadAssetAtPath<Material>(FxDir + "/Water.mat");
                if (wm == null) { wm = new Material(waterShader); AssetDatabase.CreateAsset(wm, FxDir + "/Water.mat"); }
                sea.waterVisual.GetComponent<MeshRenderer>().sharedMaterial = wm; fx.water = wm;
            }
        }

        static ParticleSystem Particles(Transform parent, string name, Material mat, float life, Vector2 size, Color color, float gravity, int max)
        {
            var go = new GameObject(name); go.transform.SetParent(parent);
            var ps = go.AddComponent<ParticleSystem>(); ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            var main = ps.main; main.playOnAwake = false; main.loop = false; main.duration = 1f; main.startLifetime = new ParticleSystem.MinMaxCurve(life * 0.6f, life);
            main.startSize = new ParticleSystem.MinMaxCurve(size.x, size.y); main.startColor = color; main.gravityModifier = gravity; main.maxParticles = max;
            main.simulationSpace = ParticleSystemSimulationSpace.World; main.startSpeed = 0;
            var em = ps.emission; em.rateOverTime = 0;
            var shape = ps.shape; shape.enabled = true; shape.shapeType = ParticleSystemShapeType.Sphere; shape.radius = 0.08f;
            var col = ps.colorOverLifetime; col.enabled = true;
            var g = new Gradient(); g.SetKeys(new[] { new GradientColorKey(Color.white, 0), new GradientColorKey(Color.white, 1) }, new[] { new GradientAlphaKey(1, 0), new GradientAlphaKey(0.8f, 0.4f), new GradientAlphaKey(0, 1) });
            col.color = g;
            var limit = ps.limitVelocityOverLifetime; limit.enabled = true; limit.dampen = 0.12f; limit.limit = 8f;
            var r = go.GetComponent<ParticleSystemRenderer>(); r.sharedMaterial = mat; r.shadowCastingMode = ShadowCastingMode.Off; r.receiveShadows = false;
            return ps;
        }

        static Texture2D Tex(string name, Func<float, float, Color> f)
        {
            string path = $"{FxDir}/{name}.png";
            if (!File.Exists(path))
            {
                const int N = 128; var t = new Texture2D(N, N, TextureFormat.RGBA32, false);
                for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) t.SetPixel(x, y, f((x + 0.5f) / N, (y + 0.5f) / N));
                File.WriteAllBytes(path, t.EncodeToPNG()); UnityEngine.Object.DestroyImmediate(t);
                AssetDatabase.ImportAsset(path);
                var imp = (TextureImporter)AssetImporter.GetAtPath(path); imp.alphaIsTransparency = true; imp.wrapMode = TextureWrapMode.Clamp; imp.mipmapEnabled = true; imp.SaveAndReimport();
            }
            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        static Material Mat(string name, string shader, Texture2D tex, bool additive)
        {
            string path = $"{FxDir}/{name}.mat";
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m == null) { m = new Material(Shader.Find(shader)); AssetDatabase.CreateAsset(m, path); }
            m.SetTexture("_BaseMap", tex); m.SetFloat("_Surface", 1f); m.SetFloat("_Blend", additive ? 2f : 0f);
            m.SetOverrideTag("RenderType", "Transparent"); m.renderQueue = (int)RenderQueue.Transparent;
            m.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha); m.SetFloat("_DstBlend", additive ? (float)BlendMode.One : (float)BlendMode.OneMinusSrcAlpha); m.SetFloat("_ZWrite", 0f);
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT"); EditorUtility.SetDirty(m);
            return m;
        }

        static VolumeProfile Profile(string name, Action<VolumeProfile> fill)
        {
            string path = $"{FxDir}/{name}.asset";
            var p = AssetDatabase.LoadAssetAtPath<VolumeProfile>(path);
            if (p != null) return p;
            p = ScriptableObject.CreateInstance<VolumeProfile>(); AssetDatabase.CreateAsset(p, path);
            fill(p);
            foreach (var c in p.components) { c.hideFlags = HideFlags.HideInInspector | HideFlags.HideInHierarchy; AssetDatabase.AddObjectToAsset(c, p); }
            EditorUtility.SetDirty(p); AssetDatabase.SaveAssets();
            return p;
        }

        /// <summary>Adds URP's Decal renderer feature to the PC and Mobile renderers once (the way the renderer inspector does).</summary>
        static void EnsureDecalFeature()
        {
            foreach (var path in new[] { "Assets/Settings/PC_Renderer.asset", "Assets/Settings/Mobile_Renderer.asset" })
            {
                var data = AssetDatabase.LoadAssetAtPath<UniversalRendererData>(path);
                if (data == null || data.rendererFeatures.Any(f => f is DecalRendererFeature)) continue;
                var feat = ScriptableObject.CreateInstance<DecalRendererFeature>(); feat.name = "Decal";
                AssetDatabase.AddObjectToAsset(feat, data);
                AssetDatabase.TryGetGUIDAndLocalFileIdentifier(feat, out _, out long localId);
                var so = new SerializedObject(data);
                var feats = so.FindProperty("m_RendererFeatures"); var map = so.FindProperty("m_RendererFeatureMap");
                feats.arraySize++; feats.GetArrayElementAtIndex(feats.arraySize - 1).objectReferenceValue = feat;
                map.arraySize++; map.GetArrayElementAtIndex(map.arraySize - 1).longValue = localId;
                so.ApplyModifiedProperties(); EditorUtility.SetDirty(data);
                Debug.Log("[GameSceneBuilder] added the Decal renderer feature to " + path);
            }
            AssetDatabase.SaveAssets();
        }
    }
}
