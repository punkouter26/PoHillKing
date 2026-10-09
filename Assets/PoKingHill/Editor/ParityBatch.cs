using System;
using System.IO;
using System.Linq;
using Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoKingHill.EditorTools
{
    /// <summary>
    /// Editor-side authoring of the parity testbed scenes, runnable from the menu or headless:
    ///   Unity.exe -batchmode -projectPath . -executeMethod PoKingHill.EditorTools.ParityBatch.ImportAll -quit
    ///   Unity.exe -batchmode -projectPath . -executeMethod PoKingHill.EditorTools.ParityBatch.RunHold -kothScene flat_1p
    /// Import uses the plugin's own MJCF importer on training/assets/g1/scene_&lt;tag&gt;_unity.xml, so the Unity
    /// hierarchy (MjBody / MjGeom / MjHingeJoint / MjActuator) is derived from the exact training model.
    /// </summary>
    public static class ParityBatch
    {
        const string SceneDir = "Assets/PoKingHill/Scenes";
        const string ModelDir = "Assets/PoKingHill/Models";

        // Which fighter's body a testbed scene holds: -kothRobot g1 (default) or kim. Kim's files sit in kim subfolders.
        static string Robot => Arg("-kothRobot", "g1");
        static string ModelsOf(string robot) => robot == "g1" ? ModelDir : $"{ModelDir}/{robot}";
        static string SceneOf(string robot, string tag) => $"{SceneDir}/Testbed_{(robot == "g1" ? "" : robot + "_")}{tag}.unity";

        static string Arg(string name, string fallback)
        {
            var a = Environment.GetCommandLineArgs();
            int i = Array.IndexOf(a, name);
            return i >= 0 && i + 1 < a.Length ? a[i + 1] : fallback;
        }

        [MenuItem("PoKingHill/Import testbed scenes (flat_1p + koth_2p)")]
        public static void ImportAll()
        {
            try { Import("flat_1p", Robot); Import("koth_1p", Robot); Import("koth_2p", Robot); if (Robot == "g1") Import("koth_4p", Robot); }
            catch (Exception e) { Debug.LogError("[ParityBatch] IMPORT FAILED: " + e); if (HasFlag("-kothExit")) EditorApplication.Exit(3); throw; }
            if (HasFlag("-kothExit")) EditorApplication.Exit(0);
        }

        /// <summary>Import one testbed scene only: -kothScene &lt;tag&gt; (for example koth_4p), -kothRobot as usual.</summary>
        public static void ImportOne()
        {
            try { Import(Arg("-kothScene", "koth_4p"), Robot); }
            catch (Exception e) { Debug.LogError("[ParityBatch] IMPORT FAILED: " + e); if (HasFlag("-kothExit")) EditorApplication.Exit(3); throw; }
            if (HasFlag("-kothExit")) EditorApplication.Exit(0);
        }

        static bool HasFlag(string f) => Array.IndexOf(Environment.GetCommandLineArgs(), f) >= 0;

        public static void Import(string tag, string robot = "g1")
        {
            string xml = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "training", "assets", robot, $"scene_{tag}_unity.xml"));
            if (!File.Exists(xml)) throw new FileNotFoundException(xml);
            Directory.CreateDirectory(SceneDir);
            AssetDatabase.Refresh();
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var root = new MjImporterWithAssets().ImportFile(xml);
            if (root == null) throw new Exception($"MJCF import failed for {xml} (see console)");
            root.name = $"Mj_{tag}"; string models = ModelsOf(robot), label = robot == "g1" ? tag : $"{robot}_{tag}";

            // Names in the compiled model must equal the MJCF names (JointMap resolves by name).
            var settings = UnityEngine.Object.FindAnyObjectByType<MjGlobalSettings>();
            if (settings == null) settings = new GameObject("MjGlobalSettings").AddComponent<MjGlobalSettings>();
            settings.UseRawGameObjectNames = true;

            var jointMap = AssetDatabase.LoadAssetAtPath<TextAsset>($"{models}/joint_map.json");
            var dump = AssetDatabase.LoadAssetAtPath<TextAsset>($"{models}/model_dump_{tag}.json");
            if (jointMap == null || dump == null) throw new Exception("joint_map.json / model_dump json missing under " + models + " (run koth.build_mjcf)");

            var rig = new GameObject("KothRig");
            // koth_5p is the game scene: four G1 and Kim (e_), who has her own joint map and wears her scanned mesh.
            var kimMap = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelsOf("kim")}/joint_map.json");
            foreach (string prefix in tag.EndsWith("5p") ? new[] { "a_", "b_", "c_", "d_", "e_" } : tag.EndsWith("4p") ? new[] { "a_", "b_", "c_", "d_" } : tag.EndsWith("2p") ? new[] { "a_", "b_" } : new[] { "a_" })
            {
                bool kim = robot == "kim" || prefix == "e_";
                var go = new GameObject("Policy_" + prefix); go.transform.SetParent(rig.transform);
                var pr = go.AddComponent<PolicyRunner>(); pr.jointMapJson = kim && robot != "kim" ? kimMap : jointMap; pr.robotPrefix = prefix; pr.body = kim ? "kim" : "g1";
                if (prefix == "e_") AddKimSkin(prefix, rig);
            }
            rig.AddComponent<ProjectilePool>().jointMapJson = jointMap;
            rig.AddComponent<ModelDumpCheck>().modelDumpJson = dump;
            rig.AddComponent<ParityProbe>().outputFile = $"unity_hold_trace_{label}.json";
            rig.AddComponent<AutoShot>();                 // inert unless started with -kothShot <file>

            var cam = new GameObject("Main Camera") { tag = "MainCamera" }.AddComponent<Camera>();   // 9:16 portrait framing
            cam.transform.position = new Vector3(0f, 1.2f, -6.5f); cam.transform.LookAt(new Vector3(0f, 0.7f, 0f)); cam.fieldOfView = 40f;
            var light = new GameObject("Directional Light").AddComponent<Light>(); light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            // The plugin's own renderer for the arena geom drew nothing (verified by screenshots: robots appeared to
            // stand on open water), although the imported mesh asset is correct. Draw the hill with a separate
            // render-only object that shows the same mesh asset with a two-sided material. No physics component.
            foreach (var geom in UnityEngine.Object.FindObjectsByType<MjGeom>(FindObjectsInactive.Include))
            {
                if (geom.gameObject.name != "arena" || geom.Mesh == null || geom.Mesh.Mesh == null) continue;
                var own = geom.GetComponent<MeshRenderer>(); if (own != null) own.enabled = false;
                var visual = new GameObject("ArenaVisual (render only)");
                visual.transform.SetPositionAndRotation(geom.transform.position, geom.transform.rotation);
                visual.AddComponent<MeshFilter>().sharedMesh = geom.Mesh.Mesh;
                var mat = new Material(MjcfImporter.GetLitShader()) { color = new Color(0.47f, 0.42f, 0.36f, 1f), doubleSidedGI = true };
                mat.SetFloat("_Cull", 0f);
                AssetDatabase.CreateAsset(mat, $"{SceneDir}/ArenaMaterial_{label}.mat");
                visual.AddComponent<MeshRenderer>().sharedMaterial = mat;
                Debug.Log($"[ParityBatch] arena visual: {geom.Mesh.Mesh.vertexCount} verts, bounds {geom.Mesh.Mesh.bounds.center} / {geom.Mesh.Mesh.bounds.size}");
            }
            int colliders = UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsInactive.Include).Length;
            int bodies = UnityEngine.Object.FindObjectsByType<Rigidbody>(FindObjectsInactive.Include).Length;
            if (colliders + bodies > 0) throw new Exception($"PhysX components present after import: {colliders} colliders, {bodies} rigidbodies");

            if (robot == "kim")
            {
                foreach (string prefix in tag.EndsWith("2p") ? new[] { "a_", "b_" } : new[] { "a_" }) AddKimSkin(prefix, rig);
                cam.transform.position = new Vector3(2.6f, 1.25f, -2.4f); cam.transform.LookAt(new Vector3(0f, 0.85f, 0f));   // she faces +X
            }
            string path = SceneOf(robot, tag);
            EditorSceneManager.SaveScene(scene, path);
            Debug.Log($"[ParityBatch] imported {tag}: {UnityEngine.Object.FindObjectsByType<MjBody>(FindObjectsInactive.Include).Length} MjBody, " +
                      $"{UnityEngine.Object.FindObjectsByType<MjGeom>(FindObjectsInactive.Include).Length} MjGeom, " +
                      $"{UnityEngine.Object.FindObjectsByType<MjActuator>(FindObjectsInactive.Include).Length} MjActuator -> {path}");
        }

        const string KimDir = "Assets/PoKingHill/Fighters/Kim";

        /// <summary>Dress one fighter of a Kim scene in her scanned mesh (kim.fbx from tools/make_kim.py): the mesh
        /// follows the MuJoCo bodies through SkinFollower and the collision shapes stop being drawn.</summary>
        static void AddKimSkin(string prefix, GameObject rig)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>($"{KimDir}/kim.fbx");
            var table = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelsOf("kim")}/kim_skin.json");
            if (prefab == null || table == null) throw new Exception("kim.fbx or kim_skin.json missing (run tools/make_kim.py in Blender, then koth.build_mjcf with KOTH_ROBOT=kim)");
            var skin = (GameObject)PrefabUtility.InstantiatePrefab(prefab); skin.name = "KimSkin_" + prefix;
            var mat = AssetDatabase.LoadAssetAtPath<Material>($"{KimDir}/Kim.mat");
            if (mat == null)
            {
                mat = new Material(Shader.Find("Universal Render Pipeline/Lit")); mat.SetFloat("_Smoothness", 0.2f);
                mat.SetTexture("_BaseMap", MapTexture("kim_albedo", false, true, KimDir));
                mat.SetTexture("_BumpMap", MapTexture("kim_normal", true, false, KimDir)); mat.EnableKeyword("_NORMALMAP");
                AssetDatabase.CreateAsset(mat, $"{KimDir}/Kim.mat");
            }
            foreach (var r in skin.GetComponentsInChildren<SkinnedMeshRenderer>()) r.sharedMaterial = mat;
            var follow = rig.AddComponent<SkinFollower>(); follow.skinJson = table; follow.robotPrefix = prefix; follow.skinRoot = skin.transform;
            var pelvis = GameObject.Find(prefix + "pelvis"); if (pelvis == null) throw new Exception("no body " + prefix + "pelvis in the imported scene");
            foreach (var r in pelvis.GetComponentsInChildren<MeshRenderer>()) r.enabled = false;
        }

        /// <summary>Policy gate: open Testbed_&lt;tag&gt;, attach &lt;rung&gt;_policy.onnx to the runner, add the parity probe
        /// with &lt;rung&gt;_reference_trajectory.json, enter play mode. Scene changes are in memory only (not saved).</summary>
        public static void RunPolicy()
        {
            string tag = Arg("-kothScene", "flat_1p"), rung = Arg("-kothRung", "r0");
            AssetDatabase.Refresh();
            EditorSceneManager.OpenScene(SceneOf(Robot, tag), OpenSceneMode.Single); string models = ModelsOf(Robot);
            var policy = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{models}/{rung}_policy.onnx");
            var runners = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include).OrderBy(r => r.robotPrefix).ToArray();
            foreach (var hold in UnityEngine.Object.FindObjectsByType<ParityProbe>(FindObjectsInactive.Include)) hold.enabled = false;
            int probes = 0;
            foreach (var runner in runners)
            {
                string suffix = runners.Length == 1 ? "" : "_" + runner.robotPrefix.Trim('_');
                var reference = AssetDatabase.LoadAssetAtPath<TextAsset>($"{models}/{rung}_reference_trajectory{suffix}.json");
                if (policy == null || reference == null) continue;
                runner.policy = policy;
                runner.opponent = runners.FirstOrDefault(r => r != runner);
                var probe = runner.gameObject.AddComponent<PolicyParityProbe>();
                probe.referenceJson = reference; probe.policy = policy; probe.runner = runner; probe.rung = rung; probe.suffix = suffix;
                probes++;
            }
            if (probes != runners.Length) { Debug.LogError($"[ParityBatch] missing {rung}_policy.onnx or reference json in {models} ({probes}/{runners.Length} robots)"); if (HasFlag("-kothExit")) EditorApplication.Exit(3); return; }
            EditorApplication.EnterPlaymode();
        }

        [MenuItem("PoKingHill/Build duel demo scene")]
        public static void BuildDuelDemo() => BuildDuelDemo("koth_5p");

        /// <summary>Builds and saves Demo_duel.unity from an imported arena scene: the game uses the one with four G1
        /// and Kim (koth_5p, default), the two-fighter parity gates the scene the brains were trained in (koth_2p). The menu
        /// seats the ticked agents on the bodies, everyone starts on the summit and the sea rises as the round clock.
        /// No PhysX components are created (the water disc loses its collider).</summary>
        public static void BuildDuelDemo(string tag)
        {
            AssetDatabase.Refresh();
            var scene = EditorSceneManager.OpenScene($"{SceneDir}/Testbed_{tag}.unity", OpenSceneMode.Single);
            var attacker = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/attacker_policy.onnx");
            if (attacker == null) throw new Exception("attacker_policy.onnx missing in " + ModelDir);
            var runners = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include).OrderBy(r => r.robotPrefix).ToArray();
            var a = runners[0]; var b = runners[1];
            // Kim fights with her walking brain and the goal law (walk at the nearest opponent); she is locked in the
            // menu until that brain exists (scripts/export_policy.py --name kim/walker).
            var kimBrain = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelsOf("kim")}/walker_policy.onnx");
            bool kimBody = runners.Any(r => r.body == "kim");
            foreach (var r in runners) r.opponent = r == a ? b : a;      // brains and opponents are assigned at launch by DemoDirector
            foreach (var probe in UnityEngine.Object.FindObjectsByType<ParityProbe>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(probe);
            var rig = a.transform.parent.gameObject;

            var water = GameObject.CreatePrimitive(PrimitiveType.Cylinder); water.name = "Sea (render only)";
            UnityEngine.Object.DestroyImmediate(water.GetComponent<Collider>());                 // PhysX stays out of the scene
            water.transform.localScale = new Vector3(900f, 0.01f, 900f);         // reaches past the distant peaks
            var mat = new Material(MjcfImporter.GetLitShader()) { color = new Color(0.10f, 0.35f, 0.55f, 1f) };
            AssetDatabase.CreateAsset(mat, $"{SceneDir}/SeaMaterial.mat"); water.GetComponent<MeshRenderer>().sharedMaterial = mat;
            var sea = rig.AddComponent<Sea>(); sea.waterVisual = water.transform;

            var director = rig.AddComponent<DemoDirector>(); director.fighters = runners; director.sea = sea;
            Fighter F(string name, string file, int dim, GoalMode goal) => new Fighter { name = name, obsDim = dim, goal = goal, stopDist = 0f, vmax = 1f,
                policy = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/{file}_policy.onnx") };
            director.roster = new[] { F("G1 All-rounder (champion)", "attacker", 112, GoalMode.Opponent), F("G1 Duelist (generation 7)", "duelist", 112, GoalMode.Opponent),
                                      F("G1 Rammer (generation 1)", "rammer", 112, GoalMode.Opponent), F("G1 Walker (stands its ground)", "r1", 103, GoalMode.None),
                                      // Kim has a body (Testbed_kim_* scenes) but no trained brain yet, and this scene holds two G1 bodies.
                                      new Fighter { name = "Kim", body = "kim", policy = kimBrain, obsDim = 103, goal = GoalMode.Opponent, stopDist = 0f, vmax = 0.8f,
                                                    available = kimBrain != null && kimBody, inGame = kimBrain != null && kimBody,
                                                    note = !kimBody ? "no Kim body in this scene" : kimBrain == null ? "no brain yet" : "walks at you, never trained to fight" } };
            if (director.roster.Any(f => f.available && f.policy == null)) throw new Exception("a roster policy is missing in " + ModelDir);
            director.maps = new[] { "Mountain top", "Summit (baseline)" };
            director.mapVisuals = new[] { BuildMountainVisual(), GameObject.Find("ArenaVisual (render only)") };
            RenderSettings.fog = true; RenderSettings.fogMode = FogMode.Exponential; RenderSettings.fogDensity = 0.0022f;      // haze gives the peaks distance
            RenderSettings.fogColor = new Color(0.72f, 0.80f, 0.90f, 1f);
            var cam = Camera.main;                      // 9:16 portrait framing of the summit
            cam.transform.position = new Vector3(0f, 2.6f, -6.2f); cam.transform.LookAt(new Vector3(0f, 0.5f, 0f)); cam.fieldOfView = 38f;
            var mc = cam.gameObject.AddComponent<MatchCamera>(); mc.director = director;
            if (cam.GetComponent<AudioListener>() == null) cam.gameObject.AddComponent<AudioListener>();
            var synth = rig.AddComponent<ImpactSynth>(); synth.robots = runners; synth.sea = sea;   // adds the AudioSource it requires
            director.synth = synth;
            rig.AddComponent<PerfProbe>();              // inert unless started with -kothPerf <seconds>
            int physx = UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsInactive.Include).Length + UnityEngine.Object.FindObjectsByType<Rigidbody>(FindObjectsInactive.Include).Length;
            if (physx > 0) throw new Exception($"PhysX components present in the demo scene: {physx}");
            string path = $"{SceneDir}/Demo_duel.unity";
            EditorSceneManager.SaveScene(scene, path);
            Debug.Log("[ParityBatch] saved " + path);
        }

        const string MapDir = "Assets/PoKingHill/Maps/Mountain";

        static Texture2D MapTexture(string name, bool normal, bool srgb, string dir = MapDir)
        {
            string path = $"{dir}/{name}.png"; var imp = AssetImporter.GetAtPath(path) as TextureImporter;
            if (imp == null) throw new Exception(path + " missing (run tools/make_mountain_map.py in Blender)");
            var type = normal ? TextureImporterType.NormalMap : TextureImporterType.Default;
            if (imp.textureType != type || imp.sRGBTexture != srgb) { imp.textureType = type; imp.sRGBTexture = srgb; imp.SaveAndReimport(); }
            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        /// <summary>The "Mountain top" map: scenery exported from Blender by tools/make_mountain_map.py, shown instead of
        /// the plain dome. Render only: no collider, the collision shape stays the MuJoCo arena mesh.</summary>
        static GameObject BuildMountainVisual()
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>($"{MapDir}/mountain.fbx");
            if (prefab == null) throw new Exception($"{MapDir}/mountain.fbx missing (run tools/make_mountain_map.py in Blender)");
            var root = (GameObject)PrefabUtility.InstantiatePrefab(prefab); root.name = "MountainVisual (render only)";
            var lit = Shader.Find("Universal Render Pipeline/Lit");
            var rock = new Material(lit); var mask = MapTexture("mountain_mask", false, false);
            rock.SetTexture("_BaseMap", MapTexture("mountain_albedo", false, true));
            rock.SetTexture("_BumpMap", MapTexture("mountain_normal", true, false)); rock.EnableKeyword("_NORMALMAP");
            rock.SetTexture("_MetallicGlossMap", mask); rock.EnableKeyword("_METALLICSPECGLOSSMAP"); rock.SetFloat("_Smoothness", 1f);   // smoothness = mask alpha
            rock.SetTexture("_OcclusionMap", mask); rock.EnableKeyword("_OCCLUSIONMAP");
            rock.SetTexture("_DetailAlbedoMap", MapTexture("rock_detail_albedo", false, false));
            rock.SetTexture("_DetailNormalMap", MapTexture("rock_detail_normal", true, false));
            rock.SetTextureScale("_DetailAlbedoMap", new Vector2(10.97f, 10.97f)); rock.EnableKeyword("_DETAIL_MULX2");  // value printed by make_mountain_map.py (2.7 m tile)
            var far = new Material(lit); far.SetTexture("_BaseMap", MapTexture("backdrop_albedo", false, true)); far.SetFloat("_Smoothness", 0.1f);
            var boulder = new Material(lit); boulder.SetTexture("_BaseMap", MapTexture("boulder_albedo", false, true)); boulder.SetFloat("_Smoothness", 0.15f);
            boulder.SetTexture("_BumpMap", MapTexture("rock_detail_normal", true, false)); boulder.EnableKeyword("_NORMALMAP");
            AssetDatabase.CreateAsset(rock, $"{MapDir}/MountainRock.mat"); AssetDatabase.CreateAsset(far, $"{MapDir}/BackdropRock.mat"); AssetDatabase.CreateAsset(boulder, $"{MapDir}/BoulderRock.mat");
            foreach (var r in root.GetComponentsInChildren<MeshRenderer>()) r.sharedMaterial = r.name.StartsWith("Backdrop") ? far : r.name.StartsWith("Rocks") ? boulder : rock;
            foreach (var c in root.GetComponentsInChildren<Collider>()) UnityEngine.Object.DestroyImmediate(c);
            foreach (var f in root.GetComponentsInChildren<MeshFilter>()) Debug.Log($"[ParityBatch] map mesh {f.name}: {f.sharedMesh.vertexCount} verts, world bounds {f.GetComponent<Renderer>().bounds}");
            return root;
        }

        /// <summary>Still picture of the whole map from outside (the match camera only ever frames the summit):
        /// renders Demo_duel in edit mode with the sea at its starting level to -kothOut (default map_overview.png).</summary>
        public static void MapOverview()
        {
            BuildDuelDemo();
            var water = GameObject.Find("Sea (render only)"); if (water != null) water.transform.position = new Vector3(0f, -6f, 0f);
            var dd = UnityEngine.Object.FindAnyObjectByType<DemoDirector>();                 // in edit mode nothing has picked a map yet
            for (int i = 0; i < dd.mapVisuals.Length; i++) if (dd.mapVisuals[i] != null) dd.mapVisuals[i].SetActive(i == dd.selectedMap);
            var cam = new GameObject("OverviewCamera").AddComponent<Camera>(); cam.fieldOfView = 45f; cam.farClipPlane = 2000f;
            cam.transform.position = new Vector3(11f, 1.5f, -13f); cam.transform.LookAt(new Vector3(0f, -3f, 0f));
            var rt = new RenderTexture(1280, 960, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB); cam.targetTexture = rt; cam.Render();
            RenderTexture.active = rt; var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0); tex.Apply(); RenderTexture.active = null;
            string path = Path.GetFullPath(Arg("-kothOut", "map_overview.png")); File.WriteAllBytes(path, tex.EncodeToPNG());
            Debug.Log("[ParityBatch] wrote " + path);
            if (HasFlag("-kothExit")) EditorApplication.Exit(0);
        }

        /// <summary>Parity gate for the combat policy: Demo_duel with the director replaced by one probe per robot,
        /// compared against &lt;rung&gt;_reference_trajectory_a/_b.json (recorded with the same two ONNX files).</summary>
        public static void RunDuelParity()
        {
            string rung = Arg("-kothRung", "r4duel");
            BuildDuelDemo("koth_2p");
            EditorSceneManager.OpenScene($"{SceneDir}/Demo_duel.unity", OpenSceneMode.Single);
            var champion = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/attacker_policy.onnx");
            foreach (var r in UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include)) { r.policy = champion; r.policyObsDim = 112; r.goal = GoalMode.Opponent; r.goalStopDist = 0f; r.goalVmax = 1f; }
            foreach (var dd in UnityEngine.Object.FindObjectsByType<DemoDirector>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(dd);
            foreach (var sea in UnityEngine.Object.FindObjectsByType<Sea>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(sea);
            foreach (var runner in UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include))
            {
                string suffix = "_" + runner.robotPrefix.Trim('_');
                var reference = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelDir}/{rung}_reference_trajectory{suffix}.json");
                if (reference == null) { Debug.LogError("[ParityBatch] missing reference for " + runner.robotPrefix); if (HasFlag("-kothExit")) EditorApplication.Exit(3); return; }
                var probe = runner.gameObject.AddComponent<PolicyParityProbe>();
                probe.referenceJson = reference; probe.policy = runner.policy; probe.runner = runner; probe.rung = rung; probe.suffix = suffix;
            }
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Statistical parity gate for the combat rungs: play N fast rounds of the duel demo without the sea
        /// (fixed 25 s bell, as in training evaluation) and write duel_stats_unity.json.</summary>
        public static void RunDuelStats()
        {
            BuildDuelDemo("koth_2p");
            EditorSceneManager.OpenScene($"{SceneDir}/Demo_duel.unity", OpenSceneMode.Single);
            foreach (var sea in UnityEngine.Object.FindObjectsByType<Sea>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(sea);
            var dd = UnityEngine.Object.FindAnyObjectByType<DemoDirector>(); dd.sea = null; dd.roundSeconds = 25f;
            dd.statsRounds = int.Parse(Arg("-kothRounds", "200")); dd.selectedA = int.Parse(Arg("-kothA", "0")); dd.selectedB = int.Parse(Arg("-kothB", "0"));
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Windows player of the duel demo in Builds/Win (release; add -kothDev for a development build).</summary>
        [MenuItem("PoKingHill/Build Windows player")]
        public static void BuildPlayer()
        {
            try
            {
                BuildDuelDemo();
                var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                {
                    scenes = new[] { $"{SceneDir}/Demo_duel.unity" }, locationPathName = "Builds/Win/PoKingHill.exe",
                    target = BuildTarget.StandaloneWindows64, options = HasFlag("-kothDev") ? BuildOptions.Development : BuildOptions.None,
                });
                Debug.Log($"[ParityBatch] build {report.summary.result}: {report.summary.totalErrors} errors, {report.summary.totalSize / (1024 * 1024)} MB -> {report.summary.outputPath}");
                if (HasFlag("-kothExit")) EditorApplication.Exit(report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded ? 0 : 3);
            }
            catch (Exception e) { Debug.LogError("[ParityBatch] BUILD FAILED: " + e); if (HasFlag("-kothExit")) EditorApplication.Exit(3); throw; }
        }

        /// <summary>Build the duel demo, open it and press Play (used with a normal, visible editor launch).</summary>
        public static void PlayDuelDemo()
        {
            BuildDuelDemo();
            EditorSceneManager.OpenScene($"{SceneDir}/Demo_duel.unity", OpenSceneMode.Single);
            EditorApplication.ExecuteMenuItem("Window/General/Game");      // the Simulator view turns clicks into touches, which the IMGUI menu ignores
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Open Testbed_&lt;tag&gt; and enter play mode. ParityProbe writes the trace and exits the editor in batch mode.</summary>
        public static void RunHold()
        {
            string tag = Arg("-kothScene", "flat_1p");
            EditorSceneManager.OpenScene(SceneOf(Robot, tag), OpenSceneMode.Single);
            EditorApplication.EnterPlaymode();
        }
    }
}
