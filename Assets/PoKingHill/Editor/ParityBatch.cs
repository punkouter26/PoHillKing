using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoKingHill.EditorTools
{
    /// <summary>
    /// Editor-side authoring of the parity testbed scenes, runnable from the menu or from the command line (batch mode
    /// needs a licence this project does not have, so automated runs use a normal editor launch and -kothExit):
    ///   Unity.exe -projectPath . -executeMethod PoKingHill.EditorTools.ParityBatch.ImportAll -kothExit
    ///   Unity.exe -projectPath . -executeMethod PoKingHill.EditorTools.ParityBatch.RunHold -kothScene flat_1p -kothExit
    /// Import uses the plugin's own MJCF importer on training/assets/g1/scene_&lt;tag&gt;_unity.xml, so the Unity
    /// hierarchy (MjBody / MjGeom / MjHingeJoint / MjActuator) is derived from the exact training model.
    /// </summary>
    public static class ParityBatch
    {
        const string SceneDir = "Assets/PoKingHill/Scenes";
        const string ModelDir = "Assets/PoKingHill/Models";
        const string DemoScene = SceneDir + "/Demo_duel.unity";

        // Which fighter's body a testbed scene holds: -kothRobot g1 (default), kim or trump. Human bodies' files sit in subfolders.
        static string Robot => Cli.Arg("-kothRobot", "g1");
        static string ModelsOf(string robot) => robot == "g1" ? ModelDir : $"{ModelDir}/{robot}";
        static string SceneOf(string robot, string tag) => $"{SceneDir}/Testbed_{(robot == "g1" ? "" : robot + "_")}{tag}.unity";

        [MenuItem("PoKingHill/Import testbed scenes (flat_1p, koth_1p, koth_2p, koth_4p)")]
        public static void ImportAll() => ImportOrExit(() => { Import("flat_1p", Robot); Import("koth_1p", Robot); Import("koth_2p", Robot); if (Robot == "g1") Import("koth_4p", Robot); });

        /// <summary>Import one testbed scene only: -kothScene &lt;tag&gt; (for example koth_4p), -kothRobot as usual.</summary>
        public static void ImportOne() => ImportOrExit(() => Import(Cli.Arg("-kothScene", "koth_4p"), Robot));

        static void ImportOrExit(Action import)
        {
            try { import(); }
            catch (Exception e) { Debug.LogError("[ParityBatch] IMPORT FAILED: " + e); if (Cli.Exit) EditorApplication.Exit(3); throw; }
            if (Cli.Exit) EditorApplication.Exit(0);
        }

        // Gate B.2 at authoring time: MuJoCo is the only simulator, so a scene must never hold a PhysX component.
        static void AssertNoPhysX(string where)
        {
            int colliders = UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsInactive.Include).Length;
            int bodies = UnityEngine.Object.FindObjectsByType<Rigidbody>(FindObjectsInactive.Include).Length;
            if (colliders + bodies > 0) throw new Exception($"PhysX components present {where}: {colliders} colliders, {bodies} rigidbodies");
        }

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
            // One runner per robot the model holds, as listed by build_mjcf in the dump (koth_6p: four G1, Kim as e_ and Trump as f_).
            // Each body uses its own kind's joint map, and every Kim body wears her scanned mesh.
            foreach (var r in RobotsOf(dump, tag, robot))
            {
                var go = new GameObject("Policy_" + r.prefix); go.transform.SetParent(rig.transform);
                var pr = go.AddComponent<PolicyRunner>(); pr.robotPrefix = r.prefix; pr.body = r.body;
                pr.jointMapJson = r.body == robot ? jointMap : AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelsOf(r.body)}/joint_map.json");
                if (r.body != "g1") AddSkin(r.prefix, r.body, rig);
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
            AssertNoPhysX("after import");

            if (robot != "g1") { cam.transform.position = new Vector3(2.6f, 1.25f, -2.4f); cam.transform.LookAt(new Vector3(0f, 0.85f, 0f)); }   // humans face +X
            string path = SceneOf(robot, tag);
            EditorSceneManager.SaveScene(scene, path);
            Debug.Log($"[ParityBatch] imported {tag}: {UnityEngine.Object.FindObjectsByType<MjBody>(FindObjectsInactive.Include).Length} MjBody, " +
                      $"{UnityEngine.Object.FindObjectsByType<MjGeom>(FindObjectsInactive.Include).Length} MjGeom, " +
                      $"{UnityEngine.Object.FindObjectsByType<MjActuator>(FindObjectsInactive.Include).Length} MjActuator -> {path}");
        }

        /// <summary>Observation width a policy was exported with (103, 112 or 121).</summary>
        static int InputsOf(Unity.InferenceEngine.ModelAsset policy) => Unity.InferenceEngine.ModelLoader.Load(policy).inputs[0].shape.Get(1);

        [Serializable] public class RobotEntry { public string prefix, body; }
        [Serializable] class DumpRobots { public RobotEntry[] robots; }

        /// <summary>The robots a model holds, from the "robots" list build_mjcf.py writes into model_dump_&lt;tag&gt;.json;
        /// dumps written before that key existed fall back to the tag (1p/2p/4p, 5p = four G1 and Kim as e_).</summary>
        public static RobotEntry[] RobotsOf(TextAsset dump, string tag, string robot)
        {
            var listed = JsonUtility.FromJson<DumpRobots>(dump.text)?.robots;
            if (listed != null && listed.Length > 0) return listed;
            int n = tag.EndsWith("5p") ? 5 : tag.EndsWith("4p") ? 4 : tag.EndsWith("2p") ? 2 : 1;
            return Enumerable.Range(0, n).Select(i => new RobotEntry { prefix = (char)('a' + i) + "_", body = tag.EndsWith("5p") && i == 4 ? "kim" : robot }).ToArray();
        }

        /// <summary>Fighters/&lt;Body&gt;: a human body's mesh folder (Kim, Trump).</summary>
        static string SkinDir(string body) => $"Assets/PoKingHill/Fighters/{char.ToUpperInvariant(body[0])}{body.Substring(1)}";

        /// <summary>Dress one human fighter in its mesh: kim.fbx (tools/make_kim.py, with a URP material built from her
        /// textures) or &lt;body&gt;.glb (tools/make_glb_fighter.py, imported by glTFast with its own materials). The mesh
        /// follows the MuJoCo bodies through SkinFollower and the collision shapes stop being drawn.</summary>
        static void AddSkin(string prefix, string body, GameObject rig)
        {
            string dir = SkinDir(body);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>($"{dir}/{body}.fbx") ?? AssetDatabase.LoadAssetAtPath<GameObject>($"{dir}/{body}.glb");
            var table = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelsOf(body)}/{body}_skin.json");
            if (prefab == null || table == null) throw new Exception($"{dir}/{body}.fbx|.glb or {body}_skin.json missing (tools/make_kim.py or tools/make_glb_fighter.py, then koth.build_mjcf with KOTH_ROBOT={body})");
            var skin = (GameObject)PrefabUtility.InstantiatePrefab(prefab); skin.name = $"{char.ToUpperInvariant(body[0])}{body.Substring(1)}Skin_{prefix}";
            if (body == "kim")
            {
                var mat = AssetDatabase.LoadAssetAtPath<Material>($"{dir}/Kim.mat");
                if (mat == null)
                {
                    mat = new Material(Shader.Find("Universal Render Pipeline/Lit")); mat.SetFloat("_Smoothness", 0.2f);
                    mat.SetTexture("_BaseMap", MapTexture("kim_albedo", false, true, dir));
                    mat.SetTexture("_BumpMap", MapTexture("kim_normal", true, false, dir)); mat.EnableKeyword("_NORMALMAP");
                    AssetDatabase.CreateAsset(mat, $"{dir}/Kim.mat");
                }
                foreach (var r in skin.GetComponentsInChildren<SkinnedMeshRenderer>()) r.sharedMaterial = mat;
            }
            var follow = rig.AddComponent<SkinFollower>(); follow.skinJson = table; follow.robotPrefix = prefix; follow.skinRoot = skin.transform;
            var pelvis = GameObject.Find(prefix + "pelvis"); if (pelvis == null) throw new Exception("no body " + prefix + "pelvis in the imported scene");
            foreach (var r in pelvis.GetComponentsInChildren<MeshRenderer>()) r.enabled = false;
        }

        /// <summary>Policy gate: open Testbed_&lt;tag&gt;, attach &lt;rung&gt;_policy.onnx to the runner, add the parity probe
        /// with &lt;rung&gt;_reference_trajectory.json, enter play mode. Scene changes are in memory only (not saved).</summary>
        public static void RunPolicy() => RunPolicy(Cli.Arg("-kothScene", "flat_1p"), Cli.Arg("-kothRung", "r0"), Robot);

        /// <summary>The same gate with explicit arguments (for an editor that is already open, driven through the Unity CLI).</summary>
        public static void RunPolicy(string tag, string rung, string robot)
        {
            AssetDatabase.Refresh();
            EditorSceneManager.OpenScene(SceneOf(robot, tag), OpenSceneMode.Single); string models = ModelsOf(robot);
            var policy = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{models}/{rung}_policy.onnx");
            var runners = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include).OrderBy(r => r.robotPrefix).ToArray();
            foreach (var hold in UnityEngine.Object.FindObjectsByType<ParityProbe>(FindObjectsInactive.Include)) hold.enabled = false;
            int probes = 0;
            foreach (var runner in runners)
            {
                string suffix = runners.Length == 1 ? "" : "_" + runner.robotPrefix.Trim('_');
                var reference = AssetDatabase.LoadAssetAtPath<TextAsset>($"{models}/{rung}_reference_trajectory{suffix}.json");
                if (policy == null || reference == null) continue;
                runner.policy = policy; runner.policyObsDim = InputsOf(policy);     // 103, 112 (combat) or 121 (free-for-all)
                runner.opponent = runners.FirstOrDefault(r => r != runner);
                var probe = runner.gameObject.AddComponent<PolicyParityProbe>();
                probe.referenceJson = reference; probe.policy = policy; probe.runner = runner; probe.rung = rung; probe.suffix = suffix;
                probes++;
            }
            if (probes != runners.Length) { Debug.LogError($"[ParityBatch] missing {rung}_policy.onnx or reference json in {models} ({probes}/{runners.Length} robots)"); if (Cli.Exit) EditorApplication.Exit(3); return; }
            EditorApplication.EnterPlaymode();
        }

        [MenuItem("PoKingHill/Build duel demo scene")]
        public static void BuildDuelDemo() => BuildDuelDemo("koth_6p");

        /// <summary>The two-fighter gates' scene (git-ignored), so a gate run never overwrites the game scene.</summary>
        const string GateScene = SceneDir + "/Gate_duel_2p.unity";

        /// <summary>Builds the game scene from an imported arena scene and saves it to <paramref name="savePath"/>
        /// (Demo_duel.unity by default): the game uses the one with four G1, Kim and Trump (koth_6p, default), the two-fighter
        /// parity gates the scene the brains were trained in (koth_2p, saved as Gate_duel_2p.unity). The menu seats the
        /// ticked agents on the bodies, everyone starts on the summit and the sea rises as the round clock. Presentation
        /// (HUD, cameras, effects) comes from GameSceneBuilder. No PhysX components are created.</summary>
        public static void BuildDuelDemo(string tag, string savePath = DemoScene, string robot = "g1")
        {
            AssetDatabase.Refresh();
            var scene = EditorSceneManager.OpenScene(SceneOf(robot, tag), OpenSceneMode.Single);
            var attacker = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/attacker_policy.onnx");
            if (attacker == null) throw new Exception("attacker_policy.onnx missing in " + ModelDir);
            var runners = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include).OrderBy(r => r.robotPrefix).ToArray();
            // Body kinds from the model dump: testbeds imported before PolicyRunner had a body field say "g1" for Kim too.
            var dumpAsset = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelsOf(robot)}/model_dump_{tag}.json");
            if (dumpAsset != null)
                foreach (var entry in RobotsOf(dumpAsset, tag, robot)) { var r = runners.FirstOrDefault(x => x.robotPrefix == entry.prefix); if (r != null) r.body = entry.body; }
            var a = runners[0]; var b = runners[1];
            foreach (var r in runners) r.opponent = r == a ? b : a;      // brains and opponents are assigned at launch by DemoDirector
            foreach (var probe in UnityEngine.Object.FindObjectsByType<ParityProbe>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(probe);
            var rig = a.transform.parent.gameObject;

            var water = GameObject.CreatePrimitive(PrimitiveType.Cylinder); water.name = "Sea (render only)";
            UnityEngine.Object.DestroyImmediate(water.GetComponent<Collider>());                 // PhysX stays out of the scene
            water.transform.localScale = new Vector3(900f, 0.01f, 900f);         // reaches past the distant peaks
            var mat = AssetDatabase.LoadAssetAtPath<Material>($"{SceneDir}/SeaMaterial.mat");          // fallback look if the water shader is missing
            if (mat == null) { mat = new Material(MjcfImporter.GetLitShader()) { color = new Color(0.10f, 0.35f, 0.55f, 1f) }; AssetDatabase.CreateAsset(mat, $"{SceneDir}/SeaMaterial.mat"); }
            water.GetComponent<MeshRenderer>().sharedMaterial = mat;
            var sea = rig.AddComponent<Sea>(); sea.waterVisual = water.transform;     // robotPrefixes left empty: every fighter body floats

            var director = rig.AddComponent<DemoDirector>(); director.fighters = runners; director.sea = sea;
            Unity.InferenceEngine.ModelAsset Onnx(string file) => AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/{file}.onnx");
            // value = the critic exported next to the policy by export_policy.py --value, when present (drives the win-odds bar)
            Fighter F(string name, string badge, string file, int dim, GoalMode goal) => new Fighter { name = name, shortName = badge, obsDim = dim, goal = goal,
                policy = Onnx(file + "_policy"), value = Onnx(file + "_value") };
            var walker = F("G1 Walker (stands its ground)", "Walker", "r1", 103, GoalMode.None); walker.steerable = true;
            var roster = new List<Fighter> { F("G1 All-rounder (champion)", "All-rounder", "attacker", 112, GoalMode.Opponent), F("G1 Duelist (generation 7)", "Duelist", "duelist", 112, GoalMode.Opponent),
                                             F("G1 Rammer (generation 1)", "Rammer", "rammer", 112, GoalMode.Opponent), walker };
            // The free-for-all brain (scripts/train.py --rung r5ffa) joins when it has been exported; its input size comes from the file.
            var ffa = Onnx("ffa_policy");
            if (ffa != null) roster.Insert(1, new Fighter { name = "G1 Free-for-all (trained 4-way)", shortName = "Free-for-all", obsDim = InputsOf(ffa), goal = GoalMode.Opponent, policy = ffa, value = Onnx("ffa_value") });
            // A human is offered only when this scene has its body (koth_6p: Kim e_, Trump f_) and a brain of theirs exists:
            // the fighting brain (Models/<body>/fighter_policy.onnx, 112 inputs) when exported, else the walker that
            // walks at the nearest opponent. vmax: the walking speed the fighter was trained with (Trump's style: 0.7).
            // style: a brain trained with a styled action path (Models/<body>/<style>_policy.onnx, env.py style_pose),
            // preferred over the plain fighter when it exists. Trump's is "bull": torso forward, arms back, knees low.
            Fighter Human(string body, string name, string badge, float fighterVmax, string style = null)
            {
                Unity.InferenceEngine.ModelAsset Own(string file) => AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelsOf(body)}/{file}.onnx");
                string file = style != null && Own(style + "_policy") != null ? style : Own("fighter_policy") != null ? "fighter" : "walker";
                var policy = Own(file + "_policy"); bool fights = file != "walker", hasBody = runners.Any(r => r.body == body);
                return new Fighter { name = name, shortName = badge, body = body, policy = policy, obsDim = policy != null ? InputsOf(policy) : 103, goal = GoalMode.Opponent, stopDist = 0f,
                                     vmax = fights ? fighterVmax : 0.8f, available = policy != null && hasBody, inGame = policy != null && hasBody,
                                     shove = file == style, stylePose = file == style ? style : "shove", value = Own(file + "_value"),
                                     note = !hasBody ? $"no {badge} body in this scene" : policy == null ? "no brain yet" : fights ? "" : "walks at you, never trained to fight" };
            }
            roster.Add(Human("kim", "Kim", "Kim", 1f));
            roster.Add(Human("trump", "Trump (Bulldozer: 102 kg, slow, shoves with his body)", "Trump", 0.7f, "bull"));
            director.roster = roster.ToArray();
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
            GameSceneBuilder.Dress(rig, director, sea, synth, cam, mc, rig.GetComponent<ProjectilePool>());
            AssertNoPhysX("in the demo scene");
            EditorSceneManager.SaveScene(scene, savePath);
            Debug.Log("[ParityBatch] saved " + savePath);
        }

        const string MapDir = "Assets/PoKingHill/Maps/Mountain";

        // Reuse the material asset when it exists (a new one every build gave the scene and the .mat a fresh internal id each time).
        static Material MapMaterial(string name, Shader shader)
        {
            string path = $"{MapDir}/{name}.mat";
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m == null) { m = new Material(shader); AssetDatabase.CreateAsset(m, path); }
            return m;
        }

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
            var rock = MapMaterial("MountainRock", lit); var mask = MapTexture("mountain_mask", false, false);
            rock.SetTexture("_BaseMap", MapTexture("mountain_albedo", false, true));
            rock.SetTexture("_BumpMap", MapTexture("mountain_normal", true, false)); rock.EnableKeyword("_NORMALMAP");
            rock.SetTexture("_MetallicGlossMap", mask); rock.EnableKeyword("_METALLICSPECGLOSSMAP"); rock.SetFloat("_Smoothness", 1f);   // smoothness = mask alpha
            rock.SetTexture("_OcclusionMap", mask); rock.EnableKeyword("_OCCLUSIONMAP");
            rock.SetTexture("_DetailAlbedoMap", MapTexture("rock_detail_albedo", false, false));
            rock.SetTexture("_DetailNormalMap", MapTexture("rock_detail_normal", true, false));
            rock.SetTextureScale("_DetailAlbedoMap", new Vector2(10.97f, 10.97f)); rock.EnableKeyword("_DETAIL_MULX2");  // value printed by make_mountain_map.py (2.7 m tile)
            var far = MapMaterial("BackdropRock", lit); far.SetTexture("_BaseMap", MapTexture("backdrop_albedo", false, true)); far.SetFloat("_Smoothness", 0.1f);
            var boulder = MapMaterial("BoulderRock", lit); boulder.SetTexture("_BaseMap", MapTexture("boulder_albedo", false, true)); boulder.SetFloat("_Smoothness", 0.15f);
            boulder.SetTexture("_BumpMap", MapTexture("rock_detail_normal", true, false)); boulder.EnableKeyword("_NORMALMAP");
            foreach (var m in new[] { rock, far, boulder }) EditorUtility.SetDirty(m);
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
            string path = Path.GetFullPath(Cli.Arg("-kothOut", "map_overview.png")); File.WriteAllBytes(path, tex.EncodeToPNG());
            Debug.Log("[ParityBatch] wrote " + path);
            if (Cli.Exit) EditorApplication.Exit(0);
        }

        /// <summary>Parity gate for the combat policy: Demo_duel with the director replaced by one probe per robot,
        /// compared against &lt;rung&gt;_reference_trajectory_a/_b.json (recorded with the same two ONNX files).</summary>
        public static void RunDuelParity()
        {
            string rung = Cli.Arg("-kothRung", "r4duel");
            BuildDuelDemo("koth_2p", GateScene);
            EditorSceneManager.OpenScene(GateScene, OpenSceneMode.Single);
            var champion = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/attacker_policy.onnx");
            foreach (var r in UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include)) { r.policy = champion; r.policyObsDim = 112; r.goal = GoalMode.Opponent; r.goalStopDist = 0f; r.goalVmax = 1f; }
            foreach (var dd in UnityEngine.Object.FindObjectsByType<DemoDirector>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(dd);
            foreach (var sea in UnityEngine.Object.FindObjectsByType<Sea>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(sea);
            foreach (var runner in UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include))
            {
                string suffix = "_" + runner.robotPrefix.Trim('_');
                var reference = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelDir}/{rung}_reference_trajectory{suffix}.json");
                if (reference == null) { Debug.LogError("[ParityBatch] missing reference for " + runner.robotPrefix); if (Cli.Exit) EditorApplication.Exit(3); return; }
                var probe = runner.gameObject.AddComponent<PolicyParityProbe>();
                probe.referenceJson = reference; probe.policy = runner.policy; probe.runner = runner; probe.rung = rung; probe.suffix = suffix;
            }
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Statistical parity gate for the combat rungs: play N fast rounds of the duel demo without the sea
        /// (fixed 25 s bell, as in training evaluation) and write duel_stats_unity.json.</summary>
        public static void RunDuelStats() => RunDuelStats("g1", int.Parse(Cli.Arg("-kothRounds", "200")), int.Parse(Cli.Arg("-kothA", "0")), int.Parse(Cli.Arg("-kothB", "0")));

        /// <summary>Kim's duel gate: her fighting brain against itself on her two-body testbed (compare with
        /// KOTH_ROBOT=kim scripts/duel_stats.py --policy fighter).</summary>
        public static void RunKimDuelStats() => RunDuelStats("kim", int.Parse(Cli.Arg("-kothRounds", "200")), -1, -1);
        public static void RunTrumpDuelStats() => RunDuelStats("trump", int.Parse(Cli.Arg("-kothRounds", "200")), -1, -1);

        /// <summary>The statistics gate on a robot's two-body testbed; agent index -1 = that robot's own roster entry.</summary>
        public static void RunDuelStats(string robot, int rounds, int a, int b)
        {
            string gate = robot == "g1" ? GateScene : $"{SceneDir}/Gate_duel_{robot}_2p.unity";
            BuildDuelDemo("koth_2p", gate, robot);
            EditorSceneManager.OpenScene(gate, OpenSceneMode.Single);
            foreach (var sea in UnityEngine.Object.FindObjectsByType<Sea>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(sea);
            var dd = UnityEngine.Object.FindAnyObjectByType<DemoDirector>(); dd.sea = null; dd.roundSeconds = 25f;
            int own = Array.FindIndex(dd.roster, f => f.body == robot && f.available);
            dd.statsRounds = rounds; dd.selectedA = a < 0 ? own : a; dd.selectedB = b < 0 ? own : b;
            EditorApplication.EnterPlaymode();
        }

        const string GateFfaScene = SceneDir + "/Gate_ffa_4p.unity";

        /// <summary>Free-for-all statistics gate: the candidate (-kothCandidate short name, default the Free-for-all brain,
        /// else the All-rounder) on body A against three opponents drawn every round from the pool eval.py --rung r5ffa uses
        /// (All-rounder twice, Duelist, Rammer, standing Walker), four G1 bodies, no sea, 20 s bell, -kothRounds N (default
        /// 200). Writes training/assets/g1/ffa_stats_unity.json: share of rounds the candidate is last one in / out first /
        /// undecided, to compare with eval.py's learner_last_one_in / learner_out / undecided.</summary>
        public static void RunFfaStats()
        {
            BuildDuelDemo("koth_4p", GateFfaScene);
            EditorSceneManager.OpenScene(GateFfaScene, OpenSceneMode.Single);
            foreach (var sea in UnityEngine.Object.FindObjectsByType<Sea>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(sea);
            var dd = UnityEngine.Object.FindAnyObjectByType<DemoDirector>(); dd.sea = null; dd.roundSeconds = 20f;
            int Find(string badge) => Array.FindIndex(dd.roster, f => f.Badge == badge && f.available);
            string want = Cli.Arg("-kothCandidate", Find("Free-for-all") >= 0 ? "Free-for-all" : "All-rounder");
            dd.statsCandidate = Find(want);
            if (dd.statsCandidate < 0) { Debug.LogError("[ParityBatch] no available roster agent named " + want); if (Cli.Exit) EditorApplication.Exit(3); return; }
            dd.statsPool = new[] { Find("All-rounder"), Find("All-rounder"), Find("Duelist"), Find("Rammer"), Find("Walker") }.Where(i => i >= 0).ToArray();
            dd.statsRounds = int.Parse(Cli.Arg("-kothRounds", "200"));
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
                    scenes = new[] { DemoScene }, locationPathName = "Builds/Win/PoKingHill.exe",
                    target = BuildTarget.StandaloneWindows64, options = Cli.Flag("-kothDev") ? BuildOptions.Development : BuildOptions.None,
                });
                Debug.Log($"[ParityBatch] build {report.summary.result}: {report.summary.totalErrors} errors, {report.summary.totalSize / (1024 * 1024)} MB -> {report.summary.outputPath}");
                if (Cli.Exit) EditorApplication.Exit(report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded ? 0 : 3);
            }
            catch (Exception e) { Debug.LogError("[ParityBatch] BUILD FAILED: " + e); if (Cli.Exit) EditorApplication.Exit(3); throw; }
        }

        /// <summary>Build the duel demo, open it and press Play (used with a normal, visible editor launch).</summary>
        public static void PlayDuelDemo()
        {
            BuildDuelDemo();
            EditorSceneManager.OpenScene(DemoScene, OpenSceneMode.Single);
            EditorApplication.ExecuteMenuItem("Window/General/Game");      // the Simulator view turns clicks into touches, which the IMGUI menu ignores
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Open Testbed_&lt;tag&gt; and enter play mode. ParityProbe writes the trace and, with -kothExit, closes the editor.</summary>
        public static void RunHold()
        {
            string tag = Cli.Arg("-kothScene", "flat_1p");
            EditorSceneManager.OpenScene(SceneOf(Robot, tag), OpenSceneMode.Single);
            EditorApplication.EnterPlaymode();
        }
    }
}
