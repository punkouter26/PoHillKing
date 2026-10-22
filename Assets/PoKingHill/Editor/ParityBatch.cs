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

        // Which fighter's body a testbed scene holds: -kothRobot g1 (default), kim, trump or grandpa. Human bodies' files sit in subfolders.
        static string Robot => Cli.Arg("-kothRobot", "g1");
        internal static string ModelsOf(string robot) => robot == "g1" ? ModelDir : $"{ModelDir}/{robot}";
        internal static string SceneOf(string robot, string tag) => $"{SceneDir}/Testbed_{(robot == "g1" ? "" : robot + "_")}{tag}.unity";

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
        internal static void AssertNoPhysX(string where)
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
            // One runner per robot the model holds, as listed by build_mjcf in the dump (koth_7p: four G1, Kim as e_, Trump as f_,
            // Grandpa as g_). Each body uses its own kind's joint map, and every human body wears its mesh.
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
                // one shared material for every imported arena (FX/Arena.mat), created on the first import
                const string arenaMat = "Assets/PoKingHill/FX/Arena.mat";
                var mat = AssetDatabase.LoadAssetAtPath<Material>(arenaMat);
                if (mat == null)
                {
                    mat = new Material(MjcfImporter.GetLitShader()) { color = new Color(0.47f, 0.42f, 0.36f, 1f), doubleSidedGI = true };
                    mat.SetFloat("_Cull", 0f);
                    AssetDatabase.CreateAsset(mat, arenaMat);
                }
                visual.AddComponent<MeshRenderer>().sharedMaterial = mat;
                Debug.Log($"[ParityBatch] arena visual: {geom.Mesh.Mesh.vertexCount} verts, bounds {geom.Mesh.Mesh.bounds.center} / {geom.Mesh.Mesh.bounds.size}");
            }
            AssertNoPhysX("after import");

            if (robot != "g1") { cam.transform.position = new Vector3(2.6f, 1.25f, -2.4f); cam.transform.LookAt(new Vector3(0f, 0.85f, 0f)); }   // humans face +X
            string path = SceneOf(robot, tag);
            EditorSceneManager.SaveScene(scene, path);
            foreach (var g in Directory.GetFiles(SceneDir, "Gate_*.unity")) AssetDatabase.DeleteAsset(g.Replace('\\', '/'));
            Debug.Log($"[ParityBatch] imported {tag}: {UnityEngine.Object.FindObjectsByType<MjBody>(FindObjectsInactive.Include).Length} MjBody, " +
                      $"{UnityEngine.Object.FindObjectsByType<MjGeom>(FindObjectsInactive.Include).Length} MjGeom, " +
                      $"{UnityEngine.Object.FindObjectsByType<MjActuator>(FindObjectsInactive.Include).Length} MjActuator -> {path}");
        }

        /// <summary>Observation width a policy was exported with (103, 112 or 121).</summary>
        internal static int InputsOf(Unity.InferenceEngine.ModelAsset policy) => Unity.InferenceEngine.ModelLoader.Load(policy).inputs[0].shape.Get(1);

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

        /// <summary>Fighters/&lt;Body&gt;: a human body's mesh folder (Kim, Trump, Grandpa).</summary>
        static string SkinDir(string body) => $"Assets/PoKingHill/Fighters/{char.ToUpperInvariant(body[0])}{body.Substring(1)}";

        /// <summary>Dress one human fighter in its mesh: &lt;body&gt;.fbx (tools/make_kim.py, tools/make_grandpa.py, with a
        /// URP material built from &lt;body&gt;_albedo / _normal) or &lt;body&gt;.glb (tools/make_glb_fighter.py, imported by
        /// glTFast with its own materials). The mesh follows the MuJoCo bodies through SkinFollower and the collision
        /// shapes stop being drawn.</summary>
        static void AddSkin(string prefix, string body, GameObject rig)
        {
            string dir = SkinDir(body), Body = $"{char.ToUpperInvariant(body[0])}{body.Substring(1)}";
            var fbx = AssetDatabase.LoadAssetAtPath<GameObject>($"{dir}/{body}.fbx");
            var prefab = fbx ?? AssetDatabase.LoadAssetAtPath<GameObject>($"{dir}/{body}.glb");
            var table = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelsOf(body)}/{body}_skin.json");
            if (prefab == null || table == null) throw new Exception($"{dir}/{body}.fbx|.glb or {body}_skin.json missing (tools/make_kim.py, make_grandpa.py or make_glb_fighter.py, then koth.build_mjcf with KOTH_ROBOT={body})");
            var skin = (GameObject)PrefabUtility.InstantiatePrefab(prefab); skin.name = $"{Body}Skin_{prefix}";
            if (fbx != null)
            {
                var mat = AssetDatabase.LoadAssetAtPath<Material>($"{dir}/{Body}.mat");
                if (mat == null)
                {
                    mat = new Material(Shader.Find("Universal Render Pipeline/Lit")); mat.SetFloat("_Smoothness", 0.2f);
                    mat.SetTexture("_BaseMap", MapTexture($"{body}_albedo", false, true, dir));
                    mat.SetTexture("_BumpMap", MapTexture($"{body}_normal", true, false, dir)); mat.EnableKeyword("_NORMALMAP");
                    AssetDatabase.CreateAsset(mat, $"{dir}/{Body}.mat");
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

        /// <summary>The two-fighter gates' scene (git-ignored): the game scene with the koth_2p bodies, so a gate run never
        /// changes the game scene.</summary>
        const string GateScene = SceneDir + "/Gate_duel_2p.unity";

        static Texture2D MapTexture(string name, bool normal, bool srgb, string dir)
        {
            string path = $"{dir}/{name}.png"; var imp = AssetImporter.GetAtPath(path) as TextureImporter;
            if (imp == null) throw new Exception(path + " missing (run tools/make_mountain_map.py in Blender)");
            var type = normal ? TextureImporterType.NormalMap : TextureImporterType.Default;
            if (imp.textureType != type || imp.sRGBTexture != srgb) { imp.textureType = type; imp.sRGBTexture = srgb; imp.SaveAndReimport(); }
            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        /// <summary>Still picture of the whole map from outside (the match camera only ever frames the summit):
        /// renders the game scene in edit mode with the sea at its starting level to -kothOut (default map_overview.png).
        /// Nothing is saved.</summary>
        public static void MapOverview()
        {
            EditorSceneManager.OpenScene(GameScene.Path, OpenSceneMode.Single);
            var water = GameObject.Find("Sea (render only)"); if (water != null) water.transform.position = new Vector3(0f, -6f, 0f);
            var dd = UnityEngine.Object.FindAnyObjectByType<MatchDirector>();                 // in edit mode nothing has picked a map yet
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

        /// <summary>Parity gate for the combat policy: the two-fighter gate scene with the director replaced by one probe per robot,
        /// compared against &lt;rung&gt;_reference_trajectory_a/_b.json (recorded with the same two ONNX files).</summary>
        public static void RunDuelParity()
        {
            string rung = Cli.Arg("-kothRung", "r4duel");
            GameScene.Compose("koth_2p", "g1", GateScene);
            var champion = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/attacker_policy.onnx");
            foreach (var r in UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include)) { r.policy = champion; r.policyObsDim = 112; r.goal = GoalMode.Opponent; r.goalStopDist = 0f; r.goalVmax = 1f; }
            foreach (var dd in UnityEngine.Object.FindObjectsByType<MatchDirector>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(dd);
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

        /// <summary>Statistical parity gate for the combat rungs: play N fast rounds of the two-fighter gate scene without the sea
        /// (fixed 25 s bell, as in training evaluation) and write duel_stats_unity.json.</summary>
        public static void RunDuelStats() => RunDuelStats("g1", int.Parse(Cli.Arg("-kothRounds", "200")), int.Parse(Cli.Arg("-kothA", "0")), int.Parse(Cli.Arg("-kothB", "0")));

        /// <summary>Kim's duel gate: her fighting brain against itself on her two-body testbed (compare with
        /// KOTH_ROBOT=kim scripts/duel_stats.py --policy fighter).</summary>
        public static void RunKimDuelStats() => RunDuelStats("kim", int.Parse(Cli.Arg("-kothRounds", "200")), -1, -1);
        public static void RunTrumpDuelStats() => RunDuelStats("trump", int.Parse(Cli.Arg("-kothRounds", "200")), -1, -1);
        public static void RunGrandpaDuelStats() => RunDuelStats("grandpa", int.Parse(Cli.Arg("-kothRounds", "200")), -1, -1);

        /// <summary>The statistics gate on a robot's two-body testbed; agent index -1 = that robot's own roster entry.</summary>
        public static void RunDuelStats(string robot, int rounds, int a, int b) => RunDuelStats(robot, rounds, a, b, play: true);

        static void RunDuelStats(string robot, int rounds, int a, int b, bool play)
        {
            string gate = robot == "g1" ? GateScene : $"{SceneDir}/Gate_duel_{robot}_2p.unity";
            GameScene.Compose("koth_2p", robot, gate);
            foreach (var sea in UnityEngine.Object.FindObjectsByType<Sea>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(sea);
            var dd = UnityEngine.Object.FindAnyObjectByType<MatchDirector>(); dd.sea = null; dd.roundSeconds = 25f;
            int own = Array.FindIndex(dd.roster, f => f.body == robot && f.available);
            dd.statsRounds = rounds; dd.selectedA = a < 0 ? own : a; dd.selectedB = b < 0 ? own : b;
            if (play) EditorApplication.EnterPlaymode();
        }

        /// <summary>Get-up gate: Testbed_flat_1p with the get-up brain (Models[/robot]/getup_policy.onnx, -kothRobot),
        /// -kothRounds N drops (default 200) with the training recipe; GetupProbe writes getup_stats_unity.json to compare
        /// with scripts/getup_stats.py (success share within the 5 s count and time-to-stand quartiles).</summary>
        public static void RunGetupStats() => RunGetupStats(Robot, int.Parse(Cli.Arg("-kothRounds", "200")));

        /// <summary>The get-up gate for a CLI eval, which cannot set -kothRobot / -kothRounds.</summary>
        public static void RunGetupStats(string robot, int drops)
        {
            EditorSceneManager.OpenScene(SceneOf(robot, "flat_1p"), OpenSceneMode.Single);
            var policy = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelsOf(robot)}/getup_policy.onnx");
            var runner = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include).FirstOrDefault();
            if (policy == null || runner == null) { Debug.LogError("[ParityBatch] RunGetupStats needs getup_policy.onnx and a PolicyRunner"); if (Cli.Exit) EditorApplication.Exit(3); return; }
            // the testbed's hold-parity probe writes unity_hold_trace_flat_1p.json whenever the scene plays: not in this gate
            foreach (var hold in UnityEngine.Object.FindObjectsByType<ParityProbe>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(hold);
            var probe = runner.gameObject.AddComponent<GetupProbe>(); probe.runner = runner; probe.policy = policy; probe.drops = drops;
            EditorApplication.EnterPlaymode();
        }

        /// <summary>The duel statistics gate with the get-up rule (MatchDirector.getupInStats): compare with
        /// scripts/duel_stats.py --getup on the same pair.</summary>
        public static void RunGetupDuelStats() => RunGetupDuelStats("g1", int.Parse(Cli.Arg("-kothRounds", "200")), int.Parse(Cli.Arg("-kothA", "0")), int.Parse(Cli.Arg("-kothB", "0")), Cli.Flag("-kothHill"));

        /// <summary>The get-up duel gate for a robot's two-body testbed (agent -1 = its own roster entry). hill = the game's
        /// rule (no count, no limit, anywhere on the top; duel_stats.py --getup --hill-rule), else the older 5 s count,
        /// 2 get-ups, plateau only (duel_stats.py --getup).</summary>
        public static void RunGetupDuelStats(string robot, int rounds, int a, int b, bool hill)
        {
            RunDuelStats(robot, rounds, a, b, play: false);
            var dd = UnityEngine.Object.FindAnyObjectByType<MatchDirector>(); dd.getupInStats = true;
            if (hill) { dd.getupSeconds = 0f; dd.maxGetups = 0; dd.getupOnPlateauOnly = false; dd.statsFile = "duel_stats_hill_unity.json"; }
            else { dd.getupSeconds = 5f; dd.maxGetups = 2; dd.getupOnPlateauOnly = true; dd.statsFile = "duel_stats_getup_unity.json"; }
            if (dd.getupBrains.Length == 0) { Debug.LogError($"[ParityBatch] no getup_policy.onnx for {robot}"); if (Cli.Exit) EditorApplication.Exit(3); return; }
            EditorApplication.EnterPlaymode();
        }

        const string GateFfaScene = SceneDir + "/Gate_ffa_4p.unity", GateFfaMixedScene = SceneDir + "/Gate_ffa_7p.unity";

        /// <summary>Free-for-all statistics gate: the candidate (-kothCandidate short name, default the Free-for-all brain,
        /// else the All-rounder) on body A against three opponents drawn every round from the pool eval.py --rung r5ffa uses
        /// (All-rounder twice, Duelist, Rammer, standing Walker), four G1 bodies, no sea, 20 s bell, -kothRounds N (default
        /// 200). Writes training/assets/g1/ffa_stats_unity.json: share of rounds the candidate is last one in / out first /
        /// undecided, to compare with eval.py's learner_last_one_in / learner_out / undecided.</summary>
        public static void RunFfaStats() => RunFfaStats(Cli.Arg("-kothCandidate", ""));

        /// <summary>The free-for-all gate for any candidate (roster badge; "" = the G1 Free-for-all brain, else the
        /// All-rounder). A human candidate (Kim, Trump, Grandpa) plays in the seven-body scene (koth_7p, Gate_ffa_7p.unity)
        /// against three G1 drawn from the same pool, the other human bodies parked: the Python reference is eval.py --rung
        /// &lt;body&gt;_ffa with that pool. Writes training/assets/&lt;candidate body&gt;/ffa_stats_unity.json.</summary>
        public static void RunFfaStats(string candidate) { if (RunFfaStatsScene(candidate)) EditorApplication.EnterPlaymode(); }

        static bool RunFfaStatsScene(string candidate)
        {
            bool human = candidate == "Kim" || candidate == "Trump" || candidate == "Grandpa";
            string gate = human ? GateFfaMixedScene : GateFfaScene;
            GameScene.Compose(human ? "koth_7p" : "koth_4p", "g1", gate);
            foreach (var sea in UnityEngine.Object.FindObjectsByType<Sea>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(sea);
            var dd = UnityEngine.Object.FindAnyObjectByType<MatchDirector>(); dd.sea = null; dd.roundSeconds = 20f;
            int Find(string badge) => Array.FindIndex(dd.roster, f => f.Badge == badge && f.available);
            string want = candidate != "" ? candidate : Find("Free-for-all") >= 0 ? "Free-for-all" : "All-rounder";
            dd.statsCandidate = Find(want);
            if (dd.statsCandidate < 0) { Debug.LogError("[ParityBatch] no available roster agent named " + want); if (Cli.Exit) EditorApplication.Exit(3); return false; }
            dd.statsPool = new[] { Find("All-rounder"), Find("All-rounder"), Find("Duelist"), Find("Rammer"), Find("Walker") }.Where(i => i >= 0).ToArray();
            dd.statsRounds = int.Parse(Cli.Arg("-kothRounds", "200"));
            return true;
        }

        const string GateHumanFfaScene = SceneDir + "/Gate_ffa_humans.unity";

        /// <summary>The humans' three-way gate: the game's bodies (koth_game), &lt;body&gt;'s fighter against the other two
        /// humans' fighters, one each (as eval.py --rung &lt;body&gt;_hill with the shipped fighters' checkpoints :slot1 / :slot2),
        /// no sea, 20 s bell, falls are out (no get-up rule).
        /// Writes training/assets/&lt;body&gt;/ffa_stats_unity.json: share of rounds the candidate is last one in / out / undecided.</summary>
        public static void RunHumanFfaStats(string body, int rounds = 200)
        {
            GameScene.Compose(GameScene.GameTag, "g1", GateHumanFfaScene);
            foreach (var sea in UnityEngine.Object.FindObjectsByType<Sea>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(sea);
            var dd = UnityEngine.Object.FindAnyObjectByType<MatchDirector>(); dd.sea = null; dd.roundSeconds = 20f;
            Agent Brain(string b, string file)
            {
                var policy = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelsOf(b)}/{file}_policy.onnx");
                if (policy == null) throw new Exception($"{ModelsOf(b)}/{file}_policy.onnx missing");
                return new Agent { name = $"{b} {file}", shortName = b, body = b, policy = policy, obsDim = InputsOf(policy), goal = GoalMode.Opponent, vmax = b == "trump" ? 0.7f : 1f,
                                   action = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelsOf(b)}/{file}_action.json") };
            }
            var others = new[] { "kim", "trump", "grandpa" }.Where(h => h != body);
            dd.roster = new[] { Brain(body, "fighter") }.Concat(others.Select(o => Brain(o, "fighter"))).ToArray();
            dd.statsCandidate = 0; dd.statsPool = new[] { 1, 2 }; dd.statsPoolInOrder = true; dd.statsRounds = rounds;
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Observation dump for training/scripts/obs_audit.py: the duel statistics gate (robot, roster seats a / b,
        /// N rounds) or, with ffaCandidate, the free-for-all gate (a human candidate plays in the six-body scene), with an
        /// ObsRecorder that writes training/assets/&lt;body&gt;/unity_obs_&lt;tag&gt;.jsonl when the gate leaves play mode.
        /// Compare with: obs_audit.py --ckpt &lt;the brain's checkpoint&gt; --unity training/assets/&lt;body&gt;/unity_obs_&lt;tag&gt;.jsonl.</summary>
        public static void RunObsDump() => RunObsDump(Cli.Arg("-kothRobot", "g1"), int.Parse(Cli.Arg("-kothRounds", "60")), int.Parse(Cli.Arg("-kothA", "0")), int.Parse(Cli.Arg("-kothB", "0")),
                                                      Cli.Arg("-kothCandidate", null), Cli.Arg("-kothTag", null), Cli.Arg("-kothBrain", ""));

        public static void RunObsDump(string robot, int rounds, int a, int b, string ffaCandidate = null, string tag = null, string brainFilter = "")
        {
            if (ffaCandidate != null) { if (!RunFfaStatsScene(ffaCandidate)) return; }
            else RunDuelStats(robot, rounds, a, b, play: false);
            var dd = UnityEngine.Object.FindAnyObjectByType<MatchDirector>(); dd.statsRounds = rounds;
            var rec = dd.gameObject.AddComponent<ObsRecorder>();
            rec.tag_ = tag ?? (ffaCandidate != null ? $"ffa_{ffaCandidate.ToLowerInvariant().Replace(' ', '_')}" : $"duel_{robot}_{a}_{b}"); rec.brainFilter = brainFilter ?? "";
            dd.statsFile = $"unity_obs_{rec.tag_}_stats.json";       // beside the dump: the gate's own reference files stay as they are
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Windows player of the game scene in Builds/Win (release; add -kothDev for a development build).</summary>
        [MenuItem("PoKingHill/Build Windows player")]
        public static void BuildPlayer()
        {
            try
            {
                var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                {
                    scenes = new[] { GameScene.Path }, locationPathName = "Builds/Win/PoKingHill.exe",
                    target = BuildTarget.StandaloneWindows64, options = Cli.Flag("-kothDev") ? BuildOptions.Development : BuildOptions.None,
                });
                Debug.Log($"[ParityBatch] build {report.summary.result}: {report.summary.totalErrors} errors, {report.summary.totalSize / (1024 * 1024)} MB -> {report.summary.outputPath}");
                if (Cli.Exit) EditorApplication.Exit(report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded ? 0 : 3);
            }
            catch (Exception e) { Debug.LogError("[ParityBatch] BUILD FAILED: " + e); if (Cli.Exit) EditorApplication.Exit(3); throw; }
        }

        /// <summary>Open the game scene and press Play (used with a normal, visible editor launch).</summary>
        public static void PlayGame()
        {
            EditorSceneManager.OpenScene(GameScene.Path, OpenSceneMode.Single);
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
