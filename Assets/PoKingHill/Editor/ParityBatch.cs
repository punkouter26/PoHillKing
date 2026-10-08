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

        static string Arg(string name, string fallback)
        {
            var a = Environment.GetCommandLineArgs();
            int i = Array.IndexOf(a, name);
            return i >= 0 && i + 1 < a.Length ? a[i + 1] : fallback;
        }

        [MenuItem("PoKingHill/Import testbed scenes (flat_1p + koth_2p)")]
        public static void ImportAll()
        {
            try { Import("flat_1p"); Import("koth_1p"); Import("koth_2p"); }
            catch (Exception e) { Debug.LogError("[ParityBatch] IMPORT FAILED: " + e); if (HasFlag("-kothExit")) EditorApplication.Exit(3); throw; }
            if (HasFlag("-kothExit")) EditorApplication.Exit(0);
        }

        static bool HasFlag(string f) => Array.IndexOf(Environment.GetCommandLineArgs(), f) >= 0;

        public static void Import(string tag)
        {
            string xml = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "training", "assets", "g1", $"scene_{tag}_unity.xml"));
            if (!File.Exists(xml)) throw new FileNotFoundException(xml);
            Directory.CreateDirectory(SceneDir);
            AssetDatabase.Refresh();
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var root = new MjImporterWithAssets().ImportFile(xml);
            if (root == null) throw new Exception($"MJCF import failed for {xml} (see console)");
            root.name = $"Mj_{tag}";

            // Names in the compiled model must equal the MJCF names (JointMap resolves by name).
            var settings = UnityEngine.Object.FindAnyObjectByType<MjGlobalSettings>();
            if (settings == null) settings = new GameObject("MjGlobalSettings").AddComponent<MjGlobalSettings>();
            settings.UseRawGameObjectNames = true;

            var jointMap = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelDir}/joint_map.json");
            var dump = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelDir}/model_dump_{tag}.json");
            if (jointMap == null || dump == null) throw new Exception("joint_map.json / model_dump json missing under " + ModelDir + " (run koth.build_mjcf)");

            var rig = new GameObject("KothRig");
            foreach (string prefix in tag.EndsWith("2p") ? new[] { "a_", "b_" } : new[] { "a_" })
            {
                var go = new GameObject("Policy_" + prefix); go.transform.SetParent(rig.transform);
                var pr = go.AddComponent<PolicyRunner>(); pr.jointMapJson = jointMap; pr.robotPrefix = prefix;
            }
            rig.AddComponent<ProjectilePool>().jointMapJson = jointMap;
            rig.AddComponent<ModelDumpCheck>().modelDumpJson = dump;
            rig.AddComponent<ParityProbe>().outputFile = $"unity_hold_trace_{tag}.json";

            var cam = new GameObject("Main Camera") { tag = "MainCamera" }.AddComponent<Camera>();   // 9:16 portrait framing
            cam.transform.position = new Vector3(0f, 1.2f, -6.5f); cam.transform.LookAt(new Vector3(0f, 0.7f, 0f)); cam.fieldOfView = 40f;
            var light = new GameObject("Directional Light").AddComponent<Light>(); light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            int colliders = UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsInactive.Include).Length;
            int bodies = UnityEngine.Object.FindObjectsByType<Rigidbody>(FindObjectsInactive.Include).Length;
            if (colliders + bodies > 0) throw new Exception($"PhysX components present after import: {colliders} colliders, {bodies} rigidbodies");

            string path = $"{SceneDir}/Testbed_{tag}.unity";
            EditorSceneManager.SaveScene(scene, path);
            Debug.Log($"[ParityBatch] imported {tag}: {UnityEngine.Object.FindObjectsByType<MjBody>(FindObjectsInactive.Include).Length} MjBody, " +
                      $"{UnityEngine.Object.FindObjectsByType<MjGeom>(FindObjectsInactive.Include).Length} MjGeom, " +
                      $"{UnityEngine.Object.FindObjectsByType<MjActuator>(FindObjectsInactive.Include).Length} MjActuator -> {path}");
        }

        /// <summary>Policy gate: open Testbed_&lt;tag&gt;, attach &lt;rung&gt;_policy.onnx to the runner, add the parity probe
        /// with &lt;rung&gt;_reference_trajectory.json, enter play mode. Scene changes are in memory only (not saved).</summary>
        public static void RunPolicy()
        {
            string tag = Arg("-kothScene", "flat_1p"), rung = Arg("-kothRung", "r0");
            AssetDatabase.Refresh();
            EditorSceneManager.OpenScene($"{SceneDir}/Testbed_{tag}.unity", OpenSceneMode.Single);
            var policy = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/{rung}_policy.onnx");
            var runners = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include).OrderBy(r => r.robotPrefix).ToArray();
            foreach (var hold in UnityEngine.Object.FindObjectsByType<ParityProbe>(FindObjectsInactive.Include)) hold.enabled = false;
            int probes = 0;
            foreach (var runner in runners)
            {
                string suffix = runners.Length == 1 ? "" : "_" + runner.robotPrefix.Trim('_');
                var reference = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelDir}/{rung}_reference_trajectory{suffix}.json");
                if (policy == null || reference == null) continue;
                runner.policy = policy;
                runner.opponent = runners.FirstOrDefault(r => r != runner);
                var probe = runner.gameObject.AddComponent<PolicyParityProbe>();
                probe.referenceJson = reference; probe.policy = policy; probe.runner = runner; probe.rung = rung; probe.suffix = suffix;
                probes++;
            }
            if (probes != runners.Length) { Debug.LogError($"[ParityBatch] missing {rung}_policy.onnx or reference json in {ModelDir} ({probes}/{runners.Length} robots)"); if (HasFlag("-kothExit")) EditorApplication.Exit(3); return; }
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Builds and saves Demo_duel.unity from the imported two-robot arena: both robots run the latest
        /// Attacker brain (112 inputs) and charge each other, both start on the summit, the sea rises as the round
        /// clock, and rounds restart automatically. No PhysX components are created (the water disc loses its collider).</summary>
        [MenuItem("PoKingHill/Build duel demo scene")]
        public static void BuildDuelDemo()
        {
            AssetDatabase.Refresh();
            var scene = EditorSceneManager.OpenScene($"{SceneDir}/Testbed_koth_2p.unity", OpenSceneMode.Single);
            var attacker = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/attacker_policy.onnx");
            if (attacker == null) throw new Exception("attacker_policy.onnx missing in " + ModelDir);
            var runners = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include).OrderBy(r => r.robotPrefix).ToArray();
            var a = runners[0]; var b = runners[1];
            foreach (var (r, o) in new[] { (a, b), (b, a) })
            {
                r.policy = attacker; r.policyObsDim = 112; r.goal = GoalMode.Opponent; r.goalStopDist = 0f; r.goalVmax = 1f; r.opponent = o;
            }
            foreach (var probe in UnityEngine.Object.FindObjectsByType<ParityProbe>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(probe);
            var rig = a.transform.parent.gameObject;

            var water = GameObject.CreatePrimitive(PrimitiveType.Cylinder); water.name = "Sea (render only)";
            UnityEngine.Object.DestroyImmediate(water.GetComponent<Collider>());                 // PhysX stays out of the scene
            water.transform.localScale = new Vector3(60f, 0.01f, 60f);
            var mat = new Material(MjcfImporter.GetLitShader()) { color = new Color(0.10f, 0.35f, 0.55f, 1f) };
            AssetDatabase.CreateAsset(mat, $"{SceneDir}/SeaMaterial.mat"); water.GetComponent<MeshRenderer>().sharedMaterial = mat;
            var sea = rig.AddComponent<Sea>(); sea.waterVisual = water.transform;

            var director = rig.AddComponent<DemoDirector>(); director.attacker = a; director.defender = b; director.sea = sea;
            director.labelA = "Robot A"; director.labelB = "Robot B"; director.behaviour = "King of the hill: Attacker vs Attacker";
            var cam = Camera.main;                      // 9:16 portrait framing of the summit
            cam.transform.position = new Vector3(0f, 2.6f, -6.2f); cam.transform.LookAt(new Vector3(0f, 0.5f, 0f)); cam.fieldOfView = 38f;
            int physx = UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsInactive.Include).Length + UnityEngine.Object.FindObjectsByType<Rigidbody>(FindObjectsInactive.Include).Length;
            if (physx > 0) throw new Exception($"PhysX components present in the demo scene: {physx}");
            string path = $"{SceneDir}/Demo_duel.unity";
            EditorSceneManager.SaveScene(scene, path);
            Debug.Log("[ParityBatch] saved " + path);
        }

        /// <summary>Parity gate for the combat policy: Demo_duel with the director replaced by one probe per robot,
        /// compared against &lt;rung&gt;_reference_trajectory_a/_b.json (recorded with the same two ONNX files).</summary>
        public static void RunDuelParity()
        {
            string rung = Arg("-kothRung", "r4duel");
            BuildDuelDemo();
            EditorSceneManager.OpenScene($"{SceneDir}/Demo_duel.unity", OpenSceneMode.Single);
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
            BuildDuelDemo();
            EditorSceneManager.OpenScene($"{SceneDir}/Demo_duel.unity", OpenSceneMode.Single);
            foreach (var sea in UnityEngine.Object.FindObjectsByType<Sea>(FindObjectsInactive.Include)) UnityEngine.Object.DestroyImmediate(sea);
            var dd = UnityEngine.Object.FindAnyObjectByType<DemoDirector>(); dd.sea = null; dd.roundSeconds = 25f;
            dd.statsRounds = int.Parse(Arg("-kothRounds", "200"));
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Build the duel demo, open it and press Play (used with a normal, visible editor launch).</summary>
        public static void PlayDuelDemo()
        {
            BuildDuelDemo();
            EditorSceneManager.OpenScene($"{SceneDir}/Demo_duel.unity", OpenSceneMode.Single);
            EditorApplication.EnterPlaymode();
        }

        /// <summary>Open Testbed_&lt;tag&gt; and enter play mode. ParityProbe writes the trace and exits the editor in batch mode.</summary>
        public static void RunHold()
        {
            string tag = Arg("-kothScene", "flat_1p");
            EditorSceneManager.OpenScene($"{SceneDir}/Testbed_{tag}.unity", OpenSceneMode.Single);
            EditorApplication.EnterPlaymode();
        }
    }
}
