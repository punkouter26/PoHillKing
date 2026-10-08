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
            string tag = Arg("-kothScene", "flat_1p"), rung = Arg("-kothRung", "r0"), prefix = Arg("-kothPrefix", "a_");
            AssetDatabase.Refresh();
            EditorSceneManager.OpenScene($"{SceneDir}/Testbed_{tag}.unity", OpenSceneMode.Single);
            var policy = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{ModelDir}/{rung}_policy.onnx");
            var reference = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ModelDir}/{rung}_reference_trajectory.json");
            if (policy == null || reference == null) { Debug.LogError($"[ParityBatch] missing {rung}_policy.onnx or reference json in {ModelDir}"); if (HasFlag("-kothExit")) EditorApplication.Exit(3); return; }
            var runner = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include).First(r => r.robotPrefix == prefix);
            runner.policy = policy;
            foreach (var hold in UnityEngine.Object.FindObjectsByType<ParityProbe>(FindObjectsInactive.Include)) hold.enabled = false;
            var probe = runner.gameObject.AddComponent<PolicyParityProbe>();
            probe.referenceJson = reference; probe.policy = policy; probe.runner = runner; probe.rung = rung;
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
