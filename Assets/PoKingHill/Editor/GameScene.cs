using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoKingHill.EditorTools
{
    /// <summary>
    /// The game scene, Scenes/Game.unity, is composed in the editor and saved like any hand-made scene: the map, the sea,
    /// the lights, the camera, the match logic on KothRig and the Presentation prefab instance are tuned by hand and never
    /// rebuilt. Code only refreshes what comes out of training, in place:
    ///   RefreshBrains   the roster (one Agent per brain in Models/) and the get-up brains: after export_policy.py
    ///   RefreshBodies   the MuJoCo model, the runners and the skins, taken from the imported testbed
    ///                   Testbed_koth_game.unity: after python -m koth.build_mjcf game and ParityBatch.ImportOne
    ///   Compose         a gate scene: a copy of the game scene with the bodies of another testbed (koth_2p, koth_4p, ...)
    /// Unity CLI: unity command eval --code 'PoKingHill.EditorTools.GameScene.RefreshBrains(); return "ok";'
    /// </summary>
    public static class GameScene
    {
        public const string Path = "Assets/PoKingHill/Scenes/Game.unity";
        public const string GameTag = "koth_game";
        const string ModelDir = "Assets/PoKingHill/Models";

        [MenuItem("PoKingHill/Game scene/Refresh brains (after export_policy.py)")]
        public static void RefreshBrains()
        {
            var scene = EditorSceneManager.OpenScene(Path, OpenSceneMode.Single);
            FillRoster(Director(scene));
            Save(scene);
        }

        [MenuItem("PoKingHill/Game scene/Refresh bodies from Testbed_koth_game")]
        public static void RefreshBodies()
        {
            var scene = EditorSceneManager.OpenScene(Path, OpenSceneMode.Single);
            SwapBodies(scene, GameTag, "g1");
            FillRoster(Director(scene));
            Save(scene);
        }

        /// <summary>A copy of the game scene at <paramref name="savePath"/> holding the bodies of Testbed_[robot_]tag
        /// (the statistics and parity gates: the scene the brains were trained in). Left open.</summary>
        public static void Compose(string tag, string robot, string savePath)
        {
            AssetDatabase.DeleteAsset(savePath);
            if (!AssetDatabase.CopyAsset(Path, savePath)) throw new Exception($"could not copy {Path} to {savePath}");
            var scene = EditorSceneManager.OpenScene(savePath, OpenSceneMode.Single);
            SwapBodies(scene, tag, robot);
            FillRoster(Director(scene));
            Save(scene);
        }

        static MatchDirector Director(Scene scene) =>
            scene.GetRootGameObjects().Select(r => r.GetComponentInChildren<MatchDirector>(true)).FirstOrDefault(d => d != null)
            ?? throw new Exception(scene.path + " has no MatchDirector");

        static void Save(Scene scene)
        {
            ParityBatch.AssertNoPhysX("in " + scene.path);
            EditorSceneManager.SaveScene(scene);
            Debug.Log("[GameScene] saved " + scene.path);
        }

        static bool IsImport(GameObject root) => root.name.StartsWith("Mj_");

        // ------------------------------------------------------------------ bodies
        /// <summary>Replaces the scene's MuJoCo tree, runners and skins with the testbed's; everything else stays.</summary>
        static void SwapBodies(Scene scene, string tag, string robot)
        {
            var director = Director(scene); var rig = director.gameObject;
            foreach (var r in rig.GetComponentsInChildren<PolicyRunner>(true)) UnityEngine.Object.DestroyImmediate(r.gameObject);
            foreach (var f in rig.GetComponents<SkinFollower>()) { if (f.skinRoot != null) UnityEngine.Object.DestroyImmediate(f.skinRoot.gameObject); UnityEngine.Object.DestroyImmediate(f); }
            foreach (var root in scene.GetRootGameObjects().Where(IsImport)) UnityEngine.Object.DestroyImmediate(root);

            var testbed = EditorSceneManager.OpenScene(ParityBatch.SceneOf(robot, tag), OpenSceneMode.Additive);
            try
            {
                var source = testbed.GetRootGameObjects().Select(r => r.GetComponentInChildren<PolicyRunner>(true)).FirstOrDefault(r => r != null)?.transform.parent.gameObject
                             ?? throw new Exception(testbed.path + " has no PolicyRunner");
                foreach (var root in testbed.GetRootGameObjects().Where(IsImport)) SceneManager.MoveGameObjectToScene(root, scene);
                foreach (var r in source.GetComponentsInChildren<PolicyRunner>(true))
                {
                    r.transform.SetParent(null); SceneManager.MoveGameObjectToScene(r.gameObject, scene); r.transform.SetParent(rig.transform, false);
                }
                foreach (var f in source.GetComponents<SkinFollower>())
                {
                    SceneManager.MoveGameObjectToScene(f.skinRoot.gameObject, scene);
                    EditorUtility.CopySerialized(f, rig.AddComponent<SkinFollower>());
                }
                rig.GetComponent<ModelDumpCheck>().modelDumpJson = source.GetComponent<ModelDumpCheck>().modelDumpJson;
                rig.GetComponent<ProjectilePool>().jointMapJson = source.GetComponent<ProjectilePool>().jointMapJson;
                // the plain dome map draws the imported arena mesh
                var arena = testbed.GetRootGameObjects().FirstOrDefault(r => r.name == "ArenaVisual (render only)")?.GetComponent<MeshFilter>();
                var dome = scene.GetRootGameObjects().FirstOrDefault(r => r.name == "ArenaVisual (render only)")?.GetComponent<MeshFilter>();
                if (arena != null && dome != null) dome.sharedMesh = arena.sharedMesh;
            }
            finally { EditorSceneManager.CloseScene(testbed, true); }

            var runners = rig.GetComponentsInChildren<PolicyRunner>(true).OrderBy(r => r.robotPrefix).ToArray();
            // body kinds from the model dump: testbeds imported before PolicyRunner had a body field say "g1" for Kim too
            var dump = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ParityBatch.ModelsOf(robot)}/model_dump_{tag}.json");
            if (dump != null)
                foreach (var e in ParityBatch.RobotsOf(dump, tag, robot)) { var r = runners.FirstOrDefault(x => x.robotPrefix == e.prefix); if (r != null) r.body = e.body; }
            foreach (var r in runners) r.opponent = r == runners[0] ? runners[1] : runners[0];      // the director points brains at opponents at launch
            director.bodies = runners;
            if (director.synth != null) director.synth.robots = runners;
            EditorSceneManager.MarkSceneDirty(scene);
        }

        // ------------------------------------------------------------------ brains
        /// <summary>The roster offered in the menu, from the brains in Models/ and the bodies this scene has.</summary>
        static void FillRoster(MatchDirector director)
        {
            var runners = director.bodies;
            var roster = new List<Agent>();
            // The G1 agents left the game on 2026-10-20: a gate scene with G1 bodies (koth_2p, koth_4p, koth_7p) gets them back.
            if (runners.Any(r => r.body == "g1")) roster.AddRange(G1Roster());
            // A human is offered only when this scene has its body and a brain of theirs exists. The newest brain wins: ffa
            // (four-way, 121 inputs), the styled brain (style), xg1 (duels against G1), fighter, walker. A brain trained with a
            // styled action path (env.py style_pose) has <file>_style.txt beside it naming the pose; the style brain itself
            // is styled. vmax: the walking speed the fighter was trained with (Trump's style: 0.7).
            Agent Human(string body, string name, string badge, float fighterVmax, string style = null)
            {
                string dir = ParityBatch.ModelsOf(body);
                string file = new[] { "ffa", style, "xg1", "fighter", "walker" }.FirstOrDefault(f => f != null && Onnx(dir, f + "_policy") != null) ?? "walker";
                var sidecar = AssetDatabase.LoadAssetAtPath<TextAsset>($"{dir}/{file}_style.txt");
                string pose = sidecar != null ? sidecar.text.Trim() : file == style ? style : null;
                var policy = Onnx(dir, file + "_policy"); bool fights = file != "walker", hasBody = runners.Any(r => r.body == body);
                return new Agent { name = name, shortName = badge, body = body, policy = policy, obsDim = policy != null ? ParityBatch.InputsOf(policy) : 103, goal = GoalMode.Opponent, stopDist = 0f,
                                   vmax = fights ? fighterVmax : 0.8f, available = policy != null && hasBody, inGame = policy != null && hasBody,
                                   shove = pose != null, stylePose = pose ?? "shove", value = Onnx(dir, file + "_value"), action = Action(dir, file),
                                   note = !hasBody ? $"no {badge} body in this scene" : policy == null ? "no brain yet" : fights ? "" : "walks at you, never trained to fight" };
            }
            roster.Add(Human("kim", "Kim", "Kim", 1f));
            roster.Add(Human("trump", "Trump (Bulldozer: 102 kg, slow, shoves with his body)", "Trump", 0.7f, "bull"));
            roster.Add(Human("grandpa", "Grandpa (86 kg, in his eighties, leans in and shoves)", "Grandpa", 1f));
            // the puppet: a walking brain that follows the stick (velocity commands) on a body this scene has; the G1 Walker
            // when there are G1 bodies, else the first human walker (Kim's), which takes one of her two bodies
            if (!roster.Any(f => f.steerable))
                foreach (var body in new[] { "kim", "trump", "grandpa" })
                {
                    string dir = ParityBatch.ModelsOf(body);
                    var w = Onnx(dir, "walker_policy");
                    if (w == null || !runners.Any(r => r.body == body)) continue;
                    string Name = $"{char.ToUpperInvariant(body[0])}{body.Substring(1)}";
                    roster.Add(new Agent { name = $"{Name} (you steer)", shortName = $"{Name} walker", body = body, policy = w, obsDim = 103, goal = GoalMode.None, vmax = 0.8f,
                                           steerable = true, inGame = false, value = Onnx(dir, "walker_value") });
                    break;
                }
            director.roster = roster.ToArray();
            if (director.roster.Any(f => f.available && f.policy == null)) throw new Exception("a roster policy is missing in " + ModelDir);
            // get-up brains (training rung getup, exported as Models[/<body>]/getup_policy.onnx) for the bodies that have one
            director.getupBrains = new[] { "g1", "kim", "trump", "grandpa" }.Where(b => runners.Any(r => r.body == b))
                .Select(b => new BodyBrain { body = b, policy = Onnx(ParityBatch.ModelsOf(b), "getup_policy") })
                .Where(b => b.policy != null).ToArray();
            EditorUtility.SetDirty(director);
        }

        static Unity.InferenceEngine.ModelAsset Onnx(string dir, string file) => AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>($"{dir}/{file}.onnx");
        // <file>_action.json (export_policy.py), when the brain was trained with a changed action path
        static TextAsset Action(string dir, string file) => AssetDatabase.LoadAssetAtPath<TextAsset>($"{dir}/{file}_action.json");

        /// <summary>The G1 agents, for the gate scenes that still hold G1 bodies: All-rounder (champion careful1),
        /// Free-for-all when exported, Duelist (gen7) and Rammer (gen1), which the free-for-all gate's opponent pool
        /// needs (eval.py --rung r5ffa), and the Walker (the puppet).</summary>
        static IEnumerable<Agent> G1Roster()
        {
            Agent A(string name, string badge, string file, int dim, GoalMode goal) => new Agent { name = name, shortName = badge, obsDim = dim, goal = goal,
                policy = Onnx(ModelDir, file + "_policy"), value = Onnx(ModelDir, file + "_value"), action = Action(ModelDir, file) };
            yield return A("G1 All-rounder (champion)", "All-rounder", "attacker", 112, GoalMode.Opponent);
            var ffa = Onnx(ModelDir, "ffa_policy");
            if (ffa != null) yield return A("G1 Free-for-all (trained 4-way)", "Free-for-all", "ffa", ParityBatch.InputsOf(ffa), GoalMode.Opponent);
            yield return A("G1 Duelist (generation 7)", "Duelist", "duelist", 112, GoalMode.Opponent);
            yield return A("G1 Rammer (generation 1)", "Rammer", "rammer", 112, GoalMode.Opponent);
            var walker = A("G1 Walker (stands its ground)", "Walker", "r1", 103, GoalMode.None); walker.steerable = true;
            yield return walker;
        }
    }
}
