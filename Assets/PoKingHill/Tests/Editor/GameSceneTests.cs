using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using NUnit.Framework;
using Unity.Cinemachine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoKingHill.Tests
{
    /// <summary>Scene integrity of the game (missing scripts, unwired references) and the pure rules behind the HUD.</summary>
    public class GameSceneTests
    {
        const string Game = "Assets/PoKingHill/Scenes/Demo_duel.unity";

        static T One<T>(UnityEngine.SceneManagement.Scene s) where T : Component =>
            s.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<T>(true)).FirstOrDefault();

        [Test]
        public void GameSceneHasNoMissingScriptsAndEverythingIsWired()
        {
            var scene = EditorSceneManager.OpenScene(Game, OpenSceneMode.Additive);
            try
            {
                int missing = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true))
                                   .Sum(t => GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(t.gameObject));
                Assert.Zero(missing, "GameObjects with missing scripts");

                var d = One<DemoDirector>(scene); Assert.NotNull(d, "DemoDirector");
                Assert.GreaterOrEqual(d.fighters.Length, 2); Assert.NotNull(d.sea); Assert.NotNull(d.synth);
                foreach (var f in d.roster.Where(f => f.available)) Assert.NotNull(f.policy, f.name + " has no policy");
                Assert.AreEqual(1, d.roster.Count(f => f.steerable), "exactly one steerable (puppet) brain");
                Assert.IsEmpty(d.sea.robotPrefixes, "the sea must float every fighter (empty list = all PolicyRunners)");

                var hud = One<HudView>(scene); Assert.NotNull(hud, "HudView");
                Assert.NotNull(hud.style, "HUD stylesheet"); Assert.NotNull(hud.director); Assert.NotNull(hud.odds); Assert.NotNull(hud.replay);
                Assert.NotNull(hud.log); Assert.NotNull(hud.hazards); Assert.NotNull(hud.chat);
                var doc = hud.GetComponent<UIDocument>(); Assert.NotNull(doc.panelSettings, "panel settings");
                Assert.AreEqual(PanelScaleMode.ScaleWithScreenSize, doc.panelSettings.scaleMode);

                var bc = One<BroadcastCamera>(scene); Assert.NotNull(bc, "BroadcastCamera");
                foreach (var cam in new[] { bc.summit, bc.faller, bc.winner, bc.replay, bc.menu }) Assert.NotNull(cam, "broadcast shot camera");
                Assert.NotNull(bc.brain); Assert.NotNull(bc.classic); Assert.NotNull(bc.group); Assert.NotNull(bc.impulse);

                var fx = One<MatchFx>(scene); Assert.NotNull(fx, "MatchFx");
                foreach (var ps in new[] { fx.dust, fx.sparks, fx.splash, fx.ripple, fx.confetti }) Assert.NotNull(ps, "particle system");
                Assert.NotNull(fx.hitVolume); Assert.NotNull(fx.sun); Assert.NotNull(fx.kingSpot);

                foreach (var c in new System.Type[] { typeof(MatchLog), typeof(ReplaySystem), typeof(PuppetControl), typeof(WinOdds), typeof(Commentator), typeof(Hazards), typeof(ChatVotes), typeof(BodyLook) })
                    Assert.IsTrue(scene.GetRootGameObjects().Any(r => r.GetComponentInChildren(c, true) != null), c.Name + " missing");
            }
            finally { EditorSceneManager.CloseScene(scene, true); }
        }

        [Test]
        public void ArenaRuleMatchesTheTrainingConstants()
        {
            var a = new ArenaSpec();
            Assert.AreEqual(1.5, a.plateau_r); Assert.AreEqual(1.0 / 9.0, a.slope_k); Assert.AreEqual(0.3, a.fall_height); Assert.AreEqual(0.3, a.fall_up);
            Assert.AreEqual(0.0, a.Ground(1.2)); Assert.AreEqual(-(1.0 / 9.0) * 4.0, a.Ground(3.5), 1e-15);
            // when build_mjcf.py has written the arena block, it must equal the C# defaults bit for bit
            string jm = "Assets/PoKingHill/Models/joint_map.json";
            var spec = JsonUtility.FromJson<JointMapSpec>(File.ReadAllText(jm));
            if (spec.arena != null && spec.arena.plateau_r != 0)
            {
                Assert.AreEqual(a.plateau_r, spec.arena.plateau_r); Assert.AreEqual(a.slope_k, spec.arena.slope_k);
                Assert.AreEqual(a.fall_height, spec.arena.fall_height); Assert.AreEqual(a.fall_up, spec.arena.fall_up);
            }
        }

        [TestCase("Assets/PoKingHill/Models", "flat_1p", "g1", "a_")]
        [TestCase("Assets/PoKingHill/Models", "koth_2p", "g1", "a_ b_")]
        [TestCase("Assets/PoKingHill/Models", "koth_4p", "g1", "a_ b_ c_ d_")]
        [TestCase("Assets/PoKingHill/Models", "koth_5p", "g1", "a_ b_ c_ d_ e_:kim")]
        [TestCase("Assets/PoKingHill/Models", "koth_6p", "g1", "a_ b_ c_ d_ e_:kim f_:trump")]
        [TestCase("Assets/PoKingHill/Models/kim", "koth_2p", "kim", "a_:kim b_:kim")]
        [TestCase("Assets/PoKingHill/Models/trump", "koth_2p", "trump", "a_:trump b_:trump")]
        public void ModelDumpListsItsRobots(string dir, string tag, string robot, string expected)
        {
            var dump = AssetDatabase.LoadAssetAtPath<TextAsset>($"{dir}/model_dump_{tag}.json");
            var got = EditorTools.ParityBatch.RobotsOf(dump, tag, robot).Select(r => r.body == "g1" ? r.prefix : r.prefix + ":" + r.body);
            Assert.AreEqual(expected, string.Join(" ", got));
        }

        /// <summary>Unity serializes a component only through the script file named after its class; a second
        /// MonoBehaviour in one file works in the session that adds it and is a missing script after a restart.</summary>
        [Test]
        public void EveryComponentLivesInAFileOfItsOwnName()
        {
            foreach (var path in Directory.GetFiles("Assets/PoKingHill/Scripts", "*.cs"))
            {
                string text = File.ReadAllText(path), file = Path.GetFileNameWithoutExtension(path);
                foreach (Match m in Regex.Matches(text, @"class\s+(\w+)\s*:\s*MonoBehaviour"))
                    Assert.AreEqual(file, m.Groups[1].Value, $"{path} declares MonoBehaviour {m.Groups[1].Value}: move it to {m.Groups[1].Value}.cs");
            }
        }

        [Test]
        public void BuildVersionIsYYMMDD() => StringAssert.IsMatch(@"^\d{6}$", BuildInfo.Version);

        [Test]
        public void LadderCountsBodiesNotAgents()
        {
            string tmp = Path.Combine(Path.GetTempPath(), "koth_ladder_test.json"); File.Delete(tmp);
            Ladder.FileOverride = tmp;
            try
            {
                var roster = new[] { new Fighter { name = "A" }, new Fighter { name = "B" } };
                // mirror match: nothing changes
                Ladder.Record(new RoundResult(1, "A", "", RoundEnd.Knockout, 3f, new[] { 0, 0 }), roster);
                Assert.AreEqual(Ladder.Start, Ladder.Rating("A"));
                // A beats B: symmetric update, A up
                Ladder.Record(new RoundResult(0, "A", "", RoundEnd.Knockout, 3f, new[] { 0, 1 }), roster);
                Assert.Greater(Ladder.Rating("A"), Ladder.Start); Assert.AreEqual(2 * Ladder.Start, Ladder.Rating("A") + Ladder.Rating("B"), 1e-3);
                // A + its mirror + B, the mirror copy (body 1) wins: A still beats B through body 1, B loses to body 1 only
                float a0 = Ladder.Rating("A");
                Ladder.Record(new RoundResult(1, "A", "", RoundEnd.Knockout, 3f, new[] { 0, 0, 1 }), roster);
                Assert.Greater(Ladder.Rating("A"), a0);
            }
            finally { Ladder.FileOverride = null; File.Delete(tmp); }
        }

        [Test]
        public void HudStylesheetHasNoUnsupportedSelectors()
        {
            string uss = File.ReadAllText("Assets/PoKingHill/UI/Hud.uss");
            Assert.IsFalse(Regex.IsMatch(uss, @":(last-child|first-child|nth-child)"), "USS has no structural pseudo-classes");
        }
    }
}
