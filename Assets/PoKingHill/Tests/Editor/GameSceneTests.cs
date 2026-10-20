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
                Assert.NotNull(hud.bio); Assert.NotNull(hud.ledger); Assert.NotNull(hud.stamina); Assert.NotNull(hud.tactician); Assert.NotNull(hud.weather);
                Assert.NotNull(d.ledger, "director.ledger (Grudge/Vulture)"); Assert.NotNull(d.bio, "director.bio (Vulture)");
                foreach (var f in d.roster.Where(f => f.available && f.goal == GoalMode.Opponent && f.policy != null))
                    Assert.NotNull(f.mlp, f.name + " has no _mlp.bytes for the Tactician (training/scripts/export_mlp.py)");
                var doc = hud.GetComponent<UIDocument>(); Assert.NotNull(doc.panelSettings, "panel settings");
                Assert.AreEqual(PanelScaleMode.ScaleWithScreenSize, doc.panelSettings.scaleMode);

                var bc = One<BroadcastCamera>(scene); Assert.NotNull(bc, "BroadcastCamera");
                foreach (var cam in new[] { bc.summit, bc.faller, bc.winner, bc.replay, bc.menu, bc.crane, bc.intro }) Assert.NotNull(cam, "broadcast shot camera");
                Assert.NotNull(bc.crane.GetComponent<CinemachineSplineDolly>()?.Spline, "intro crane spline");
                Assert.NotNull(bc.brain); Assert.NotNull(bc.classic); Assert.NotNull(bc.group); Assert.NotNull(bc.impulse);

                var fx = One<MatchFx>(scene); Assert.NotNull(fx, "MatchFx");
                foreach (var ps in new[] { fx.dust, fx.sparks, fx.splash, fx.ripple, fx.confetti }) Assert.NotNull(ps, "particle system");
                Assert.NotNull(fx.hitVolume); Assert.NotNull(fx.sun); Assert.NotNull(fx.kingSpot); Assert.NotNull(fx.steam); Assert.NotNull(fx.stamina); Assert.NotNull(fx.weather);
                Assert.NotNull(fx.shockRing); Assert.NotNull(fx.drips); Assert.NotNull(fx.lightning); Assert.NotNull(fx.bolt); Assert.NotNull(fx.danger);
                Assert.NotNull(fx.synth); Assert.NotNull(fx.bio); Assert.NotNull(fx.quality); Assert.NotNull(fx.hazards, "MatchFx.hazards (banner wind)");

                // the Mountain map (tools/make_mountain_map.py): three levels of detail, the props, the swaying banners
                var map = d.mapVisuals[0]; Assert.NotNull(map, "mountain map visual");
                var lods = map.GetComponentInChildren<LODGroup>(true); Assert.NotNull(lods, "mountain LODGroup"); Assert.AreEqual(3, lods.lodCount);
                var parts = map.GetComponentsInChildren<MeshRenderer>(true);
                foreach (var n in new[] { "Crags", "Ruins", "BannerPoles", "Banners", "Rocks", "Backdrop" }) Assert.IsTrue(parts.Any(r => r.name == n), n + " missing from the map");
                Assert.AreEqual("PoKingHill/Banner", parts.First(r => r.name == "Banners").sharedMaterial.shader.name);
                Assert.IsTrue(parts.All(r => r.GetComponent<Collider>() == null), "the map is render only");

                // sound: 3D one-shots and what the ambience follows
                Assert.NotNull(d.synth.pool, "ImpactSynth.pool (SoundPool)"); Assert.NotNull(d.synth.director); Assert.NotNull(d.synth.hazards);
                Assert.NotNull(d.synth.weather); Assert.NotNull(d.synth.bio);
                // fighter surface: the material must be an asset on the project shader, so a player build keeps it
                var look = One<BodyLook>(scene); Assert.NotNull(look.fighterMaterial, "BodyLook.fighterMaterial (FX/Fighter.mat)"); Assert.NotNull(look.bio);
                Assert.AreEqual("PoKingHill/Fighter", look.fighterMaterial.shader.name);
                Assert.IsTrue(AssetDatabase.Contains(look.fighterMaterial), "the fighter material is an asset");
                var warp = One<TimeWarp>(scene); Assert.NotNull(warp, "TimeWarp"); Assert.NotNull(warp.director); Assert.NotNull(warp.bio);
                var quality = One<QualityTier>(scene); Assert.NotNull(quality, "QualityTier"); Assert.NotNull(quality.cam); Assert.NotNull(quality.sun); Assert.NotNull(quality.director);
                Assert.NotNull(hud.quality); Assert.NotNull(hud.synth); Assert.NotNull(hud.warp);
                Assert.NotNull(One<MatchLog>(scene).perf, "MatchLog.perf");

                foreach (var c in new System.Type[] { typeof(MatchLog), typeof(ReplaySystem), typeof(PuppetControl), typeof(WinOdds), typeof(Commentator), typeof(Hazards), typeof(ChatVotes), typeof(BodyLook),
                                                      typeof(ContactLedger), typeof(Biometrics), typeof(Stamina), typeof(Tactician), typeof(WeatherFeed), typeof(VoiceCaster) })
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
