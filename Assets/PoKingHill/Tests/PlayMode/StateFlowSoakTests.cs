using System.Collections;
using Mujoco;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace PoKingHill.Tests
{
    /// <summary>
    /// State-flow integrity under abuse: random taps at random moments (launches, skip, menu, continue, replay play and
    /// stop, debug panel, sheet hide and peek, restart, tab switches, the player's walker) for a few hundred steps, with
    /// invariants checked every frame: replay state matches the director's, physics runs whenever no replay does, the
    /// sheet only shows in Menu or Result, every seated fighter has a badge, the stick is centred outside a round, and the
    /// game never sits in one state for longer than a round can last. Afterwards the menu must come back clean. Also
    /// the regression for the replay race: entering the menu during a replay keeps the menu's placement.
    /// </summary>
    public class StateFlowSoakTests
    {
        static IEnumerator Until(System.Func<bool> ok, float seconds, string what)
        {
            float end = Time.realtimeSinceStartup + seconds;
            while (!ok()) { if (Time.realtimeSinceStartup > end) Assert.Fail("timed out waiting for " + what); yield return null; }
        }

        static void Invariants(MatchDirector d, HudView hud, string after)
        {
            string ctx = $" (state {d.State}, after {after})";
            Assert.AreEqual(d.State == MatchState.Replay, ReplaySystem.Active, "replay flag matches the director" + ctx);
            if (!ReplaySystem.Active) Assert.IsTrue(MjScene.Instance.enabled, "physics runs whenever no replay does" + ctx);
            if (hud.SheetShown) Assert.IsTrue(d.State == MatchState.Menu || d.State == MatchState.Result, "a sheet only shows in Menu or Result" + ctx);
            if (d.State == MatchState.Fight) Assert.AreEqual(d.Playing, hud.BadgeCount, "every seated fighter has a badge" + ctx);
            if (d.State != MatchState.Fight) Assert.AreEqual(Vector2.zero, PuppetControl.Stick, "the stick is centred outside a round" + ctx);
        }

        [UnityTest, Timeout(900000)]
        public IEnumerator RandomTapsNeverStrandTheGame()
        {
            using var guard = new ConsoleGuard();
            bool intro = GameSettings.Intro, replays = GameSettings.Replays;
            SceneManager.LoadScene("Game");
            yield return Until(() => MatchDirector.Instance != null && MatchDirector.Instance.State == MatchState.Menu, 60, "the menu");
            var d = MatchDirector.Instance; var hud = Object.FindAnyObjectByType<HudView>(); var replay = Object.FindAnyObjectByType<ReplaySystem>();
            var rng = new System.Random(20261019);
            string[] names = { "quick", "launch all", "skip intro", "menu", "continue", "replay", "stop replay", "debug", "hide sheet", "peek", "restart", "tab", "puppet" };
            int menus = 0, steps = 0; var last = d.State; float since = Time.realtimeSinceStartup;
            try
            {
                GameSettings.Intro = true; GameSettings.Replays = true;
                d.pauseBetweenRounds = 1.5f; if (replay != null) replay.autoDelay = 0.3f;
                Time.timeScale = 4f;
                while (steps < 300 && menus < 25)
                {
                    int a = rng.Next(names.Length); steps++;
                    switch (a)
                    {
                        case 0: if (d.State == MatchState.Menu) d.QuickMatch(); break;
                        case 1: if (d.State == MatchState.Menu) { foreach (var f in d.roster) f.inGame = f.available && rng.Next(3) > 0; if (d.TickedCount > 0) d.Launch(); } break;
                        case 2: d.SkipIntro(); break;
                        case 3: if (d.State != MatchState.Menu) { d.EnterMenu(); menus++; } break;
                        case 4: d.Continue(); break;
                        case 5: replay?.Play(); break;
                        case 6: replay?.Stop(); break;
                        case 7: hud.ToggleDebug(); break;
                        case 8: hud.SetCollapsed(true); break;
                        case 9: hud.SetCollapsed(false); break;
                        case 10: if (d.State == MatchState.Fight) d.NewRound(); break;
                        case 11: hud.Tab = rng.Next(HudView.TabCount); hud.Section = rng.Next(HudView.SectionCount); break;
                        case 12: d.PuppetRequested = !d.PuppetRequested; break;
                    }
                    Invariants(d, hud, names[a]);
                    // wait a random moment, checking every frame; a state that never changes is a dead end
                    float until = Time.realtimeSinceStartup + (float)rng.NextDouble() * 1.5f;
                    while (Time.realtimeSinceStartup < until)
                    {
                        yield return null;
                        Invariants(d, hud, names[a] + " + wait");
                        if (d.State != last) { if (d.State == MatchState.Menu && last != MatchState.Menu) menus++; last = d.State; since = Time.realtimeSinceStartup; }
                        if (d.State != MatchState.Menu && Time.realtimeSinceStartup - since > 120f) Assert.Fail($"stuck in {d.State} for 2 minutes (last action {names[a]})");
                    }
                }
                d.EnterMenu(); hud.SetCollapsed(false);
                yield return null; yield return null;
                Assert.AreEqual(MatchState.Menu, d.State);
                Assert.IsFalse(ReplaySystem.Active);
                Assert.IsTrue(MjScene.Instance.enabled);
                Assert.IsTrue(hud.SheetShown, "the menu sheet is back");
                Assert.AreEqual(0, hud.BadgeCount, "no badges in the menu");
                Debug.Log($"[StateFlowSoak] {steps} steps, {menus} returns to the menu");
            }
            finally
            {
                Time.timeScale = 1f; d.PuppetRequested = false; GameSettings.Intro = intro; GameSettings.Replays = replays;
                if (hud.DebugShown) hud.ToggleDebug();
            }
            guard.AssertClean("the soak");
        }

        /// <summary>Entering the menu while the knock-out replay plays used to let the replay restore the end of the
        /// round over the menu's fresh placement a frame later (the fallen fighter back in the sea).</summary>
        [UnityTest, Timeout(240000)]
        public IEnumerator MenuDuringReplayKeepsTheMenuPlacement()
        {
            using var guard = new ConsoleGuard();
            bool intro = GameSettings.Intro;
            SceneManager.LoadScene("Game");
            yield return Until(() => MatchDirector.Instance != null && MatchDirector.Instance.State == MatchState.Menu, 60, "the menu");
            var d = MatchDirector.Instance; var replay = Object.FindAnyObjectByType<ReplaySystem>();
            Assert.NotNull(replay, "the game scene has a ReplaySystem");
            try
            {
                GameSettings.Intro = false; d.PuppetRequested = false; replay.autoDelay = 999f; d.pauseBetweenRounds = 600f;
                for (int attempt = 0; attempt < 4 && !replay.Available; attempt++)
                {
                    d.QuickMatch(); Time.timeScale = 6f;
                    yield return Until(() => d.State == MatchState.Result, 200, "the round to end");
                    Time.timeScale = 1f;
                }
                if (!replay.Available) Assert.Inconclusive("no knock-out in four rounds, nothing to replay");
                replay.Play();
                Assert.IsTrue(ReplaySystem.Active);
                yield return new WaitForSecondsRealtime(0.5f);
                d.EnterMenu();
                Assert.IsFalse(ReplaySystem.Active, "entering the menu ends the replay");
                for (int i = 0; i < 10; i++) yield return null;
                Assert.AreEqual(MatchState.Menu, d.State);
                Assert.IsTrue(MjScene.Instance.enabled);
                for (int i = 0; i < 2; i++)
                {
                    var p = d.bodies[i].PelvisTransform.position;
                    Assert.Less(new Vector2(p.x, p.z).magnitude, 1.5f, $"fighter {i} stands where the menu placed it");
                }
            }
            finally { Time.timeScale = 1f; GameSettings.Intro = intro; }
            guard.AssertClean("the replay race");
        }
    }
}
