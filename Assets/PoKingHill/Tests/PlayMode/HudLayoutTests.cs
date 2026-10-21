using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace PoKingHill.Tests
{
    /// <summary>
    /// The HUD fits every portrait phone from 16:9 to 21:9 without scrolling or clipping: every menu tab, the intro
    /// card, a six-way fight (one row of badges), the debug panel on each tab during a fight, and the results sheet. HudView lays
    /// itself out for a simulated screen (SimulateScreen) and lists what does not fit (LayoutProblems): elements outside
    /// the column, text wider than its box, a sheet reaching the top bar or the bottom corners, badges on two rows, the
    /// top bar on a cutout. The console stays clean throughout (ConsoleGuard).
    /// </summary>
    public class HudLayoutTests
    {
        static readonly Vector2Int[] Phones = { new(1080, 1920), new(1080, 2340), new(1080, 2520) };

        static IEnumerator Until(System.Func<bool> ok, float seconds, string what)
        {
            float end = Time.realtimeSinceStartup + seconds;
            while (!ok()) { if (Time.realtimeSinceStartup > end) Assert.Fail("timed out waiting for " + what); yield return null; }
        }

        static IEnumerator Frames(int n) { for (int i = 0; i < n; i++) yield return null; }

        /// <summary>Lays the HUD out on each phone and asserts it fits; the view is the one on show (menu tab, fight...).</summary>
        static IEnumerator FitsEveryPhone(HudView hud, string what)
        {
            foreach (var p in Phones)
            {
                hud.SimulateScreen(p.x, p.y);
                yield return Frames(4);
                var problems = hud.LayoutProblems();
                Assert.IsEmpty(problems, $"{what} at {p.x}x{p.y}:\n" + string.Join("\n", problems));
            }
        }

        [UnityTest, Timeout(360000)]
        public IEnumerator EveryScreenFitsEveryPhone()
        {
            using var guard = new ConsoleGuard();
            bool intro = GameSettings.Intro, replays = GameSettings.Replays, telemetry = GameSettings.TelemetryHud, dev = GameSettings.DevTools;
            SceneManager.LoadScene("Game");
            yield return Until(() => MatchDirector.Instance != null && MatchDirector.Instance.State == MatchState.Menu, 60, "the menu");
            var d = MatchDirector.Instance; var hud = Object.FindAnyObjectByType<HudView>();
            var replay = Object.FindAnyObjectByType<ReplaySystem>(); if (replay != null) replay.autoDelay = 999f;
            try
            {
                GameSettings.DevTools = true; GameSettings.TelemetryHud = true;       // the FPS pill, two lines tall
                for (int t = 0; t < HudView.TabCount - 1; t++) { hud.Tab = t; yield return FitsEveryPhone(hud, $"menu tab {t}"); }
                hud.Tab = HudView.TabCount - 1;
                for (int s = 0; s < HudView.SectionCount; s++) { hud.Section = s; yield return FitsEveryPhone(hud, $"settings section {s}"); }

                // six-way: intro card, fight, results
                foreach (var f in d.roster) f.inGame = f.available;
                GameSettings.Intro = true;
                d.Launch();
                Assert.AreEqual(MatchState.Intro, d.State);
                yield return Until(() => d.IntroFeatured >= 0, 10, "a fighter's card");
                yield return FitsEveryPhone(hud, "intro card");
                d.SkipIntro(); yield return Frames(2);
                Assert.AreEqual(MatchState.Fight, d.State);
                Assert.AreEqual(d.Playing, hud.BadgeCount);
                MatchEvents.RaiseCaption("Duelist is off: Free-for-all and All-rounder teamed up on the summit.");     // a caption that wraps
                yield return FitsEveryPhone(hud, $"{d.Playing}-way fight");

                d.pauseBetweenRounds = 600f; Time.timeScale = 6f;
                yield return Until(() => d.State == MatchState.Result, 200, "the round to end");
                Time.timeScale = 1f;
                yield return FitsEveryPhone(hud, "results sheet");

                // a quick match with the debug panel open on each tab
                GameSettings.Intro = false;
                d.QuickMatch(); yield return Frames(3);
                hud.ToggleDebug();
                for (int t = 0; t < 3; t++) { hud.SetDebugTab(t); yield return FitsEveryPhone(hud, $"fight + debug tab {t}"); }
                hud.ToggleDebug();
            }
            finally
            {
                hud.ClearSimulation(); Time.timeScale = 1f;
                GameSettings.Intro = intro; GameSettings.Replays = replays; GameSettings.TelemetryHud = telemetry; GameSettings.DevTools = dev;
            }
            guard.AssertClean("the layout pass");
        }

        /// <summary>A phone that reports a full-screen safe area but has a punch-hole camera in the middle of the top
        /// edge: the FPS pill steps aside. A corner notch pushes the whole top bar down instead.</summary>
        [UnityTest, Timeout(120000)]
        public IEnumerator TopBarClearsCutouts()
        {
            using var guard = new ConsoleGuard();
            SceneManager.LoadScene("Game");
            yield return Until(() => MatchDirector.Instance != null && MatchDirector.Instance.State == MatchState.Menu, 60, "the menu");
            var hud = Object.FindAnyObjectByType<HudView>();
            bool dev = GameSettings.DevTools; GameSettings.DevTools = true;      // the FPS pill is a developer tool
            try
            {
                // 1080 x 2400 at 0.8 panel units per pixel: the pill sits about 70-140 px below the top edge
                hud.SimulateScreen(1080, 2400, default, new[] { new Rect(505, 2400 - 150, 70, 70) });
                yield return Frames(6);
                var problems = hud.LayoutProblems();
                Assert.IsEmpty(problems, "punch hole:\n" + string.Join("\n", problems));

                hud.SimulateScreen(1080, 2400, default, new[] { new Rect(0, 2400 - 140, 300, 140) });
                yield return Frames(6);
                problems = hud.LayoutProblems();
                Assert.IsEmpty(problems, "corner notch:\n" + string.Join("\n", problems));

                // a phone that does report its notch and home bar: the safe area pads the column and the menu sheet
                // still ends above the bottom corners; hidden, it leaves a peek pill, and the debug panel fits too
                hud.SimulateScreen(1080, 2400, new Rect(0, 90, 1080, 2400 - 90 - 140), new[] { new Rect(505, 2400 - 120, 70, 70) });
                yield return Frames(6);
                problems = hud.LayoutProblems();
                Assert.IsEmpty(problems, "safe area with notch and home bar:\n" + string.Join("\n", problems));
                hud.SetCollapsed(true); hud.ToggleDebug();
                yield return Frames(6);
                problems = hud.LayoutProblems();
                hud.ToggleDebug(); hud.SetCollapsed(false);
                Assert.IsEmpty(problems, "peek pill and debug panel in the safe area:\n" + string.Join("\n", problems));
            }
            finally { hud.ClearSimulation(); GameSettings.DevTools = dev; }
            guard.AssertClean("the cutout pass");
        }
    }
}
