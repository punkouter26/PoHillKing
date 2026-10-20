using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace PoKingHill.Tests
{
    /// <summary>The game's state flow in the real scene: Menu -> (Intro ->) Fight -> Result -> Menu, with no error or
    /// exception logged (the test runner fails on any unexpected error log). Physics runs at 4x speed to keep the test
    /// short. The intro (tale of the tape, GameSettings.Intro) is skipped where the test is about the rounds.</summary>
    public class MatchFlowTests
    {
        static IEnumerator Until(System.Func<bool> ok, float seconds, string what)
        {
            float end = Time.realtimeSinceStartup + seconds;
            while (!ok()) { if (Time.realtimeSinceStartup > end) Assert.Fail("timed out waiting for " + what); yield return null; }
        }

        [UnityTest, Timeout(240000)]
        public IEnumerator MenuFightResultMenu()
        {
            SceneManager.LoadScene("Demo_duel");
            yield return Until(() => DemoDirector.Instance != null && DemoDirector.Instance.State == MatchState.Menu, 60, "the menu");
            var d = DemoDirector.Instance;
            d.pauseBetweenRounds = 1f; Time.timeScale = 4f;
            var replay = Object.FindAnyObjectByType<ReplaySystem>(); if (replay != null) replay.autoDelay = 999f;
            d.QuickMatch(); d.SkipIntro();
            Assert.AreEqual(MatchState.Fight, d.State);
            Assert.GreaterOrEqual(d.Playing, 2);
            yield return Until(() => d.State == MatchState.Result, 150, "the round to end");
            Assert.IsNotNull(d.LastResult.Agents); Assert.AreEqual(d.Playing, d.LastResult.Agents.Length);
            d.Continue();
            yield return Until(() => d.State == MatchState.Menu, 20, "the menu after one round");
            Time.timeScale = 1f;
        }

        [UnityTest, Timeout(120000)]
        public IEnumerator ReturningToTheMenuMidRoundIsClean()
        {
            SceneManager.LoadScene("Demo_duel");
            yield return Until(() => DemoDirector.Instance != null && DemoDirector.Instance.State == MatchState.Menu, 60, "the menu");
            var d = DemoDirector.Instance;
            d.Launch(); yield return new WaitForSeconds(1f);
            d.EnterMenu();                    // from the intro or the round, whichever is on
            Assert.AreEqual(MatchState.Menu, d.State);
            yield return null;
            d.QuickMatch(); d.SkipIntro(); yield return new WaitForSeconds(0.5f);
            Assert.AreEqual(MatchState.Fight, d.State);
            d.NewRound();                     // reset mid-round
            Assert.AreEqual(MatchState.Fight, d.State);
        }

        /// <summary>The tale of the tape plays when it is on: Intro with every fighter standing and no round clock, then the
        /// first round on its own, brains armed. The setting is restored afterwards.</summary>
        [UnityTest, Timeout(120000)]
        public IEnumerator IntroLeadsIntoTheFirstRound()
        {
            bool was = GameSettings.Intro;
            try
            {
                GameSettings.Intro = true;
                SceneManager.LoadScene("Demo_duel");
                yield return Until(() => DemoDirector.Instance != null && DemoDirector.Instance.State == MatchState.Menu, 60, "the menu");
                var d = DemoDirector.Instance;
                d.QuickMatch();
                Assert.AreEqual(MatchState.Intro, d.State);
                Assert.AreEqual(0.0, d.RoundTime, "no round clock during the intro");
                Assert.IsNull(d.fighters[0].policy, "brains stand passively during the intro");
                yield return Until(() => d.IntroFeatured >= 0, 10, "the first fighter's card");
                yield return Until(() => d.State == MatchState.Fight, 20, "the round after the intro");
                Assert.IsNotNull(d.fighters[0].policy, "brains armed for the round");
            }
            finally { GameSettings.Intro = was; }
        }
    }
}
