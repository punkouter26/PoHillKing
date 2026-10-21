using System.IO;
using NUnit.Framework;

namespace PoKingHill.Tests
{
    /// <summary>The pure parts behind the broadcast features: the ladder's record and credits, the balance geometry.</summary>
    public class FeatureTests
    {
        [Test]
        public void LadderKeepsRecordAndCredits()
        {
            string tmp = Path.Combine(Path.GetTempPath(), "koth_ladder_features.json"); File.Delete(tmp);
            Ladder.FileOverride = tmp;
            try
            {
                var roster = new[] { new Agent { name = "A" }, new Agent { name = "B" }, new Agent { name = "C" } };
                Ladder.Record(new RoundResult(0, "A", "", RoundEnd.Knockout, 3f, new[] { 0, 1, 2 }), roster);
                Ladder.Record(new RoundResult(-1, "No winner", "", RoundEnd.FullTime, 25f, new[] { 0, 1 }), roster);      // a tie changes no record
                Assert.AreEqual(1, Ladder.Wins("A")); Assert.AreEqual(0, Ladder.Losses("A"));
                Assert.AreEqual(1, Ladder.Losses("B")); Assert.AreEqual(1, Ladder.Losses("C"));
                Ladder.AddCredits("A", 2, 1); Ladder.AddCredits("A", 1, 0);
                Assert.AreEqual(3, Ladder.Knockouts("A")); Assert.AreEqual(1, Ladder.Assists("A"));
                Ladder.FileOverride = tmp;                          // reload from disk: the counts were saved
                Assert.AreEqual(3, Ladder.Knockouts("A")); Assert.AreEqual(1, Ladder.Wins("A"));
            }
            finally { Ladder.FileOverride = null; File.Delete(tmp); }
        }

        [Test]
        public void CapturePointDistanceToTheFeet()
        {
            Assert.AreEqual(0.0, Biometrics.SegmentDistance(0, 0, -0.1, 0, 0.1, 0), 1e-12);        // between two feet
            Assert.AreEqual(0.3, Biometrics.SegmentDistance(0, 0.3, -0.1, 0, 0.1, 0), 1e-12);      // in front of the stance
            Assert.AreEqual(0.2, Biometrics.SegmentDistance(0.3, 0, -0.1, 0, 0.1, 0), 1e-12);      // past the right foot
            Assert.AreEqual(0.5, Biometrics.SegmentDistance(0.3, 0.4, 0, 0, 0, 0), 1e-12);         // one foot (degenerate segment)
        }
    }
}
