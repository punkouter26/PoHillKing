using System.IO;
using System.Linq;
using NUnit.Framework;
using PoKingHill.EditorTools;

namespace PoKingHill.Tests
{
    /// <summary>Self-collision of the imported fighters (AGENTS.md section 3): in the zero pose no pair the simulation
    /// collides may interpenetrate. training/scripts/collision_audit.py has the full audit (rest pose, T-pose, swing
    /// arcs, pairs that would touch with every self-collision on) and the HTML report.</summary>
    public class CollisionTests
    {
        // Known, measured by collision_audit.py (261019): Kim's foot boxes interpenetrate 3.5 mm in her zero and rest
        // pose (her stance is narrower than the feet). Fixing it changes her physics, so it waits for her next retrain;
        // the allowance only stops it getting worse.
        static readonly (string robot, string a, string b, double mm)[] Known =
        {
            ("kim", "a_left_foot_box_collision", "a_right_foot_box_collision", 3.6),
            ("kim", "a_left_foot_box_collision", "a_right_foot3_collision", 0.6),
            ("kim", "a_left_foot1_collision", "a_right_foot_box_collision", 0.6),
        };

        [TestCase("g1"), TestCase("kim"), TestCase("trump"), TestCase("grandpa")]
        public void CollidingSelfPairsDoNotOverlapInTheZeroPose(string robot)
        {
            string xml = Path.GetFullPath($"training/assets/{robot}/scene_koth_2p_unity.xml");
            Assert.IsTrue(File.Exists(xml), xml + " missing (python -m koth.build_mjcf)");
            var found = CollisionAudit.ZeroPose(xml);
            var unexpected = found.Where(o => !Known.Any(k => k.robot == robot && k.a == o.A && k.b == o.B && o.DepthMm <= k.mm)).ToList();
            Assert.IsEmpty(unexpected, robot + ": " + string.Join(", ", unexpected));
        }
    }
}
