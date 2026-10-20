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
        // No exceptions since 2026-10-20: Kim's foot boxes, which overlapped 3.6 mm in her zero and rest pose, now keep a
        // 4 mm gap (koth/build_human.py FOOT_GAP). Add an entry here only for a measured overlap that waits for a retrain.
        static readonly (string robot, string a, string b, double mm)[] Known = { };

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
