using NUnit.Framework;
using UnityEngine;

namespace PoKingHill.Tests
{
    /// <summary>ActionMap (C#) against training/koth/action_map.py's NumpyMap on the same inputs, and the sidecar format
    /// export_policy.py writes. Reference numbers: NumpyMap(dict(map="tanh", scale=[0.3] * 29, window=True), 0.5, kp 200,
    /// kv 5, tau 50) with rest 0.1, q 0.05.</summary>
    public class ActionMapTests
    {
        static TextAsset Sidecar(string json) => new TextAsset(json) { name = "test_action" };

        static string Scale(float v) => "[" + string.Join(",", System.Linq.Enumerable.Repeat(v.ToString(System.Globalization.CultureInfo.InvariantCulture), 29)) + "]";

        [Test]
        public void NoSidecarIsTheOriginalPath() => Assert.IsNull(ActionMap.Parse(null));

        [Test]
        public void TanhScaleAndWindowMatchPython()
        {
            var m = ActionMap.Parse(Sidecar("{\"map\": \"tanh\", \"scale\": " + Scale(0.3f) + ", \"window\": true}"));
            Assert.IsTrue(m.Tanh && m.Window && m.Scale.Length == 29);
            float a = m.ToAction(2f);
            Assert.AreEqual(0.964027584, a, 1e-6);                                      // np.tanh(2) as float32
            // joint moving at +4 rad/s: the window is [-0.10, 0.40], the target 0.1 + 0.3 tanh(2) = 0.3892 is inside
            Assert.AreEqual(0.389208275, m.Target(0, 0.1, a, 0.5f, 0.05, 4.0, 200, 5, 50, -1, 1), 1e-6);
            // moving at -4 rad/s the window's top is 0.05 + (5 * -4 + 50) / 200 = 0.20: the target is cut there
            Assert.AreEqual(0.2, m.Target(0, 0.1, a, 0.5f, 0.05, -4.0, 200, 5, 50, -1, 1), 1e-9);
            // and the control range still applies last
            Assert.AreEqual(0.15, m.Target(0, 0.1, a, 0.5f, 0.05, 4.0, 200, 5, 50, -1, 0.15), 1e-9);
        }

        [Test]
        public void ClipMapKeepsTheJointMapScale()
        {
            var m = ActionMap.Parse(Sidecar("{\"map\": \"clip\", \"scale\": null, \"window\": false}"));
            Assert.IsFalse(m.Tanh || m.Window); Assert.IsNull(m.Scale);
            Assert.AreEqual(1f, m.ToAction(1.7f)); Assert.AreEqual(-1f, m.ToAction(-3f)); Assert.AreEqual(0.2f, m.ToAction(0.2f));
            Assert.AreEqual(0.5, m.Target(0, 0.1, 1f, 0.5f, 0, 0, 200, 5, 50, -1, 0.5), 1e-9);    // NumpyMap: 0.1 + 0.5 * 1 clipped to 0.5
        }

        [Test]
        public void WrongScaleLengthIsRejected() =>
            Assert.Throws<System.ArgumentException>(() => ActionMap.Parse(Sidecar("{\"map\": \"tanh\", \"scale\": [0.1, 0.2], \"window\": false}")));
    }
}
