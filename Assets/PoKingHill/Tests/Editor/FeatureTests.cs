using System.IO;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace PoKingHill.Tests
{
    /// <summary>The pure parts behind the broadcast features: the Tactician's C# networks against Sentis, the ladder's
    /// record and credits, the balance geometry and the voice's PCM decoding.</summary>
    public class FeatureTests
    {
        // MlpNet (worker-thread rollouts) must compute what the ONNX brain computes in PolicyRunner.
        [TestCase("Assets/PoKingHill/Models/attacker_policy.onnx", "Assets/PoKingHill/Models/attacker_mlp.bytes")]
        [TestCase("Assets/PoKingHill/Models/ffa_policy.onnx", "Assets/PoKingHill/Models/ffa_mlp.bytes")]
        [TestCase("Assets/PoKingHill/Models/ffa_value.onnx", "Assets/PoKingHill/Models/ffa_value_mlp.bytes")]
        [TestCase("Assets/PoKingHill/Models/trump/fighter_policy.onnx", "Assets/PoKingHill/Models/trump/fighter_mlp.bytes")]
        public void MlpNetMatchesSentis(string onnx, string bytes)
        {
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(onnx); var weights = AssetDatabase.LoadAssetAtPath<TextAsset>(bytes);
            Assert.NotNull(model, onnx); Assert.NotNull(weights, bytes + " (run training/scripts/export_mlp.py)");
            var net = MlpNet.Load(weights.bytes);
            using var worker = new Worker(ModelLoader.Load(model), BackendType.CPU);
            int dim = net.InputDim; var x = new float[dim]; var y = new float[net.OutputDim]; var scratch = net.NewScratch();
            using var input = new Tensor<float>(new TensorShape(1, dim));
            var rng = new System.Random(1); float worst = 0, scale = 1;
            for (int trial = 0; trial < 20; trial++)
            {
                for (int i = 0; i < dim; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
                net.Forward(x, y, scratch);
                input.Upload(x); worker.Schedule(input);
                var output = worker.PeekOutput() as Tensor<float>; output.CompleteAllPendingOperations(); var reference = output.AsReadOnlySpan();
                for (int i = 0; i < y.Length; i++) { worst = Mathf.Max(worst, Mathf.Abs(y[i] - reference[i])); scale = Mathf.Max(scale, Mathf.Abs(reference[i])); }
            }
            Assert.Less(worst, 2e-4f * scale, $"MlpNet differs from Sentis by {worst}");
        }

        [Test]
        public void MlpNetRejectsOtherFiles() => Assert.Throws<System.FormatException>(() => MlpNet.Load(new byte[16]));

        [Test]
        public void LadderKeepsRecordAndCredits()
        {
            string tmp = Path.Combine(Path.GetTempPath(), "koth_ladder_features.json"); File.Delete(tmp);
            Ladder.FileOverride = tmp;
            try
            {
                var roster = new[] { new Fighter { name = "A" }, new Fighter { name = "B" }, new Fighter { name = "C" } };
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

        [Test]
        public void VoicePcmDecodes()
        {
            var pcm = new byte[] { 0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80, 0x00, 0x40 };      // 0, 32767, -32768, 16384
            var clip = VoiceCaster.ToClip(pcm, "t");
            try
            {
                Assert.AreEqual(4, clip.samples); Assert.AreEqual(VoiceCaster.SampleRate, clip.frequency);
                var s = new float[4]; clip.GetData(s, 0);
                Assert.AreEqual(0f, s[0], 1e-4f); Assert.AreEqual(32767f / 32768f, s[1], 1e-4f); Assert.AreEqual(-1f, s[2], 1e-4f); Assert.AreEqual(0.5f, s[3], 1e-4f);
            }
            finally { Object.DestroyImmediate(clip); }
            Assert.IsNull(VoiceCaster.ToClip(new byte[2], "empty"));
        }

        [Test]
        public void WeatherCodesReadAsWords()
        {
            Assert.AreEqual("clear sky", WeatherFeed.Describe(0)); Assert.AreEqual("rain", WeatherFeed.Describe(63));
            Assert.AreEqual("thunderstorm", WeatherFeed.Describe(95)); Assert.AreEqual("fog", WeatherFeed.Describe(45));
        }
    }
}
