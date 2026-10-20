using System;

namespace PoKingHill
{
    /// <summary>
    /// A brain's MLP in plain C# (weights from training/scripts/export_mlp.py, "KMLP" files): x = (obs - mean) / divisor,
    /// then Linear/ELU layers, the last one linear, as koth/actor.py. Immutable after loading; Forward takes the caller's
    /// scratch buffer, so worker threads can share one net (Sentis workers belong to the main thread). Used by the
    /// Tactician's rollouts; an EditMode test checks it against Sentis on the same ONNX.
    /// </summary>
    public sealed class MlpNet
    {
        public readonly int InputDim, OutputDim, MaxWidth;
        readonly int[] _dims; readonly float[] _mean, _invDiv; readonly float[][] _w, _b;

        MlpNet(int[] dims, float[] mean, float[] div, float[][] w, float[][] b)
        {
            _dims = dims; _mean = mean; _w = w; _b = b; InputDim = dims[0]; OutputDim = dims[^1];
            _invDiv = new float[div.Length]; for (int i = 0; i < div.Length; i++) _invDiv[i] = 1f / div[i];
            int max = 0; foreach (int d in dims) max = Math.Max(max, d); MaxWidth = max;
        }

        public static MlpNet Load(byte[] bytes)
        {
            int at = 0;
            int I() { int v = BitConverter.ToInt32(bytes, at); at += 4; return v; }
            float[] F(int n) { var a = new float[n]; Buffer.BlockCopy(bytes, at, a, 0, 4 * n); at += 4 * n; return a; }
            if (bytes == null || bytes.Length < 12 || I() != 0x504C4D4B || I() != 1) throw new FormatException("not a KMLP v1 file (training/scripts/export_mlp.py)");
            int layers = I(); var dims = new int[layers + 1]; for (int i = 0; i <= layers; i++) dims[i] = I();
            var mean = F(dims[0]); var div = F(dims[0]);
            var w = new float[layers][]; var b = new float[layers][];
            for (int l = 0; l < layers; l++) { w[l] = F(dims[l + 1] * dims[l]); b[l] = F(dims[l + 1]); }
            if (at != bytes.Length) throw new FormatException($"KMLP size mismatch: read {at} of {bytes.Length} bytes");
            return new MlpNet(dims, mean, div, w, b);
        }

        /// <summary>Scratch for one caller (two layer-width buffers).</summary>
        public float[] NewScratch() => new float[2 * MaxWidth];

        /// <summary>y[0..OutputDim) = net(x[0..InputDim)). Thread-safe with a scratch buffer per thread.</summary>
        public void Forward(float[] x, float[] y, float[] scratch)
        {
            int w0 = MaxWidth; int n = InputDim;
            for (int i = 0; i < n; i++) scratch[i] = (x[i] - _mean[i]) * _invDiv[i];
            int src = 0, dst = w0;
            for (int l = 0; l < _w.Length; l++)
            {
                int inD = _dims[l], outD = _dims[l + 1]; var W = _w[l]; var B = _b[l]; bool last = l == _w.Length - 1;
                for (int o = 0; o < outD; o++)
                {
                    int row = o * inD; float s0 = 0, s1 = 0, s2 = 0, s3 = 0; int i = 0;
                    for (; i + 3 < inD; i += 4)
                    {
                        s0 += W[row + i] * scratch[src + i]; s1 += W[row + i + 1] * scratch[src + i + 1];
                        s2 += W[row + i + 2] * scratch[src + i + 2]; s3 += W[row + i + 3] * scratch[src + i + 3];
                    }
                    for (; i < inD; i++) s0 += W[row + i] * scratch[src + i];
                    float v = s0 + s1 + s2 + s3 + B[o];
                    if (last) y[o] = v; else scratch[dst + o] = v > 0 ? v : MathF.Exp(v) - 1f;
                }
                (src, dst) = (dst, src);
            }
        }
    }
}
