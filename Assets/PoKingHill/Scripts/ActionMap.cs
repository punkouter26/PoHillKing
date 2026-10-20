using System;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Mirror of training/koth/action_map.py (keep both in step): how a brain's output becomes a joint target. A brain exported with
    /// &lt;name&gt;_action.json beside its ONNX uses this; without one (every brain up to 2026-10-20) PolicyRunner keeps the
    /// original path, a = clamp(raw, -1, 1), target = clamp(default + offset + action_scale * a, ctrl range).
    ///   map "tanh"   a = tanh(raw)
    ///   scale[29]    per-joint action scale instead of joint_map's single one
    ///   window       target clamped to the torque-feasible window q + (kv qd -+ tau) / kp (nominal gains)
    /// </summary>
    public sealed class ActionMap
    {
        [Serializable] class Json { public string map = "clip"; public float[] scale; public bool window; }

        public readonly bool Tanh, Window;
        public readonly float[] Scale;     // null = joint_map's action_scale for every joint

        ActionMap(bool tanh, float[] scale, bool window) { Tanh = tanh; Scale = scale; Window = window; }

        /// <summary>Parse a sidecar; null (the original path) for no file.</summary>
        public static ActionMap Parse(TextAsset json)
        {
            if (json == null || string.IsNullOrWhiteSpace(json.text)) return null;
            var j = JsonUtility.FromJson<Json>(json.text);
            if (j.scale != null && j.scale.Length == 0) j.scale = null;
            if (j.scale != null && j.scale.Length != 29) throw new ArgumentException($"{json.name}: scale has {j.scale.Length} values, not 29");
            return new ActionMap(j.map == "tanh", j.scale, j.window);
        }

        public float ToAction(float raw) => Tanh ? (float)Math.Tanh(raw) : Mathf.Clamp(raw, -1f, 1f);

        /// <summary>Joint target for joint i: rest = default (+ style offset), f = the (filtered) action, q / qd the joint's
        /// current state, kp / kv / tau its nominal gains and force limit, lo / hi its control range.</summary>
        public double Target(int i, double rest, float f, float baseScale, double q, double qd, double kp, double kv, double tau, double lo, double hi)
        {
            double t = rest + (Scale != null ? Scale[i] : baseScale) * f;
            if (Window && kp > 0) t = Math.Clamp(t, q + (kv * qd - tau) / kp, q + (kv * qd + tau) / kp);
            return Math.Clamp(t, lo, hi);
        }
    }
}
