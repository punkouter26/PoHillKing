using System;
using Mujoco;

namespace PoKingHill
{
    /// <summary>The numbers on a fighter's intro card, read from the compiled model (the same one the brains fight in).</summary>
    public static unsafe class TaleOfTheTape
    {
        /// <summary>Body mass (pelvis subtree, kg) and the strongest joint's force limit (N m).</summary>
        public static void Body(PolicyRunner r, out float kg, out float strongestNm)
        {
            kg = 0; strongestNm = 0;
            if (r == null || r.Map == null || !MjScene.InstanceExists || MjScene.Instance.Model == null) return;
            var m = MjScene.Instance.Model;
            kg = (float)m->body_subtreemass[r.Map.PelvisBodyId];
            foreach (int a in r.Map.ActId)
                strongestNm = Math.Max(strongestNm, (float)Math.Max(Math.Abs(m->actuator_forcerange[2 * a]), Math.Abs(m->actuator_forcerange[2 * a + 1])));
        }

        /// <summary>What kind of brain an agent has, from its input size and style.</summary>
        public static string Brain(Agent f)
        {
            if (f == null) return "-";
            string kind = f.obsDim >= ObsBuilder.ObsDim + 18 ? "4-way" : f.obsDim > ObsBuilder.ObsDim ? "duel" : "walker";
            return f.shove ? $"{kind}, {f.stylePose}" : kind;
        }
    }
}
