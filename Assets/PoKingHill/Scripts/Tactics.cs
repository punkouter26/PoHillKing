namespace PoKingHill
{
    /// <summary>What a fighting brain is asked to walk at. Press is the trained law (straight at the opponent); the
    /// others move only the goal point fed to the same GoalCommand law, so the commands stay inside the trained ranges.</summary>
    public enum Tactic { Press, FlankLeft, FlankRight, Regroup, Lurk }

    public static unsafe class Tactics
    {
        public const float FlankOffset = 0.55f, LurkRadius = 0.6f;

        /// <summary>World xy vector from me to the goal point of <paramref name="t"/> and its stop distance.
        /// me / opp = root qpos (x, y, z, ...) in the MuJoCo frame. Press gives (opp - me, baseStop), as trained.</summary>
        public static void Goal(Tactic t, double* me, double* opp, float baseStop, out double gx, out double gy, out float stop)
        {
            double dx = opp[0] - me[0], dy = opp[1] - me[1];
            switch (t)
            {
                case Tactic.FlankLeft:
                case Tactic.FlankRight:
                {
                    // a point beside the opponent, so the push comes from the side instead of head on
                    double len = System.Math.Sqrt(dx * dx + dy * dy) + 1e-6, side = t == Tactic.FlankLeft ? 1 : -1;
                    gx = dx + side * (-dy / len) * FlankOffset; gy = dy + side * (dx / len) * FlankOffset; stop = 0f; return;
                }
                case Tactic.Regroup: gx = -me[0]; gy = -me[1]; stop = 0.2f; return;        // back to the middle of the summit
                case Tactic.Lurk:
                {
                    // hold a spot on my own bearing at LurkRadius from the centre, out of the fight
                    double r = System.Math.Sqrt(me[0] * me[0] + me[1] * me[1]);
                    if (r < 0.05) { gx = 0; gy = 0; stop = 0.1f; return; }
                    gx = me[0] / r * LurkRadius - me[0]; gy = me[1] / r * LurkRadius - me[1]; stop = 0.1f; return;
                }
                default: gx = dx; gy = dy; stop = baseStop; return;
            }
        }
    }
}
