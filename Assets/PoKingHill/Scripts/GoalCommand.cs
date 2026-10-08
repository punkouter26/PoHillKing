using UnityEngine;

namespace PoKingHill
{
    public enum GoalMode { None, Center, Opponent }

    /// <summary>Mirror of training/koth/obs.py goal_command(): velocity command that walks toward a goal point.</summary>
    public static unsafe class GoalCommand
    {
        /// <param name="quat">root quaternion w x y z (MuJoCo frame)</param>
        /// <param name="gx">world x of (goal - robot)</param>
        /// <param name="gy">world y of (goal - robot)</param>
        public static void Compute(double* quat, double gx, double gy, float stopDist, float vmax, float[] cmd)
        {
            double w = quat[0], x = quat[1], y = quat[2], z = quat[3];
            double yaw = System.Math.Atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
            double dist = System.Math.Sqrt(gx * gx + gy * gy);
            double ang = System.Math.Atan2(gy, gx) - yaw;
            ang = System.Math.Atan2(System.Math.Sin(ang), System.Math.Cos(ang));
            double speed = System.Math.Clamp(dist - stopDist, 0.0, vmax);
            cmd[0] = (float)System.Math.Clamp(speed * System.Math.Cos(ang), -1.0, 1.0);
            cmd[1] = (float)System.Math.Clamp(speed * System.Math.Sin(ang), -0.5, 0.5);
            cmd[2] = speed > 0.05 ? (float)System.Math.Clamp(2.0 * ang, -1.0, 1.0) : 0f;
        }
    }
}
