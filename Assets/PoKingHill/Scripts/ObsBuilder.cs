using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>Mirror of training/koth/obs.py. Reads raw mjData in the MuJoCo frame; never touches a Transform.</summary>
    public static unsafe class ObsBuilder
    {
        public const int ObsDim = 103;
        public const float GaitFreqHz = 1.5f;

        // Rotate world vector v into the frame of unit quaternion q=(w,x,y,z). Same math as obs.quat_rotate_inverse.
        static void QuatRotateInverse(double* q, double vx, double vy, double vz, out float ox, out float oy, out float oz)
        {
            double w = q[0], x = q[1], y = q[2], z = q[3];
            // t = 2 * cross(xyz, v)
            double tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
            // r = v - w*t + cross(xyz, t)
            ox = (float)(vx - w * tx + (y * tz - z * ty));
            oy = (float)(vy - w * ty + (z * tx - x * tz));
            oz = (float)(vz - w * tz + (x * ty - y * tx));
        }

        /// <summary>Mirror of obs.build_combat(): 9 values about the opponent and the ring, written at obs[o..o+8].
        /// Heading frame = world rotated by my yaw only.</summary>
        public static void FillCombat(float[] obs, int o, MujocoLib.mjData_* d, JointMap me, JointMap opp)
        {
            double* q = d->qpos + me.RootQposAdr; double* v = d->qvel + me.RootDofAdr;
            double* oq = d->qpos + opp.RootQposAdr; double* ov = d->qvel + opp.RootDofAdr;
            double w = q[3], x = q[4], y = q[5], z = q[6];
            double yaw = System.Math.Atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), c = System.Math.Cos(yaw), s = System.Math.Sin(yaw);
            void H(double vx, double vy, int at) { obs[at] = (float)(c * vx + s * vy); obs[at + 1] = (float)(-s * vx + c * vy); }
            H(oq[0] - q[0], oq[1] - q[1], o);
            H(ov[0] - v[0], ov[1] - v[1], o + 2);
            H(-q[0], -q[1], o + 4);
            obs[o + 6] = (float)System.Math.Sqrt(q[0] * q[0] + q[1] * q[1]);
            obs[o + 7] = (float)System.Math.Sqrt(oq[0] * oq[0] + oq[1] * oq[1]);
            obs[o + 8] = (float)(1 - 2 * (oq[4] * oq[4] + oq[5] * oq[5]));
        }

        public static void Fill(float[] obs, MujocoLib.mjData_* d, JointMap jm, float[] command, float[] lastAction, float phase)
        {
            double* q = d->qpos + jm.RootQposAdr;      // x y z qw qx qy qz
            double* v = d->qvel + jm.RootDofAdr;       // vx vy vz (world) wx wy wz (body)
            QuatRotateInverse(q + 3, v[0], v[1], v[2], out obs[0], out obs[1], out obs[2]);
            obs[3] = (float)v[3]; obs[4] = (float)v[4]; obs[5] = (float)v[5];
            QuatRotateInverse(q + 3, 0, 0, -1, out obs[6], out obs[7], out obs[8]);
            obs[9] = command[0]; obs[10] = command[1]; obs[11] = command[2];
            for (int i = 0; i < jm.N; i++)
            {
                obs[12 + i] = (float)d->qpos[jm.QposAdr[i]] - jm.DefaultPose[i];
                obs[41 + i] = (float)d->qvel[jm.DofAdr[i]];
                obs[70 + i] = lastAction[i];
            }
            obs[99] = Mathf.Cos(phase); obs[100] = Mathf.Sin(phase);
            obs[101] = Mathf.Cos(phase + Mathf.PI); obs[102] = Mathf.Sin(phase + Mathf.PI);
        }
    }
}
