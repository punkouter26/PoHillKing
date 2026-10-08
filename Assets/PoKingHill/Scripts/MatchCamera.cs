using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Autonomous 9:16 match camera. Combat framing keeps both robots and the summit in view; when a round ends
    /// with a loser, the camera follows that robot down the slope, then returns to the summit.
    /// Reads robot positions from mjData (MuJoCo x, y, z maps to Unity x, z, y); render only, no physics.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public unsafe class MatchCamera : MonoBehaviour
    {
        public PolicyRunner robotA, robotB;
        public DemoDirector director;
        public float height = 2.4f, baseDistance = 5.6f, distancePerMetre = 1.2f, smooth = 2.5f, followSmooth = 4f;

        Vector3 _look = new(0f, 0.6f, 0f);

        static Vector3 Pelvis(PolicyRunner r)
        {
            double* q = MjScene.Instance.Data->qpos + r.Map.RootQposAdr;
            return new Vector3((float)q[0], (float)q[2], (float)q[1]);
        }

        void LateUpdate()
        {
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null || robotA == null || robotA.Map == null || robotB.Map == null) return;
            Vector3 a = Pelvis(robotA), b = Pelvis(robotB);
            var loser = director != null ? director.FollowTarget : null;
            Vector3 look, pos; float k;
            if (loser != null)
            {                                                   // ejection sequence: track the defeated robot
                Vector3 p = Pelvis(loser); Vector3 outward = new Vector3(p.x, 0f, p.z).normalized;
                look = p; pos = p + outward * 3.2f + Vector3.up * 1.6f; k = followSmooth;
            }
            else
            {                                                   // combat framing: both robots, summit rim in frame
                look = (a + b) * 0.5f; look.y = Mathf.Max(0.5f, look.y * 0.8f);
                float sep = Vector3.Distance(a, b);
                pos = new Vector3(look.x * 0.5f, height, look.z * 0.5f - (baseDistance + distancePerMetre * sep)); k = smooth;
            }
            float t = 1f - Mathf.Exp(-k * Time.unscaledDeltaTime);
            _look = Vector3.Lerp(_look, look, t);
            transform.position = Vector3.Lerp(transform.position, pos, t);
            transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.LookRotation(_look - transform.position, Vector3.up), t);
        }
    }
}
