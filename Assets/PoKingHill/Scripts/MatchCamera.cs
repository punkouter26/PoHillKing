using System.Collections.Generic;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Autonomous 9:16 match camera. It stays on the fighter nearest the centre of the summit, pulled back far enough
    /// to keep the others that are still in the round in view. When a fighter that was knocked off reaches the sea,
    /// the camera cuts to it until it has gone under (at most cutSeconds), then cuts back; a newer faller takes over.
    /// Reads robot positions from mjData (MuJoCo x, y, z maps to Unity x, z, y); render only, no physics.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public unsafe class MatchCamera : MonoBehaviour
    {
        public DemoDirector director;
        public float height = 2.4f, baseDistance = 5.6f, distancePerMetre = 1.2f, smooth = 2.5f;
        [Tooltip("Cut to a fallen fighter when its pelvis is this far above the water, or lower")] public float cutAbove = 0.5f;
        [Tooltip("...and back once its pelvis is this far under the surface")] public float cutUnder = 0.6f;
        public float cutSeconds = 3f;
        [Tooltip("Another fighter takes over as the centre only when it is this much closer to it (no flicker between two)")] public float centreMargin = 0.3f;

        Vector3 _look = new(0f, 0.6f, 0f);
        PolicyRunner _centre, _cut; float _cutUntil; bool _snap;
        readonly HashSet<PolicyRunner> _shown = new();

        static Vector3 Pelvis(PolicyRunner r)
        {
            double* q = MjScene.Instance.Data->qpos + r.Map.RootQposAdr;
            return new Vector3((float)q[0], (float)q[2], (float)q[1]);
        }
        static float FromCentre(Vector3 p) => Mathf.Sqrt(p.x * p.x + p.z * p.z);

        void LateUpdate()
        {
            if (director == null || !MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var alive = director.Alive; var fallen = director.Fallen;
            if (alive.Count == 0 || alive[0].Map == null) return;

            // Cut-away: the newest fallen fighter to reach the water, shown once.
            if (fallen.Count == 0) { _shown.Clear(); if (_cut != null) { _cut = null; _snap = true; } }
            if (director.sea != null)
                foreach (var r in fallen)
                    if (!_shown.Contains(r) && Pelvis(r).y < director.sea.Level + cutAbove)
                    {
                        _shown.Add(r); _cut = r; _cutUntil = Time.unscaledTime + cutSeconds; _snap = true;
                        Debug.Log($"[MatchCamera] cut to {r.robotPrefix} at the water (sea {director.sea.Level:0.0} m)");
                        director.Shot("splash");
                    }
            if (_cut != null && (Time.unscaledTime > _cutUntil || Pelvis(_cut).y < director.sea.Level - cutUnder))
            {
                Debug.Log($"[MatchCamera] back to the summit ({(Time.unscaledTime > _cutUntil ? "time limit" : "gone under")})");
                _cut = null; _snap = true;
            }

            Vector3 look, pos;
            if (_cut != null)
            {
                Vector3 p = Pelvis(_cut); Vector3 outward = new Vector3(p.x, 0f, p.z).normalized;
                look = p; pos = p + outward * 3.2f + Vector3.up * 1.6f;
            }
            else
            {
                // The fighter nearest the centre of the summit; the current one keeps the job unless clearly beaten.
                if (_centre != null && !Contains(alive, _centre)) _centre = null;
                float best = _centre != null ? FromCentre(Pelvis(_centre)) - centreMargin : float.MaxValue;
                foreach (var r in alive) { float d = FromCentre(Pelvis(r)); if (d < best) { best = d; _centre = r; } }
                look = Pelvis(_centre); float reach = 0f;
                foreach (var r in alive) reach = Mathf.Max(reach, Vector3.Distance(Pelvis(r), look));
                look.y = Mathf.Max(0.5f, look.y * 0.8f);
                pos = new Vector3(look.x * 0.5f, height, look.z * 0.5f - (baseDistance + distancePerMetre * 2f * reach));
            }
            float t = _snap ? 1f : 1f - Mathf.Exp(-smooth * Time.unscaledDeltaTime); _snap = false;      // cuts are instant, the rest glides
            _look = Vector3.Lerp(_look, look, t);
            transform.position = Vector3.Lerp(transform.position, pos, t);
            transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.LookRotation(_look - transform.position, Vector3.up), t);
        }

        static bool Contains(IReadOnlyList<PolicyRunner> list, PolicyRunner r)
        {
            foreach (var x in list) if (x == r) return true;
            return false;
        }
    }
}
