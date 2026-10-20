using System.Collections.Generic;
using Mujoco;
using Unity.Cinemachine;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Broadcast direction with Cinemachine (GameSettings.CameraStyle = Broadcast; Classic hands the camera back to
    /// MatchCamera). Shots, highest priority wins:
    ///   replay   slow orbit close to the fighter whose knock-out is being replayed
    ///   faller   hard cut to a knocked-out fighter as it reaches the sea (once per faller, at most 3 s; cut-aways setting)
    ///   winner   orbit around the winner while the result is up
    ///   crane    intro: a dolly along a spline from low over the sea up to the summit (Splines + CinemachineSplineDolly)
    ///   intro    intro: a close, slow orbit of the fighter whose tale-of-the-tape card is up
    ///   summit   group framing of everyone still in, the fighter nearest the centre weighted double
    ///   menu     slow orbit of the empty summit
    /// Big impacts shake the active camera through a Cinemachine impulse (Effects setting).
    /// Targets are the pelvis transforms the MuJoCo plugin moves (render side only, no physics).
    /// </summary>
    public unsafe class BroadcastCamera : MonoBehaviour
    {
        public DemoDirector director;
        public CinemachineBrain brain;
        public MatchCamera classic;
        public CinemachineCamera summit, faller, winner, replay, menu;
        [Tooltip("Intro shots (optional): the spline crane and the close orbit of the featured fighter")] public CinemachineCamera crane, intro;
        public CinemachineTargetGroup group;
        public CinemachineImpulseSource impulse;
        public Transform summitAnchor;
        public float orbitDegPerSecond = 18f, cutAbove = 0.5f, cutUnder = 0.6f, cutSeconds = 3f;

        readonly HashSet<PolicyRunner> _shown = new();
        PolicyRunner _cut; float _cutUntil; CinemachineCamera _live; int _introFor = -1;

        void OnEnable() { MatchEvents.Impact += OnImpact; MatchEvents.RoundStarted += OnRound; }
        void OnDisable() { MatchEvents.Impact -= OnImpact; MatchEvents.RoundStarted -= OnRound; }
        void OnRound() { _shown.Clear(); _cut = null; }

        static Transform Pelvis(PolicyRunner r) => r != null ? r.PelvisTransform : null;

        void OnImpact(Vector3 p, float strength, PolicyRunner r)
        {
            if (impulse == null || !GameSettings.Effects || strength < 0.45f || !brain.enabled) return;
            impulse.GenerateImpulseAt(p, new Vector3(0.2f, -1f, 0.1f) * (0.12f * strength));
        }

        void LateUpdate()
        {
            if (director == null) return;
            bool broadcast = GameSettings.CameraStyle == CameraStyle.Broadcast;
            if (brain.enabled != broadcast) brain.enabled = broadcast;
            if (classic != null && classic.enabled == broadcast) classic.enabled = !broadcast;
            if (!broadcast || !MjScene.InstanceExists || MjScene.Instance.Data == null) return;

            var alive = director.Alive; var fallen = director.Fallen;
            // the summit group: everyone still in, the one nearest the centre counts double
            PolicyRunner centre = null; float best = float.MaxValue;
            for (int i = 0; i < alive.Count; i++) { var t = Pelvis(alive[i]); if (t == null) continue; float d = new Vector2(t.position.x, t.position.z).magnitude; if (d < best) { best = d; centre = alive[i]; } }
            SyncGroup(alive, centre);

            CinemachineCamera next;
            if (ReplaySystem.Active && Pelvis(ReplaySystem.Focus) != null) { Aim(replay, Pelvis(ReplaySystem.Focus)); Spin(replay, 0.5f); next = replay; }
            else if (director.State == MatchState.Menu || director.State == MatchState.Booting) { Spin(menu, 0.6f); next = menu; }
            else if (director.State == MatchState.Intro && crane != null && intro != null) next = IntroShot();
            else if (director.State == MatchState.Result && director.LastResult.Winner >= 0 && director.LastResult.Winner < director.fighters.Length)
            { Aim(winner, Pelvis(director.fighters[director.LastResult.Winner])); Spin(winner, 1f); next = winner; }
            else if (FallerShot(fallen)) next = faller;
            else next = summit;
            Activate(next);
        }

        CinemachineCamera IntroShot()
        {
            int f = director.IntroFeatured;
            if (f < 0)
            {
                var dolly = crane.GetComponent<CinemachineSplineDolly>();
                if (dolly != null) dolly.CameraPosition = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(director.IntroElapsed / Mathf.Max(0.1f, director.introCrane)));
                return crane;
            }
            var p = f < director.Playing ? Pelvis(director.fighters[f]) : null;
            if (p != null)
            {
                Aim(intro, p);
                var orbit = intro.GetComponent<CinemachineOrbitalFollow>();
                // each card starts from outside the ring looking in, so no other fighter stands between camera and subject
                if (f != _introFor && orbit != null) { _introFor = f; orbit.HorizontalAxis.Value = Mathf.Atan2(-p.position.x, -p.position.z) * Mathf.Rad2Deg - 20f; }
                Spin(intro, 0.6f);
            }
            return intro;
        }

        bool FallerShot(IReadOnlyList<PolicyRunner> fallen)
        {
            var sea = director.sea;
            if (fallen.Count == 0 || sea == null || !GameSettings.CutAways) { _cut = null; return false; }
            for (int i = 0; i < fallen.Count; i++)
            {
                var t = Pelvis(fallen[i]);
                if (t != null && !_shown.Contains(fallen[i]) && t.position.y < sea.Level + cutAbove) { _shown.Add(fallen[i]); _cut = fallen[i]; _cutUntil = Time.unscaledTime + cutSeconds; director.Shot("splash"); }
            }
            if (_cut == null) return false;
            var p = Pelvis(_cut);
            if (p == null || Time.unscaledTime > _cutUntil || p.position.y < sea.Level - cutUnder) { _cut = null; return false; }
            Vector3 outward = new Vector3(p.position.x, 0, p.position.z).normalized;
            var follow = faller.GetComponent<CinemachineFollow>(); if (follow != null) follow.FollowOffset = outward * 3.2f + Vector3.up * 1.6f;
            Aim(faller, p);
            return true;
        }

        void SyncGroup(IReadOnlyList<PolicyRunner> alive, PolicyRunner centre)
        {
            if (group == null) return;
            var targets = group.Targets;
            // rebuild only when membership changes; weights each frame
            bool same = targets.Count == alive.Count;
            for (int i = 0; same && i < alive.Count; i++) same = targets[i].Object == Pelvis(alive[i]);
            if (!same)
            {
                targets.Clear();
                for (int i = 0; i < alive.Count; i++) { var t = Pelvis(alive[i]); if (t != null) targets.Add(new CinemachineTargetGroup.Target { Object = t, Weight = 1, Radius = 0.6f }); }
                if (targets.Count == 0 && summitAnchor != null) targets.Add(new CinemachineTargetGroup.Target { Object = summitAnchor, Weight = 1, Radius = 1.5f });
            }
            for (int i = 0; i < targets.Count && i < alive.Count; i++) { var t = targets[i]; t.Weight = alive[i] == centre ? 2f : 1f; targets[i] = t; }
        }

        static void Aim(CinemachineCamera cam, Transform t) { if (t == null) return; cam.Target.TrackingTarget = t; cam.Target.LookAtTarget = t; }

        void Spin(CinemachineCamera cam, float speed)
        {
            var orbit = cam.GetComponent<CinemachineOrbitalFollow>(); if (orbit == null) return;
            orbit.HorizontalAxis.Value = Mathf.Repeat(orbit.HorizontalAxis.Value + orbitDegPerSecond * speed * Time.unscaledDeltaTime + 180f, 360f) - 180f;
        }

        void Activate(CinemachineCamera cam)
        {
            if (cam == _live) return;
            // cuts to and from the faller are instant, everything else blends
            bool hard = cam == faller || _live == faller || cam == replay || _live == replay || _live == null || cam == crane;
            brain.DefaultBlend = hard ? new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.Cut, 0f)
                                      : new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.EaseInOut, _live == intro && cam == intro ? 0.5f : 0.9f);
            foreach (var c in new[] { summit, faller, winner, replay, menu, crane, intro }) if (c != null) c.Priority = c == cam ? 20 : 10;
            _live = cam;
        }
    }
}
