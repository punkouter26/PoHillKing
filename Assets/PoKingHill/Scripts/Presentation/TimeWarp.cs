using System;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Presentation time (GameSettings.Cinematic):
    ///   slow motion  0.3x for 0.7 s of real time, the first time in a round a fighter's capture point crosses the rim
    ///                heading out (Biometrics.doomed: it is going over unless something stops it)
    ///   hit stop     a 70 ms near-freeze on the hardest hits (MatchEvents.Impact strength 0.85 and up)
    /// Only Time.timeScale changes. Every physics step is still one 2 ms mj_step with the same inputs, so the match
    /// plays out exactly as it would at full speed, only shown slower. Never in the statistics gates (they own
    /// timeScale), during replays (ReplaySystem has its own pace), or while anything else has changed timeScale.
    /// ImpactSynth reads Muffle to low-pass the mix while time is slowed.
    /// </summary>
    public class TimeWarp : MonoBehaviour
    {
        public MatchDirector director;
        public Biometrics bio;
        public float slowScale = 0.3f, slowSeconds = 0.7f, stopScale = 0.05f, stopSeconds = 0.07f, stopStrength = 0.85f, cooldownSeconds = 2.5f;

        /// <summary>0..1: how much the audio should be muffled right now.</summary>
        public static float Muffle { get; private set; }
        public int SlowMos { get; private set; }
        public int HitStops { get; private set; }

        float _from, _until, _target = 1f, _written = 1f, _muffle, _nextAllowed; bool[] _warned = Array.Empty<bool>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => Muffle = 0f;

        void OnEnable() { MatchEvents.Impact += OnImpact; MatchEvents.RoundStarted += OnRound; }
        void OnDisable()
        {
            MatchEvents.Impact -= OnImpact; MatchEvents.RoundStarted -= OnRound;
            Release(); Muffle = 0f;
        }

        void OnRound() { if (_warned.Length > 0) Array.Clear(_warned, 0, _warned.Length); }

        bool Allowed => GameSettings.Cinematic && director != null && director.statsRounds == 0 && director.State == MatchState.Fight && !ReplaySystem.Active;
        // nobody else is using timeScale (gates run at 50x, the get-up probe too)
        bool Free => Time.timeScale == _written && (_written == 1f || _until > 0f);

        void OnImpact(Vector3 p, float strength, PolicyRunner r)
        {
            if (strength < stopStrength || !Allowed || !Free || _until > Time.unscaledTime) return;
            Begin(stopScale, stopSeconds, 0.35f); HitStops++;
        }

        void Begin(float scale, float seconds, float muffle)
        {
            _from = Time.unscaledTime; _until = _from + seconds; _target = scale; _muffle = muffle;
            Write(scale);
        }

        void Write(float s) { Time.timeScale = s; _written = s; }

        void Release()
        {
            if (_until > 0f && Time.timeScale == _written) Write(1f);
            _until = 0f; _target = 1f;
        }

        void Update()
        {
            if (Time.timeScale != _written) { _written = Time.timeScale; _until = 0f; }      // someone else took timeScale: hand it over
            if (_until > 0f && (!Allowed || Time.unscaledTime >= _until)) Release();
            if (_until > 0f)
            {
                // ease back to full speed over the last fifth of the effect
                float t = Mathf.InverseLerp(_until - 0.2f * (_until - _from), _until, Time.unscaledTime);
                Write(Mathf.Lerp(_target, 1f, t * t));
            }
            Muffle = Mathf.MoveTowards(Muffle, _until > 0f ? _muffle : 0f, Time.unscaledDeltaTime * 4f);

            if (bio == null || !Allowed) return;
            int n = director.bodies.Length; if (_warned.Length != n) _warned = new bool[n];
            for (int i = 0; i < director.Playing && i < n; i++)
            {
                var v = bio.Of(director.bodies[i]);
                if (v == null || !v.doomed || _warned[i]) continue;
                _warned[i] = true;
                if (Free && _until <= 0f && Time.unscaledTime >= _nextAllowed) { Begin(slowScale, slowSeconds, 0.6f); _nextAllowed = Time.unscaledTime + cooldownSeconds; SlowMos++; }
            }
        }
    }
}
