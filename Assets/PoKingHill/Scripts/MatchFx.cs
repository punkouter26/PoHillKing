using Mujoco;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PoKingHill
{
    /// <summary>
    /// Render-side reactions to the match (Effects setting; nothing here touches physics):
    ///   impacts    dust burst sized by the hit, sparks on hard hits, a short chromatic/lens punch (post-process volume)
    ///   footfalls  small dust puffs; a scuff decal (URP Decal Projector) where a foot lands while still sliding
    ///   water      splash spray and a ripple ring where a body enters; the water shader's level and shoreline foam
    ///   light      a new time of day each round, the sky darkening toward a storm as the sea rises, and a spotlight
    ///              on the fighter nearest the centre
    ///   winner     confetti over the winner
    ///   stamina    steam off fighters that are worn out, a spark burst on adrenaline (Stamina)
    ///   weather    with a live weather city (WeatherFeed): night, overcast or clear sky, and a stormier sea when it rains
    /// The particle systems, decal pool, volume and lights are scene objects built by the editor builder, so they
    /// can be tuned by hand; this component only drives them.
    /// </summary>
    public unsafe class MatchFx : MonoBehaviour
    {
        public DemoDirector director;
        public ParticleSystem dust, sparks, splash, ripple, confetti, steam;
        public Stamina stamina;
        public WeatherFeed weather;
        public DecalProjector[] scuffs = System.Array.Empty<DecalProjector>();
        public Volume hitVolume;
        public Light sun, kingSpot;
        public Material water;
        public float scuffSeconds = 10f;

        static readonly int SeaLevel = Shader.PropertyToID("_SeaLevel"), Storm = Shader.PropertyToID("_Storm");
        float[] _scuffBorn; int _nextScuff; float _punch; Color _fogCalm, _sunCalm; float _sunCalmIntensity;
        readonly System.Random _rng = new();
        static readonly (Vector3 euler, Color color, float intensity)[] TimesOfDay =
        {
            (new Vector3(50f, -30f, 0f), new Color(1.00f, 0.96f, 0.90f), 1.10f),     // midday
            (new Vector3(22f, 60f, 0f), new Color(1.00f, 0.78f, 0.55f), 1.00f),      // late afternoon
            (new Vector3(12f, -120f, 0f), new Color(1.00f, 0.62f, 0.42f), 0.85f),    // sunset
            (new Vector3(35f, 150f, 0f), new Color(0.85f, 0.90f, 1.00f), 0.95f),     // overcast morning
            (new Vector3(28f, 200f, 0f), new Color(0.55f, 0.65f, 0.95f), 0.40f),     // moonlight (live weather at night)
        };
        float _nextSteam;

        void OnEnable()
        {
            _scuffBorn = new float[scuffs.Length];
            // Drive a runtime copy of the water material: setting the level on the shared asset would rewrite Water.mat on disk.
            if (water != null && director != null && director.sea != null && director.sea.waterVisual != null && !water.name.EndsWith("(runtime)"))
            {
                water = new Material(water) { name = water.name + " (runtime)" };
                director.sea.waterVisual.GetComponent<Renderer>().sharedMaterial = water;
            }
            foreach (var s in scuffs) if (s != null) s.fadeFactor = 0;
            _fogCalm = RenderSettings.fogColor;
            if (sun != null) { _sunCalm = sun.color; _sunCalmIntensity = sun.intensity; }
            MatchEvents.Impact += OnImpact; MatchEvents.Footfall += OnFootfall; MatchEvents.Splash += OnSplash;
            MatchEvents.RoundStarted += OnRound; MatchEvents.RoundEnded += OnEnd; MatchEvents.Hyped += OnHype;
        }
        void OnDisable()
        {
            MatchEvents.Impact -= OnImpact; MatchEvents.Footfall -= OnFootfall; MatchEvents.Splash -= OnSplash;
            MatchEvents.RoundStarted -= OnRound; MatchEvents.RoundEnded -= OnEnd; MatchEvents.Hyped -= OnHype;
        }

        void OnHype(PolicyRunner r)
        {
            if (!On || r.PelvisTransform == null) return;
            Burst(sparks, r.PelvisTransform.position + Vector3.up * 0.3f, 40, 2.5f); _punch = Mathf.Max(_punch, 0.5f);
        }

        bool On => GameSettings.Effects && director != null && director.statsRounds == 0;

        void Burst(ParticleSystem ps, Vector3 p, int count, float speed)
        {
            if (ps == null || count <= 0) return;
            var e = new ParticleSystem.EmitParams { position = p, applyShapeToPosition = true };
            for (int i = 0; i < count; i++)
            {
                var dir = new Vector3((float)_rng.NextDouble() * 2 - 1, (float)_rng.NextDouble() * 0.9f + 0.3f, (float)_rng.NextDouble() * 2 - 1).normalized;
                e.velocity = dir * speed * (0.5f + (float)_rng.NextDouble()); ps.Emit(e, 1);
            }
        }

        void OnImpact(Vector3 p, float s, PolicyRunner r)
        {
            if (!On) return;
            if (s > 0.5f) Burst(dust, p + Vector3.down * 0.6f, 1 + (int)(8 * s), 0.6f + 1.2f * s);
            if (s > 0.6f) Burst(sparks, p, (int)(18 * s), 3f);
            _punch = Mathf.Max(_punch, Mathf.Clamp01(s * 1.2f));
        }

        void OnFootfall(Vector3 p, float s, float slip)
        {
            if (!On) return;
            if (s > 0.6f || slip > 0.5f) Burst(dust, p, 1 + (int)(2 * s), 0.35f);
            if (slip > 0.35f && scuffs.Length > 0) PlaceScuff(p, slip);
        }

        void PlaceScuff(Vector3 p, float slip)
        {
            var d = scuffs[_nextScuff]; if (d == null) return;
            // project straight down onto the hill; size grows with the slide
            d.transform.SetPositionAndRotation(p + Vector3.up * 0.25f, Quaternion.Euler(90f, (float)_rng.NextDouble() * 360f, 0f));
            float len = Mathf.Lerp(0.22f, 0.55f, Mathf.Clamp01(slip / 1.5f));
            d.size = new Vector3(len, 0.18f, 0.6f); d.fadeFactor = 0.85f;
            _scuffBorn[_nextScuff] = Time.time; _nextScuff = (_nextScuff + 1) % scuffs.Length;
        }

        void OnSplash(Vector3 p, float s)
        {
            if (!On) return;
            Burst(splash, p, 10 + (int)(40 * s), 2.5f + 2f * s);
            if (ripple != null) ripple.Emit(new ParticleSystem.EmitParams { position = p + Vector3.up * 0.02f, applyShapeToPosition = false }, 1);
        }

        void OnRound()
        {
            if (director == null || director.statsRounds > 0 || sun == null) return;
            // live weather: night -> moonlight, cloudy or wet -> overcast, else a daylight pick; without it any of the four
            int pick = weather != null && weather.Ready ? (!weather.IsDay ? 4 : weather.Wet || weather.CloudCover > 70 ? 3 : _rng.Next(3)) : _rng.Next(4);
            var t = TimesOfDay[pick];
            sun.transform.rotation = Quaternion.Euler(t.euler); _sunCalm = t.color; _sunCalmIntensity = t.intensity;
        }

        void OnEnd(RoundResult r)
        {
            if (!On || confetti == null || r.Winner < 0 || r.Winner >= director.fighters.Length) return;
            var t = director.fighters[r.Winner].PelvisTransform; if (t == null) return;
            confetti.transform.position = t.position + Vector3.up * 1.6f; confetti.Play();
        }

        void Update()
        {
            if (director == null) return;
            // post-process punch decays over ~0.25 s
            _punch = Mathf.MoveTowards(_punch, 0f, Time.unscaledDeltaTime * 4f);
            if (hitVolume != null) hitVolume.weight = GameSettings.Effects ? _punch : 0f;
            for (int i = 0; i < scuffs.Length; i++)
                if (scuffs[i] != null && scuffs[i].fadeFactor > 0) scuffs[i].fadeFactor = Mathf.Lerp(0.85f, 0f, (Time.time - _scuffBorn[i]) / scuffSeconds);

            var sea = director.sea;
            float storm = sea != null && director.State == MatchState.Fight ? Mathf.Clamp01((sea.Level + sea.startDepth) / sea.startDepth) : 0f;
            storm = storm * storm;
            if (weather != null && weather.Ready && weather.Wet && director.State == MatchState.Fight) storm = Mathf.Max(storm, 0.35f);
            if (steam != null && stamina != null && GameSettings.Stamina && On && director.State == MatchState.Fight && Time.time >= _nextSteam)
            {
                _nextSteam = Time.time + 0.2f;
                for (int i = 0; i < director.Playing; i++)
                {
                    var r = director.fighters[i]; float s = stamina.Level(r);
                    if (s < 0.4f && !director.IsOut(i) && r.PelvisTransform != null) Burst(steam, r.PelvisTransform.position + Vector3.up * 0.55f, s < 0.2f ? 2 : 1, 0.25f);
                }
            }
            if (water != null && sea != null) { water.SetFloat(SeaLevel, sea.Level); water.SetFloat(Storm, storm); }
            if (sun != null)
            {
                sun.color = Color.Lerp(_sunCalm, new Color(0.62f, 0.68f, 0.80f), storm * 0.8f);
                sun.intensity = Mathf.Lerp(_sunCalmIntensity, _sunCalmIntensity * 0.55f, storm);
            }
            RenderSettings.fogColor = Color.Lerp(_fogCalm, new Color(0.42f, 0.47f, 0.56f), storm * 0.85f);

            if (kingSpot != null)
            {
                Transform king = null; float best = float.MaxValue;
                var alive = director.Alive;
                if (director.State == MatchState.Fight || director.State == MatchState.Result)
                    for (int i = 0; i < alive.Count; i++)
                    {
                        var t = alive[i].PelvisTransform; if (t == null) continue;
                        float d = new Vector2(t.position.x, t.position.z).magnitude; if (d < best) { best = d; king = t; }
                    }
                kingSpot.enabled = king != null && GameSettings.Effects;
                if (king != null) { kingSpot.transform.position = king.position + new Vector3(0f, 4.5f, -0.8f); kingSpot.transform.LookAt(king.position); }
            }
        }
    }
}
