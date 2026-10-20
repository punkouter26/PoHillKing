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
    ///   shockwave  a hard hit (strength 0.7+) sends a screen-space ring out from it (ShockwaveFeature) and a dust ring
    ///              along the ground
    ///   lightning  once the storm is past half way (or the live weather has a thunderstorm), a strike every 5-12 s
    ///              over the sea: a jagged bolt, a double flash of the lightning light, thunder from ImpactSynth
    ///              delayed by the distance
    ///   danger     a pulsing red light on the fighter whose capture point is closest past the rim (Biometrics)
    ///   drips      a fighter that has been in the sea drips for a few seconds once out of it
    /// Budgets (particles, decals, shockwaves) follow QualityTier.
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
        public ParticleSystem shockRing, drips;
        public Light lightning, danger;
        public LineRenderer bolt;
        public ImpactSynth synth;                 // optional: thunder
        public Biometrics bio;                    // optional: the danger light
        public QualityTier quality;               // optional: budgets
        float _nextBolt = -1f, _boltAt = -10f; float[] _wetUntil = System.Array.Empty<float>(); float _nextDrip;
        readonly Vector3[] _boltPts = new Vector3[12];

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
            if (quality != null) count = Mathf.RoundToInt(count * quality.ParticleScale);
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
            if (s > 0.6f && r != null && r.body == "g1") Burst(sparks, p, (int)(18 * s), 3f);       // metal sparks; humans only raise dust
            _punch = Mathf.Max(_punch, Mathf.Clamp01(s * 1.2f));
            if (s >= 0.7f)
            {
                if (quality == null || quality.Shockwaves) Shockwave.Trigger(p, s);
                if (shockRing != null) shockRing.Emit(new ParticleSystem.EmitParams { position = p + Vector3.down * 0.62f, startSize = 0.3f, applyShapeToPosition = false }, 1);
            }
        }

        void OnFootfall(Vector3 p, float s, float slip)
        {
            if (!On) return;
            if (s > 0.6f || slip > 0.5f) Burst(dust, p, 1 + (int)(2 * s), 0.35f);
            if (slip > 0.35f && scuffs.Length > 0) PlaceScuff(p, slip);
        }

        void PlaceScuff(Vector3 p, float slip)
        {
            int pool = quality != null ? Mathf.Min(scuffs.Length, quality.Decals) : scuffs.Length;
            if (_nextScuff >= pool) _nextScuff = 0;
            var d = scuffs[_nextScuff]; if (d == null) return;
            // project straight down onto the hill; size grows with the slide
            d.transform.SetPositionAndRotation(p + Vector3.up * 0.25f, Quaternion.Euler(90f, (float)_rng.NextDouble() * 360f, 0f));
            float len = Mathf.Lerp(0.22f, 0.55f, Mathf.Clamp01(slip / 1.5f));
            d.size = new Vector3(len, 0.18f, 0.6f); d.fadeFactor = 0.85f;
            _scuffBorn[_nextScuff] = Time.time; _nextScuff = (_nextScuff + 1) % pool;
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

            bool fight = director.State == MatchState.Fight && On;
            Lightning(fight, storm);
            Danger(fight);
            Drips(fight);
        }

        // A strike over the sea every 5-12 s once the storm is past half way (or with a live thunderstorm).
        void Lightning(bool fight, float storm)
        {
            bool thunderstorm = weather != null && weather.Ready && weather.Code >= 95;
            float since = Time.unscaledTime - _boltAt;
            if (lightning != null) lightning.intensity = since < 0.4f ? 4f * (Flick(since, 0f) + 0.7f * Flick(since, 0.16f)) : 0f;
            if (lightning != null) lightning.enabled = since < 0.4f;
            if (bolt != null) bolt.enabled = since < 0.22f;
            if (!fight || (storm < 0.5f && !thunderstorm)) { _nextBolt = -1f; return; }
            if (_nextBolt < 0f) { _nextBolt = Time.unscaledTime + 2f + (float)_rng.NextDouble() * 4f; return; }
            if (Time.unscaledTime < _nextBolt) return;
            _nextBolt = Time.unscaledTime + 5f + (float)_rng.NextDouble() * 7f; _boltAt = Time.unscaledTime;
            // where it strikes the sea: most of the time a point the camera sees (a ray through the upper part of the
            // frame, 40-260 m out), otherwise anywhere around, heard more than seen
            var cam = Camera.main; float seaY = director.sea != null ? director.sea.Level : 0f;
            Vector3 ground; float dist;
            var ray = cam != null ? cam.ViewportPointToRay(new Vector3(0.15f + 0.7f * (float)_rng.NextDouble(), 0.55f + 0.4f * (float)_rng.NextDouble(), 0f)) : default;
            float hit = cam != null && ray.direction.y < -1e-3f ? (seaY - ray.origin.y) / ray.direction.y : -1f;
            if (hit > 0f && _rng.NextDouble() < 0.75)
            {
                ground = ray.origin + ray.direction * hit; Vector3 flat = new(ground.x, 0f, ground.z);
                dist = Mathf.Clamp(flat.magnitude, 40f, 260f); ground = flat.normalized * dist + Vector3.up * seaY;
            }
            else
            {
                float ang = (float)_rng.NextDouble() * 360f; dist = 60f + 200f * (float)_rng.NextDouble();
                ground = Quaternion.Euler(0f, ang, 0f) * Vector3.forward * dist + Vector3.up * seaY;
            }
            Vector3 dir = new Vector3(ground.x, 0f, ground.z).normalized;
            if (bolt != null)
            {
                Vector3 top = ground + Vector3.up * (0.35f * dist + 20f * (float)_rng.NextDouble()) + dir * 8f;     // tall enough to leave the frame
                for (int i = 0; i < _boltPts.Length; i++)
                {
                    float t = i / (float)(_boltPts.Length - 1);
                    Vector3 jitter = i == 0 || i == _boltPts.Length - 1 ? Vector3.zero : new Vector3((float)_rng.NextDouble() - 0.5f, 0f, (float)_rng.NextDouble() - 0.5f) * (0.15f * (top.y - ground.y));
                    _boltPts[i] = Vector3.Lerp(top, ground, t) + jitter;
                }
                bolt.positionCount = _boltPts.Length; bolt.SetPositions(_boltPts); bolt.widthMultiplier = 0.004f * dist;     // ~2 px wide whatever the distance
            }
            if (lightning != null) lightning.transform.rotation = Quaternion.LookRotation((Vector3.zero - (ground + Vector3.up * 60f)).normalized);
            if (synth != null) synth.Thunder(ground, Mathf.Lerp(0.9f, 0.5f, dist / 260f));
        }

        static float Flick(float t, float at) { float x = t - at; return x < 0f ? 0f : Mathf.Exp(-x / 0.05f); }

        // A red light pulsing over whoever is furthest past the rim (and still in).
        void Danger(bool fight)
        {
            if (danger == null) return;
            Transform worst = null; float best = 0.12f;
            if (fight && bio != null)
                for (int i = 0; i < director.Playing; i++)
                {
                    if (director.IsOut(i)) continue; var v = bio.Of(director.fighters[i]); if (v == null) continue;
                    float d = 0.12f - v.rim; if (d > best) { best = d; worst = director.fighters[i].PelvisTransform; }
                }
            danger.enabled = worst != null;
            if (worst == null) return;
            danger.transform.position = worst.position + Vector3.up * 1.4f;
            danger.intensity = (2f + 3f * Mathf.Clamp01(best / 0.4f)) * (0.6f + 0.4f * Mathf.Sin(Time.unscaledTime * 14f));
        }

        // Fighters that were in the sea drip for a while once out of it.
        void Drips(bool fight)
        {
            if (drips == null || director.sea == null) return;
            int n = director.fighters.Length; if (_wetUntil.Length != n) _wetUntil = new float[n];
            bool emit = Time.unscaledTime >= _nextDrip; if (emit) _nextDrip = Time.unscaledTime + 0.08f;
            for (int i = 0; i < n && i < director.Playing; i++)
            {
                var t = director.fighters[i].PelvisTransform; if (t == null) continue;
                if (t.position.y < director.sea.Level) { _wetUntil[i] = Time.unscaledTime + 6f; continue; }
                if (emit && On && Time.unscaledTime < _wetUntil[i]) Burst(drips, t.position + new Vector3(0f, 0.2f, 0f), 2, 0.05f);
            }
        }
    }
}
