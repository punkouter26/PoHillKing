using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Runtime-synthesized sound, no audio files. Driven only by the MuJoCo state and the match events:
    ///   impact        a fighter's pelvis changes velocity sharply within 20 ms (hit, shove, landing). What it hits picks
    ///                 the sound: G1 against G1 rings a metal clank over the thud, a human taking a hit gives a flesh
    ///                 smack and now and then a grunt in that body's voice (Trump low, Kim high), a fall is a thud. These
    ///                 one-shots play in 3D at the hit through SoundPool (distance, direction, doppler); without a pool
    ///                 they fall back to the stereo voices below.
    ///   footfall      a foot's downward speed stops near the ground: a short click scaled by that speed
    ///   friction      every 20 ms the contacts are scanned: a planted foot that slides scrapes the rock (band-passed
    ///                 noise, louder with the slip speed), two fighters rubbing grind (metal for G1 pairs, a duller rub
    ///                 when a human is in it), sustained for as long as the sliding lasts
    ///   splash        a pelvis crosses the water surface going down; then a quiet bubbling hum while submerged
    ///   ambience      wind over the summit (gusts from Hazards), surf that grows as the
    ///                 sea rises and the storm builds, a crowd that murmurs and swells on hype, knock-outs, the end of a
    ///                 round and a fighter teetering on the rim (Biometrics), a low rumble during a quake
    /// Mix: MasterVolume x SfxVolume for hits and footsteps, MasterVolume x AmbienceVolume for the beds;
    /// everything is low-passed while the camera is under water or time is slowed.
    /// The stereo voices are panned and attenuated from the camera's point of view. The same triggers are announced
    /// through MatchEvents for particles, decals, camera shake and commentary. The physics callback writes trigger
    /// values; the audio thread turns them into decaying noise and sine bursts. Silent beds in the statistics gates.
    /// </summary>
    [RequireComponent(typeof(AudioSource))]
    public unsafe class ImpactSynth : MonoBehaviour
    {
        public PolicyRunner[] robots;
        public Sea sea;
        public SoundPool pool;                    // optional: 3D one-shots
        public DemoDirector director;             // optional: ambience follows the match state
        public Hazards hazards;                   // optional: gusts and quakes in the ambience
        public Biometrics bio;                    // optional: the crowd gasps at a fighter on the rim
        [Range(0f, 1f)] public float volume = 0.6f;
        public float impactThreshold = 0.45f;     // m/s of pelvis velocity change within 20 ms
        public float reach = 0.8f;                // another pelvis this close makes a hit a fighter-on-fighter hit
        long _tick; double _quietUntil;

        // Verification counters. Triggers are counted where the physics raises them; Blocks and Peak come from the
        // audio thread, which the editor only runs while it has focus.
        public int Thuds, Clicks, Splashes, Clanks, Smacks, Grunts, Blocks; public float Peak, Slide, Grind;
        /// <summary>RMS of each ambience bed over the last audio block, after the mix (verification and tuning).</summary>
        public float WindRms, SurfRms, CrowdRms, RumbleRms;
        double[][] _prevVel; double[] _prevFootVz; bool[] _wasUnder; int[] _footBody; int[] _robotOfBody; bool[] _isFoot; double[] _gruntAt;
        // triggers written by the physics callback, consumed by the audio thread (strength, pan -1..1)
        volatile float _thud, _thudPitch = 70f, _thudPan, _click, _clickPan, _clank, _clankPan, _splash, _splashPan, _under;
        // sustained levels (targets) and ambience, written by the main thread
        volatile float _slide, _slidePan, _grind, _grindMetal, _wind, _surf, _crowd, _rumble, _muffle, _sfxGain = 1f, _ambGain = 1f;
        float _eThud, _eClick, _eClank, _eSplash, _pThud, _pClick, _pClank, _pSplash, _phase, _lp, _bp1, _bp2, _c1, _c2, _c3;
        float _sSlide, _sGrind, _sWind, _sSurf, _sCrowd, _sRumble, _sMuffle, _wl, _wr, _brown, _rl, _mL, _mR, _gb1, _gb2, _surfPh, _lfo1, _lfo2, _lfo3;
        float _swell, _gasp; bool[] _gasped = System.Array.Empty<bool>();
        Reson _crowdA, _crowdB, _crowdC;
        int _rate = 48000; uint _rng = 12345u; Transform _listener;

        void OnEnable()
        {
            _rate = AudioSettings.outputSampleRate;
            _crowdA = new Reson(480f, 260f, _rate); _crowdB = new Reson(1150f, 420f, _rate); _crowdC = new Reson(2600f, 900f, _rate);
            var src = GetComponent<AudioSource>(); src.playOnAwake = true; src.loop = true; src.spatialBlend = 0f;
            if (src.clip == null) src.clip = AudioClip.Create("silence", _rate, 1, _rate, false);   // carrier so OnAudioFilterRead runs
            src.Play();
            MjScene.Instance.postInitEvent += OnInit; MjScene.Instance.postUpdateEvent += Post;
            MatchEvents.Knockout += OnKo; MatchEvents.Hyped += OnHype; MatchEvents.RoundEnded += OnEnd; MatchEvents.RoundStarted += OnRound;
        }
        void OnDisable()
        {
            MatchEvents.Knockout -= OnKo; MatchEvents.Hyped -= OnHype; MatchEvents.RoundEnded -= OnEnd; MatchEvents.RoundStarted -= OnRound;
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.postInitEvent -= OnInit; MjScene.Instance.postUpdateEvent -= Post;
        }

        void OnInit(object s, MjStepArgs a) { _prevVel = null; }
        bool Gate => director != null && director.statsRounds > 0;
        void OnKo(PolicyRunner r) { if (!Gate) _swell = Mathf.Max(_swell, 1f); }
        void OnHype(PolicyRunner r) { if (!Gate) _swell = Mathf.Max(_swell, 0.7f); }
        void OnEnd(RoundResult r) { if (!Gate && r.Winner >= 0) _swell = Mathf.Max(_swell, 0.9f); }
        void OnRound() { if (_gasped.Length > 0) System.Array.Clear(_gasped, 0, _gasped.Length); }

        /// <summary>A lightning strike at a world position: thunder from there, delayed by the distance (343 m/s).</summary>
        public void Thunder(Vector3 strike, float strength)
        {
            if (pool == null || _listener == null || Gate) return;
            float dist = Vector3.Distance(strike, _listener.position);
            // the strike is far beyond the pool's roll-off: play it from the strike's direction at a hearable distance
            Vector3 at = _listener.position + (strike - _listener.position).normalized * 20f;
            pool.Play(SoundPool.Kind.Thunder, at, strength * GameSettings.MasterVolume * GameSettings.AmbienceVolume, Mathf.Lerp(1.05f, 0.85f, dist / 400f), Mathf.Min(dist / 343f, 2.5f));
        }

        void Update()
        {
            volume = GameSettings.MasterVolume; _sfxGain = GameSettings.SfxVolume; _ambGain = GameSettings.AmbienceVolume;
            if (_listener == null && Camera.main != null) _listener = Camera.main.transform;
            float dt = Time.unscaledDeltaTime;
            _swell = Mathf.MoveTowards(_swell, 0f, dt * 0.35f); _gasp = Mathf.MoveTowards(_gasp, 0f, dt * 0.8f);

            // ambience targets
            bool gate = Gate, live = director != null && (director.State == MatchState.Fight || director.State == MatchState.Intro || director.RoundOver);
            float level = sea != null ? Mathf.Clamp01((sea.Level + sea.startDepth) / Mathf.Max(0.1f, sea.startDepth)) : 0f;
            float wind = 0.1375f + (hazards != null && hazards.Gusting ? 0.55f : 0f);
            float surf = 0.08f + 0.45f * level * level;
            if (bio != null && director != null && director.State == MatchState.Fight)
            {
                if (_gasped.Length != director.fighters.Length) _gasped = new bool[director.fighters.Length];
                for (int i = 0; i < director.Playing; i++)
                {
                    var v = bio.Of(director.fighters[i]);
                    if (v != null && v.rim < 0.25f && !director.IsOut(i) && !_gasped[i]) { _gasped[i] = true; _gasp = 1f; }
                }
            }
            float crowd = live ? 0.10f + 0.55f * _swell + 0.3f * _gasp : director != null && director.State == MatchState.Menu ? 0.04f : 0f;
            _wind = gate ? 0f : wind; _surf = gate ? 0f : surf; _crowd = gate ? 0f : crowd;
            _rumble = gate || hazards == null || !hazards.Quaking ? 0f : 0.9f;

            // under water (the camera below the surface) or slowed time: close the low-pass
            float under = sea != null && _listener != null && _listener.position.y < sea.Level ? 1f : 0f;
            _muffle = Mathf.Max(under, TimeWarp.Muffle);
            if (pool != null) pool.cutoff = Mathf.Lerp(22000f, 650f, Mathf.Sqrt(_muffle));
        }

        // MuJoCo (x, y, z) -> Unity (x, z, y); pan and distance gain from the camera's point of view.
        static Vector3 U(double x, double y, double z) => new((float)x, (float)z, (float)y);
        void Place(Vector3 p, out float pan, out float gain)
        {
            if (_listener == null) { pan = 0; gain = 1; return; }
            Vector3 rel = _listener.InverseTransformPoint(p); float dist = rel.magnitude;
            pan = Mathf.Clamp(rel.x / Mathf.Max(1f, dist), -1f, 1f); gain = Mathf.Clamp01(7f / Mathf.Max(7f, dist));
        }

        void Setup(MujocoLib.mjModel_* m, int n)
        {
            _prevVel = new double[n][]; _prevFootVz = new double[2 * n]; _wasUnder = new bool[n]; _footBody = new int[2 * n]; _gruntAt = new double[n];
            for (int i = 0; i < n; i++)
            {
                _prevVel[i] = new double[3];
                _footBody[2 * i] = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, robots[i].robotPrefix + "left_ankle_roll_link");
                _footBody[2 * i + 1] = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, robots[i].robotPrefix + "right_ankle_roll_link");
            }
            int nb = (int)m->nbody; _robotOfBody = new int[nb]; _isFoot = new bool[nb];
            for (int b = 0; b < nb; b++)
            {
                _robotOfBody[b] = -1;
                string name = System.Runtime.InteropServices.Marshal.PtrToStringAnsi((System.IntPtr)MujocoLib.mj_id2name(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, b)) ?? "";
                for (int k = 0; k < n; k++) if (name.StartsWith(robots[k].robotPrefix)) { _robotOfBody[b] = k; break; }
            }
            foreach (int f in _footBody) if (f >= 0) _isFoot[f] = true;
        }

        bool Parked(MujocoLib.mjData_* d, int i) => robots[i].Map == null || robots[i].paused && robots[i].policy == null && d->qpos[robots[i].Map.RootQposAdr + 2] > 20;

        void Post(object s, MjStepArgs a)
        {
            if (robots == null || robots.Length == 0 || robots[0].Map == null) return;
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data; int n = robots.Length;
            if (_prevVel == null) Setup(m, n);
            float under = 0f; _tick++;
            for (int i = 0; i < n; i++)
            {
                if (Parked(d, i)) continue;   // parked in the sky
                double* v = d->qvel + robots[i].Map.RootDofAdr; double* q = d->qpos + robots[i].Map.RootQposAdr;
                // impact: pelvis velocity change over a 20 ms window (a single 2 ms step never shows a big enough jump)
                if (_tick % 10 == 0)
                {
                    double dv = System.Math.Sqrt((v[0] - _prevVel[i][0]) * (v[0] - _prevVel[i][0]) + (v[1] - _prevVel[i][1]) * (v[1] - _prevVel[i][1]) + (v[2] - _prevVel[i][2]) * (v[2] - _prevVel[i][2]));
                    if (dv > impactThreshold && dv < 20 && d->time > _quietUntil)
                    {
                        float strength = Mathf.Clamp01((float)dv / 2.5f); Vector3 p = U(q[0], q[1], q[2]);
                        _quietUntil = d->time + 0.12; Thuds++;
                        Hit(d, i, p, strength, (float)dv);
                        MatchEvents.RaiseImpact(p, strength, robots[i]);
                    }
                    for (int k = 0; k < 3; k++) _prevVel[i][k] = v[k];
                }
                // footfall: a foot that was moving down faster than 0.25 m/s comes to rest
                for (int f = 0; f < 2; f++)
                {
                    int b = _footBody[2 * i + f]; if (b < 0) continue; int idx = 2 * i + f;
                    double vz = d->cvel[6 * b + 5];
                    if (vz < -0.25) _prevFootVz[idx] = System.Math.Min(_prevFootVz[idx], vz);                 // remember the fastest descent
                    else if (vz > -0.05 && _prevFootVz[idx] < -0.25)
                    {
                        float strength = Mathf.Clamp01((float)(-_prevFootVz[idx]) / 1.5f);
                        double sx = d->cvel[6 * b + 3], sy = d->cvel[6 * b + 4]; float slip = (float)System.Math.Sqrt(sx * sx + sy * sy);
                        Vector3 p = U(d->xpos[3 * b], d->xpos[3 * b + 1], d->xpos[3 * b + 2]); Place(p, out float pan, out float gain);
                        _click = Mathf.Max(_click, strength * gain); _clickPan = pan;
                        _prevFootVz[idx] = 0; Clicks++;
                        MatchEvents.RaiseFootfall(p, strength, slip);
                    }
                }
                if (sea != null)
                {
                    bool isUnder = q[2] < sea.Level;
                    if (isUnder && !_wasUnder[i])
                    {
                        Vector3 p = U(q[0], q[1], sea.Level); float strength = Mathf.Clamp01((float)System.Math.Abs(v[2]) / 4f + 0.4f);
                        if (pool != null) pool.Play(SoundPool.Kind.Splash, p, strength * volume * _sfxGain);
                        else { Place(p, out float pan, out float gain); _splash = Mathf.Max(_splash, strength * gain); _splashPan = pan; }
                        Splashes++;
                        MatchEvents.RaiseSplash(p, strength);
                    }
                    _wasUnder[i] = isUnder; if (isUnder) under = 1f;
                }
            }
            _under = under;
            if (_tick % 10 == 5) Friction(m, d);
        }

        // What a hit sounds like depends on who is in it.
        void Hit(MujocoLib.mjData_* d, int i, Vector3 p, float strength, float dv)
        {
            int other = NearestOther(d, i);
            bool human = robots[i].body != "g1", otherHuman = other >= 0 && robots[other].body != "g1";
            float g = volume * _sfxGain, pitch = Mathf.Lerp(1.15f, 0.85f, Mathf.Clamp01(dv / 3f));
            if (pool == null)
            {
                Place(p, out float pan, out float gain);
                _thud = Mathf.Max(_thud, strength * gain); _thudPitch = Mathf.Lerp(55f, 110f, Mathf.Clamp01(dv / 3f)); _thudPan = pan;
                if (other >= 0) { _clank = Mathf.Max(_clank, strength * gain); _clankPan = pan; Clanks++; }
                return;
            }
            pool.Play(SoundPool.Kind.Thud, p, strength * g, pitch * (robots[i].body == "trump" ? 0.85f : robots[i].body == "grandpa" ? 0.9f : 1f));
            if (other < 0) return;
            if (!human && !otherHuman) { pool.Play(SoundPool.Kind.Clank, p, strength * g * 0.8f, pitch); Clanks++; }
            else if (human)
            {
                pool.Play(SoundPool.Kind.Smack, p, strength * g * 0.9f, pitch); Smacks++;
                if (strength > 0.25f && d->time > _gruntAt[i])
                {
                    _gruntAt[i] = d->time + 1.2;
                    pool.Play(robots[i].body == "kim" ? SoundPool.Kind.GruntHigh : SoundPool.Kind.GruntLow, p + Vector3.up * 0.6f, Mathf.Lerp(0.5f, 0.9f, strength) * g); Grunts++;
                }
            }
            else { pool.Play(SoundPool.Kind.Clank, p, strength * g * 0.4f, pitch * 0.9f); Clanks++; }    // a human shoving a G1: duller metal
        }

        // Sliding feet and fighters rubbing, from the contact list every 20 ms.
        void Friction(MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
        {
            double* va = stackalloc double[6]; double* vb = stackalloc double[6];
            float slide = 0f, grind = 0f, metal = 0f; Vector3 slideAt = default;
            for (int c = 0; c < d->ncon; c++)
            {
                var con = d->contact + c; int b1 = m->geom_bodyid[con->geom1], b2 = m->geom_bodyid[con->geom2];
                int r1 = _robotOfBody[b1], r2 = _robotOfBody[b2];
                if (r1 < 0 && r2 < 0) continue;
                if ((r1 < 0 || r2 < 0) && (b1 == 0 || b2 == 0))
                {
                    int foot = b1 == 0 ? b2 : b1; if (!_isFoot[foot]) continue;
                    MujocoLib.mj_objectVelocity(m, d, (int)MujocoLib.mjtObj.mjOBJ_BODY, foot, va, 0);
                    float slip = (float)System.Math.Sqrt(va[3] * va[3] + va[4] * va[4]);
                    float s = Mathf.Clamp01((slip - 0.15f) / 1.0f);
                    if (s > slide) { slide = s; slideAt = U(d->xpos[3 * foot], d->xpos[3 * foot + 1], d->xpos[3 * foot + 2]); }
                }
                else if (r1 >= 0 && r2 >= 0 && r1 != r2)
                {
                    MujocoLib.mj_objectVelocity(m, d, (int)MujocoLib.mjtObj.mjOBJ_BODY, b1, va, 0);
                    MujocoLib.mj_objectVelocity(m, d, (int)MujocoLib.mjtObj.mjOBJ_BODY, b2, vb, 0);
                    double dx = va[3] - vb[3], dy = va[4] - vb[4], dz = va[5] - vb[5];
                    float rel = Mathf.Clamp01(((float)System.Math.Sqrt(dx * dx + dy * dy + dz * dz) - 0.2f) / 1.2f);
                    if (rel > grind) { grind = rel; metal = robots[r1].body == "g1" && robots[r2].body == "g1" ? 1f : 0f; }
                }
            }
            if (slide > 0f) { Place(slideAt, out float pan, out float gain); slide *= gain; _slidePan = pan; }
            _slide = slide; _grind = grind; _grindMetal = metal; Slide = slide; Grind = grind;
        }

        int NearestOther(MujocoLib.mjData_* d, int i)
        {
            double* q = d->qpos + robots[i].Map.RootQposAdr; int best = -1; double bestD = reach * reach;
            for (int j = 0; j < robots.Length; j++)
            {
                if (j == i || Parked(d, j)) continue; double* o = d->qpos + robots[j].Map.RootQposAdr;
                double dx = o[0] - q[0], dy = o[1] - q[1], dz = o[2] - q[2], dd = dx * dx + dy * dy + dz * dz;
                if (dd < bestD) { bestD = dd; best = j; }
            }
            return best;
        }

        float Noise() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return (_rng & 0xFFFF) / 32768f - 1f; }

        void OnAudioFilterRead(float[] data, int channels)
        {
            Blocks++;
            float t = _thud; if (t > 0) { _eThud = Mathf.Max(_eThud, t); _pThud = _thudPan; _thud = 0; }
            float c = _click; if (c > 0) { _eClick = Mathf.Max(_eClick, c); _pClick = _clickPan; _click = 0; }
            float k = _clank; if (k > 0) { _eClank = Mathf.Max(_eClank, k); _pClank = _clankPan; _clank = 0; }
            float sp = _splash; if (sp > 0) { _eSplash = Mathf.Max(_eSplash, sp); _pSplash = _splashPan; _splash = 0; }
            float dThud = Mathf.Exp(-1f / (0.09f * _rate)), dClick = Mathf.Exp(-1f / (0.012f * _rate));
            float dClank = Mathf.Exp(-1f / (0.28f * _rate)), dSplash = Mathf.Exp(-1f / (0.35f * _rate));
            float w = 2f * Mathf.PI * _thudPitch / _rate, under = _under;
            float w1 = 2f * Mathf.PI * 330f / _rate, w2 = w1 * 2.76f, w3 = w1 * 5.40f;
            // sustained voices glide to their targets (about 60 ms); ambience over about 0.5 s
            float glide = 1f - Mathf.Exp(-1f / (0.06f * _rate)), drift = 1f - Mathf.Exp(-1f / (0.5f * _rate));
            float slideT = _slide, slidePan = _slidePan, grindT = _grind, metal = _grindMetal, windT = _wind, surfT = _surf, crowdT = _crowd, rumbleT = _rumble;
            float sfx = volume * _sfxGain, amb = volume * _ambGain, muffleT = _muffle;
            float surfW = 2f * Mathf.PI / (6.5f * _rate), lfoW1 = 2f * Mathf.PI * 0.23f / _rate, lfoW2 = 2f * Mathf.PI * 0.071f / _rate, lfoW3 = 2f * Mathf.PI * 1.7f / _rate;
            float wg = 2f * Mathf.PI * 1900f / _rate; double eW = 0, eS = 0, eC = 0, eR = 0;
            for (int i = 0; i < data.Length; i += channels)
            {
                float nz = Noise(), nz2 = Noise();
                _lp += 0.08f * (nz - _lp);                                       // low-passed noise for splash and bubbles
                _bp1 += 0.35f * (nz - _bp1); _bp2 += 0.35f * (_bp1 - _bp2);      // band-passed noise for the rock scrape
                float scrape = _bp1 - _bp2;
                _phase += w; if (_phase > 2f * Mathf.PI) _phase -= 2f * Mathf.PI;
                _c1 += w1; _c2 += w2; _c3 += w3; if (_c1 > 6.2831853f) _c1 -= 6.2831853f; if (_c2 > 6.2831853f) _c2 -= 6.2831853f; if (_c3 > 6.2831853f) _c3 -= 6.2831853f;
                _sSlide += glide * (slideT - _sSlide); _sGrind += glide * (grindT - _sGrind);
                _sWind += drift * (windT - _sWind); _sSurf += drift * (surfT - _sSurf); _sCrowd += drift * (crowdT - _sCrowd); _sRumble += drift * (rumbleT - _sRumble);
                _sMuffle += drift * (muffleT - _sMuffle);
                _surfPh += surfW; if (_surfPh > 6.2831853f) _surfPh -= 6.2831853f;
                _lfo1 += lfoW1; _lfo2 += lfoW2; _lfo3 += lfoW3; if (_lfo1 > 6.2831853f) _lfo1 -= 6.2831853f; if (_lfo2 > 6.2831853f) _lfo2 -= 6.2831853f; if (_lfo3 > 6.2831853f) _lfo3 -= 6.2831853f;

                // one-shots (stereo fallback) and footsteps
                float sThud = _eThud * (0.8f * Mathf.Sin(_phase) + 0.25f * nz), sClick = _eClick * 0.5f * nz;
                float sClank = _eClank * 0.35f * (Mathf.Sin(_c1) + 0.6f * Mathf.Sin(_c2) + 0.35f * Mathf.Sin(_c3));
                float sSplash = _eSplash * 1.6f * _lp, sUnder = under * 0.05f * _lp;
                _eThud *= dThud; _eClick *= dClick; _eClank *= dClank; _eSplash *= dSplash;
                // friction: a sliding foot scrapes, rubbing fighters grind (metal: a ringing partial over the noise)
                float sSlide = _sSlide * 0.8f * scrape;
                _gb1 += 0.5f * (nz2 - _gb1); _gb2 += 0.5f * (_gb1 - _gb2);
                float sGrind = _sGrind * (metal > 0.5f ? 0.35f * (_gb1 - _gb2) * (1f + 0.8f * Mathf.Sin(_c2 * 2.3f)) : 0.5f * _lp);
                // ambience: wind (two decorrelated low-passed noises, gusting), surf (brown noise swelling every ~6.5 s),
                // crowd (noise through three vowel-ish formants, murmuring), quake rumble
                float gust = 0.65f + 0.35f * Mathf.Sin(_lfo1) * Mathf.Sin(_lfo2 * 3f);
                float cut = 0.015f + 0.05f * gust * Mathf.Clamp01(_sWind * 1.5f);
                _wl += cut * (nz - _wl); _wr += cut * (nz2 - _wr);
                _brown = Mathf.Clamp(_brown + 0.015f * nz, -1f, 1f) * 0.9993f;
                float wave = 0.35f + 0.65f * Mathf.Pow(0.5f + 0.5f * Mathf.Sin(_surfPh), 3f);
                float sSurf = _sSurf * 1.4f * _brown * wave;
                _rl += 0.004f * (nz - _rl);
                float sRumble = _sRumble * 9f * _rl * (0.7f + 0.3f * Mathf.Sin(_lfo3));
                float murmur = 0.6f + 0.4f * Mathf.Sin(_lfo3 * 1.3f + Mathf.Sin(_lfo1 * 7f) * 2f);
                float sCrowd = _sCrowd * 0.5f * murmur * (_crowdA.Step(nz) + 0.7f * _crowdB.Step(nz2) + 0.35f * _crowdC.Step(nz) * Mathf.Clamp01(_sCrowd * 2f));
                float windL = _sWind * 6f * _wl * gust, windR = _sWind * 6f * _wr * gust;

                // equal-power-ish pan per voice: g = 0.5 (1 ± pan) + 0.5
                float L = sThud * (1 - 0.5f * _pThud) + sClick * (1 - 0.5f * _pClick) + sSlide * (1 - 0.5f * slidePan) + sClank * (1 - 0.5f * _pClank) + sSplash * (1 - 0.5f * _pSplash) + sUnder + sGrind;
                float R = sThud * (1 + 0.5f * _pThud) + sClick * (1 + 0.5f * _pClick) + sSlide * (1 + 0.5f * slidePan) + sClank * (1 + 0.5f * _pClank) + sSplash * (1 + 0.5f * _pSplash) + sUnder + sGrind;
                eW += windL * windL * amb * amb; eS += sSurf * sSurf * amb * amb; eC += sCrowd * sCrowd * amb * amb; eR += sRumble * sRumble * amb * amb;
                L = L * sfx + (windL + sSurf + sCrowd + sRumble) * amb; R = R * sfx + (windR + sSurf * 0.9f + sCrowd * 0.95f + sRumble) * amb;
                // muffle: a one-pole low-pass from 20 kHz down to ~650 Hz under water or in slowed time
                float a = 1f - Mathf.Exp(-wg * Mathf.Lerp(10f, 0.34f, Mathf.Sqrt(Mathf.Clamp01(_sMuffle))));
                _mL += a * (L - _mL); _mR += a * (R - _mR); L = _mL; R = _mR;
                L = Mathf.Clamp(L, -1f, 1f); R = Mathf.Clamp(R, -1f, 1f);
                float pk = Mathf.Max(Mathf.Abs(L), Mathf.Abs(R)); if (pk > Peak) Peak = pk;
                if (channels >= 2) { data[i] = L; data[i + 1] = R; for (int ch = 2; ch < channels; ch++) data[i + ch] = 0.5f * (L + R); }
                else data[i] = 0.5f * (L + R);
            }
            int frames = Mathf.Max(1, data.Length / Mathf.Max(1, channels));
            WindRms = Mathf.Sqrt((float)(eW / frames)); SurfRms = Mathf.Sqrt((float)(eS / frames)); CrowdRms = Mathf.Sqrt((float)(eC / frames)); RumbleRms = Mathf.Sqrt((float)(eR / frames));
        }

        /// <summary>Two-pole resonator (a formant): centre frequency and bandwidth in Hz.</summary>
        struct Reson
        {
            float a1, a2, g, y1, y2;
            public Reson(float f, float bw, int rate)
            {
                float r = Mathf.Exp(-Mathf.PI * bw / rate); a1 = 2 * r * Mathf.Cos(2 * Mathf.PI * f / rate); a2 = -r * r; g = 1 - r; y1 = y2 = 0;
            }
            public float Step(float x) { float y = g * x + a1 * y1 + a2 * y2; y2 = y1; y1 = y; return y; }
        }
    }
}
