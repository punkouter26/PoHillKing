using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Runtime-synthesized sound, no audio files. Driven only by the MuJoCo state:
    ///   impact thud   a robot pelvis changes velocity sharply within 20 ms (hit, shove, landing); loudness and pitch
    ///                 scale with the velocity jump. When another robot is within reach it is a robot-on-robot hit and
    ///                 a metallic clank (three inharmonic decaying partials) rings over the thud
    ///   footfall      a foot's downward speed stops near the ground; a short click scaled by that speed, plus a rock
    ///                 scrape (band-passed noise) when the foot is still sliding as it lands
    ///   splash        a pelvis crosses the water surface going down; then a quiet bubbling hum while submerged
    /// Every voice is panned and attenuated by where it happened relative to the camera (stereo, listener-relative).
    /// The same triggers are announced through MatchEvents for particles, decals, camera shake and commentary.
    /// Physics callback writes trigger values; the audio thread turns them into decaying noise and sine bursts.
    /// </summary>
    [RequireComponent(typeof(AudioSource))]
    public unsafe class ImpactSynth : MonoBehaviour
    {
        public PolicyRunner[] robots;
        public Sea sea;
        [Range(0f, 1f)] public float volume = 0.6f;
        public float impactThreshold = 0.45f;     // m/s of pelvis velocity change within 20 ms
        public float reach = 0.8f;                // another pelvis this close makes a hit a robot-on-robot clank
        long _tick; double _quietUntil;

        // Verification counters. Triggers are counted where the physics raises them; Blocks and Peak come from the
        // audio thread, which the editor only runs while it has focus.
        public int Thuds, Clicks, Splashes, Clanks, Blocks; public float Peak;
        double[][] _prevVel; double[] _prevFootVz; bool[] _wasUnder; int[] _footBody;
        // triggers written by the physics callback, consumed by the audio thread (strength, pan -1..1)
        volatile float _thud, _thudPitch = 70f, _thudPan, _click, _clickPan, _scrape, _scrapePan, _clank, _clankPan, _splash, _splashPan, _under;
        float _eThud, _eClick, _eScrape, _eClank, _eSplash, _pThud, _pClick, _pScrape, _pClank, _pSplash, _phase, _lp, _bp1, _bp2; float _c1, _c2, _c3;
        int _rate = 48000; uint _rng = 12345u; Transform _listener;

        void OnEnable()
        {
            _rate = AudioSettings.outputSampleRate;
            var src = GetComponent<AudioSource>(); src.playOnAwake = true; src.loop = true; src.spatialBlend = 0f;
            if (src.clip == null) src.clip = AudioClip.Create("silence", _rate, 1, _rate, false);   // carrier so OnAudioFilterRead runs
            src.Play();
            MjScene.Instance.postInitEvent += OnInit; MjScene.Instance.postUpdateEvent += Post;
        }
        void OnDisable()
        {
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.postInitEvent -= OnInit; MjScene.Instance.postUpdateEvent -= Post;
        }

        void OnInit(object s, MjStepArgs a) { _prevVel = null; }
        void Update() { volume = GameSettings.Volume; if (_listener == null && Camera.main != null) _listener = Camera.main.transform; }

        // MuJoCo (x, y, z) -> Unity (x, z, y); pan and distance gain from the camera's point of view.
        static Vector3 U(double x, double y, double z) => new((float)x, (float)z, (float)y);
        void Place(Vector3 p, out float pan, out float gain)
        {
            if (_listener == null) { pan = 0; gain = 1; return; }
            Vector3 rel = _listener.InverseTransformPoint(p); float dist = rel.magnitude;
            pan = Mathf.Clamp(rel.x / Mathf.Max(1f, dist), -1f, 1f); gain = Mathf.Clamp01(7f / Mathf.Max(7f, dist));
        }

        void Post(object s, MjStepArgs a)
        {
            if (robots == null || robots.Length == 0 || robots[0].Map == null) return;
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data; int n = robots.Length;
            if (_prevVel == null)
            {
                _prevVel = new double[n][]; _prevFootVz = new double[2 * n]; _wasUnder = new bool[n]; _footBody = new int[2 * n];
                for (int i = 0; i < n; i++)
                {
                    _prevVel[i] = new double[3];
                    _footBody[2 * i] = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, robots[i].robotPrefix + "left_ankle_roll_link");
                    _footBody[2 * i + 1] = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, robots[i].robotPrefix + "right_ankle_roll_link");
                }
            }
            float under = 0f; _tick++;
            for (int i = 0; i < n; i++)
            {
                if (robots[i].Map == null || robots[i].paused && robots[i].policy == null && d->qpos[robots[i].Map.RootQposAdr + 2] > 20) continue;   // parked in the sky
                double* v = d->qvel + robots[i].Map.RootDofAdr; double* q = d->qpos + robots[i].Map.RootQposAdr;
                // impact: pelvis velocity change over a 20 ms window (a single 2 ms step never shows a big enough jump)
                if (_tick % 10 == 0)
                {
                    double dv = System.Math.Sqrt((v[0] - _prevVel[i][0]) * (v[0] - _prevVel[i][0]) + (v[1] - _prevVel[i][1]) * (v[1] - _prevVel[i][1]) + (v[2] - _prevVel[i][2]) * (v[2] - _prevVel[i][2]));
                    if (dv > impactThreshold && dv < 20 && d->time > _quietUntil)
                    {
                        float strength = Mathf.Clamp01((float)dv / 2.5f); Vector3 p = U(q[0], q[1], q[2]); Place(p, out float pan, out float gain);
                        _thud = Mathf.Max(_thud, strength * gain); _thudPitch = Mathf.Lerp(55f, 110f, Mathf.Clamp01((float)dv / 3f)); _thudPan = pan;
                        _quietUntil = d->time + 0.12; Thuds++;
                        if (NearOther(d, i)) { _clank = Mathf.Max(_clank, strength * gain); _clankPan = pan; Clanks++; }
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
                        if (slip > 0.25f) { _scrape = Mathf.Max(_scrape, Mathf.Clamp01(slip / 1.2f) * gain); _scrapePan = pan; }
                        _prevFootVz[idx] = 0; Clicks++;
                        MatchEvents.RaiseFootfall(p, strength, slip);
                    }
                }
                if (sea != null)
                {
                    bool isUnder = q[2] < sea.Level;
                    if (isUnder && !_wasUnder[i])
                    {
                        Vector3 p = U(q[0], q[1], sea.Level); Place(p, out float pan, out float gain); float strength = Mathf.Clamp01((float)System.Math.Abs(v[2]) / 4f + 0.4f);
                        _splash = Mathf.Max(_splash, strength * gain); _splashPan = pan; Splashes++;
                        MatchEvents.RaiseSplash(p, strength);
                    }
                    _wasUnder[i] = isUnder; if (isUnder) under = 1f;
                }
            }
            _under = under;
        }

        bool NearOther(MujocoLib.mjData_* d, int i)
        {
            double* q = d->qpos + robots[i].Map.RootQposAdr;
            for (int j = 0; j < robots.Length; j++)
            {
                if (j == i || robots[j].Map == null) continue; double* o = d->qpos + robots[j].Map.RootQposAdr;
                double dx = o[0] - q[0], dy = o[1] - q[1], dz = o[2] - q[2]; if (dx * dx + dy * dy + dz * dz < reach * reach) return true;
            }
            return false;
        }

        float Noise() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return (_rng & 0xFFFF) / 32768f - 1f; }

        void OnAudioFilterRead(float[] data, int channels)
        {
            Blocks++;
            float t = _thud; if (t > 0) { _eThud = Mathf.Max(_eThud, t); _pThud = _thudPan; _thud = 0; }
            float c = _click; if (c > 0) { _eClick = Mathf.Max(_eClick, c); _pClick = _clickPan; _click = 0; }
            float sc = _scrape; if (sc > 0) { _eScrape = Mathf.Max(_eScrape, sc); _pScrape = _scrapePan; _scrape = 0; }
            float k = _clank; if (k > 0) { _eClank = Mathf.Max(_eClank, k); _pClank = _clankPan; _clank = 0; }
            float sp = _splash; if (sp > 0) { _eSplash = Mathf.Max(_eSplash, sp); _pSplash = _splashPan; _splash = 0; }
            float dThud = Mathf.Exp(-1f / (0.09f * _rate)), dClick = Mathf.Exp(-1f / (0.012f * _rate)), dScrape = Mathf.Exp(-1f / (0.16f * _rate));
            float dClank = Mathf.Exp(-1f / (0.28f * _rate)), dSplash = Mathf.Exp(-1f / (0.35f * _rate));
            float w = 2f * Mathf.PI * _thudPitch / _rate, under = _under;
            float w1 = 2f * Mathf.PI * 330f / _rate, w2 = w1 * 2.76f, w3 = w1 * 5.40f;
            for (int i = 0; i < data.Length; i += channels)
            {
                float nz = Noise(); _lp += 0.08f * (nz - _lp);                 // low-passed noise for splash and bubbles
                _bp1 += 0.35f * (nz - _bp1); _bp2 += 0.35f * (_bp1 - _bp2);      // band-passed noise for the rock scrape
                float scrape = _bp1 - _bp2;
                _phase += w; if (_phase > 2f * Mathf.PI) _phase -= 2f * Mathf.PI;
                _c1 += w1; _c2 += w2; _c3 += w3; if (_c1 > 6.2831853f) _c1 -= 6.2831853f; if (_c2 > 6.2831853f) _c2 -= 6.2831853f; if (_c3 > 6.2831853f) _c3 -= 6.2831853f;
                float sThud = _eThud * (0.8f * Mathf.Sin(_phase) + 0.25f * nz), sClick = _eClick * 0.5f * nz, sScrape = _eScrape * 0.9f * scrape;
                float sClank = _eClank * 0.35f * (Mathf.Sin(_c1) + 0.6f * Mathf.Sin(_c2) + 0.35f * Mathf.Sin(_c3));
                float sSplash = _eSplash * 1.6f * _lp, sUnder = under * 0.05f * _lp;
                _eThud *= dThud; _eClick *= dClick; _eScrape *= dScrape; _eClank *= dClank; _eSplash *= dSplash;
                // equal-power-ish pan per voice: g = 0.5 (1 ± pan) + 0.5
                float L = sThud * (1 - 0.5f * _pThud) + sClick * (1 - 0.5f * _pClick) + sScrape * (1 - 0.5f * _pScrape) + sClank * (1 - 0.5f * _pClank) + sSplash * (1 - 0.5f * _pSplash) + sUnder;
                float R = sThud * (1 + 0.5f * _pThud) + sClick * (1 + 0.5f * _pClick) + sScrape * (1 + 0.5f * _pScrape) + sClank * (1 + 0.5f * _pClank) + sSplash * (1 + 0.5f * _pSplash) + sUnder;
                L = Mathf.Clamp(L * volume, -1f, 1f); R = Mathf.Clamp(R * volume, -1f, 1f);
                float pk = Mathf.Max(Mathf.Abs(L), Mathf.Abs(R)); if (pk > Peak) Peak = pk;
                if (channels >= 2) { data[i] = L; data[i + 1] = R; for (int ch = 2; ch < channels; ch++) data[i + ch] = 0.5f * (L + R); }
                else data[i] = 0.5f * (L + R);
            }
        }
    }
}
