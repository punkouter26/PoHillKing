using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Runtime-synthesized sound, no audio files. Driven only by the MuJoCo state:
    ///   impact thud   a robot pelvis changes velocity sharply within one physics step (hit, shove, landing);
    ///                 loudness and pitch scale with the velocity jump
    ///   footfall      a foot's downward speed stops near the ground; a short click scaled by that speed
    ///   splash        a pelvis crosses the water surface going down; then a quiet bubbling hum while submerged
    /// Physics thread writes trigger values; the audio thread turns them into decaying noise and sine bursts.
    /// </summary>
    [RequireComponent(typeof(AudioSource))]
    public unsafe class ImpactSynth : MonoBehaviour
    {
        public PolicyRunner[] robots;
        public Sea sea;
        [Range(0f, 1f)] public float volume = 0.6f;
        public float impactThreshold = 0.45f;     // m/s of pelvis velocity change within 20 ms
        long _tick; double _quietUntil;

        // Verification counters. Triggers are counted where the physics raises them; Blocks and Peak come from the
        // audio thread, which the editor only runs while it has focus.
        public int Thuds, Clicks, Splashes, Blocks; public float Peak;
        double[][] _prevVel; double[] _prevFootVz; bool[] _wasUnder; int[] _footBody;
        // triggers written by the physics callback, consumed by the audio thread
        volatile float _thud, _thudPitch = 70f, _click, _splash, _under;
        float _eThud, _eClick, _eSplash, _phase, _lp; int _rate = 48000; uint _rng = 12345u;

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
                if (robots[i].Map == null) continue;
                double* v = d->qvel + robots[i].Map.RootDofAdr; double* q = d->qpos + robots[i].Map.RootQposAdr;
                // impact: pelvis velocity change over a 20 ms window (a single 2 ms step never shows a big enough jump)
                if (_tick % 10 == 0)
                {
                    double dv = System.Math.Sqrt((v[0] - _prevVel[i][0]) * (v[0] - _prevVel[i][0]) + (v[1] - _prevVel[i][1]) * (v[1] - _prevVel[i][1]) + (v[2] - _prevVel[i][2]) * (v[2] - _prevVel[i][2]));
                    if (dv > impactThreshold && dv < 20 && d->time > _quietUntil)
                    {
                        _thud = Mathf.Max(_thud, Mathf.Clamp01((float)dv / 2.5f)); _thudPitch = Mathf.Lerp(55f, 110f, Mathf.Clamp01((float)dv / 3f));
                        _quietUntil = d->time + 0.12; Thuds++;
                    }
                    for (int k = 0; k < 3; k++) _prevVel[i][k] = v[k];
                }
                // footfall: a foot that was moving down faster than 0.25 m/s comes to rest
                for (int f = 0; f < 2; f++)
                {
                    int b = _footBody[2 * i + f]; if (b < 0) continue; int idx = 2 * i + f;
                    double vz = d->cvel[6 * b + 5];
                    if (vz < -0.25) _prevFootVz[idx] = System.Math.Min(_prevFootVz[idx], vz);                 // remember the fastest descent
                    else if (vz > -0.05 && _prevFootVz[idx] < -0.25) { _click = Mathf.Max(_click, Mathf.Clamp01((float)(-_prevFootVz[idx]) / 1.5f)); _prevFootVz[idx] = 0; Clicks++; }
                }
                if (sea != null)
                {
                    bool isUnder = q[2] < sea.Level;
                    if (isUnder && !_wasUnder[i]) { _splash = Mathf.Max(_splash, Mathf.Clamp01((float)System.Math.Abs(v[2]) / 4f + 0.4f)); Splashes++; }
                    _wasUnder[i] = isUnder; if (isUnder) under = 1f;
                }
            }
            _under = under;
        }

        float Noise() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return (_rng & 0xFFFF) / 32768f - 1f; }

        void OnAudioFilterRead(float[] data, int channels)
        {
            Blocks++;
            float t = _thud; if (t > 0) { _eThud = Mathf.Max(_eThud, t); _thud = 0; }
            float c = _click; if (c > 0) { _eClick = Mathf.Max(_eClick, c); _click = 0; }
            float sp = _splash; if (sp > 0) { _eSplash = Mathf.Max(_eSplash, sp); _splash = 0; }
            float dThud = Mathf.Exp(-1f / (0.09f * _rate)), dClick = Mathf.Exp(-1f / (0.012f * _rate)), dSplash = Mathf.Exp(-1f / (0.35f * _rate));
            float w = 2f * Mathf.PI * _thudPitch / _rate, under = _under;
            for (int i = 0; i < data.Length; i += channels)
            {
                float nz = Noise(); _lp += 0.08f * (nz - _lp);                 // low-passed noise for splash and bubbles
                _phase += w; if (_phase > 2f * Mathf.PI) _phase -= 2f * Mathf.PI;
                float s = _eThud * (0.8f * Mathf.Sin(_phase) + 0.25f * nz) + _eClick * 0.5f * nz + _eSplash * 1.6f * _lp + under * 0.05f * _lp;
                _eThud *= dThud; _eClick *= dClick; _eSplash *= dSplash;
                s = Mathf.Clamp(s * volume, -1f, 1f); if (s > Peak) Peak = s; else if (-s > Peak) Peak = -s;
                for (int ch = 0; ch < channels; ch++) data[i + ch] = s;
            }
        }
    }
}
