using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Runs duel rounds on the summit: both robots start on top of the hill, a round ends when one leaves the
    /// plateau (radius &gt; 1.7 m), falls, or the bell rings, then both are reset by writing mjData (no scene reload).
    /// Also draws a minimal HUD (title, score, round clock, FPS, physics and inference cost).
    /// </summary>
    public unsafe class DemoDirector : MonoBehaviour
    {
        public PolicyRunner attacker, defender;
        public float roundSeconds = 15f, outRadius = 1.7f, pauseBetweenRounds = 1.2f;

        int _winsA, _winsB, _draws; double _roundStart; float _pauseUntil; string _last = ""; bool _pending;
        float _fps; readonly System.Diagnostics.Stopwatch _sw = new(); double _stepMs;

        void OnEnable()
        {
            MjScene.Instance.postInitEvent += OnInit;
            MjScene.Instance.preUpdateEvent += Pre; MjScene.Instance.postUpdateEvent += Post;
        }
        void OnDisable()
        {
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.postInitEvent -= OnInit; MjScene.Instance.preUpdateEvent -= Pre; MjScene.Instance.postUpdateEvent -= Post;
        }

        void OnInit(object s, MjStepArgs a) => _pending = true;   // place the robots on the first physics step, after every runner has its map
        void Pre(object s, MjStepArgs a)
        {
            if (_pending && attacker.Map != null && defender.Map != null) { NewRound(); _pending = false; }
            _sw.Restart();
        }

        void NewRound()
        {
            // Everyone starts on the summit: attacker near the middle, defender standing near the rim.
            float bearing = Random.Range(0f, 2f * Mathf.PI), ra = Random.Range(0f, 0.5f), rb = Random.Range(1.2f, 1.3f);
            attacker.ResetPose(ra * Mathf.Cos(bearing), ra * Mathf.Sin(bearing), Random.Range(-Mathf.PI, Mathf.PI));
            defender.ResetPose(rb * Mathf.Cos(bearing + Mathf.PI), rb * Mathf.Sin(bearing + Mathf.PI), Random.Range(-Mathf.PI, Mathf.PI));
            attacker.paused = defender.paused = false;
            _roundStart = MjScene.Instance.Data->time;
        }

        static bool Out(MujocoLib.mjData_* d, PolicyRunner r, float outRadius)
        {
            double* q = d->qpos + r.Map.RootQposAdr;
            double rad = System.Math.Sqrt(q[0] * q[0] + q[1] * q[1]);
            double ground = -(1.0 / 9.0) * System.Math.Pow(System.Math.Max(rad - 1.5, 0), 2);
            double up = 1 - 2 * (q[4] * q[4] + q[5] * q[5]);
            return rad > outRadius || up < 0.3 || q[2] - ground < 0.3;
        }

        void Post(object s, MjStepArgs a)
        {
            _stepMs = 0.95 * _stepMs + 0.05 * _sw.Elapsed.TotalMilliseconds;
            if (attacker.Map == null || defender.Map == null || _pending) return;
            var d = MjScene.Instance.Data;
            if (_pauseUntil > 0) { if (Time.unscaledTime >= _pauseUntil) { _pauseUntil = 0; NewRound(); } return; }
            bool aOut = Out(d, attacker, outRadius), bOut = Out(d, defender, outRadius);
            double t = d->time - _roundStart;
            if (!aOut && !bOut && t < roundSeconds) return;
            if (bOut) { _winsA++; _last = $"Attacker wins in {t:0.0} s"; }
            else if (aOut) { _winsB++; _last = $"Defender wins in {t:0.0} s"; }
            else { _draws++; _last = "Draw"; }
            _pauseUntil = Time.unscaledTime + pauseBetweenRounds;
        }

        void Update() => _fps = Mathf.Lerp(_fps, 1f / Mathf.Max(Time.unscaledDeltaTime, 1e-4f), 0.05f);

        void OnGUI()
        {
            int w = Screen.width, h = Screen.height; float k = Mathf.Max(1f, h / 900f);
            var st = new GUIStyle(GUI.skin.label) { fontSize = Mathf.RoundToInt(15 * k), normal = { textColor = Color.white } };
            var big = new GUIStyle(st) { fontSize = Mathf.RoundToInt(20 * k), fontStyle = FontStyle.Bold };
            var right = new GUIStyle(st) { alignment = TextAnchor.UpperRight }; var mid = new GUIStyle(st) { alignment = TextAnchor.UpperCenter };
            float pad = 12 * k, lh = 26 * k;
            GUI.Label(new Rect(pad, pad, w, lh), "PoKingHill", big);                                                          // top left: title
            double t = MjScene.InstanceExists && MjScene.Instance.Data != null ? MjScene.Instance.Data->time - _roundStart : 0;
            GUI.Label(new Rect(0, pad, w, lh * 2), $"{_fps:0} FPS   physics {_stepMs:0.00} ms   brain {attacker.LastInferenceMs:0.00} ms\nround {t:0.0} s", mid);   // top centre: telemetry
            GUI.Label(new Rect(0, pad, w - pad, lh), "Duel: Attacker vs standing Walker", right);                          // top right: behaviour
            GUI.Label(new Rect(pad, h - pad - lh * 2, w, lh * 2), $"Attacker {_winsA}   Defender {_winsB}   Draws {_draws}\n{_last}", st);   // bottom left
            GUI.Label(new Rect(0, h - pad - lh, w - pad, lh), "v0.1 parity build", right);                                    // bottom right: version
        }
    }
}
