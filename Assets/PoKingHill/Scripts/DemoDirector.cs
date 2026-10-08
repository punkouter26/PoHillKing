using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Runs king-of-the-hill rounds on the summit. Both robots start on top of the hill. A round ends when one
    /// leaves the plateau (radius &gt; outRadius), falls, or the sea reaches the summit (tie). Robots are then reset
    /// by writing mjData (no scene reload). Draws a minimal HUD: title, telemetry, behaviour, score, version.
    /// </summary>
    public unsafe class DemoDirector : MonoBehaviour
    {
        public PolicyRunner attacker, defender;          // robot A and robot B
        public Sea sea;                                  // optional; without it the round lasts roundSeconds
        public string labelA = "Robot A", labelB = "Robot B", behaviour = "King of the hill";
        public Vector2 spawnRadiusA = new(0.5f, 1.2f), spawnRadiusB = new(0.5f, 1.2f);
        public float roundSeconds = 25f, outRadius = 1.7f, pauseBetweenRounds = 3.5f;
        [Tooltip("Statistical parity gate: play this many rounds fast, write duel_stats_unity.json, then stop. 0 = off.")]
        public int statsRounds = 0;
        /// <summary>The robot the camera should follow after a decided round (the loser), else null.</summary>
        public PolicyRunner FollowTarget { get; private set; }
        readonly System.Collections.Generic.List<double> _times = new();

        int _winsA, _winsB, _draws; double _roundStart; float _pauseUntil; string _last = ""; bool _pending;
        float _fps; readonly System.Diagnostics.Stopwatch _sw = new(); double _stepMs;

        void OnEnable()
        {
            Application.targetFrameRate = 60;            // the editor otherwise renders uncapped and starves GPU training
            // The unfocused editor renders about one frame per second, so let each frame carry up to 2 s of physics.
            if (statsRounds > 0) { Application.runInBackground = true; Time.timeScale = 50f; Time.maximumDeltaTime = 2f; pauseBetweenRounds = 0f; }
            MjScene.Instance.postInitEvent += OnInit;
            MjScene.Instance.preUpdateEvent += Pre; MjScene.Instance.postUpdateEvent += Post;
        }
        void OnDisable()
        {
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.postInitEvent -= OnInit; MjScene.Instance.preUpdateEvent -= Pre; MjScene.Instance.postUpdateEvent -= Post;
        }

        void OnInit(object s, MjStepArgs a) => _pending = true;   // place the robots on the first physics step, once every runner has its map
        void Pre(object s, MjStepArgs a)
        {
            if (_pending && attacker.Map != null && defender.Map != null) { NewRound(); _pending = false; }
            _sw.Restart();
        }

        void NewRound()
        {
            float bearing = Random.Range(0f, 2f * Mathf.PI);
            float ra = Random.Range(spawnRadiusA.x, spawnRadiusA.y), rb = Random.Range(spawnRadiusB.x, spawnRadiusB.y);
            attacker.ResetPose(ra * Mathf.Cos(bearing), ra * Mathf.Sin(bearing), Random.Range(-Mathf.PI, Mathf.PI));
            defender.ResetPose(rb * Mathf.Cos(bearing + Mathf.PI), rb * Mathf.Sin(bearing + Mathf.PI), Random.Range(-Mathf.PI, Mathf.PI));
            _roundStart = MjScene.Instance.Data->time; FollowTarget = null;
            if (sea != null) sea.Restart();
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
            bool bell = sea != null ? sea.ReachedSummit : t >= roundSeconds;
            if (!aOut && !bOut && !bell) return;
            if (bOut && !aOut) { _winsA++; _times.Add(t); _last = $"{labelA} wins in {t:0.0} s"; FollowTarget = defender; }
            else if (aOut && !bOut) { _winsB++; _times.Add(t); _last = $"{labelB} wins in {t:0.0} s"; FollowTarget = attacker; }
            else { _draws++; _last = bell ? "Tie: the sea took the summit" : "Tie: both out"; }
            _pauseUntil = Time.unscaledTime + pauseBetweenRounds;
            if (statsRounds > 0 && _winsA + _winsB + _draws >= statsRounds) WriteStats();
            else if (statsRounds > 0) { _pauseUntil = 0; NewRound(); }
        }

        void WriteStats()
        {
            _times.Sort(); int n = _winsA + _winsB + _draws;
            double Q(double f) => _times.Count == 0 ? -1 : _times[System.Math.Min(_times.Count - 1, (int)(f * _times.Count))];
            double mean = 0; foreach (var x in _times) mean += x; mean = _times.Count > 0 ? mean / _times.Count : -1;
            string F(double v) => v.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture);
            string json = $"{{\n \"rounds\": {n}, \"wins_a\": {_winsA}, \"wins_b\": {_winsB}, \"ties\": {_draws}, \"decided\": {F((double)(_winsA + _winsB) / n)},\n" +
                          $" \"median_time_s\": {F(Q(0.5))}, \"mean_time_s\": {F(mean)}, \"p25_time_s\": {F(Q(0.25))}, \"p75_time_s\": {F(Q(0.75))}\n}}\n";
            string path = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "training", "assets", "g1", "duel_stats_unity.json"));
            System.IO.File.WriteAllText(path, json); Debug.Log("[DemoDirector] wrote " + path + "\n" + json);
            statsRounds = 0; attacker.paused = defender.paused = true;
#if UNITY_EDITOR
            if (System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "-kothExit") >= 0) UnityEditor.EditorApplication.Exit(0);
#endif
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
            string seaText = sea != null ? $"   sea {sea.Level:0.0} m, arrives at {sea.Duration:0} s" : "";
            GUI.Label(new Rect(0, pad + lh, w, lh * 2), $"{_fps:0} FPS   physics {_stepMs:0.00} ms   brain {attacker.LastInferenceMs:0.00} ms\nround {t:0.0} s{seaText}", mid);   // top centre: telemetry
            GUI.Label(new Rect(0, pad, w - pad, lh), behaviour, right);                                                    // top right: behaviour
            GUI.Label(new Rect(pad, h - pad - lh * 2, w, lh * 2), $"{labelA} {_winsA}   {labelB} {_winsB}   Ties {_draws}\n{_last}", st);   // bottom left
            GUI.Label(new Rect(0, h - pad - lh, w - pad, lh), "v0.2", right);                                               // bottom right: version
        }
    }
}
