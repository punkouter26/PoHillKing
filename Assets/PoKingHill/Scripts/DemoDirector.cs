using System;
using System.Collections.Generic;
using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoKingHill
{
    [Serializable]
    public class Fighter
    {
        public string name = "Fighter";
        public ModelAsset policy;
        [Tooltip("103 = locomotion inputs, 112 = with opponent/ring inputs")] public int obsDim = 112;
        public GoalMode goal = GoalMode.Opponent;
        public float stopDist = 0f, vmax = 1f;
        [Tooltip("Trained in the shove style (arm pose + action smoothing)")] public bool shove;
    }

    /// <summary>
    /// Match flow for king of the hill: pre-match menu (fighter A, fighter B, random matchup, map), launch, round,
    /// result, back to the menu. Both robots always start on top of the hill. A round ends when one leaves the
    /// plateau (radius &gt; outRadius), falls, or the sea reaches the summit (tie). Everything is done by writing
    /// mjData (no scene reload, no Instantiate/Destroy). HUD anchors: TL title, TC telemetry, TR menu / behaviour,
    /// BL score and reset, BR version.
    /// </summary>
    public unsafe class DemoDirector : MonoBehaviour
    {
        public PolicyRunner attacker, defender;          // robot A and robot B
        public Sea sea;                                  // optional; without it the round lasts roundSeconds
        public ImpactSynth synth;                        // optional, for the audio counters in the verification file
        public Fighter[] roster = Array.Empty<Fighter>();
        public int selectedA, selectedB;
        public bool randomMatchup;
        public string[] maps = { "Summit (baseline)" };
        public int selectedMap;
        public Vector2 spawnRadiusA = new(0.5f, 1.2f), spawnRadiusB = new(0.5f, 1.2f);
        public float roundSeconds = 25f, outRadius = 1.7f, pauseBetweenRounds = 3.5f;
        [Tooltip("Rounds per launch before returning to the menu. 0 = keep fighting.")] public int roundsPerMatch = 1;
        [Tooltip("Skip the menu and start fighting immediately.")] public bool autoLaunch;
        [Tooltip("Statistical parity gate: play this many rounds fast, write duel_stats_unity.json, then stop. 0 = off.")]
        public int statsRounds = 0;

        /// <summary>The robot the camera should follow after a decided round (the loser), else null.</summary>
        public PolicyRunner FollowTarget { get; private set; }
        public bool InMenu { get; private set; } = true;

        readonly List<double> _times = new();
        int _winsA, _winsB, _draws, _roundsThisMatch; double _roundStart; float _pauseUntil; string _last = ""; bool _pending;
        float _fps; readonly System.Diagnostics.Stopwatch _sw = new(); double _stepMs;
        string _shotDir; readonly HashSet<string> _shots = new(); float _menuShotAt = -1;

        static string Arg(string name)
        {
            var a = Environment.GetCommandLineArgs(); int i = Array.IndexOf(a, name);
            return i >= 0 && i + 1 < a.Length ? a[i + 1] : null;
        }
        static bool Flag(string name) => Array.IndexOf(Environment.GetCommandLineArgs(), name) >= 0;

        void OnEnable()
        {
            Application.targetFrameRate = 60;            // the editor otherwise renders uncapped and starves GPU training
            Application.runInBackground = true;
            _shotDir = Arg("-kothShots");
            // Player-side statistical gate (-kothRounds N): same rules as ParityBatch.RunDuelStats, no sea and a 25 s bell.
            if (!Application.isEditor && int.TryParse(Arg("-kothRounds"), out int rounds) && rounds > 0)
            {
                statsRounds = rounds; roundSeconds = 25f;
                if (int.TryParse(Arg("-kothA"), out int fa)) selectedA = fa;
                if (int.TryParse(Arg("-kothB"), out int fb)) selectedB = fb;
                if (sea != null) { sea.enabled = false; sea = null; }
            }
            if (statsRounds > 0 || Flag("-kothAutoLaunch")) autoLaunch = true;
            // The unfocused editor renders about one frame per second, so in stats mode let each frame carry up to 2 s of physics.
            if (statsRounds > 0) { Time.timeScale = 50f; Time.maximumDeltaTime = 2f; pauseBetweenRounds = 0f; roundsPerMatch = 0; }
            MjScene.Instance.postInitEvent += OnInit;
            MjScene.Instance.preUpdateEvent += Pre; MjScene.Instance.postUpdateEvent += Post;
        }
        void OnDisable()
        {
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.postInitEvent -= OnInit; MjScene.Instance.preUpdateEvent -= Pre; MjScene.Instance.postUpdateEvent -= Post;
        }

        void OnInit(object s, MjStepArgs a) => _pending = true;   // act on the first physics step, once every runner has its map
        void Pre(object s, MjStepArgs a)
        {
            if (_pending && attacker.Map != null && defender.Map != null)
            {
                _pending = false;
                if (autoLaunch) Launch(); else EnterMenu();
            }
            _sw.Restart();
        }

        // ------------------------------------------------------------------ flow
        void EnterMenu()
        {
            InMenu = true; FollowTarget = null; _pauseUntil = 0; _roundsThisMatch = 0;
            attacker.SetBrain(null, ObsBuilder.ObsDim, GoalMode.None, 0.3f, 0.8f);      // passive stance while the menu is up
            defender.SetBrain(null, ObsBuilder.ObsDim, GoalMode.None, 0.3f, 0.8f);
            Place();
            if (sea != null) sea.Park();
            _menuShotAt = Time.unscaledTime + 1.5f;
        }

        public void Launch()
        {
            if (roster.Length > 0)
            {
                if (randomMatchup) { selectedA = UnityEngine.Random.Range(0, roster.Length); selectedB = UnityEngine.Random.Range(0, roster.Length); }
                var fa = roster[Mathf.Clamp(selectedA, 0, roster.Length - 1)]; var fb = roster[Mathf.Clamp(selectedB, 0, roster.Length - 1)];
                attacker.SetBrain(fa.policy, fa.obsDim, fa.goal, fa.stopDist, fa.vmax, fa.shove);
                defender.SetBrain(fb.policy, fb.obsDim, fb.goal, fb.stopDist, fb.vmax, fb.shove);
            }
            InMenu = false; _roundsThisMatch = 0;
            NewRound();
        }

        void Place()
        {
            float bearing = UnityEngine.Random.Range(0f, 2f * Mathf.PI);
            float ra = UnityEngine.Random.Range(spawnRadiusA.x, spawnRadiusA.y), rb = UnityEngine.Random.Range(spawnRadiusB.x, spawnRadiusB.y);
            // Everyone starts on the summit, on opposite sides. Yaw is random, as in training and duel_stats.py.
            attacker.ResetPose(ra * Mathf.Cos(bearing), ra * Mathf.Sin(bearing), UnityEngine.Random.Range(-Mathf.PI, Mathf.PI));
            defender.ResetPose(rb * Mathf.Cos(bearing + Mathf.PI), rb * Mathf.Sin(bearing + Mathf.PI), UnityEngine.Random.Range(-Mathf.PI, Mathf.PI));
        }

        void NewRound()
        {
            Place();
            _roundStart = MjScene.Instance.Data->time; FollowTarget = null;
            if (sea != null) sea.Restart();
        }

        static bool Out(MujocoLib.mjData_* d, PolicyRunner r, float outRadius)
        {
            double* q = d->qpos + r.Map.RootQposAdr;
            double rad = Math.Sqrt(q[0] * q[0] + q[1] * q[1]);
            double ground = -(1.0 / 9.0) * Math.Pow(Math.Max(rad - 1.5, 0), 2);
            double up = 1 - 2 * (q[4] * q[4] + q[5] * q[5]);
            return rad > outRadius || up < 0.3 || q[2] - ground < 0.3;
        }

        void Post(object s, MjStepArgs a)
        {
            _stepMs = 0.95 * _stepMs + 0.05 * _sw.Elapsed.TotalMilliseconds;
            if (attacker.Map == null || defender.Map == null || _pending || InMenu) return;
            var d = MjScene.Instance.Data;
            if (_pauseUntil > 0)
            {
                if (FollowTarget != null && Time.unscaledTime >= _pauseUntil - pauseBetweenRounds + 1.0f) Shot("ejection");
                if (Time.unscaledTime < _pauseUntil) return;
                _pauseUntil = 0;
                if (ShotsDone()) return;
                if (roundsPerMatch > 0 && _roundsThisMatch >= roundsPerMatch && !(_shotDir != null && Flag("-kothExit"))) EnterMenu(); else NewRound();
                return;
            }
            bool aOut = Out(d, attacker, outRadius), bOut = Out(d, defender, outRadius);
            double t = d->time - _roundStart;
            if (t > 1.2) Shot("combat");
            bool bell = sea != null ? sea.ReachedSummit : t >= roundSeconds;
            if (!aOut && !bOut && !bell) return;
            string na = NameOf(selectedA, "Robot A"), nb = NameOf(selectedB, "Robot B");
            if (bOut && !aOut) { _winsA++; _times.Add(t); _last = $"A ({na}) wins in {t:0.0} s"; FollowTarget = defender; }
            else if (aOut && !bOut) { _winsB++; _times.Add(t); _last = $"B ({nb}) wins in {t:0.0} s"; FollowTarget = attacker; }
            else { _draws++; _last = bell ? "Tie: the sea took the summit" : "Tie: both out"; }
            _roundsThisMatch++;
            _pauseUntil = Time.unscaledTime + pauseBetweenRounds;
            if (statsRounds > 0 && _winsA + _winsB + _draws >= statsRounds) WriteStats();
            else if (statsRounds > 0) { _pauseUntil = 0; NewRound(); }
        }

        string NameOf(int i, string fallback) => roster.Length > 0 ? roster[Mathf.Clamp(i, 0, roster.Length - 1)].name : fallback;

        // ------------------------------------------------------------------ verification helpers
        void Shot(string name)
        {
            if (_shotDir == null || !_shots.Add(name)) return;
            System.IO.Directory.CreateDirectory(_shotDir);
            ScreenCapture.CaptureScreenshot(System.IO.Path.Combine(_shotDir, name + ".png"));
            Debug.Log("[DemoDirector] screenshot " + name);
        }
        // Automated capture run (-kothShots dir -kothExit): leave once menu, combat and ejection frames exist.
        bool ShotsDone()
        {
            if (_shotDir == null || !Flag("-kothExit")) return false;
            if (!_shots.Contains("menu") || !_shots.Contains("combat") || !_shots.Contains("ejection")) return false;
            if (synth != null)
                System.IO.File.WriteAllText(System.IO.Path.Combine(_shotDir, "audio.json"),
                    $"{{\"thuds\": {synth.Thuds}, \"footfalls\": {synth.Clicks}, \"splashes\": {synth.Splashes}, \"peak_sample\": {synth.Peak.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture)}, \"audio_blocks\": {synth.Blocks}}}\n");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.Exit(0);
#endif
            return true;
        }

        void WriteStats()
        {
            _times.Sort(); int n = _winsA + _winsB + _draws;
            double Q(double f) => _times.Count == 0 ? -1 : _times[Math.Min(_times.Count - 1, (int)(f * _times.Count))];
            double mean = 0; foreach (var x in _times) mean += x; mean = _times.Count > 0 ? mean / _times.Count : -1;
            string F(double v) => v.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture);
            string json = $"{{\n \"rounds\": {n}, \"wins_a\": {_winsA}, \"wins_b\": {_winsB}, \"ties\": {_draws}, \"decided\": {F((double)(_winsA + _winsB) / n)},\n" +
                          $" \"median_time_s\": {F(Q(0.5))}, \"mean_time_s\": {F(mean)}, \"p25_time_s\": {F(Q(0.25))}, \"p75_time_s\": {F(Q(0.75))}\n}}\n";
            // The player has no training folder beside it: it writes next to the executable, like perf_unity.json.
            string path = System.IO.Path.GetFullPath(Application.isEditor ? System.IO.Path.Combine(Application.dataPath, "..", "training", "assets", "g1", "duel_stats_unity.json")
                                                                          : System.IO.Path.Combine(Application.dataPath, "..", "duel_stats_unity.json"));
            System.IO.File.WriteAllText(path, json); Debug.Log("[DemoDirector] wrote " + path + "\n" + json);
            statsRounds = 0; attacker.paused = defender.paused = true;
#if UNITY_EDITOR
            if (Flag("-kothExit")) UnityEditor.EditorApplication.Exit(0);
#else
            Application.Quit();
#endif
        }

        void Update()
        {
            _fps = Mathf.Lerp(_fps, 1f / Mathf.Max(Time.unscaledDeltaTime, 1e-4f), 0.05f);
            if (InMenu && _menuShotAt > 0 && Time.unscaledTime >= _menuShotAt)
            {
                _menuShotAt = -1; Shot("menu");
                if (_shotDir != null && Flag("-kothExit")) _launchAt = Time.unscaledTime + 1.0f;     // automated capture run: continue into a round
            }
            if (_launchAt > 0 && Time.unscaledTime >= _launchAt) { _launchAt = -1; Launch(); }
        }
        float _launchAt = -1;

        // ------------------------------------------------------------------ HUD and menu (IMGUI)
        static void Label(Rect r, string text, GUIStyle st)
        {
            var c = st.normal.textColor; st.normal.textColor = new Color(0f, 0f, 0f, 0.75f);
            GUI.Label(new Rect(r.x + 2, r.y + 2, r.width, r.height), text, st); st.normal.textColor = c; GUI.Label(r, text, st);
        }

        GUIStyle _st, _big, _right, _mid, _btn, _toggle, _launch; float _styleK;
        void OnGUI()
        {
            int w = Screen.width, h = Screen.height; float k = Mathf.Max(1f, h / 900f);
            if (_st == null || _styleK != k)             // styles are rebuilt only when the window height changes, not every OnGUI call
            {
                _styleK = k;
                _st = new GUIStyle(GUI.skin.label) { fontSize = Mathf.RoundToInt(15 * k), normal = { textColor = Color.white } };
                _big = new GUIStyle(_st) { fontSize = Mathf.RoundToInt(20 * k), fontStyle = FontStyle.Bold };
                _right = new GUIStyle(_st) { alignment = TextAnchor.UpperRight }; _mid = new GUIStyle(_st) { alignment = TextAnchor.UpperCenter };
                _btn = new GUIStyle(GUI.skin.button) { fontSize = Mathf.RoundToInt(14 * k) };
                _toggle = new GUIStyle(GUI.skin.toggle) { fontSize = _st.fontSize, normal = { textColor = Color.white } };
                _launch = new GUIStyle(_btn) { fontSize = Mathf.RoundToInt(18 * k), fontStyle = FontStyle.Bold };
            }
            GUIStyle st = _st, big = _big, right = _right, mid = _mid, btn = _btn;
            float pad = 12 * k, lh = 26 * k;
            Label(new Rect(pad, pad, w, lh), "PoKingHill", big);                                                          // top left: title
            double t = !InMenu && MjScene.InstanceExists && MjScene.Instance.Data != null ? MjScene.Instance.Data->time - _roundStart : 0;
            string seaText = sea != null && !InMenu ? $"   sea {sea.Level:0.0} m, arrives at {sea.Duration:0} s" : "";
            float brainMs = attacker != null ? (float)attacker.LastInferenceMs : 0f;
            Label(new Rect(0, pad + lh, w, lh * 2), $"{_fps:0} FPS   physics {_stepMs:0.00} ms   brain {brainMs:0.00} ms\n" + (InMenu ? "menu" : $"round {t:0.0} s{seaText}"), mid);   // top centre: telemetry
            Label(new Rect(pad, h - pad - lh * 2, w, lh * 2), $"A {_winsA}   B {_winsB}   Ties {_draws}\n{_last}", st);    // bottom left: score
            Label(new Rect(0, h - pad - lh, w - pad, lh), "v0.3", right);                                               // bottom right: version
            if (!InMenu)
            {
                Label(new Rect(0, pad + lh * 2.9f, w, lh), $"{NameOf(selectedA, "A")}  vs  {NameOf(selectedB, "B")}", mid);
                if (statsRounds == 0 && GUI.Button(new Rect(w - pad - 80 * k, pad, 80 * k, lh), "Menu", btn)) EnterMenu();              // top right: menu
                if (statsRounds == 0 && GUI.Button(new Rect(pad, h - pad - lh * 3.3f, 90 * k, lh), "Reset", btn)) NewRound();           // bottom left: reset
                return;
            }
            // ---- pre-match menu
            float mw = Mathf.Min(w - 2 * pad, 420 * k), x = (w - mw) / 2, y = h * 0.16f;
            GUI.Box(new Rect(x - pad, y - pad, mw + 2 * pad, lh * (7.5f + 2 * roster.Length) + 4 * pad), GUIContent.none);
            var names = new string[roster.Length]; for (int i = 0; i < names.Length; i++) names[i] = roster[i].name;
            Label(new Rect(x, y, mw, lh), "Fighter A", big); y += lh * 1.2f;
            GUI.enabled = !randomMatchup;
            selectedA = GUI.SelectionGrid(new Rect(x, y, mw, lh * roster.Length), selectedA, names, 1, btn); y += lh * roster.Length + pad;
            Label(new Rect(x, y, mw, lh), "Fighter B", big); y += lh * 1.2f;
            selectedB = GUI.SelectionGrid(new Rect(x, y, mw, lh * roster.Length), selectedB, names, 1, btn); y += lh * roster.Length + pad;
            GUI.enabled = true;
            randomMatchup = GUI.Toggle(new Rect(x, y, mw, lh), randomMatchup, " Random matchup", _toggle); y += lh * 1.2f;
            Label(new Rect(x, y, mw * 0.3f, lh), "Map", st);
            selectedMap = GUI.SelectionGrid(new Rect(x + mw * 0.3f, y, mw * 0.7f, lh), selectedMap, maps, 1, btn); y += lh * 1.6f;
            if (GUI.Button(new Rect(x, y, mw, lh * 1.5f), "Launch", _launch)) Launch();
        }
    }
}
