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
        [Tooltip("Ticked in the menu: this agent takes part in matches")] public bool inGame = true;
        [Tooltip("False = listed in the menu but cannot be ticked yet; note says why")] public bool available = true;
        public string note = "";
    }

    /// <summary>
    /// Match flow for king of the hill: pre-match menu (a tick per agent, map), launch, round, result, back to the
    /// menu. Every ticked agent gets one of the scene's fighter bodies (two to fighters.Length of them; unused bodies
    /// are parked in the sky) and all start on top of the hill. A fighter is out when it leaves the plateau
    /// (radius &gt; outRadius) or falls; the round ends when one is left (it wins), none is left, or the sea reaches
    /// the summit (tie). The brains were trained one against one: in a larger fight each one sees and chases the
    /// nearest fighter that is still in. Everything is done by writing
    /// mjData (no scene reload, no Instantiate/Destroy). HUD anchors: TL title, TC telemetry, TR menu / behaviour,
    /// BL score and reset, BR version.
    /// </summary>
    public unsafe class DemoDirector : MonoBehaviour
    {
        [Tooltip("Every fighter body in the scene (a_, b_, ...). A match seats agents on the first ones.")]
        public PolicyRunner[] fighters = Array.Empty<PolicyRunner>();
        public PolicyRunner attacker => fighters[0];     // seat A and seat B: the pair the two-fighter gates measure
        public PolicyRunner defender => fighters[1];
        public Sea sea;                                  // optional; without it the round lasts roundSeconds
        public ImpactSynth synth;                        // optional, for the audio counters in the verification file
        public Fighter[] roster = Array.Empty<Fighter>();
        public int selectedA, selectedB;
        public string[] maps = { "Summit (baseline)" };
        [Tooltip("Render-only scenery per map, same order as maps. Physics is the same dome for every map.")]
        public GameObject[] mapVisuals = Array.Empty<GameObject>();
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
        /// <summary>Fighters of the current round that are still in (what the camera frames).</summary>
        public IReadOnlyList<PolicyRunner> Alive => _alive;
        /// <summary>Fighters knocked out of the current round, oldest first (the camera cuts to them at the water).</summary>
        public IReadOnlyList<PolicyRunner> Fallen => _fallen;
        readonly List<PolicyRunner> _alive = new(), _fallen = new();
        int[] _seat = Array.Empty<int>(), _wins = Array.Empty<int>(), _score = Array.Empty<int>();   // roster index per body (-1 = parked); wins per seat; wins per agent
        bool[] _out = Array.Empty<bool>(); int _playing;

        readonly List<double> _times = new();
        int _draws, _roundsThisMatch; double _roundStart; float _pauseUntil; string _last = ""; bool _pending;
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
            if (_pending && Ready())
            {
                _pending = false;
                _seat = new int[fighters.Length]; _wins = new int[fighters.Length]; _out = new bool[fighters.Length]; _score = new int[Mathf.Max(1, roster.Length)];
                if (autoLaunch) Launch(); else EnterMenu();
            }
            if (!_pending && _seat.Length > 0)
            {
                for (int i = _playing; i < fighters.Length; i++) Park(i);
                Retarget();
            }
            _sw.Restart();
        }

        bool Ready()
        {
            if (fighters.Length < 2) return false;
            foreach (var f in fighters) if (f == null || f.Map == null) return false;
            return true;
        }

        // A body nobody is seated on waits far above the sea, re-pinned before every physics step.
        void Park(int i)
        {
            fighters[i].ResetPose(100f + 2f * i, 0f, 0f);
            MjScene.Instance.Data->qpos[fighters[i].Map.RootQposAdr + 2] = 50.0;
        }

        // Each fighter watches the nearest one that is still in (with two fighters: always the other one).
        void Retarget()
        {
            var d = MjScene.Instance.Data;
            for (int i = 0; i < _playing; i++)
            {
                double* q = d->qpos + fighters[i].Map.RootQposAdr; double best = double.MaxValue; PolicyRunner pick = null, any = null;
                for (int j = 0; j < _playing; j++)
                {
                    if (j == i) continue;
                    double* o = d->qpos + fighters[j].Map.RootQposAdr; double dist = (o[0] - q[0]) * (o[0] - q[0]) + (o[1] - q[1]) * (o[1] - q[1]);
                    any ??= fighters[j];
                    if (!_out[j] && dist < best) { best = dist; pick = fighters[j]; }
                }
                fighters[i].opponent = pick ?? any;
            }
        }

        void Seat(int count)
        {
            _playing = count;
            for (int i = 0; i < fighters.Length; i++) { fighters[i].paused = i >= count; if (i >= count) _seat[i] = -1; }
        }

        // ------------------------------------------------------------------ flow
        void EnterMenu()
        {
            InMenu = true; FollowTarget = null; _pauseUntil = 0; _roundsThisMatch = 0;
            foreach (var f in fighters) f.SetBrain(null, ObsBuilder.ObsDim, GoalMode.None, 0.3f, 0.8f);      // passive stance while the menu is up
            Seat(2); _seat[0] = _seat[1] = 0;
            Place(); ClearOut();
            if (sea != null) sea.Park();
            _menuShotAt = Time.unscaledTime + 1.5f;
        }

        public void Launch()
        {
            if (roster.Length > 0)
            {
                // Menu launch: every ticked agent takes a seat, in random order (one ticked = it fights itself; more
                // ticked than bodies = that many are drawn). The statistics gate keeps the pair it was started with.
                var seats = new List<int>();
                if (statsRounds > 0) { seats.Add(Mathf.Clamp(selectedA, 0, roster.Length - 1)); seats.Add(Mathf.Clamp(selectedB, 0, roster.Length - 1)); }
                else
                {
                    var pool = InGame(); if (pool.Count == 0) return;
                    if (pool.Count == 1) pool.Add(pool[0]);
                    while (pool.Count > 0 && seats.Count < fighters.Length) { int k = UnityEngine.Random.Range(0, pool.Count); seats.Add(pool[k]); pool.RemoveAt(k); }
                    selectedA = seats[0]; selectedB = seats[1];
                }
                Seat(seats.Count);
                for (int i = 0; i < fighters.Length; i++)
                {
                    if (i >= _playing) { fighters[i].SetBrain(null, ObsBuilder.ObsDim, GoalMode.None, 0.3f, 0.8f); continue; }
                    var f = roster[seats[i]]; _seat[i] = seats[i];
                    fighters[i].SetBrain(f.policy, f.obsDim, f.goal, f.stopDist, f.vmax, f.shove);
                }
            }
            else Seat(2);
            InMenu = false; _roundsThisMatch = 0;
            NewRound();
        }

        List<int> InGame()
        {
            var pool = new List<int>();
            for (int i = 0; i < roster.Length; i++) if (roster[i].available && roster[i].inGame) pool.Add(i);
            return pool;
        }

        void Place()
        {
            // Everyone starts on the summit, evenly spread around it (two fighters: opposite sides). Yaw is random,
            // as in training and duel_stats.py.
            float bearing = UnityEngine.Random.Range(0f, 2f * Mathf.PI);
            for (int i = 0; i < _playing; i++)
            {
                var band = i == 1 ? spawnRadiusB : spawnRadiusA; float r = UnityEngine.Random.Range(band.x, band.y), a = bearing + 2f * Mathf.PI * i / _playing;
                fighters[i].ResetPose(r * Mathf.Cos(a), r * Mathf.Sin(a), UnityEngine.Random.Range(-Mathf.PI, Mathf.PI));
            }
        }

        void ClearOut()
        {
            _alive.Clear(); _fallen.Clear();
            for (int i = 0; i < _playing; i++) { _out[i] = false; fighters[i].paused = false; _alive.Add(fighters[i]); }
        }

        void NewRound()
        {
            Place(); ClearOut();
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
            if (_pending || InMenu || !Ready()) return;
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
            double t = d->time - _roundStart;
            if (t > 1.2) Shot("combat");
            PolicyRunner justOut = null; int alive = 0, last = -1;
            for (int i = 0; i < _playing; i++)
            {
                if (!_out[i] && Out(d, fighters[i], outRadius))
                {
                    _out[i] = true; justOut = fighters[i]; _alive.Remove(fighters[i]); _fallen.Add(fighters[i]);
                    if (_playing > 2) fighters[i].paused = true;        // in a free-for-all an eliminated fighter goes limp
                }
                if (!_out[i]) { alive++; last = i; }
            }
            bool bell = sea != null ? sea.ReachedSummit : t >= roundSeconds;
            if (alive > 1 && !bell) return;
            if (alive == 1)
            {
                _wins[last]++; if (_seat[last] >= 0 && _seat[last] < _score.Length) _score[_seat[last]]++;
                _times.Add(t); _last = $"{(char)('A' + last)} ({NameOf(_seat[last], "Robot")}) wins in {t:0.0} s"; FollowTarget = justOut;
            }
            else { _draws++; _last = alive > 1 ? "Tie: the sea took the summit" : "Tie: nobody left on the summit"; }
            _roundsThisMatch++;
            _pauseUntil = Time.unscaledTime + pauseBetweenRounds;
            if (statsRounds > 0 && _wins[0] + _wins[1] + _draws >= statsRounds) WriteStats();
            else if (statsRounds > 0) { _pauseUntil = 0; NewRound(); }
        }

        string NameOf(int i, string fallback) => roster.Length > 0 ? roster[Mathf.Clamp(i, 0, roster.Length - 1)].name : fallback;
        // "G1 All-rounder (champion)" -> "All-rounder": what fits on a line when four fight.
        string ShortName(int i)
        {
            string n = NameOf(i, "Robot"); int p = n.IndexOf(" ("); if (p > 0) n = n.Substring(0, p);
            return n.StartsWith("G1 ") ? n.Substring(3) : n;
        }

        // ------------------------------------------------------------------ verification helpers
        public void Shot(string name)
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
            if (Flag("-kothWaitSplash") && !_shots.Contains("splash")) return false;      // keep playing until the camera has cut to a faller at the water
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
            _times.Sort(); int _winsA = _wins[0], _winsB = _wins[1], n = _winsA + _winsB + _draws;
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
            for (int i = 0; i < mapVisuals.Length; i++) if (mapVisuals[i] != null && mapVisuals[i].activeSelf != (i == selectedMap)) mapVisuals[i].SetActive(i == selectedMap);
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
            float brainMs = fighters.Length > 0 && fighters[0] != null ? (float)fighters[0].LastInferenceMs : 0f;
            Label(new Rect(0, pad + lh, w, lh * 2), $"{_fps:0} FPS   physics {_stepMs:0.00} ms   brain {brainMs:0.00} ms\n" + (InMenu ? "menu" : $"round {t:0.0} s{seaText}"), mid);   // top centre: telemetry
            var score = new System.Text.StringBuilder();                                                                  // bottom left: wins per agent
            for (int i = 0; i < roster.Length && i < _score.Length; i++) if (roster[i].available && (roster[i].inGame || _score[i] > 0)) score.Append($"{ShortName(i)} {_score[i]}   ");
            Label(new Rect(pad, h - pad - lh * 2, w, lh * 2), $"{score}Ties {_draws}\n{_last}", st);
            Label(new Rect(0, h - pad - lh, w - pad, lh), "v0.3", right);                                               // bottom right: version
            if (!InMenu)
            {
                string title = _playing == 2 ? $"{NameOf(_seat[0], "A")}  vs  {NameOf(_seat[1], "B")}" : "";
                for (int i = 0; _playing > 2 && i < _playing; i++) title += (i > 0 ? "  vs  " : "") + ShortName(_seat[i]);
                Label(new Rect(0, pad + lh * 2.9f, w, lh), title, mid);
                if (statsRounds == 0 && GUI.Button(new Rect(w - pad - 80 * k, pad, 80 * k, lh), "Menu", btn)) EnterMenu();              // top right: menu
                if (statsRounds == 0 && GUI.Button(new Rect(pad, h - pad - lh * 3.3f, 90 * k, lh), "Reset", btn)) NewRound();           // bottom left: reset
                return;
            }
            // ---- pre-match menu
            float mw = Mathf.Min(w - 2 * pad, 420 * k), x = (w - mw) / 2, y = h * 0.16f;
            GUI.Box(new Rect(x - pad, y - pad, mw + 2 * pad, lh * (6.2f + 1.1f * roster.Length) + 4 * pad), GUIContent.none);
            Label(new Rect(x, y, mw, lh), "Agents in the game", big); y += lh * 1.2f;
            foreach (var f in roster)                 // one tick box per agent; an agent without a brain is listed but locked
            {
                GUI.enabled = f.available;
                bool on = f.available && f.inGame;      // the stock tick box is a few pixels wide at this scale, so the mark is drawn as text
                f.inGame = GUI.Toggle(new Rect(x, y, mw, lh), on, (on ? "[x]  " : "[  ]  ") + f.name + (f.note.Length > 0 ? "  (" + f.note + ")" : ""), st);
                y += lh * 1.1f;
            }
            GUI.enabled = true; int ticked = InGame().Count;
            Label(new Rect(x, y, mw, lh), ticked == 0 ? "Tick at least one agent" : ticked == 1 ? "One agent: it fights itself" : ticked == 2 ? "These two fight" : ticked <= fighters.Length ? $"All {ticked} fight at once: the last one on the summit wins" : $"{fighters.Length} of them are drawn for each match", st); y += lh * 1.3f;
            Label(new Rect(x, y, mw * 0.3f, lh), "Map", st);
            selectedMap = GUI.SelectionGrid(new Rect(x + mw * 0.3f, y, mw * 0.7f, lh), selectedMap, maps, 1, btn); y += lh * 1.6f;
            GUI.enabled = ticked > 0;
            if (GUI.Button(new Rect(x, y, mw, lh * 1.5f), "Launch", _launch)) Launch();
            GUI.enabled = true;
        }
    }
}
