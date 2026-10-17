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
        [Tooltip("What fits on a badge, e.g. All-rounder")] public string shortName = "";
        [Tooltip("The kind of body this brain drives (g1, kim)")] public string body = "g1";
        public ModelAsset policy;
        [Tooltip("Optional critic exported by export_policy.py --value: drives the win-odds bar")] public ModelAsset value;
        [Tooltip("103 = locomotion inputs, 112 = with opponent/ring inputs")] public int obsDim = 112;
        public GoalMode goal = GoalMode.Opponent;
        public float stopDist = 0f, vmax = 1f;
        [Tooltip("Trained in the shove style (arm pose + action smoothing)")] public bool shove;
        [Tooltip("A walking brain that follows velocity commands: the player can steer it")] public bool steerable;
        [Tooltip("Ticked in the menu: this agent takes part in matches")] public bool inGame = true;
        [Tooltip("False = listed in the menu but cannot be ticked yet; note says why")] public bool available = true;
        public string note = "";
        public string Badge => shortName.Length > 0 ? shortName : name;
    }

    /// <summary>
    /// Match flow for king of the hill, as an explicit state machine: Menu -> Fight -> Result (-> Replay) -> Fight or
    /// Menu. Every seated agent gets one of the scene's fighter bodies of its own kind (unused bodies are parked in the
    /// sky) and all start on top of the hill. A fighter is out when it leaves the plateau (radius &gt; OutRadiusNow) or
    /// falls; the round ends when one is left (it wins), none is left, or the sea reaches the summit (the one nearest
    /// the centre wins). The brains were trained one against one: in a larger fight each one is pointed at one
    /// opponent (nearest, or the easiest to push off). Everything is done by writing mjData (no scene reload, no
    /// Instantiate/Destroy). Presentation listens to <see cref="MatchEvents"/>; the HUD is HudView.
    /// </summary>
    public unsafe class DemoDirector : MonoBehaviour
    {
        [Tooltip("Every fighter body in the scene (a_, b_, ...). A match seats agents on the first ones.")]
        public PolicyRunner[] fighters = Array.Empty<PolicyRunner>();
        public Sea sea;                                  // optional; without it the round lasts roundSeconds
        public ImpactSynth synth;                        // optional, for the audio counters in the verification file
        public Fighter[] roster = Array.Empty<Fighter>();
        public int selectedA, selectedB;
        public string[] maps = { "Summit (baseline)" };
        [Tooltip("Render-only scenery per map, same order as maps. Physics is the same dome for every map.")]
        public GameObject[] mapVisuals = Array.Empty<GameObject>();
        public int selectedMap;
        public Vector2 spawnRadiusA = new(0.5f, 1.2f), spawnRadiusB = new(0.5f, 1.2f);
        public float roundSeconds = 25f, outRadius = 1.7f;
        [Tooltip("How long the winner panel stays up before the next round or the menu")] public float pauseBetweenRounds = 5f;
        [Tooltip("Rounds per launch before returning to the menu. 0 = keep fighting. GameSettings overrides it at launch.")] public int roundsPerMatch = 1;
        [Tooltip("Skip the menu and start fighting immediately.")] public bool autoLaunch;
        [Tooltip("Statistical parity gate: play this many rounds fast, write duel_stats_unity.json, then stop. 0 = off.")]
        public int statsRounds = 0;

        /// <summary>Colour of each body's badge and team trim, in fighters[] order after seating.</summary>
        public static readonly Color[] SeatColors =
            { new(0.95f, 0.36f, 0.30f), new(0.27f, 0.62f, 0.98f), new(0.98f, 0.78f, 0.25f), new(0.45f, 0.85f, 0.45f), new(0.80f, 0.50f, 0.95f) };

        public static DemoDirector Instance { get; private set; }
        public MatchState State { get; private set; } = MatchState.Booting;
        public bool InMenu => State <= MatchState.Menu;
        /// <summary>The round is decided and the winner panel (or its replay) is up.</summary>
        public bool RoundOver => State == MatchState.Result || State == MatchState.Replay;
        /// <summary>Fighters of the current round that are still in (what the camera frames).</summary>
        public IReadOnlyList<PolicyRunner> Alive => _alive;
        /// <summary>Fighters knocked out of the current round, oldest first (the camera cuts to them at the water).</summary>
        public IReadOnlyList<PolicyRunner> Fallen => _fallen;
        public int Playing => _playing;
        public int AgentOf(int body) => body >= 0 && body < _seat.Length ? _seat[body] : -1;
        public int Wins(int agent) => agent >= 0 && agent < _score.Length ? _score[agent] : 0;
        /// <summary>Wins of one body this session; in a mirror match the badges show these, not the shared agent's.</summary>
        public int BodyWins(int body) => body >= 0 && body < _wins.Length ? _wins[body] : 0;
        public bool AgentSeatedTwice(int agent) { int n = 0; for (int i = 0; i < _playing; i++) if (_seat[i] == agent) n++; return n > 1; }
        public int Draws => _draws;
        public bool IsOut(int body) => body >= 0 && body < _out.Length && _out[body];
        public double RoundTime => State >= MatchState.Fight && MjScene.InstanceExists && MjScene.Instance.Data != null ? MjScene.Instance.Data->time - _roundStart : 0;
        public float ResultSecondsLeft => Mathf.Max(0f, _resultUntil - Time.unscaledTime);
        public RoundResult LastResult { get; private set; }
        public string Status => _last;
        public double StepMs => _stepMs;
        /// <summary>Plateau radius that counts as "still in" this round (a hazard may shrink it).</summary>
        public float OutRadiusNow { get; set; }
        /// <summary>The player steers this body with the stick (walker brain, velocity commands), -1 = nobody.</summary>
        public int PuppetBody { get; private set; } = -1;
        /// <summary>Ticked in the menu: the next launch seats the player on the first body.</summary>
        public bool PuppetRequested { get; set; }
        public bool LastRoundOfMatch => _bracket.Count == 0 && roundsPerMatch > 0 && _roundsThisMatch >= roundsPerMatch;
        public IReadOnlyList<BracketMatch> Bracket => _bracket;
        public bool InTournament => _bracket.Count > 0;
        public string TournamentChampion { get; private set; } = "";

        public class BracketMatch { public int a, b, winner = -1, tries; public string label; }
        readonly List<BracketMatch> _bracket = new(); int _bracketAt;

        readonly List<PolicyRunner> _alive = new(), _fallen = new();
        int[] _seat = Array.Empty<int>(), _wins = Array.Empty<int>(), _score = Array.Empty<int>();   // roster index per body (-1 = parked); wins per body; wins per agent
        bool[] _out = Array.Empty<bool>(); int _playing;
        PolicyRunner[] _target = Array.Empty<PolicyRunner>();
        readonly List<double> _times = new();
        int _draws, _roundsThisMatch; double _roundStart; float _resultUntil; string _last = ""; bool _pending, _knockout;
        readonly System.Diagnostics.Stopwatch _sw = new(); double _stepMs;
        string _shotDir; readonly HashSet<string> _shots = new(); float _menuShotAt = -1, _launchAt = -1;
        ArenaSpec _arena = new();

        void OnEnable()
        {
            Instance = this;
            Application.targetFrameRate = 60;            // the editor otherwise renders uncapped and starves GPU training
            Application.runInBackground = true;
            _shotDir = Cli.Arg("-kothShots");
            // Player-side statistical gate (-kothRounds N): same rules as ParityBatch.RunDuelStats, no sea and a 25 s bell.
            if (!Application.isEditor && int.TryParse(Cli.Arg("-kothRounds"), out int rounds) && rounds > 0)
            {
                statsRounds = rounds; roundSeconds = 25f;
                if (int.TryParse(Cli.Arg("-kothA"), out int fa)) selectedA = fa;
                if (int.TryParse(Cli.Arg("-kothB"), out int fb)) selectedB = fb;
                if (sea != null) { sea.enabled = false; sea = null; }
            }
            if (statsRounds > 0 || Cli.Flag("-kothAutoLaunch")) autoLaunch = true;
            // The unfocused editor renders about one frame per second, so in stats mode let each frame carry up to 2 s of physics.
            if (statsRounds > 0) { Time.timeScale = 50f; Time.maximumDeltaTime = 2f; pauseBetweenRounds = 0f; roundsPerMatch = 0; }
            OutRadiusNow = outRadius;
            MjScene.Instance.postInitEvent += OnInit;
            MjScene.Instance.preUpdateEvent += Pre; MjScene.Instance.postUpdateEvent += Post;
        }
        void OnDisable()
        {
            if (Instance == this) Instance = null;
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.postInitEvent -= OnInit; MjScene.Instance.preUpdateEvent -= Pre; MjScene.Instance.postUpdateEvent -= Post;
        }

        void SetState(MatchState s) { if (State == s) return; State = s; MatchEvents.RaiseState(s); }

        void OnInit(object s, MjStepArgs a) => _pending = true;   // act on the first physics step, once every runner has its map
        void Pre(object s, MjStepArgs a)
        {
            if (_pending && Ready())
            {
                _pending = false;
                _seat = new int[fighters.Length]; _wins = new int[fighters.Length]; _out = new bool[fighters.Length]; _target = new PolicyRunner[fighters.Length];
                _score = new int[Mathf.Max(1, roster.Length)];
                _arena = fighters[0].Spec?.arena ?? new ArenaSpec();
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

        // Each fighter is pointed at one opponent still in. With two fighters: always the other one. With more, either
        // the nearest (as trained) or, with Targeting.Opportunist, the one easiest to push off: near the rim or unsteady.
        void Retarget()
        {
            var d = MjScene.Instance.Data;
            bool opportunist = _playing > 2 && statsRounds == 0 && GameSettings.Targeting == Targeting.Opportunist;
            for (int i = 0; i < _playing; i++)
            {
                double* q = d->qpos + fighters[i].Map.RootQposAdr; double best = double.MaxValue, current = double.MaxValue; PolicyRunner pick = null, any = null;
                for (int j = 0; j < _playing; j++)
                {
                    if (j == i) continue;
                    double* o = d->qpos + fighters[j].Map.RootQposAdr;
                    double dist = Math.Sqrt((o[0] - q[0]) * (o[0] - q[0]) + (o[1] - q[1]) * (o[1] - q[1]));
                    any ??= fighters[j];
                    if (_out[j]) continue;
                    double score = dist;
                    if (opportunist)
                    {
                        double r = Math.Sqrt(o[0] * o[0] + o[1] * o[1]), rim = Math.Clamp((r - 0.8) / Math.Max(0.1, OutRadiusNow - 0.8), 0, 1);
                        score = dist - 1.2 * rim - 0.8 * (1 - ObsBuilder.Up(o + 3));
                    }
                    else score = dist * dist;          // nearest, compared exactly as before (squared distance)
                    if (fighters[j] == _target[i]) current = score;
                    if (score < best) { best = score; pick = fighters[j]; }
                }
                // Keep the current target unless another is clearly better (no flicker between two).
                if (opportunist && pick != null && _target[i] != null && current < double.MaxValue && best > current - 0.25) pick = _target[i];
                _target[i] = pick;
                fighters[i].opponent = pick ?? any;
            }
        }

        void Seat(int count)
        {
            _playing = count;
            for (int i = 0; i < fighters.Length; i++) { fighters[i].paused = i >= count; if (i >= count) _seat[i] = -1; _target[i] = null; }
        }

        // ------------------------------------------------------------------ flow
        public void EnterMenu()
        {
            _bracket.Clear(); PuppetBody = -1;
            _resultUntil = 0; _roundsThisMatch = 0;
            foreach (var f in fighters) f.SetBrain(null, ObsBuilder.ObsDim, GoalMode.None, 0.3f, 0.8f);      // passive stance while the menu is up
            Seat(2); _seat[0] = _seat[1] = 0;
            Place(); ClearOut();
            if (sea != null) sea.Park();
            _menuShotAt = Time.unscaledTime + 1.5f;
            SetState(MatchState.Menu);
        }

        /// <summary>Menu launch: every ticked agent takes a seat, in random order (one ticked = it fights itself; more
        /// ticked than bodies = that many are drawn). The statistics gate keeps the pair it was started with.</summary>
        public void Launch()
        {
            var agents = new List<int>();
            if (roster.Length > 0)
            {
                if (statsRounds > 0) { agents.Add(Mathf.Clamp(selectedA, 0, roster.Length - 1)); agents.Add(Mathf.Clamp(selectedB, 0, roster.Length - 1)); }
                else
                {
                    var pool = InGame(); if (pool.Count == 0) return;
                    if (pool.Count == 1) pool.Add(pool[0]);
                    while (pool.Count > 0) { int k = UnityEngine.Random.Range(0, pool.Count); agents.Add(pool[k]); pool.RemoveAt(k); }
                }
            }
            _bracket.Clear();
            StartMatch(agents, statsRounds == 0 && PuppetRequested);
        }

        /// <summary>One tap from the menu: the champion against itself on the first map.</summary>
        public void QuickMatch()
        {
            _bracket.Clear(); selectedMap = 0;
            StartMatch(PuppetRequested ? new List<int> { 0 } : new List<int> { 0, 0 }, PuppetRequested);     // with the player: you vs the champion
        }

        /// <summary>Single elimination between up to four available agents (the ticked ones, or all when fewer than two
        /// are ticked): semi-finals, then the final. A tie is replayed once; a second tie goes to the higher rating.</summary>
        public void StartTournament()
        {
            var pool = InGame(); if (pool.Count < 2) { pool.Clear(); for (int i = 0; i < roster.Length; i++) if (roster[i].available) pool.Add(i); }
            for (int i = pool.Count - 1; i > 0; i--) { int k = UnityEngine.Random.Range(0, i + 1); (pool[i], pool[k]) = (pool[k], pool[i]); }
            if (pool.Count > 4) pool.RemoveRange(4, pool.Count - 4);
            if (pool.Count < 2) { _last = "Need two agents with a brain for a tournament"; return; }
            _bracket.Clear(); TournamentChampion = "";
            if (pool.Count == 2) _bracket.Add(new BracketMatch { a = pool[0], b = pool[1], label = "Final" });
            else
            {
                _bracket.Add(new BracketMatch { a = pool[0], b = pool[1], label = "Semi-final 1" });
                if (pool.Count == 4) _bracket.Add(new BracketMatch { a = pool[2], b = pool[3], label = "Semi-final 2" });
                _bracket.Add(new BracketMatch { a = -1, b = pool.Count == 3 ? pool[2] : -1, label = "Final" });     // filled by the semi-final winners
            }
            _bracketAt = 0; PlayBracketMatch();
        }

        void PlayBracketMatch()
        {
            var m = _bracket[_bracketAt];
            StartMatch(new List<int> { m.a, m.b }, false);
            _last = m.label;
        }

        void StartMatch(List<int> agents, bool puppet)
        {
            PuppetBody = -1;
            if (roster.Length > 0)
            {
                int steer = Array.FindIndex(roster, f => f.steerable && f.available && f.policy != null);
                if (puppet && steer >= 0) agents.Insert(0, steer);
                if (agents.Count == 1) agents.Add(agents[0]);
                // Each agent takes a free body of its own kind; the bodies in use move to the front of fighters[].
                var seats = new List<int>();
                foreach (int agent in agents)
                {
                    int k = -1;
                    for (int i = seats.Count; i < fighters.Length && k < 0; i++) if (fighters[i].body == roster[agent].body) k = i;
                    if (k < 0) continue;
                    (fighters[seats.Count], fighters[k]) = (fighters[k], fighters[seats.Count]); seats.Add(agent);
                }
                if (seats.Count < 2) { _last = "That selection has only one body to fight with: tick another agent"; return; }
                selectedA = seats[0]; selectedB = seats[1];
                Seat(seats.Count);
                for (int i = 0; i < fighters.Length; i++)
                {
                    if (i >= _playing) { fighters[i].SetBrain(null, ObsBuilder.ObsDim, GoalMode.None, 0.3f, 0.8f); continue; }
                    var f = roster[seats[i]]; _seat[i] = seats[i];
                    bool steered = puppet && steer >= 0 && i == 0;
                    fighters[i].SetBrain(f.policy, f.obsDim, steered ? GoalMode.None : f.goal, f.stopDist, f.vmax, f.shove);
                }
                if (puppet && steer >= 0) PuppetBody = 0;
            }
            else Seat(2);
            if (statsRounds == 0 && !autoLaunch) roundsPerMatch = _bracket.Count > 0 ? 1 : GameSettings.RoundsPerMatch;   // PerfProbe/auto runs keep theirs
            _roundsThisMatch = 0;
            NewRound();
        }

        List<int> InGame()
        {
            var pool = new List<int>();
            for (int i = 0; i < roster.Length; i++) if (roster[i].available && roster[i].inGame) pool.Add(i);
            return pool;
        }
        public int TickedCount { get { int n = 0; foreach (var f in roster) if (f.available && f.inGame) n++; return n; } }

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

        /// <summary>Start the next round of the current match (also the HUD's Reset).</summary>
        public void NewRound()
        {
            Place(); ClearOut(); _knockout = false;
            _roundStart = MjScene.Instance.Data->time; OutRadiusNow = outRadius; _resultUntil = 0;
            if (sea != null) { sea.pace = statsRounds > 0 ? 1f : GameSettings.SeaPace; sea.Restart(); }
            SetState(MatchState.Fight);
            MatchEvents.RaiseRoundStarted();
        }

        bool Out(MujocoLib.mjData_* d, PolicyRunner r)
        {
            double* q = d->qpos + r.Map.RootQposAdr;
            double rad = Math.Sqrt(q[0] * q[0] + q[1] * q[1]);
            return rad > OutRadiusNow || ObsBuilder.Up(q + 3) < _arena.fall_up || q[2] - _arena.Ground(rad) < _arena.fall_height;
        }

        void Post(object s, MjStepArgs a)
        {
            _stepMs = 0.95 * _stepMs + 0.05 * _sw.Elapsed.TotalMilliseconds;
            if (_pending || !Ready() || State != MatchState.Fight) return;
            var d = MjScene.Instance.Data;
            double t = d->time - _roundStart;
            if (t > 1.2) Shot("combat");
            int alive = 0, last = -1;
            for (int i = 0; i < _playing; i++)
            {
                if (!_out[i] && Out(d, fighters[i]))
                {
                    _out[i] = true; _knockout = true; _alive.Remove(fighters[i]); _fallen.Add(fighters[i]);
                    if (_playing > 2) fighters[i].paused = true;        // in a free-for-all an eliminated fighter goes limp
                    MatchEvents.RaiseKnockout(fighters[i]);
                }
                if (!_out[i]) { alive++; last = i; }
            }
            bool bell = sea != null ? sea.ReachedSummit : t >= roundSeconds;
            if (alive > 1 && !bell) return;
            EndRound(d, alive, last, t);
        }

        void EndRound(MujocoLib.mjData_* d, int alive, int last, double t)
        {
            int winner = -1; string how; RoundEnd end;
            if (alive == 1)
            {
                winner = last; end = RoundEnd.Knockout; _times.Add(t);
                how = $"last one on the summit after {t:0.0} s";
                _last = $"{(char)('A' + last)} ({NameOf(_seat[last], "Robot")}) wins in {t:0.0} s";
            }
            else if (alive > 1 && sea != null)
            {
                // The sea has reached the top with several still standing: the hill belongs to whoever is nearest its
                // centre. (Without a sea, in the statistics gate, a full-time round stays a tie as in training.)
                double best = double.MaxValue;
                for (int i = 0; i < _playing; i++)
                {
                    if (_out[i]) continue;
                    double* q = d->qpos + fighters[i].Map.RootQposAdr; double r2 = q[0] * q[0] + q[1] * q[1];
                    if (r2 < best) { best = r2; winner = i; }
                }
                end = RoundEnd.SeaArrived;
                how = $"nearest the centre when the sea arrived ({Math.Sqrt(best):0.0} m, {alive} still standing)";
                _last = $"{(char)('A' + winner)} ({NameOf(_seat[winner], "Robot")}) holds the summit as the sea arrives";
            }
            else
            {
                end = alive > 1 ? RoundEnd.FullTime : RoundEnd.NobodyLeft;
                how = alive > 1 ? "full time" : "nobody was left on the summit";
                _last = alive > 1 ? "Tie: full time" : "Tie: nobody left on the summit";
            }
            if (winner >= 0) Award(winner); else _draws++;
            _roundsThisMatch++;
            var agents = new int[_playing]; for (int i = 0; i < _playing; i++) agents[i] = _seat[i];
            LastResult = new RoundResult(winner, winner >= 0 ? NameOf(_seat[winner], "Robot") : "No winner", how, end, (float)t, agents);
            if (statsRounds > 0)
            {
                if (_wins[0] + _wins[1] + _draws >= statsRounds) WriteStats(); else NewRound();
                return;
            }
            if (_bracket.Count > 0) RecordBracket(winner);
            _resultUntil = Time.unscaledTime + pauseBetweenRounds;
            MatchEvents.RaiseRoundEnded(LastResult);       // first: the ladder and telemetry settle before the results sheet reads them
            SetState(MatchState.Result);
        }

        void RecordBracket(int winnerBody)
        {
            var m = _bracket[_bracketAt];
            if (winnerBody < 0 && m.tries++ == 0) { _last = m.label + ": tie, replayed once"; return; }
            int w = winnerBody >= 0 ? _seat[winnerBody] : Ladder.Rating(roster[m.a].name) >= Ladder.Rating(roster[m.b].name) ? m.a : m.b;
            m.winner = w;
            if (m.label != "Final")
            {
                var final = _bracket[^1];
                if (final.a < 0) final.a = w; else final.b = w;
            }
            else TournamentChampion = roster[w].name;
        }

        /// <summary>Leave the winner panel now (its button).</summary>
        public void Continue() { if (State == MatchState.Result) _resultUntil = Time.unscaledTime; }

        /// <summary>ReplaySystem holds the result while it plays the knockout back.</summary>
        public void BeginReplay() { if (State == MatchState.Result) SetState(MatchState.Replay); }
        public void EndReplay() { if (State != MatchState.Replay) return; _resultUntil = Mathf.Max(_resultUntil, Time.unscaledTime + 2f); SetState(MatchState.Result); }

        void Advance()
        {
            if (ShotsDone()) return;
            if (_bracket.Count > 0)
            {
                var m = _bracket[_bracketAt];
                if (m.winner < 0) { PlayBracketMatch(); return; }                  // replay a tied match
                if (++_bracketAt < _bracket.Count) { PlayBracketMatch(); return; }
                _last = "Tournament champion: " + TournamentChampion;
                _bracket.Clear(); EnterMenu(); return;
            }
            if (LastRoundOfMatch && !(_shotDir != null && Cli.Exit)) EnterMenu(); else NewRound();
        }

        void Award(int body) { _wins[body]++; if (_seat[body] >= 0 && _seat[body] < _score.Length) _score[_seat[body]]++; }

        public string NameOf(int agent, string fallback) => roster.Length > 0 && agent >= 0 ? roster[Mathf.Clamp(agent, 0, roster.Length - 1)].name : fallback;
        public string BadgeOf(int agent) => roster.Length > 0 && agent >= 0 ? roster[Mathf.Clamp(agent, 0, roster.Length - 1)].Badge : "Robot";

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
            if (_shotDir == null || !Cli.Exit) return false;
            if (!_shots.Contains("menu") || !_shots.Contains("combat") || !_shots.Contains("ejection")) return false;
            if (Cli.Flag("-kothWaitSplash") && !_shots.Contains("splash")) return false;      // keep playing until the camera has cut to a faller at the water
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
            statsRounds = 0; fighters[0].paused = fighters[1].paused = true;      // seats A and B: the pair the two-fighter gate measures
#if UNITY_EDITOR
            if (Cli.Exit) UnityEditor.EditorApplication.Exit(0);
#else
            Application.Quit();
#endif
        }

        void Update()
        {
            for (int i = 0; i < mapVisuals.Length; i++) if (mapVisuals[i] != null && mapVisuals[i].activeSelf != (i == selectedMap)) mapVisuals[i].SetActive(i == selectedMap);
            if (State == MatchState.Menu && _menuShotAt > 0 && Time.unscaledTime >= _menuShotAt)
            {
                _menuShotAt = -1; Shot("menu");
                if (_shotDir != null && Cli.Exit) _launchAt = Time.unscaledTime + 1.0f;     // automated capture run: continue into a round
            }
            if (_launchAt > 0 && Time.unscaledTime >= _launchAt) { _launchAt = -1; Launch(); }
            if (State == MatchState.Result)
            {
                float shownFor = pauseBetweenRounds - (_resultUntil - Time.unscaledTime);
                if (shownFor >= 1f) { if (_knockout) Shot("ejection"); Shot("winner"); }
                if (Time.unscaledTime >= _resultUntil) Advance();
            }
        }
    }
}
