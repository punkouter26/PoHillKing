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
        [Tooltip("The kind of body this brain drives (g1, kim, trump, grandpa)")] public string body = "g1";
        public ModelAsset policy;
        [Tooltip("Optional critic exported by export_policy.py --value: drives the win-odds bar")] public ModelAsset value;
        [Tooltip("103 = locomotion inputs, 112 = with opponent/ring inputs")] public int obsDim = 112;
        public GoalMode goal = GoalMode.Opponent;
        public float stopDist = 0f, vmax = 1f;
        [Tooltip("Trained in a styled action path (rest pose + action smoothing)")] public bool shove;
        [Tooltip("Which styled rest pose with shove: shove (arms forward) or bull (Trump's charge)")] public string stylePose = "shove";
        [Tooltip("A walking brain that follows velocity commands: the player can steer it")] public bool steerable;
        [Tooltip("Ticked in the menu: this agent takes part in matches")] public bool inGame = true;
        [Tooltip("False = listed in the menu but cannot be ticked yet; note says why")] public bool available = true;
        public string note = "";
        [Tooltip("Optional weights of the policy for Tactician rollouts (scripts/export_mlp.py, <file>_mlp.bytes)")] public TextAsset mlp;
        [Tooltip("Optional action path of the policy (<file>_action.json, training/koth/action_map.py); empty = the original clamp path")] public TextAsset action;
        [Tooltip("Optional critic weights (<file>_value_mlp.bytes) that score Tactician rollouts")] public TextAsset valueMlp;
        public string Badge => shortName.Length > 0 ? shortName : name;
    }

    /// <summary>A brain that belongs to a kind of body rather than to an agent (the get-up brain).</summary>
    [Serializable]
    public class BodyBrain { public string body = "g1"; public ModelAsset policy; }

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
        const float MinSpawnGap = 0.9f;     // metres between neighbouring pelvises at the start of a round with more than two
        public float roundSeconds = 25f, outRadius = 1.7f;
        [Tooltip("How long the winner panel stays up before the next round or the menu")] public float pauseBetweenRounds = 5f;
        [Tooltip("Rounds per launch before returning to the menu. 0 = keep fighting. GameSettings overrides it at launch.")] public int roundsPerMatch = 1;
        [Tooltip("Skip the menu and start fighting immediately.")] public bool autoLaunch;
        [Tooltip("Statistical parity gate: play this many rounds fast, write duel_stats_unity.json, then stop. 0 = off.")]
        public int statsRounds = 0;
        [Tooltip("File name the statistics gate writes (empty = duel_stats_unity.json / ffa_stats_unity.json, the parity references); diagnostics runs set their own")]
        public string statsFile = "";
        [Tooltip("Free-for-all statistics gate: the candidate (roster index) on seat A against three opponents drawn from this pool every round (roster indices). Empty = the two-fighter gate.")]
        public int[] statsPool = Array.Empty<int>();
        public int statsCandidate = -1;
        [Tooltip("Get-up brain per body kind (training rung getup): a fighter that falls on the plateau gets getupSeconds to stand again (GameSettings.GetUp)")]
        public BodyBrain[] getupBrains = Array.Empty<BodyBrain>();
        public float getupSeconds = 5f; public int maxGetups = 2;
        [Tooltip("The statistics gate plays with the get-up rule (duel_stats.py --getup)")] public bool getupInStats;
        [Tooltip("Tale of the tape: crane shot length, then this many seconds per fighter (clamped to fit about 7 s)")] public float introCrane = 1.6f, introPerFighter = 1.6f;
        public ContactLedger ledger;                     // optional: Grudge / Vulture targeting
        public Biometrics bio;                           // optional: Vulture targeting

        /// <summary>Colour of each body's badge and team trim, in fighters[] order after seating.</summary>
        public static readonly Color[] SeatColors =
            { new(0.95f, 0.36f, 0.30f), new(0.27f, 0.62f, 0.98f), new(0.98f, 0.78f, 0.25f), new(0.45f, 0.85f, 0.45f), new(0.80f, 0.50f, 0.95f),
              new(0.30f, 0.85f, 0.85f), new(0.98f, 0.52f, 0.72f) };

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
        /// <summary>Fallen on the plateau, the get-up brain driving and the count running (still in the round).</summary>
        public bool IsDown(int body) => body >= 0 && body < _playing && body < _downAt.Length && _downAt[body] >= 0 && !_out[body];
        /// <summary>Seconds left on a down fighter's count.</summary>
        public float DownLeft(int body) => IsDown(body) ? Mathf.Max(0f, (float)(_downAt[body] + getupSeconds - MjScene.Instance.Data->time)) : 0f;
        /// <summary>During the intro: the seat being presented (-1 during the crane shot), and how far into its card (0..1).</summary>
        public int IntroFeatured => State == MatchState.Intro ? (int)Mathf.Floor((Time.unscaledTime - _introStart - introCrane) / Mathf.Max(0.01f, _introPer)) : -1;
        public float IntroPhase => State == MatchState.Intro ? Mathf.Repeat((Time.unscaledTime - _introStart - introCrane) / Mathf.Max(0.01f, _introPer), 1f) : 0f;
        public float IntroElapsed => State == MatchState.Intro ? Time.unscaledTime - _introStart : 0f;
        public float IntroLength => _introLength;
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
        double[] _downAt = Array.Empty<double>(), _stoodAt = Array.Empty<double>(), _lurkAt = Array.Empty<double>(); int[] _getups = Array.Empty<int>(); bool[] _steered = Array.Empty<bool>(), _lurking = Array.Empty<bool>();
        float _introStart, _introPer, _introLength;
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
                _downAt = new double[fighters.Length]; _stoodAt = new double[fighters.Length]; _lurkAt = new double[fighters.Length]; _getups = new int[fighters.Length];
                _steered = new bool[fighters.Length]; _lurking = new bool[fighters.Length];
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
        // Free-for-all modes beyond Opportunist (GameSettings.Targeting): Dogpile = everyone at the king (nearest the
        // centre; the king itself takes the nearest), Grudge = whoever pushed me in the last 6 s (ContactLedger), Vulture =
        // hold a spot out of the fight (Tactic.Lurk) while two others push each other, then go for the weakest
        // (Biometrics stability, or the one nearest the rim). Brains still see one opponent, as trained.
        void Retarget()
        {
            var d = MjScene.Instance.Data;
            var mode = _playing > 2 && statsRounds == 0 ? GameSettings.Targeting : Targeting.Nearest;
            if (mode > Targeting.Opportunist && State == MatchState.Fight) { RetargetMode(d, mode); return; }
            if (mode != Targeting.Vulture) for (int i = 0; i < _playing; i++) if (_lurking[i]) { _lurking[i] = false; fighters[i].tactic = Tactic.Press; }
            bool opportunist = mode == Targeting.Opportunist;
            for (int i = 0; i < _playing; i++)
            {
                double* q = d->qpos + fighters[i].Map.RootQposAdr; double best = double.MaxValue, current = double.MaxValue; PolicyRunner pick = null, any = null;
                // nearest and second-nearest still in, as training ranks them (koth/env.py _refresh_nbr): an eliminated
                // fighter only when nobody else is left; with one opponent left the second is the first
                double d1 = double.MaxValue, d2 = double.MaxValue, dAny = double.MaxValue; PolicyRunner n1 = null, n2 = null;
                for (int j = 0; j < _playing; j++)
                {
                    if (j == i) continue;
                    double* o = d->qpos + fighters[j].Map.RootQposAdr;
                    double dist = Math.Sqrt((o[0] - q[0]) * (o[0] - q[0]) + (o[1] - q[1]) * (o[1] - q[1]));
                    if (dist < dAny) { dAny = dist; any = fighters[j]; }
                    if (_out[j]) continue;
                    if (dist < d1) { d2 = d1; n2 = n1; d1 = dist; n1 = fighters[j]; } else if (dist < d2) { d2 = dist; n2 = fighters[j]; }
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
                n1 ??= any; n2 ??= n1;
                // A free-for-all brain (121 inputs) gets what it was trained on: nearest and second-nearest. Two-fighter
                // brains get the chosen target (nearest, or the opportunist pick).
                bool ffaBrain = fighters[i].policyObsDim >= ObsBuilder.ObsDim + 18;
                fighters[i].opponent = ffaBrain ? n1 : pick ?? any;
                fighters[i].opponent2 = n2;
            }
        }

        void RetargetMode(MujocoLib.mjData_* d, Targeting mode)
        {
            double R(int j) { double* o = d->qpos + fighters[j].Map.RootQposAdr; return Math.Sqrt(o[0] * o[0] + o[1] * o[1]); }
            double Dist(int i, int j) { double* a = d->qpos + fighters[i].Map.RootQposAdr; double* b = d->qpos + fighters[j].Map.RootQposAdr; return Math.Sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])); }
            int king = -1; for (int j = 0; j < _playing; j++) if (!_out[j] && (king < 0 || R(j) < R(king))) king = j;
            for (int i = 0; i < _playing; i++)
            {
                int nearest = -1, any = -1, second = -1;
                for (int j = 0; j < _playing; j++)
                {
                    if (j == i) continue; if (any < 0 || Dist(i, j) < Dist(i, any)) any = j;
                    if (_out[j]) continue;
                    if (nearest < 0 || Dist(i, j) < Dist(i, nearest)) { second = nearest; nearest = j; } else if (second < 0 || Dist(i, j) < Dist(i, second)) second = j;
                }
                if (nearest < 0) nearest = any; if (second < 0) second = nearest;
                int pick = nearest; bool lurk = false;
                if (mode == Targeting.Dogpile)
                {
                    pick = king >= 0 && king != i ? king : nearest;
                    int cur = Array.IndexOf(fighters, _target[i]);           // keep the current king unless the new one is clearly more central
                    if (cur >= 0 && cur != i && cur < _playing && !_out[cur] && pick != cur && king != i && R(pick) > R(cur) - 0.15) pick = cur;
                }
                else if (mode == Targeting.Grudge && ledger != null)
                {
                    var by = ledger.LastPusher(fighters[i], 6.0); int j = Array.IndexOf(fighters, by);
                    if (j >= 0 && j < _playing && !_out[j]) pick = j;
                }
                else if (mode == Targeting.Vulture)
                {
                    int weakest = -1, alive = 0; float weak = float.MaxValue;
                    for (int j = 0; j < _playing; j++)
                    {
                        if (_out[j]) continue; alive++; if (j == i) continue;
                        var v = bio != null ? bio.Of(fighters[j]) : null; float s = v != null ? Mathf.Min(v.stability, v.rim / 0.6f) : 1f;
                        if (s < weak) { weak = s; weakest = j; }
                    }
                    bool engaged = ledger != null && ledger.Engaged(fighters[i], 0.8);
                    if (weakest >= 0 && weak < 0.45f) { pick = weakest; _lurkAt[i] = -1; }  // someone is about to go: pounce
                    else if (_lurking[i] && d->time - _lurkAt[i] < 1.5) { lurk = true; pick = Array.IndexOf(fighters, _target[i]); if (pick < 0 || pick >= _playing || _out[pick]) pick = nearest; }   // hold the lurk a moment: no flicker
                    else if (!engaged && alive >= 3 && ledger != null)
                        for (int a = 0; a < _playing && !lurk; a++)
                            for (int b = a + 1; b < _playing && !lurk; b++)
                                if (a != i && b != i && !_out[a] && !_out[b] && ledger.Fighting(fighters[a], fighters[b], 1.0))
                                { lurk = true; var va = bio?.Of(fighters[a]); var vb = bio?.Of(fighters[b]); pick = va != null && vb != null && vb.stability < va.stability ? b : a; }
                }
                if (lurk != _lurking[i]) { _lurking[i] = lurk; fighters[i].tactic = lurk ? Tactic.Lurk : Tactic.Press; if (lurk) _lurkAt[i] = d->time; }
                _target[i] = fighters[pick];
                fighters[i].opponent = fighters[pick];
                int other = second != pick ? second : nearest != pick ? nearest : pick;
                fighters[i].opponent2 = fighters[other];
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
            ReplaySystem.StopActive();          // a replay restores its saved state when it ends: end it before placing anyone
            _bracket.Clear(); PuppetBody = -1;
            _resultUntil = 0; _roundsThisMatch = 0;
            foreach (var f in fighters) f.SetBrain(null, ObsBuilder.ObsDim, GoalMode.None, 0.3f, 0.8f);      // passive stance while the menu is up
            Seat(2);
            for (int i = 0; i < 2 && i < fighters.Length; i++)
            {
                int a = Array.FindIndex(roster, r => r.body == fighters[i].body && !r.steerable);
                _seat[i] = a >= 0 ? a : 0;
            }
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
                if (statsRounds > 0 && FfaStats) agents = FfaDraw();
                else if (statsRounds > 0) { agents.Add(Mathf.Clamp(selectedA, 0, roster.Length - 1)); agents.Add(Mathf.Clamp(selectedB, 0, roster.Length - 1)); }
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
            ReplaySystem.StopActive();
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
                for (int i = 0; i < _playing; i++) { _seat[i] = seats[i]; _steered[i] = puppet && steer >= 0 && i == 0; }
                if (puppet && steer >= 0) PuppetBody = 0;
            }
            else Seat(2);
            if (statsRounds == 0 && !autoLaunch) roundsPerMatch = _bracket.Count > 0 ? 1 : GameSettings.RoundsPerMatch;   // PerfProbe/auto runs keep theirs
            _roundsThisMatch = 0;
            if (statsRounds == 0 && !autoLaunch && GameSettings.Intro && roster.Length > 0) { BeginIntro(); return; }
            ArmBrains();
            NewRound();
        }

        // Each seated body gets its agent's brain (and its body's get-up brain); the rest stand passively.
        void ArmBrains()
        {
            if (roster.Length == 0) return;
            for (int i = 0; i < fighters.Length; i++)
            {
                if (i >= _playing) { fighters[i].SetBrain(null, ObsBuilder.ObsDim, GoalMode.None, 0.3f, 0.8f); continue; }
                var f = roster[_seat[i]];
                fighters[i].SetBrain(f.policy, f.obsDim, _steered[i] ? GoalMode.None : f.goal, f.stopDist, f.vmax, f.shove, f.stylePose, f.action);
                fighters[i].getupPolicy = Array.Find(getupBrains, b => b != null && b.body == fighters[i].body)?.policy;
            }
        }

        // ------------------------------------------------------------------ intro (tale of the tape)
        // The seated fighters stand passively on their spawn spots (no brain, hold ctrl, as in the menu) while the camera
        // cranes up to the summit and then visits each one; HudView shows its card. Then the brains are armed and the
        // first round starts from a fresh placement, exactly as without the intro.
        void BeginIntro()
        {
            for (int i = 0; i < fighters.Length; i++) fighters[i].SetBrain(null, ObsBuilder.ObsDim, GoalMode.None, 0.3f, 0.8f);
            Place(); ClearOut();
            if (sea != null) sea.Park();
            _introPer = Mathf.Clamp(6f / Mathf.Max(1, _playing), 0.9f, introPerFighter);
            _introStart = Time.unscaledTime; _introLength = introCrane + _playing * _introPer;
            SetState(MatchState.Intro);
        }

        /// <summary>End the tale of the tape now (HUD button) and start the first round.</summary>
        public void SkipIntro() { if (State == MatchState.Intro) EndIntro(); }
        void EndIntro() { ArmBrains(); NewRound(); }

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
            // as in training and duel_stats.py. With more than two, neighbours start at least MinSpawnGap apart: six
            // on the 0.5 m ring are 0.5 m apart, less than Trump is wide, and overlapping bodies burst off the summit.
            float bearing = UnityEngine.Random.Range(0f, 2f * Mathf.PI);
            float rMin = _playing > 2 ? MinSpawnGap / (2f * Mathf.Sin(Mathf.PI / _playing)) : 0f;
            for (int i = 0; i < _playing; i++)
            {
                var band = i == 1 ? spawnRadiusB : spawnRadiusA; float lo = Mathf.Min(Mathf.Max(band.x, rMin), band.y);
                float r = UnityEngine.Random.Range(lo, band.y), a = bearing + 2f * Mathf.PI * i / _playing;
                fighters[i].ResetPose(r * Mathf.Cos(a), r * Mathf.Sin(a), UnityEngine.Random.Range(-Mathf.PI, Mathf.PI));
            }
        }

        void ClearOut()
        {
            _alive.Clear(); _fallen.Clear();
            for (int i = 0; i < fighters.Length; i++)
            {
                bool inPlay = i < _playing;
                _out[i] = !inPlay; fighters[i].paused = !inPlay; if (inPlay) _alive.Add(fighters[i]);
                _downAt[i] = -1; _stoodAt[i] = -1; _getups[i] = 0; _lurking[i] = false; fighters[i].tactic = Tactic.Press;
            }
        }

        /// <summary>Start the next round of the current match (also the HUD's Reset).</summary>
        public void NewRound()
        {
            ReplaySystem.StopActive();
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

        /// <summary>The get-up rule is on: in play with GameSettings.GetUp, or in a gate started with getupInStats.</summary>
        public bool GetUpRule => statsRounds > 0 ? getupInStats : GameSettings.GetUp;

        // Get-up rule: a ring-out is still out at once. A fall inside the plateau (radius < plateau_r) hands the body to
        // its get-up brain and starts a count of getupSeconds; standing again (up > 0.9, pelvis above 85 % of the standing
        // height, for 0.4 s) gives it back to the fight brain, the count running out is a knock-out. At most maxGetups
        // per fighter per round; a fighter with no get-up brain is out on a fall as before. Mirrored by
        // training/scripts/duel_stats.py --getup.
        bool OutWithGetUp(MujocoLib.mjData_* d, int i, double t)
        {
            var r = fighters[i]; double* q = d->qpos + r.Map.RootQposAdr;
            double rad = Math.Sqrt(q[0] * q[0] + q[1] * q[1]), h = q[2] - _arena.Ground(rad), up = ObsBuilder.Up(q + 3);
            if (rad > OutRadiusNow) return true;
            if (_downAt[i] >= 0)
            {
                if (up > 0.9 && h > 0.85 * (r.Spec?.key_root_z ?? 0.75f))
                {
                    if (_stoodAt[i] < 0) _stoodAt[i] = t;
                    if (t - _stoodAt[i] >= 0.4) { _downAt[i] = -1; _stoodAt[i] = -1; r.SetGetUp(false); MatchEvents.RaiseRecovered(r); }
                }
                else _stoodAt[i] = -1;
                return _downAt[i] >= 0 && t - _downAt[i] > getupSeconds;
            }
            bool fallen = up < _arena.fall_up || h < _arena.fall_height;
            if (!fallen) return false;
            if (rad < _arena.plateau_r && _getups[i] < maxGetups && r.SetGetUp(true))
            {
                _downAt[i] = t; _stoodAt[i] = -1; _getups[i]++; MatchEvents.RaiseDown(r);
                return false;
            }
            return true;
        }

        void Post(object s, MjStepArgs a)
        {
            _stepMs = 0.95 * _stepMs + 0.05 * _sw.Elapsed.TotalMilliseconds;
            if (_pending || !Ready() || State != MatchState.Fight) return;
            var d = MjScene.Instance.Data;
            double t = d->time - _roundStart;
            if (t > 1.2) Shot("combat");
            int alive = 0, last = -1; bool getup = GetUpRule;
            for (int i = 0; i < _playing; i++)
            {
                if (!_out[i] && (getup ? OutWithGetUp(d, i, t) : Out(d, fighters[i])))
                {
                    if (_downAt[i] >= 0) { _downAt[i] = -1; fighters[i].SetGetUp(false); }
                    _out[i] = true; _knockout = true; _alive.Remove(fighters[i]); _fallen.Add(fighters[i]);
                    if (_playing > 2) fighters[i].paused = true;        // in a free-for-all an eliminated fighter goes limp
                    MatchEvents.RaiseKnockout(fighters[i]);
                }
                if (!_out[i]) { alive++; last = i; }
            }
            bool bell = sea != null ? sea.ReachedSummit : t >= roundSeconds;
            if (statsRounds > 0 && FfaStats && (_out[0] || alive == 1 || bell)) { FfaRoundOver(t); return; }    // as eval.py: the candidate's round ends when it is out
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
                _last = $"{(char)('A' + last)} ({BadgeOf(_seat[last])}) wins in {t:0.0} s";
            }
            else if (alive > 1 && sea != null)
            {
                // The sea has reached the top with several still standing: the hill belongs to whoever is nearest its
                // centre. (Without a sea, in the statistics gate, a full-time round stays a tie as in training.)
                double best = double.MaxValue;
                for (int pass = 0; pass < 2 && winner < 0; pass++)          // a fighter that is down only wins if everyone left is down
                    for (int i = 0; i < _playing; i++)
                    {
                        if (_out[i] || (pass == 0 && IsDown(i))) continue;
                        double* q = d->qpos + fighters[i].Map.RootQposAdr; double r2 = q[0] * q[0] + q[1] * q[1];
                        if (r2 < best) { best = r2; winner = i; }
                    }
                end = RoundEnd.SeaArrived;
                how = $"nearest the centre when the sea arrived ({Math.Sqrt(best):0.0} m, {alive} still standing)";
                _last = $"{(char)('A' + winner)} ({BadgeOf(_seat[winner])}) holds the summit as the sea arrives";
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
            LastResult = new RoundResult(winner, winner >= 0 ? BadgeOf(_seat[winner]) : "No winner", how, end, (float)t, agents);
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

        // ------------------------------------------------------------------ free-for-all statistics gate
        bool FfaStats => statsPool != null && statsPool.Length > 0 && statsCandidate >= 0;
        int _ffaWin, _ffaLoss, _ffaTie; readonly List<double> _ffaTimes = new();

        // Candidate first (it takes body A), then three opponents drawn from the pool, as eval.py draws per world.
        List<int> FfaDraw()
        {
            var agents = new List<int> { statsCandidate };
            for (int k = 0; k < Mathf.Min(3, fighters.Length - 1); k++) agents.Add(statsPool[UnityEngine.Random.Range(0, statsPool.Length)]);
            return agents;
        }

        void FfaRoundOver(double t)
        {
            int alive = 0; for (int i = 0; i < _playing; i++) if (!_out[i]) alive++;
            if (_out[0]) _ffaLoss++; else if (alive == 1) { _ffaWin++; _ffaTimes.Add(t); } else _ffaTie++;
            int n = _ffaWin + _ffaLoss + _ffaTie;
            if (n < statsRounds) { StartMatch(FfaDraw(), false); return; }
            _ffaTimes.Sort(); double median = _ffaTimes.Count > 0 ? _ffaTimes[_ffaTimes.Count / 2] : -1;
            string F(double v) => v.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture);
            string json = $"{{\n \"candidate\": \"{NameOf(statsCandidate, "?")}\", \"rounds\": {n}, \"last_one_in\": {F((double)_ffaWin / n)}, \"out\": {F((double)_ffaLoss / n)}, " +
                          $"\"undecided\": {F((double)_ffaTie / n)}, \"median_win_time_s\": {F(median)}\n}}\n";
            string path = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "training", "assets", fighters[0].body, statsFile.Length > 0 ? statsFile : "ffa_stats_unity.json"));
            System.IO.File.WriteAllText(path, json); Debug.Log("[DemoDirector] wrote " + path + "\n" + json);
            foreach (var f in fighters) f.paused = true;
            StatsDone();
        }

        // End of a statistics gate: back to real time (50x with 2 s frames keeps the editor's main thread busy, so the
        // CLI cannot reach it), then leave play mode, the editor, or the player.
        void StatsDone()
        {
            statsRounds = 0; Time.timeScale = 1f; Time.maximumDeltaTime = 1f / 3f;
#if UNITY_EDITOR
            if (Cli.Exit) UnityEditor.EditorApplication.Exit(0); else UnityEditor.EditorApplication.ExitPlaymode();
#else
            Application.Quit();
#endif
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
            string path = System.IO.Path.GetFullPath(Application.isEditor ? System.IO.Path.Combine(Application.dataPath, "..", "training", "assets", fighters[0].body, statsFile.Length > 0 ? statsFile : "duel_stats_unity.json")
                                                                          : System.IO.Path.Combine(Application.dataPath, "..", "duel_stats_unity.json"));
            System.IO.File.WriteAllText(path, json); Debug.Log("[DemoDirector] wrote " + path + "\n" + json);
            fighters[0].paused = fighters[1].paused = true;      // seats A and B: the pair the two-fighter gate measures
            StatsDone();
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
            if (State == MatchState.Intro && Time.unscaledTime >= _introStart + _introLength) EndIntro();
            if (State == MatchState.Result)
            {
                float shownFor = pauseBetweenRounds - (_resultUntil - Time.unscaledTime);
                if (shownFor >= 1f) { if (_knockout) Shot("ejection"); Shot("winner"); }
                if (Time.unscaledTime >= _resultUntil) Advance();
            }
        }
    }
}
