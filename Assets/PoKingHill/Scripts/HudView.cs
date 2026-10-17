using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;

namespace PoKingHill
{
    /// <summary>
    /// The whole UI in UI Toolkit, one 9:16 column that never scrolls (styles in UI/Hud.uss).
    ///   corners   TL title · TC FPS (plus physics/brain ms with the telemetry HUD) · TR menu · BL debug · BR version
    ///   live      score badges, round clock and sea bar, win-odds bar, hazard chip, caption line, puppet stick
    ///   sheets    Menu (tabs Play | Agents | Settings) and Results (winner, per-fighter telemetry, replay, next)
    ///   debug     two-finger tap or the BL button: timings, overlays, intent arrows, hazard triggers, replay
    /// Safe-area padding follows Screen.safeArea. State comes from DemoDirector and MatchEvents; nothing here touches
    /// the simulation except through the director's public methods.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public class HudView : MonoBehaviour
    {
        public DemoDirector director;
        public WinOdds odds;
        public ReplaySystem replay;
        public MatchLog log;
        public Hazards hazards;
        public ChatVotes chat;
        public StyleSheet style;

        UIDocument _doc;
        VisualElement _root, _column, _safe, _scoreboard, _roundInfo, _seaFill, _odds, _puppet, _debug, _debugRows, _sheet, _sheetBody, _knob, _stick;
        Label _fps, _clock, _hazard, _caption, _status, _oddsLabel, _debugText, _countdown;
        Button _next;
        readonly List<VisualElement> _badges = new(); readonly List<VisualElement> _oddsSegs = new();
        enum SheetKind { None, Menu, Result }
        SheetKind _sheetKind; int _tab; float _fpsAvg = 60, _nextSlow, _captionUntil; string _lastStatus = ""; float _statusUntil;
        Rect _safeArea; Vector2 _screen;

        void OnEnable()
        {
            _doc = GetComponent<UIDocument>();
            Build();
            MatchEvents.StateChanged += OnState; MatchEvents.RoundStarted += RebuildScoreboard; MatchEvents.RoundEnded += OnRoundEnded; MatchEvents.Caption += OnCaption;
            GameSettings.Changed += OnSettings;
            if (director != null) OnState(director.State);
        }
        void OnDisable()
        {
            MatchEvents.StateChanged -= OnState; MatchEvents.RoundStarted -= RebuildScoreboard; MatchEvents.RoundEnded -= OnRoundEnded; MatchEvents.Caption -= OnCaption;
            GameSettings.Changed -= OnSettings;
        }

        // ------------------------------------------------------------------ construction
        static VisualElement El(string cls, VisualElement parent = null) { var e = new VisualElement(); foreach (var c in cls.Split(' ', StringSplitOptions.RemoveEmptyEntries)) e.AddToClassList(c); parent?.Add(e); return e; }
        static Label Text(string text, string cls, VisualElement parent) { var l = new Label(text); foreach (var c in cls.Split(' ', StringSplitOptions.RemoveEmptyEntries)) l.AddToClassList(c); parent.Add(l); return l; }
        static Button Btn(string text, string cls, Action click, VisualElement parent) { var b = new Button(click) { text = text }; foreach (var c in cls.Split(' ', StringSplitOptions.RemoveEmptyEntries)) b.AddToClassList(c); parent.Add(b); return b; }

        void Build()
        {
            var top = _doc.rootVisualElement; top.Clear();
            if (style != null) top.styleSheets.Add(style);
            _root = El("root", top); _root.style.flexGrow = 1;
            _column = El("column", _root);
            _root.RegisterCallback<GeometryChangedEvent>(_ => FitColumn());
            _safe = El("safe", _column);

            // top bar: TL title · TC FPS · TR menu
            var bar = El("top-bar", _safe);
            var mid = El("top-center", bar); mid.pickingMode = PickingMode.Ignore;      // FPS centred on the screen, under title and menu
            _fps = Text("60 FPS", "pill fps", mid);
            Text("PoKingHill", "title shadow", bar);
            var right = El("top-right", bar);
            var menu = Btn("", "icon-btn", ToggleMenu, right); menu.Add(new Icon(Icon.Kind.Menu)); menu.tooltip = "Menu";

            _scoreboard = El("scoreboard", _safe);
            _roundInfo = El("round-info", _safe);
            _clock = Text("", "clock shadow", _roundInfo);
            var seaBar = El("sea-bar", _roundInfo); _seaFill = El("sea-fill", seaBar);
            _odds = El("odds", _safe);
            _oddsLabel = Text("", "odds-label dim shadow", _safe);
            _hazard = Text("", "hazard-chip", _safe);
            El("spacer", _safe);
            _caption = Text("", "caption", _safe); _caption.style.opacity = 0;

            _puppet = El("puppet", _safe); BuildPuppet();

            var bottom = El("bottom-bar", _safe);
            var dbg = Btn("", "icon-btn", ToggleDebug, bottom); dbg.Add(new Icon(Icon.Kind.Sliders)); dbg.tooltip = "Debug (or tap with two fingers)";
            _status = Text("", "status shadow", bottom);
            Text(BuildInfo.Version, "version shadow", bottom);

            _debug = El("debug-panel", _column); _debugText = Text("", "", _debug); _debugRows = El("", _debug); _debug.style.display = DisplayStyle.None;

            _sheet = El("sheet", _column); _sheet.style.display = DisplayStyle.None;
            El("grabber", _sheet);
            _sheetBody = El("", _sheet);
        }

        void FitColumn()
        {
            float w = _root.layout.width, h = _root.layout.height; if (float.IsNaN(w) || h <= 0) return;
            float col = Mathf.Min(w, h * 9f / 16f);
            bool changed = Mathf.Abs(_column.resolvedStyle.width - col) > 0.5f;
            if (changed) _column.style.width = col;
            ApplySafeArea(changed);
        }

        void ApplySafeArea(bool force)
        {
            var sa = Screen.safeArea; var screen = new Vector2(Screen.width, Screen.height);
            if (!force && sa == _safeArea && screen == _screen) return;
            _safeArea = sa; _screen = screen;
            float k = _root.layout.height / Mathf.Max(1, Screen.height); if (float.IsNaN(k) || k <= 0) return;
            bool fullWidth = Mathf.Abs(_column.layout.width - _root.layout.width) < 2;
            _safe.style.paddingTop = 24 + (Screen.height - sa.yMax) * k; _safe.style.paddingBottom = 24 + sa.yMin * k;
            _safe.style.paddingLeft = 28 + (fullWidth ? sa.xMin * k : 0); _safe.style.paddingRight = 28 + (fullWidth ? (Screen.width - sa.xMax) * k : 0);
            _sheet.style.bottom = 120 + sa.yMin * k;       // the sheet ends above the bottom corners (debug, version)
        }

        void BuildPuppet()
        {
            _stick = El("stick", _puppet); _knob = El("stick-knob", _stick);
            _stick.RegisterCallback<PointerDownEvent>(e => { _stick.CapturePointer(e.pointerId); MoveStick(e.localPosition); });
            _stick.RegisterCallback<PointerMoveEvent>(e => { if (_stick.HasPointerCapture(e.pointerId)) MoveStick(e.localPosition); });
            _stick.RegisterCallback<PointerUpEvent>(e => { _stick.ReleasePointer(e.pointerId); PuppetControl.Stick = Vector2.zero; _knob.style.left = 90; _knob.style.top = 90; });
            var side = El("side-btns", _puppet);
            foreach (var (label, dir) in new[] { ("Step L", 1f), ("Step R", -1f) })
            {
                var b = Btn(label, "side-btn", null, side);
                b.RegisterCallback<PointerDownEvent>(_ => PuppetControl.Side = dir, TrickleDown.TrickleDown);
                b.RegisterCallback<PointerUpEvent>(_ => PuppetControl.Side = 0, TrickleDown.TrickleDown);
                b.RegisterCallback<PointerLeaveEvent>(_ => PuppetControl.Side = 0);
            }
            _puppet.style.display = DisplayStyle.None;
        }

        void MoveStick(Vector2 local)
        {
            Vector2 c = new(150, 150), v = Vector2.ClampMagnitude(local - c, 110f);
            _knob.style.left = 90 + v.x; _knob.style.top = 90 + v.y;
            PuppetControl.Stick = new Vector2(v.x / 110f, -v.y / 110f);
        }

        // ------------------------------------------------------------------ events
        void OnState(MatchState s)
        {
            if (s == MatchState.Menu) ShowSheet(SheetKind.Menu);
            else if (s == MatchState.Result) ShowSheet(SheetKind.Result);
            else ShowSheet(SheetKind.None);
            RebuildScoreboard();
        }
        void OnRoundEnded(RoundResult r) { }      // the Result state change rebuilds the sheet
        void OnCaption(string text) { _caption.text = text; _caption.style.opacity = 1; _captionUntil = Time.unscaledTime + 3.5f; }
        void OnSettings() { if (_sheetKind == SheetKind.Menu) RebuildSheet(); }

        void ToggleMenu()
        {
            if (director == null) return;
            if (director.State == MatchState.Menu) return;
            if (ReplaySystem.Active) replay.Stop();
            director.EnterMenu();
        }
        void ToggleDebug() { bool on = _debug.style.display == DisplayStyle.None; _debug.style.display = on ? DisplayStyle.Flex : DisplayStyle.None; if (on) BuildDebugRows(); }

        void ShowSheet(SheetKind k)
        {
            _sheetKind = k; _sheet.style.display = k == SheetKind.None ? DisplayStyle.None : DisplayStyle.Flex;
            if (k != SheetKind.None) RebuildSheet();
        }

        void RebuildSheet()
        {
            _sheetBody.Clear(); _next = null; _countdown = null;
            if (_sheetKind == SheetKind.Menu) BuildMenu(); else if (_sheetKind == SheetKind.Result) BuildResult();
        }

        // ------------------------------------------------------------------ menu
        void BuildMenu()
        {
            var tabs = El("tabs", _sheetBody);
            string[] names = { "Play", "Agents", "Settings" };
            for (int i = 0; i < names.Length; i++) { int t = i; var b = Btn(names[i], "tab", () => { _tab = t; RebuildSheet(); }, tabs); if (i == _tab) b.AddToClassList("selected"); }
            if (_tab == 0) BuildPlay(); else if (_tab == 1) BuildAgents(); else BuildSettings();
        }

        void BuildPlay()
        {
            Text("King of the hill", "h1", _sheetBody);
            Text("Stay on the summit while the sea rises. Last one standing wins.", "body", _sheetBody);
            Btn("Quick match", "primary", () => director.QuickMatch(), _sheetBody);
            var row = El("row", _sheetBody);
            int ticked = director.TickedCount;
            Btn(ticked == 0 ? "Custom (tick agents)" : $"Custom · {ticked} agents", "secondary", () => director.Launch(), row).SetEnabled(ticked > 0);
            Btn("Tournament", "secondary", () => director.StartTournament(), row);
            Text("Map", "h2", _sheetBody);
            Seg(director.maps, director.selectedMap, i => { director.selectedMap = i; RebuildSheet(); });
            bool canSteer = Array.Exists(director.roster, f => f.steerable && f.available && f.policy != null);
            if (canSteer)
            {
                var grid = El("grid", _sheetBody);
                Chip(grid, "You play: steer a walker (stick or WASD)", director.PuppetRequested, v => { director.PuppetRequested = v; RebuildSheet(); });
                grid[0].style.width = Length.Percent(100);
            }
            if (director.Status.Length > 0) Text(director.Status, "body", _sheetBody);
        }

        void BuildAgents()
        {
            Text("Agents", "h1", _sheetBody);
            Text(TickHint(), "body", _sheetBody);
            var grid = El("grid", _sheetBody); grid.style.marginTop = 16;
            for (int i = 0; i < director.roster.Length; i++)
            {
                var f = director.roster[i]; int idx = i;
                var card = El("agent", grid); if (f.available && f.inGame) card.AddToClassList("on"); if (!f.available) card.AddToClassList("locked");
                var topRow = El("agent-top", card);
                var av = El("avatar", topRow); av.style.backgroundColor = DemoDirector.SeatColors[i % DemoDirector.SeatColors.Length];
                Text(Initials(f.Badge), "", av);
                Text(f.Badge, "name", topRow);
                string meta = f.available ? $"Elo {Ladder.Rating(f.name):0} · {Ladder.Rounds(f.name)} rounds · {director.Wins(i)} wins" : "Locked: " + f.note;
                if (f.available && f.note.Length > 0) meta += " · " + f.note;
                Text(meta, "meta", card);
                if (f.available) card.RegisterCallback<ClickEvent>(_ => { f.inGame = !f.inGame; RebuildSheet(); });
                card.tooltip = f.name;
            }
        }

        string TickHint()
        {
            int n = director.TickedCount, bodies = director.fighters.Length;
            return n == 0 ? "Tap agents to tick them for a custom match." : n == 1 ? "One ticked: it fights itself." : n == 2 ? "These two fight." :
                   n <= bodies ? $"All {n} fight at once: the last one on the summit wins." : $"{bodies} of them are drawn for each match.";
        }

        static string Initials(string s) { var parts = s.Split(new[] { ' ', '-' }, StringSplitOptions.RemoveEmptyEntries); return parts.Length == 0 ? "?" : parts.Length == 1 ? parts[0].Substring(0, Math.Min(2, parts[0].Length)).ToUpperInvariant() : (parts[0][0].ToString() + parts[1][0]).ToUpperInvariant(); }

        void BuildSettings()
        {
            Text("Match", "h2", _sheetBody);
            int rounds = GameSettings.RoundsPerMatch; int[] roundOpts = { 1, 3, 5, 0 };
            Labeled("Rounds", new[] { "1", "3", "5", "Endless" }, Array.IndexOf(roundOpts, rounds), i => GameSettings.RoundsPerMatch = roundOpts[i]);
            float[] paces = { 1.5f, 1f, 0.7f };
            Labeled("Sea", new[] { "Slow", "Normal", "Fast" }, Nearest(paces, GameSettings.SeaPace), i => GameSettings.SeaPace = paces[i]);
            Labeled("Targets", new[] { "Nearest", "Opportunist" }, (int)GameSettings.Targeting, i => GameSettings.Targeting = (Targeting)i);
            Text("Show", "h2", _sheetBody);
            Labeled("Camera", new[] { "Broadcast", "Classic" }, (int)GameSettings.Camera, i => GameSettings.Camera = (CameraStyle)i);
            float[] vols = { 0f, 0.3f, 0.6f, 1f };
            Labeled("Sound", new[] { "Off", "Low", "Mid", "High" }, Nearest(vols, GameSettings.Volume), i => GameSettings.Volume = vols[i]);
            var grid = El("grid", _sheetBody);
            Chip(grid, "Hazards", GameSettings.Hazards, v => GameSettings.Hazards = v);
            Chip(grid, "Camera cut-aways", GameSettings.CutAways, v => GameSettings.CutAways = v);
            Chip(grid, "Instant replays", GameSettings.Replays, v => GameSettings.Replays = v);
            Chip(grid, "Effects", GameSettings.Effects, v => GameSettings.Effects = v);
            Chip(grid, "Win odds", GameSettings.WinOdds, v => GameSettings.WinOdds = v);
            Chip(grid, "Commentary", GameSettings.Commentary, v => GameSettings.Commentary = v);
            Text("Connect", "h2", _sheetBody);
            var tf = new TextField { value = GameSettings.TwitchChannel, textEdition = { placeholder = "Twitch channel for chat votes (!wind !ice !quake !shrink !boxes)" } };
            tf.AddToClassList("field"); tf.RegisterCallback<FocusOutEvent>(_ => GameSettings.TwitchChannel = tf.value); _sheetBody.Add(tf);
            var grid2 = El("grid", _sheetBody);
            Chip(grid2, Commentator.ClaudeAvailable ? "Claude commentary" : "Claude commentary (set ANTHROPIC_API_KEY)", GameSettings.ClaudeCommentary, v => GameSettings.ClaudeCommentary = v, Commentator.ClaudeAvailable);
            bool sas = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("KOTH_AZURE_TABLE_SAS"));
            Chip(grid2, sas ? "Upload telemetry (Azure)" : "Upload telemetry (set KOTH_AZURE_TABLE_SAS)", GameSettings.UploadTelemetry, v => GameSettings.UploadTelemetry = v, sas);
            var reset = Btn("Reset all settings", "secondary", () => { GameSettings.ResetAll(); RebuildSheet(); }, _sheetBody);
            reset.style.flexGrow = 0; reset.style.flexShrink = 0; reset.style.height = 76;
        }

        static int Nearest(float[] opts, float v) { int best = 0; for (int i = 1; i < opts.Length; i++) if (Mathf.Abs(opts[i] - v) < Mathf.Abs(opts[best] - v)) best = i; return best; }

        void Labeled(string label, string[] opts, int selected, Action<int> pick)
        {
            var row = El("row", _sheetBody); row.style.alignItems = Align.Center; row.style.marginLeft = 0; row.style.marginRight = 0;
            var l = Text(label, "body", row); l.style.width = 170;
            var seg = Seg(opts, selected, pick, row); seg.style.flexGrow = 1;
        }

        VisualElement Seg(string[] opts, int selected, Action<int> pick, VisualElement parent = null)
        {
            var seg = El("seg", parent ?? _sheetBody);
            for (int i = 0; i < opts.Length; i++) { int k = i; var b = Btn(opts[i], "seg-btn", () => { pick(k); RebuildSheet(); }, seg); if (i == selected) b.AddToClassList("selected"); }
            return seg;
        }

        void Chip(VisualElement parent, string label, bool on, Action<bool> set, bool enabled = true)
        {
            var chip = El("chip", parent); if (on) chip.AddToClassList("on"); if (!enabled) chip.AddToClassList("locked");
            var l = Text(label, "", chip); l.style.flexShrink = 1;
            Text(on ? "ON" : "OFF", "state", chip);
            if (enabled) chip.RegisterCallback<ClickEvent>(_ => { set(!on); if (_sheetKind == SheetKind.Menu) RebuildSheet(); else BuildDebugRows(); });
        }

        // ------------------------------------------------------------------ results
        void BuildResult()
        {
            var r = director.LastResult;
            if (director.TournamentChampion.Length > 0 && !director.InTournament)
                Text("Tournament champion: " + director.TournamentChampion, "h1", _sheetBody);
            Text(r.Winner >= 0 ? "Winner: " + r.WinnerName : "No winner", "h1", _sheetBody);
            var stripe = El("winner-stripe", _sheetBody);
            stripe.style.backgroundColor = r.Winner >= 0 ? DemoDirector.SeatColors[r.Winner % DemoDirector.SeatColors.Length] : new Color(1, 1, 1, 0.2f);
            Text(char.ToUpper(r.How[0]) + r.How.Substring(1), "body", _sheetBody);

            var head = El("res-row res-head", _sheetBody); head.style.marginTop = 14;
            Text("Fighter", "res-name", head); foreach (var h in new[] { "Result", "Out at", "Walked", "Top" }) Text(h, "res-cell", head); Text("Elo", "res-cell elo", head);
            for (int i = 0; i < r.Agents.Length; i++)
            {
                var row = El("res-row", _sheetBody);
                var nameBox = El("res-name", row); nameBox.style.flexDirection = FlexDirection.Row; nameBox.style.alignItems = Align.Center;
                var dot = El("dot", nameBox); dot.style.backgroundColor = DemoDirector.SeatColors[i % DemoDirector.SeatColors.Length];
                Text(director.BadgeOf(r.Agents[i]) + (director.PuppetBody == i ? " (you)" : ""), "", nameBox);
                string res = r.Winner < 0 ? "Tie" : r.Winner == i ? "Win" : "Loss";
                Text(res, "res-cell " + (res == "Win" ? "win" : res == "Loss" ? "loss" : ""), row);
                float ko = log != null ? log.KnockedOutAt(i) : -1;
                Text(ko >= 0 ? $"{ko:0.0} s" : "-", "res-cell", row);
                Text(log != null ? $"{log.PathMetres(i):0.0} m" : "", "res-cell", row);
                Text(log != null ? $"{log.TopSpeed(i):0.0} m/s" : "", "res-cell", row);
                float dElo = log != null ? log.RatingDelta(i) : 0;
                float elo = Ladder.Rating(director.NameOf(r.Agents[i], "Robot"));
                Text(Mathf.Abs(dElo) < 0.5f ? $"{elo:0}" : $"{elo:0} ({(dElo >= 0 ? "+" : "")}{dElo:0})", "res-cell elo", row);
            }
            if (director.InTournament)
            {
                var b = El("bracket", _sheetBody);
                foreach (var m in director.Bracket)
                    Text($"{m.label}: {(m.a >= 0 ? director.BadgeOf(m.a) : "?")} vs {(m.b >= 0 ? director.BadgeOf(m.b) : "?")}" + (m.winner >= 0 ? $": {director.BadgeOf(m.winner)} wins" : ""), "body", b);
            }
            var row2 = El("row", _sheetBody); row2.style.marginTop = 18;
            if (replay != null && replay.Available) Btn("Replay knock-out", "secondary", () => replay.Play(), row2);
            _next = Btn("", "primary next", () => director.Continue(), row2);
            UpdateCountdown();
        }

        void UpdateCountdown()
        {
            if (_next == null) return;
            string what = director.InTournament ? "Next match" : director.LastRoundOfMatch ? "Main menu" : "Next round";
            _next.text = $"{what} ({Mathf.CeilToInt(director.ResultSecondsLeft)})";
        }

        // ------------------------------------------------------------------ live strip
        void RebuildScoreboard()
        {
            if (director == null) return;
            _scoreboard.Clear(); _badges.Clear(); _odds.Clear(); _oddsSegs.Clear();
            if (director.InMenu) return;
            for (int i = 0; i < director.Playing; i++)
            {
                var b = El("badge", _scoreboard); if (director.PuppetBody == i) b.AddToClassList("you");
                var dot = El("dot", b); dot.style.backgroundColor = DemoDirector.SeatColors[i % DemoDirector.SeatColors.Length];
                Text(director.BadgeOf(director.AgentOf(i)) + (director.PuppetBody == i ? " (you)" : ""), "", b);
                int agent = director.AgentOf(i);
                Text((director.AgentSeatedTwice(agent) ? director.BodyWins(i) : director.Wins(agent)).ToString(), "wins", b);
                _badges.Add(b);
                var seg = El("odds-seg", _odds); seg.style.backgroundColor = DemoDirector.SeatColors[i % DemoDirector.SeatColors.Length]; seg.style.width = Length.Percent(100f / director.Playing);
                _oddsSegs.Add(seg);
            }
            if (director.Draws > 0) { var t = El("badge", _scoreboard); Text("Ties " + director.Draws, "", t); }
        }

        void Update()
        {
            if (director == null || _root == null) return;
            _fpsAvg = Mathf.Lerp(_fpsAvg, 1f / Mathf.Max(Time.unscaledDeltaTime, 1e-4f), 0.05f);
            bool fight = director.State == MatchState.Fight, live = fight || director.RoundOver;
            if (Time.unscaledTime >= _nextSlow)
            {
                _nextSlow = Time.unscaledTime + 0.25f;
                FitColumn();
                _fps.text = GameSettings.Telemetry ? $"{_fpsAvg:0} FPS · phys {director.StepMs:0.00} · brain {(director.fighters.Length > 0 ? director.fighters[0].LastInferenceMs : 0):0.00} ms" : $"{_fpsAvg:0} FPS";
                if (_debug.style.display == DisplayStyle.Flex) _debugText.text = DebugText();
                UpdateCountdown();
            }
            _roundInfo.style.display = live ? DisplayStyle.Flex : DisplayStyle.None;
            if (live)
            {
                _clock.text = ReplaySystem.Active ? "REPLAY" : $"{director.RoundTime:0.0} s";
                var sea = director.sea;
                _seaFill.style.width = Length.Percent(sea != null ? Mathf.Clamp01((sea.Level + sea.startDepth) / sea.startDepth) * 100f : 0f);
                for (int i = 0; i < _badges.Count; i++) _badges[i].EnableInClassList("out", director.IsOut(i));
            }
            bool showOdds = fight && GameSettings.WinOdds && odds != null && _oddsSegs.Count > 1;
            _odds.style.display = showOdds ? DisplayStyle.Flex : DisplayStyle.None; _oddsLabel.style.display = _odds.style.display;
            if (showOdds)
            {
                for (int i = 0; i < _oddsSegs.Count; i++) _oddsSegs[i].style.width = Length.Percent(odds.Odds[i] * 100f);
                _oddsLabel.text = odds.FromCritic ? "Win odds · critic" : "Win odds · estimate";
            }
            string hz = hazards != null && fight ? hazards.Active : "";
            if (chat != null && chat.WindowLeft > 0) hz = $"Chat vote {chat.WindowLeft:0}s";
            _hazard.text = hz; _hazard.style.display = hz.Length > 0 ? DisplayStyle.Flex : DisplayStyle.None;
            if (_captionUntil > 0 && Time.unscaledTime > _captionUntil) { _captionUntil = 0; _caption.style.opacity = 0; }
            _caption.style.visibility = _sheet.style.display == DisplayStyle.Flex ? Visibility.Hidden : Visibility.Visible;   // never under a sheet
            if (director.Status != _lastStatus) { _lastStatus = director.Status; _statusUntil = Time.unscaledTime + 4f; }
            _status.text = Time.unscaledTime < _statusUntil ? _lastStatus : "";
            _puppet.style.display = fight && director.PuppetBody >= 0 ? DisplayStyle.Flex : DisplayStyle.None;
            _sheet.style.display = _sheetKind != SheetKind.None && !ReplaySystem.Active ? DisplayStyle.Flex : DisplayStyle.None;
            _debug.style.visibility = _sheet.style.display == DisplayStyle.Flex ? Visibility.Hidden : Visibility.Visible;     // a sheet always wins
            TwoFingerTap();
        }

        void TwoFingerTap()
        {
            var ts = Touchscreen.current; if (ts == null) return;
            int down = 0, began = 0;
            foreach (var t in ts.touches) { if (t.isInProgress) down++; if (t.press.wasPressedThisFrame) began++; }
            if (down == 2 && began > 0) ToggleDebug();
        }

        // ------------------------------------------------------------------ debug panel
        void BuildDebugRows()
        {
            _debugRows.Clear();
            var seg = El("seg", _debugRows); string[] modes = { "No overlay", "Strain", "Fatigue", "Colliders" };
            for (int i = 0; i < modes.Length; i++) { int k = i; var b = Btn(modes[i], "seg-btn", () => { GameSettings.Overlay = (Overlay)k; BuildDebugRows(); }, seg); if ((int)GameSettings.Overlay == i) b.AddToClassList("selected"); }
            var grid = El("grid", _debugRows);
            Chip(grid, "Intent arrows", GameSettings.IntentArrows, v => GameSettings.IntentArrows = v);
            Chip(grid, "Telemetry HUD", GameSettings.Telemetry, v => GameSettings.Telemetry = v);
            var hz = El("debug-row", _debugRows);
            foreach (Hazard h in Enum.GetValues(typeof(Hazard))) { var hh = h; Btn(h.ToString(), "seg-btn", () => hazards?.Trigger(hh), hz).style.flexGrow = 1; }
            var r = El("debug-row", _debugRows);
            Btn("Replay", "seg-btn", () => replay?.Play(), r).style.flexGrow = 1;
            Btn("Restart round", "seg-btn", () => { if (director.State == MatchState.Fight) director.NewRound(); }, r).style.flexGrow = 1;
        }

        string DebugText()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append($"state {director.State} · round {director.RoundTime:0.0} s · out radius {director.OutRadiusNow:0.00} m\n");
            sb.Append($"physics tick {director.StepMs:0.000} ms · ");
            for (int i = 0; i < director.Playing && i < director.fighters.Length; i++) sb.Append($"{director.fighters[i].robotPrefix} {director.fighters[i].LastInferenceMs:0.00} ms  ");
            sb.Append($"\nGC {GC.CollectionCount(0)} gen0 · telemetry {MatchLog.Folder}");
            if (chat != null) sb.Append($"\nchat: {chat.Status}");
            return sb.ToString();
        }
    }

    /// <summary>Small vector icons drawn with Painter2D, so no font glyph or texture is needed.</summary>
    public class Icon : VisualElement
    {
        public enum Kind { Menu, Sliders }
        readonly Kind _kind;
        public Icon(Kind kind) { _kind = kind; AddToClassList("icon"); pickingMode = PickingMode.Ignore; generateVisualContent += Draw; }

        void Draw(MeshGenerationContext ctx)
        {
            var p = ctx.painter2D; var r = contentRect; float s = Mathf.Min(r.width, r.height); var c = r.center; float u = s * 0.18f;
            p.strokeColor = new Color(0.96f, 0.97f, 0.98f); p.lineWidth = Mathf.Max(2f, s * 0.07f); p.lineCap = LineCap.Round;
            if (_kind == Kind.Menu)
                for (int i = -1; i <= 1; i++) { p.BeginPath(); p.MoveTo(new Vector2(c.x - u * 1.4f, c.y + i * u)); p.LineTo(new Vector2(c.x + u * 1.4f, c.y + i * u)); p.Stroke(); }
            else
                for (int i = -1; i <= 1; i += 2)
                {
                    p.BeginPath(); p.MoveTo(new Vector2(c.x - u * 1.4f, c.y + i * u * 0.8f)); p.LineTo(new Vector2(c.x + u * 1.4f, c.y + i * u * 0.8f)); p.Stroke();
                    p.fillColor = new Color(1f, 0.77f, 0.25f); p.BeginPath(); p.Arc(new Vector2(c.x + i * u * 0.6f, c.y + i * u * 0.8f), u * 0.42f, 0, 360); p.Fill();
                }
        }
    }
}
