using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;

namespace PoKingHill
{
    /// <summary>
    /// The whole UI in UI Toolkit, one 9:16 column that never scrolls (styles in UI/Hud.uss).
    ///   corners   TL title · TC FPS (plus frame p95 and quality with the telemetry HUD; hold it for the debug panel)
    ///             · TR menu · BL debug · BR version. The FPS pill steps aside from a punch-hole camera (Screen.cutouts).
    ///   live      one row of fighter badges (seat-coloured initials, wins pip, status tag, vitals strip; tap =
    ///             adrenaline), round clock, sea bar and ties, win-odds bar, hazard chip, knock-out feed, caption line,
    ///             replay bar with a draggable playhead, puppet stick that appears under the thumb
    ///   intro     tale of the tape: one card per fighter (mass, strongest joint, Elo, record, knock-outs, brain, weather)
    ///   sheets    Menu (tabs Play | Agents | Match | Show | Connect) and Results (winner, replay strip, per-fighter
    ///             telemetry, next). A sheet swipes down (or its grabber is tapped) to a peek pill; a swipe up from the
    ///             bottom edge brings it back, or opens the menu during a round. A sheet taller than the room between
    ///             the corners turns dense.
    ///   debug     two-finger tap, holding the FPS pill or the BL button: tabs Perf | View | Hazards
    /// Safe-area padding follows Screen.safeArea. State comes from DemoDirector and MatchEvents; nothing here touches
    /// the simulation except through the director's and the replay's public methods. Tests and captures can lay the HUD
    /// out for another screen (SimulateScreen) and list what does not fit (LayoutProblems).
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
        public Biometrics bio;
        public ContactLedger ledger;
        public Stamina stamina;
        public Tactician tactician;
        public WeatherFeed weather;
        public QualityTier quality;               // optional: the Show tab's quality row and the telemetry line
        public ImpactSynth synth;                 // optional: audio levels in the debug panel
        public TimeWarp warp;                     // optional: slow-motion counts in the debug panel
        public StyleSheet style;

        const float EdgeBand = 220f, SwipeDistance = 180f, DismissDistance = 140f, HoldSeconds = 0.5f;

        UIDocument _doc;
        VisualElement _root, _column, _safe, _topBar, _bottomBar, _menuBtn, _scoreboard, _roundInfo, _seaFill, _odds, _puppet, _debug, _debugTabs, _debugBody,
                      _sheet, _sheetBody, _knob, _stick, _stickZone, _feed, _intro, _replayBar, _replayFill, _replayHead, _peekRow, _resFill, _resHead;
        Label _title, _fps, _clock, _ties, _hazard, _caption, _status, _oddsLabel, _debugText;
        Button _next, _peek;
        readonly List<VisualElement> _badges = new(); readonly List<VisualElement> _oddsSegs = new();
        readonly List<VitalsBar> _vitals = new(); readonly List<Label> _tags = new(); readonly List<Label> _pips = new();
        readonly List<(VisualElement row, float until)> _feedRows = new(); int _introShown = -2;
        enum SheetKind { None, Menu, Result }
        SheetKind _sheetKind; int _tab, _debugTab; bool _collapsed;
        float _fpsAvg = 60, _nextSlow, _captionUntil, _pressAt = -1; string _lastStatus = ""; float _statusUntil;
        Vector2 _stickCenter;
        Rect _safeArea; Vector2 _screen; float _bottomInset;
        // SimulateScreen: a virtual screen in pixels (safe area and cutouts with the origin bottom-left, like Screen's)
        bool _sim; Vector2 _simSize; Rect _simSafe; Rect[] _simCutouts = Array.Empty<Rect>();

        void OnEnable()
        {
            _doc = GetComponent<UIDocument>();
            Build();
            MatchEvents.StateChanged += OnState; MatchEvents.RoundStarted += OnRoundStarted; MatchEvents.RoundEnded += OnRoundEnded; MatchEvents.Caption += OnCaption;
            MatchEvents.KnockoutCredited += OnCredited;
            GameSettings.Changed += OnSettings;
            if (director != null) OnState(director.State);
        }
        void OnDisable()
        {
            MatchEvents.StateChanged -= OnState; MatchEvents.RoundStarted -= OnRoundStarted; MatchEvents.RoundEnded -= OnRoundEnded; MatchEvents.Caption -= OnCaption;
            MatchEvents.KnockoutCredited -= OnCredited;
            GameSettings.Changed -= OnSettings;
            PuppetControl.Stick = Vector2.zero; PuppetControl.Side = 0;
        }
        void OnRoundStarted() { RebuildScoreboard(); foreach (var (row, _) in _feedRows) row.RemoveFromHierarchy(); _feedRows.Clear(); }

        // ------------------------------------------------------------------ public hooks (tests, captures, CLI)
        /// <summary>The menu tab on show: 0 Play, 1 Agents, 2 Match, 3 Show, 4 Connect.</summary>
        public int Tab { get => _tab; set { _tab = Mathf.Clamp(value, 0, 4); if (_sheetKind == SheetKind.Menu) RebuildSheet(); } }
        public bool SheetShown => _sheet != null && _sheet.style.display == DisplayStyle.Flex;
        public bool SheetCollapsed => _collapsed;
        public bool DebugShown => _debug != null && _debug.style.display == DisplayStyle.Flex;
        public bool PuppetShown => _puppet != null && _puppet.style.display == DisplayStyle.Flex;
        public bool ReplayBarShown => _replayBar != null && _replayBar.style.display == DisplayStyle.Flex;
        public int BadgeCount => _badges.Count;
        public void SetCollapsed(bool collapsed) { _collapsed = collapsed && _sheetKind != SheetKind.None; ApplyVisibility(); }
        public void SetDebugTab(int tab) { _debugTab = Mathf.Clamp(tab, 0, 2); if (DebugShown) BuildDebugRows(); }

        /// <summary>Lays the HUD out as on a screen of this many pixels, with this safe area and these cutouts (origin
        /// bottom-left, as Screen reports them; an empty safe area = the whole screen). Clear with ClearSimulation.</summary>
        public void SimulateScreen(int width, int height, Rect safeArea = default, Rect[] cutouts = null)
        {
            _sim = true; _simSize = new Vector2(width, height);
            _simSafe = safeArea.width > 0 ? safeArea : new Rect(0, 0, width, height);
            _simCutouts = cutouts ?? Array.Empty<Rect>();
            _safeArea = default; FitColumn();
        }
        public void ClearSimulation() { _sim = false; _safeArea = default; _fps.style.translate = StyleKeyword.Null; FitColumn(); }

        Vector2 ScreenSize => _sim ? _simSize : new Vector2(Screen.width, Screen.height);
        Rect SafeArea => _sim ? _simSafe : Screen.safeArea;
        Rect[] Cutouts => _sim ? _simCutouts : Screen.cutouts;

        // ------------------------------------------------------------------ construction
        static VisualElement El(string cls, VisualElement parent = null) { var e = new VisualElement(); foreach (var c in cls.Split(' ', StringSplitOptions.RemoveEmptyEntries)) e.AddToClassList(c); parent?.Add(e); return e; }
        static Label Text(string text, string cls, VisualElement parent) { var l = new Label(text); foreach (var c in cls.Split(' ', StringSplitOptions.RemoveEmptyEntries)) l.AddToClassList(c); parent.Add(l); return l; }
        static Button Btn(string text, string cls, Action click, VisualElement parent) { var b = new Button(click) { text = text }; foreach (var c in cls.Split(' ', StringSplitOptions.RemoveEmptyEntries)) b.AddToClassList(c); parent.Add(b); return b; }

        void Build()
        {
            var top = _doc.rootVisualElement; top.Clear();
            if (style != null) top.styleSheets.Add(style);
            _root = El("root", top);
            _column = El("column", _root);
            _root.RegisterCallback<GeometryChangedEvent>(_ => FitColumn());
            _safe = El("safe", _column);

            // top bar: TL title · TC FPS · TR menu
            _topBar = El("top-bar", _safe);
            var mid = El("top-center", _topBar); mid.pickingMode = PickingMode.Ignore;      // FPS centred on the screen, under title and menu
            _fps = Text("60 FPS", "pill fps", mid); _fps.tooltip = "Hold for the debug panel";
            _fps.RegisterCallback<PointerDownEvent>(_ => _pressAt = Time.unscaledTime);
            _fps.RegisterCallback<PointerUpEvent>(_ => _pressAt = -1);
            _fps.RegisterCallback<PointerLeaveEvent>(_ => _pressAt = -1);
            _title = Text("PoKingHill", "title shadow", _topBar);
            var right = El("top-right", _topBar);
            _menuBtn = Btn("", "icon-btn", ToggleMenu, right); _menuBtn.Add(new Icon(Icon.Kind.Menu)); _menuBtn.tooltip = "Menu";

            _scoreboard = El("scoreboard", _safe);
            _roundInfo = El("round-info", _safe);
            _clock = Text("", "clock shadow", _roundInfo);
            var seaBar = El("sea-bar clip", _roundInfo); _seaFill = El("sea-fill", seaBar);
            _ties = Text("", "ties", _roundInfo);
            _odds = El("odds clip", _safe);
            _oddsLabel = Text("", "odds-label dim shadow", _safe);
            _hazard = Text("", "hazard-chip", _safe);
            _feed = El("feed", _safe); _feed.pickingMode = PickingMode.Ignore;
            El("spacer", _safe);
            _intro = El("intro", _safe); _intro.style.display = DisplayStyle.None;
            _caption = Text("", "caption", _safe); _caption.style.opacity = 0;
            _replayBar = El("replay-bar", _safe); BuildReplayBar();
            _puppet = El("puppet", _safe); BuildPuppet();

            _bottomBar = El("bottom-bar", _safe);
            var dbg = Btn("", "icon-btn", ToggleDebug, _bottomBar); dbg.Add(new Icon(Icon.Kind.Sliders)); dbg.tooltip = "Debug (or tap with two fingers, or hold the FPS pill)";
            _status = Text("", "status shadow clip", _bottomBar);
            Text(BuildInfo.Version, "version shadow", _bottomBar);

            _debug = El("debug-panel", _column); _debug.style.display = DisplayStyle.None;
            _debugTabs = El("tabs", _debug); _debugBody = El("", _debug); _debugText = new Label();

            _sheet = El("sheet", _column); _sheet.style.display = DisplayStyle.None;
            var grab = El("grab-zone", _sheet); El("grabber", grab); grab.tooltip = "Swipe down (or tap) to hide";
            SheetDrag(grab);
            _sheetBody = El("", _sheet);
            _sheet.RegisterCallback<GeometryChangedEvent>(_ => FitSheet());

            _peekRow = El("peek-row", _column); _peekRow.pickingMode = PickingMode.Ignore;
            _peek = Btn("", "peek", () => SetCollapsed(false), _peekRow);
            _peekRow.style.display = DisplayStyle.None;
            EdgeSwipe();
        }

        void FitColumn()
        {
            float w = _root.layout.width, h = _root.layout.height; if (float.IsNaN(w) || h <= 0) return;
            var scr = ScreenSize;
            float panelW = _sim ? h * scr.x / Mathf.Max(1, scr.y) : w;          // the panel is always 1920 units tall
            float col = Mathf.Min(panelW, h * 9f / 16f);
            bool changed = Mathf.Abs(_column.resolvedStyle.width - col) > 0.5f;
            if (changed) _column.style.width = col;
            ApplySafeArea(changed);
        }

        void ApplySafeArea(bool force)
        {
            var sa = SafeArea; var screen = ScreenSize;
            if (!force && sa == _safeArea && screen == _screen) return;
            _safeArea = sa; _screen = screen;
            float k = _root.layout.height / Mathf.Max(1, screen.y); if (float.IsNaN(k) || k <= 0) return;
            float colW = _column.style.width.value.value > 0 ? _column.style.width.value.value : _column.layout.width;
            float panelW = screen.x * k; bool fullWidth = Mathf.Abs(colW - panelW) < 2;
            float top = (screen.y - sa.yMax) * k, bottom = sa.yMin * k;
            // a cutout the safe area does not cover, away from the middle (a corner notch), pushes the top bar down;
            // one in the middle is a punch hole the FPS pill steps around (DodgeCutouts)
            foreach (var c in Cutouts)
            {
                float cx = c.center.x / screen.x;
                if (c.yMin * k < screen.y * k * 0.2f) continue;                 // a cutout at the bottom
                if (cx > 0.3f && cx < 0.7f) continue;
                top = Mathf.Max(top, (screen.y - c.yMin) * k);
            }
            // the insets pad the column; .safe keeps its own padding from the stylesheet inside them
            _column.style.paddingTop = top; _column.style.paddingBottom = bottom;
            _column.style.paddingLeft = fullWidth ? sa.xMin * k : 0; _column.style.paddingRight = fullWidth ? (screen.x - sa.xMax) * k : 0;
            _sheet.style.bottom = 120 + bottom;       // the sheet ends above the bottom corners (debug, version)
            _peekRow.style.bottom = 128 + bottom;
            _bottomInset = bottom;
            _sheet.RemoveFromClassList("dense");       // re-measured at the new size (FitSheet)
            _fps.schedule.Execute(DodgeCutouts);
        }

        /// <summary>Cutout in panel units (origin top-left), from a screen rectangle (origin bottom-left).</summary>
        Rect PanelRect(Rect screenRect)
        {
            var scr = ScreenSize; float k = _root.layout.height / Mathf.Max(1, scr.y);
            float offsetX = _root.worldBound.xMin + (_root.layout.width - scr.x * k) * 0.5f;
            return new Rect(offsetX + screenRect.xMin * k, _root.worldBound.yMin + (scr.y - screenRect.yMax) * k, screenRect.width * k, screenRect.height * k);
        }

        void DodgeCutouts()
        {
            var pill = _fps.worldBound; if (float.IsNaN(pill.width) || pill.width <= 0) return;
            var shift = _fps.resolvedStyle.translate; pill.position -= new Vector2(shift.x, shift.y);    // where it sits untranslated
            var dodge = Vector2.zero;
            foreach (var c in Cutouts)
            {
                var r = PanelRect(c); r.xMin -= 10; r.xMax += 10; r.yMax += 10;
                if (!r.Overlaps(pill)) continue;
                float right = r.xMax - pill.xMin, left = r.xMin - pill.xMax;      // shift needed to clear it on either side
                float roomRight = _menuBtn.worldBound.xMin - 12 - pill.xMax, roomLeft = pill.xMin - (_title.worldBound.xMax + 12);
                dodge = right <= roomRight ? new Vector2(right, 0) : -left <= roomLeft ? new Vector2(left, 0) : new Vector2(0, r.yMax - pill.yMin);
                break;
            }
            if (dodge == Vector2.zero) _fps.style.translate = StyleKeyword.Null; else _fps.style.translate = new Translate(dodge.x, dodge.y);
        }

        /// <summary>A sheet taller than the room between the top bar and the bottom corners tightens its rows.</summary>
        void FitSheet()
        {
            if (_sheet.style.display == DisplayStyle.None || _sheet.ClassListContains("dense")) return;
            if (_sheet.worldBound.yMin < _topBar.worldBound.yMax + 8) _sheet.AddToClassList("dense");
        }

        // ------------------------------------------------------------------ gestures
        void SheetDrag(VisualElement grab)
        {
            float startY = 0, dy = 0; bool moved = false;
            grab.RegisterCallback<PointerDownEvent>(e => { grab.CapturePointer(e.pointerId); startY = e.position.y; dy = 0; moved = false; _sheet.AddToClassList("dragging"); });
            grab.RegisterCallback<PointerMoveEvent>(e =>
            {
                if (!grab.HasPointerCapture(e.pointerId)) return;
                dy = Mathf.Max(0, e.position.y - startY); moved |= dy > 12; _sheet.style.translate = new Translate(0, dy);
            });
            grab.RegisterCallback<PointerUpEvent>(e =>
            {
                if (!grab.HasPointerCapture(e.pointerId)) return;
                grab.ReleasePointer(e.pointerId); _sheet.RemoveFromClassList("dragging"); _sheet.style.translate = StyleKeyword.Null;
                if (!moved || dy > DismissDistance) SetCollapsed(true);
            });
        }

        /// <summary>A swipe up that starts on the bottom edge (not on a control, the stick or a sheet): brings a hidden
        /// sheet back, or opens the menu during the intro or a round (as the menu button does).</summary>
        void EdgeSwipe()
        {
            int id = -1; float startY = 0;
            _root.RegisterCallback<PointerDownEvent>(e =>
            {
                id = -1;
                if (e.target is Button || Inside(e.target as VisualElement, _puppet) || Inside(e.target as VisualElement, _sheet) || Inside(e.target as VisualElement, _debug)) return;
                if (e.position.y < _column.worldBound.yMax - EdgeBand) return;
                id = e.pointerId; startY = e.position.y;
            });
            _root.RegisterCallback<PointerUpEvent>(e => { if (e.pointerId != id) return; id = -1; if (startY - e.position.y > SwipeDistance) SwipeUp(); });
        }

        static bool Inside(VisualElement e, VisualElement ancestor) { for (; e != null; e = e.parent) if (e == ancestor) return true; return false; }

        void SwipeUp()
        {
            if (director == null) return;
            if (_sheetKind != SheetKind.None) { if (_collapsed) SetCollapsed(false); return; }
            if (director.State == MatchState.Fight || director.State == MatchState.Intro) ToggleMenu();
        }

        void BuildReplayBar()
        {
            Text("REPLAY", "replay-label", _replayBar);
            Scrub(_replayBar, out _replayFill, out _replayHead, t => replay?.Seek(t));
            Btn("Skip", "replay-skip", () => replay?.Stop(), _replayBar);
            _replayBar.style.display = DisplayStyle.None;
        }

        /// <summary>A track with a fill and a head; dragging holds the replay and seeks, releasing lets it run on.</summary>
        VisualElement Scrub(VisualElement parent, out VisualElement fill, out VisualElement head, Action<float> seek, Action<float> begin = null)
        {
            var scrub = El("scrub", parent); var track = El("scrub-track", scrub); var f = El("scrub-fill", track); var h = El("scrub-head", track);
            float At(Vector2 world) => Mathf.Clamp01((world.x - track.worldBound.xMin) / Mathf.Max(1, track.worldBound.width));
            scrub.RegisterCallback<PointerDownEvent>(e =>
            {
                e.StopPropagation(); float t = At(e.position);
                if (!ReplaySystem.Active) { begin?.Invoke(t); return; }          // the results strip: play from there
                scrub.CapturePointer(e.pointerId);
                if (replay != null) { replay.Hold = true; seek(t); }
            });
            scrub.RegisterCallback<PointerMoveEvent>(e => { if (scrub.HasPointerCapture(e.pointerId) && ReplaySystem.Active) seek(At(e.position)); });
            scrub.RegisterCallback<PointerUpEvent>(e => { if (scrub.HasPointerCapture(e.pointerId)) scrub.ReleasePointer(e.pointerId); });
            scrub.RegisterCallback<PointerCaptureOutEvent>(_ => { if (replay != null) replay.Hold = false; });     // released, or hidden mid-drag
            fill = f; head = h; return scrub;
        }

        void BuildPuppet()
        {
            // the stick rests bottom-left and jumps to wherever a thumb lands in the zone
            _stickZone = El("stick-zone", _puppet);
            _stick = El("stick", _stickZone); _knob = El("stick-knob", _stick); _stick.pickingMode = PickingMode.Ignore; _knob.pickingMode = PickingMode.Ignore;
            _stickZone.RegisterCallback<PointerDownEvent>(e => { _stickZone.CapturePointer(e.pointerId); PlaceStick(e.localPosition); MoveStick(e.localPosition); });
            _stickZone.RegisterCallback<PointerMoveEvent>(e => { if (_stickZone.HasPointerCapture(e.pointerId)) MoveStick(e.localPosition); });
            _stickZone.RegisterCallback<PointerUpEvent>(e => { _stickZone.ReleasePointer(e.pointerId); RestStick(); });
            _stickZone.RegisterCallback<PointerCaptureOutEvent>(_ => RestStick());
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

        void PlaceStick(Vector2 local)
        {
            var z = _stickZone.contentRect;
            float x = Mathf.Clamp(local.x, 150, Mathf.Max(150, z.width - 150)), y = Mathf.Clamp(local.y, 150, Mathf.Max(150, z.height - 150));
            _stickCenter = new Vector2(x, y);
            _stick.style.left = x - 150; _stick.style.top = y - 150; _stick.style.bottom = StyleKeyword.Auto;
            _stick.AddToClassList("held");
        }

        void MoveStick(Vector2 local)
        {
            Vector2 v = Vector2.ClampMagnitude(local - _stickCenter, 110f);
            _knob.style.left = 87 + v.x; _knob.style.top = 87 + v.y;
            PuppetControl.Stick = new Vector2(v.x / 110f, -v.y / 110f);
        }

        void RestStick()
        {
            PuppetControl.Stick = Vector2.zero;
            _stick.style.left = StyleKeyword.Null; _stick.style.top = StyleKeyword.Null; _stick.style.bottom = StyleKeyword.Null;
            _knob.style.left = StyleKeyword.Null; _knob.style.top = StyleKeyword.Null;
            _stick.RemoveFromClassList("held");
        }

        // ------------------------------------------------------------------ events
        void OnState(MatchState s)
        {
            if (s == MatchState.Menu) ShowSheet(SheetKind.Menu);
            else if (s == MatchState.Result) ShowSheet(SheetKind.Result);
            else if (s != MatchState.Replay) ShowSheet(SheetKind.None);          // a replay keeps the result sheet behind it
            RebuildScoreboard();
            // a finger on the stick when the round ends must not steer the next one
            if (_stickZone != null) RestStick();
            PuppetControl.Side = 0;
            ApplyVisibility();
        }
        void OnRoundEnded(RoundResult r) { }      // the Result state change rebuilds the sheet
        void OnCaption(string text) { _caption.text = text; _caption.style.opacity = 1; _captionUntil = Time.unscaledTime + 3.5f; }
        void OnSettings() { if (_sheetKind == SheetKind.Menu) RebuildSheet(); }

        void ToggleMenu()
        {
            if (director == null) return;
            if (director.State == MatchState.Menu) { SetCollapsed(!_collapsed); return; }
            director.EnterMenu();                 // ends a running replay first (ReplaySystem.StopActive)
        }
        public void ToggleDebug() { bool on = _debug.style.display == DisplayStyle.None; _debug.style.display = on ? DisplayStyle.Flex : DisplayStyle.None; if (on) BuildDebugRows(); }

        void ShowSheet(SheetKind k)
        {
            if (k != _sheetKind) _collapsed = false;
            _sheetKind = k;
            if (k != SheetKind.None) RebuildSheet();
            ApplyVisibility();
        }

        void RebuildSheet()
        {
            _sheetBody.Clear(); _next = null; _resFill = _resHead = null;
            _sheet.RemoveFromClassList("dense");
            if (_sheetKind == SheetKind.Menu) BuildMenu(); else if (_sheetKind == SheetKind.Result) BuildResult();
        }

        void ApplyVisibility()
        {
            bool replaying = ReplaySystem.Active;
            bool sheet = _sheetKind != SheetKind.None && !replaying;
            _sheet.style.display = sheet && !_collapsed ? DisplayStyle.Flex : DisplayStyle.None;
            _peekRow.style.display = sheet && _collapsed ? DisplayStyle.Flex : DisplayStyle.None;
            _debug.style.visibility = _sheet.style.display == DisplayStyle.Flex ? Visibility.Hidden : Visibility.Visible;     // a sheet always wins
            _caption.style.visibility = _sheet.style.display == DisplayStyle.Flex ? Visibility.Hidden : Visibility.Visible;   // never under a sheet
        }

        // ------------------------------------------------------------------ menu
        void BuildMenu()
        {
            var tabs = El("tabs", _sheetBody);
            string[] names = { "Play", "Agents", "Match", "Show", "Connect" };
            for (int i = 0; i < names.Length; i++) { int t = i; var b = Btn(names[i], "tab", () => { _tab = t; RebuildSheet(); }, tabs); if (i == _tab) b.AddToClassList("selected"); }
            if (_tab == 0) BuildPlay(); else if (_tab == 1) BuildAgents(); else if (_tab == 2) BuildMatch(); else if (_tab == 3) BuildShow(); else BuildConnect();
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
                Chip(grid, "You play: steer a walker (stick or WASD)", director.PuppetRequested, v => { director.PuppetRequested = v; RebuildSheet(); }).AddToClassList("wide");
            }
            if (director.Status.Length > 0) Text(director.Status, "body", _sheetBody);
        }

        void BuildAgents()
        {
            Text("Agents", "h1", _sheetBody);
            Text(TickHint(), "body", _sheetBody);
            var grid = El("grid spaced", _sheetBody);
            for (int i = 0; i < director.roster.Length; i++)
            {
                var f = director.roster[i];
                var card = El("agent", grid); if (f.available && f.inGame) card.AddToClassList("on"); if (!f.available) card.AddToClassList("locked");
                var topRow = El("agent-top", card);
                // the seat palette while it is long enough; with more agents than colours (seven since Trump) evenly spaced hues, so none repeat
                int n = director.roster.Length;
                var av = El("avatar", topRow); av.style.backgroundColor = n <= DemoDirector.SeatColors.Length ? DemoDirector.SeatColors[i] : Color.HSVToRGB((float)i / n, 0.62f, 0.95f);
                Text(Initials(f.Badge), "", av);
                Text(f.Badge, "name", topRow);
                string meta = f.available ? $"Elo {Ladder.Rating(f.name):0} · {Ladder.Wins(f.name)}-{Ladder.Losses(f.name)} · {Ladder.Knockouts(f.name)} KOs" : "Locked: " + f.note;
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

        // The rules of a match.
        static readonly string[] TargetingLabels = { "Nearest", "Weakest", "Dogpile", "Grudge", "Vulture" };
        static readonly string[] TargetingHints =
        {
            "each brain pushes the nearest fighter", "each brain goes for the fighter easiest to push off", "everyone goes for the king of the hill",
            "brains go back for whoever pushed them last", "brains wait out a scrap, then pick off the weakest",
        };

        void BuildMatch()
        {
            Text("Match", "h2", _sheetBody);
            int rounds = GameSettings.RoundsPerMatch; int[] roundOpts = { 1, 3, 5, 0 };
            Labeled("Rounds", new[] { "1", "3", "5", "Endless" }, Array.IndexOf(roundOpts, rounds), i => GameSettings.RoundsPerMatch = roundOpts[i]);
            float[] paces = { 1.5f, 1f, 0.7f };
            Labeled("Sea", new[] { "Slow", "Normal", "Fast" }, Nearest(paces, GameSettings.SeaPace), i => GameSettings.SeaPace = paces[i]);
            int target = (int)GameSettings.Targeting;
            Text("Free-for-all targets: " + TargetingHints[Mathf.Clamp(target, 0, TargetingHints.Length - 1)], "hint", _sheetBody);
            Seg(TargetingLabels, target, i => GameSettings.Targeting = (Targeting)i).AddToClassList("seg-tight");
            Text("Rules", "h2", _sheetBody);
            var game = El("grid", _sheetBody);
            Chip(game, "Tale of the tape", GameSettings.Intro, v => GameSettings.Intro = v);
            Chip(game, "Stamina + hype", GameSettings.Stamina, v => GameSettings.Stamina = v);
            bool getupBrain = director.getupBrains != null && Array.Exists(director.getupBrains, b => b != null && b.policy != null);
            Chip(game, getupBrain ? "Get back up" : "Get back up (no brain yet)", GameSettings.GetUp, v => GameSettings.GetUp = v, getupBrain);
            bool planner = Array.Exists(director.roster, f => f.mlp != null);
            Chip(game, planner ? "Tactician (beta)" : "Tactician (no weights)", GameSettings.Tactician, v => GameSettings.Tactician = v, planner);
            Chip(game, "Hazards", GameSettings.Hazards, v => GameSettings.Hazards = v);
        }

        // How the match is shown and heard.
        void BuildShow()
        {
            Text("Picture", "h2", _sheetBody);
            Labeled("Camera", new[] { "Broadcast", "Classic" }, (int)GameSettings.Camera, i => GameSettings.Camera = (CameraStyle)i);
            // the tier Auto resolved to is on the telemetry line and in the debug panel (a longer label overflows the segment)
            Labeled("Quality", new[] { "Auto", "Low", "Medium", "High" }, (int)GameSettings.Quality, i => GameSettings.Quality = (Quality)i);
            var grid = El("grid", _sheetBody);
            Chip(grid, "Effects", GameSettings.Effects, v => GameSettings.Effects = v);
            Chip(grid, "Slow motion", GameSettings.SlowMotion, v => GameSettings.SlowMotion = v);
            Chip(grid, "Cut-aways", GameSettings.CutAways, v => GameSettings.CutAways = v);
            Chip(grid, "Instant replays", GameSettings.Replays, v => GameSettings.Replays = v);
            Chip(grid, "Win odds", GameSettings.WinOdds, v => GameSettings.WinOdds = v);
            Chip(grid, "Commentary", GameSettings.Commentary, v => GameSettings.Commentary = v);
            Text("Sound", "h2", _sheetBody);
            float[] vols = { 0f, 0.3f, 0.6f, 1f }, mix = { 0f, 0.4f, 0.7f, 1f };
            string[] levels = { "Off", "Low", "Mid", "High" };
            Labeled("Master", levels, Nearest(vols, GameSettings.Volume), i => GameSettings.Volume = vols[i]);
            Labeled("Hits", levels, Nearest(mix, GameSettings.SfxVolume), i => GameSettings.SfxVolume = mix[i]);
            Labeled("Ambience", levels, Nearest(mix, GameSettings.AmbienceVolume), i => GameSettings.AmbienceVolume = mix[i]);
            Labeled("Voice", levels, Nearest(mix, GameSettings.VoiceVolume), i => GameSettings.VoiceVolume = mix[i]);
        }

        // Network features: each one names what it needs when it is not configured, so nothing here fails silently.
        void BuildConnect()
        {
            Text("Viewers", "h2", _sheetBody);
            Field(GameSettings.TwitchChannel, "Twitch channel (votes, !hype)", v => GameSettings.TwitchChannel = v);
            Field(GameSettings.WeatherCity, "City for live summit weather", v => GameSettings.WeatherCity = v);
            if (weather != null && weather.Status.Length > 0) Text(weather.Status, "body", _sheetBody);
            Text("Commentary and telemetry", "h2", _sheetBody);
            var grid = El("grid", _sheetBody);
            Chip(grid, Commentator.ClaudeAvailable ? "Claude commentary" : "Claude commentary (set ANTHROPIC_API_KEY)", GameSettings.ClaudeCommentary, v => GameSettings.ClaudeCommentary = v, Commentator.ClaudeAvailable);
            Chip(grid, VoiceCaster.Available ? $"Voice ({VoiceCaster.Provider})" : "Voice (set AZURE_SPEECH_KEY or ELEVENLABS_API_KEY)", GameSettings.Voice, v => GameSettings.Voice = v, VoiceCaster.Available);
            bool sas = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("KOTH_AZURE_TABLE_SAS"));
            Chip(grid, sas ? "Upload telemetry (Azure)" : "Upload telemetry (set KOTH_AZURE_TABLE_SAS)", GameSettings.UploadTelemetry, v => GameSettings.UploadTelemetry = v, sas);
            foreach (var c in grid.Children()) c.AddToClassList("wide");
            Btn("Reset all settings", "secondary compact", () => { GameSettings.ResetAll(); RebuildSheet(); }, _sheetBody);
        }

        void Field(string value, string placeholder, Action<string> set)
        {
            var tf = new TextField { value = value, textEdition = { placeholder = placeholder } };
            tf.AddToClassList("field"); tf.AddToClassList("clip"); tf.RegisterCallback<FocusOutEvent>(_ => set(tf.value)); _sheetBody.Add(tf);
        }

        static int Nearest(float[] opts, float v) { int best = 0; for (int i = 1; i < opts.Length; i++) if (Mathf.Abs(opts[i] - v) < Mathf.Abs(opts[best] - v)) best = i; return best; }

        void Labeled(string label, string[] opts, int selected, Action<int> pick)
        {
            var row = El("row labeled", _sheetBody);
            Text(label, "body", row);
            Seg(opts, selected, pick, row);
        }

        VisualElement Seg(string[] opts, int selected, Action<int> pick, VisualElement parent = null)
        {
            var seg = El("seg", parent ?? _sheetBody);
            for (int i = 0; i < opts.Length; i++) { int k = i; var b = Btn(opts[i], "seg-btn", () => { pick(k); RebuildSheet(); }, seg); if (i == selected) b.AddToClassList("selected"); }
            return seg;
        }

        VisualElement Chip(VisualElement parent, string label, bool on, Action<bool> set, bool enabled = true)
        {
            var chip = El("chip", parent); if (on) chip.AddToClassList("on"); if (!enabled) chip.AddToClassList("locked");
            Text(label, "", chip);
            Text(on ? "ON" : "OFF", "state", chip);
            if (enabled) chip.RegisterCallback<ClickEvent>(_ => { set(!on); if (_sheetKind == SheetKind.Menu) RebuildSheet(); else BuildDebugRows(); });
            return chip;
        }

        // ------------------------------------------------------------------ results
        void BuildResult()
        {
            var r = director.LastResult; if (r.Agents == null) return;
            if (director.TournamentChampion.Length > 0 && !director.InTournament)
                Text("Tournament champion: " + director.TournamentChampion, "h1", _sheetBody);
            Text(r.Winner >= 0 ? "Winner: " + r.WinnerName : "No winner", "h1", _sheetBody);
            var stripe = El("winner-stripe", _sheetBody);
            stripe.style.backgroundColor = r.Winner >= 0 ? DemoDirector.SeatColors[r.Winner % DemoDirector.SeatColors.Length] : new Color(1, 1, 1, 0.2f);
            Text(char.ToUpper(r.How[0]) + r.How.Substring(1), "body", _sheetBody);

            // the knock-out replay, merged into the sheet: tap anywhere on the strip to play from there, drag to scrub
            if (replay != null && replay.Available)
            {
                var strip = El("res-scrub", _sheetBody);
                strip.Add(new Icon(Icon.Kind.Play) { name = "play" }); strip[0].AddToClassList("inline");
                Text("Replay", "", strip);
                Scrub(strip, out _resFill, out _resHead, t => replay.Seek(t), t => { if (!ReplaySystem.Active) { replay.Play(); replay.Seek(t); } });
                strip.RegisterCallback<ClickEvent>(_ => { if (!ReplaySystem.Active) replay.Play(); });
            }

            // units live in the header, so the name column gets the room
            var head = El("res-row res-head", _sheetBody);
            var nameHead = El("res-name grow", head); Text("Fighter", "", nameHead);
            Text("", "res-cell narrow", head);
            Text("Out s", "res-cell", head); Text(ledger != null ? "KO" : "Walk m", "res-cell narrow", head); Text("Top m/s", "res-cell", head);
            var eloHead = El("res-cell elo", head); Text("Elo", "", eloHead);
            for (int i = 0; i < r.Agents.Length; i++)
            {
                var row = El("res-row", _sheetBody);
                var nameBox = El("res-name grow clip", row);
                var dot = El("dot", nameBox); dot.style.backgroundColor = DemoDirector.SeatColors[i % DemoDirector.SeatColors.Length];
                Text(director.BadgeOf(r.Agents[i]) + (director.PuppetBody == i ? " (you)" : ""), "ellipsis", nameBox);
                string res = r.Winner < 0 ? "T" : r.Winner == i ? "W" : "L";
                Text(res, "res-cell narrow " + (res == "W" ? "win" : res == "L" ? "loss" : ""), row).tooltip = res == "W" ? "win" : res == "L" ? "loss" : "tie";
                float ko = log != null ? log.KnockedOutAt(i) : -1;
                Text(ko >= 0 ? $"{ko:0.0}" : "-", "res-cell", row);
                if (ledger != null)
                {
                    int kos = ledger.Knockouts(director.fighters[i]), ast = ledger.Assists(director.fighters[i]);
                    Text(kos + ast == 0 ? "-" : ast == 0 ? kos.ToString() : $"{kos}+{ast}", "res-cell narrow", row).tooltip = "knock-outs + assists";
                }
                else Text(log != null ? $"{log.PathMetres(i):0.0}" : "", "res-cell narrow", row);
                Text(log != null ? $"{log.TopSpeed(i):0.0}" : "", "res-cell", row);
                float dElo = log != null ? log.RatingDelta(i) : 0;
                float elo = Ladder.Rating(director.NameOf(r.Agents[i], "Robot"));
                var eloBox = El("res-cell elo", row); Text($"{elo:0}", "", eloBox);
                if (Mathf.Abs(dElo) >= 0.5f) Text($"{(dElo >= 0 ? "+" : "")}{dElo:0}", "delta " + (dElo >= 0 ? "up" : "down"), eloBox);
            }
            if (director.InTournament)
            {
                var b = El("bracket", _sheetBody);
                foreach (var m in director.Bracket)
                    Text($"{m.label}: {(m.a >= 0 ? director.BadgeOf(m.a) : "?")} vs {(m.b >= 0 ? director.BadgeOf(m.b) : "?")}" + (m.winner >= 0 ? $": {director.BadgeOf(m.winner)} wins" : ""), "body", b);
            }
            var row2 = El("row spaced", _sheetBody);
            _next = Btn("", "primary next", () => director.Continue(), row2);
            UpdateCountdown();
        }

        void UpdateCountdown()
        {
            string what = director.InTournament ? "Next match" : director.LastRoundOfMatch ? "Main menu" : "Next round";
            int left = Mathf.CeilToInt(director.ResultSecondsLeft);
            if (_next != null) _next.text = $"{what} ({left})";
            string peek = _sheetKind == SheetKind.Result ? $"Results · {left}" : "Menu";
            if (_peek.text != peek) _peek.text = peek;
        }

        // ------------------------------------------------------------------ live strip
        void RebuildScoreboard()
        {
            if (director == null) return;
            _scoreboard.Clear(); _badges.Clear(); _odds.Clear(); _oddsSegs.Clear(); _vitals.Clear(); _tags.Clear(); _pips.Clear();
            _ties.style.display = DisplayStyle.None;
            if (director.InMenu) return;
            for (int i = 0; i < director.Playing; i++)
            {
                int agent = director.AgentOf(i), seat = i;
                var b = El("badge", _scoreboard); if (director.PuppetBody == i) b.AddToClassList("you");
                var wrap = El("avatar-wrap", b);
                var av = El("avatar", wrap); av.style.backgroundColor = DemoDirector.SeatColors[i % DemoDirector.SeatColors.Length];
                Text(Initials(director.BadgeOf(agent)), "", av);
                _pips.Add(Text((director.AgentSeatedTwice(agent) ? director.BodyWins(i) : director.Wins(agent)).ToString(), "wins-pip", wrap));
                var tag = Text("", "badge-tag", wrap); tag.style.display = DisplayStyle.None; _tags.Add(tag);
                Text(director.BadgeOf(agent) + (director.PuppetBody == i ? " (you)" : ""), "badge-name ellipsis", b);
                var vit = new VitalsBar(); vit.style.display = bio != null ? DisplayStyle.Flex : DisplayStyle.None; b.Add(vit); _vitals.Add(vit);
                b.tooltip = director.NameOf(agent, "Robot");
                b.RegisterCallback<ClickEvent>(_ => Hype(seat));         // a tap on a badge cheers that fighter on (Stamina + hype)
                _badges.Add(b);
                var seg = El("odds-seg", _odds); seg.style.backgroundColor = DemoDirector.SeatColors[i % DemoDirector.SeatColors.Length]; seg.style.width = Length.Percent(100f / director.Playing);
                _oddsSegs.Add(seg);
            }
            if (director.Draws > 0) { _ties.text = "Ties " + director.Draws; _ties.style.display = DisplayStyle.Flex; }
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
                _fps.text = !GameSettings.Telemetry ? $"{_fpsAvg:0} FPS" : quality != null
                    // a second, smaller line: one long line spread over the title (physics and brain times are in the debug panel)
                    ? $"{quality.Fps:0} FPS\n<size=20>p95 {quality.FrameP95Ms:0} ms · {quality.Current}</size>"
                    : $"{_fpsAvg:0} FPS · phys {director.StepMs:0.00} · brain {(director.fighters.Length > 0 ? director.fighters[0].LastInferenceMs : 0):0.00} ms";
                if (DebugShown && _debugTab == 0) _debugText.text = DebugText();
                UpdateCountdown();
            }
            if (_pressAt > 0 && Time.unscaledTime - _pressAt >= HoldSeconds) { _pressAt = -1; ToggleDebug(); }
            _roundInfo.style.display = live ? DisplayStyle.Flex : DisplayStyle.None;
            if (live)
            {
                _clock.text = ReplaySystem.Active ? "REPLAY" : $"{director.RoundTime:0.0} s";
                var sea = director.sea;
                _seaFill.style.width = Length.Percent(sea != null ? Mathf.Clamp01((sea.Level + sea.startDepth) / sea.startDepth) * 100f : 0f);
                for (int i = 0; i < _badges.Count; i++) _badges[i].EnableInClassList("out", director.IsOut(i));
            }
            bool replaying = ReplaySystem.Active;
            _replayBar.style.display = replaying ? DisplayStyle.Flex : DisplayStyle.None;
            if (replaying) SetScrub(_replayFill, _replayHead, replay != null ? replay.Progress : 0f);
            if (_resFill != null) SetScrub(_resFill, _resHead, replaying && replay != null ? replay.Progress : 0f);
            if (fight || director.State == MatchState.Intro) UpdateBadges();
            for (int k = _feedRows.Count - 1; k >= 0; k--) if (Time.unscaledTime > _feedRows[k].until) { _feedRows[k].row.RemoveFromHierarchy(); _feedRows.RemoveAt(k); }
            UpdateIntro();
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
            if (director.Status != _lastStatus) { _lastStatus = director.Status; _statusUntil = Time.unscaledTime + 4f; }
            _status.text = Time.unscaledTime < _statusUntil ? _lastStatus : "";
            bool puppet = fight && director.PuppetBody >= 0;
            if (puppet != PuppetShown) { _puppet.style.display = puppet ? DisplayStyle.Flex : DisplayStyle.None; if (!puppet) RestStick(); }
            // with the stick on screen the panel hangs under the live strip (badges, clock, odds, hazard) instead of standing on the bottom corners
            if (puppet) { _debug.style.top = _feed.worldBound.yMin - _column.worldBound.yMin + 8; _debug.style.bottom = StyleKeyword.Auto; }
            else { _debug.style.top = StyleKeyword.Null; _debug.style.bottom = 128 + _bottomInset; }
            ApplyVisibility();
            TwoFingerTap();
        }

        static void SetScrub(VisualElement fill, VisualElement head, float t)
        {
            var pct = Length.Percent(Mathf.Clamp01(t) * 100f);
            if (fill.style.width != pct) fill.style.width = pct;
            if (head.style.left != pct) head.style.left = pct;
        }

        // ------------------------------------------------------------------ vitals, tags, feed, intro
        void UpdateBadges()
        {
            for (int i = 0; i < _badges.Count && i < director.Playing; i++)
            {
                var r = director.fighters[i]; var v = bio != null ? bio.Of(r) : null; bool outNow = director.IsOut(i);
                if (v != null && i < _vitals.Count) _vitals[i].Set(v, stamina != null && GameSettings.Stamina ? stamina.Level(r) : -1f, outNow);
                bool down = director.IsDown(i), hyped = stamina != null && stamina.IsHyped(r) && !outNow, edge = v != null && v.doomed && !outNow;
                _badges[i].EnableInClassList("down", down); _badges[i].EnableInClassList("edge", edge && !down); _badges[i].EnableInClassList("hyped", hyped && !down);
                string tag = down ? $"DOWN {director.DownLeft(i):0}" : edge ? "EDGE" : hyped ? "HYPE" : !outNow && r.tactic != Tactic.Press && director.State == MatchState.Fight ? TacticLabel(r.tactic) : "";
                var t = _tags[i]; if (t.text != tag) t.text = tag; t.style.display = tag.Length > 0 ? DisplayStyle.Flex : DisplayStyle.None;
            }
        }

        static string TacticLabel(Tactic t) => t switch { Tactic.FlankLeft => "flank L", Tactic.FlankRight => "flank R", Tactic.Regroup => "regroup", Tactic.Lurk => "lurk", _ => "" };

        void Hype(int seat)
        {
            if (stamina == null || director.State != MatchState.Fight || seat >= director.Playing) return;
            if (stamina.TryHype(director.fighters[seat])) MatchEvents.RaiseCaption($"The crowd fires up {director.BadgeOf(director.AgentOf(seat))}!");
        }

        void OnCredited(PolicyRunner victim, PolicyRunner by, PolicyRunner assist)
        {
            if (director == null || director.statsRounds > 0) return;
            int v = Array.IndexOf(director.fighters, victim), b = Array.IndexOf(director.fighters, by), a = Array.IndexOf(director.fighters, assist);
            if (v < 0) return;
            var row = El("feed-row", _feed); row.pickingMode = PickingMode.Ignore;
            void Who(int seat) { var d = El("dot", row); d.style.backgroundColor = DemoDirector.SeatColors[seat % DemoDirector.SeatColors.Length]; Text(director.BadgeOf(director.AgentOf(seat)), "", row); }
            if (b >= 0) { Who(b); Text("pushes off", "feed-verb", row); }
            Who(v); if (b < 0) Text("falls off", "feed-verb", row);
            if (a >= 0) { Text("+", "feed-verb", row); Who(a); }
            _feedRows.Add((row, Time.unscaledTime + 6f));
            while (_feedRows.Count > 4) { _feedRows[0].row.RemoveFromHierarchy(); _feedRows.RemoveAt(0); }
        }

        // The tale-of-the-tape card: the field during the crane shot, then one fighter at a time.
        void UpdateIntro()
        {
            bool on = director.State == MatchState.Intro;
            _intro.style.display = on ? DisplayStyle.Flex : DisplayStyle.None;
            if (!on) { _introShown = -2; return; }
            int f = Mathf.Min(director.IntroFeatured, director.Playing - 1);
            if (f == _introShown) return;
            _introShown = f; _intro.Clear();
            if (f < 0)
            {
                Text("Tonight on the summit", "intro-kicker", _intro);
                var sb = new System.Text.StringBuilder();
                for (int i = 0; i < director.Playing; i++) sb.Append(i == 0 ? "" : i == director.Playing - 1 ? " vs " : ", ").Append(director.BadgeOf(director.AgentOf(i)));
                Text(sb.ToString(), "intro-name", _intro);
                if (weather != null && weather.Summary.Length > 0) Text(weather.Summary, "intro-weather", _intro);
            }
            else
            {
                var r = director.fighters[f]; int agent = director.AgentOf(f); var entry = agent >= 0 && agent < director.roster.Length ? director.roster[agent] : null;
                var stripe = El("intro-stripe", _intro); stripe.style.backgroundColor = DemoDirector.SeatColors[f % DemoDirector.SeatColors.Length];
                Text($"Fighter {f + 1} of {director.Playing}", "intro-kicker", _intro);
                Text(director.BadgeOf(agent), "intro-name", _intro);
                if (entry != null && entry.name != entry.Badge) Text(entry.name, "body", _intro);
                var grid = El("intro-grid", _intro);
                TaleOfTheTape.Body(r, out float kg, out float strongest);
                string name = director.NameOf(agent, "Robot");
                void Tile(string k, string v) { var t = El("intro-tile", grid); Text(v, "intro-value", t); Text(k, "intro-key", t); }
                Tile("mass", $"{kg:0} kg"); Tile("strongest joint", $"{strongest:0} N m"); Tile("Elo", $"{Ladder.Rating(name):0}");
                Tile("record", $"{Ladder.Wins(name)}-{Ladder.Losses(name)}"); Tile("knock-outs", Ladder.Knockouts(name).ToString()); Tile("brain", TaleOfTheTape.Brain(entry));
            }
            var row = El("row spaced", _intro);
            Btn("Skip intro", "secondary", () => director.SkipIntro(), row);
        }

        void TwoFingerTap()
        {
            var ts = Touchscreen.current; if (ts == null) return;
            int down = 0, began = 0;
            foreach (var t in ts.touches) { if (t.isInProgress) down++; if (t.press.wasPressedThisFrame) began++; }
            if (down == 2 && began > 0) ToggleDebug();
        }

        // ------------------------------------------------------------------ debug panel: Perf | View | Hazards
        void BuildDebugRows()
        {
            _debugTabs.Clear(); _debugBody.Clear();
            string[] names = { "Perf", "View", "Hazards" };
            for (int i = 0; i < names.Length; i++) { int t = i; var b = Btn(names[i], "tab", () => { _debugTab = t; BuildDebugRows(); }, _debugTabs); if (i == _debugTab) b.AddToClassList("selected"); }
            if (_debugTab == 0) { _debugBody.Add(_debugText); _debugText.text = DebugText(); return; }
            if (_debugTab == 1)
            {
                var seg = El("seg", _debugBody); string[] modes = { "None", "Strain", "Fatigue", "Colliders" };
                for (int i = 0; i < modes.Length; i++) { int k = i; var b = Btn(modes[i], "seg-btn", () => { GameSettings.Overlay = (Overlay)k; BuildDebugRows(); }, seg); if ((int)GameSettings.Overlay == i) b.AddToClassList("selected"); }
                var grid = El("grid", _debugBody);
                Chip(grid, "Intent arrows", GameSettings.IntentArrows, v => GameSettings.IntentArrows = v);
                Chip(grid, "Telemetry HUD", GameSettings.Telemetry, v => GameSettings.Telemetry = v);
                return;
            }
            var hz = El("seg wrap", _debugBody);
            foreach (Hazard h in Enum.GetValues(typeof(Hazard))) { var hh = h; Btn(h.ToString(), "seg-btn", () => hazards?.Trigger(hh), hz); }
            var r = El("seg", _debugBody);
            Btn("Replay", "seg-btn", () => replay?.Play(), r);
            Btn("Restart round", "seg-btn", () => { if (director.State == MatchState.Fight) director.NewRound(); }, r);
        }

        string DebugText()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append($"state {director.State} · round {director.RoundTime:0.0} s · out radius {director.OutRadiusNow:0.00} m\n");
            sb.Append($"physics tick {director.StepMs:0.000} ms · ");
            for (int i = 0; i < director.Playing && i < director.fighters.Length; i++) sb.Append($"{director.fighters[i].robotPrefix} {director.fighters[i].LastInferenceMs:0.00} ms  ");
            sb.Append($"\nGC {GC.CollectionCount(0)} gen0 · telemetry {MatchLog.Folder}");
            if (quality != null) sb.Append($"\nframe {quality.FrameMs:0.0} ms (p95 {quality.FrameP95Ms:0.0}) · {quality.GcKbPerFrame:0.00} KB/frame · quality {quality.Label}, {quality.Downgrades} auto steps");
            if (synth != null) sb.Append($"\naudio: wind {synth.WindRms:0.000} surf {synth.SurfRms:0.000} crowd {synth.CrowdRms:0.000} · slide {synth.Slide:0.00} grind {synth.Grind:0.00} · {synth.Clanks} clanks {synth.Smacks} smacks {synth.Grunts} grunts");
            if (warp != null) sb.Append($" · {warp.SlowMos} slow-mos {warp.HitStops} hit stops");
            if (chat != null) sb.Append($"\nchat: {chat.Status}");
            if (tactician != null && GameSettings.Tactician) sb.Append($"\ntactician: {tactician.Decisions} decisions, last batch {tactician.LastBatchMs:0} ms");
            if (weather != null && weather.Status.Length > 0) sb.Append($"\nweather: {weather.Status}");
            return sb.ToString();
        }

        // ------------------------------------------------------------------ layout audit (tests, captures)
        /// <summary>
        /// What does not fit on this screen: an element outside the column, a label or button whose text is wider than
        /// its box (outside "clip" and "ellipsis" containers, which cut on purpose), a sheet that reaches the top bar
        /// or the bottom corners, badges on more than one row, an intro card under the strip, the FPS pill on a cutout.
        /// Empty = the HUD fits.
        /// </summary>
        public List<string> LayoutProblems()
        {
            var list = new List<string>(); var col = _column.worldBound;
            void Walk(VisualElement e, bool clipped)
            {
                if (e.resolvedStyle.display == DisplayStyle.None || e.resolvedStyle.visibility == Visibility.Hidden) return;
                var b = e.worldBound;
                if (!clipped && b.width > 0.5f && b.height > 0.5f && (b.xMin < col.xMin - 1 || b.xMax > col.xMax + 1 || b.yMin < col.yMin - 1 || b.yMax > col.yMax + 1))
                    list.Add($"{Describe(e)} outside the column ({b.xMin:0},{b.yMin:0} {b.width:0}x{b.height:0})");
                if (!clipped && e is TextElement t && !(e is TextField) && !string.IsNullOrEmpty(t.text) && !e.ClassListContains("ellipsis") && t.resolvedStyle.whiteSpace == WhiteSpace.NoWrap)
                {
                    var need = t.MeasureTextSize(t.text, 0, VisualElement.MeasureMode.Undefined, 0, VisualElement.MeasureMode.Undefined);
                    if (need.x > t.contentRect.width + 2) list.Add($"{Describe(e)} needs {need.x:0} units, has {t.contentRect.width:0}");
                }
                else if (!clipped && e is TextElement w && !(e is TextField) && !string.IsNullOrEmpty(w.text) && w.contentRect.width > 0)
                {
                    // wrapped text: taller than its box spills out of the background
                    var need = w.MeasureTextSize(w.text, w.contentRect.width + 2, VisualElement.MeasureMode.Exactly, 0, VisualElement.MeasureMode.Undefined);      // slack: a label sized to its text can wrap on rounding alone
                    if (need.y > w.contentRect.height + 2) list.Add($"{Describe(e)} needs {need.y:0} units of height, has {w.contentRect.height:0}");
                }
                bool clip = clipped || e is TextField || e.ClassListContains("clip") || e.ClassListContains("ellipsis");
                foreach (var c in e.Children()) Walk(c, clip);
            }
            Walk(_safe, false); Walk(_sheet, false); Walk(_debug, false); Walk(_peekRow, false);
            if (SheetShown)
            {
                var s = _sheet.worldBound;
                if (s.yMin < _topBar.worldBound.yMax - 1) list.Add($"sheet reaches the top bar ({s.yMin:0} < {_topBar.worldBound.yMax:0})");
                if (s.yMax > _bottomBar.worldBound.yMin + 1) list.Add($"sheet covers the bottom corners ({s.yMax:0} > {_bottomBar.worldBound.yMin:0})");
            }
            if (DebugShown && _debug.resolvedStyle.visibility == Visibility.Visible)
            {
                var g = _debug.worldBound;
                if (g.yMax > _bottomBar.worldBound.yMin + 1) list.Add("debug panel covers the bottom corners");
                if (g.yMin < _topBar.worldBound.yMax - 1) list.Add("debug panel reaches the top bar");
                if (PuppetShown && g.Overlaps(_stickZone.worldBound)) list.Add("debug panel covers the stick");
                if (PuppetShown && _roundInfo.resolvedStyle.display == DisplayStyle.Flex && g.Overlaps(_roundInfo.worldBound)) list.Add("debug panel covers the round clock");
            }
            if (_peekRow.resolvedStyle.display == DisplayStyle.Flex && _peek.worldBound.yMax > _bottomBar.worldBound.yMin + 1) list.Add("peek pill covers the bottom corners");
            if (_badges.Count > 1)
                foreach (var b in _badges) if (Mathf.Abs(b.worldBound.yMin - _badges[0].worldBound.yMin) > 1) { list.Add("badges wrap onto a second row"); break; }
            if (_intro.resolvedStyle.display == DisplayStyle.Flex && _intro.worldBound.yMin < _scoreboard.worldBound.yMax - 1) list.Add("intro card overlaps the badges");
            foreach (var c in Cutouts)
            {
                var r = PanelRect(c);
                if (r.Overlaps(_fps.worldBound)) list.Add("FPS pill on a cutout");
                if (r.Overlaps(_title.worldBound)) list.Add("title on a cutout");
                if (r.Overlaps(_menuBtn.worldBound)) list.Add("menu button on a cutout");
            }
            return list;
        }

        static string Describe(VisualElement e)
        {
            if (e is TextElement t && !string.IsNullOrEmpty(t.text)) return $"\"{t.text.Split('\n')[0]}\"";
            string classes = string.Join(".", e.GetClasses());
            return classes.Length > 0 ? "." + classes : e.GetType().Name;
        }
    }

    /// <summary>
    /// A badge's vitals strip, drawn with Painter2D: balance bar (Biometrics stability; green, amber, red), the last 3 s
    /// of joint strain as a sparkline (white, red above 60 % of the force limits), and stamina as a thin gold bar when
    /// stamina is on. Repainted only when a value changed.
    /// </summary>
    public class VitalsBar : VisualElement
    {
        float _stab = 1f, _stamina = -1f; bool _out; int _head = -1; Biometrics.Vitals _v;
        static readonly Color Good = new(0.43f, 0.84f, 0.47f), Warn = new(1f, 0.77f, 0.25f), Bad = new(0.94f, 0.38f, 0.34f), Track = new(1f, 1f, 1f, 0.14f);

        public VitalsBar() { AddToClassList("vitals"); pickingMode = PickingMode.Ignore; generateVisualContent += Draw; }

        public void Set(Biometrics.Vitals v, float stamina, bool isOut)
        {
            if (v == _v && v.head == _head && Mathf.Abs(v.stability - _stab) < 0.01f && Mathf.Abs(stamina - _stamina) < 0.01f && isOut == _out) return;
            _v = v; _head = v.head; _stab = v.stability; _stamina = stamina; _out = isOut; MarkDirtyRepaint();
        }

        void Draw(MeshGenerationContext ctx)
        {
            var p = ctx.painter2D; var r = contentRect; if (r.width < 4 || _v == null) return;
            float w = r.width, barH = r.height * 0.32f;
            Bar(p, 0, w, barH, Track); if (!_out) Bar(p, 0, w * Mathf.Clamp01(_stab), barH, _stab > 0.6f ? Good : _stab > 0.3f ? Warn : Bad);
            // strain sparkline, oldest sample on the left
            float y0 = barH + r.height * 0.08f, sh = r.height * (_stamina >= 0 ? 0.42f : 0.6f);
            p.lineWidth = 2f; p.lineJoin = LineJoin.Round; p.strokeColor = _out ? Track : new Color(1f, 1f, 1f, 0.8f);
            p.BeginPath(); float peak = 0;
            for (int k = 0; k < Biometrics.History; k++)
            {
                float s = Mathf.Clamp01(_v.strainHistory[(_v.head + k) % Biometrics.History] / 0.8f); peak = Mathf.Max(peak, s);
                var pt = new Vector2(w * k / (Biometrics.History - 1), y0 + sh * (1f - s));
                if (k == 0) p.MoveTo(pt); else p.LineTo(pt);
            }
            if (peak > 0.75f && !_out) p.strokeColor = Bad;
            p.Stroke();
            if (_stamina >= 0) { float y = r.height - 4; Bar(p, 0, w, 4, Track, y); Bar(p, 0, w * Mathf.Clamp01(_stamina), 4, Warn, y); }
        }

        static void Bar(Painter2D p, float x, float w, float h, Color c, float y = 0)
        {
            if (w <= 0) return;
            p.fillColor = c; p.BeginPath(); p.MoveTo(new Vector2(x, y)); p.LineTo(new Vector2(x + w, y)); p.LineTo(new Vector2(x + w, y + h)); p.LineTo(new Vector2(x, y + h)); p.ClosePath(); p.Fill();
        }
    }

    /// <summary>Small vector icons drawn with Painter2D, so no font glyph or texture is needed.</summary>
    public class Icon : VisualElement
    {
        public enum Kind { Menu, Sliders, Play }
        readonly Kind _kind;
        public Icon(Kind kind) { _kind = kind; AddToClassList("icon"); pickingMode = PickingMode.Ignore; generateVisualContent += Draw; }

        void Draw(MeshGenerationContext ctx)
        {
            var p = ctx.painter2D; var r = contentRect; float s = Mathf.Min(r.width, r.height); var c = r.center; float u = s * 0.18f;
            p.strokeColor = new Color(0.96f, 0.97f, 0.98f); p.lineWidth = Mathf.Max(2f, s * 0.07f); p.lineCap = LineCap.Round;
            if (_kind == Kind.Menu)
                for (int i = -1; i <= 1; i++) { p.BeginPath(); p.MoveTo(new Vector2(c.x - u * 1.4f, c.y + i * u)); p.LineTo(new Vector2(c.x + u * 1.4f, c.y + i * u)); p.Stroke(); }
            else if (_kind == Kind.Play)
            {
                p.fillColor = new Color(1f, 0.77f, 0.25f); p.BeginPath();
                p.MoveTo(new Vector2(c.x - s * 0.3f, c.y - s * 0.36f)); p.LineTo(new Vector2(c.x + s * 0.38f, c.y)); p.LineTo(new Vector2(c.x - s * 0.3f, c.y + s * 0.36f)); p.ClosePath(); p.Fill();
            }
            else
                for (int i = -1; i <= 1; i += 2)
                {
                    p.BeginPath(); p.MoveTo(new Vector2(c.x - u * 1.4f, c.y + i * u * 0.8f)); p.LineTo(new Vector2(c.x + u * 1.4f, c.y + i * u * 0.8f)); p.Stroke();
                    p.fillColor = new Color(1f, 0.77f, 0.25f); p.BeginPath(); p.Arc(new Vector2(c.x + i * u * 0.6f, c.y + i * u * 0.8f), u * 0.42f, 0, 360); p.Fill();
                }
        }
    }
}
