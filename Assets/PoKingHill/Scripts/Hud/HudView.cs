using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;

namespace PoKingHill
{
    /// <summary>
    /// The whole UI in UI Toolkit, one 9:16 column that never scrolls (styles in UI/Hud.uss). Split by concern:
    /// this file (layout, gestures, live strip, intro), HudView.Menu.cs (menu, settings, results), HudView.Debug.cs
    /// (debug panel, layout audit), HudWidgets.cs (vitals strip, icons).
    ///   corners   TL title · TR menu · BR version; with GameSettings.DevTools also TC FPS (hold it for the debug
    ///             panel; it steps aside from a punch-hole camera, Screen.cutouts) and BL debug
    ///   live      one row of fighter badges (seat-coloured initials, wins pip, status tag, vitals strip), round
    ///             clock, sea bar and ties, knock-out feed, replay bar with a draggable playhead, puppet stick that
    ///             appears under the thumb; with developer tools also the win-odds bar and the caption line
    ///   intro     tale of the tape: one card per fighter (mass, strongest joint, Elo, record, knock-outs, brain)
    ///   sheets    Menu (tabs Play | Agents | Settings) and Results (winner, replay strip, per-fighter telemetry,
    ///             next). A sheet swipes down (or its grabber is tapped) to a peek pill; a swipe up from the bottom
    ///             edge brings it back, or opens the menu during a round. A sheet taller than the room between the
    ///             corners turns dense.
    ///   debug     developer tools only: two-finger tap, holding the FPS pill or the BL button: tabs Perf | View | Tools
    /// Safe-area padding follows Screen.safeArea. State comes from MatchDirector and MatchEvents; nothing here touches
    /// the simulation except through the director's and the replay's public methods. Tests and captures can lay the HUD
    /// out for another screen (SimulateScreen) and list what does not fit (LayoutProblems).
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public partial class HudView : MonoBehaviour
    {
        public MatchDirector director;
        public WinOdds odds;
        public ReplaySystem replay;
        public MatchLog log;
        public Biometrics bio;
        public ContactLedger ledger;
        public QualityTier quality;               // optional: the telemetry line and the debug panel's frame times
        public ImpactSynth synth;                 // optional: audio levels in the debug panel
        public TimeWarp warp;                     // optional: slow-motion counts in the debug panel
        public StyleSheet style;

        const float EdgeBand = 220f, SwipeDistance = 180f, DismissDistance = 140f, HoldSeconds = 0.5f;

        UIDocument _doc;
        VisualElement _root, _column, _safe, _topBar, _bottomBar, _menuBtn, _scoreboard, _roundInfo, _seaFill, _odds, _puppet, _debug, _debugTabs, _debugBody,
                      _sheet, _sheetBody, _knob, _stick, _stickZone, _feed, _intro, _replayBar, _replayFill, _replayHead, _peekRow, _resFill, _resHead;
        Label _title, _fps, _clock, _ties, _caption, _status, _oddsLabel, _debugText;
        Button _next, _peek, _debugBtn;
        readonly List<VisualElement> _badges = new(); readonly List<VisualElement> _oddsSegs = new();
        readonly List<VitalsBar> _vitals = new(); readonly List<Label> _tags = new(); readonly List<Label> _pips = new();
        readonly List<(VisualElement row, float until)> _feedRows = new(); int _introShown = -2;
        enum SheetKind { None, Menu, Result }
        SheetKind _sheetKind; int _tab, _section, _debugTab; bool _collapsed;
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
        public const int TabCount = 3, SectionCount = 4;
        /// <summary>The menu tab on show: 0 Play, 1 Agents, 2 Settings.</summary>
        public int Tab { get => _tab; set { _tab = Mathf.Clamp(value, 0, TabCount - 1); if (_sheetKind == SheetKind.Menu) RebuildSheet(); } }
        /// <summary>The settings section on show: 0 Match, 1 Picture, 2 Sound, 3 Advanced.</summary>
        public int Section { get => _section; set { _section = Mathf.Clamp(value, 0, SectionCount - 1); if (_sheetKind == SheetKind.Menu) RebuildSheet(); } }
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
            _safeArea = default; UpdateFps(); FitColumn();
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
            _fps.RegisterCallback<GeometryChangedEvent>(_ => DodgeCutouts());       // a wider text (telemetry line) may reach the cutout again
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
            _feed = El("feed", _safe); _feed.pickingMode = PickingMode.Ignore;
            El("spacer", _safe);
            _intro = El("intro", _safe); _intro.style.display = DisplayStyle.None;
            _caption = Text("", "caption", _safe); _caption.style.opacity = 0;
            _replayBar = El("replay-bar", _safe); BuildReplayBar();
            _puppet = El("puppet", _safe); BuildPuppet();

            _bottomBar = El("bottom-bar", _safe);
            _debugBtn = Btn("", "icon-btn", ToggleDebug, _bottomBar); _debugBtn.Add(new Icon(Icon.Kind.Sliders)); _debugBtn.tooltip = "Debug (or tap with two fingers, or hold the FPS pill)";
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
            ApplyDevTools();
        }

        /// <summary>The FPS pill and the debug button are developer tools (GameSettings.DevTools); the version stays.
        /// Hidden with visibility, not display, so the top bar and the bottom corners keep their layout.</summary>
        void ApplyDevTools()
        {
            bool on = GameSettings.DevTools, turnedOff = _devTools && !on; _devTools = on;
            var v = on ? Visibility.Visible : Visibility.Hidden;
            _fps.style.visibility = v; _debugBtn.style.visibility = v;
            if (turnedOff && DebugShown) ToggleDebug();
        }
        bool _devTools;

        void UpdateFps()
        {
            if (!_devTools || _fps == null || director == null) return;
            string tier = quality != null && quality.Current == Quality.Medium ? "Med" : quality != null ? quality.Current.ToString() : "";
            _fps.text = !GameSettings.TelemetryHud ? $"{_fpsAvg:0} FPS" : quality != null
                // a second, smaller line: one long line spread over the title (physics and brain times are in the debug panel)
                ? $"{quality.Fps:0} FPS\n<size=20>p95 {quality.FrameP95Ms:0} ms · {tier}</size>"
                : $"{_fpsAvg:0} FPS · phys {director.StepMs:0.00} · brain {(director.bodies.Length > 0 ? director.bodies[0].LastInferenceMs : 0):0.00} ms";
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
        void OnSettings() { ApplyDevTools(); UpdateFps(); if (_sheetKind == SheetKind.Menu) RebuildSheet(); }

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

        // ------------------------------------------------------------------ live strip
        void RebuildScoreboard()
        {
            if (director == null) return;
            _scoreboard.Clear(); _badges.Clear(); _odds.Clear(); _oddsSegs.Clear(); _vitals.Clear(); _tags.Clear(); _pips.Clear();
            _ties.style.display = DisplayStyle.None;
            if (director.InMenu) return;
            for (int i = 0; i < director.Playing; i++)
            {
                int agent = director.AgentOf(i);
                var b = El("badge", _scoreboard); if (director.PuppetBody == i) b.AddToClassList("you");
                var wrap = El("avatar-wrap", b);
                var av = El("avatar", wrap); av.style.backgroundColor = MatchDirector.SeatColors[i % MatchDirector.SeatColors.Length];
                Text(Initials(director.BadgeOf(agent)), "", av);
                _pips.Add(Text((director.AgentSeatedTwice(agent) ? director.BodyWins(i) : director.Wins(agent)).ToString(), "wins-pip", wrap));
                var tag = Text("", "badge-tag", wrap); tag.style.display = DisplayStyle.None; _tags.Add(tag);
                Text(director.BadgeOf(agent) + (director.PuppetBody == i ? " (you)" : ""), "badge-name ellipsis", b);
                var vit = new VitalsBar(); vit.style.display = bio != null ? DisplayStyle.Flex : DisplayStyle.None; b.Add(vit); _vitals.Add(vit);
                b.tooltip = director.NameOf(agent, "Robot");
                _badges.Add(b);
                var seg = El("odds-seg", _odds); seg.style.backgroundColor = MatchDirector.SeatColors[i % MatchDirector.SeatColors.Length]; seg.style.width = Length.Percent(100f / director.Playing);
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
                if (!_sim) UpdateFps();
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
            if (_captionUntil > 0 && Time.unscaledTime > _captionUntil) { _captionUntil = 0; _caption.style.opacity = 0; }
            if (director.Status != _lastStatus) { _lastStatus = director.Status; _statusUntil = Time.unscaledTime + 4f; }
            _status.text = Time.unscaledTime < _statusUntil ? _lastStatus : "";
            bool puppet = fight && director.PuppetBody >= 0;
            if (puppet != PuppetShown) { _puppet.style.display = puppet ? DisplayStyle.Flex : DisplayStyle.None; if (!puppet) RestStick(); }
            // with the stick on screen the panel hangs under the live strip (badges, clock, odds) instead of standing on the bottom corners
            if (puppet) { _debug.style.top = _feed.worldBound.yMin - _column.worldBound.yMin + 8; _debug.style.bottom = StyleKeyword.Auto; }
            else { _debug.style.top = StyleKeyword.Null; _debug.style.bottom = 128 + _bottomInset; }
            ApplyVisibility();
            if (_devTools) TwoFingerTap();
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
                var r = director.bodies[i]; var v = bio != null ? bio.Of(r) : null; bool outNow = director.IsOut(i);
                if (v != null && i < _vitals.Count) _vitals[i].Set(v, outNow);
                bool down = director.IsDown(i), edge = v != null && v.doomed && !outNow;
                _badges[i].EnableInClassList("down", down); _badges[i].EnableInClassList("edge", edge && !down);
                string tag = down ? $"DOWN {director.DownLeft(i):0}" : edge ? "EDGE" : "";
                var t = _tags[i]; if (t.text != tag) t.text = tag; t.style.display = tag.Length > 0 ? DisplayStyle.Flex : DisplayStyle.None;
            }
        }

        void OnCredited(PolicyRunner victim, PolicyRunner by, PolicyRunner assist)
        {
            if (director == null || director.statsRounds > 0) return;
            int v = Array.IndexOf(director.bodies, victim), b = Array.IndexOf(director.bodies, by), a = Array.IndexOf(director.bodies, assist);
            if (v < 0) return;
            var row = El("feed-row", _feed); row.pickingMode = PickingMode.Ignore;
            void Who(int seat) { var d = El("dot", row); d.style.backgroundColor = MatchDirector.SeatColors[seat % MatchDirector.SeatColors.Length]; Text(director.BadgeOf(director.AgentOf(seat)), "", row); }
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
            }
            else
            {
                var r = director.bodies[f]; int agent = director.AgentOf(f); var entry = agent >= 0 && agent < director.roster.Length ? director.roster[agent] : null;
                var stripe = El("intro-stripe", _intro); stripe.style.backgroundColor = MatchDirector.SeatColors[f % MatchDirector.SeatColors.Length];
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
    }
}
