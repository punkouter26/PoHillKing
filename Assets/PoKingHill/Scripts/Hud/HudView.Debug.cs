using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoKingHill
{
    public partial class HudView
    {
        // ------------------------------------------------------------------ debug panel: Perf | View | Tools
        void BuildDebugRows()
        {
            _debugTabs.Clear(); _debugBody.Clear();
            string[] names = { "Perf", "View", "Tools" };
            for (int i = 0; i < names.Length; i++) { int t = i; var b = Btn(names[i], "tab", () => { _debugTab = t; BuildDebugRows(); }, _debugTabs); if (i == _debugTab) b.AddToClassList("selected"); }
            if (_debugTab == 0) { _debugBody.Add(_debugText); _debugText.text = DebugText(); return; }
            if (_debugTab == 1)
            {
                var seg = El("seg", _debugBody); string[] modes = { "None", "Strain", "Colliders" };
                for (int i = 0; i < modes.Length; i++) { int k = i; var b = Btn(modes[i], "seg-btn", () => { GameSettings.Overlay = (Overlay)k; BuildDebugRows(); }, seg); if ((int)GameSettings.Overlay == i) b.AddToClassList("selected"); }
                var grid = El("grid", _debugBody);
                Chip(grid, "Intent arrows", GameSettings.IntentArrows, v => GameSettings.IntentArrows = v);
                Chip(grid, "Telemetry HUD", GameSettings.TelemetryHud, v => GameSettings.TelemetryHud = v);
                Chip(grid, "Win odds", GameSettings.WinOdds, v => GameSettings.WinOdds = v);
                Chip(grid, "Captions", GameSettings.Commentary, v => GameSettings.Commentary = v);
                return;
            }
            var r = El("seg", _debugBody);
            Btn("Replay", "seg-btn", () => replay?.Play(), r);
            Btn("Restart round", "seg-btn", () => { if (director.State == MatchState.Fight) director.NewRound(); }, r);
        }

        string DebugText()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append($"state {director.State} · round {director.RoundTime:0.0} s · out radius {director.OutRadiusNow:0.00} m\n");
            sb.Append($"physics tick {director.StepMs:0.000} ms · ");
            for (int i = 0; i < director.Playing && i < director.bodies.Length; i++) sb.Append($"{director.bodies[i].robotPrefix} {director.bodies[i].LastInferenceMs:0.00} ms  ");
            sb.Append($"\nGC {GC.CollectionCount(0)} gen0 · telemetry {MatchLog.Folder}");
            if (quality != null) sb.Append($"\nframe {quality.FrameMs:0.0} ms (p95 {quality.FrameP95Ms:0.0}) · {quality.GcKbPerFrame:0.00} KB/frame · quality {quality.Label}, {quality.Downgrades} auto steps");
            if (synth != null) sb.Append($"\naudio: wind {synth.WindRms:0.000} surf {synth.SurfRms:0.000} crowd {synth.CrowdRms:0.000} · slide {synth.Slide:0.00} grind {synth.Grind:0.00} · {synth.Clanks} clanks {synth.Smacks} smacks {synth.Grunts} grunts");
            if (warp != null) sb.Append($" · {warp.SlowMos} slow-mos {warp.HitStops} hit stops");
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
            }
            if (_peekRow.resolvedStyle.display == DisplayStyle.Flex && _peek.worldBound.yMax > _bottomBar.worldBound.yMin + 1) list.Add("peek pill covers the bottom corners");
            if (_badges.Count > 1)
                foreach (var b in _badges) if (Mathf.Abs(b.worldBound.yMin - _badges[0].worldBound.yMin) > 1) { list.Add("badges wrap onto a second row"); break; }
            if (_intro.resolvedStyle.display == DisplayStyle.Flex && _intro.worldBound.yMin < _scoreboard.worldBound.yMax - 1) list.Add("intro card overlaps the badges");
            foreach (var c in Cutouts)
            {
                var r = PanelRect(c);
                if (_devTools && r.Overlaps(_fps.worldBound)) list.Add("FPS pill on a cutout");
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
}
