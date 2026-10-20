using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoKingHill
{
    public partial class HudView
    {
        // ------------------------------------------------------------------ menu: Play | Agents | Settings
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
            Btn(ticked == 0 ? "Custom (pick in Agents)" : $"Custom · {ticked} agents", "secondary", () => director.Launch(), row).SetEnabled(ticked > 0);
            Btn("Tournament", "secondary", () => director.StartTournament(), row);
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

        // ------------------------------------------------------------------ settings: Match | Picture | Sound | Advanced
        // Every optional setting lives on this one page; the defaults (GameSettings) are the plain game.
        void BuildSettings()
        {
            Seg(new[] { "Match", "Picture", "Sound", "Advanced" }, _section, i => _section = i);
            if (_section == 0) BuildMatchSettings(); else if (_section == 1) BuildPictureSettings(); else if (_section == 2) BuildSoundSettings(); else BuildAdvancedSettings();
        }

        void BuildMatchSettings()
        {
            int[] roundOpts = { 1, 3, 5, 0 };
            Labeled("Rounds", new[] { "1", "3", "5", "Endless" }, Array.IndexOf(roundOpts, GameSettings.RoundsPerMatch), i => GameSettings.RoundsPerMatch = roundOpts[i]);
            float[] paces = { 1.5f, 1f, 0.7f };
            Labeled("Sea", new[] { "Slow", "Normal", "Fast" }, Nearest(paces, GameSettings.SeaPace), i => GameSettings.SeaPace = paces[i]);
            if (director.maps.Length > 1) { Text("Map", "hint", _sheetBody); Seg(director.maps, director.selectedMap, i => director.selectedMap = i); }
            var grid = El("grid", _sheetBody);
            Chip(grid, "Tale of the tape", GameSettings.Intro, v => GameSettings.Intro = v);
            Chip(grid, "Instant replays", GameSettings.Replays, v => GameSettings.Replays = v);
        }

        void BuildPictureSettings()
        {
            Labeled("Camera", new[] { "Broadcast", "Classic" }, (int)GameSettings.CameraStyle, i => GameSettings.CameraStyle = (CameraStyle)i);
            // the tier Auto resolved to is on the telemetry line and in the debug panel (a longer label overflows the segment)
            Labeled("Quality", new[] { "Auto", "Low", "Medium", "High" }, (int)GameSettings.Quality, i => GameSettings.Quality = (Quality)i);
            var grid = El("grid", _sheetBody);
            Chip(grid, "Effects", GameSettings.Effects, v => GameSettings.Effects = v);
            Chip(grid, "Slow motion", GameSettings.SlowMotion, v => GameSettings.SlowMotion = v);
            Chip(grid, "Cut-aways", GameSettings.CutAways, v => GameSettings.CutAways = v);
            Chip(grid, "Win odds", GameSettings.WinOdds, v => GameSettings.WinOdds = v);
            Chip(grid, "Captions", GameSettings.Commentary, v => GameSettings.Commentary = v);
        }

        void BuildSoundSettings()
        {
            float[] vols = { 0f, 0.3f, 0.6f, 1f }, mix = { 0f, 0.4f, 0.7f, 1f };
            string[] levels = { "Off", "Low", "Mid", "High" };
            Labeled("Master", levels, Nearest(vols, GameSettings.MasterVolume), i => GameSettings.MasterVolume = vols[i]);
            Labeled("Hits", levels, Nearest(mix, GameSettings.SfxVolume), i => GameSettings.SfxVolume = mix[i]);
            Labeled("Ambience", levels, Nearest(mix, GameSettings.AmbienceVolume), i => GameSettings.AmbienceVolume = mix[i]);
        }

        // Who free-for-all brains go for (brains were trained 1v1).
        static readonly string[] TargetingLabels = { "Nearest", "Weakest", "Dogpile", "Grudge", "Vulture" };
        static readonly string[] TargetingHints =
        {
            "each brain pushes the nearest fighter", "each brain goes for the fighter easiest to push off", "everyone goes for the king of the hill",
            "brains go back for whoever pushed them last", "brains wait out a scrap, then pick off the weakest",
        };

        void BuildAdvancedSettings()
        {
            int target = (int)GameSettings.Targeting;
            Text("Free-for-all targets: " + TargetingHints[Mathf.Clamp(target, 0, TargetingHints.Length - 1)], "hint", _sheetBody);
            Seg(TargetingLabels, target, i => GameSettings.Targeting = (Targeting)i).AddToClassList("seg-tight");
            var grid = El("grid", _sheetBody);
            Chip(grid, "Stamina + hype", GameSettings.Stamina, v => GameSettings.Stamina = v);
            bool getupBrain = director.getupBrains != null && Array.Exists(director.getupBrains, b => b != null && b.policy != null);
            Chip(grid, getupBrain ? "Get back up" : "Get back up (no brain yet)", GameSettings.GetUp, v => GameSettings.GetUp = v, getupBrain);
            Chip(grid, "Hazards", GameSettings.Hazards, v => GameSettings.Hazards = v);
            bool planner = Array.Exists(director.roster, f => f.mlp != null);
            Chip(grid, planner ? "Tactician (beta)" : "Tactician (no weights)", GameSettings.Tactician, v => GameSettings.Tactician = v, planner);
            Chip(grid, "Developer tools", GameSettings.DevTools, v => GameSettings.DevTools = v);
            Btn("Reset all settings", "secondary compact", () => { GameSettings.ResetAll(); RebuildSheet(); }, _sheetBody);
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
    }
}
