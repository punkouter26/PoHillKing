using System;
using UnityEngine;

namespace PoKingHill
{
    public enum CameraStyle { Broadcast, Classic }
    /// <summary>Who each brain is pointed at in a free-for-all (brains were trained 1v1): Nearest (as trained), Opportunist
    /// (easiest to push off), Dogpile (everyone at the king, the one nearest the centre), Grudge (whoever pushed me last),
    /// Vulture (hang back while two others fight, then go for the weakest).</summary>
    public enum Targeting { Nearest, Opportunist, Dogpile, Grudge, Vulture }
    public enum Overlay { Off, Strain, Fatigue, Colliders }
    /// <summary>Render budget (QualityTier): Auto starts High on desktop, Medium on phones, and steps down in a player
    /// build when the frame rate stays low.</summary>
    public enum Quality { Auto, Low, Medium, High }

    /// <summary>
    /// Every optional setting, saved in PlayerPrefs. The defaults are the plain game, with the least on screen: one
    /// round per launch, the sea at its normal pace, the fight starts at once (no tale of the tape, no automatic
    /// replay), no captions, no win-odds bar, no hazards or extra rules, no developer tools. Read anywhere; the
    /// Settings page of the menu writes them.
    /// </summary>
    public static class GameSettings
    {
        public static event Action Changed;

        // Match
        public static int RoundsPerMatch { get => I("rounds", 1); set => Set("rounds", Mathf.Clamp(value, 0, 9)); }
        /// <summary>Multiplier on the sea's arrival time (1 = 20-30 s).</summary>
        public static float SeaPace { get => F("seaPace", 1f); set => Set("seaPace", Mathf.Clamp(value, 0.5f, 2f)); }
        /// <summary>Tale of the tape before a match's first round: a crane shot, then a card per fighter.</summary>
        public static bool Intro { get => B("intro", false); set => Set("intro", value); }
        /// <summary>The knock-out plays again on its own after the round (the results sheet always offers it).</summary>
        public static bool Replays { get => B("replays", false); set => Set("replays", value); }

        // Picture
        public static CameraStyle CameraStyle { get => (CameraStyle)I("camera", (int)CameraStyle.Broadcast); set => Set("camera", (int)value); }
        public static Quality Quality { get => (Quality)I("quality", (int)Quality.Auto); set => Set("quality", (int)value); }
        public static bool Effects { get => B("effects", true); set => Set("effects", value); }
        /// <summary>Slow motion when a fighter's balance point crosses the rim, a short freeze on the biggest hits (TimeWarp).</summary>
        public static bool SlowMotion { get => B("slowmo", true); set => Set("slowmo", value); }
        public static bool CutAways { get => B("cutaways", true); set => Set("cutaways", value); }
        public static bool WinOdds { get => B("winOdds", false); set => Set("winOdds", value); }
        /// <summary>Caption lines from the Commentator.</summary>
        public static bool Commentary { get => B("commentary", false); set => Set("commentary", value); }

        // Sound: mix channels are multipliers on MasterVolume (hits and footsteps; the wind, surf, crowd and thunder bed)
        public static float MasterVolume { get => F("volume", 0.6f); set => Set("volume", Mathf.Clamp01(value)); }
        public static float SfxVolume { get => F("sfxVolume", 1f); set => Set("sfxVolume", Mathf.Clamp01(value)); }
        public static float AmbienceVolume { get => F("ambVolume", 0.7f); set => Set("ambVolume", Mathf.Clamp01(value)); }

        // Advanced: extra rules and experiments
        public static Targeting Targeting { get => (Targeting)I("targeting", (int)Targeting.Opportunist); set => Set("targeting", (int)value); }
        public static bool Hazards { get => B("hazards", false); set => Set("hazards", value); }
        /// <summary>Fighters tire with the work their joints do (PD stiffness down to 85 %, inside the 80-120 % the
        /// brains were randomised over); a badge tap gives a short adrenaline boost.</summary>
        public static bool Stamina { get => B("stamina", false); set => Set("stamina", value); }
        /// <summary>A fighter that falls on the plateau gets a count to stand up again with its get-up brain.</summary>
        public static bool GetUp { get => B("getup", false); set => Set("getup", value); }
        /// <summary>Fighters with a lookahead planner choose press / flank / regroup from simulated futures.</summary>
        public static bool Tactician { get => B("tactician", false); set => Set("tactician", value); }
        /// <summary>The FPS pill, the debug button and the two-finger tap that open the debug panel.</summary>
        public static bool DevTools { get => B("devTools", false); set => Set("devTools", value); }

        // Debug panel
        /// <summary>A second line on the FPS pill: frame time p95 and the quality tier.</summary>
        public static bool TelemetryHud { get => B("telemetryHud", false); set => Set("telemetryHud", value); }
        public static Overlay Overlay { get => (Overlay)I("overlay", (int)Overlay.Off); set => Set("overlay", (int)value); }
        public static bool IntentArrows { get => B("intent", false); set => Set("intent", value); }

        static readonly string[] Keys = { "rounds", "seaPace", "intro", "replays", "camera", "quality", "effects", "slowmo", "cutaways", "winOdds", "commentary",
                                          "volume", "sfxVolume", "ambVolume", "targeting", "hazards", "stamina", "getup", "tactician", "devTools",
                                          "telemetryHud", "overlay", "intent" };
        /// <summary>Back to the defaults: deletes only this game's keys, never other PlayerPrefs.</summary>
        public static void ResetAll() { foreach (var k in Keys) PlayerPrefs.DeleteKey("koth." + k); PlayerPrefs.Save(); Changed?.Invoke(); }

        static int I(string k, int d) { try { return PlayerPrefs.GetInt("koth." + k, d); } catch { return d; } }
        static float F(string k, float d) { try { return PlayerPrefs.GetFloat("koth." + k, d); } catch { return d; } }
        static bool B(string k, bool d) => I(k, d ? 1 : 0) != 0;
        static void Set(string k, int v) { PlayerPrefs.SetInt("koth." + k, v); Changed?.Invoke(); }
        static void Set(string k, float v) { PlayerPrefs.SetFloat("koth." + k, v); Changed?.Invoke(); }
        static void Set(string k, bool v) => Set(k, v ? 1 : 0);

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetEvents() => Changed = null;
    }

    /// <summary>Build version shown bottom right: YYMMDD of the last commit, written by Editor/BuildVersion.cs.</summary>
    public static class BuildInfo
    {
        static string _v;
        public static string Version => _v ??= LoadVersion();
        static string LoadVersion()
        {
            var t = Resources.Load<TextAsset>("build_version");
            return t != null && t.text.Trim().Length > 0 ? t.text.Trim() : Application.version;
        }
    }
}
