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

    /// <summary>
    /// Every optional setting, saved in PlayerPrefs. Defaults are the plain game: one round per launch, the sea at its
    /// normal pace, no hazards, no overlays, no network features. Read anywhere; the Settings tab writes them.
    /// </summary>
    public static class GameSettings
    {
        public static event Action Changed;

        // Match
        public static int RoundsPerMatch { get => I("rounds", 1); set => Set("rounds", Mathf.Clamp(value, 0, 9)); }
        /// <summary>Multiplier on the sea's arrival time (1 = 20-30 s).</summary>
        public static float SeaPace { get => F("seaPace", 1f); set => Set("seaPace", Mathf.Clamp(value, 0.5f, 2f)); }
        public static Targeting Targeting { get => (Targeting)I("targeting", (int)Targeting.Opportunist); set => Set("targeting", (int)value); }
        public static bool Hazards { get => B("hazards", false); set => Set("hazards", value); }
        /// <summary>Tale of the tape before a match's first round: a crane shot, then a card per fighter.</summary>
        public static bool Intro { get => B("intro", true); set => Set("intro", value); }
        /// <summary>Fighters tire with the work their joints do (PD stiffness down to 85 %, inside the 80-120 % the
        /// brains were randomised over); adrenaline from chat or a badge tap gives a short boost.</summary>
        public static bool Stamina { get => B("stamina", false); set => Set("stamina", value); }
        /// <summary>A fighter that falls on the plateau gets a count to stand up again with its get-up brain.</summary>
        public static bool GetUp { get => B("getup", false); set => Set("getup", value); }
        /// <summary>Fighters with a lookahead planner choose press / flank / regroup from simulated futures.</summary>
        public static bool Tactician { get => B("tactician", false); set => Set("tactician", value); }

        // Presentation
        public static CameraStyle Camera { get => (CameraStyle)I("camera", (int)CameraStyle.Broadcast); set => Set("camera", (int)value); }
        public static bool CutAways { get => B("cutaways", true); set => Set("cutaways", value); }
        public static bool Replays { get => B("replays", true); set => Set("replays", value); }
        public static bool Effects { get => B("effects", true); set => Set("effects", value); }
        public static float Volume { get => F("volume", 0.6f); set => Set("volume", Mathf.Clamp01(value)); }
        public static bool WinOdds { get => B("winOdds", true); set => Set("winOdds", value); }
        public static bool Commentary { get => B("commentary", true); set => Set("commentary", value); }

        // Debug
        public static bool Telemetry { get => B("telemetryHud", false); set => Set("telemetryHud", value); }
        public static Overlay Overlay { get => (Overlay)I("overlay", (int)Overlay.Off); set => Set("overlay", (int)value); }
        public static bool IntentArrows { get => B("intent", false); set => Set("intent", value); }

        // Network features, all off until configured. Secrets are never stored here: the Claude key comes from the
        // ANTHROPIC_API_KEY environment variable and the Azure table SAS URL from KOTH_AZURE_TABLE_SAS.
        public static string TwitchChannel { get => S("twitch", ""); set => Set("twitch", (value ?? "").Trim().TrimStart('#').ToLowerInvariant()); }
        public static bool ClaudeCommentary { get => B("claude", false); set => Set("claude", value); }
        public static bool UploadTelemetry { get => B("azure", false); set => Set("azure", value); }
        /// <summary>Read the commentary aloud (Azure Speech with AZURE_SPEECH_KEY + AZURE_SPEECH_REGION, or ElevenLabs with ELEVENLABS_API_KEY).</summary>
        public static bool Voice { get => B("voice", false); set => Set("voice", value); }
        /// <summary>City whose live weather (Open-Meteo, no key) sets the summit's sky and how often gusts come; "" = off.</summary>
        public static string WeatherCity { get => S("weather", ""); set => Set("weather", (value ?? "").Trim()); }

        static readonly string[] Keys = { "rounds", "seaPace", "targeting", "hazards", "intro", "stamina", "getup", "tactician", "camera", "cutaways", "replays",
                                          "effects", "volume", "winOdds", "commentary", "telemetryHud", "overlay", "intent", "twitch", "claude", "azure", "voice", "weather" };
        /// <summary>Back to the defaults: deletes only this game's keys, never other PlayerPrefs.</summary>
        public static void ResetAll() { foreach (var k in Keys) PlayerPrefs.DeleteKey("koth." + k); PlayerPrefs.Save(); Changed?.Invoke(); }

        static int I(string k, int d) { try { return PlayerPrefs.GetInt("koth." + k, d); } catch { return d; } }
        static float F(string k, float d) { try { return PlayerPrefs.GetFloat("koth." + k, d); } catch { return d; } }
        static bool B(string k, bool d) => I(k, d ? 1 : 0) != 0;
        static string S(string k, string d) { try { return PlayerPrefs.GetString("koth." + k, d); } catch { return d; } }
        static void Set(string k, int v) { PlayerPrefs.SetInt("koth." + k, v); Changed?.Invoke(); }
        static void Set(string k, float v) { PlayerPrefs.SetFloat("koth." + k, v); Changed?.Invoke(); }
        static void Set(string k, bool v) => Set(k, v ? 1 : 0);
        static void Set(string k, string v) { PlayerPrefs.SetString("koth." + k, v); Changed?.Invoke(); }

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
