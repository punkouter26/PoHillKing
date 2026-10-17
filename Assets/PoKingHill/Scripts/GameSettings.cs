using System;
using UnityEngine;

namespace PoKingHill
{
    public enum CameraStyle { Broadcast, Classic }
    public enum Targeting { Nearest, Opportunist }
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

        public static void ResetAll() { PlayerPrefs.DeleteAll(); PlayerPrefs.Save(); Changed?.Invoke(); }

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
