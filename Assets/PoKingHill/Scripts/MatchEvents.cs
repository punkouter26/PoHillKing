using System;
using UnityEngine;

namespace PoKingHill
{
    public enum MatchState { Booting, Menu, Fight, Result, Replay }

    public enum RoundEnd { Knockout, SeaArrived, FullTime, NobodyLeft }

    /// <summary>What the winner panel, the ladder, telemetry and the commentator are told when a round ends.</summary>
    public readonly struct RoundResult
    {
        public readonly int Winner;            // body index in DemoDirector.fighters, -1 = no winner
        public readonly string WinnerName, How;
        public readonly RoundEnd End;
        public readonly float Seconds;
        public readonly int[] Agents;          // roster index per seated body, -1 = parked
        public RoundResult(int winner, string winnerName, string how, RoundEnd end, float seconds, int[] agents)
        { Winner = winner; WinnerName = winnerName; How = how; End = end; Seconds = seconds; Agents = agents; }
    }

    /// <summary>
    /// One place where the match announces what happens, so presentation (HUD, camera, VFX, audio, commentary,
    /// telemetry, ladder, chat) listens instead of being called by the director. Positions are Unity world space.
    /// Raised on the main thread (MuJoCo steps in FixedUpdate); handlers must be cheap and must not allocate per call.
    /// </summary>
    public static class MatchEvents
    {
        public static event Action<MatchState> StateChanged;
        public static event Action RoundStarted;
        public static event Action<PolicyRunner> Knockout;
        public static event Action<RoundResult> RoundEnded;
        /// <summary>A sharp change of a pelvis velocity: position, strength 0..1, the robot hit.</summary>
        public static event Action<Vector3, float, PolicyRunner> Impact;
        /// <summary>A foot coming to rest: position, strength 0..1, horizontal slip speed at touchdown (m/s).</summary>
        public static event Action<Vector3, float, float> Footfall;
        public static event Action<Vector3, float> Splash;
        /// <summary>Free text for the caption line (commentary, chat votes, hazards).</summary>
        public static event Action<string> Caption;

        public static void RaiseState(MatchState s) => StateChanged?.Invoke(s);
        public static void RaiseRoundStarted() => RoundStarted?.Invoke();
        public static void RaiseKnockout(PolicyRunner r) => Knockout?.Invoke(r);
        public static void RaiseRoundEnded(in RoundResult r) => RoundEnded?.Invoke(r);
        public static void RaiseImpact(Vector3 p, float s, PolicyRunner r) => Impact?.Invoke(p, s, r);
        public static void RaiseFootfall(Vector3 p, float s, float slip) => Footfall?.Invoke(p, s, slip);
        public static void RaiseSplash(Vector3 p, float s) => Splash?.Invoke(p, s);
        public static void RaiseCaption(string text) => Caption?.Invoke(text);

        // Domain reload is off in fast enter-play mode: listeners from the previous play session must not linger.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void Reset() { StateChanged = null; RoundStarted = null; Knockout = null; RoundEnded = null; Impact = null; Footfall = null; Splash = null; Caption = null; }
    }
}
