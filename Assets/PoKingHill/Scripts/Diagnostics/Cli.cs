using System;

namespace PoKingHill
{
    /// <summary>The -koth* command-line switches, shared by the runtime components and the editor batch tools.</summary>
    public static class Cli
    {
        // Read once: GetCommandLineArgs returns a fresh array on every call, and Exit is polled from physics callbacks.
        static readonly string[] Args = Environment.GetCommandLineArgs();

        /// <summary>Value after <paramref name="name"/>, or <paramref name="fallback"/> when the switch is absent.</summary>
        public static string Arg(string name, string fallback = null)
        {
            int i = Array.IndexOf(Args, name);
            return i >= 0 && i + 1 < Args.Length ? Args[i + 1] : fallback;
        }

        public static bool Flag(string name) => Array.IndexOf(Args, name) >= 0;

        /// <summary>Automated run: leave the editor when the job is done.</summary>
        public static readonly bool Exit = Flag("-kothExit");
    }
}
