using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace PoKingHill.Tests
{
    /// <summary>
    /// Console health for a test: collects every exception, error and assert logged while it lives, plus the UI Toolkit
    /// warnings that mean a broken layout or stylesheet (layout passes that do not settle, unknown USS properties or
    /// variables, missing panel settings). The test runner already fails on an unexpected error log; this also catches
    /// those warnings and lists everything at once. <c>using var guard = new ConsoleGuard(); ... guard.AssertClean();</c>
    /// </summary>
    public sealed class ConsoleGuard : IDisposable
    {
        static readonly string[] UiWarnings =
        {
            "Layout update is struggling", "UIElements", "UI Toolkit", "USS", "StyleSheet", "Unknown property", "Unknown custom property",
            "PanelSettings", "VisualElement", "Painter2D", "Invalid value for",
        };
        readonly List<string> _found = new();

        public ConsoleGuard() { Application.logMessageReceived += OnLog; }
        public void Dispose() { Application.logMessageReceived -= OnLog; }

        // PlayerTestAssemblyProvider.ResetStaticsOnLoad (SubsystemRegistration) calls m_LoadedAssemblies?.Clear()
        // instead of setting it to null, leaving it non-null and empty when Enter Play Mode skips domain reload.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void ResetTestAssemblyCache() =>
            typeof(UnityEngine.TestTools.UnityTestAttribute).Assembly.GetType("UnityEngine.TestTools.Utils.PlayerTestAssemblyProvider")
                ?.GetField("m_LoadedAssemblies", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)?.SetValue(null, null);

        void OnLog(string text, string stack, LogType type)
        {
            if (type == LogType.Exception || type == LogType.Error || type == LogType.Assert) { _found.Add($"{type}: {text}"); return; }
            if (type != LogType.Warning) return;
            foreach (var w in UiWarnings) if (text.Contains(w)) { _found.Add($"Warning: {text}"); return; }
        }

        public IReadOnlyList<string> Found => _found;
        public void AssertClean(string what = "the console")
        {
            Assert.IsEmpty(_found, $"{what} logged {_found.Count} problem(s):\n" + string.Join("\n", _found));
        }
    }
}
