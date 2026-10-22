using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace PoKingHill.EditorTools
{
    /// <summary>
    /// Signed release Android App Bundle for Google Play, written to <see cref="OUTPUT"/>.
    /// Menu Tools/PoHillKing/Build Android AAB (Play release), or headless:
    ///   Unity.exe -batchmode -quit -projectPath . -buildTarget Android -executeMethod PoKingHill.EditorTools.PlayAab.Build
    /// The outcome is the "AAB BUILD RESULT:" log line (and the exit code in batch mode).
    ///
    /// The upload key and its password (the .pass beside it) live in the OneDrive vault, never in this repo.
    /// Once the first bundle is on Play, losing that key means losing updates; see KEYSTORES-README.txt there.
    /// Signing and the .aab switch are restored afterwards, so a plain side-load APK build still works.
    /// </summary>
    public static class PlayAab
    {
        /// PERMANENT once the first bundle is uploaded: Play keys the app on it.
        public const string APP_ID = "com.punkoutersoftware.pohillking";
        const string KEYSTORE = "C:/Users/punko/OneDrive/VAULT/_CODE/pohillking-upload.jks";
        const string ALIAS = "pohillking-upload";
        const string OUTPUT = "Builds/Android/PoHillKing.aab";

        [MenuItem("Tools/PoHillKing/Build Android AAB (Play release)")]
        public static void Build()
        {
            string passFile = Path.ChangeExtension(KEYSTORE, ".pass");
            if (EditorApplication.isPlaying)
                Finish("Aborted — exit Play mode first", false);
            else if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
                Finish("Aborted — switch the editor to Android first (a cross-target build can strip URP shaders)", false);
            else if (!File.Exists(KEYSTORE) || !File.Exists(passFile))
                Finish("Aborted — keystore or .pass not found at " + KEYSTORE, false);
            else if (Prepare() is string missing)
                Finish("Aborted — " + missing, false);
            else
                Run(File.ReadAllText(passFile).Trim());
        }

        const string APK = "Builds/Android/PoHillKing.apk";

        /// <summary>Side-load APK for a phone on USB (debug keystore, no versionCode bump, no Play upload), written to
        /// Builds/Android/PoHillKing.apk; then: adb install -r Builds/Android/PoHillKing.apk. Logs "APK BUILD RESULT:".</summary>
        [MenuItem("Tools/PoHillKing/Build Android APK (side-load)")]
        public static void SideloadApk()
        {
            if (EditorApplication.isPlaying) { Debug.LogError("APK BUILD RESULT: aborted, exit Play mode first"); return; }
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android) { Debug.LogError("APK BUILD RESULT: aborted, switch the editor to Android first"); return; }
            if (Prepare() is string missing) { Debug.LogError("APK BUILD RESULT: aborted, " + missing); return; }
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, APP_ID);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            bool wasCustomKeystore = PlayerSettings.Android.useCustomKeystore, wasBundle = EditorUserBuildSettings.buildAppBundle;
            PlayerSettings.Android.useCustomKeystore = false; EditorUserBuildSettings.buildAppBundle = false;
            string[] scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray();
            Directory.CreateDirectory(Path.GetDirectoryName(APK));
            BuildSummary summary;
            try { summary = BuildPipeline.BuildPlayer(scenes, APK, BuildTarget.Android, BuildOptions.None).summary; }
            finally { PlayerSettings.Android.useCustomKeystore = wasCustomKeystore; EditorUserBuildSettings.buildAppBundle = wasBundle; }
            string msg = $"APK BUILD RESULT: {summary.result} | {APP_ID} | {scenes.Length} scenes | {summary.totalErrors} errors | {summary.totalSize / (1024 * 1024)} MB | {APK} | {summary.totalTime.TotalMinutes:0.0} min";
            if (summary.result == BuildResult.Succeeded) Debug.Log(msg); else Debug.LogError(msg);
        }

        /// Project-specific setup before the build; returns what is missing, or null when ready.
        static string Prepare()
        {
            // Assets/Plugins/MuJoCo/Android/arm64-v8a/libmujoco.so (3.15.0, NDK r27c) imports aligned_alloc: API 28+.
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel28;
            return File.Exists("Assets/Plugins/MuJoCo/Android/arm64-v8a/libmujoco.so") ? null
                : "the Android MuJoCo library is missing (Assets/Plugins/MuJoCo/Android/arm64-v8a/libmujoco.so)";
        }

        static void Run(string pass)
        {
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, APP_ID);
            PlayerSettings.companyName = "Punkouter Software";
            PlayerSettings.Android.targetSdkVersion = (AndroidSdkVersions)36;   // Play's floor since 2026-08-31
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.bundleVersionCode++;   // Play rejects a reused versionCode
            AssetDatabase.SaveAssets();

            bool wasCustomKeystore = PlayerSettings.Android.useCustomKeystore, wasBundle = EditorUserBuildSettings.buildAppBundle;
            PlayerSettings.Android.useCustomKeystore = true;
            PlayerSettings.Android.keystoreName = KEYSTORE;
            PlayerSettings.Android.keyaliasName = ALIAS;
            PlayerSettings.Android.keystorePass = PlayerSettings.Android.keyaliasPass = pass;
            EditorUserBuildSettings.buildAppBundle = true;
            EditorUserBuildSettings.development = false;

            string[] scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray();
            Directory.CreateDirectory(Path.GetDirectoryName(OUTPUT));
            BuildSummary summary;
            try
            {
                summary = BuildPipeline.BuildPlayer(scenes, OUTPUT, BuildTarget.Android, BuildOptions.None).summary;
            }
            finally
            {
                PlayerSettings.Android.keystorePass = PlayerSettings.Android.keyaliasPass = "";
                PlayerSettings.Android.useCustomKeystore = wasCustomKeystore;
                EditorUserBuildSettings.buildAppBundle = wasBundle;
            }
            Finish($"{summary.result} | {APP_ID} v{PlayerSettings.bundleVersion} code {PlayerSettings.Android.bundleVersionCode} | " +
                   $"{scenes.Length} scenes | {summary.totalErrors} errors | {summary.totalSize / (1024 * 1024)} MB | {OUTPUT}",
                   summary.result == BuildResult.Succeeded);
        }

        static void Finish(string message, bool ok)
        {
            if (ok) Debug.Log("AAB BUILD RESULT: " + message);
            else Debug.LogError("AAB BUILD RESULT: " + message);
            if (Application.isBatchMode) EditorApplication.Exit(ok ? 0 : 1);
        }
    }
}
