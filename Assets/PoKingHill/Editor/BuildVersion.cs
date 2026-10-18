using System;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace PoKingHill.EditorTools
{
    /// <summary>
    /// Writes the HUD's build version, YYMMDD of the last git commit (today's date when git is unavailable), to
    /// Assets/PoKingHill/Resources/build_version.txt (git-ignored) on every editor load and before every player build,
    /// and, for a build only, sets PlayerSettings.bundleVersion to the same string (doing it on every load would rewrite
    /// ProjectSettings.asset each day).
    /// </summary>
    [InitializeOnLoad]
    public class BuildVersion : IPreprocessBuildWithReport
    {
        const string Dir = "Assets/PoKingHill/Resources", FilePath = Dir + "/build_version.txt";
        public int callbackOrder => 0;

        static BuildVersion() => EditorApplication.delayCall += () => Write(false);
        public void OnPreprocessBuild(BuildReport report) => Write(true);

        static void Write(bool forBuild)
        {
            string v = CommitDate() ?? DateTime.Now.ToString("yyMMdd");
            if (forBuild) PlayerSettings.bundleVersion = v;
            if (File.Exists(FilePath) && File.ReadAllText(FilePath).Trim() == v) return;
            Directory.CreateDirectory(Dir); File.WriteAllText(FilePath, v + "\n");
            if (!forBuild) AssetDatabase.ImportAsset(FilePath);
            UnityEngine.Debug.Log("[BuildVersion] " + v);
        }

        static string CommitDate()
        {
            try
            {
                var p = Process.Start(new ProcessStartInfo("git", "log -1 --format=%cd --date=format:%y%m%d")
                    { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = Path.GetFullPath(".") });
                string s = p.StandardOutput.ReadToEnd().Trim(); p.WaitForExit(3000);
                return s.Length == 6 ? s : null;
            }
            catch { return null; }
        }
    }
}
