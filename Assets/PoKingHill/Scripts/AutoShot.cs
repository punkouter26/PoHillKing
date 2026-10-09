using System;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Verification helper for play-mode runs without the match director: started with <c>-kothShot &lt;file.png&gt;</c>
    /// it saves one screenshot after <c>-kothShotAt</c> seconds (default 3) and, with <c>-kothExit</c>, closes the editor.
    /// </summary>
    public class AutoShot : MonoBehaviour
    {
        string _path; float _at = 3f; int _doneFrame = -1;

        static string Arg(string name)
        {
            var a = Environment.GetCommandLineArgs(); int i = Array.IndexOf(a, name);
            return i >= 0 && i + 1 < a.Length ? a[i + 1] : null;
        }

        void OnEnable()
        {
            _path = Arg("-kothShot"); if (_path == null) { enabled = false; return; }
            if (float.TryParse(Arg("-kothShotAt"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float t)) _at = t;
            Application.targetFrameRate = 60; Application.runInBackground = true;
        }

        void Update()
        {
            if (_doneFrame < 0 && Time.time >= _at) { ScreenCapture.CaptureScreenshot(_path); _doneFrame = Time.frameCount; Debug.Log("[AutoShot] " + _path); }
#if UNITY_EDITOR
            if (_doneFrame >= 0 && Time.frameCount > _doneFrame + 3 && Array.IndexOf(Environment.GetCommandLineArgs(), "-kothExit") >= 0) UnityEditor.EditorApplication.Exit(0);
#endif
        }
    }
}
