using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PoKingHill
{
    /// <summary>
    /// Render budget and live frame telemetry (GameSettings.Quality). Presets, applied to the game camera and the sun
    /// (never to the shared pipeline assets in the editor, where a change would be saved to disk):
    ///   Low      no post-processing, no shadows, no anti-aliasing, 40 % of the particles, 8 scuff decals, no shockwave,
    ///            the map's lighter mesh (LODGroup forced to LOD1), render scale 0.75 (player builds only)
    ///   Medium   post-processing, hard shadows, FXAA, 70 % of the particles, 16 decals, render scale 0.9 (player builds)
    ///   High     everything: soft shadows, SMAA, all particles and 24 decals, render scale 1
    ///   Auto     starts High on desktop and Medium on phones; in a player build it steps down a tier when a fight runs
    ///            below 45 FPS for 4 s (the editor's own overhead would trip it, so it only reports there)
    /// Frame telemetry for the HUD's telemetry line and MatchLog: mean FPS, mean and 95th-percentile frame time over the
    /// last 120 frames, and managed bytes allocated per frame on the main thread. Allocation-free.
    /// </summary>
    public class QualityTier : MonoBehaviour
    {
        public MatchDirector director;
        public Camera cam;
        public Light sun;
        public float downgradeFps = 45f, downgradeSeconds = 4f;

        /// <summary>The preset in force (never Auto).</summary>
        public Quality Current { get; private set; } = Quality.High;
        public float ParticleScale => Current == Quality.Low ? 0.4f : Current == Quality.Medium ? 0.7f : 1f;
        public int Decals => Current == Quality.Low ? 8 : Current == Quality.Medium ? 16 : 24;
        public bool Shockwaves => Current != Quality.Low;
        public float Fps { get; private set; } = 60f;
        public float FrameMs { get; private set; }
        public float FrameP95Ms { get; private set; }
        public float GcKbPerFrame { get; private set; }
        public int Downgrades { get; private set; }
        public string Label => GameSettings.Quality == Quality.Auto ? "Auto · " + Current : Current.ToString();

        const int N = 120;
        readonly float[] _ms = new float[N], _sorted = new float[N];
        int _head, _count; float _nextStats, _slowSince = -1f; long _lastBytes; double _gcSum; int _gcFrames;
        Quality _autoTier; bool _applied; float _baseScale = 1f;

        void OnEnable()
        {
            _autoTier = Application.isMobilePlatform ? Quality.Medium : Quality.High;
            if (GraphicsSettings.currentRenderPipeline is UniversalRenderPipelineAsset urp) _baseScale = urp.renderScale;
            GameSettings.Changed += Apply; Apply();
        }
        void OnDisable() { GameSettings.Changed -= Apply; }

        void Apply()
        {
            var want = GameSettings.Quality == Quality.Auto ? _autoTier : GameSettings.Quality;
            if (_applied && want == Current) return;
            Current = want; _applied = true;
            if (cam != null)
            {
                var data = cam.GetUniversalAdditionalCameraData();
                data.renderPostProcessing = Current != Quality.Low;
                data.renderShadows = Current != Quality.Low;
                data.antialiasing = Current == Quality.High ? AntialiasingMode.SubpixelMorphologicalAntiAliasing : Current == Quality.Medium ? AntialiasingMode.FastApproximateAntialiasing : AntialiasingMode.None;
            }
            // the Mountain map's levels of detail (tools/make_mountain_map.py): Low keeps the lighter mesh even up close
            foreach (var lg in FindObjectsByType<LODGroup>(FindObjectsInactive.Include)) lg.ForceLOD(Current == Quality.Low ? 1 : -1);
            if (sun != null) sun.shadows = Current == Quality.High ? LightShadows.Soft : Current == Quality.Medium ? LightShadows.Hard : LightShadows.None;
            // the pipeline asset is shared: change its render scale only in a player, where nothing is saved
            if (!Application.isEditor && GraphicsSettings.currentRenderPipeline is UniversalRenderPipelineAsset urp)
                urp.renderScale = _baseScale * (Current == Quality.Low ? 0.75f : Current == Quality.Medium ? 0.9f : 1f);
        }

        void Update()
        {
            float ms = Time.unscaledDeltaTime * 1000f;
            _ms[_head] = ms; _head = (_head + 1) % N; if (_count < N) _count++;
            long bytes = GC.GetAllocatedBytesForCurrentThread();
            if (_lastBytes > 0 && bytes >= _lastBytes) { _gcSum += bytes - _lastBytes; _gcFrames++; }
            _lastBytes = bytes;

            if (Time.unscaledTime >= _nextStats && _count > 10)
            {
                _nextStats = Time.unscaledTime + 0.5f;
                double sum = 0; for (int i = 0; i < _count; i++) { sum += _ms[i]; _sorted[i] = _ms[i]; }
                Array.Sort(_sorted, 0, _count);
                FrameMs = (float)(sum / _count); Fps = 1000f / Mathf.Max(0.01f, FrameMs); FrameP95Ms = _sorted[Mathf.Min(_count - 1, (int)(0.95f * _count))];
                GcKbPerFrame = _gcFrames > 0 ? (float)(_gcSum / _gcFrames / 1024.0) : 0f; _gcSum = 0; _gcFrames = 0;
            }

            // Auto: step down when a fight stays slow (player builds only; the statistics gates run at 50x on purpose)
            bool watch = GameSettings.Quality == Quality.Auto && !Application.isEditor && director != null && director.statsRounds == 0
                         && director.State == MatchState.Fight && Time.timeScale == 1f && _autoTier > Quality.Low;
            if (!watch || Fps >= downgradeFps) { _slowSince = -1f; return; }
            if (_slowSince < 0f) _slowSince = Time.unscaledTime;
            else if (Time.unscaledTime - _slowSince > downgradeSeconds)
            {
                _autoTier--; Downgrades++; _slowSince = -1f; Apply();
                MatchEvents.RaiseCaption($"Quality lowered to {_autoTier} to keep the frame rate up.");
            }
        }
    }
}
