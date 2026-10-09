using System;
using System.Globalization;
using Mujoco;
using Unity.Profiling;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Performance pass (tasks.md D.6). Off unless the process was started with <c>-kothPerf &lt;seconds&gt;</c>: then the
    /// duel fights continuously, and after a 2 s warm-up every physics step and frame is timed for that long,
    /// perf_unity.json is written next to the project / player data folder, and the process exits.
    ///   tick   = first preUpdate listener -> first postUpdate listener (2 brains + ctrl + sea + mj_step + state sync)
    ///   mjstep = last preUpdate listener -> first postUpdate listener (mj_step + the plugin's state sync)
    ///   brain  = PolicyRunner.LastInferenceMs on each policy tick
    ///   gc     = managed bytes allocated by the main thread inside tick
    /// </summary>
    [DefaultExecutionOrder(-200)]   // subscribes before PolicyRunner (-100), so its handlers run first
    public class PerfProbe : MonoBehaviour
    {
        const float Warmup = 2f;
        float _seconds; bool _lateHooked, _done, _gcCounterWorks;
        float[] _tick, _mj, _brain, _frame; int _nTick, _nBrain, _nFrame;
        long _t0, _t1, _gc0, _gcBytes, _gcSteps, _frameGcBytes; int _gen0;
        float _start = -1; int _maxStepsInFrame, _stepsThisFrame;
        ProfilerRecorder _frameGc;
        PolicyRunner[] _runners;

        long Alloc() => _frameGc.Valid ? _frameGc.CurrentValue : GC.GetAllocatedBytesForCurrentThread();
        static double Ms(long ticks) => ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

        void OnEnable()
        {
            var a = Environment.GetCommandLineArgs(); int i = Array.IndexOf(a, "-kothPerf");
            if (i < 0 || i + 1 >= a.Length || !float.TryParse(a[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out _seconds)) { enabled = false; return; }
            int cap = (int)((_seconds + 1) / Time.fixedDeltaTime);
            _tick = new float[cap]; _mj = new float[cap]; _brain = new float[cap]; _frame = new float[cap];
            var dd = FindAnyObjectByType<DemoDirector>(); if (dd != null) { dd.autoLaunch = true; dd.roundsPerMatch = 0; dd.pauseBetweenRounds = 0f; }
            _runners = FindObjectsByType<PolicyRunner>(FindObjectsInactive.Exclude);
            foreach (var r in _runners) r.OnPolicyStep += OnBrain;
            _frameGc = ProfilerRecorder.StartNew(ProfilerCategory.Memory, "GC Allocated In Frame");
            // Mono's per-thread counter reads 0 in the player; the profiler counter exists in the editor and development builds only.
            long g = Alloc(); var junk = new byte[4096]; _gcCounterWorks = Alloc() - g >= junk.Length;
            MjScene.Instance.preUpdateEvent += PreFirst; MjScene.Instance.postUpdateEvent += Post;
        }

        // Every other component subscribes in its own OnEnable, so a handler added in Start is the last one called.
        void Start() { if (enabled) { MjScene.Instance.preUpdateEvent += PreLast; _lateHooked = true; } }

        void OnDisable()
        {
            if (_frameGc.Valid) _frameGc.Dispose();
            if (_runners != null) foreach (var r in _runners) if (r != null) r.OnPolicyStep -= OnBrain;
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.preUpdateEvent -= PreFirst; MjScene.Instance.postUpdateEvent -= Post;
            if (_lateHooked) MjScene.Instance.preUpdateEvent -= PreLast;
        }

        bool Recording => !_done && _start >= 0 && Time.unscaledTime - _start >= Warmup;

        void PreFirst(object s, MjStepArgs a) { _gc0 = Alloc(); _t0 = System.Diagnostics.Stopwatch.GetTimestamp(); }
        void PreLast(object s, MjStepArgs a) { _t1 = System.Diagnostics.Stopwatch.GetTimestamp(); }
        void Post(object s, MjStepArgs a)
        {
            long now = System.Diagnostics.Stopwatch.GetTimestamp(), gc = Alloc() - _gc0;
            _stepsThisFrame++;
            if (!Recording || _nTick >= _tick.Length) return;
            _tick[_nTick] = (float)Ms(now - _t0); _mj[_nTick++] = (float)Ms(now - _t1);
            if (gc > 0) { _gcBytes += gc; _gcSteps++; }
        }
        void OnBrain(PolicyRunner r) { if (Recording && _nBrain < _brain.Length) _brain[_nBrain++] = (float)r.LastInferenceMs; }

        void Update()
        {
            if (_done) return;
            if (_start < 0) { _start = Time.unscaledTime; _stepsThisFrame = 0; return; }
            if (Recording)
            {
                if (_nFrame == 0) _gen0 = GC.CollectionCount(0);
                if (_nFrame < _frame.Length) _frame[_nFrame++] = Time.unscaledDeltaTime * 1000f;
                _maxStepsInFrame = Mathf.Max(_maxStepsInFrame, _stepsThisFrame);
                if (_frameGc.Valid) _frameGcBytes += _frameGc.LastValue;
            }
            _stepsThisFrame = 0;
            if (Time.unscaledTime - _start >= Warmup + _seconds) Finish();
        }

        static string Stats(float[] v, int n)
        {
            if (n == 0) return "null";
            Array.Sort(v, 0, n); double sum = 0; for (int i = 0; i < n; i++) sum += v[i];
            string F(double x) => x.ToString("0.####", CultureInfo.InvariantCulture);
            return $"{{\"n\": {n}, \"mean\": {F(sum / n)}, \"p50\": {F(v[n / 2])}, \"p99\": {F(v[Math.Min(n - 1, (int)(0.99 * n))])}, \"max\": {F(v[n - 1])}}}";
        }

        void Finish()
        {
            _done = true;
            string I(double x) => x.ToString("0.##", CultureInfo.InvariantCulture);
            double frameMean = 0; for (int i = 0; i < _nFrame; i++) frameMean += _frame[i]; frameMean /= Math.Max(1, _nFrame);
            string json = "{\n" +
                $" \"build\": \"{(Application.isEditor ? "editor" : Debug.isDebugBuild ? "development player" : "release player")}\", \"unity\": \"{Application.unityVersion}\", \"cpu\": \"{SystemInfo.processorType}\",\n" +
                $" \"seconds\": {I(_seconds)}, \"fixed_dt\": {I(Time.fixedDeltaTime * 1000)}, \"vsync\": {QualitySettings.vSyncCount}, \"target_fps\": {Application.targetFrameRate}, \"window\": \"{Screen.width}x{Screen.height}\",\n" +
                $" \"fps_mean\": {I(1000.0 / Math.Max(frameMean, 1e-6))}, \"frame_ms\": {Stats(_frame, _nFrame)},\n" +
                $" \"physics_steps_per_s\": {I(_nTick / _seconds)}, \"max_steps_in_one_frame\": {_maxStepsInFrame},\n" +
                $" \"tick_ms\": {Stats(_tick, _nTick)},\n \"mjstep_ms\": {Stats(_mj, _nTick)},\n \"brain_ms\": {Stats(_brain, _nBrain)},\n" +
                $" \"gc_counter_works\": {(_gcCounterWorks ? "true" : "false")}, \"hot_path_gc_bytes_per_step\": {I((double)_gcBytes / Math.Max(1, _nTick))}, \"hot_path_steps_allocating\": {_gcSteps},\n" +
                $" \"frame_gc_recorder_valid\": {(_frameGc.Valid ? "true" : "false")}, \"frame_gc_bytes_per_frame\": {I((double)_frameGcBytes / Math.Max(1, _nFrame))}, \"gen0_collections\": {GC.CollectionCount(0) - _gen0}\n}}\n";
            string path = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "perf_unity.json"));
            System.IO.File.WriteAllText(path, json); Debug.Log("[PerfProbe] wrote " + path + "\n" + json);
#if UNITY_EDITOR
            if (Array.IndexOf(Environment.GetCommandLineArgs(), "-kothExit") >= 0) UnityEditor.EditorApplication.Exit(0);
#else
            Application.Quit();
#endif
        }
    }
}
