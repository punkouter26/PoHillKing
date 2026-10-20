using System.IO;
using System.Text;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Records the observations Unity feeds each brain, for training/scripts/obs_audit.py (are Unity's inputs inside what
    /// the brain saw in training?). Added only by ParityBatch.RunObsDump. Every <c>everyTicks</c>-th policy tick of every
    /// runner with a fight brain (the get-up brain is skipped) is copied into a preallocated buffer, so the physics tick
    /// stays allocation-free; when the component is disabled (the gate leaves play mode) it writes one JSONL file per body:
    /// training/assets/&lt;body&gt;/unity_obs_&lt;tag&gt;.jsonl, one {"obs": [...], "brain", "seat"} object per line.
    /// </summary>
    public class ObsRecorder : MonoBehaviour
    {
        public string tag_ = "koth_2p";
        public int everyTicks = 5;
        public int maxRows = 20000;
        [Tooltip("Only brains whose ONNX name contains this (empty = every brain)")] public string brainFilter = "";

        PolicyRunner[] _runners = System.Array.Empty<PolicyRunner>();
        float[] _buf; int[] _dim, _runner; string[] _brain; int _rows, _used, _tick, _seen, _skipped;
        public int Rows => _rows;

        void Start()
        {
            _runners = FindObjectsByType<PolicyRunner>(FindObjectsInactive.Exclude);
            _buf = new float[maxRows * 121]; _dim = new int[maxRows]; _runner = new int[maxRows]; _brain = new string[maxRows];
            foreach (var r in _runners) r.OnPolicyStep += Record;
            Debug.Log($"[ObsRecorder] recording {_runners.Length} runners, every {everyTicks}th tick, tag {tag_}");
        }

        void Record(PolicyRunner r)
        {
            _seen++;
            if (_rows >= maxRows || r.GettingUp || r.policy == null || r.paused) { _skipped++; return; }
            if (++_tick % everyTicks != 0) return;
            string name = r.policy.name;
            if (brainFilter.Length > 0 && !name.Contains(brainFilter)) return;
            int n = Mathf.Min(r.Obs.Length, r.policyObsDim);
            if (_used + n > _buf.Length) return;
            System.Array.Copy(r.Obs, 0, _buf, _used, n);
            _dim[_rows] = n; _runner[_rows] = System.Array.IndexOf(_runners, r); _brain[_rows] = name; _used += n; _rows++;
        }

        void OnDisable()
        {
            foreach (var r in _runners) if (r != null) r.OnPolicyStep -= Record;
            if (_rows == 0 || _buf == null) { Debug.LogWarning($"[ObsRecorder] nothing recorded for {tag_}: {_seen} policy ticks seen, {_skipped} skipped"); return; }
            string root = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "training", "assets"));
            var writers = new System.Collections.Generic.Dictionary<string, StreamWriter>();
            int at = 0; var sb = new StringBuilder();
            try
            {
                for (int i = 0; i < _rows; i++)
                {
                    var r = _runners[_runner[i]]; string body = r != null ? r.body : "g1";
                    if (!writers.TryGetValue(body, out var w))
                    {
                        string dir = Path.Combine(root, body); Directory.CreateDirectory(dir);
                        writers[body] = w = new StreamWriter(Path.Combine(dir, $"unity_obs_{tag_}.jsonl"), false, new UTF8Encoding(false));
                    }
                    sb.Clear().Append("{\"obs\":[");
                    for (int k = 0; k < _dim[i]; k++) { if (k > 0) sb.Append(','); sb.Append(_buf[at + k].ToString("R", System.Globalization.CultureInfo.InvariantCulture)); }
                    sb.Append("],\"brain\":\"").Append(_brain[i]).Append("\",\"seat\":\"").Append(r != null ? r.robotPrefix : "?").Append("\"}");
                    w.WriteLine(sb.ToString()); at += _dim[i];
                }
            }
            finally { foreach (var w in writers.Values) w.Dispose(); }
            Debug.Log($"[ObsRecorder] wrote {_rows} observations ({string.Join(", ", writers.Keys)}) as unity_obs_{tag_}.jsonl");
            _rows = 0;
        }
    }
}
