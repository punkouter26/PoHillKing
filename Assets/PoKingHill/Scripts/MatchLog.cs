using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using UnityEngine;
using UnityEngine.Networking;

namespace PoKingHill
{
    /// <summary>Elo rating per agent name, kept in persistentDataPath/ladder.json. A decided round counts as a win of
    /// the winner over every other seated agent; a tie as a draw between all of them.</summary>
    public static class Ladder
    {
        public const float Start = 1500f; const float K = 24f;
        // wins/losses (decided rounds against another agent), knock-outs and assists (ContactLedger) came later: files
        // written before them read as 0.
        [Serializable] class Entry { public string name; public float rating; public int rounds, wins, losses, kos, assists; }
        [Serializable] class File { public List<Entry> agents = new(); }
        static Dictionary<string, Entry> _byName;
        /// <summary>Tests point the ladder at a scratch file so they never touch the player's ratings.</summary>
        public static string FileOverride { get => _override; set { _override = value; _byName = null; } }
        static string _override;
        static string PathOf => _override ?? System.IO.Path.Combine(Application.persistentDataPath, "ladder.json");

        static Dictionary<string, Entry> All()
        {
            if (_byName != null) return _byName;
            _byName = new Dictionary<string, Entry>();
            try { if (System.IO.File.Exists(PathOf)) foreach (var e in JsonUtility.FromJson<File>(System.IO.File.ReadAllText(PathOf)).agents) _byName[e.name] = e; }
            catch (Exception ex) { Debug.LogWarning("[Ladder] could not read " + PathOf + ": " + ex.Message); }
            return _byName;
        }
        static Entry Get(string name) { var all = All(); if (!all.TryGetValue(name, out var e)) all[name] = e = new Entry { name = name, rating = Start }; return e; }
        public static float Rating(string name) => Get(name).rating;
        public static int Rounds(string name) => Get(name).rounds;
        public static int Wins(string name) => Get(name).wins;
        public static int Losses(string name) => Get(name).losses;
        public static int Knockouts(string name) => Get(name).kos;
        public static int Assists(string name) => Get(name).assists;

        /// <summary>Knock-outs and assists one agent was credited with in a round (saved with the next Record).</summary>
        public static void AddCredits(string name, int kos, int assists) { if (kos == 0 && assists == 0) return; var e = Get(name); e.kos += kos; e.assists += assists; Save(); }

        /// <summary>Pairs are bodies, not agents, so a mirror match (the same agent on two bodies) counts correctly: the
        /// winning body beats every other body; pairs of the same agent and pairs that both lost to a third are skipped.</summary>
        public static void Record(in RoundResult r, Fighter[] roster)
        {
            int n = r.Agents.Length; var delta = new Dictionary<string, float>(); var seen = new HashSet<string>();
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    int ai = r.Agents[i], aj = r.Agents[j];
                    if (ai < 0 || aj < 0 || ai >= roster.Length || aj >= roster.Length || ai == aj) continue;
                    float sa = r.Winner < 0 ? 0.5f : r.Winner == i ? 1f : r.Winner == j ? 0f : -1f;
                    if (sa < 0) continue;
                    string na = roster[ai].name, nb = roster[aj].name;
                    float ea = 1f / (1f + Mathf.Pow(10f, (Rating(nb) - Rating(na)) / 400f));
                    delta[na] = (delta.TryGetValue(na, out var da) ? da : 0) + K * (sa - ea);
                    delta[nb] = (delta.TryGetValue(nb, out var db) ? db : 0) - K * (sa - ea);
                }
            for (int i = 0; i < n; i++) if (r.Agents[i] >= 0 && r.Agents[i] < roster.Length) seen.Add(roster[r.Agents[i]].name);
            if (seen.Count < 2) return;
            string winner = r.Winner >= 0 && r.Winner < n && r.Agents[r.Winner] >= 0 && r.Agents[r.Winner] < roster.Length ? roster[r.Agents[r.Winner]].name : null;
            foreach (var name in seen)
            {
                var e = Get(name); e.rating += delta.TryGetValue(name, out var d) ? d : 0; e.rounds++;
                if (winner != null) { if (name == winner) e.wins++; else e.losses++; }
            }
            Save();
        }

        static void Save()
        {
            try { var f = new File(); f.agents.AddRange(All().Values); System.IO.File.WriteAllText(PathOf, JsonUtility.ToJson(f, true)); }
            catch (Exception ex) { Debug.LogWarning("[Ladder] could not write " + PathOf + ": " + ex.Message); }
        }
    }

    /// <summary>
    /// Round telemetry and the ladder. Samples every seated fighter at 10 Hz in the physics callback into preallocated
    /// buffers (pelvis position, uprightness, mean |action|), and when a round ends writes one JSON line to
    /// persistentDataPath/telemetry/&lt;YYMMDD&gt;.jsonl: agents, outcome, knock-out times, path length, top speed,
    /// action effort, final radius and the sampled kinematics. With GameSettings.UploadTelemetry and the environment
    /// variable KOTH_AZURE_TABLE_SAS (a table SAS URL) a summary row is also inserted into Azure Table Storage.
    /// </summary>
    public unsafe class MatchLog : MonoBehaviour
    {
        public DemoDirector director;
        public ContactLedger ledger;                    // optional: knock-out credits in the telemetry and the ladder
        const int Hz = 10, MaxSeconds = 90, MaxBodies = 8;
        readonly float[,] _kin = new float[MaxBodies, MaxSeconds * Hz * 4];     // x, y, z, up per sample
        readonly double[] _path = new double[MaxBodies], _topSpeed = new double[MaxBodies], _effort = new double[MaxBodies], _koAt = new double[MaxBodies];
        readonly double[] _lastX = new double[MaxBodies], _lastY = new double[MaxBodies];
        int _samples, _tick, _effortTicks;
        readonly float[] _ratingDelta = new float[MaxBodies];
        public static string Folder => Path.Combine(Application.persistentDataPath, "telemetry");

        // Per-body numbers of the current (or just finished) round, for the results sheet.
        public float PathMetres(int body) => body >= 0 && body < MaxBodies ? (float)_path[body] : 0;
        public float TopSpeed(int body) => body >= 0 && body < MaxBodies ? (float)_topSpeed[body] : 0;
        public float KnockedOutAt(int body) => body >= 0 && body < MaxBodies ? (float)_koAt[body] : -1;
        public float RatingDelta(int body) => body >= 0 && body < MaxBodies ? _ratingDelta[body] : 0;

        void OnEnable()
        {
            MatchEvents.RoundStarted += OnStart; MatchEvents.RoundEnded += OnEnd; MatchEvents.Knockout += OnKo;
            if (MjScene.InstanceExists) MjScene.Instance.postUpdateEvent += Post;
        }
        void OnDisable()
        {
            MatchEvents.RoundStarted -= OnStart; MatchEvents.RoundEnded -= OnEnd; MatchEvents.Knockout -= OnKo;
            if (MjScene.InstanceExists) MjScene.Instance.postUpdateEvent -= Post;
        }

        void OnStart()
        {
            _samples = 0; _tick = 0; _effortTicks = 0;
            for (int i = 0; i < MaxBodies; i++) { _path[i] = _topSpeed[i] = _effort[i] = 0; _koAt[i] = -1; _lastX[i] = double.NaN; }
        }
        void OnKo(PolicyRunner r) { int i = Array.IndexOf(director.fighters, r); if (i >= 0 && i < MaxBodies) _koAt[i] = director.RoundTime; }

        void Post(object s, MjStepArgs a)
        {
            if (director == null || director.State != MatchState.Fight || director.statsRounds > 0) return;
            if (++_tick % (int)Mathf.Round(1f / (Hz * Time.fixedDeltaTime)) != 0) return;
            var d = MjScene.Instance.Data; int n = Mathf.Min(director.Playing, MaxBodies); bool room = _samples < MaxSeconds * Hz;
            for (int i = 0; i < n; i++)
            {
                var r = director.fighters[i]; double* q = d->qpos + r.Map.RootQposAdr; double* v = d->qvel + r.Map.RootDofAdr;
                if (!double.IsNaN(_lastX[i])) _path[i] += Math.Sqrt((q[0] - _lastX[i]) * (q[0] - _lastX[i]) + (q[1] - _lastY[i]) * (q[1] - _lastY[i]));
                _lastX[i] = q[0]; _lastY[i] = q[1];
                _topSpeed[i] = Math.Max(_topSpeed[i], Math.Sqrt(v[0] * v[0] + v[1] * v[1]));
                double e = 0; var act = r.LastAction; for (int k = 0; k < act.Length; k++) e += Math.Abs(act[k]); _effort[i] += e / Math.Max(1, act.Length);
                if (room) { int o = 4 * _samples; _kin[i, o] = (float)q[0]; _kin[i, o + 1] = (float)q[1]; _kin[i, o + 2] = (float)q[2]; _kin[i, o + 3] = (float)ObsBuilder.Up(q + 3); }
            }
            _effortTicks++; if (room) _samples++;
        }

        void OnEnd(RoundResult r)
        {
            if (director == null || director.statsRounds > 0) return;
            for (int i = 0; i < r.Agents.Length && i < MaxBodies; i++) _ratingDelta[i] = -Ladder.Rating(director.NameOf(r.Agents[i], "Robot"));
            Ladder.Record(r, director.roster);
            for (int i = 0; i < r.Agents.Length && i < MaxBodies; i++) _ratingDelta[i] += Ladder.Rating(director.NameOf(r.Agents[i], "Robot"));
            if (ledger != null)
                for (int i = 0; i < r.Agents.Length && i < director.fighters.Length; i++)
                    if (r.Agents[i] >= 0) Ladder.AddCredits(director.NameOf(r.Agents[i], "Robot"), ledger.Knockouts(director.fighters[i]), ledger.Assists(director.fighters[i]));
            try { Write(r); } catch (Exception ex) { Debug.LogWarning("[MatchLog] " + ex.Message); }
        }

        unsafe void Write(in RoundResult r)
        {
            var inv = CultureInfo.InvariantCulture; string F(double x) => x.ToString("0.###", inv);
            var d = MjScene.Instance.Data; var sb = new StringBuilder(4096);
            string stamp = DateTime.UtcNow.ToString("o", inv);
            sb.Append("{\"utc\":\"").Append(stamp).Append("\",\"version\":\"").Append(BuildInfo.Version)
              .Append("\",\"map\":\"").Append(director.maps.Length > director.selectedMap ? director.maps[director.selectedMap] : "").Append('"')
              .Append(",\"end\":\"").Append(r.End).Append("\",\"seconds\":").Append(F(r.Seconds)).Append(",\"winner\":").Append(r.Winner)
              .Append(",\"hz\":").Append(Hz).Append(",\"fighters\":[");
            for (int i = 0; i < r.Agents.Length && i < MaxBodies; i++)
            {
                var f = director.fighters[i]; double* q = d->qpos + f.Map.RootQposAdr; int agent = r.Agents[i];
                string name = director.NameOf(agent, "Robot");
                if (i > 0) sb.Append(',');
                sb.Append("{\"agent\":\"").Append(name).Append("\",\"body\":\"").Append(f.robotPrefix).Append("\",\"result\":\"")
                  .Append(r.Winner < 0 ? "tie" : r.Winner == i ? "win" : "loss").Append("\",\"rating\":").Append(F(Ladder.Rating(name)))
                  .Append(",\"ko_s\":").Append(F(_koAt[i])).Append(",\"path_m\":").Append(F(_path[i])).Append(",\"top_speed\":").Append(F(_topSpeed[i]))
                  .Append(",\"effort\":").Append(F(_effort[i] / Math.Max(1, _effortTicks))).Append(",\"final_r\":").Append(F(Math.Sqrt(q[0] * q[0] + q[1] * q[1])))
                  .Append(",\"puppet\":").Append(director.PuppetBody == i ? "true" : "false");
                if (ledger != null)
                    sb.Append(",\"kos\":").Append(ledger.Knockouts(f)).Append(",\"assists\":").Append(ledger.Assists(f)).Append(",\"pushed_ns\":").Append(F(ledger.Pushed(f)));
                sb.Append(",\"kin\":[");
                for (int k = 0; k < 4 * _samples; k++) { if (k > 0) sb.Append(','); sb.Append(F(_kin[i, k])); }
                sb.Append("]}");
            }
            sb.Append("]}");
            Directory.CreateDirectory(Folder);
            File.AppendAllText(Path.Combine(Folder, DateTime.Now.ToString("yyMMdd") + ".jsonl"), sb.Append('\n').ToString());

            string sas = Environment.GetEnvironmentVariable("KOTH_AZURE_TABLE_SAS");
            if (GameSettings.UploadTelemetry && !string.IsNullOrEmpty(sas)) StartCoroutine(Upload(sas, r, stamp));
        }

        // Azure Table Storage "Insert Entity": POST https://<account>.table.core.windows.net/<table>?<sas> with a JSON entity.
        IEnumerator Upload(string sasUrl, RoundResult r, string stamp)
        {
            var inv = CultureInfo.InvariantCulture; var sb = new StringBuilder();
            sb.Append("{\"PartitionKey\":\"").Append(DateTime.UtcNow.ToString("yyMMdd")).Append("\",\"RowKey\":\"").Append(Guid.NewGuid().ToString("N"))
              .Append("\",\"Utc\":\"").Append(stamp).Append("\",\"Version\":\"").Append(BuildInfo.Version).Append("\",\"End\":\"").Append(r.End)
              .Append("\",\"Seconds\":").Append(r.Seconds.ToString("0.###", inv)).Append(",\"Winner\":\"").Append(Escape(r.WinnerName)).Append("\",\"Agents\":\"");
            for (int i = 0; i < r.Agents.Length; i++) sb.Append(i > 0 ? ";" : "").Append(Escape(director.NameOf(r.Agents[i], "Robot")));
            sb.Append("\"}");
            using var req = new UnityWebRequest(sasUrl, "POST") { uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(sb.ToString())), downloadHandler = new DownloadHandlerBuffer() };
            req.SetRequestHeader("Content-Type", "application/json"); req.SetRequestHeader("Accept", "application/json;odata=nometadata");
            req.SetRequestHeader("x-ms-version", "2019-02-02"); req.SetRequestHeader("Prefer", "return-no-content");
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success) Debug.LogWarning($"[MatchLog] Azure upload failed: {req.responseCode} {req.error}");
        }
        static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }
}
