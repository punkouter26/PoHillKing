using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    public enum Hazard { Wind, Ice, Quake, Shrink, Boxes }

    /// <summary>
    /// Arena hazards, all through native MuJoCo inputs (no PhysX):
    ///   Wind    a 1.5 s horizontal gust on every pelvis (xfrc_applied, 12 % of body weight)
    ///   Ice     foot sliding friction x0.7 for 6 s (0.6 -> 0.42, inside the 0.4-0.9 range the brains trained on)
    ///   Quake   2 s of random horizontal shoves on every pelvis
    ///   Shrink  the plateau that counts as "in" shrinks from 1.7 m to 1.3 m over 8 s, for the rest of the round
    ///   Boxes   one thrown box at every fighter (the training's projectile recipe)
    /// With GameSettings.Hazards a random one strikes every 8-12 s; chat votes (ChatVotes) and the debug panel can
    /// trigger any of them. Off in the statistics gate.
    /// </summary>
    public unsafe class Hazards : MonoBehaviour
    {
        public DemoDirector director;
        public ProjectilePool pool;
        public float windShare = 0.12f, iceScale = 0.7f, shrinkTo = 1.3f;

        public string Active => _windUntil > Now || _iceUntil > Now || _quakeUntil > Now ? _activeName : "";
        double _windUntil, _iceUntil, _quakeUntil, _shrinkFrom = -1; Vector2 _wind; string _activeName = "";
        float _nextRandom; int[] _feet = Array.Empty<int>(); double[] _footFriction = Array.Empty<double>(); bool _iced;
        readonly System.Random _rng = new();
        double Now => MjScene.InstanceExists && MjScene.Instance.Data != null ? MjScene.Instance.Data->time : 0;

        void OnEnable() { MatchEvents.RoundStarted += OnStart; if (MjScene.InstanceExists) MjScene.Instance.postInitEvent += OnInit; }
        // Subscribed in Start, after Sea's OnEnable: Sea assigns xfrc_applied for bodies in the water, the gust adds to it.
        void Start() { if (MjScene.InstanceExists) { MjScene.Instance.preUpdateEvent += Pre; if (MjScene.Instance.Model != null) OnInit(null, null); } }
        void OnDisable()
        {
            MatchEvents.RoundStarted -= OnStart;
            if (MjScene.InstanceExists) { MjScene.Instance.postInitEvent -= OnInit; MjScene.Instance.preUpdateEvent -= Pre; }
            Thaw();
        }

        void OnInit(object s, MjStepArgs a)
        {
            var m = MjScene.Instance.Model; var list = new System.Collections.Generic.List<int>();
            for (int g = 0; g < (int)m->ngeom; g++)
            {
                string name = System.Runtime.InteropServices.Marshal.PtrToStringAnsi((IntPtr)MujocoLib.mj_id2name(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, g)) ?? "";
                if (name.EndsWith("foot_box_collision")) list.Add(g);
            }
            _feet = list.ToArray(); _footFriction = new double[_feet.Length];
            for (int i = 0; i < _feet.Length; i++) _footFriction[i] = m->geom_friction[3 * _feet[i]];
        }

        void OnStart() { _windUntil = _iceUntil = _quakeUntil = 0; _shrinkFrom = -1; Thaw(); _nextRandom = Time.time + 8f + (float)_rng.NextDouble() * 4f; }

        public void Trigger(Hazard h)
        {
            if (director == null || director.State != MatchState.Fight || director.statsRounds > 0) return;
            double now = Now; _activeName = h.ToString();
            switch (h)
            {
                case Hazard.Wind: { float a = (float)(_rng.NextDouble() * Math.PI * 2); _wind = new Vector2(Mathf.Cos(a), Mathf.Sin(a)); _windUntil = now + 1.5; break; }
                case Hazard.Ice: _iceUntil = now + 6; Freeze(); break;
                case Hazard.Quake: _quakeUntil = now + 2; break;
                case Hazard.Shrink: if (_shrinkFrom < 0) _shrinkFrom = now; break;
                case Hazard.Boxes:
                    if (pool != null)
                        for (int i = 0; i < director.Playing; i++)
                        {
                            if (director.IsOut(i)) continue;
                            double* q = MjScene.Instance.Data->qpos + director.fighters[i].Map.RootQposAdr;
                            pool.FireAt(new Vector3((float)q[0], (float)q[1], (float)q[2]), 6f);
                        }
                    break;
            }
            MatchEvents.RaiseCaption(h switch
            {
                Hazard.Wind => "A gust sweeps the summit!", Hazard.Ice => "Ice on the summit: footing is treacherous.",
                Hazard.Quake => "The mountain shakes!", Hazard.Shrink => "The summit is crumbling: the safe ground shrinks.", _ => "Incoming boxes!",
            });
        }

        void Update()
        {
            if (director == null || director.State != MatchState.Fight || !GameSettings.Hazards || director.statsRounds > 0) return;
            if (Time.time < _nextRandom) return;
            _nextRandom = Time.time + 8f + (float)_rng.NextDouble() * 4f;
            Trigger((Hazard)_rng.Next(5));
        }

        void Pre(object s, MjStepArgs a)
        {
            if (director == null || director.State != MatchState.Fight) return;
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data; double now = d->time;
            if (_iced && now > _iceUntil) Thaw();
            if (_shrinkFrom >= 0) director.OutRadiusNow = Mathf.Lerp(director.outRadius, shrinkTo, (float)Math.Clamp((now - _shrinkFrom) / 8.0, 0, 1));
            bool wind = now < _windUntil, quake = now < _quakeUntil;
            if (!wind && !quake) return;
            double g = -m->opt.gravity[2]; bool seaAssigns = director.sea != null && director.sea.isActiveAndEnabled;
            for (int i = 0; i < director.Playing; i++)
            {
                if (director.IsOut(i)) continue;
                var r = director.fighters[i]; int b = r.Map.PelvisBodyId; double mass = m->body_subtreemass[b];
                double* F = d->xfrc_applied + 6 * b;
                if (!seaAssigns) { F[0] = 0; F[1] = 0; }        // xfrc_applied persists: without the sea re-assigning it, start from zero
                if (wind) { F[0] += windShare * mass * g * _wind.x; F[1] += windShare * mass * g * _wind.y; }
                if (quake) { F[0] += 0.1 * mass * g * (_rng.NextDouble() * 2 - 1); F[1] += 0.1 * mass * g * (_rng.NextDouble() * 2 - 1); }
            }
        }

        // xfrc_applied persists between steps: clear what the gust added once it ends (Sea re-assigns its own bodies).
        void LateUpdate()
        {
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null || director == null || Now < _windUntil || Now < _quakeUntil) return;
            if (_windUntil <= 0 && _quakeUntil <= 0) return;
            var d = MjScene.Instance.Data;
            for (int i = 0; i < director.fighters.Length; i++) { double* F = d->xfrc_applied + 6 * director.fighters[i].Map.PelvisBodyId; F[0] = F[1] = 0; }
            _windUntil = _quakeUntil = 0;
        }

        void Freeze()
        {
            if (_iced || !MjScene.InstanceExists) return; var m = MjScene.Instance.Model;
            for (int i = 0; i < _feet.Length; i++) m->geom_friction[3 * _feet[i]] = _footFriction[i] * iceScale;
            _iced = true;
        }
        void Thaw()
        {
            if (!_iced || !MjScene.InstanceExists || MjScene.Instance.Model == null) { _iced = false; return; }
            var m = MjScene.Instance.Model;
            for (int i = 0; i < _feet.Length; i++) m->geom_friction[3 * _feet[i]] = _footFriction[i];
            _iced = false;
        }
    }

    /// <summary>
    /// Viewer voting from Twitch chat (GameSettings.TwitchChannel). Joins the channel read-only and anonymously
    /// (Twitch's justinfan login, no account or token), counts !wind / !ice / !quake / !shrink / !boxes over a 15 s
    /// window and triggers the winning hazard. The socket runs on a background thread; votes reach the game through a
    /// queue read on the main thread.
    /// </summary>
    public class ChatVotes : MonoBehaviour
    {
        public Hazards hazards;
        public float windowSeconds = 15f;
        public string Status { get; private set; } = "off";
        public int[] Tally { get; } = new int[5];
        public float WindowLeft => Mathf.Max(0, _windowEnd - Time.unscaledTime);

        readonly ConcurrentQueue<Hazard> _votes = new(); Thread _thread; volatile bool _run; string _channel = ""; float _windowEnd;

        void OnEnable() { GameSettings.Changed += Apply; Apply(); }
        void OnDisable() { GameSettings.Changed -= Apply; StopChat(); }

        void Apply()
        {
            string ch = GameSettings.TwitchChannel;
            if (ch == _channel) return;
            StopChat(); _channel = ch;
            if (ch.Length == 0) { Status = "off"; return; }
            _run = true; Status = "connecting to #" + ch;
            _thread = new Thread(() => Loop(ch)) { IsBackground = true, Name = "TwitchChat" }; _thread.Start();
        }
        void StopChat() { _run = false; _thread = null; }

        void Loop(string channel)
        {
            while (_run)
            {
                try
                {
                    using var tcp = new TcpClient("irc.chat.twitch.tv", 6667) { ReceiveTimeout = 360000 };
                    using var s = tcp.GetStream(); using var rd = new StreamReader(s); using var wr = new StreamWriter(s) { NewLine = "\r\n", AutoFlush = true };
                    wr.WriteLine("NICK justinfan" + new System.Random().Next(10000, 99999)); wr.WriteLine("JOIN #" + channel);   // no Unity API off the main thread
                    Status = "listening to #" + channel;
                    string line;
                    while (_run && (line = rd.ReadLine()) != null)
                    {
                        if (line.StartsWith("PING")) { wr.WriteLine("PONG" + line.Substring(4)); continue; }
                        int at = line.IndexOf(" PRIVMSG #", StringComparison.Ordinal); if (at < 0) continue;
                        int colon = line.IndexOf(" :", at, StringComparison.Ordinal); if (colon < 0) continue;
                        string msg = line.Substring(colon + 2).Trim().ToLowerInvariant();
                        if (msg.StartsWith("!wind")) _votes.Enqueue(Hazard.Wind); else if (msg.StartsWith("!ice")) _votes.Enqueue(Hazard.Ice);
                        else if (msg.StartsWith("!quake")) _votes.Enqueue(Hazard.Quake); else if (msg.StartsWith("!shrink")) _votes.Enqueue(Hazard.Shrink);
                        else if (msg.StartsWith("!box")) _votes.Enqueue(Hazard.Boxes);
                    }
                }
                catch (Exception e) { Status = "chat error: " + e.Message; }
                for (int i = 0; i < 50 && _run; i++) Thread.Sleep(100);     // reconnect after 5 s
            }
        }

        void Update()
        {
            while (_votes.TryDequeue(out var h)) { if (_windowEnd <= 0) _windowEnd = Time.unscaledTime + windowSeconds; Tally[(int)h]++; }
            if (_windowEnd <= 0 || Time.unscaledTime < _windowEnd) return;
            int best = 0; for (int i = 1; i < Tally.Length; i++) if (Tally[i] > Tally[best]) best = i;
            if (Tally[best] > 0 && hazards != null) { MatchEvents.RaiseCaption($"Chat voted {(Hazard)best} ({Tally[best]} votes)"); hazards.Trigger((Hazard)best); }
            Array.Clear(Tally, 0, Tally.Length); _windowEnd = 0;
        }
    }
}
