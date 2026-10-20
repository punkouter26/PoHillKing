using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Viewer voting from Twitch chat (GameSettings.TwitchChannel). Joins the channel read-only and anonymously
    /// (Twitch's justinfan login, no account or token), counts !wind / !ice / !quake / !shrink / !boxes over a 15 s
    /// window and triggers the winning hazard. "!hype &lt;fighter or colour&gt;" (e.g. !hype trump, !hype red) gives that
    /// fighter adrenaline when stamina is on (Stamina.TryHype has its own cooldown). The socket runs on a background
    /// thread; votes reach the game through queues read on the main thread.
    /// </summary>
    public class ChatVotes : MonoBehaviour
    {
        public Hazards hazards;
        public Stamina stamina;
        public DemoDirector director;
        static readonly string[] ColourNames = { "red", "blue", "yellow", "green", "purple", "cyan" };      // DemoDirector.SeatColors order
        readonly ConcurrentQueue<string> _hypes = new();
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
                        else if (msg.StartsWith("!hype ") && _hypes.Count < 64) _hypes.Enqueue(msg.Substring(6).Trim());
                    }
                }
                catch (Exception e) { Status = "chat error: " + e.Message; }
                for (int i = 0; i < 50 && _run; i++) Thread.Sleep(100);     // reconnect after 5 s
            }
        }

        void Update()
        {
            while (_hypes.TryDequeue(out var who))
            {
                int seat = SeatNamed(who);
                if (seat >= 0 && stamina != null && stamina.TryHype(director.fighters[seat])) MatchEvents.RaiseCaption($"Chat hypes {director.BadgeOf(director.AgentOf(seat))}!");
            }
            while (_votes.TryDequeue(out var h)) { if (_windowEnd <= 0) _windowEnd = Time.unscaledTime + windowSeconds; Tally[(int)h]++; }
            if (_windowEnd <= 0 || Time.unscaledTime < _windowEnd) return;
            int best = 0; for (int i = 1; i < Tally.Length; i++) if (Tally[i] > Tally[best]) best = i;
            if (Tally[best] > 0 && hazards != null) { MatchEvents.RaiseCaption($"Chat voted {(Hazard)best} ({Tally[best]} votes)"); hazards.Trigger((Hazard)best); }
            Array.Clear(Tally, 0, Tally.Length); _windowEnd = 0;
        }

        /// <summary>Seat named by a chat word: a seat colour, or the start of a fighter's badge (case-insensitive); -1 if none.</summary>
        public int SeatNamed(string word)
        {
            if (director == null || string.IsNullOrEmpty(word)) return -1;
            word = word.ToLowerInvariant();
            int c = Array.IndexOf(ColourNames, word); if (c >= 0 && c < director.Playing) return c;
            for (int i = 0; i < director.Playing; i++) if (director.BadgeOf(director.AgentOf(i)).ToLowerInvariant().StartsWith(word)) return i;
            return -1;
        }
    }
}
