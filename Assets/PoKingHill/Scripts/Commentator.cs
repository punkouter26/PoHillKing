using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace PoKingHill
{
    /// <summary>
    /// One-line captions for what happens on the hill. Always works offline from a small set of templates; with
    /// GameSettings.ClaudeCommentary on and ANTHROPIC_API_KEY set in the environment it asks Claude (Messages API, raw
    /// HTTPS because the Anthropic C# SDK is a NuGet package Unity's player runtimes don't take) for a livelier line on
    /// the big moments: round start, knock-outs, the result. At most one request in flight and one every few seconds;
    /// a template line is shown at once and replaced if Claude answers in time. The key is never stored by the game.
    /// </summary>
    public class Commentator : MonoBehaviour
    {
        public DemoDirector director;
        public float minGapSeconds = 4f;
        const string Endpoint = "https://api.anthropic.com/v1/messages", Model = "claude-opus-5-5";

        public static bool ClaudeAvailable => !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY"));
        float _nextClaude; bool _busy; int _bigHits;
        static readonly System.Random Rng = new();

        void OnEnable() { MatchEvents.RoundStarted += OnStart; MatchEvents.Knockout += OnKo; MatchEvents.RoundEnded += OnEnd; MatchEvents.Impact += OnImpact; }
        void OnDisable() { MatchEvents.RoundStarted -= OnStart; MatchEvents.Knockout -= OnKo; MatchEvents.RoundEnded -= OnEnd; MatchEvents.Impact -= OnImpact; }

        bool On => GameSettings.Commentary && director != null && director.statsRounds == 0;

        void OnStart()
        {
            if (!On) return; _bigHits = 0;
            string field = Field();
            Say(Pick($"{field}: everyone on the summit, the sea is coming.", $"Round on! {field}.", $"{field}: last one standing keeps the hill."),
                $"A king-of-the-hill round between humanoid robots just started: {field}. The sea rises and ends the round in about half a minute.");
        }

        void OnKo(PolicyRunner r)
        {
            if (!On) return;
            int body = Array.IndexOf(director.fighters, r); string who = director.BadgeOf(director.AgentOf(body));
            int left = director.Alive.Count;
            Say(Pick($"{who} is off the hill!", $"Down goes {who}!", $"{who} tumbles toward the water."),
                $"{who} has just been knocked off the summit after {director.RoundTime:0.0} s; {left} fighter{(left == 1 ? "" : "s")} still standing.");
        }

        void OnImpact(Vector3 p, float strength, PolicyRunner r)
        {
            if (!On || strength < 0.8f || ++_bigHits > 3 || director.State != MatchState.Fight) return;
            int body = Array.IndexOf(director.fighters, r);
            Say(Pick($"Huge hit on {director.BadgeOf(director.AgentOf(body))}!", "What a collision!", "That one shook the mountain."), null);
        }

        void OnEnd(RoundResult res)
        {
            if (!On) return;
            string line = res.Winner >= 0 ? Pick($"{res.WinnerName} is king of the hill!", $"{res.WinnerName} takes it, {res.How}.", $"Crown for {res.WinnerName}.")
                                          : Pick("Nobody keeps the hill this time.", "A draw on the summit.");
            Say(line, $"Round over after {res.Seconds:0.0} s. {(res.Winner >= 0 ? res.WinnerName + " won, " + res.How : "No winner: " + res.How)}.");
        }

        string Field()
        {
            var sb = new StringBuilder();
            for (int i = 0; i < director.Playing; i++) sb.Append(i == 0 ? "" : i == director.Playing - 1 ? " vs " : ", ").Append(director.BadgeOf(director.AgentOf(i)));
            return sb.ToString();
        }

        static string Pick(params string[] lines) => lines[Rng.Next(lines.Length)];

        void Say(string local, string claudePrompt)
        {
            MatchEvents.RaiseCaption(local);
            if (claudePrompt == null || !GameSettings.ClaudeCommentary || _busy || Time.unscaledTime < _nextClaude || !ClaudeAvailable) return;
            _nextClaude = Time.unscaledTime + minGapSeconds;
            StartCoroutine(AskClaude(claudePrompt));
        }

        [Serializable] class Block { public string type; public string text; }
        [Serializable] class Reply { public Block[] content; public string stop_reason; }

        IEnumerator AskClaude(string situation)
        {
            _busy = true;
            const string system = "You are a lively sports commentator for a physics game where humanoid robots fight to stay on a mountain " +
                                  "summit while the sea rises. Reply with one short broadcast line (at most 14 words), no quotes, no emoji.";
            string body = "{\"model\":\"" + Model + "\",\"max_tokens\":2048,\"fallbacks\":\"default\",\"output_config\":{\"effort\":\"low\"}," +
                          "\"system\":\"" + Json(system) + "\",\"messages\":[{\"role\":\"user\",\"content\":\"" + Json(situation) + "\"}]}";
            using var req = new UnityWebRequest(Endpoint, "POST") { uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body)), downloadHandler = new DownloadHandlerBuffer(), timeout = 20 };
            req.SetRequestHeader("Content-Type", "application/json");
            req.SetRequestHeader("x-api-key", Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY"));
            req.SetRequestHeader("anthropic-version", "2023-06-01");
            req.SetRequestHeader("anthropic-beta", "server-side-fallback-2026-07-01");
            yield return req.SendWebRequest();
            _busy = false;
            if (req.result != UnityWebRequest.Result.Success) { Debug.LogWarning($"[Commentator] Claude request failed: {req.responseCode} {req.error}"); yield break; }
            Reply reply; try { reply = JsonUtility.FromJson<Reply>(req.downloadHandler.text); } catch { yield break; }
            if (reply == null || reply.stop_reason == "refusal" || reply.content == null) yield break;   // declined: keep the template line
            foreach (var b in reply.content)
                if (b.type == "text" && !string.IsNullOrWhiteSpace(b.text)) { MatchEvents.RaiseCaption(b.text.Trim().Trim('"')); yield break; }
        }

        static string Json(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n");
    }
}
