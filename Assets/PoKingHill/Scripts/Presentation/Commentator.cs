using System;
using System.Text;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// One-line captions for what happens on the hill (GameSettings.Commentary), picked offline from a small set of
    /// templates: round start, big hits, falls and get-ups, knock-outs with their credit, the result.
    /// </summary>
    public class Commentator : MonoBehaviour
    {
        public DemoDirector director;
        int _bigHits;
        static readonly System.Random Rng = new();

        void OnEnable()
        {
            MatchEvents.RoundStarted += OnStart; MatchEvents.KnockoutCredited += OnKo; MatchEvents.RoundEnded += OnEnd; MatchEvents.Impact += OnImpact;
            MatchEvents.Down += OnDown; MatchEvents.Recovered += OnUp; MatchEvents.Hyped += OnHype;
        }
        void OnDisable()
        {
            MatchEvents.RoundStarted -= OnStart; MatchEvents.KnockoutCredited -= OnKo; MatchEvents.RoundEnded -= OnEnd; MatchEvents.Impact -= OnImpact;
            MatchEvents.Down -= OnDown; MatchEvents.Recovered -= OnUp; MatchEvents.Hyped -= OnHype;
        }

        string Who(PolicyRunner r) => director.BadgeOf(director.AgentOf(Array.IndexOf(director.fighters, r)));

        void OnDown(PolicyRunner r)
        {
            if (!On) return; string who = Who(r);
            Say(Pick($"{who} is down! Can they get up?", $"{who} hits the deck, the count is on.", $"Down goes {who}, but it's not over."));
        }

        void OnUp(PolicyRunner r)
        {
            if (!On) return; string who = Who(r);
            Say(Pick($"{who} is back up!", $"{who} beats the count!", $"Up again: {who} fights on."));
        }

        void OnHype(PolicyRunner r) { if (On) Say(Pick($"The crowd lifts {Who(r)}!", $"Adrenaline for {Who(r)}!")); }

        bool On => GameSettings.Commentary && director != null && director.statsRounds == 0;

        void OnStart()
        {
            if (!On) return; _bigHits = 0;
            string field = Field();
            Say(Pick($"{field}: everyone on the summit, the sea is coming.", $"Round on! {field}.", $"{field}: last one standing keeps the hill."));
        }

        // ContactLedger credits the knock-out right after the director announces it: by = null means nobody pushed it.
        void OnKo(PolicyRunner r, PolicyRunner by, PolicyRunner assist)
        {
            if (!On) return;
            string who = Who(r);
            Say(by == null ? Pick($"{who} is off the hill!", $"{who} tumbles toward the water.", $"{who} slips off on their own!")
              : assist == null ? Pick($"{Who(by)} shoves {who} off!", $"{who} goes over, courtesy of {Who(by)}.", $"{Who(by)} sends {who} into the sea!")
              : Pick($"{Who(by)} finishes {who}, with help from {Who(assist)}!", $"{who} is off: {Who(by)} and {Who(assist)} teamed up."));
        }

        void OnImpact(Vector3 p, float strength, PolicyRunner r)
        {
            if (!On || strength < 0.8f || ++_bigHits > 3 || director.State != MatchState.Fight) return;
            int body = Array.IndexOf(director.fighters, r);
            Say(Pick($"Huge hit on {director.BadgeOf(director.AgentOf(body))}!", "What a collision!", "That one shook the mountain."));
        }

        void OnEnd(RoundResult res)
        {
            if (!On) return;
            Say(res.Winner >= 0 ? Pick($"{res.WinnerName} is king of the hill!", $"{res.WinnerName} takes it, {res.How}.", $"Crown for {res.WinnerName}.")
                                : Pick("Nobody keeps the hill this time.", "A draw on the summit."));
        }

        string Field()
        {
            var sb = new StringBuilder();
            for (int i = 0; i < director.Playing; i++) sb.Append(i == 0 ? "" : i == director.Playing - 1 ? " vs " : ", ").Append(director.BadgeOf(director.AgentOf(i)));
            return sb.ToString();
        }

        static string Pick(params string[] lines) => lines[Rng.Next(lines.Length)];
        static void Say(string line) => MatchEvents.RaiseCaption(line);
    }
}
