using System;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Stamina and adrenaline (GameSettings.Stamina). Each fighter's stamina S in [0, 1] drains with the mechanical power
    /// its joints deliver and recovers toward 1:
    ///   dS/dt = -P / (m * fullWattsPerKg) + recovery * (1 - S),   P = sum |actuator force x joint speed|, m = body mass
    /// and sets the stiffness of its position servos: kp scale = minKp + (1 - minKp) S, plus adrenalineKp while hyped.
    /// Both kp and the matching bias term are scaled, as the training's kp randomisation does (koth/env.py _randomize,
    /// +-20 %), and the scale stays inside 0.85-1.15, so every brain meets joints it was trained on. Adrenaline (chat
    /// !hype, a badge tap) lasts a few seconds, refills some stamina, and has a cooldown. Off in the statistics gate:
    /// nominal gains are restored whenever stamina is off or no round is being fought.
    /// </summary>
    public unsafe class Stamina : MonoBehaviour
    {
        public DemoDirector director;
        public float fullWattsPerKg = 150f, recovery = 0.04f, minKp = 0.85f, adrenalineKp = 0.15f, adrenalineSeconds = 4f, cooldownSeconds = 8f;
        const int NGain = 10, NBias = 10;            // mjNGAIN, mjNBIAS

        PolicyRunner[] _runners = Array.Empty<PolicyRunner>();
        float[] _s = Array.Empty<float>(), _scale = Array.Empty<float>(); double[] _hypeUntil = Array.Empty<double>(), _hypeReady = Array.Empty<double>(), _mass = Array.Empty<double>();
        double[] _gain0 = Array.Empty<double>(), _bias0 = Array.Empty<double>(); bool _ready, _applied;

        bool Active => director != null && director.statsRounds == 0 && GameSettings.Stamina && director.State == MatchState.Fight;
        int Index(PolicyRunner r) => r == null ? -1 : Array.IndexOf(_runners, r);
        /// <summary>Stamina 0..1 (1 when the feature is off).</summary>
        public float Level(PolicyRunner r) { int i = Index(r); return i >= 0 && _ready && GameSettings.Stamina ? _s[i] : 1f; }
        /// <summary>The stiffness scale applied to this fighter's servos now (1 = nominal).</summary>
        public float KpScale(PolicyRunner r) { int i = Index(r); return i >= 0 && _ready && _applied ? _scale[i] : 1f; }
        public bool IsHyped(PolicyRunner r) { int i = Index(r); return i >= 0 && _ready && Now < _hypeUntil[i]; }
        double Now => MjScene.InstanceExists && MjScene.Instance.Data != null ? MjScene.Instance.Data->time : 0;

        void OnEnable() { MatchEvents.RoundStarted += OnRound; if (MjScene.InstanceExists) MjScene.Instance.preUpdateEvent += Pre; }
        void OnDisable() { MatchEvents.RoundStarted -= OnRound; if (MjScene.InstanceExists) MjScene.Instance.preUpdateEvent -= Pre; Restore(); }

        void Init()
        {
            var m = MjScene.Instance.Model; _runners = (PolicyRunner[])director.fighters.Clone(); int n = _runners.Length;
            _s = new float[n]; _scale = new float[n]; _hypeUntil = new double[n]; _hypeReady = new double[n]; _mass = new double[n];
            _gain0 = new double[(int)m->nu]; _bias0 = new double[(int)m->nu];
            for (int a = 0; a < (int)m->nu; a++) { _gain0[a] = m->actuator_gainprm[NGain * a]; _bias0[a] = m->actuator_biasprm[NBias * a + 1]; }
            for (int k = 0; k < n; k++) _mass[k] = Math.Max(1, m->body_subtreemass[_runners[k].Map.PelvisBodyId]);
            _ready = true; OnRound();
        }

        void OnRound()
        {
            if (!_ready) return;
            for (int k = 0; k < _s.Length; k++) { _s[k] = 1f; _scale[k] = 1f; _hypeUntil[k] = 0; _hypeReady[k] = 0; }
        }

        /// <summary>Adrenaline for one fighter, if stamina is on, it is fighting and its cooldown is over.</summary>
        public bool TryHype(PolicyRunner r)
        {
            int i = Index(r); if (i < 0 || !Active || director.IsOut(Array.IndexOf(director.fighters, r))) return false;
            double now = Now; if (now < _hypeReady[i]) return false;
            _hypeUntil[i] = now + adrenalineSeconds; _hypeReady[i] = now + cooldownSeconds; _s[i] = Mathf.Min(1f, _s[i] + 0.25f);
            MatchEvents.RaiseHyped(r);
            return true;
        }

        void Pre(object sender, MjStepArgs args)
        {
            if (director == null) return;
            if (!_ready) { if (director.State >= MatchState.Menu && director.fighters.Length > 0 && director.fighters[0].Map != null) Init(); else return; }
            if (!Active) { Restore(); return; }
            var m = MjScene.Instance.Model; var d = MjScene.Instance.Data; float dt = (float)m->opt.timestep; double now = d->time;
            for (int k = 0; k < _runners.Length; k++)
            {
                var map = _runners[k].Map; double p = 0;
                for (int i = 0; i < map.N; i++) p += Math.Abs(d->actuator_force[map.ActId[i]] * d->qvel[map.DofAdr[i]]);
                float s = _s[k] + dt * (-(float)(p / (_mass[k] * fullWattsPerKg)) + recovery * (1f - _s[k]));
                _s[k] = Mathf.Clamp01(s);
                float scale = minKp + (1f - minKp) * _s[k] + (now < _hypeUntil[k] ? adrenalineKp : 0f);
                _scale[k] = Mathf.Clamp(scale, 0.8f, 1.2f);
                for (int i = 0; i < map.N; i++)
                {
                    int a = map.ActId[i];
                    m->actuator_gainprm[NGain * a] = _gain0[a] * _scale[k]; m->actuator_biasprm[NBias * a + 1] = _bias0[a] * _scale[k];
                }
            }
            _applied = true;
        }

        void Restore()
        {
            if (!_applied || !_ready || !MjScene.InstanceExists || MjScene.Instance.Model == null) { _applied = false; return; }
            var m = MjScene.Instance.Model;
            for (int a = 0; a < _gain0.Length; a++) { m->actuator_gainprm[NGain * a] = _gain0[a]; m->actuator_biasprm[NBias * a + 1] = _bias0[a]; }
            for (int k = 0; k < _scale.Length; k++) _scale[k] = 1f;
            _applied = false;
        }
    }
}
