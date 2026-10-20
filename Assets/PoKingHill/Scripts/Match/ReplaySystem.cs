using System;
using Mujoco;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Instant replay of the last knock-out. Every second physics step the whole qpos goes into a preallocated ring
    /// buffer (8 s). On request the simulation is paused (MjScene disabled), its full state saved, and the recorded
    /// poses are shown in slow motion from 2 s before the knock-out to 2.5 s after it by writing qpos, running
    /// mj_kinematics and syncing the Unity transforms. Afterwards the saved state is restored exactly and the
    /// simulation resumes as if nothing happened.
    /// </summary>
    public unsafe class ReplaySystem : MonoBehaviour
    {
        public DemoDirector director;
        public float seconds = 8f, before = 2f, after = 2.5f, speed = 0.35f, autoDelay = 0.8f;

        public static bool Active { get; private set; }
        /// <summary>The fighter whose fall is being replayed (the camera frames it).</summary>
        public static PolicyRunner Focus { get; private set; }
        // a knock-out "in the future" means mjData was reset since (MuJoCo resets after a NaN): nothing left to replay
        public bool Available => _koTime >= 0 && _count > 10 && director != null && director.State == MatchState.Result && MjScene.InstanceExists && _koTime <= MjScene.Instance.Data->time;
        public float Progress => Active ? (float)((_playT - _from) / Math.Max(1e-6, _to - _from)) : 0f;
        /// <summary>The HUD's scrubber holds the playhead while a finger is on it: no advance, no end.</summary>
        public bool Hold { get; set; }

        static ReplaySystem _live;
        /// <summary>Ends a running replay (restoring the live state). The director calls it before it places bodies.</summary>
        public static void StopActive() { if (Active && _live != null) _live.Stop(); }

        double[] _frames, _times; int _nq, _head, _count, _tick;
        double _koTime = -1, _from, _to, _playT; PolicyRunner _koRunner; float _autoAt = -1;
        // saved live state
        double[] _qpos, _qvel, _ctrl, _act, _warm, _xfrc; double _time;

        void OnEnable()
        {
            _live = this;
            MatchEvents.Knockout += OnKo; MatchEvents.RoundStarted += OnStart; MatchEvents.RoundEnded += OnEnd;
            if (MjScene.InstanceExists) { MjScene.Instance.postInitEvent += OnInit; MjScene.Instance.postUpdateEvent += Post; if (MjScene.Instance.Model != null) OnInit(null, null); }
        }
        void OnDisable()
        {
            MatchEvents.Knockout -= OnKo; MatchEvents.RoundStarted -= OnStart; MatchEvents.RoundEnded -= OnEnd;
            if (Active) Stop();
            if (MjScene.InstanceExists) { MjScene.Instance.postInitEvent -= OnInit; MjScene.Instance.postUpdateEvent -= Post; }
        }

        void OnInit(object s, MjStepArgs a)
        {
            var m = MjScene.Instance.Model; _nq = (int)m->nq;
            int n = Mathf.CeilToInt(seconds / (2 * Time.fixedDeltaTime));
            _frames = new double[n * _nq]; _times = new double[n];
            _qpos = new double[_nq]; _qvel = new double[m->nv]; _ctrl = new double[m->nu]; _act = new double[Math.Max(1, (int)m->na)];
            _warm = new double[m->nv]; _xfrc = new double[6 * m->nbody];
            _head = _count = 0;
        }

        void OnStart() { _koTime = -1; _koRunner = null; _autoAt = -1; }
        void OnKo(PolicyRunner r) { _koTime = MjScene.Instance.Data->time; _koRunner = r; }
        void OnEnd(RoundResult r) { if (GameSettings.Replays && r.End == RoundEnd.Knockout && _koTime >= 0 && director.statsRounds == 0) _autoAt = Time.unscaledTime + autoDelay; }

        void Post(object s, MjStepArgs a)
        {
            if (_frames == null || director == null || director.State < MatchState.Fight || director.statsRounds > 0) return;
            if (++_tick % 2 != 0) return;
            var d = MjScene.Instance.Data; int n = _times.Length;
            fixed (double* dst = &_frames[_head * _nq]) Buffer.MemoryCopy(d->qpos, dst, _nq * 8, _nq * 8);
            _times[_head] = d->time; _head = (_head + 1) % n; if (_count < n) _count++;
        }

        public void Play()
        {
            if (!Available || Active) return;
            var d = MjScene.Instance.Data; var m = MjScene.Instance.Model;
            double oldest = _times[(_head - _count + _times.Length) % _times.Length];
            _from = Math.Max(oldest, _koTime - before); _to = Math.Min(d->time, _koTime + after); _playT = _from;
            if (_to - _from < 0.5) return;
            Copy(d->qpos, _qpos); Copy(d->qvel, _qvel); Copy(d->ctrl, _ctrl); if (m->na > 0) Copy(d->act, _act);
            Copy(d->qacc_warmstart, _warm); Copy(d->xfrc_applied, _xfrc); _time = d->time;
            MjScene.Instance.enabled = false;                 // no FixedUpdate: physics and every listener stop
            Active = true; Focus = _koRunner; director.BeginReplay();
        }

        public void Stop()
        {
            if (!Active) return;
            var d = MjScene.Instance.Data; var m = MjScene.Instance.Model;
            Restore(_qpos, d->qpos); Restore(_qvel, d->qvel); Restore(_ctrl, d->ctrl); if (m->na > 0) Restore(_act, d->act);
            Restore(_warm, d->qacc_warmstart); Restore(_xfrc, d->xfrc_applied); d->time = _time;
            MujocoLib.mj_forward(m, d); MjScene.Instance.SyncUnityToMjState();
            MjScene.Instance.enabled = true;
            Active = false; Focus = null; Hold = false; director.EndReplay();
        }

        /// <summary>Moves the playhead to a fraction of the replayed window (the HUD's scrubber).</summary>
        public void Seek(float t01)
        {
            if (!Active) return;
            _playT = _from + (_to - _from) * Mathf.Clamp(t01, 0f, 0.999f);
            ShowFrame();
        }

        void Update()
        {
            if (_autoAt > 0 && Time.unscaledTime >= _autoAt) { _autoAt = -1; Play(); }
            if (!Active) return;
            if (director.State != MatchState.Replay) { Stop(); return; }
            if (!Hold) _playT += Time.unscaledDeltaTime * speed;
            if (_playT >= _to && !Hold) { Stop(); return; }
            ShowFrame();
        }

        void ShowFrame()
        {
            // nearest recorded frame at or before the playhead
            int n = _times.Length, best = -1;
            for (int k = 0; k < _count; k++) { int i = (_head - 1 - k + n) % n; if (_times[i] <= _playT) { best = i; break; } }
            if (best < 0) return;
            var d = MjScene.Instance.Data; var m = MjScene.Instance.Model;
            fixed (double* src = &_frames[best * _nq]) Buffer.MemoryCopy(src, d->qpos, _nq * 8, _nq * 8);
            MujocoLib.mj_kinematics(m, d); MujocoLib.mj_comPos(m, d);
            MjScene.Instance.SyncUnityToMjState();
        }

        static void Copy(double* src, double[] dst) { fixed (double* p = dst) Buffer.MemoryCopy(src, p, dst.Length * 8, dst.Length * 8); }
        static void Restore(double[] src, double* dst) { fixed (double* p = src) Buffer.MemoryCopy(p, dst, src.Length * 8, src.Length * 8); }
    }
}
