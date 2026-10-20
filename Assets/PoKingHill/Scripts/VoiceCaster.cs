using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace PoKingHill
{
    /// <summary>
    /// Reads the commentary aloud (GameSettings.Voice). Text-to-speech through one of two services, chosen by which
    /// key is in the environment (keys are never stored by the game):
    ///   Azure Speech  AZURE_SPEECH_KEY + AZURE_SPEECH_REGION (optional AZURE_SPEECH_VOICE, default en-US-DavisNeural,
    ///                 excited style), REST endpoint, raw 24 kHz 16-bit mono PCM
    ///   ElevenLabs    ELEVENLABS_API_KEY (optional ELEVENLABS_VOICE_ID), pcm_24000
    /// Listens to MatchEvents.Commentary (one settled line per moment). One line at a time: a line that arrives while
    /// another is being fetched or spoken replaces the waiting one, so the voice never falls behind the match.
    /// </summary>
    [RequireComponent(typeof(AudioSource))]
    public class VoiceCaster : MonoBehaviour
    {
        public DemoDirector director;
        public const int SampleRate = 24000;

        static string Env(string k) => Environment.GetEnvironmentVariable(k);
        public static bool Azure => !string.IsNullOrEmpty(Env("AZURE_SPEECH_KEY")) && !string.IsNullOrEmpty(Env("AZURE_SPEECH_REGION"));
        public static bool Eleven => !string.IsNullOrEmpty(Env("ELEVENLABS_API_KEY"));
        public static bool Available => Azure || Eleven;
        public static string Provider => Azure ? "Azure" : Eleven ? "ElevenLabs" : "none";

        AudioSource _src; string _waiting; bool _busy; AudioClip _clip;
        /// <summary>The voice is talking: ImpactSynth ducks the ambience under it.</summary>
        public static bool Speaking { get; private set; }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => Speaking = false;

        void OnEnable() { _src = GetComponent<AudioSource>(); _src.playOnAwake = false; _src.spatialBlend = 0f; MatchEvents.Commentary += OnLine; }
        void OnDisable() { MatchEvents.Commentary -= OnLine; }

        void OnLine(string text)
        {
            if (!GameSettings.Voice || !Available || director == null || director.statsRounds > 0 || string.IsNullOrWhiteSpace(text)) return;
            _waiting = text;
        }

        void Update()
        {
            Speaking = _src != null && _src.isPlaying;
            if (_src != null && _src.isPlaying) _src.volume = GameSettings.Volume * GameSettings.VoiceVolume;
            if (_busy || _waiting == null || _src.isPlaying) return;
            string line = _waiting; _waiting = null;
            StartCoroutine(Speak(line));
        }

        IEnumerator Speak(string text)
        {
            // a line queued in an earlier play session (fast enter-play keeps the instance) can arrive with no key set now
            if (!Available || !GameSettings.Voice) yield break;
            _busy = true;
            using var req = Azure ? AzureRequest(text) : ElevenRequest(text);
            req.timeout = 15;
            yield return req.SendWebRequest();
            _busy = false;
            if (req.result != UnityWebRequest.Result.Success) { Debug.LogWarning($"[VoiceCaster] {Provider} request failed: {req.responseCode} {req.error}"); yield break; }
            var clip = ToClip(req.downloadHandler.data, "commentary");
            if (clip == null) yield break;
            if (_clip != null) Destroy(_clip);
            _clip = clip; _src.clip = clip; _src.volume = GameSettings.Volume * GameSettings.VoiceVolume; _src.Play();
        }

        static UnityWebRequest AzureRequest(string text)
        {
            string voice = string.IsNullOrEmpty(Env("AZURE_SPEECH_VOICE")) ? "en-US-DavisNeural" : Env("AZURE_SPEECH_VOICE");
            string ssml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>" +
                          $"<voice name='{voice}'><mstts:express-as style='excited'><prosody rate='+10%'>{Xml(text)}</prosody></mstts:express-as></voice></speak>";
            var req = new UnityWebRequest($"https://{Env("AZURE_SPEECH_REGION")}.tts.speech.microsoft.com/cognitiveservices/v1", "POST")
                { uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(ssml)), downloadHandler = new DownloadHandlerBuffer() };
            req.SetRequestHeader("Ocp-Apim-Subscription-Key", Env("AZURE_SPEECH_KEY"));
            req.SetRequestHeader("Content-Type", "application/ssml+xml");
            req.SetRequestHeader("X-Microsoft-OutputFormat", "raw-24khz-16bit-mono-pcm");
            req.SetRequestHeader("User-Agent", "PoKingHill");
            return req;
        }

        static UnityWebRequest ElevenRequest(string text)
        {
            string voice = string.IsNullOrEmpty(Env("ELEVENLABS_VOICE_ID")) ? "JBFqnCBsd6RMkjVDRZzb" : Env("ELEVENLABS_VOICE_ID");
            string body = "{\"text\":\"" + text.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\",\"model_id\":\"eleven_flash_v2_5\"}";
            var req = new UnityWebRequest($"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_24000", "POST")
                { uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body)), downloadHandler = new DownloadHandlerBuffer() };
            req.SetRequestHeader("xi-api-key", Env("ELEVENLABS_API_KEY"));
            req.SetRequestHeader("Content-Type", "application/json");
            return req;
        }

        static string Xml(string s) => s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("'", "&apos;").Replace("\"", "&quot;");

        /// <summary>Raw little-endian 16-bit mono PCM at SampleRate to an AudioClip (null when there is no audio).</summary>
        public static AudioClip ToClip(byte[] pcm, string name)
        {
            if (pcm == null || pcm.Length < 4) return null;
            int n = pcm.Length / 2; var samples = new float[n];
            for (int i = 0; i < n; i++) samples[i] = (short)(pcm[2 * i] | (pcm[2 * i + 1] << 8)) / 32768f;
            var clip = AudioClip.Create(name, n, 1, SampleRate, false); clip.SetData(samples, 0);
            return clip;
        }
    }
}
