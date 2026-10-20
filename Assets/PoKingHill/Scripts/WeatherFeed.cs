using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace PoKingHill
{
    /// <summary>
    /// Live weather for the summit from Open-Meteo (free, no key): GameSettings.WeatherCity is looked up with the
    /// geocoding API, then the current temperature, wind, cloud cover, day/night and WMO weather code are read every
    /// 15 minutes. Presentation and pacing only, never the physics the brains see: MatchFx picks the sky (night,
    /// overcast, storm tint) from it and Hazards sends gusts more often when it is windy (each gust stays the
    /// trained-range 12 % of body weight). The intro card shows the summary line. Off while the city is empty.
    /// </summary>
    public class WeatherFeed : MonoBehaviour
    {
        public float refreshMinutes = 15f;
        public bool Ready { get; private set; }
        public string Summary { get; private set; } = "";
        public string Status { get; private set; } = "";
        public float WindKmh { get; private set; }
        public float TemperatureC { get; private set; }
        public float CloudCover { get; private set; }
        public bool IsDay { get; private set; } = true;
        public int Code { get; private set; }
        /// <summary>Rain, snow showers or thunder (WMO 51-67, 80-82, 95-99).</summary>
        public bool Wet => (Code >= 51 && Code <= 67) || (Code >= 80 && Code <= 82) || Code >= 95;
        public static event Action Updated;

        [Serializable] class Place { public string name, country; public float latitude, longitude; }
        [Serializable] class Places { public Place[] results; }
        [Serializable] class Current { public float temperature_2m, wind_speed_10m, cloud_cover; public int weather_code, is_day; }
        [Serializable] class Forecast { public Current current; }

        string _city = ""; float _next; Coroutine _job;

        void OnEnable() { GameSettings.Changed += Apply; Apply(); }
        void OnDisable() { GameSettings.Changed -= Apply; }

        void Apply()
        {
            string city = GameSettings.WeatherCity;
            if (city == _city) return;
            _city = city; Ready = false; Summary = ""; _next = 0;
            Status = city.Length == 0 ? "" : "looking up " + city;
            if (_job != null) { StopCoroutine(_job); _job = null; }
        }

        void Update()
        {
            if (_city.Length == 0 || _job != null || Time.unscaledTime < _next) return;
            _next = Time.unscaledTime + refreshMinutes * 60f;
            _job = StartCoroutine(Fetch(_city));
        }

        IEnumerator Fetch(string city)
        {
            using (var geo = UnityWebRequest.Get("https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name=" + UnityWebRequest.EscapeURL(city)))
            {
                geo.timeout = 10; yield return geo.SendWebRequest();
                if (geo.result != UnityWebRequest.Result.Success) { Fail("geocoding failed: " + geo.error); yield break; }
                Places found = null; try { found = JsonUtility.FromJson<Places>(geo.downloadHandler.text); } catch { }
                if (found?.results == null || found.results.Length == 0) { Fail($"no place called {city}"); yield break; }
                var p = found.results[0];
                string url = FormattableString.Invariant($"https://api.open-meteo.com/v1/forecast?latitude={p.latitude}&longitude={p.longitude}&current=temperature_2m,wind_speed_10m,cloud_cover,weather_code,is_day");
                using var wx = UnityWebRequest.Get(url); wx.timeout = 10; yield return wx.SendWebRequest();
                if (wx.result != UnityWebRequest.Result.Success) { Fail("forecast failed: " + wx.error); yield break; }
                Forecast f = null; try { f = JsonUtility.FromJson<Forecast>(wx.downloadHandler.text); } catch { }
                if (f?.current == null) { Fail("forecast had no current weather"); yield break; }
                if (city != _city) { _job = null; yield break; }         // the city changed while we were asking
                var c = f.current; WindKmh = c.wind_speed_10m; TemperatureC = c.temperature_2m; CloudCover = c.cloud_cover; IsDay = c.is_day != 0; Code = c.weather_code;
                Summary = $"Live from {p.name}{(string.IsNullOrEmpty(p.country) ? "" : ", " + p.country)}: {TemperatureC:0} °C, wind {WindKmh:0} km/h, {Describe(Code)}{(IsDay ? "" : ", night")}";
                Status = Summary; Ready = true; _job = null;
                Updated?.Invoke();
            }
        }

        void Fail(string why) { Status = why; Ready = false; _job = null; _next = Time.unscaledTime + 60f; Debug.LogWarning("[WeatherFeed] " + why); }

        /// <summary>WMO weather interpretation code in a few words.</summary>
        public static string Describe(int code) => code switch
        {
            0 => "clear sky", 1 => "mainly clear", 2 => "partly cloudy", 3 => "overcast", 45 or 48 => "fog",
            >= 51 and <= 57 => "drizzle", >= 61 and <= 67 => "rain", >= 71 and <= 77 => "snow", >= 80 and <= 82 => "showers",
            85 or 86 => "snow showers", >= 95 => "thunderstorm", _ => "mixed weather",
        };

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetEvents() => Updated = null;
    }
}
