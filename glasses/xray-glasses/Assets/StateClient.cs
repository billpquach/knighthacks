using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Text;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Reads the /state stream (server-sent events, up to 10 a second) from xray.live on the Jetson or
/// replay_server.py on the laptop (same endpoints) and exposes the newest message.
/// v2: base URL can come from baseurl.txt (XrayConfig); the live point cloud ("scene", ~1500 points per
/// message, only the web dashboard uses it) is cut off before parsing to keep the Beam Pro's frame rate up;
/// Live tells whether a message arrived recently.
/// </summary>
public class StateClient : MonoBehaviour
{
    [Tooltip("Base URL with no trailing slash, e.g. http://127.0.0.1:8080. baseurl.txt overrides it.")]
    public string baseUrl = "http://127.0.0.1:8080";
    [Tooltip("Stream path. ?cloud=0 asks xray.live to leave out the point clouds (replay_server.py ignores it).")]
    public string statePath = "/state?cloud=0";
    public float reconnectDelay = 2f;
    [Tooltip("Drop the per-message point cloud before parsing, if the server sent one anyway.")]
    public bool stripScene = true;
    [Tooltip("Seconds without a message before the stream counts as stale.")]
    public float staleAfter = 3f;

    public string BaseUrl { get; private set; }
    public JObject Latest { get; private set; }
    public float LatestAt { get; private set; } = -999f;
    public bool Connected { get; private set; }
    public bool Live => Latest != null && Time.unscaledTime - LatestAt < staleAfter;
    public event Action<JObject> OnState;

    readonly ConcurrentQueue<string> inbox = new ConcurrentQueue<string>();

    // Receives bytes on a worker thread and splits them into SSE events.
    class SseHandler : DownloadHandlerScript
    {
        readonly ConcurrentQueue<string> queue;
        readonly StringBuilder buffer = new StringBuilder();
        readonly Decoder decoder = Encoding.UTF8.GetDecoder();
        public volatile bool gotData;

        public SseHandler(ConcurrentQueue<string> q) : base(new byte[65536]) { queue = q; }

        protected override bool ReceiveData(byte[] data, int length)
        {
            if (data == null || length == 0) return false;
            gotData = true;
            var chars = new char[length + 4];
            int n = decoder.GetChars(data, 0, length, chars, 0);
            buffer.Append(chars, 0, n);

            string s = buffer.ToString();
            int i;
            while ((i = s.IndexOf("\n\n", StringComparison.Ordinal)) >= 0)
            {
                string ev = s.Substring(0, i);
                s = s.Substring(i + 2);
                foreach (var line in ev.Split('\n'))
                {
                    if (line.StartsWith("data:", StringComparison.Ordinal))
                        queue.Enqueue(line.Substring(5).Trim());
                }
            }
            buffer.Clear();
            buffer.Append(s);
            return true;
        }
    }

    void Awake() { BaseUrl = XrayConfig.ResolveBaseUrl(baseUrl); }
    void OnEnable() { StartCoroutine(Run()); }
    void OnDisable() { StopAllCoroutines(); Connected = false; }

    IEnumerator Run()
    {
        while (true)
        {
            string path = string.IsNullOrEmpty(statePath) ? "/state" : statePath;
            using (var req = new UnityWebRequest(BaseUrl + path, UnityWebRequest.kHttpVerbGET))
            {
                var handler = new SseHandler(inbox);
                req.downloadHandler = handler;
                req.SetRequestHeader("Accept", "text/event-stream");
                req.timeout = 0; // never time out a live stream
                var op = req.SendWebRequest();
                while (!op.isDone)
                {
                    Connected = handler.gotData;   // connected once bytes actually arrive
                    yield return null;
                }
                Connected = false;
                if (req.result != UnityWebRequest.Result.Success)
                    Debug.LogWarning("StateClient: " + req.error + " (" + BaseUrl + ")");
            }
            yield return new WaitForSecondsRealtime(reconnectDelay);
        }
    }

    void Update()
    {
        string last = null;
        while (inbox.TryDequeue(out var msg)) last = msg;   // only the newest one matters
        if (last == null) return;
        try
        {
            Latest = Parse(last);
            LatestAt = Time.unscaledTime;
            OnState?.Invoke(Latest);
        }
        catch (Exception e)
        {
            Debug.LogWarning("StateClient parse error: " + e.Message);
        }
    }

    JObject Parse(string json)
    {
        if (stripScene)
        {
            // xray.live adds "scene" (the live cloud) as the message's last key: cut it off before parsing.
            int i = json.LastIndexOf(",\"scene\":", StringComparison.Ordinal);
            if (i > 0)
            {
                try { return JObject.Parse(json.Substring(0, i) + "}"); }
                catch (Exception) { /* not where we expected: parse the whole thing */ }
            }
        }
        return JObject.Parse(json);
    }
}