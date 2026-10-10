using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Text;
using System.Xml.Linq;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Reads the Jetson viewer's /state stream (server-sent events, about 10 Hz) and exposes the newest message.
/// Works against replay_server.py on the laptop or the real Jetson at the same URL.
/// </summary>
public class StateClient : MonoBehaviour
{
    [Tooltip("Base URL with no trailing slash, e.g. http://192.168.1.50:8080")]
    public string baseUrl = "http://192.168.1.50:8080";
    public float reconnectDelay = 2f;

    public JObject Latest { get; private set; }
    public float LatestAt { get; private set; }
    public bool Connected { get; private set; }
    public event Action<JObject> OnState;

    readonly ConcurrentQueue<string> inbox = new ConcurrentQueue<string>();

    // Receives bytes on a worker thread and splits them into SSE events.
    class SseHandler : DownloadHandlerScript
    {
        readonly ConcurrentQueue<string> queue;
        readonly StringBuilder buffer = new StringBuilder();
        readonly Decoder decoder = Encoding.UTF8.GetDecoder();

        public SseHandler(ConcurrentQueue<string> q) : base(new byte[16384]) { queue = q; }

        protected override bool ReceiveData(byte[] data, int length)
        {
            if (data == null || length == 0) return false;
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

    void OnEnable() { StartCoroutine(Run()); }
    void OnDisable() { StopAllCoroutines(); Connected = false; }

    IEnumerator Run()
    {
        while (true)
        {
            using (var req = new UnityWebRequest(baseUrl + "/state", UnityWebRequest.kHttpVerbGET))
            {
                req.downloadHandler = new SseHandler(inbox);
                req.SetRequestHeader("Accept", "text/event-stream");
                req.timeout = 0; // never time out a live stream
                Connected = true;
                yield return req.SendWebRequest();
                Connected = false;
                if (req.result != UnityWebRequest.Result.Success)
                    Debug.LogWarning("StateClient: " + req.error);
            }
            yield return new WaitForSecondsRealtime(reconnectDelay);
        }
    }

    void Update()
    {
        string last = null;
        while (inbox.TryDequeue(out var msg)) last = msg;
        if (last == null) return;
        try
        {
            Latest = JObject.Parse(last);
            LatestAt = Time.unscaledTime;
            OnState?.Invoke(Latest);
        }
        catch (Exception e)
        {
            Debug.LogWarning("StateClient parse error: " + e.Message);
        }
    }
}