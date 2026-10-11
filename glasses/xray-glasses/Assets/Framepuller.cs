using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

/// <summary>
/// Pulls /frame.jpg one request at a time (never pushes), so the view cannot fall behind.
/// v2: asks for ?after=SEQ so the Jetson answers only when there is a newer frame (no repeats);
/// keeps the placeholder / last good frame on failure; tells NO CAMERA (404/503) from OFFLINE;
/// base URL can come from baseurl.txt (XrayConfig); keeps an AspectRatioFitter on the target in step.
/// </summary>
public class FramePuller : MonoBehaviour
{
    public enum FrameStatus { Waiting, Ok, NoCamera, Offline }

    [Tooltip("Base URL with no trailing slash, e.g. http://127.0.0.1:8080. baseurl.txt overrides it.")]
    public string baseUrl = "http://127.0.0.1:8080";
    public RawImage target;
    [Range(1f, 15f)] public float maxFps = 10f;
    [Tooltip("Ask the server to wait for a newer frame (xray.live supports ?after=SEQ).")]
    public bool waitForNewer = true;

    public string BaseUrl { get; private set; }
    public FrameStatus Status { get; private set; } = FrameStatus.Waiting;
    public string LastSeq { get; private set; } = "";
    public string LastAgeMs { get; private set; } = "";
    public int AgeMs { get; private set; } = -1;
    public float LastFrameAt { get; private set; } = -999f;
    public bool Healthy => Status == FrameStatus.Ok && Time.unscaledTime - LastFrameAt < 3f;

    Texture2D tex;
    bool assigned;

    void Awake() { BaseUrl = XrayConfig.ResolveBaseUrl(baseUrl); }

    IEnumerator Start()
    {
        tex = new Texture2D(2, 2, TextureFormat.RGB24, false);   // one texture, reused for every frame
        float period = 1f / Mathf.Max(1f, maxFps);
        long lastSeq = -1;

        while (true)
        {
            float t0 = Time.unscaledTime;
            string url = BaseUrl + "/frame.jpg" + (waitForNewer && lastSeq >= 0 ? "?after=" + lastSeq : "");
            bool failed = false;
            using (var req = UnityWebRequest.Get(url))
            {
                req.timeout = 5;
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success &&
                    ImageConversion.LoadImage(tex, req.downloadHandler.data, false))
                {
                    if (!assigned && target != null) { target.texture = tex; assigned = true; }  // placeholder stays until now
                    var fitter = target != null ? target.GetComponent<AspectRatioFitter>() : null;
                    if (fitter != null && tex.height > 0) fitter.aspectRatio = (float)tex.width / tex.height;

                    LastSeq = req.GetResponseHeader("X-Seq") ?? "";
                    LastAgeMs = req.GetResponseHeader("X-Age-Ms") ?? "";
                    AgeMs = int.TryParse(LastAgeMs, out var age) ? age : -1;
                    lastSeq = long.TryParse(LastSeq, out var seq) ? seq : -1;
                    LastFrameAt = Time.unscaledTime;
                    Status = FrameStatus.Ok;
                }
                else
                {
                    failed = true;
                    long code = req.responseCode;
                    Status = (code == 404 || code == 503) ? FrameStatus.NoCamera : FrameStatus.Offline;
                    if (Status == FrameStatus.Offline) lastSeq = -1;   // the server may have restarted
                }
            }
            float wait = (failed ? 1f : period) - (Time.unscaledTime - t0);
            if (wait > 0f) yield return new WaitForSecondsRealtime(wait);
            else yield return null;
        }
    }

    void OnDestroy() { if (tex != null) Destroy(tex); }
}