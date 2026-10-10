using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

/// <summary>
/// Pulls /frame.jpg one request at a time (never pushes), so the view cannot fall behind.
/// Put it next to a RawImage.
/// </summary>
public class FramePuller : MonoBehaviour
{
    [Tooltip("Base URL with no trailing slash, e.g. http://192.168.1.50:8080")]
    public string baseUrl = "http://192.168.1.50:8080";
    public RawImage target;
    [Range(1f, 15f)] public float maxFps = 10f;

    public string LastSeq { get; private set; } = "";
    public string LastAgeMs { get; private set; } = "";
    public bool Healthy { get; private set; }

    Texture2D tex;

    IEnumerator Start()
    {
        tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
        if (target != null) target.texture = tex;
        float period = 1f / maxFps;

        while (true)
        {
            float t0 = Time.unscaledTime;
            using (var req = UnityWebRequest.Get(baseUrl + "/frame.jpg"))
            {
                req.timeout = 3;
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                {
                    ImageConversion.LoadImage(tex, req.downloadHandler.data);
                    LastSeq = req.GetResponseHeader("X-Seq") ?? "";
                    LastAgeMs = req.GetResponseHeader("X-Age-Ms") ?? "";
                    Healthy = true;
                }
                else
                {
                    Healthy = false;
                }
            }
            float wait = period - (Time.unscaledTime - t0);
            if (wait > 0f) yield return new WaitForSecondsRealtime(wait);
            else yield return null;
        }
    }
}