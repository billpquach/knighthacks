using System.Text;
using Newtonsoft.Json.Linq;
using TMPro;
using UnityEngine;

/// <summary>
/// Turns the newest /state message into readable text for the glasses.
/// Fields follow Bill's state contract: phase, fps, target_id, objects[], ball.
/// </summary>
public class StateHud : MonoBehaviour
{
    public StateClient client;
    public FramePuller frames;
    public TMP_Text label;

    void OnEnable() { if (client != null) client.OnState += Render; }
    void OnDisable() { if (client != null) client.OnState -= Render; }

    void Update()
    {
        if (client != null && label != null && client.Latest == null)
            label.text = client.Connected ? "Waiting for state..." : "Not connected to " + client.baseUrl;
    }

    void Render(JObject s)
    {
        if (label == null) return;
        var sb = new StringBuilder();

        string phase = (string)s["phase"] ?? "?";
        float? fps = (float?)s["fps"];
        sb.Append(phase.ToUpper());
        if (fps.HasValue) sb.Append("  ").Append(fps.Value.ToString("0")).Append(" fps");
        if (frames != null && !string.IsNullOrEmpty(frames.LastSeq))
            sb.Append("  frame ").Append(frames.LastSeq);
        sb.AppendLine();

        int targetId = (int?)s["target_id"] ?? -1;
        var objs = s["objects"] as JArray;
        if (objs != null)
        {
            foreach (var o in objs)
            {
                int id = (int?)o["id"] ?? -1;
                string state = (string)o["state"] ?? "?";
                float since = (float?)o["since_seen_s"] ?? 0f;
                bool isTarget = id == targetId;
                sb.Append(isTarget ? "> " : "  ")
                  .Append("cup ").Append(id).Append("  ").Append(state);
                if (state != "VISIBLE") sb.Append("  seen ").Append(since.ToString("0")).Append("s ago");
                if ((bool?)o["has_ball"] == true) sb.Append("  has ball");
                sb.AppendLine();
            }
        }

        var ball = s["ball"] as JObject;
        if (ball != null)
        {
            string color = (string)ball["color"] ?? "ball";
            string bstate = (string)ball["state"] ?? "?";
            sb.Append(color).Append(" ball: ").Append(bstate);
            if (bstate == "INSIDE") sb.Append(" cup ").Append((int?)ball["inside"] ?? -1);
            var verdict = ball["verdict"];
            if (verdict != null && verdict.Type == JTokenType.Object)
            {
                bool right = (bool?)verdict["right"] ?? false;
                sb.Append("  REVEAL ").Append(right ? "RIGHT" : "WRONG");
            }
            sb.AppendLine();
        }

        label.text = sb.ToString();
    }
}
