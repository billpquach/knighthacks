using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using TMPro;
using UnityEngine;

/// <summary>
/// The glasses HUD: turns the newest /state message (xray.live / replay_server.py) into large, readable text.
/// The x-ray frame already carries Bill's small overlay; this shows the few things that must read at a glance:
/// connection, the target ball's cup, each cup's state (colour-coded), and a big REVEAL RIGHT/WRONG banner.
///
/// State contract (core/xray/track.py, shell.py): phase "scan" | "track"; target_id = the cup holding the yellow
/// ball (null before it is hidden and after the reveal); objects[] {id, label, state, since_seen_s, target,
/// has_ball, balls_inside[]}; ball {id "yellow ball", color, state VISIBLE|INSIDE|OCCLUDED|OUT_OF_VIEW|LOST,
/// inside (cup id), verdict {believed, actual, right, cm} | null, balls[] (every ball, same fields)}.
///
/// Uses TextMeshPro rich text, so one label is enough; give it a second label in "banner" for a separate,
/// bigger reveal banner. Refreshes about 10 times a second.
/// </summary>
public class StateHud : MonoBehaviour
{
    public StateClient client;
    public FramePuller frames;
    public TMP_Text label;
    [Tooltip("Optional: a separate large text for the REVEAL banner. Empty = banner goes at the top of Label.")]
    public TMP_Text banner;
    [Tooltip("How long the REVEAL banner stays up (xray.live shows its own for 5 s).")]
    public float bannerSeconds = 6f;
    public int maxObjects = 6;

    // Bright colours: on the glasses dark pixels are see-through, so avoid dark text.
    const string Green = "#4CFF6A", Yellow = "#FFD84C", Grey = "#B4B4B4", Red = "#FF5A5A", Cyan = "#4CD2FF", Amber = "#FFB040";

    readonly StringBuilder sb = new StringBuilder(1024);
    readonly Dictionary<string, string> lastVerdicts = new Dictionary<string, string>();
    bool primed;                 // the first message only records verdicts (no banner for an old reveal)
    string bannerText = "";
    float bannerUntil = -1f, nextRefresh;

    void Update()
    {
        if (label == null || Time.unscaledTime < nextRefresh) return;
        nextRefresh = Time.unscaledTime + 0.1f;

        JObject s = client != null ? client.Latest : null;
        if (s != null) WatchVerdicts(s);

        sb.Clear();
        AppendStatus(s);
        if (s == null)
        {
            sb.Append("\n<size=120%>").Append(client != null && client.Connected ? "Waiting for state..." : "Connecting to " + (client != null ? client.BaseUrl : "?")).Append("</size>");
        }
        else if (Str(s["phase"]) == "scan")
        {
            sb.Append("\n<size=150%><color=").Append(Cyan).Append(">SCANNING</color></size>\nWalk slowly around the table\n");
        }
        else if (Str(s["phase"]) == "ask")
        {
            AppendAsk(s);
        }
        else if (Str(s["phase"]) == "starting")
        {
            AppendStarting(s);
        }
        else
        {
            AppendTarget(s);
            AppendObjects(s);
            AppendOtherBalls(s);
        }

        bool showBanner = Time.unscaledTime < bannerUntil;
        if (banner != null)
        {
            banner.text = showBanner ? bannerText : "";
            label.text = sb.ToString();
        }
        else
        {
            label.text = showBanner ? "<size=170%>" + bannerText + "</size>\n" + sb : sb.ToString();
        }
    }

    // ------------------------------------------------------------------ parts
    void AppendStatus(JObject s)
    {
        bool live = client != null && client.Live;
        if (live) sb.Append("<color=").Append(Green).Append(">LIVE</color>");
        else
        {
            sb.Append("<color=").Append(Red).Append(">NO SIGNAL</color>");
            if (client != null && client.Latest != null)
                sb.Append(" (").Append((Time.unscaledTime - client.LatestAt).ToString("0", CultureInfo.InvariantCulture)).Append(" s)");
        }
        float? fps = s != null ? (float?)s["fps"] : null;
        if (fps.HasValue) sb.Append("  ").Append(fps.Value.ToString("0", CultureInfo.InvariantCulture)).Append(" fps");
        if (frames != null)
        {
            switch (frames.Status)
            {
                case FramePuller.FrameStatus.NoCamera: sb.Append("  <color=").Append(Grey).Append(">NO CAMERA</color>"); break;
                case FramePuller.FrameStatus.Offline: sb.Append("  <color=").Append(Red).Append(">NO FRAMES</color>"); break;
                case FramePuller.FrameStatus.Ok:
                    if (!frames.Healthy) sb.Append("  <color=").Append(Yellow).Append(">FRAMES STALLED</color>");
                    else if (frames.AgeMs >= 0) sb.Append("  cam ").Append(frames.AgeMs).Append(" ms");
                    break;
            }
        }
        sb.Append('\n');
    }

    void AppendTarget(JObject s)
    {
        var ball = s["ball"] as JObject;
        if (ball != null)
        {
            string name = Cap(Str(ball["color"]) ?? "") + " ball";
            string st = Str(ball["state"]) ?? "?";
            sb.Append("\n<size=140%>");
            if (st == "INSIDE")
                sb.Append("<color=").Append(Yellow).Append(">").Append(name).Append(" -> under CUP ").Append(Str(ball["inside"]) ?? "?").Append("</color>");
            else
                sb.Append("<color=").Append(StateColour(st)).Append(">").Append(name).Append(": ").Append(Words(st)).Append("</color>");
            sb.Append("</size>\n");
            return;
        }
        string tid = Str(s["target_id"]);                       // no shell game: a target picked by Qwen-3D
        if (tid == null) return;
        if (s["objects"] is JArray objs)
            foreach (var o in objs)
                if (Str(o["id"]) == tid)
                {
                    string st = Str(o["state"]) ?? "?";
                    sb.Append("\n<size=140%><color=").Append(Amber).Append(">TARGET: ").Append(Str(o["label"]) ?? "object").Append(' ').Append(tid)
                      .Append("</color>  <color=").Append(StateColour(st)).Append(">").Append(Words(st)).Append("</color></size>\n");
                    return;
                }
    }

    void AppendObjects(JObject s)
    {
        if (!(s["objects"] is JArray objs) || objs.Count == 0) return;
        string tid = Str(s["target_id"]);
        sb.Append('\n');
        int n = 0;
        foreach (var o in objs)
        {
            if (n++ >= maxObjects) { sb.Append("  ...\n"); break; }
            string id = Str(o["id"]) ?? "?";
            string st = Str(o["state"]) ?? "?";
            bool isTarget = (bool?)o["target"] == true || (tid != null && id == tid);
            sb.Append(isTarget ? "<color=" + Amber + ">> </color>" : "   ");
            sb.Append("<color=").Append(StateColour(st)).Append(">").Append(Str(o["label"]) ?? "object").Append(' ').Append(id)
              .Append("  ").Append(Words(st));
            if (st != "VISIBLE")
            {
                float? since = (float?)o["since_seen_s"];
                if (since.HasValue) sb.Append(' ').Append(since.Value.ToString("0", CultureInfo.InvariantCulture)).Append('s');
            }
            sb.Append("</color>");
            if (o["balls_inside"] is JArray inside && inside.Count > 0)
            {
                sb.Append("  (");
                for (int i = 0; i < inside.Count; i++) sb.Append(i > 0 ? " + " : "").Append(Str(inside[i]));
                sb.Append(" inside)");
            }
            sb.Append('\n');
        }
    }

    void AppendOtherBalls(JObject s)
    {
        var ball = s["ball"] as JObject;
        if (ball == null || !(ball["balls"] is JArray balls)) return;
        string targetId = Str(ball["id"]);
        foreach (var b in balls)
        {
            if (Str(b["id"]) == targetId) continue;
            string st = Str(b["state"]) ?? "?";
            sb.Append("   <color=").Append(Grey).Append(">").Append(Str(b["id"]) ?? "ball").Append(": ")
              .Append(st == "INSIDE" ? "under cup " + (Str(b["inside"]) ?? "?") : Words(st)).Append("</color>\n");
        }
    }

    // ------------------------------------------------------------------ xray.live --ask: Qwen-3D questions
    // phase "ask": objects = answers q1, q2, ... {query, score, band, seconds, target (the newest)};
    // "qwen" = {status, ready, thinking, answers}. phase "starting": Qwen-3D done, YOLOE loading.
    void AppendAsk(JObject s)
    {
        var q = s["qwen"] as JObject;
        bool ready = q != null && (bool?)q["ready"] == true;
        string thinking = q != null ? Str(q["thinking"]) : null;
        string status = (q != null ? Str(q["status"]) : null) ?? "Qwen-3D";
        sb.Append("\n<size=140%><color=").Append(Cyan).Append(">ASK QWEN-3D</color></size>\n");
        sb.Append("<color=").Append(thinking != null ? Yellow : ready ? Green : Grey).Append(">").Append(status).Append("</color>\n");
        AppendAnswers(s, 4);
    }

    void AppendStarting(JObject s)
    {
        sb.Append("\n<size=140%><color=").Append(Cyan).Append(">STARTING TRACKING</color></size>\nQwen-3D done, loading YOLOE...\n");
        AppendAnswers(s, 1);
    }

    void AppendAnswers(JObject s, int max)
    {
        if (!(s["objects"] is JArray objs) || objs.Count == 0) return;
        sb.Append('\n');
        for (int i = Mathf.Max(0, objs.Count - max); i < objs.Count; i++)
        {
            var o = objs[i];
            bool target = (bool?)o["target"] == true;
            float? score = (float?)o["score"];
            string band = Str(o["band"]);
            sb.Append(target ? "<color=" + Amber + ">> " : "<color=" + Grey + ">   ")
              .Append(target ? "TARGET: " : "").Append('"').Append(Str(o["query"]) ?? Str(o["label"]) ?? "answer").Append('"');
            if (score.HasValue) sb.Append("  ").Append(score.Value.ToString("0.00", CultureInfo.InvariantCulture));
            if (band != null) sb.Append(" (").Append(band).Append(')');
            sb.Append("</color>\n");
        }
    }

    // ------------------------------------------------------------------ the reveal banner
    void WatchVerdicts(JObject s)
    {
        var ball = s["ball"] as JObject;
        if (ball == null) return;
        var all = ball["balls"] as JArray;
        IEnumerable<JToken> list = all != null && all.Count > 0 ? (IEnumerable<JToken>)all : new JToken[] { ball };
        foreach (var b in list)
        {
            string key = Str(b["id"]) ?? "ball";
            var v = b["verdict"] as JObject;
            string sig = v != null ? v.ToString(Formatting.None) : null;
            lastVerdicts.TryGetValue(key, out var prev);
            if (primed && v != null && sig != prev)
            {
                bool right = (bool?)v["right"] == true;
                bannerText = "<color=" + (right ? Green : Red) + ">REVEAL: " + (right ? "RIGHT" : "WRONG") + "</color>\n"
                           + "<size=55%>" + Cap(key) + ": we said cup " + (Str(v["believed"]) ?? "?") + ", it was cup " + (Str(v["actual"]) ?? "?") + "</size>";
                bannerUntil = Time.unscaledTime + bannerSeconds;
            }
            lastVerdicts[key] = sig;
        }
        primed = true;
    }

    // ------------------------------------------------------------------ helpers
    static string Str(JToken t) => t == null || t.Type == JTokenType.Null ? null : t.ToString();

    static string Cap(string s) => string.IsNullOrEmpty(s) ? s : char.ToUpperInvariant(s[0]) + s.Substring(1);

    static string Words(string state) => state == "OUT_OF_VIEW" ? "OUT OF VIEW" : state;

    static string StateColour(string state)
    {
        switch (state)
        {
            case "VISIBLE": return Green;
            case "OCCLUDED": return Yellow;
            case "OUT_OF_VIEW": return Grey;
            case "LOST": return Red;
            case "INSIDE": return Cyan;
            default: return Grey;
        }
    }
}