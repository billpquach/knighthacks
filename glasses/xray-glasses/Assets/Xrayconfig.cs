using System.IO;
using UnityEngine;

/// <summary>
/// Where the glasses get their data from, decided at runtime so changing the server needs no rebuild.
/// If a file named baseurl.txt exists in Application.persistentDataPath, its first line wins
/// (e.g. http://100.69.44.56:8080). Otherwise the URL typed in the Inspector is used.
/// On the Beam Pro, push it with:
///   adb push baseurl.txt /sdcard/Android/data/PACKAGE_NAME/files/baseurl.txt
/// </summary>
public static class XrayConfig
{
    public const string FileName = "baseurl.txt";

    public static string ResolveBaseUrl(string inspectorUrl)
    {
        try
        {
            string path = Path.Combine(Application.persistentDataPath, FileName);
            if (File.Exists(path))
            {
                string line = File.ReadAllText(path).Trim();
                int nl = line.IndexOf('\n');
                if (nl >= 0) line = line.Substring(0, nl).Trim();
                if (line.StartsWith("http"))
                {
                    Debug.Log("XrayConfig: using " + line + " from " + path);
                    return line.TrimEnd('/');
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("XrayConfig: could not read " + FileName + ": " + e.Message);
        }
        return (inspectorUrl ?? "").Trim().TrimEnd('/');
    }
}