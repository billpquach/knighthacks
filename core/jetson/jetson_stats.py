"""What the Jetson is doing while something runs: RAM, GPU load, CPU load, temperature and board
power (tegrastats), plus RAM from /proc/meminfo (that part works on any Linux, without tegrastats).

    python3 jetson_stats.py                 # one line per second until Ctrl-C
    python3 jetson_stats.py --json          # the same as JSON lines (for a dashboard or a log)

From Python:
    from jetson_stats import Sampler
    s = Sampler(interval=0.5)               # samples in a background thread from now on
    s.mark("model load")                    # names what happens from now on
    ...
    s.mark("queries")
    s.stop()
    s.segments()                            # per phase: seconds, RAM peak, GPU %, mean/peak W, energy J, peak C
    s.latest()                              # the newest sample (for a live panel)

On the Jetson the CPU and the GPU share one RAM, so "RAM used" is everything: model weights, CUDA
buffers and the rest. RAM comes from /proc/meminfo (MemTotal - MemAvailable, the same number the
Qwen-3D runner and the live runner report); the rest from tegrastats. Fields tegrastats doesn't
print, or a machine without it, come out as None.
"""
import argparse
import json
import re
import shutil
import subprocess
import threading
import time
from collections import deque

RE_RAM = re.compile(r"\bRAM (\d+)/(\d+)MB")
RE_SWAP = re.compile(r"\bSWAP (\d+)/(\d+)MB")
RE_CPU = re.compile(r"\bCPU \[([^\]]*)\]")
RE_GPU = re.compile(r"\bGR3D_FREQ (\d+)%")
RE_TEMP = re.compile(r"\b([A-Za-z]\w*)@(-?\d+(?:\.\d+)?)C\b")
RE_RAIL = re.compile(r"\b([A-Z][A-Z0-9_]*) (\d+)mW/(\d+)mW")
TOTAL_RAILS = ("VDD_IN", "VIN_SYS_5V0", "POM_5V_IN")      # whole-board input: Orin Nano/NX, AGX, older


def parse_tegrastats(line):
    """One tegrastats line -> dict. Keys are present only for the fields the line has."""
    s = {}
    m = RE_RAM.search(line)
    if m:
        s["ram_used_mb"], s["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = RE_SWAP.search(line)
    if m:
        s["swap_used_mb"] = int(m.group(1))
    m = RE_CPU.search(line)
    if m:
        loads = [int(c.split("%")[0]) for c in m.group(1).split(",") if "%" in c]     # "off" cores skipped
        if loads:
            s["cpu_pct"] = round(sum(loads) / len(loads), 1)
    m = RE_GPU.search(line)
    if m:
        s["gpu_pct"] = int(m.group(1))
    temps = {k.lower(): float(v) for k, v in RE_TEMP.findall(line)}
    temps = {k: v for k, v in temps.items() if -40 < v < 150 and k != "pmic"}   # -256: sensor off; PMIC: fixed 100
    if temps:
        s["temps_c"] = temps
        s["temp_c"] = temps.get("tj", max(temps.values()))         # tj: the hottest point of the chip
    rails = {name: int(now) for name, now, _avg in RE_RAIL.findall(line)}
    if rails:
        s["rails_mw"] = rails
        total = next((rails[r] for r in TOTAL_RAILS if r in rails), None)
        if total is not None:
            s["power_w"] = round(total / 1000, 2)
    return s


def meminfo_gb():
    """(used, total) GB, used = MemTotal - MemAvailable."""
    m = {}
    with open("/proc/meminfo") as f:
        for ln in f:
            k, v = ln.split(":", 1)
            m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) / 1024 ** 2, m["MemTotal"] / 1024 ** 2


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _r(x, nd=2):
    return None if x is None else round(x, nd)


class Sampler:
    """Samples every `interval` s in a background thread: RAM from /proc/meminfo, everything else
    from the newest tegrastats line (tegrastats runs as its own process while the sampler runs)."""

    def __init__(self, interval=0.5, tegrastats=True, keep=20000):
        self.interval = interval
        self.t0 = time.monotonic()
        self.samples = deque(maxlen=keep)
        self.marks = [(0.0, "start")]
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.ts_last, self.ts_lines = None, 0
        self.proc = None
        exe = shutil.which("tegrastats") if tegrastats else None
        if exe:
            ms = str(max(100, int(interval * 1000)))
            # stdbuf: tegrastats writes to a pipe through stdio, which would otherwise hold lines back
            cmd = (["stdbuf", "-oL"] if shutil.which("stdbuf") else []) + [exe, "--interval", ms]
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                             text=True, bufsize=1, start_new_session=True)
                threading.Thread(target=self._read, daemon=True).start()
            except OSError:
                self.proc = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def has_tegrastats(self):
        return self.ts_lines > 0

    def _read(self):
        for line in self.proc.stdout:
            s = parse_tegrastats(line)
            if s:
                self.ts_last = (time.monotonic(), s)
                self.ts_lines += 1

    def _sample(self):
        now = time.monotonic()
        s = {"t": round(now - self.t0, 3)}
        try:
            used, total = meminfo_gb()
            s["ram_used_gb"], s["ram_total_gb"] = round(used, 3), round(total, 3)
        except (OSError, KeyError, ValueError):
            pass
        last = self.ts_last
        if last is not None and now - last[0] < max(2.5 * self.interval, 1.5):     # stale lines are not used
            ts = last[1]
            for k in ("gpu_pct", "cpu_pct", "temp_c", "power_w", "swap_used_mb"):
                if k in ts:
                    s[k] = ts[k]
        return s

    def _run(self):
        while not self.stopped.is_set():
            s = self._sample()
            with self.lock:
                self.samples.append(s)
            self.stopped.wait(self.interval)

    def mark(self, name):
        """Everything sampled from now until the next mark belongs to phase `name`."""
        with self.lock:
            self.marks.append((time.monotonic() - self.t0, name))

    def latest(self):
        with self.lock:
            return dict(self.samples[-1]) if self.samples else self._sample()

    def stop(self):
        if self.stopped.is_set():
            return
        with self.lock:
            self.samples.append(self._sample())        # one last sample at the very end
        self.stopped.set()
        self.thread.join(timeout=2)
        self.t_end = time.monotonic() - self.t0
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()

    def segments(self):
        """One dict per phase (the time between one mark and the next), in order."""
        end = getattr(self, "t_end", time.monotonic() - self.t0)
        with self.lock:
            samples, marks = list(self.samples), list(self.marks)
        out = []
        for i, (t_a, name) in enumerate(marks):
            t_b = marks[i + 1][0] if i + 1 < len(marks) else end
            seg = [s for s in samples if t_a <= s["t"] < t_b or (i + 1 == len(marks) and s["t"] >= t_a)]
            if t_b - t_a <= 0:
                continue
            ram = [s["ram_used_gb"] for s in seg if "ram_used_gb" in s]
            gpu = [s["gpu_pct"] for s in seg if "gpu_pct" in s]
            cpu = [s["cpu_pct"] for s in seg if "cpu_pct" in s]
            pw = [s["power_w"] for s in seg if "power_w" in s]
            temp = [s["temp_c"] for s in seg if "temp_c" in s]
            p_mean = _mean(pw)
            out.append(dict(phase=name, start_s=_r(t_a), seconds=_r(t_b - t_a), samples=len(seg),
                            ram_peak_gb=_r(max(ram) if ram else None), ram_mean_gb=_r(_mean(ram)),
                            gpu_mean_pct=_r(_mean(gpu), 0), gpu_max_pct=max(gpu) if gpu else None,
                            cpu_mean_pct=_r(_mean(cpu), 0),
                            power_mean_w=_r(p_mean), power_peak_w=_r(max(pw) if pw else None),
                            energy_j=_r(p_mean * (t_b - t_a), 0) if p_mean is not None else None,
                            temp_peak_c=_r(max(temp) if temp else None, 1)))
        return out


def line(s):
    """A sample as one short human line."""
    parts = []
    if "ram_used_gb" in s:
        parts.append(f"RAM {s['ram_used_gb']:.2f}/{s['ram_total_gb']:.2f} GB")
    for k, fmt in (("gpu_pct", "GPU {:.0f}%"), ("cpu_pct", "CPU {:.0f}%"), ("power_w", "{:.1f} W"), ("temp_c", "{:.1f} C")):
        if s.get(k) is not None:
            parts.append(fmt.format(s[k]))
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--json", action="store_true", help="JSON lines instead of text")
    args = ap.parse_args()
    s = Sampler(args.interval)
    try:
        time.sleep(min(1.0, args.interval) + 0.2)          # let tegrastats print its first line
        while True:
            x = s.latest()
            print(json.dumps(x) if args.json else line(x), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        s.stop()


if __name__ == "__main__":
    main()
