"""Qwen-3D benchmark on the Jetson: runs the Qwen-3D runner (jetson_init.py, one-shot mode) on a set of
scans and queries, and reports what each answer costs and whether it is right.

For every scan (and every --frames / --quant setting asked for) it starts the runner once, the way a
real INIT starts it, and measures:
  * time: Python + torch import, model load, scene prep (the ViT and the rest), each query, and the
    preloaded INIT time (scene prep + first query: what the live demo waits for after the scan)
  * memory: RAM peak (the runner's own, and seen from outside at 10 Hz), out-of-memory retries, NvMap
    memory-pressure warnings
  * with tegrastats: GPU load, board power (W), energy (J) and peak temperature, per phase
  * answers: score, confidence band and centre of every query, checked against what a cases file
    expects, and against the laptop's answers when the scan has them (<scan>/results/)

Run it from the repo, with the venv that runs Qwen-3D:
    source ~/q3d/bin/activate
    python3 core/qwen3d_jetson/bench.py ~/permanence/runs/live_183221/scan -q "the cup next to the can" -q "the cup next to the ball"
    python3 core/qwen3d_jetson/bench.py --cases core/qwen3d_jetson/cases.example.json
    python3 core/qwen3d_jetson/bench.py --cases my_cases.json --frames 15 10      # also with 10 frames: less RAM, faster
    python3 core/qwen3d_jetson/bench.py --cases my_cases.json --dry-run           # checks and commands only

The runner and the weights stay where they are (--runner, default ~/qwen3d_jetson_test). Each scan is
linked into the output folder, so the runner's answers land there and the scan itself is not touched.
Before a run: reboot, desktop off (sudo systemctl isolate multi-user.target), sudo jetson_clocks,
nothing else on the GPU (it refuses to start next to a live run or another Qwen-3D). Outputs in
runs/qwen_bench_HHMMSS/: results.json, summary.md, and per run a .log and a .samples.jsonl (one
sample per 0.5 s: RAM, GPU, W, C). Send back everything after "=== BENCH SUMMARY".
"""
import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jetson"))
try:
    import jetson_stats
except ImportError:          # bench.py copied somewhere on its own: no GPU / power numbers, the rest works
    jetson_stats = None

FOUND_SCORE, LOW_SCORE = 0.5, 0.3      # the runner's confidence bands: high / low / none
INIT_BUDGET_S = 40                     # live demo: INIT answers within 40 s of the end of the scan
CENTER_TOL_CM = 3.0                    # the runner's own rule for "the same answer": centres within 3 cm
SCAN_NEEDS = ("color", "depth", "conf", "intrinsic", "pose")
SCAN_EXTRA = ("meta.json", "results")  # results/: the laptop's answers, which the runner compares with
RUNNER_NEEDS = ("jetson_init.py", "run_test.py", "scene_prep.py", "testpack.pt", "heads.pth", "qwen")
HEAVY = ("permanence.live_test", "permanence.replay", "jetson_init.py")

# the runner's output lines (jetson_init.py, one-shot mode)
RE_TORCH = re.compile(r"^torch (\S+), (.+), RAM in use ([\d.]+) of ([\d.]+) GB$")
RE_WEIGHTS = re.compile(r"^\s+weights\s+(\d+)/(\d+):")
RE_LOADED = re.compile(r"^model loaded in (\d+) s")
RE_RETRY = re.compile(r"^out of memory during (.+?) with (\d+) frames .*retrying with (\d+) frames")
RE_SCENE = re.compile(r"^scene: (\d+) frames -> (\d+) point tokens, (\d+) 2 cm voxels; prep ([\d.]+) s \((.*)\), "
                      r"RAM in use ([\d.]+) GB")
RE_QUERY = re.compile(r"^  '(?P<query>.*)' \(target '(?P<target>.*)'\): score (?P<score>-?[\d.]+) \((?P<band>\w+)\), "
                      r"(?P<points>\d+) points(?: at \[(?P<center>[^\]]*)\])?, (?P<seconds>[\d.]+) s"
                      r"(?: \| (?P<laptop>.*))?$")
RE_DEVICE = re.compile(r"^device: (.+)$")
RE_RAM = re.compile(r"^RAM in use: idle ([\d.]+) GB, peak ([\d.]+) GB of ([\d.]+) GB")
RE_FATAL = re.compile(r"^OUT OF MEMORY during (.+?):")
NVMAP = "NvMapMemAllocInternalTagged"


def band(score):
    return "high" if score >= FOUND_SCORE else "low" if score >= LOW_SCORE else "none"


def slug(q):
    """The runner's file name for a query's answer (jetson_init.slug)."""
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:40]


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-") or "run"


def used_gb():
    """System RAM in use (MemTotal - MemAvailable): on the Jetson the GPU allocates from it too."""
    m = {}
    with open("/proc/meminfo") as f:
        for ln in f:
            k, v = ln.split(":", 1)
            m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) / 1024 ** 2


class RamPeak(threading.Thread):
    """RAM in use, sampled every 0.1 s from outside the runner (its own peak is lost if it crashes)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = self.now = used_gb()
        self.done = threading.Event()
        self.start()

    def run(self):
        while not self.done.wait(0.1):
            self.now = used_gb()
            self.peak = max(self.peak, self.now)

    def stop(self):
        self.done.set()
        self.join(timeout=1)
        return self.peak


# ---------------------------------------------------------------------------------------- checks before running
def heavy_processes():
    """Other heavy GPU processes ([(pid, command line)]): the 8 GB fit one at a time."""
    found, skip = [], {os.getpid(), os.getppid()}
    for p in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(p.name)
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError):
            continue
        if pid in skip or not argv or b"python" not in Path(argv[0].decode(errors="ignore")).name.encode():
            continue
        cmd = b" ".join(argv).decode(errors="ignore").strip()
        if any(h in cmd for h in HEAVY):
            found.append((pid, cmd[:160]))
    return found


def board_info():
    """What the numbers depend on: L4T release, power mode, desktop on or off, RAM already in use."""
    info = {"ram_used_gb": round(used_gb(), 2), "tegrastats": bool(shutil.which("tegrastats"))}
    try:
        first = Path("/etc/nv_tegra_release").read_text().splitlines()[0]
        m = re.search(r"R(\d+) \(release\), REVISION: ([\d.]+)", first)
        info["l4t"] = f"R{m.group(1)}.{m.group(2)}" if m else first.strip("# ").strip()
    except (OSError, IndexError):
        pass
    for key, cmd in (("power_mode", ["nvpmodel", "-q"]), ("desktop", ["systemctl", "is-active", "graphical.target"])):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
        if key == "power_mode":
            m = re.search(r"NV Power Mode:\s*(\S+)", out)
            if m:
                info[key] = m.group(1)
        elif out:
            info[key] = "on" if out == "active" else "off"
    return info


def preflight(args):
    """Problems that stop the benchmark, and warnings that only go in the report."""
    problems, warnings = [], []
    missing = [n for n in RUNNER_NEEDS if not (args.runner / n).exists()]
    if missing:
        problems.append(f"the runner folder {args.runner} has no {', '.join(missing)} (--runner points at the folder "
                        f"with jetson_init.py and the weights)")
    for pid, cmd in heavy_processes():
        problems.append(f"another heavy process is running (pid {pid}: {cmd}); stop it first (or --force)")
    info = board_info()
    if info["ram_used_gb"] > 1.5:
        warnings.append(f"{info['ram_used_gb']:.2f} GB of RAM already in use (about 0.5-1.5 GB after a reboot with "
                        f"the desktop off): the numbers will be worse than on a clean board")
    if info.get("desktop") == "on":
        warnings.append("the desktop is on (about 1 GB): sudo systemctl isolate multi-user.target")
    if not info["tegrastats"] or jetson_stats is None:
        warnings.append("no tegrastats (or core/jetson/jetson_stats.py): no GPU, power or temperature numbers")
    return problems, warnings, info


def link_scan(scan, dst):
    """dst/ = links to the scan's parts, so the runner reads the scan and writes its answers into dst/."""
    scan = scan.expanduser().resolve()
    missing = [n for n in SCAN_NEEDS if not (scan / n).is_dir()]
    if missing:
        raise SystemExit(f"{scan} is not a scan folder (no {', '.join(missing)}/)")
    dst.mkdir(parents=True, exist_ok=True)
    for n in SCAN_NEEDS + SCAN_EXTRA:
        if (scan / n).exists() and not (dst / n).exists():
            (dst / n).symlink_to(scan / n, target_is_directory=(scan / n).is_dir())
    return dst


# ---------------------------------------------------------------------------------------- the plan
def plan(args):
    """[case] from the cases file and the scans given on the command line, then one job per case,
    --frames and --quant setting."""
    cases = []
    if args.cases:
        spec = json.loads(Path(args.cases).expanduser().read_text())
        for i, c in enumerate(spec.get("cases", [])):
            name = c.get("name") or f"case{i + 1}"
            if not c.get("scan") or not c.get("queries"):
                raise SystemExit(f"cases file: case '{name}' needs a scan and at least one query")
            extra = set(c.get("expect", {})) - set(c["queries"])
            if extra:
                raise SystemExit(f"cases file: case '{name}' expects answers for queries it doesn't ask: {sorted(extra)}")
            cases.append(dict(name=name, scan=c["scan"], queries=list(c["queries"]), expect=c.get("expect", {})))
    if args.scans and not args.query:
        raise SystemExit("give the queries for the scans on the command line with -q \"...\" (one -q per query)")
    for scan in args.scans:
        p = Path(scan).expanduser()
        name = p.resolve().parent.name if p.name == "scan" else p.name
        cases.append(dict(name=name, scan=scan, queries=list(args.query), expect={}))
    if not cases:
        raise SystemExit("nothing to run: give a scan folder and -q queries, or --cases FILE")
    jobs, seen = [], set()
    for c in cases:
        for frames in args.frames:
            for quant in args.quant:
                jid = safe_name(f"{c['name']}-f{frames}-{quant}")
                while jid in seen:
                    jid += "x"
                seen.add(jid)
                jobs.append(dict(c, id=jid, frames=frames, quant=quant))
    return jobs


def command(job, args, scan_dir):
    return [args.python, "-u", str(args.runner / "jetson_init.py"), str(scan_dir), *job["queries"],
            "--frames", str(job["frames"]), "--quant", job["quant"]]


# ---------------------------------------------------------------------------------------- one run
def stop_group(proc, grace=10):
    """The runner and anything it started: SIGTERM, then SIGKILL. Its memory must be free before the next run."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(5)
    except ProcessLookupError:
        pass


def floats(s):
    return [float(x) for x in s.split(",")]


def parse_line(ln, t, res, times, mark):
    """Fold one line of the runner's output (arrived t s after its start) into res."""
    m = RE_TORCH.match(ln)
    if m:
        res.update(torch=m.group(1), device=m.group(2), ram_before_gb=float(m.group(3)), ram_total_gb=float(m.group(4)))
        times["imported"] = t
        mark("model load")
        return
    m = RE_LOADED.match(ln)
    if m:
        res["model_load_s"] = int(m.group(1))
        times["loaded"] = t
        mark("scene prep")
        return
    m = RE_RETRY.match(ln)
    if m:
        res["oom_retries"].append(dict(stage=m.group(1), frames=int(m.group(2)), retry_frames=int(m.group(3)), t=round(t, 1)))
        return
    m = RE_SCENE.match(ln)
    if m:
        parts = {}
        for kv in m.group(5).split(", "):
            k, _, v = kv.rpartition(" ")
            try:
                parts[k] = float(v)
            except ValueError:
                pass
        res.update(frames_used=int(m.group(1)), point_tokens=int(m.group(2)), voxels=int(m.group(3)),
                   scene_prep_s=float(m.group(4)), scene_parts=parts, ram_after_scene_gb=float(m.group(6)))
        times["scene"] = t
        mark("queries")
        return
    m = RE_QUERY.match(ln)
    if m:
        laptop = m.group("laptop")
        res["queries"].append(dict(
            query=m.group("query"), target=m.group("target"), score=float(m.group("score")), band=m.group("band"),
            points=int(m.group("points")), center=floats(m.group("center")) if m.group("center") else None,
            seconds=float(m.group("seconds")), done_s=round(t, 2), laptop=laptop,
            laptop_match=None if laptop is None else bool(re.search(r"\bMATCH\b", laptop)) and "DIFFERENT" not in laptop))
        return
    if ln.startswith("=== SUMMARY"):
        times["summary"] = t
        mark("exit")
        return
    m = RE_DEVICE.match(ln)
    if m:
        res["config"] = m.group(1)
        return
    m = RE_RAM.match(ln)
    if m:
        res.update(ram_idle_gb=float(m.group(1)), ram_peak_gb=float(m.group(2)), ram_total_gb=float(m.group(3)))
        return
    m = RE_FATAL.match(ln.strip())
    if m:
        res["oom_fatal"] = m.group(1)
        res["error"] = ln.strip()
        return
    if NVMAP in ln:
        res["nvmap_warnings"] += 1


def check_answer(q, exp):
    """One answer against what the cases file expects (None if it expects nothing for this query):
    score -> the same confidence band; band; min_score; center -> within tol_cm (default 3 cm)."""
    if not exp:
        return None
    c = {}
    if "score" in exp:
        c["band"] = band(q["score"]) == band(exp["score"])
        c["score_delta"] = round(q["score"] - exp["score"], 3)
    if "band" in exp:
        c["band"] = band(q["score"]) == exp["band"]
    if "min_score" in exp:
        c["min_score"] = q["score"] >= exp["min_score"]
    if "center" in exp:
        if q.get("center"):
            d = 100 * math.dist(q["center"], exp["center"])
            c["center_cm"] = round(d, 1)
            c["center"] = d <= exp.get("tol_cm", CENTER_TOL_CM)
        else:
            c["center"] = False
    c["ok"] = all(v for v in c.values() if isinstance(v, bool))
    return c


def run_one(job, args, out, n, total):
    rid = job["id"]
    scan_dir = link_scan(Path(job["scan"]), out / rid / "scan")
    cmd = command(job, args, scan_dir)
    print(f"\n== run {n}/{total}: {rid}: {len(job['queries'])} quer{'y' if len(job['queries']) == 1 else 'ies'} on "
          f"{job['scan']} ({job['frames']} frames, {job['quant']})", flush=True)
    res = dict(id=rid, case=job["name"], scan=str(Path(job["scan"]).expanduser()), frames_asked=job["frames"],
               quant=job["quant"], command=cmd, queries=[], oom_retries=[], oom_fatal=None, nvmap_warnings=0, error=None)
    sampler = jetson_stats.Sampler(args.interval, tegrastats=not args.no_tegrastats) if jetson_stats else None
    mark = sampler.mark if sampler else (lambda name: None)
    mark("import")
    ram = RamPeak()
    times, tail = {}, deque(maxlen=20)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=str(args.runner), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True, errors="replace", bufsize=1,
                            env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
    timed_out = threading.Event()

    def on_timeout():
        timed_out.set()
        stop_group(proc)
    timer = threading.Timer(args.timeout, on_timeout)
    timer.daemon = True
    timer.start()
    try:
        with open(out / f"{rid}.log", "w") as log:
            log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            for raw in proc.stdout:
                t = time.monotonic() - t0
                ln = raw.rstrip("\n")
                log.write(f"[{t:7.2f}] {ln}\n")
                log.flush()
                nv = res["nvmap_warnings"]
                parse_line(ln, t, res, times, mark)
                w = RE_WEIGHTS.match(ln)
                if (w and w.group(1) != w.group(2)) or (NVMAP in ln and nv > 0):
                    continue                            # one line per weights file, NvMap warnings: counted instead
                print("   | " + ln, flush=True)
                if ln.strip() and NVMAP not in ln:
                    tail.append(ln.strip())
            proc.wait()
    finally:
        timer.cancel()
        stop_group(proc)                                # (also on Ctrl-C: it runs in its own session)
        proc.stdout.close()
        if sampler:
            sampler.stop()
        res["ram_peak_outside_gb"] = round(ram.stop(), 2)
    res["wall_s"] = round(time.monotonic() - t0, 1)
    res["exit_code"] = proc.returncode
    res["timed_out"] = timed_out.is_set()
    if res["timed_out"]:
        res["error"] = f"no answer within --timeout {args.timeout:.0f} s: stopped"
    elif proc.returncode != 0 and res["error"] is None:
        res["error"] = tail[-1] if tail else f"exit code {proc.returncode}"

    # wall-clock phases (independent of the runner's own timers)
    if "imported" in times:
        res["import_s"] = round(times["imported"], 1)
    if "loaded" in times and "imported" in times:
        res["model_load_wall_s"] = round(times["loaded"] - times["imported"], 1)
    if "scene" in times and "loaded" in times:
        res["scene_wall_s"] = round(times["scene"] - times["loaded"], 1)

    # the runner's answer files add what its log line leaves out
    for q in res["queries"]:
        f = scan_dir / "results_jetson" / f"{slug(q['query'])}.json"
        try:
            j = json.loads(f.read_text())
        except (OSError, ValueError):
            j = {}
        if j.get("query") == q["query"]:
            for k in ("top5_scores", "llm_seconds", "peak_gpu_gb", "llm_tokens", "size_xyz_m"):
                if k in j:
                    q[k] = j[k]
        q["check"] = check_answer(q, job["expect"].get(q["query"]))

    first = res["queries"][0] if res["queries"] else None
    if first is not None and "scene_prep_s" in res:
        res["init_s"] = round(res["scene_prep_s"] + first["seconds"], 1)
    no_oom = not res["oom_retries"] and not res["oom_fatal"]
    res["checks"] = {
        "finished": proc.returncode == 0 and not res["timed_out"] and len(res["queries"]) == len(job["queries"]),
        "init_within_budget": res.get("init_s") is not None and res["init_s"] <= INIT_BUDGET_S,
        "no_out_of_memory": no_oom,
        "first_query_high": first is not None and first["score"] >= FOUND_SCORE,
        "expected_answers": [q["check"]["ok"] for q in res["queries"] if q.get("check")],
    }
    if res["error"] is None and not res["checks"]["finished"]:
        res["error"] = (f"{len(job['queries']) - len(res['queries'])} of {len(job['queries'])} answers missing from the "
                        f"runner's output (see {rid}.log)")
    res["phases"] = sampler.segments() if sampler else []
    res["tegrastats"] = bool(sampler and sampler.has_tegrastats)
    if sampler:
        with open(out / f"{rid}.samples.jsonl", "w") as f:
            for s in list(sampler.samples):
                f.write(json.dumps(s) + "\n")
    print("-> " + one_line(res), flush=True)
    return res


def settle(baseline, max_s):
    """Wait until the last run's memory is back: the next run needs all of it."""
    end = time.monotonic() + max_s
    while time.monotonic() < end and used_gb() > baseline + 0.3:
        time.sleep(0.5)
    return used_gb()


# ---------------------------------------------------------------------------------------- report
def f_(x, spec="", unit="", none="-"):
    return none if x is None else f"{x:{spec}}{unit}"


def one_line(r):
    first = r["queries"][0] if r["queries"] else None
    return (f"{r['id']}: " + ("FINISHED" if r["checks"]["finished"] else f"FAILED ({r['error']})")
            + f" | load {f_(r.get('model_load_s'), 'd', ' s')}, scene prep {f_(r.get('scene_prep_s'), '.1f', ' s')}"
            + f", INIT {f_(r.get('init_s'), '.1f', ' s')}, RAM peak {f_(r.get('ram_peak_gb') or r.get('ram_peak_outside_gb'), '.2f', ' GB')}"
            + (f" | first query {first['score']:.3f} ({first['band']})" if first else ""))


def answer_line(q):
    s = f"'{q['query']}': {q['score']:.3f} {q['band']:4s} {q['points']:5d} pts"
    s += f" at [{', '.join(f'{c:.3f}' for c in q['center'])}]" if q.get("center") else " (no points)"
    s += f", {q['seconds']:.2f} s"
    c = q.get("check")
    if c:
        bits = []
        if "band" in c:
            bits.append("band " + ("OK" if c["band"] else "DIFFERENT")
                        + (f" ({c['score_delta']:+.3f})" if "score_delta" in c else ""))
        if "min_score" in c:
            bits.append("score " + ("OK" if c["min_score"] else "TOO LOW"))
        if "center" in c:
            bits.append(f"centre {f_(c.get('center_cm'), '.1f', ' cm')} " + ("OK" if c["center"] else "DIFFERENT"))
        s += " | expected: " + ", ".join(bits)
    if q.get("laptop_match") is not None:
        s += " | laptop " + ("MATCH" if q["laptop_match"] else "DIFFERENT")
    return s


def checks_line(r):
    c = r["checks"]
    pf = lambda ok: "PASS" if ok else "FAIL"            # noqa: E731
    first = r["queries"][0] if r["queries"] else None
    exp = c["expected_answers"]
    return (f"INIT <= {INIT_BUDGET_S} s {pf(c['init_within_budget'])} ({f_(r.get('init_s'), '.1f', ' s')}) | "
            f"no out-of-memory {pf(c['no_out_of_memory'])}"
            + (f" ({len(r['oom_retries'])} retr{'y' if len(r['oom_retries']) == 1 else 'ies'})" if r["oom_retries"] else "")
            + f" | first query >= {FOUND_SCORE} {pf(c['first_query_high'])}"
            + (f" ({first['score']:.3f})" if first else "")
            + (f" | expected answers {sum(exp)}/{len(exp)}" if exp else ""))


PHASES = ("import", "model load", "scene prep", "queries")


def phase_rows(r):
    rows = [p for p in r["phases"] if p["phase"] in PHASES]
    init_j = [p["energy_j"] for p in rows if p["phase"] in ("scene prep", "queries")]
    return rows, (sum(init_j) if init_j and None not in init_j else None)


def summary_text(info, runs, args, out):
    w = max([len(r["id"]) for r in runs] + [4])
    L = ["", "=== BENCH SUMMARY (send me this) ===",
         "board: " + ", ".join(f"{k} {v}" for k, v in (("L4T", info.get("l4t")), ("power mode", info.get("power_mode")),
                                                       ("desktop", info.get("desktop")),
                                                       ("RAM in use before", f"{info['ram_used_gb']:.2f} GB"),
                                                       ("tegrastats", "yes" if info["tegrastats"] else "no")) if v),
         f"runner: {args.runner}" + (f" | {runs[0]['config']}" if runs and runs[0].get("config") else "")]
    for wn in info.get("warnings", []):
        L.append(f"WARNING: {wn}")
    L += ["", f"{'run':{w}s}  exit  load   scene prep (ViT)    1st query  INIT*   RAM peak        OOM  NvMap",
          *[f"{r['id']:{w}s}  {f_(r['exit_code'], '>4')}  {f_(r.get('model_load_s'), '>3d', ' s'):>5s}  "
            f"{f_(r.get('scene_prep_s'), '.1f', ' s'):>7s} ({f_(r.get('scene_parts', {}).get('vit'), '.1f', ' s'):>6s})   "
            f"{f_(r['queries'][0]['seconds'] if r['queries'] else None, '.2f', ' s'):>7s}  "
            f"{f_(r.get('init_s'), '.1f', ' s'):>6s}  "
            f"{f_(r.get('ram_peak_gb') or r.get('ram_peak_outside_gb'), '.2f')} / {f_(r.get('ram_total_gb'), '.2f', ' GB')}  "
            f"{('fatal' if r['oom_fatal'] else len(r['oom_retries']) or '-')!s:>3s}  {r['nvmap_warnings'] or '-'!s:>5s}"
            for r in runs],
          f"* INIT = scene prep + first query: what the live demo waits for after the scan, with Qwen-3D preloaded "
          f"(budget {INIT_BUDGET_S} s). The live runner adds its own memory (phone stream, viewer) on top of the RAM peak.",
          "", "answers:"]
    for r in runs:
        for q in r["queries"]:
            L.append(f"  {r['id']:{w}s}  {answer_line(q)}")
        if r["error"]:
            L.append(f"  {r['id']:{w}s}  ERROR: {r['error']}")
    if any(r["phases"] for r in runs):
        L += ["", f"Jetson, per phase ({'tegrastats' if any(r['tegrastats'] for r in runs) else 'RAM only: no tegrastats'}):",
              f"  {'run':{w}s}  {'phase':10s}  {'s':>6s}  {'RAM peak':>8s}  {'GPU':>4s}  {'W avg / peak':>12s}  {'J':>6s}  {'C peak':>6s}"]
        for r in runs:
            rows, init_j = phase_rows(r)
            for i, p in enumerate(rows):
                L.append(f"  {r['id'] if i == 0 else '':{w}s}  {p['phase']:10s}  {f_(p['seconds'], '6.1f')}  "
                         f"{f_(p['ram_peak_gb'], '.2f', ' GB'):>8s}  {f_(p['gpu_mean_pct'], '.0f', '%'):>4s}  "
                         f"{f_(p['power_mean_w'], '.1f'):>5s} / {f_(p['power_peak_w'], '.1f'):<5s}  "
                         f"{f_(p['energy_j'], '.0f'):>6s}  {f_(p['temp_peak_c'], '.1f'):>6s}")
            if init_j is not None:
                L.append(f"  {'':{w}s}  INIT energy (scene prep + queries): {init_j:.0f} J")
    L += ["", "live-demo checks:"] + [f"  {r['id']:{w}s}  {checks_line(r)}" for r in runs]
    L.append(f"files: {out}/ (results.json, summary.md, a .log and .samples.jsonl per run)")
    return "\n".join(L)


def summary_md(info, runs, args):
    L = ["# Qwen-3D on the Jetson", "",
         "| | |", "|---|---|"]
    for k, v in (("L4T", info.get("l4t")), ("power mode", info.get("power_mode")), ("desktop", info.get("desktop")),
                 ("RAM in use before", f"{info['ram_used_gb']:.2f} GB"), ("runner", str(args.runner)),
                 ("model", runs[0].get("config") if runs else None)):
        if v:
            L.append(f"| {k} | {v} |")
    L += ["", "| run | frames | model load | scene prep (ViT) | first query | INIT | RAM peak | OOM | finished |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['id']} | {f_(r.get('frames_used'))} | {f_(r.get('model_load_s'), 'd', ' s')} | "
                 f"{f_(r.get('scene_prep_s'), '.1f', ' s')} ({f_(r.get('scene_parts', {}).get('vit'), '.1f', ' s')}) | "
                 f"{f_(r['queries'][0]['seconds'] if r['queries'] else None, '.2f', ' s')} | {f_(r.get('init_s'), '.1f', ' s')} | "
                 f"{f_(r.get('ram_peak_gb') or r.get('ram_peak_outside_gb'), '.2f')} / {f_(r.get('ram_total_gb'), '.2f', ' GB')} | "
                 f"{'fatal' if r['oom_fatal'] else len(r['oom_retries']) or '-'} | {'yes' if r['checks']['finished'] else 'NO'} |")
    L += ["", "| run | query | score | band | centre (m) | s | expected | laptop |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for q in r["queries"]:
            c = q.get("check")
            L.append(f"| {r['id']} | {q['query']} | {q['score']:.3f} | {q['band']} | "
                     f"{', '.join(f'{x:.3f}' for x in q['center']) if q.get('center') else '-'} | {q['seconds']:.2f} | "
                     f"{'-' if not c else 'OK' if c['ok'] else 'DIFFERENT'} | "
                     f"{'-' if q.get('laptop_match') is None else 'MATCH' if q['laptop_match'] else 'DIFFERENT'} |")
    if any(r["phases"] for r in runs):
        L += ["", "| run | phase | s | RAM peak (GB) | GPU avg | W avg | W peak | J | C peak |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            rows, _ = phase_rows(r)
            for p in rows:
                L.append(f"| {r['id']} | {p['phase']} | {f_(p['seconds'], '.1f')} | {f_(p['ram_peak_gb'], '.2f')} | "
                         f"{f_(p['gpu_mean_pct'], '.0f', '%')} | {f_(p['power_mean_w'], '.1f')} | {f_(p['power_peak_w'], '.1f')} | "
                         f"{f_(p['energy_j'], '.0f')} | {f_(p['temp_peak_c'], '.1f')} |")
    L += ["", "INIT = scene prep + first query (the wait after the scan with Qwen-3D preloaded; budget "
          f"{INIT_BUDGET_S} s). Checks per run:", ""] + [f"- {r['id']}: {checks_line(r)}" for r in runs]
    return "\n".join(L) + "\n"


def write(out, info, runs, args, partial=False):
    doc = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), partial=partial, board=info, runner=str(args.runner),
               python=args.python, frames=args.frames, quant=args.quant, runs=runs)
    (out / "results.json").write_text(json.dumps(doc, indent=2))
    (out / "summary.md").write_text(summary_md(info, runs, args))


# ---------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Qwen-3D benchmark on the Jetson (see the top of this file).")
    ap.add_argument("scans", nargs="*", help="scan folders (color/ depth/ conf/ intrinsic/ pose/), all asked the -q queries")
    ap.add_argument("-q", "--query", action="append", default=[], help="a query for the scans given here (repeat -q)")
    ap.add_argument("--cases", help="JSON file: scans, queries and expected answers (see cases.example.json)")
    default_runner = HERE if (HERE / "jetson_init.py").exists() else Path.home() / "qwen3d_jetson_test"
    ap.add_argument("--runner", default=str(default_runner), help="folder with jetson_init.py and the weights")
    ap.add_argument("--python", default=sys.executable, help="python that runs the runner (default: this one)")
    ap.add_argument("--frames", type=int, nargs="+", default=[15], help="scan frames for the model; several = one run each")
    ap.add_argument("--quant", nargs="+", default=["w8"], choices=["w8", "w4"], help="weights: w8 (the default) or w4")
    ap.add_argument("--timeout", type=float, default=900, help="seconds before a run is stopped")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between Jetson samples")
    ap.add_argument("--settle", type=float, default=60, help="seconds to wait at most for RAM to come back between runs")
    ap.add_argument("--no-tegrastats", action="store_true", help="RAM only")
    ap.add_argument("--out", default=f"runs/qwen_bench_{time.strftime('%H%M%S')}")
    ap.add_argument("--force", action="store_true", help="run even next to another heavy process (numbers will be off)")
    ap.add_argument("--dry-run", action="store_true", help="checks and the commands it would run; runs nothing")
    args = ap.parse_args(argv)
    args.runner = Path(args.runner).expanduser().resolve()

    jobs = plan(args)
    problems, warnings, info = preflight(args)
    info["warnings"] = warnings
    print("board: " + ", ".join(f"{k} {v}" for k, v in info.items() if k != "warnings"))
    for wn in warnings:
        print(f"WARNING: {wn}")
    for p in problems:
        print(f"STOP: {p}")
    if problems and not (args.force and all("another heavy process" in p for p in problems)):
        return 2
    for j in jobs:
        missing = [n for n in SCAN_NEEDS if not (Path(j["scan"]).expanduser() / n).is_dir()]
        if missing:
            print(f"STOP: {j['scan']} is not a scan folder (no {', '.join(missing)}/)")
            return 2
    out = Path(args.out).expanduser().resolve()           # absolute: the runner starts in its own folder
    print(f"{len(jobs)} run{'s' if len(jobs) != 1 else ''} (~2-3 min each: model load ~95 s, scene ~20 s, ~5 s per query):")
    for j in jobs:
        print(f"  {j['id']}: " + " ".join(shlex.quote(c) for c in command(j, args, out / j["id"] / "scan")))
    if args.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    baseline = used_gb()
    runs = []
    try:
        for n, job in enumerate(jobs, 1):
            if n > 1:
                now = settle(baseline, args.settle)
                if now > baseline + 0.3:
                    print(f"(RAM still {now:.2f} GB, {now - baseline:.2f} GB above the start: the next numbers may be off)")
            runs.append(run_one(job, args, out, n, len(jobs)))
            write(out, info, runs, args, partial=n < len(jobs))
    except KeyboardInterrupt:
        print("\n(stopped: the report covers the runs that finished)")
        if runs:
            write(out, info, runs, args, partial=True)
            print(summary_text(info, runs, args, out))
        return 130
    print(summary_text(info, runs, args, out))
    return 0 if all(r["checks"]["finished"] for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
