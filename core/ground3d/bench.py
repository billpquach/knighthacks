"""Qwen-3D benchmark on the Jetson: runs the grounding runner (python -m ground3d.ground, one shot) on a set
of scans and queries, and reports what each answer costs and whether it is right.

For every scan (and every --frames setting asked for) it starts the runner once, the way a real INIT
starts it, and reports:
  * time: model load, scene prep (and its ViT part), each query, and INIT = scene prep + first query
    (what the live demo waits for after the scan, with the model preloaded)
  * memory: RAM peak (the runner's own and seen from outside at 10 Hz), out-of-memory retries, NvMap
    memory-pressure warnings
  * with tegrastats: GPU load, board power (W), energy (J) and peak temperature, per phase
  * answers: score, confidence band and centre of every query, checked against a cases file

Run from core/, with the venv that runs Qwen-3D:
    python -m ground3d.bench <scan> -q "the cup closest to the can" -q "the cup next to the ball"
    python -m ground3d.bench --cases ground3d/cases.example.json [--frames 15 10] [--dry-run]
    python -m ground3d.bench --cases my_cases.json --update-cases     # write the answers in as the expected ones

Before a run: reboot, desktop off (sudo systemctl isolate multi-user.target), sudo jetson_clocks, nothing
else on the GPU (it refuses to start next to the live runner or another Qwen-3D). Outputs in
runs/ground3d_bench_HHMMSS/: results.json, summary.md, and per run the runner's output folder, a .log and
a .samples.jsonl (RAM, GPU, W, C every 0.5 s). Send back everything after "=== BENCH SUMMARY".
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
CORE = HERE.parent
sys.path.insert(0, str(CORE / "jetson"))
try:
    import jetson_stats
except ImportError:          # no GPU / power numbers without it; the rest works
    jetson_stats = None

HIGH, LOW = 0.5, 0.3                   # confidence bands of a score
INIT_BUDGET_S = 40                     # live demo: INIT answers within 40 s of the end of the scan
CENTER_TOL_CM = 3.0                    # "the same answer": centres within 3 cm
SCAN_NEEDS = ("color", "depth", "conf", "intrinsic", "pose")
HEAVY = ("ground3d.ground", "xray.live")   # processes that need the GPU's memory
NVMAP = "NvMapMemAllocInternalTagged"


def band(score):
    return "high" if score >= HIGH else "low" if score >= LOW else "none"


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
    """RAM in use, every 0.1 s, from outside the runner (its own peak is lost if it crashes)."""

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
    """Other GPU-heavy processes ([(pid, command line)]): the 8 GB fit one at a time."""
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
    if not (args.core / "ground3d" / "ground.py").exists():
        problems.append(f"{args.core} has no ground3d/ground.py (--core is the repo's core/ folder)")
    up, models = Path(args.upstream).expanduser(), Path(args.models).expanduser()
    if not (up / "qwen3d" / "modeling_qwen2_5_vl_modified.py").exists():
        problems.append(f"no Qwen-3D code at {up} (git clone https://github.com/ll220/qwen3d {up})")
    for need in ("qwen/config.json", "heads.pth", "processor"):
        if not (models / need).exists():
            problems.append(f"{models} has no {need} (--models: the folder with qwen/, heads.pth, processor/)")
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


# ---------------------------------------------------------------------------------------- the plan
def plan(args):
    """[case] from the cases file and the scans on the command line, then one job per case and --frames."""
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
            cases.append(dict(name=name, scan=c["scan"], queries=list(c["queries"]), expect=c.get("expect", {}), index=i))
    if args.scans and not args.query:
        raise SystemExit("give the queries for the scans on the command line with -q \"...\" (one -q per query)")
    for scan in args.scans:
        p = Path(scan).expanduser()
        name = p.resolve().parent.name if p.name == "scan" else p.name
        cases.append(dict(name=name, scan=scan, queries=list(args.query), expect={}, index=None))
    if not cases:
        raise SystemExit("nothing to run: give a scan folder and -q queries, or --cases FILE")
    jobs, seen = [], set()
    for c in cases:
        for frames in args.frames:
            jid = safe_name(f"{c['name']}-f{frames}")
            while jid in seen:
                jid += "x"
            seen.add(jid)
            jobs.append(dict(c, id=jid, frames=frames))
    return jobs


def command(job, args, out_dir):
    cmd = [args.python, "-u", "-m", "ground3d.ground", str(Path(job["scan"]).expanduser()), "--frames", str(job["frames"]),
           "--out", str(out_dir), "--upstream", str(args.upstream), "--models", str(args.models)]
    for q in job["queries"]:
        cmd += ["-q", q]
    return cmd


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


def phase_of(line):
    """The runner's progress lines that start a phase (ground3d/ground.py prints them)."""
    if line.startswith("torch "):
        return "model load"
    if line.startswith("loaded in "):
        return "scene prep"
    if line.startswith("scene: "):
        return "queries"
    if line.startswith("=== GROUND3D SUMMARY"):
        return "exit"
    return None


def check_answer(q, exp):
    """One answer against what the cases file expects (None if nothing): score -> the same band; band;
    min_score; center -> within tol_cm (default 3 cm)."""
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
    run_dir = out / rid
    cmd = command(job, args, run_dir)
    print(f"\n== run {n}/{total}: {rid}: {len(job['queries'])} quer{'y' if len(job['queries']) == 1 else 'ies'} on "
          f"{job['scan']} ({job['frames']} frames)", flush=True)
    res = dict(id=rid, case=job["name"], scan=str(Path(job["scan"]).expanduser()), frames_asked=job["frames"],
               command=cmd, answers=[], retries=[], nvmap_warnings=0, error=None)
    sampler = jetson_stats.Sampler(args.interval, tegrastats=not args.no_tegrastats) if jetson_stats else None
    mark = sampler.mark if sampler else (lambda name: None)
    mark("import")
    ram = RamPeak()
    tail = deque(maxlen=20)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=str(args.core), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
                ph = phase_of(ln)
                if ph:
                    mark(ph)
                if NVMAP in ln:
                    res["nvmap_warnings"] += 1
                    if res["nvmap_warnings"] > 1:
                        continue                        # counted, printed once
                elif re.match(r"^\s+weights \d+/\d+", ln) and not re.match(r"^\s+weights (\d+)/\1\b", ln):
                    continue                            # one line per weights file: only the last one is shown
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

    try:
        r = json.loads((run_dir / "results.json").read_text())
    except (OSError, ValueError):
        r = {}
    for k in ("timing", "frames", "voxels", "target_points", "ram_gb", "device", "torch"):
        if k in r:
            res[k] = r[k]
    res["answers"] = r.get("answers", [])
    res["retries"] = r.get("retries", [])
    res["error"] = r.get("error")
    if res["timed_out"]:
        res["error"] = f"no answer within --timeout {args.timeout:.0f} s: stopped"
    elif res["error"] is None and proc.returncode != 0:
        res["error"] = tail[-1] if tail else f"exit code {proc.returncode}"
    for q in res["answers"]:
        q["check"] = check_answer(q, job["expect"].get(q["query"]))

    scene = res.get("timing", {}).get("scene", {})
    first = res["answers"][0] if res["answers"] else None
    if first is not None and "total" in scene:
        res["init_s"] = round(scene["total"] + first["seconds"], 1)
    res["checks"] = {
        "finished": proc.returncode == 0 and not res["timed_out"] and not res["error"]
                    and len(res["answers"]) == len(job["queries"]),
        "init_within_budget": res.get("init_s") is not None and res["init_s"] <= INIT_BUDGET_S,
        "no_out_of_memory": not res["retries"] and "OUT OF MEMORY" not in (res["error"] or ""),
        "first_query_high": first is not None and first["score"] >= HIGH,
        "expected_answers": [q["check"]["ok"] for q in res["answers"] if q.get("check")],
    }
    if res["error"] is None and not res["checks"]["finished"]:
        res["error"] = f"{len(job['queries']) - len(res['answers'])} of {len(job['queries'])} answers missing (see {rid}.log)"
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


def ram_peak(r):
    return (r.get("ram_gb") or {}).get("peak") or r.get("ram_peak_outside_gb")


def one_line(r):
    first = r["answers"][0] if r["answers"] else None
    t = r.get("timing", {})
    return (f"{r['id']}: " + ("FINISHED" if r["checks"]["finished"] else f"FAILED ({r['error']})")
            + f" | load {f_(t.get('load'), '.0f', ' s')}, scene prep {f_(t.get('scene', {}).get('total'), '.1f', ' s')}"
            + f", INIT {f_(r.get('init_s'), '.1f', ' s')}, RAM peak {f_(ram_peak(r), '.2f', ' GB')}"
            + (f" | first query {first['score']:.3f} ({first['band']})" if first else ""))


def answer_line(q):
    s = f"'{q['query']}': {q['score']:.3f} {q['band']:4s} {q['n_points']:5d} pts"
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
    return s


def checks_line(r):
    c = r["checks"]
    pf = lambda ok: "PASS" if ok else "FAIL"            # noqa: E731
    first = r["answers"][0] if r["answers"] else None
    exp = c["expected_answers"]
    return (f"INIT <= {INIT_BUDGET_S} s {pf(c['init_within_budget'])} ({f_(r.get('init_s'), '.1f', ' s')}) | "
            f"no out-of-memory {pf(c['no_out_of_memory'])}"
            + (f" ({len(r['retries'])} retr{'y' if len(r['retries']) == 1 else 'ies'})" if r["retries"] else "")
            + f" | first query >= {HIGH} {pf(c['first_query_high'])}" + (f" ({first['score']:.3f})" if first else "")
            + (f" | expected answers {sum(exp)}/{len(exp)}" if exp else ""))


PHASES = ("import", "model load", "scene prep", "queries")


def phase_rows(r):
    rows = [p for p in r["phases"] if p["phase"] in PHASES]
    init_j = [p["energy_j"] for p in rows if p["phase"] in ("scene prep", "queries")]
    return rows, (sum(init_j) if init_j and None not in init_j else None)


def summary_text(info, runs, args, out):
    w = max([len(r["id"]) for r in runs] + [4])
    first = next((r for r in runs if r.get("device")), {})
    L = ["", "=== BENCH SUMMARY (send me this) ===",
         "board: " + ", ".join(f"{k} {v}" for k, v in (("L4T", info.get("l4t")), ("power mode", info.get("power_mode")),
                                                       ("desktop", info.get("desktop")),
                                                       ("RAM in use before", f"{info['ram_used_gb']:.2f} GB"),
                                                       ("tegrastats", "yes" if info["tegrastats"] else "no")) if v),
         f"runner: ground3d ({first.get('device', '?')}, torch {first.get('torch', '?')}), models {args.models}"]
    for wn in info.get("warnings", []):
        L.append(f"WARNING: {wn}")
    L += ["", f"{'run':{w}s}  exit  load   scene prep (ViT)    1st query  INIT*   RAM peak        OOM  NvMap"]
    for r in runs:
        t, sc = r.get("timing", {}), r.get("timing", {}).get("scene", {})
        L.append(f"{r['id']:{w}s}  {f_(r['exit_code'], '>4')}  {f_(t.get('load'), '>3.0f', ' s'):>5s}  "
                 f"{f_(sc.get('total'), '.1f', ' s'):>7s} ({f_(sc.get('vit'), '.1f', ' s'):>6s})   "
                 f"{f_(r['answers'][0]['seconds'] if r['answers'] else None, '.2f', ' s'):>7s}  "
                 f"{f_(r.get('init_s'), '.1f', ' s'):>6s}  "
                 f"{f_(ram_peak(r), '.2f')} / {f_((r.get('ram_gb') or {}).get('total'), '.2f', ' GB')}  "
                 f"{('fatal' if 'OUT OF MEMORY' in (r['error'] or '') else len(r['retries']) or '-')!s:>3s}  "
                 f"{r['nvmap_warnings'] or '-'!s:>5s}")
    L += [f"* INIT = scene prep + first query: what the live demo waits for after the scan, with Qwen-3D preloaded "
          f"(budget {INIT_BUDGET_S} s). The live runner adds its own memory (phone stream, viewer) on top of the RAM peak.",
          "", "answers:"]
    for r in runs:
        for q in r["answers"]:
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
    L.append(f"files: {out}/ (results.json, summary.md, per run: the runner's folder, a .log and a .samples.jsonl)")
    return "\n".join(L)


def summary_md(info, runs, args):
    L = ["# Qwen-3D on the Jetson", "", "| | |", "|---|---|"]
    for k, v in (("L4T", info.get("l4t")), ("power mode", info.get("power_mode")), ("desktop", info.get("desktop")),
                 ("RAM in use before", f"{info['ram_used_gb']:.2f} GB"), ("models", str(args.models))):
        if v:
            L.append(f"| {k} | {v} |")
    L += ["", "| run | frames | model load | scene prep (ViT) | first query | INIT | RAM peak | OOM | finished |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        t, sc = r.get("timing", {}), r.get("timing", {}).get("scene", {})
        L.append(f"| {r['id']} | {f_(r.get('frames'))} | {f_(t.get('load'), '.0f', ' s')} | "
                 f"{f_(sc.get('total'), '.1f', ' s')} ({f_(sc.get('vit'), '.1f', ' s')}) | "
                 f"{f_(r['answers'][0]['seconds'] if r['answers'] else None, '.2f', ' s')} | {f_(r.get('init_s'), '.1f', ' s')} | "
                 f"{f_(ram_peak(r), '.2f')} / {f_((r.get('ram_gb') or {}).get('total'), '.2f', ' GB')} | "
                 f"{len(r['retries']) or '-'} | {'yes' if r['checks']['finished'] else 'NO'} |")
    L += ["", "| run | query | score | band | centre (m) | s | expected |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        for q in r["answers"]:
            c = q.get("check")
            L.append(f"| {r['id']} | {q['query']} | {q['score']:.3f} | {q['band']} | "
                     f"{', '.join(f'{x:.3f}' for x in q['center']) if q.get('center') else '-'} | {q['seconds']:.2f} | "
                     f"{'-' if not c else 'OK' if c['ok'] else 'DIFFERENT'} |")
    if any(r["phases"] for r in runs):
        L += ["", "| run | phase | s | RAM peak (GB) | GPU avg | W avg | W peak | J | C peak |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            rows, _ = phase_rows(r)
            for p in rows:
                L.append(f"| {r['id']} | {p['phase']} | {f_(p['seconds'], '.1f')} | {f_(p['ram_peak_gb'], '.2f')} | "
                         f"{f_(p['gpu_mean_pct'], '.0f', '%')} | {f_(p['power_mean_w'], '.1f')} | {f_(p['power_peak_w'], '.1f')} | "
                         f"{f_(p['energy_j'], '.0f')} | {f_(p['temp_peak_c'], '.1f')} |")
    L += ["", f"INIT = scene prep + first query (budget {INIT_BUDGET_S} s). Checks per run:", ""]
    L += [f"- {r['id']}: {checks_line(r)}" for r in runs]
    return "\n".join(L) + "\n"


def write(out, info, runs, args, partial=False):
    doc = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), partial=partial, board=info, models=str(args.models),
               upstream=str(args.upstream), frames=args.frames, runs=runs)
    (out / "results.json").write_text(json.dumps(doc, indent=2))
    (out / "summary.md").write_text(summary_md(info, runs, args))


def update_cases(path, jobs, runs):
    """Write each finished run's answers into the cases file as what it expects (score and centre)."""
    spec = json.loads(path.read_text())
    n = 0
    for job, r in zip(jobs, runs):
        if job["index"] is None or not r["checks"]["finished"]:
            continue
        exp = spec["cases"][job["index"]].setdefault("expect", {})
        for q in r["answers"]:
            exp[q["query"]] = {k: v for k, v in (("score", q["score"]), ("center", q["center"])) if v is not None}
            n += 1
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"{path}: expected answers updated ({n})")


# ---------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Qwen-3D benchmark on the Jetson (see the top of this file).")
    ap.add_argument("scans", nargs="*", help="scan folders (color/ depth/ conf/ intrinsic/ pose/), all asked the -q queries")
    ap.add_argument("-q", "--query", action="append", default=[], help="a query for the scans given here (repeat -q)")
    ap.add_argument("--cases", help="JSON file: scans, queries and expected answers (see cases.example.json)")
    ap.add_argument("--update-cases", action="store_true", help="write the answers into the cases file as the expected ones")
    ap.add_argument("--frames", type=int, nargs="+", default=[15], help="scan frames for the model; several = one run each")
    ap.add_argument("--upstream", default=str(Path.home() / "qwen3d"), help="the Qwen-3D code checkout")
    ap.add_argument("--models", default=str(Path.home() / "models" / "qwen3d"), help="folder with qwen/, heads.pth, processor/")
    ap.add_argument("--core", default=str(CORE), help="the repo's core/ folder (the runner is python -m ground3d.ground there)")
    ap.add_argument("--python", default=sys.executable, help="python that runs the runner (default: this one)")
    ap.add_argument("--timeout", type=float, default=900, help="seconds before a run is stopped")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between Jetson samples")
    ap.add_argument("--settle", type=float, default=60, help="seconds to wait at most for RAM to come back between runs")
    ap.add_argument("--no-tegrastats", action="store_true", help="RAM only")
    ap.add_argument("--out", default=f"runs/ground3d_bench_{time.strftime('%H%M%S')}")
    ap.add_argument("--force", action="store_true", help="run even next to another heavy process (numbers will be off)")
    ap.add_argument("--dry-run", action="store_true", help="checks and the commands it would run; runs nothing")
    args = ap.parse_args(argv)
    args.core = Path(args.core).expanduser().resolve()
    if args.update_cases and not args.cases:
        ap.error("--update-cases needs --cases")

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
    out = Path(args.out).expanduser().resolve()           # absolute: the runner starts in core/
    print(f"{len(jobs)} run{'s' if len(jobs) != 1 else ''} (~2-3 min each: model load ~100 s, scene ~20 s, ~5 s per query):")
    for j in jobs:
        print(f"  {j['id']}: " + " ".join(shlex.quote(c) for c in command(j, args, out / j["id"])))
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
    if args.update_cases:
        update_cases(Path(args.cases).expanduser(), jobs, runs)
    return 0 if all(r["checks"]["finished"] for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
