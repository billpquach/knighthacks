"""Qwen-3D on the Jetson: a scan and a sentence ("the cup closest to the can") -> that object's 3D points.

    # one shot: load (~1.5 min), study the scan (~20 s), answer each query (~5 s), exit
    python -m ground3d.ground <scan> -q "the cup closest to the can" -q "the cup next to the ball"
    # preloaded, for the live demo: load now, then answer jobs (job.json in the folder; "more": true keeps it loaded
    # for the next question) and exit (also when the folder gets a "stop" file); python -m xray.live --ask runs this
    python -m ground3d.ground --serve ~/ground_jobs [--parent PID]

Run from core/, with the venv that has torch (CUDA), transformers 4.51.1 and accelerate. It needs:
  --upstream   the Qwen-3D code: git clone https://github.com/ll220/qwen3d ~/qwen3d
  --models     a folder with the model's files (default ~/models/qwen3d):
                 qwen/       its safetensors + config.json
                 heads.pth   the trained connector + mask decoder weights
                 processor/  the Qwen2.5-VL processor files

Memory plan for the 8 GB board: 8-bit language model, ViT and embedding table (quant.py); the ViT is
unloaded once the scan is encoded; 15 frames, or 10 if 15 runs out of memory (the ViT is still there then).

Outputs (one-shot): <out>/results.json (timings, RAM, every answer), a .ply of each answer's points, and
<slug>_refined.ply: the object collected densely from the scan around the answer (ground3d.refine, on the CPU).
A job's answer: <jobs>/result.json. The first query's answer is the target for the tracker.
"""
import argparse
import ctypes
import gc
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")    # freed GPU memory goes back

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from . import quant, refine as refine_mod, scene as scene_mod, upstream  # noqa: E402

GB = 1024 ** 3
HIGH, LOW = 0.5, 0.3                          # confidence bands for a score
RELATION = (r"\b(next to|beside|besides|near|nearest to|closest to|farthest from|furthest from|on top of|on|under|"
            r"underneath|below|above|over|behind|in front of|to the left of|to the right of|left of|right of|"
            r"between|inside|in|at|by|with|that|which|who|of)\b")
HOME = Path.home()


def band(score):
    return "high" if score >= HIGH else "low" if score >= LOW else "none"


def target_phrase(query):
    """The object the sentence is about: the words before the first relation, without the article
    ('the cup closest to the can' -> 'cup')."""
    head = re.split(RELATION, query, maxsplit=1, flags=re.I)[0].strip(" ,.") or query
    return re.sub(r"^(the|a|an|that|this)\s+", "", head, flags=re.I) or head


def slug(q):
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:48] or "query"


def is_oom(e):
    m = str(e)
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in m or "NVML" in m or "ALLOC" in m


class Ram(threading.Thread):
    """System RAM in use (on the Jetson the GPU's memory is part of it), every 0.1 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.now = self.peak = self.used()
        self.start()

    @staticmethod
    def used():
        m = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                k, v = ln.split(":", 1)
                m[k] = int(v.split()[0])
        return (m["MemTotal"] - m["MemAvailable"]) / 1024 ** 2

    @staticmethod
    def total():
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) / 1024 ** 2

    def run(self):
        while True:
            self.now = self.used()
            self.peak = max(self.peak, self.now)
            time.sleep(0.1)


def free_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Grounder:
    def __init__(self, upstream_root, weights, heads, processor, device="cuda", vit8=True, emb8=True, log=print):
        self.log, self.device = log, device
        t = time.time()
        self.up = upstream.load(upstream_root)
        self.cfg = self.up.cfg
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(str(processor), min_pixels=self.cfg.INPUT.MIN_PIXEL,
                                                       max_pixels=self.cfg.INPUT.MAX_PIXEL)
        tok = self.tokenizer = self.processor.tokenizer
        tok.add_special_tokens({"additional_special_tokens": ["<|pointcloud_pad|>"]})
        self.ids = {name: tok.convert_tokens_to_ids(name) for name in
                    ("<|pointcloud_pad|>", "<|im_start|>", "<|im_end|>", "<|vision_end|>")}
        config = self.up.Config.from_pretrained(str(weights))
        if config.pointcloud_token_id != self.ids["<|pointcloud_pad|>"]:
            log(f"note: the tokenizer gives <|pointcloud_pad|> = {self.ids['<|pointcloud_pad|>']}, the weights' config "
                f"{config.pointcloud_token_id}; using the tokenizer's")
            config.pointcloud_token_id = self.ids["<|pointcloud_pad|>"]
        self.timing = {"import": round(time.time() - t, 1)}
        t = time.time()
        self.model = quant.load_streaming(self.up.Model, config, weights, device, vit8=vit8, emb8=emb8, log=log)
        self.hidden = None
        self.model.model.norm.register_forward_hook(lambda m, i, o: setattr(self, "hidden", o))
        self.heads = self._heads(heads)
        self.timing["load"] = round(time.time() - t, 1)
        self.scene = self.voxel_feats = None
        self.retries = []                                    # out-of-memory retries during prepare()

    def _heads(self, path):
        """The parts upstream trains on top of the language model: two projections to the decoder width
        and the mask decoder (built from upstream's config, weights from the checkpoint)."""
        cfg, dim = self.cfg, self.cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        hid = self.model.config.hidden_size
        heads = nn.Module()
        heads.connector = nn.Sequential(nn.Linear(hid, dim), nn.LayerNorm(dim))
        heads.text_connector = nn.Sequential(nn.Linear(hid, dim), nn.LayerNorm(dim))
        heads.mask_decoder = self.up.MaskDecoder(**self.up.MaskDecoder.from_config(cfg, cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM, True))
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if k.split(".")[0] in ("connector", "text_connector", "mask_decoder")}
        missing, unexpected = heads.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"{path} doesn't fit the heads built from the upstream config: missing {missing[:4]}, "
                             f"unexpected {unexpected[:4]}")
        return heads.to(self.device).eval()

    # ------------------------------------------------------------------------------------ the scene
    @torch.no_grad()
    def prepare(self, scan, frames=15, max_depth=3.0, retry_frames=10):
        """Encode the scan once: ViT on each frame, tokens averaged into 5 cm voxels. Then the ViT is
        unloaded (it isn't needed for queries). Out of memory -> once more with fewer frames."""
        tries = [frames] + ([retry_frames] if frames > retry_frames else [])
        for i, n in enumerate(tries):
            try:
                t = time.time()
                sc = scene_mod.prepare(scan, self.processor.image_processor, self.cfg, n, max_depth, self.device)
                tv = time.time()
                feats = []
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    for pv, g in zip(sc.pixel_values, sc.grid_thw):
                        feats.append(self.model.visual(pv.to(self.device, torch.bfloat16), grid_thw=g.view(1, 3).to(self.device)))
                fc = torch.cat(feats)
                if fc.shape[0] != sc.token_voxel.shape[0]:
                    raise SystemExit(f"the ViT gave {fc.shape[0]} tokens for {sc.token_voxel.shape[0]} token points")
                self.voxel_feats = scene_mod.group_mean(fc, sc.token_voxel, sc.n_voxels).to(torch.bfloat16)
                sc.timing["vit"] = time.time() - tv
                sc.timing["total"] = time.time() - t
                del feats, fc
                break
            except RuntimeError as e:
                if not is_oom(e) or i == len(tries) - 1:
                    raise
                err = str(e).splitlines()[0][:120]
            sc = feats = fc = None                      # (outside the except: the traceback's tensors are gone)
            free_memory()
            self.retries.append(dict(frames=n, retry_frames=tries[i + 1], error=err))
            self.log(f"out of memory with {n} frames ({err}); retrying with {tries[i + 1]}")
        self.model.visual = None
        sc.pixel_values = None
        free_memory()
        self.scene = sc
        return sc

    # ------------------------------------------------------------------------------------ a query
    def _prompt(self, query):
        """Token ids with the scene's voxel tokens spliced in at the point-cloud placeholder."""
        tok, n = self.tokenizer, self.scene.n_voxels
        ids0 = tok([self.up.prompt_3d.format(target_sentence=query)], return_tensors="pt")["input_ids"][0]
        at = (ids0 == self.ids["<|pointcloud_pad|>"]).nonzero().flatten()
        if len(at) != 1:
            raise SystemExit("the prompt template should have exactly one point-cloud placeholder")
        at = int(at[0])
        ids = torch.cat([ids0[:at], ids0.new_full((n,), self.ids["<|pointcloud_pad|>"]), ids0[at + 1:]])
        vis_end = int((ids == self.ids["<|vision_end|>"]).nonzero().flatten()[-1])
        ends = (ids == self.ids["<|im_end|>"]).nonzero().flatten()
        span = (vis_end + 1, int(ends[ends > vis_end][0]))                # the sentence's tokens
        # the answer starts after "<|im_start|>assistant\n"; at inference there is none yet: answer == len(ids)
        answer = min(len(ids), int(len(ids) - 1 - (ids.flip(0) == self.ids["<|im_start|>"]).float().argmax()) + 3)
        return ids0, at, ids, span, answer

    def _target_positions(self, query, span_len):
        phrase = target_phrase(query)
        a = query.lower().find(phrase.lower())
        b = a + len(phrase)
        enc = self.tokenizer(query, return_offsets_mapping=True, add_special_tokens=False)
        if len(enc["input_ids"]) != span_len:
            return phrase, list(range(span_len))          # tokenized differently inside the prompt: use them all
        pos = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < b and e > a]
        return phrase, pos or list(range(span_len))

    @torch.no_grad()
    def query(self, query):
        sc, dev, cfg = self.scene, self.device, self.cfg
        t0 = time.time()
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ids0, at, ids, (s0, s1), answer = self._prompt(query)
        n, L = sc.n_voxels, len(ids)
        emb = self.model.get_input_embeddings()(ids0.to(dev))
        emb = torch.cat([emb[:at], self.voxel_feats.to(emb.dtype), emb[at + 1:]])[None]
        pad = torch.ones(1, L, dtype=torch.long, device=dev)
        allowed = self.up.build_full_mask(pad, dev)                      # [1, 1, L, L]
        if not cfg.CAUSAL_MASK:                                          # the prompt sees itself both ways, the answer is causal
            allowed[0, 0, :answer, answer:] = False
            allowed[0, 0, answer:, answer:] = torch.tril(torch.ones(L - answer, L - answer, dtype=torch.bool, device=dev))
        attn = torch.where(allowed, 0.0, torch.finfo(torch.bfloat16).min).to(torch.bfloat16)
        del allowed
        self.hidden = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            self.model(input_ids=ids[None].to(dev), attention_mask=attn if not cfg.CAUSAL_MASK else pad, padding_mask=pad,
                       inputs_embeds=emb, output_hidden_states=False, return_dict=True, use_cache=False,
                       rope_type=cfg.ROPE_TYPE, pointcloud_pixel_pos=[sc.voxel_pixel], pointcloud_xyz=sc.voxel_xyz[None],
                       use_causal_mask=cfg.CAUSAL_MASK, generate_mode=False,
                       answer_start=torch.tensor([answer], device=dev))
            hidden = self.hidden
            self.hidden = None
            del emb, attn
            t_llm = time.time() - t0
            point = self.heads.connector(hidden[:, at:at + n].float()).float()[0]  # [V, C]
            text = self.heads.text_connector(hidden[:, s0:s1].float())             # [1, S, C]
            del hidden
            up = torch.zeros(len(sc.target_xyz), point.shape[1], device=dev)       # onto the target cloud
            for j in range(sc.knn_idx.shape[1]):
                up.add_(point[sc.knn_idx[:, j]] * sc.knn_w[:, j:j + 1])
            m = len(sc.target_xyz)
            out = self.heads.mask_decoder(up.T[None, :, :, None], shape=[1, len(sc.frames)], mask_features_xyz=sc.target_xyz[None],
                                          segments=torch.arange(m, device=dev)[None], decoder_3d=True,
                                          actual_decoder_3d=True, scannet_all_masks_batched=None, max_valid_points=[m],
                                          qwen_3d_text_features=text)
        phrase, pos = self._target_positions(query, s1 - s0)
        scores = out["pred_logits"][0].float().sigmoid()[:, pos].mean(-1)         # one per decoder query
        order = scores.argsort(descending=True)
        best = int(order[0])
        mask = out["pred_masks"][0, best].float().sigmoid() > 0.5
        pts = sc.target_xyz[mask].cpu().numpy()
        if dev == "cuda":
            torch.cuda.synchronize()
        r = dict(query=query, target=phrase, score=round(float(scores[best]), 4), band=band(float(scores[best])),
                 top5=[round(float(scores[i]), 4) for i in order[:5]], n_points=int(len(pts)),
                 center=np.round(pts.mean(0), 3).tolist() if len(pts) else None,
                 size=np.round(pts.max(0) - pts.min(0), 3).tolist() if len(pts) else None,
                 seconds=round(time.time() - t0, 2), llm_seconds=round(t_llm, 2), tokens=L,
                 gpu_peak_gb=round(torch.cuda.max_memory_allocated() / GB, 2) if dev == "cuda" else None)
        del out, up, point, text
        if dev == "cuda":
            torch.cuda.empty_cache()
        return r, pts


def write_ply(path, pts):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, pts, fmt="%.4f")


def describe(r):
    return (f"'{r['query']}' (target '{r['target']}'): score {r['score']:.3f} ({r['band']}), {r['n_points']} points"
            + (f" at {r['center']}" if r["center"] else "") + f", {r['seconds']:.2f} s")


def refine_answers(scan, answers, points, out=None, log=print):
    """Each answer's dense, clean cloud from the scan (ground3d.refine, CPU); adds answer["refined"] and,
    with `out`, writes <slug>_refined.ply. A failure here is noted, never fatal: the raw answer stands."""
    try:
        keyframes = refine_mod.load_keyframes(scan)
        table = refine_mod.table_height(keyframes)
        if table is None:                                   # no table at table height: the objects stand on the floor
            table = refine_mod.table_height(keyframes, lo=-0.15, hi=0.3)   # (or the scan saw only the table, at z = 0)
        # detector masks if the scan already has them cached (YOLOE is not loaded here: Qwen-3D holds the GPU)
        dets = refine_mod.scan_detections(scan, refine_mod.DEFAULT_PROMPT, use_detector=False, log=log)
    except Exception as e:
        for a in answers:
            a["refined"] = {"error": f"{type(e).__name__}: {e}"}
        return
    for a, pts in zip(answers, points):
        if not len(pts):
            continue
        try:
            r = refine_mod.refine(pts, keyframes, table, label=a.get("target") or a["query"], dets=dets)
            a["refined"] = refine_mod.summary(r)
            if out is not None and r["n_points"]:
                refine_mod.write_ply(Path(out) / f"{slug(a['query'])}_refined.ply", r["points"])
            log(f"  refined '{a['query']}': {len(pts)} -> {r['n_points']} points in {r['seconds']:.2f} s"
                + (f", centre {r['center']}, size {refine_mod.size_cm(r['size'])}" if r["center"] else ""))
        except Exception as e:
            a["refined"] = {"error": f"{type(e).__name__}: {e}"}


def serve(g, jobs, parent, ram, args):
    """Preloaded: answer jobs/job.json ({"scan", "queries" or "query", "frames", "more"}) into jobs/result.json. A job
    with "more": true keeps it loaded for the next one (on the same scan, encoded once: the ViT is unloaded after the
    first); otherwise, or as soon as jobs/stop appears, it exits so the memory is free for tracking."""
    (jobs / "ready").write_text(str(os.getpid()))
    print(f"READY ({ram.now:.2f} GB in use): waiting for {jobs / 'job.json'}", flush=True)
    sc = scan = None
    while True:
        while not (jobs / "job.json").exists():
            if (jobs / "stop").exists():
                (jobs / "ready").unlink(missing_ok=True)
                print("stop: exiting to free the memory", flush=True)
                return
            if parent and os.getppid() != parent:
                (jobs / "ready").unlink(missing_ok=True)
                sys.exit("the process that started this has ended: exiting to free the memory")
            time.sleep(0.2)
        time.sleep(0.1)
        job = json.loads((jobs / "job.json").read_text())
        (jobs / "job.json").rename(jobs / "job.taken")
        t = time.time()
        res = dict(scan=job["scan"])
        try:
            if sc is None:
                sc, scan = g.prepare(job["scan"], int(job.get("frames", args.frames)), args.max_depth), job["scan"]
            elif job["scan"] != scan:
                raise RuntimeError(f"this load encoded {scan}; a new scan needs a new load")
            res.update(frames=len(sc.frames), voxels=sc.n_voxels, target_points=len(sc.target_xyz),
                       scene_s={k: round(v, 2) for k, v in sc.timing.items()})
            answers, points = [], []
            for q in job.get("queries") or [job["query"]]:
                r, pts = g.query(q)
                answers.append(r)
                points.append(pts)
                print(describe(r), flush=True)
                if len(pts):                                # Qwen-3D's own cloud: drawn if the dense one fails
                    write_ply(jobs / f"{slug(q)}.ply", pts)
            refine_answers(job["scan"], answers, points, out=jobs)
            res.update(answers=answers, target=answers[0])
        except Exception as e:                              # the live runner reads the error and falls back
            res["error"] = ("OUT OF MEMORY: " if is_oom(e) else "") + f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
            print(res["error"], flush=True)
        res.update(seconds_after_job=round(time.time() - t, 1), ram_peak_gb=round(ram.peak, 2), ram_gb=round(ram.now, 2))
        (jobs / "result.tmp").write_text(json.dumps(res, indent=2))
        (jobs / "result.tmp").rename(jobs / "result.json")
        if not job.get("more"):
            break
    (jobs / "ready").unlink(missing_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Qwen-3D grounding on the Jetson (see the top of this file)")
    ap.add_argument("scan", nargs="?", help="scan folder")
    ap.add_argument("-q", "--query", action="append", default=[], help="a sentence; repeat -q; the first is the target")
    ap.add_argument("--serve", metavar="JOBS", help="preload, answer jobs from JOBS/job.json, then exit")
    ap.add_argument("--parent", type=int, help="with --serve: exit if this process ends first")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--max-depth", type=float, default=3.0)
    ap.add_argument("--upstream", default=str(HOME / "qwen3d"), help="the Qwen-3D code checkout")
    ap.add_argument("--models", default=str(HOME / "models" / "qwen3d"), help="folder with qwen/, heads.pth, processor/")
    ap.add_argument("--weights", help="default: <models>/qwen")
    ap.add_argument("--heads", help="default: <models>/heads.pth")
    ap.add_argument("--processor", help="default: <models>/processor")
    ap.add_argument("--no-vit8", action="store_true", help="keep the ViT in bf16 (more memory)")
    ap.add_argument("--no-emb8", action="store_true", help="keep the embedding table in bf16 (more memory)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-refine", action="store_true", help="skip the dense object clouds (ground3d.refine)")
    ap.add_argument("--out", help="one-shot output folder (default: runs/ground3d_HHMMSS)")
    args = ap.parse_args(argv)
    if not args.serve and not (args.scan and args.query):
        ap.error("give a scan folder and -q queries, or --serve JOBS")
    models = Path(args.models).expanduser()
    args.weights = args.weights or str(models / "qwen")
    args.heads = args.heads or str(models / "heads.pth")
    args.processor = args.processor or str(models / "processor")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("torch has no CUDA here (on the Jetson: the JetPack torch wheel, and the venv activated)")
    ram = Ram()
    time.sleep(0.2)
    ram_idle = ram.now
    jobs = Path(args.serve).expanduser() if args.serve else None
    if jobs:
        jobs.mkdir(parents=True, exist_ok=True)
        for f in ("job.json", "result.json", "ready", "stop"):
            (jobs / f).unlink(missing_ok=True)
        (jobs / "pid").write_text(str(os.getpid()))
        print(f"torch {torch.__version__}, RAM in use {ram_idle:.2f} of {Ram.total():.2f} GB", flush=True)
        g = Grounder(args.upstream, args.weights, args.heads, args.processor, args.device,
                     vit8=not args.no_vit8, emb8=not args.no_emb8)
        print(f"loaded in {g.timing['import'] + g.timing['load']:.0f} s, RAM in use {ram.now:.2f} GB", flush=True)
        serve(g, jobs, args.parent, ram, args)
        return 0

    # one shot: results.json is written whatever happens, so a benchmark can always read the outcome
    out = Path(args.out or f"runs/ground3d_{time.strftime('%H%M%S')}").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    res = dict(scan=str(Path(args.scan).expanduser().resolve()), queries=list(args.query), torch=torch.__version__,
               device=torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
               settings=dict(frames_asked=args.frames, vit8=not args.no_vit8, emb8=not args.no_emb8),
               answers=[], retries=[], error=None)
    g = sc = None
    ram_loaded = None
    print(f"torch {torch.__version__}, {res['device']}, RAM in use {ram_idle:.2f} of {Ram.total():.2f} GB", flush=True)
    try:
        g = Grounder(args.upstream, args.weights, args.heads, args.processor, args.device,
                     vit8=not args.no_vit8, emb8=not args.no_emb8)
        ram_loaded = ram.now
        print(f"loaded in {g.timing['import'] + g.timing['load']:.0f} s, RAM in use {ram_loaded:.2f} GB", flush=True)
        sc = g.prepare(args.scan, args.frames, args.max_depth)
        print(f"scene: {len(sc.frames)} frames, {sc.grid[0]}x{sc.grid[1]} tokens each -> {sc.n_voxels} voxels, "
              f"{len(sc.target_xyz)} target points; " + ", ".join(f"{k} {v:.1f} s" for k, v in sc.timing.items())
              + f"; RAM in use {ram.now:.2f} GB", flush=True)
        points = []
        for q in args.query:
            r, pts = g.query(q)
            if len(pts):
                write_ply(out / f"{slug(q)}.ply", pts)
            res["answers"].append(r)
            points.append(pts)
            print("  " + describe(r), flush=True)
        if not args.no_refine:
            refine_answers(args.scan, res["answers"], points, out=out)
    except (Exception, SystemExit) as e:
        msg = str(e).splitlines()[0][:300] if str(e) else type(e).__name__
        res["error"] = ("OUT OF MEMORY: " if isinstance(e, Exception) and is_oom(e) else "") + f"{type(e).__name__}: {msg}"
        print(f"FAILED: {res['error']}", flush=True)
    if g is not None:
        res["timing"] = dict(g.timing)
        res["retries"] = list(g.retries)
    if sc is not None:
        res["timing"]["scene"] = {k: round(v, 2) for k, v in sc.timing.items()}
        res.update(frames=len(sc.frames), voxels=sc.n_voxels, target_points=len(sc.target_xyz))
    res["ram_gb"] = dict(idle=round(ram_idle, 2), loaded=None if ram_loaded is None else round(ram_loaded, 2),
                         peak=round(ram.peak, 2), total=round(Ram.total(), 2))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("\n=== GROUND3D SUMMARY (send me this) ===")
    if res["error"]:
        print(f"FAILED: {res['error']}")
    if sc is not None and res["answers"]:
        first = res["answers"][0]
        print(f"load {g.timing['import'] + g.timing['load']:.0f} s | scene {sc.timing['total']:.1f} s (ViT {sc.timing['vit']:.1f} s, "
              f"{len(sc.frames)} frames, {sc.n_voxels} voxels) | first query {first['seconds']:.2f} s | "
              f"INIT (scene + first query) {sc.timing['total'] + first['seconds']:.1f} s")
    print(f"RAM in use: idle {ram_idle:.2f}, loaded {res['ram_gb']['loaded']}, peak {ram.peak:.2f} of {Ram.total():.2f} GB")
    for r in res["answers"]:
        print(describe(r))
    print(f"files: {out}/ (results.json, one .ply per answer)")
    return 1 if res["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
