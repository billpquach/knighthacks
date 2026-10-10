"""The upstream Qwen-3D code (github.com/ll220/qwen3d), imported for inference only.

Upstream's modules import its training stack when they load: detectron2 (config machinery, registries),
torch_scatter, a CUDA point-ops extension, ipdb. The Jetson has none of them and inference doesn't run
them, so this puts small stand-ins in sys.modules (only for what is really missing) and registers the
`qwen3d` package without running its __init__.py, which imports every dataset. No upstream code is copied
here; the model, the mask decoder and the default config values all come from the checkout.

    up = load("~/qwen3d")      # the upstream checkout: the folder that holds qwen3d/ and libs/
    up.Model, up.Config, up.MaskDecoder, up.build_full_mask, up.cfg, up.prompt_3d
"""
import functools
import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import torch


# ---------------------------------------------------------------------------------------- config nodes
class Cfg(dict):
    """detectron2-style config node: attribute access, nested nodes, merge from a dict (yaml). While
    upstream's default-config functions run (lenient), reading a field detectron2 would have defined gives
    an empty node instead of an error (those are training settings); afterwards a missing field is an error."""
    lenient = False

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            if Cfg.lenient and not k.startswith("_"):
                return Cfg()
            raise AttributeError(f"config has no {k}") from None

    def __setattr__(self, k, v):
        self[k] = v

    def merge(self, d):
        for k, v in d.items():
            if k == "_BASE_":
                continue
            if isinstance(v, dict):
                node = self.get(k)
                if not isinstance(node, Cfg):
                    node = self[k] = Cfg()
                node.merge(v)
            else:
                self[k] = v
        return self


def _skeleton():
    """The nodes detectron2's own defaults create, which upstream's config functions add fields to, and
    the two detectron2 default values the mask decoder reads that upstream doesn't set (checked when the
    trained heads load: a different size would not load)."""
    cfg = Cfg()
    for path in ("MODEL.MASK_FORMER.TEST", "MODEL.SEM_SEG_HEAD", "MODEL.SWIN", "MODEL.INS_EMBED_HEAD", "INPUT.CROP",
                 "DATASETS", "DATALOADER", "SOLVER.AMP", "TEST"):
        node = cfg
        for k in path.split("."):
            node = node.setdefault(k, Cfg())
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 54
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128
    return cfg


# ---------------------------------------------------------------------------------------- stand-ins
def _broadcast(index, src, dim):
    """torch_scatter's rule: a 1-D index runs along `dim`; otherwise it lines up with src's leading
    dimensions; either way it is expanded over the remaining ones."""
    if index.dim() == 1:
        index = index.view([1] * dim + [-1])
    while index.dim() < src.dim():
        index = index.unsqueeze(-1)
    return index.expand_as(src)


def _scatter(src, index, dim, reduce, dim_size=None):
    dim = dim % src.dim()
    idx = _broadcast(index.long(), src, dim)
    n = int(dim_size) if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
    shape = list(src.shape)
    shape[dim] = n
    if reduce in ("sum", "mean"):
        out = torch.zeros(shape, dtype=torch.float32, device=src.device).scatter_add_(dim, idx, src.float())
        if reduce == "mean":
            cnt = torch.zeros(shape, dtype=torch.float32, device=src.device).scatter_add_(
                dim, idx, torch.ones(src.shape, dtype=torch.float32, device=src.device))
            out /= cnt.clamp(min=1)
        return out.to(src.dtype)
    return torch.zeros(shape, dtype=src.dtype, device=src.device).scatter_reduce(dim, idx, src, reduce, include_self=False)


def _torch_scatter():
    def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
        return _scatter(src, index, dim, "mean", dim_size)

    def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
        return _scatter(src, index, dim, "sum", dim_size)

    def scatter_min(src, index, dim=-1, out=None, dim_size=None):
        return _scatter(src, index, dim, "amin", dim_size), None        # (no arg-min: inference never reads it)

    def scatter_max(src, index, dim=-1, out=None, dim_size=None):
        return _scatter(src, index, dim, "amax", dim_size), None

    return dict(scatter_mean=scatter_mean, scatter_sum=scatter_sum, scatter_add=scatter_sum,
                scatter_min=scatter_min, scatter_max=scatter_max)


def configurable(init_func=None, *, from_config=None):
    """detectron2's @configurable: __init__ may be called with a config (it goes through from_config) or
    with keyword arguments."""
    if init_func is None:
        return lambda f: configurable(f, from_config=from_config)

    @functools.wraps(init_func)
    def wrapped(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            kwargs = type(self).from_config(*args, **kwargs)
            args = ()
        return init_func(self, *args, **kwargs)
    return wrapped


class Registry(dict):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def register(self, obj=None):
        if obj is None:
            return lambda o: self.register(o)
        self[obj.__name__] = obj
        return obj

    def get(self, name):
        return self[name]


def _missing_point_ops(name):
    raise RuntimeError(f"pointops.{name} is a CUDA extension that inference here doesn't use (or build)")


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__path__ = []                                          # importable as a package too
    m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    sys.modules[name] = m
    return m


def _install_standins():
    """Stand-ins only for what is missing: a machine with the real packages uses them. transformers is
    imported first: it probes for detectron2 itself and must not see the stand-in."""
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl  # noqa: F401
    def have(mod):
        try:
            importlib.import_module(mod)
            return True
        except Exception:
            return False
    if not have("ipdb"):
        _stub("ipdb", set_trace=lambda *a, **k: None)
    if not have("torch_scatter"):
        _stub("torch_scatter", **_torch_scatter())
    if not have("detectron2.config"):
        _stub("detectron2")
        _stub("detectron2.config", configurable=configurable, CfgNode=Cfg)
        _stub("detectron2.utils")
        _stub("detectron2.utils.registry", Registry=Registry)
    for name in ("libs", "libs.pointops2", "libs.pointops2.functions"):
        if name not in sys.modules:
            _stub(name)
    pointops = _stub("libs.pointops2.functions.pointops")
    pointops.__getattr__ = lambda name: functools.partial(_missing_point_ops, name)
    if not have("scipy.spatial.transform"):
        _stub("scipy")
        _stub("scipy.spatial")
        _stub("scipy.spatial.transform", Rotation=None)


# ---------------------------------------------------------------------------------------- loading
class Upstream(types.SimpleNamespace):
    pass


def build_cfg(root):
    """Upstream's default config (its own config functions) + its yaml chain for the 3D model."""
    import yaml
    cfgmod = importlib.import_module("qwen3d.config")
    cfg = _skeleton()
    Cfg.lenient = True
    try:
        for fn in ("add_maskformer2_config", "add_maskformer2_video_config"):
            if hasattr(cfgmod, fn):
                getattr(cfgmod, fn)(cfg)
    finally:
        Cfg.lenient = False
    chain, f = [], root / "qwen3d" / "configs" / "qwen_3d.yaml"
    while f is not None:
        d = yaml.safe_load(f.read_text()) or {}
        chain.append(d)
        f = (f.parent / d["_BASE_"]) if d.get("_BASE_") else None
    for d in reversed(chain):                               # base first, the 3D config last
        cfg.merge(d)
    return cfg


def load(root):
    root = Path(root).expanduser().resolve()
    if not (root / "qwen3d" / "modeling_qwen2_5_vl_modified.py").exists():
        raise SystemExit(f"{root} is not the Qwen-3D checkout (git clone https://github.com/ll220/qwen3d): "
                         f"no qwen3d/modeling_qwen2_5_vl_modified.py")
    _install_standins()
    if "qwen3d" not in sys.modules:                         # the package, without its __init__.py
        pkg = types.ModuleType("qwen3d")
        pkg.__path__ = [str(root / "qwen3d")]
        sys.modules["qwen3d"] = pkg
    model = importlib.import_module("qwen3d.modeling_qwen2_5_vl_modified")
    if not torch.cuda.is_available():
        # upstream pins the memory-efficient attention kernel, which only exists on CUDA; without a GPU (the
        # CPU test) any kernel may run. On the Jetson this stays as upstream wrote it.
        import contextlib
        model.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()
    config = importlib.import_module("qwen3d.config_qwen2_5_vl_modified")
    decoder = importlib.import_module("qwen3d.modeling.transformer_decoder.simple_mask_decoder")
    template = importlib.import_module("qwen3d.conversation_template")
    return Upstream(root=root, Model=model.Qwen2_5_VLForConditionalGeneration, Config=config.Qwen2_5_3D_VLConfig,
                    build_full_mask=model.build_full_mask, MaskDecoder=decoder.SimpleMaskDecoder,
                    prompt_3d=template.TEXT_INPUT_GEN_3D, cfg=build_cfg(root))
