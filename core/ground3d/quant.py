"""8-bit weights for Qwen-3D on the 8 GB Jetson, in plain PyTorch (no bitsandbytes, no custom kernels).

A Linear keeps its weight as int8 with one bf16 scale per output row (symmetric absmax) and expands it to
bf16 only for its own matmul, so the full bf16 model (~7 GB) never exists at once. The token-embedding
table is int8 per row too, and only the rows looked up get expanded.

With bits=4 the language model's Linears are 4-bit instead (two weights per byte, a bf16 scale and minimum per
64 weights of a row; about 1.2 GB less): for keeping Qwen-3D loaded beside the tracker. Its answers can differ from 8-bit's:
check them on a saved scan before relying on it.

load_streaming() builds the model with empty weights, then reads the safetensors one tensor at a time,
quantizing on the way in, and drops each file from the page cache once read: on the Jetson the page cache
and the GPU share the same RAM, and 7 GB of cached weights would crowd out the model itself.
"""
import json
import os
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

GB = 1024 ** 3


@torch.no_grad()
def quantize_rows(w, device, rows=4096):
    """w (any device, any float dtype) -> (int8 [out, in], bf16 scale [out, 1]) on `device`, one per-row
    absmax scale; a few thousand rows at a time, so no full-size fp32 copy is ever made (the embedding
    table alone would be 1.2 GB in fp32)."""
    q = torch.empty(w.shape, dtype=torch.int8, device=device)
    scale = torch.empty(w.shape[0], 1, dtype=torch.bfloat16, device=device)
    for i in range(0, w.shape[0], rows):
        part = w[i:i + rows].to(device).float()
        s = part.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8) / 127
        q[i:i + rows] = torch.round(part / s).clamp_(-127, 127).to(torch.int8)
        scale[i:i + rows] = s.to(torch.bfloat16)
    return q, scale


class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.in_features, self.out_features, self.has_bias = in_features, out_features, bias
        self.register_buffer("q", None)
        self.register_buffer("scale", None)
        self.bias = None

    @torch.no_grad()
    def set_weight(self, w, device):
        self.q, self.scale = quantize_rows(w, device)

    def weight_bf16(self):
        return self.q.to(torch.bfloat16) * self.scale

    def forward(self, x):
        w = self.weight_bf16()
        return F.linear(x.to(w.dtype), w, None if self.bias is None else self.bias.to(w.dtype))


@torch.no_grad()
def quantize_groups4(w, device, group=64, rows=2048):
    """w [out, in] -> (uint8 [out, in / 2]: two 4-bit codes per byte, bf16 scale and bf16 minimum [out, in / group]):
    each group of `group` weights along a row spans its own min..max in 16 steps (asymmetric: a lopsided group
    loses less than with a scale from its largest magnitude), a few thousand rows at a time."""
    out, n = w.shape
    if n % group:
        raise ValueError(f"4-bit needs in_features divisible by {group}, got {n}")
    packed = torch.empty(out, n // 2, dtype=torch.uint8, device=device)
    scale = torch.empty(out, n // group, dtype=torch.bfloat16, device=device)
    low = torch.empty(out, n // group, dtype=torch.bfloat16, device=device)
    for i in range(0, out, rows):
        part = w[i:i + rows].to(device).float().view(-1, n // group, group)
        lo = part.amin(dim=2, keepdim=True)
        s = (part.amax(dim=2, keepdim=True) - lo).clamp_(min=1e-8) / 15
        q = torch.round((part - lo) / s).clamp_(0, 15).to(torch.uint8).view(-1, n)
        packed[i:i + rows] = q[:, 0::2] | (q[:, 1::2] << 4)
        scale[i:i + rows], low[i:i + rows] = s.squeeze(2).to(torch.bfloat16), lo.squeeze(2).to(torch.bfloat16)
    return packed, scale, low


class Int4Linear(Int8Linear):
    """Int8Linear's layout with 4-bit weights (quantize_groups4), expanded to bf16 only for its own matmul."""
    GROUP = 64

    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("low", None)

    @torch.no_grad()
    def set_weight(self, w, device):
        self.q, self.scale, self.low = quantize_groups4(w, device, self.GROUP)

    def weight_bf16(self):
        p, out = self.q, self.q.shape[0]
        q = torch.stack((p & 15, p >> 4), dim=-1).view(out, -1, self.GROUP).to(torch.bfloat16)
        return (q * self.scale.unsqueeze(-1) + self.low.unsqueeze(-1)).view(out, -1)


class Int8Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.register_buffer("q", None)
        self.register_buffer("scale", None)

    @torch.no_grad()
    def set_weight(self, w, device):
        self.q, self.scale = quantize_rows(w, device)

    def forward(self, ids):
        return self.q[ids].to(torch.bfloat16) * self.scale[ids]


def chunked_mlp(mlp, chunk=1024):
    """The MLP acts on each token alone, so running it on `chunk` tokens at a time gives the same result
    with a fraction of the [tokens x 11008] intermediate tensors (~3,000 point tokens per query)."""
    full = mlp.forward

    def forward(self, x):
        if x.shape[-2] <= chunk:
            return full(x)
        return torch.cat([full(x[..., i:i + chunk, :]) for i in range(0, x.shape[-2], chunk)], dim=-2)
    mlp.forward = types.MethodType(forward, mlp)


def _swap_linears(module, names, cls=None):
    cls = cls or Int8Linear
    for name in names:
        lin = getattr(module, name, None)
        if isinstance(lin, nn.Linear):
            setattr(module, name, cls(lin.in_features, lin.out_features, lin.bias is not None))


def _drop_page_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (AttributeError, OSError):
        pass


def load_streaming(model_cls, config, weights_dir, device="cuda", vit8=True, emb8=True, bits=8, log=print):
    """model_cls(config) with int8 decoder Linears (4-bit with bits=4; ViT Linears int8 with vit8, embeddings with emb8),
    weights read from weights_dir/*.safetensors one tensor at a time. lm_head is dropped: Qwen-3D reads
    hidden states, not logits."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    weights_dir = Path(weights_dir)
    with init_empty_weights(include_buffers=False):
        model = model_cls._from_config(config, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model.lm_head = nn.Identity()
    decoder = Int4Linear if bits == 4 else Int8Linear
    for layer in model.model.layers:
        _swap_linears(layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj"), decoder)
        _swap_linears(layer.mlp, ("gate_proj", "up_proj", "down_proj"), decoder)
        chunked_mlp(layer.mlp)
    if vit8:
        for blk in model.visual.blocks:
            _swap_linears(blk.attn, ("qkv", "proj"))
            _swap_linears(blk.mlp, ("gate_proj", "up_proj", "down_proj"))
    if emb8:
        e = model.model.embed_tokens
        model.model.embed_tokens = Int8Embedding(e.num_embeddings, e.embedding_dim)

    index = weights_dir / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    t0 = time.time()
    for n, f in enumerate(files, 1):
        with safe_open(str(weights_dir / f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key.startswith("lm_head."):
                    continue
                mod_name, pname = key.rsplit(".", 1)
                try:
                    mod = model.get_submodule(mod_name)
                except AttributeError:
                    raise SystemExit(f"{f} has {key}, which the model doesn't: wrong weights folder or model code?")
                t = sf.get_tensor(key)
                if isinstance(mod, (Int8Linear, Int8Embedding)) and pname == "weight":
                    mod.set_weight(t, device)
                elif isinstance(mod, Int8Linear) and pname == "bias":
                    mod.bias = nn.Parameter(t.to(device, torch.bfloat16), requires_grad=False)
                else:
                    set_module_tensor_to_device(model, key, device, value=t.to(torch.bfloat16))
                del t
        _drop_page_cache(str(weights_dir / f))
        if device == "cuda":
            torch.cuda.empty_cache()
            log(f"  weights {n}/{len(files)} ({time.time() - t0:.0f} s): {torch.cuda.memory_allocated() / GB:.2f} GB on the GPU")
    for m in model.modules():                                # buffers made at init (rotary tables)
        for name, b in list(m._buffers.items()):
            if b is not None and b.device.type != torch.device(device).type:
                m._buffers[name] = b.to(device)
    empty = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.device.type == "meta"]
    empty += [n for n, m in model.named_modules() if isinstance(m, (Int8Linear, Int8Embedding)) and m.q is None]
    empty += [n for n, m in model.named_modules() if isinstance(m, Int8Linear) and m.has_bias and m.bias is None]
    if empty:
        raise SystemExit(f"weights missing from {weights_dir} for: {empty[:6]}{' ...' if len(empty) > 6 else ''}")
    return model.eval()
