"""python3 -m unittest discover -s tests

The parts of ground3d that need nothing heavy run everywhere. The end-to-end test runs the whole runner on
the CPU with a tiny random Qwen-3D (same architecture, 2 layers), a word-level tokenizer and a synthetic
scan: it checks every call and shape, not the answers. It needs torch, transformers 4.51.1, accelerate,
safetensors, pyyaml and the upstream checkout (QWEN3D_UPSTREAM, ~/qwen3d or /home/user/ll220/qwen3d);
it is skipped otherwise.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

try:
    import numpy as np
    import torch
    from ground3d import ground, scene
except ImportError:
    torch = None


def upstream_root():
    for p in (os.environ.get("QWEN3D_UPSTREAM"), "~/qwen3d", "/home/user/ll220/qwen3d"):
        if p and (Path(p).expanduser() / "qwen3d" / "modeling_qwen2_5_vl_modified.py").exists():
            return Path(p).expanduser()
    return None


def heavy_available():
    try:
        import accelerate  # noqa: F401
        import safetensors  # noqa: F401
        import transformers
        import yaml  # noqa: F401
        return transformers.__version__ == "4.51.1" and upstream_root() is not None
    except ImportError:
        return False


@unittest.skipIf(torch is None, "needs torch and numpy")
class TestPieces(unittest.TestCase):
    def test_target_phrase(self):
        for q, t in [("the cup closest to the can", "cup"), ("The red solo cup next to the ball", "red solo cup"),
                     ("a can", "can"), ("the cup on the left", "cup"), ("cup", "cup")]:
            self.assertEqual(ground.target_phrase(q), t, q)

    def test_token_grid(self):
        self.assertEqual(scene.token_grid(640, 480), (23, 17))      # a portrait phone frame
        self.assertEqual(scene.token_grid(480, 640), (17, 23))
        gh, gw = scene.token_grid(1440, 1920)                        # over the pixel budget: scaled down
        self.assertLessEqual(gh * gw * 28 * 28, 390000)

    def test_voxels_and_points(self):
        xyz = torch.tensor([[0.0, 0, 0], [0.01, 0, 0], [0.2, 0, 0], [0.21, 0.01, 0]])
        ids = scene.voxel_ids(xyz, 0.05)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[2], ids[3])
        self.assertNotEqual(ids[0], ids[2])
        m = scene.group_mean(xyz, ids)
        self.assertEqual(m.shape, (2, 3))
        K = torch.tensor([[100.0, 0, 2], [0, 100.0, 1], [0, 0, 1]])
        T = torch.eye(4)
        T[:3, 3] = torch.tensor([1.0, 2, 3])
        p = scene.world_points(torch.ones(3, 5), K, T)
        self.assertTrue(torch.allclose(p[1, 2], torch.tensor([1.0, 2, 4])))   # principal point: straight ahead


def make_scan(root, n=5, h=640, w=480):
    """A table (z = 0.7) with two boxes, seen from n poses looking down; recorder format."""
    from PIL import Image
    for d in ("color", "depth", "conf", "intrinsic", "pose"):
        (root / d).mkdir(parents=True)
    K = np.array([[450.0, 0, w / 2], [0, 450.0, h / 2], [0, 0, 1]])
    rng = np.random.default_rng(0)
    for k in range(n):
        cam = np.array([0.3 * k - 0.6, -0.8, 1.4])
        fwd = np.array([0, 0.8, 0.7]) - cam
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        T = np.eye(4)
        T[:3, :3] = np.stack([right, down, fwd], 1)
        T[:3, 3] = cam
        v, u = np.mgrid[0:h, 0:w]
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1) @ T[:3, :3].T
        t = (0.7 - cam[2]) / np.where(np.abs(rays[..., 2]) < 1e-6, -1e-6, rays[..., 2])     # hit the table plane
        depth = np.clip(t * 1.0, 0, 4)                                 # (camera z = distance along the axis here)
        rgb = (rng.random((h, w, 3)) * 60 + 100).astype(np.uint8)
        Image.fromarray(rgb).save(root / "color" / f"{k}.jpg")
        Image.fromarray((depth * 1000).astype(np.uint16)).save(root / "depth" / f"{k}.png")
        Image.fromarray(np.full((h, w), 2, np.uint8)).save(root / "conf" / f"{k}.png")
        np.savetxt(root / "intrinsic" / f"{k}.txt", K)
        np.savetxt(root / "pose" / f"{k}.txt", T)


def make_model(root, upstream):
    """A 2-layer Qwen-3D (head size 128, as the real one) with random weights, a word-level tokenizer and the real Qwen image processor."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen2TokenizerFast
    from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
               "<|image_pad|>", "<|video_pad|>"]
    words = ("system you are a helpful assistant user the cup closest to can next ball on table red solo left right "
             "of which \n , .").split(" ")
    vocab = {t: i for i, t in enumerate(special + words + ["[UNK]"])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = Qwen2TokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.add_special_tokens({"additional_special_tokens": special})
    proc = Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(), tokenizer=tok)
    proc.save_pretrained(str(root / "processor"))

    pc_id = len(tok) + 0                                              # what add_special_tokens will give it
    cfg = up_config(upstream)(vocab_size=len(tok) + 8, hidden_size=256, intermediate_size=256, num_hidden_layers=2,
                              num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                              # head size 128 and the 3D rope sections as in the real model (the rope assumes them)
                              rope_scaling={"type": "default", "mrope_section": [16, 24, 24]}, tie_word_embeddings=True,
                              pointcloud_token_id=pc_id,
                              vision_start_token_id=tok.convert_tokens_to_ids("<|vision_start|>"),
                              vision_end_token_id=tok.convert_tokens_to_ids("<|vision_end|>"),
                              vision_token_id=tok.convert_tokens_to_ids("<|vision_pad|>"),
                              image_token_id=tok.convert_tokens_to_ids("<|image_pad|>"),
                              video_token_id=tok.convert_tokens_to_ids("<|video_pad|>"),
                              vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                                                 out_hidden_size=256, patch_size=14, spatial_merge_size=2,
                                                 temporal_patch_size=2, window_size=112, fullatt_block_indexes=[1],
                                                 in_channels=3))
    torch.manual_seed(0)
    model = upstream.Model._from_config(cfg, attn_implementation="sdpa", torch_dtype=torch.float32)
    model.save_pretrained(str(root / "weights"), safe_serialization=True)
    return cfg


def up_config(upstream):
    return upstream.Config


def make_heads(path, upstream):
    import torch.nn as nn
    c = upstream.cfg
    dim = c.MODEL.MASK_FORMER.HIDDEN_DIM
    heads = nn.Module()
    heads.connector = nn.Sequential(nn.Linear(256, dim), nn.LayerNorm(dim))
    heads.text_connector = nn.Sequential(nn.Linear(256, dim), nn.LayerNorm(dim))
    heads.mask_decoder = upstream.MaskDecoder(**upstream.MaskDecoder.from_config(c, c.MODEL.SEM_SEG_HEAD.CONVS_DIM, True))
    torch.save(heads.state_dict(), str(path))


@unittest.skipIf(torch is None or not heavy_available(), "needs transformers 4.51.1, accelerate, safetensors and the upstream checkout")
class TestEndToEndCPU(unittest.TestCase):
    def test_quantize_rows(self):
        from ground3d import quant
        w = torch.randn(10000, 64)
        q, s = quant.quantize_rows(w, "cpu", rows=3000)
        back = q.float() * s.float()
        self.assertLess((back - w).abs().max().item(), (w.abs().amax(1, keepdim=True) / 127 * 0.51 + 0.01).max().item())

    def test_geometry_matches_upstream(self):
        """Token points and 5 cm voxels exactly as upstream's own unproject / resize / voxelize produce them."""
        import types
        from ground3d import upstream as up_mod
        up_mod.load(upstream_root())
        import qwen3d.modeling.backproject.backproject as bp
        cfg = types.SimpleNamespace(MULTIVIEW_XYZ_SCALES=[28, 7], INPUT=types.SimpleNamespace(MIN_PIXEL=3136, MAX_PIXEL=390000,
                                    VOXEL_SIZE=[0.05, 0.05]), MODEL=types.SimpleNamespace(INTERPOLATION_METHOD="nearest"),
                                    get=lambda k, d=None: d)
        with tempfile.TemporaryDirectory() as t:
            make_scan(Path(t) / "scan", n=3)
            frames = [scene.load_frame(Path(t) / "scan", k) for k in range(3)]
            gh, gw = scene.token_grid(640, 480)
            depths = torch.stack([torch.from_numpy(f[1]) for f in frames])[None]
            poses = torch.stack([torch.from_numpy(f[4]).float() for f in frames])[None]
            Ks = torch.stack([torch.from_numpy(f[3]).float() for f in frames])[None]
            xyz = bp.unproject(Ks, poses, depths, mask_valid=False)
            ref = bp.interpolate_depth(xyz, [torch.zeros(3, 1, gh, gw)], method="nearest")[0].reshape(1, -1, 3)
            ref_ids = bp.voxelization(ref.clone(), 0.05)[0]
            mine = torch.cat([torch.nn.functional.interpolate(
                scene.world_points(torch.from_numpy(f[1]), torch.from_numpy(f[3]).float(), torch.from_numpy(f[4]).float())
                .permute(2, 0, 1)[None], size=(gh, gw), mode="nearest")[0].permute(1, 2, 0).reshape(-1, 3) for f in frames])
            self.assertTrue(torch.allclose(mine, ref[0], atol=1e-5))
            self.assertTrue(torch.equal(scene.voxel_ids(mine, 0.05), ref_ids))



    def test_tiny_model(self):
        from ground3d import upstream
        up = upstream.load(upstream_root())
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            make_scan(root / "scan")
            make_model(root, up)
            make_heads(root / "heads.pth", up)
            out = root / "out"
            rc = ground.main([str(root / "scan"), "-q", "the cup closest to the can", "-q", "the ball", "--device", "cpu",
                              "--upstream", str(upstream_root()), "--weights", str(root / "weights"),
                              "--heads", str(root / "heads.pth"), "--processor", str(root / "processor"),
                              "--frames", "4", "--out", str(out)])
            self.assertEqual(rc, 0)
            res = json.loads((out / "results.json").read_text())
            self.assertEqual(res["frames"], 4)
            self.assertGreater(res["voxels"], 10)
            self.assertGreater(res["target_points"], 10)
            a = res["answers"]
            self.assertEqual([x["query"] for x in a], ["the cup closest to the can", "the ball"])
            self.assertEqual(a[0]["target"], "cup")
            for x in a:
                self.assertTrue(0 <= x["score"] <= 1)
                self.assertEqual(len(x["top5"]), 5)


if __name__ == "__main__":
    unittest.main()
