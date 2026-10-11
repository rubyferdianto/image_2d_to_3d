"""Full 360° object reconstruction from a single image with TripoSR (third_party/TripoSR, MIT).

Unlike the depth pipeline (2.5D, only the visible side), this predicts the whole
object — back and sides included. Works best on one object with a clear outline.
"""
from __future__ import annotations

import re
import sys
import threading
import types
from dataclasses import replace
from functools import lru_cache, wraps
from pathlib import Path

import numpy as np
from PIL import Image

from .blender import BlenderOptions, find_blender, render_mesh

TRIPOSR_DIR = Path(__file__).resolve().parent.parent / "third_party" / "TripoSR"


def _install_mcubes_shim() -> None:
    """TripoSR imports `torchmcubes` (needs a CUDA compile); route it to PyMCubes instead."""
    if "torchmcubes" in sys.modules:
        return
    import mcubes
    import torch

    def marching_cubes(volume, iso):
        v, f = mcubes.marching_cubes(volume.cpu().numpy(), iso)
        # torchmcubes returns (x, y, z) = (k, j, i); TripoSR swaps back to index order.
        v = np.ascontiguousarray(v[:, ::-1], dtype=np.float32)
        return torch.from_numpy(v), torch.from_numpy(f.astype(np.int64))

    sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=marching_cubes)


_VIT_RENAMES = [  # transformers<5 ViT names (checkpoint) -> transformers>=5 names (model)
    (re.compile(r"\.encoder\.layer\.(\d+)\.attention\.attention\.query\."), r".layers.\1.attention.q_proj."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.attention\.attention\.key\."), r".layers.\1.attention.k_proj."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.attention\.attention\.value\."), r".layers.\1.attention.v_proj."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.attention\.output\.dense\."), r".layers.\1.attention.o_proj."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.intermediate\.dense\."), r".layers.\1.mlp.fc1."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.output\.dense\."), r".layers.\1.mlp.fc2."),
    (re.compile(r"\.encoder\.layer\.(\d+)\."), r".layers.\1."),
]


def _adapt_state_dict(ckpt: dict, expected: set[str]) -> dict:
    """Rename old ViT keys only if the installed transformers uses the new layout."""
    if all(k in expected for k in ckpt):
        return ckpt
    out = {}
    for k, v in ckpt.items():
        for pat, rep in _VIT_RENAMES:
            k2 = pat.sub(rep, k)
            if k2 != k:
                k = k2
                break
        out[k] = v
    return out


def _load_once(fn):
    """Cache like lru_cache, but a concurrent first call (e.g. Generate 3D clicked during the startup
    warmup) waits for the load in progress instead of downloading and loading a second copy."""
    cached, lock = lru_cache(maxsize=1)(fn), threading.Lock()

    @wraps(fn)
    def wrapper():
        with lock:
            return cached()

    return wrapper


@_load_once
def _load():
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    _install_mcubes_shim()
    if str(TRIPOSR_DIR) not in sys.path:
        sys.path.insert(0, str(TRIPOSR_DIR))
    from tsr.system import TSR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(hf_hub_download("stabilityai/TripoSR", "config.yaml"))
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    ckpt = torch.load(hf_hub_download("stabilityai/TripoSR", "model.ckpt"), map_location="cpu")
    model.load_state_dict(_adapt_state_dict(ckpt, set(model.state_dict())))
    model.renderer.set_chunk_size(131072)  # ~3 GB VRAM peak; 2x faster than 8192
    return model.to(device).eval(), device


@_load_once
def _rembg_session():
    import rembg

    # rembg>=2.0.8x defaults to bria-rmbg (1 GB, CC-BY-NC, slow on CPU); u2net is what TripoSR was built with.
    return rembg.new_session("u2net")


def prepare(image: Image.Image, foreground_ratio: float = 0.85, remove_bg: bool = True) -> Image.Image:
    """Cut out the object, center it, and composite on the neutral gray TripoSR expects."""
    from tsr.utils import remove_background, resize_foreground  # noqa: E402 (path set in _load)

    if not remove_bg:
        return image.convert("RGB")
    rgba = resize_foreground(remove_background(image, _rembg_session()), foreground_ratio)
    a = np.asarray(rgba).astype(np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4]) * 0.5
    return Image.fromarray((rgb * 255).astype(np.uint8))


def warmup() -> None:
    """Load TripoSR and the background remover up front (~15 s) so the first request is fast."""
    _load()
    _rembg_session()


def reconstruct(image: Image.Image | str | Path, out_dir: str | Path, mc_resolution: int = 256,
                foreground_ratio: float = 0.85, remove_bg: bool = True, turntable: bool = True,
                n_views: int = 24, view_size: int = 256, renderer: str = "native",
                blender: BlenderOptions | None = None) -> dict[str, Path]:
    """Reconstruct a mesh, then render it with TripoSR or optional headless Blender."""
    if renderer not in ("native", "blender"):
        raise ValueError(f"Unknown renderer: {renderer!r}; choose 'native' or 'blender'.")
    blender_opts = blender or BlenderOptions()
    if renderer == "blender":
        blender_opts = replace(blender_opts, executable=find_blender(blender_opts.executable))

    import torch

    model, device = _load()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    files = {"input": out_dir / "object_input.png"}
    inp = prepare(image, foreground_ratio, remove_bg)
    inp.save(files["input"])

    with torch.no_grad():
        codes = model([inp], device=device)

    mesh = model.extract_mesh(codes, True, resolution=mc_resolution)[0]
    if mesh.volume < 0:  # keep normals pointing outward regardless of marching-cubes winding
        mesh.invert()
    files["mesh"] = out_dir / "object.glb"
    mesh.export(files["mesh"])
    files["obj"] = out_dir / "object.obj"
    mesh.export(files["obj"])

    if renderer == "blender":
        files.update(render_mesh(files["mesh"], out_dir, blender_opts, prefix="object", mode="object",
                                 turntable=turntable, n_views=n_views))
        return files

    if not turntable:
        return files

    # Neural (NeRF) turntable render of the predicted 3D object — by far the slowest step,
    # so it runs in fp16 (2x faster, visually identical).
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        views = model.render(codes, n_views=n_views, elevation_deg=15.0, height=view_size, width=view_size,
                             return_type="pil")[0]
    files["turntable"] = out_dir / "object_turntable.gif"
    views[0].save(files["turntable"], save_all=True, append_images=views[1:], duration=1000 * 3 // n_views, loop=0)
    picks = [views[i * n_views // 6] for i in range(6)]
    sheet = Image.new("RGB", (view_size * 7, view_size), (128, 128, 128))
    for i, im in enumerate([inp.resize((view_size, view_size))] + picks):
        sheet.paste(im, (i * view_size, 0))
    files["views"] = out_dir / "object_views.jpg"
    sheet.save(files["views"], quality=92)
    return files
