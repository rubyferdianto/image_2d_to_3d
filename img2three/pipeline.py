"""End-to-end: image -> depth -> {depth map, point cloud, mesh, stereo, anaglyph, wiggle GIF}."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import geometry, stereo
from .depth import estimate_depth

ALL_OUTPUTS = ("depth", "points", "mesh", "sbs", "anaglyph", "wiggle")


@dataclass
class Options:
    model: str = "small"
    fov: float = 55.0               # assumed horizontal field of view, degrees
    near: float = 1.0               # relative models only: depth of nearest / farthest pixel
    far: float = 10.0
    max_size: int = 1280            # cap on the working image's long side
    mesh_size: int = 512            # long side of the mesh grid (vertex count ~ mesh_size^2 * aspect)
    edge_threshold: float = 0.04    # drop mesh faces across relative depth jumps larger than this
    stereo_strength: float = 0.03   # max parallax as fraction of width
    convergence: float = 0.5        # disparity that sits on the screen plane
    outputs: tuple[str, ...] = field(default=ALL_OUTPUTS)


def _fit(img: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def colorize(disparity: np.ndarray) -> np.ndarray:
    bgr = cv2.applyColorMap((disparity * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def convert(image: Image.Image | str | Path, out_dir: str | Path, opts: Options | None = None) -> dict[str, Path]:
    opts = opts or Options()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = _fit(ImageOps.exif_transpose(image).convert("RGB"), opts.max_size)
    rgb = np.asarray(image)

    d = estimate_depth(image, opts.model, opts.near, opts.far)
    files: dict[str, Path] = {}

    if "depth" in opts.outputs:
        files["depth"] = out_dir / "depth.png"
        Image.fromarray(colorize(d.disparity)).save(files["depth"])
        files["depth16"] = out_dir / "depth16.png"
        Image.fromarray((d.disparity * 65535).astype(np.uint16)).save(files["depth16"])

    if {"points", "mesh"} & set(opts.outputs):
        small = _fit(image, opts.mesh_size)
        w, h = small.size
        depth_s = cv2.resize(d.depth, (w, h), interpolation=cv2.INTER_NEAREST)
        rgb_s = np.asarray(small)
        K = geometry.intrinsics(w, h, opts.fov)
        if "points" in opts.outputs:
            files["points"] = out_dir / "points.ply"
            geometry.build_point_cloud(depth_s, rgb_s, K).export(files["points"])
        if "mesh" in opts.outputs:
            files["mesh"] = out_dir / "mesh.glb"
            geometry.build_mesh(depth_s, rgb_s, K, opts.edge_threshold).export(files["mesh"])

    if {"sbs", "anaglyph"} & set(opts.outputs):
        left, right = stereo.stereo_pair(rgb, d.disparity, opts.stereo_strength, opts.convergence)
        if "sbs" in opts.outputs:
            files["sbs"] = out_dir / "stereo_sbs.jpg"
            Image.fromarray(stereo.side_by_side(left, right)).save(files["sbs"], quality=92)
        if "anaglyph" in opts.outputs:
            files["anaglyph"] = out_dir / "anaglyph.jpg"
            Image.fromarray(stereo.anaglyph(left, right)).save(files["anaglyph"], quality=92)

    if "wiggle" in opts.outputs:
        files["wiggle"] = out_dir / "wiggle.gif"
        gif_img = _fit(image, 640)
        gif_disp = cv2.resize(d.disparity, gif_img.size, interpolation=cv2.INTER_NEAREST)
        frames = stereo.wiggle_frames(np.asarray(gif_img), gif_disp, opts.stereo_strength,
                                      convergence=opts.convergence)
        stereo.save_gif(frames, files["wiggle"])

    return files
