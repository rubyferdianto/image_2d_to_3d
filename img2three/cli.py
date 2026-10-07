"""Command line: python -m img2three photo.jpg -o outputs/photo"""
from __future__ import annotations

import argparse
from pathlib import Path

from .depth import MODELS
from .pipeline import ALL_OUTPUTS, Options, convert


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Convert a 2D image into 3D (depth, point cloud, mesh, stereo).")
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("outputs"), help="output root (one subfolder per image)")
    p.add_argument("--mode", choices=["depth", "object", "both"], default="depth",
                   help="depth: 2.5D from depth map; object: full 360-degree mesh via TripoSR (single object photos)")
    p.add_argument("--model", choices=MODELS, default="small")
    p.add_argument("--outputs", default=",".join(ALL_OUTPUTS), help=f"comma list from: {','.join(ALL_OUTPUTS)}")
    p.add_argument("--fov", type=float, default=55.0)
    p.add_argument("--near", type=float, default=1.0)
    p.add_argument("--far", type=float, default=10.0)
    p.add_argument("--mesh-size", type=int, default=512)
    p.add_argument("--edge-threshold", type=float, default=0.04)
    p.add_argument("--stereo-strength", type=float, default=0.03)
    p.add_argument("--convergence", type=float, default=0.5)
    p.add_argument("--mc-resolution", type=int, default=256, help="object mode: marching cubes grid size")
    p.add_argument("--no-turntable", action="store_true", help="object mode: skip the 360-degree preview render (~5 s)")
    p.add_argument("--no-remove-bg", action="store_true", help="object mode: input already on a gray background")
    a = p.parse_args(argv)

    outputs = tuple(s.strip() for s in a.outputs.split(",") if s.strip())
    bad = set(outputs) - set(ALL_OUTPUTS)
    if bad:
        p.error(f"unknown outputs: {', '.join(sorted(bad))}")
    opts = Options(model=a.model, fov=a.fov, near=a.near, far=a.far, mesh_size=a.mesh_size,
                   edge_threshold=a.edge_threshold, stereo_strength=a.stereo_strength,
                   convergence=a.convergence, outputs=outputs)

    for img in a.images:
        files = {}
        if a.mode in ("depth", "both"):
            files |= convert(img, a.out / img.stem, opts)
        if a.mode in ("object", "both"):
            from .object3d import reconstruct

            files |= {f"object_{k}": v for k, v in reconstruct(img, a.out / img.stem, a.mc_resolution,
                                                                remove_bg=not a.no_remove_bg,
                                                                turntable=not a.no_turntable).items()}
        print(f"{img}:")
        for name, path in files.items():
            print(f"  {name:16s} {path}")


if __name__ == "__main__":
    main()
