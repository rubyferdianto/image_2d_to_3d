"""Publish the web UI (app.py) as a Hugging Face Space, i.e. a public live demo.

    .venv\\Scripts\\hf auth login                          # once, with a Write token
    .venv\\Scripts\\python scripts\\deploy_space.py         # -> <you>/image_2d_to_3d
    .venv\\Scripts\\python scripts\\deploy_space.py owner/name --out build\\space   # write files only

Uploads app.py, img2three/, samples/ and the vendored TripoSR package (third_party/TripoSR, see
README), plus a Space README (front matter) and a requirements.txt derived from ours. The free
CPU hardware works, but a 3D model takes minutes there; GPU hardware is a paid Space setting.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRIPOSR = ROOT / "third_party" / "TripoSR"
GITHUB = "https://github.com/rubyferdianto/image_2d_to_3d"


def space_readme() -> str:
    # The Space installs this exact Gradio version; app.py uses Gradio 6 APIs.
    return f"""---
title: 2D to 3D
emoji: 🧊
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: {version("gradio")}
python_version: "3.12"
app_file: app.py
short_description: Turn a photo into a 3D model, depth map or stereo image
---

Live demo of [image_2d_to_3d]({GITHUB}): single-image 3D reconstruction with
[TripoSR](https://github.com/VAST-AI-Research/TripoSR) (MIT) and monocular depth with
[Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) (Small, Apache-2.0).

On free CPU hardware a full 3D model takes a few minutes; the 360° turntable is off by default.
Uploaded images are stored on this Space only until it restarts.
"""


def space_requirements() -> str:
    """Our requirements.txt without gradio (installed by the Space SDK) and pytest, plus torch."""
    keep = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#") and not name.startswith(("gradio", "pytest")):
            keep.append(name)
    return "\n".join(["torch", *keep]) + "\n"


def files() -> dict[str, Path | bytes]:
    """Space path -> local file, or generated content."""
    if not (TRIPOSR / "tsr").is_dir():
        raise SystemExit(f"Missing {TRIPOSR}; clone it first: "
                         "git clone https://github.com/VAST-AI-Research/TripoSR third_party/TripoSR")
    out: dict[str, Path | bytes] = {"app.py": ROOT / "app.py", "third_party/TripoSR/LICENSE": TRIPOSR / "LICENSE"}
    for pattern in ("img2three/*.py", "samples/*", "third_party/TripoSR/tsr/**/*.py"):
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file():
                out[p.relative_to(ROOT).as_posix()] = p
    out["README.md"] = space_readme().encode()
    out["requirements.txt"] = space_requirements().encode()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_id", nargs="?", help="owner/space-name (default: <you>/image_2d_to_3d)")
    ap.add_argument("--out", type=Path, help="write the Space files here instead of uploading")
    args = ap.parse_args()
    bundle = files()

    if args.out:
        for rel, src in bundle.items():
            dest = args.out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src if isinstance(src, bytes) else src.read_bytes())
        print(f"Wrote {len(bundle)} files to {args.out}")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    repo_id = args.repo_id or f"{api.whoami()['name']}/image_2d_to_3d"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=src if isinstance(src, bytes) else str(src))
           for rel, src in bundle.items()]
    api.create_commit(repo_id, ops, repo_type="space", commit_message=f"Deploy from {GITHUB}")
    print(f"Uploaded {len(ops)} files. The Space builds in a few minutes: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
