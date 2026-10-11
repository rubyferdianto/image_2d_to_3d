"""Run the separately installed Blender application as a final mesh renderer."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from PIL import Image


@dataclass(frozen=True)
class BlenderOptions:
    executable: str | None = None
    engine: str = "CYCLES"
    resolution: int = 768  # longest image side, in pixels
    samples: int = 32
    timeout: float = 600

    def __post_init__(self):
        if self.engine not in {"CYCLES", "BLENDER_EEVEE_NEXT"}:
            raise ValueError("Blender engine must be CYCLES or BLENDER_EEVEE_NEXT.")
        for name in ("resolution", "samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Blender {name} must be a positive integer.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Blender timeout must be finite and positive.")


def find_blender(executable: str | Path | None = None) -> str:
    """Resolve an explicit path, BLENDER_PATH, PATH, or a standard installation."""
    requested = executable if executable is not None else os.environ.get("BLENDER_PATH")
    if requested is not None:
        path = Path(requested).expanduser()
        if str(requested).strip() and path.is_file():
            return str(path.resolve())
        found = shutil.which(str(requested)) if str(requested).strip() else None
        if found:
            return found
        raise FileNotFoundError(f"Blender executable not found: {requested!s}. Set --blender-path or BLENDER_PATH to the Blender executable.")
    found = shutil.which("blender")
    if found:
        return found
    candidates = []
    if sys.platform == "win32":
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"
        candidates = sorted(base.glob("Blender */blender.exe"),
                            key=lambda p: tuple(int(n) for n in re.findall(r"\d+", p.parent.name)),
                            reverse=True)
    elif sys.platform == "darwin":
        candidates = [Path("/Applications/Blender.app/Contents/MacOS/Blender"),
                      Path.home() / "Applications/Blender.app/Contents/MacOS/Blender"]
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    raise FileNotFoundError("Blender was not found. Install Blender 4.2 or newer, add it to PATH, "
                            "or set --blender-path / BLENDER_PATH to its executable.")


def _log_tail(path: Path) -> str:
    with path.open("rb") as log:
        log.seek(max(0, path.stat().st_size - 4000))
        return log.read().decode("utf-8", errors="replace").strip()


def render_mesh(mesh_path: str | Path, out_dir: str | Path, opts: BlenderOptions | None = None,
                *, prefix: str = "object", mode: str = "object", turntable: bool = False,
                n_views: int = 24, fov: float = 55.0, aspect_ratio: float = 1.0) -> dict[str, Path]:
    """Render a GLB in a fresh Blender process; publish outputs only after success.

    Object meshes use TripoSR's native Z-up coordinates. Depth meshes use the
    glTF camera coordinates produced by geometry.unproject, viewed from the origin.
    """
    opts = opts or BlenderOptions()
    if mode not in {"object", "depth"}:
        raise ValueError("Blender render mode must be object or depth.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("Render prefix must contain only letters, numbers, underscores or hyphens.")
    if not math.isfinite(fov) or not 0 < fov < 180:
        raise ValueError("Camera fov must be between 0 and 180 degrees.")
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("Camera aspect_ratio must be finite and positive.")
    if isinstance(n_views, bool) or not isinstance(n_views, int) or n_views < 1:
        raise ValueError("n_views must be a positive integer.")
    if turntable and mode == "depth":
        raise ValueError("A depth mesh only supports the source view, not a full turntable.")
    mesh_path = Path(mesh_path).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    executable = find_blender(opts.executable)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{prefix}_blender.log"

    with tempfile.TemporaryDirectory(prefix="img2three-blender-") as tmp:
        work = Path(tmp)
        staged = {"render": work / f"{prefix}_render.png", "blend": work / f"{prefix}_scene.blend"}
        frames_dir = work / "frames"
        if turntable:
            frames_dir.mkdir()
        job = dict(mesh_path=str(mesh_path), render_path=str(staged["render"]),
                   blend_path=str(staged["blend"]), frames_dir=str(frames_dir) if turntable else None,
                   mode=mode, engine=opts.engine, resolution=opts.resolution, samples=opts.samples,
                   n_views=n_views, fov=fov, aspect_ratio=aspect_ratio,
                   final_render_path=str((out_dir / staged["render"].name).resolve()))
        job_path = work / "job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        command = [executable, "--background", "--factory-startup", "--python-exit-code", "1",
                   "--python", str(Path(__file__).with_name("_blender_scene.py")), "--", str(job_path)]
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=opts.timeout,
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        except (OSError, subprocess.SubprocessError) as exc:
            reason = f"timed out after {opts.timeout:g}s" if isinstance(exc, subprocess.TimeoutExpired) else "failed"
            raise RuntimeError(f"Blender render {reason}. See {log_path}.\n{_log_tail(log_path)}") from exc

        required = list(staged.values())
        frames = [frames_dir / f"frame_{i:04d}.png" for i in range(n_views)] if turntable else []
        missing = [p.name for p in required + frames if not p.is_file() or not p.stat().st_size]
        if missing:
            raise RuntimeError(f"Blender did not produce expected outputs: {', '.join(missing)}. "
                               f"See {log_path}.\n{_log_tail(log_path)}")
        if frames:
            images = []
            for frame in frames:
                with Image.open(frame) as im:
                    images.append(im.convert("RGB"))
            staged["turntable"] = work / f"{prefix}_turntable.gif"
            images[0].save(staged["turntable"], save_all=True, append_images=images[1:],
                           duration=max(10, 3000 // n_views), loop=0)
            staged["views"] = work / f"{prefix}_views.jpg"
            thumb_size = (min(256, images[0].width), min(256, images[0].height))
            sheet = Image.new("RGB", (thumb_size[0] * 6, thumb_size[1]))
            for i in range(6):
                sheet.paste(images[i * n_views // 6].resize(thumb_size, Image.Resampling.LANCZOS),
                            (i * thumb_size[0], 0))
            sheet.save(staged["views"], quality=92)
        files = {key: out_dir / path.name for key, path in staged.items()}
        for key, path in staged.items():
            shutil.copyfile(path, files[key])
    return files
