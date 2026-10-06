# image_2d_to_3d

Convert a single 2D photo into 3D. Two modes:

| mode | what you get | best for |
|---|---|---|
| `--mode object` | **full 360° mesh** (back and sides predicted by TripoSR) | one object per photo: products, toys, furniture, animals |
| `--mode depth` (default) | 2.5D depth mesh, point cloud, stereo, anaglyph, 3D-photo GIF | whole scenes, rooms, landscapes |

```bash
.venv/Scripts/python -m img2three samples/chair.png --mode object -o outputs
```

Object mode writes `object.glb` / `object.obj`, `object_turntable.gif` (360° render), `object_views.jpg`
and `object_input.png` (the background-removed image the model sees). It needs the extra packages
`omegaconf einops rembg[cpu] pymcubes xatlas imageio` plus the vendored `third_party/TripoSR`
(`git clone https://github.com/VAST-AI-Research/TripoSR third_party/TripoSR`). It runs on an
8 GB GPU in about 15–20 s per image after the first model download.

## Depth mode

```
image ──► Depth Anything V2 ──► depth map ─┬─► point cloud (.ply)
                                           ├─► 2.5D vertex-colored mesh (.glb)
                                           └─► new-view synthesis (DIBR) ─┬─► side-by-side stereo
                                                                          ├─► red/cyan anaglyph
                                                                          └─► "wiggle" 3D-photo GIF
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs cu128+
.\.venv\Scripts\pip install -r requirements.txt
```

## Usage

CLI:

```powershell
.\.venv\Scripts\python -m img2three photo.jpg -o outputs
.\.venv\Scripts\python -m img2three *.jpg --model metric-indoor --outputs depth,mesh
```

Web UI (with an interactive 3D viewer):

```powershell
.\.venv\Scripts\python app.py      # http://127.0.0.1:7860
```

Python:

```python
from img2three import Options, convert
convert("photo.jpg", "outputs/photo", Options(model="base", fov=60))
```

## Outputs

| file | what |
|---|---|
| `depth.png` / `depth16.png` | colorized / 16-bit normalized disparity (white = near) |
| `points.ply` | colored point cloud (MeshLab, CloudCompare, Blender) |
| `mesh.glb` | vertex-colored mesh, depth discontinuities cut (Blender, three.js, Windows 3D Viewer) |
| `stereo_sbs.jpg` | left/right pair for VR or parallel viewing |
| `anaglyph.jpg` | red/cyan glasses |
| `wiggle.gif` | swaying-camera 3D-photo effect |

## Tuning knobs

- `--model`: `small` (fast, Apache-2.0), `base`/`large` (better, **CC-BY-NC**), `metric-indoor`/`metric-outdoor` (depth in meters).
- `--fov`: assumed camera field of view. Wrong values squash or stretch the geometry; phones are about 65–75°.
- `--near/--far`: depth range for relative models (only the ratio matters for shape).
- `--edge-threshold`: lower values cut more "skin" between foreground and background.
- `--stereo-strength`, `--convergence`: amount of parallax, and which depth sits on the screen plane.

## Layout

```
img2three/
  depth.py     depth estimation (model registry, relative→depth conversion)
  geometry.py  unprojection, point cloud, mesh triangulation w/ discontinuity culling
  stereo.py    DIBR view synthesis with z-buffer + inpainting, anaglyph, GIF
  pipeline.py  orchestration (Options, convert)
  cli.py       command line
app.py         Gradio UI
tests/         model-free unit tests (pytest)
```

## Roadmap

1. **Full 360° object meshes**: add an `object` backend using an image-to-3D model (TripoSR fits 8 GB; Hunyuan3D-2 / TRELLIS give higher quality) with background removal (`rembg`).
2. **Better disocclusion fill**: replace OpenCV inpainting with diffusion inpainting (LaMa / SD-inpaint) for larger camera moves.
3. **Layered depth / 3D Ken Burns video**: render camera paths to MP4.
4. **Textured mesh export**: a UV texture instead of vertex colors (sharper at low poly), with mesh decimation.
5. **Video input**: temporally consistent depth (Video Depth Anything) → stereo 3D video.
