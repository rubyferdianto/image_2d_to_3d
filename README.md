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
8 GB GPU in about 15–20 s per image after the first model download, using the native renderer.

Both modes can also use **Blender for final rendering**, producing a PNG and an editable
`.blend` scene with a camera and vertex-color materials. Object scenes include studio lighting;
depth scenes preserve the photograph's existing lighting. Native rendering remains the default.

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

Click a sample, drag your own image onto the preview (or click it to browse, or paste from the
clipboard), or type a local path. Uploads are copied to `outputs/_uploads/`. Without a GPU the
360° turntable is off by default, since it takes several times longer than the mesh itself.

Publish the web UI as a Hugging Face Space (a public live demo):

```powershell
.\.venv\Scripts\hf auth login                 # once, with a Write token
.\.venv\Scripts\python scripts\deploy_space.py  # -> https://huggingface.co/spaces/<you>/image_2d_to_3d
```

On a Space, typed paths are limited to `samples/` and uploads, each upload gets its own folder,
and the Blender options are hidden. Free CPU hardware works, but a 3D model takes minutes.

Python:

```python
from img2three import Options, convert
convert("photo.jpg", "outputs/photo", Options(model="base", fov=60))
```

## Blender final rendering

Install **Blender 4.2 or newer** separately from [blender.org](https://www.blender.org/download/).
It is not a Python dependency. Put the Blender executable on `PATH`, set `BLENDER_PATH` to
its full path, or pass `--blender-path` for a single run. Paths containing spaces must be quoted.
Standard installation folders on Windows and macOS are also detected automatically.

```powershell
# Example Windows installation path; adjust it to your installed version.
$env:BLENDER_PATH = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"

# Full object: Blender final image, scene, turntable GIF, and contact sheet.
.\.venv\Scripts\python -m img2three samples/chair.png --mode object --renderer blender -o outputs

# Skip the turntable for a faster final image and scene.
.\.venv\Scripts\python -m img2three samples/chair.png --mode object --renderer blender --no-turntable -o outputs

# Depth mesh, viewed from the original camera direction.
.\.venv\Scripts\python -m img2three photo.jpg --renderer blender --render-size 1024 --render-samples 64 -o outputs

# Explicit executable and the Eevee engine.
.\.venv\Scripts\python -m img2three photo.jpg --renderer blender --blender-engine eevee --blender-path "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
```

In the web UI, expand **Advanced · final renderer** in either tab and select **Blender**; the
same section sets the executable, engine, render size, and samples. The final PNG appears in the
**Blender** result tab, with a downloadable Blender scene.

Cycles is the default engine and uses the CPU for portability. More samples reduce noise
but increase render time; the default is 32 samples with a 768-pixel longest side. `--blender-engine eevee`
selects Eevee, whose background rendering requires a working graphics driver. Blender
turntables can take several minutes, especially with Cycles.

Blender renders the extracted colored mesh in object mode, so the final image and turntable
match the exported geometry. The native object preview uses TripoSR's neural renderer and
may look different. Depth mode is still a source-facing 2.5D surface: rendering in Blender
does not generate hidden backs or sides. Requesting Blender always exports the required
mesh, even with a restricted selection such as `--outputs depth --renderer blender`.

Python:

```python
from img2three import Options, convert
from img2three.blender import BlenderOptions
from img2three.object3d import reconstruct

render_options = BlenderOptions(engine="CYCLES", resolution=1024, samples=64)
files = convert("photo.jpg", "outputs/photo",
                Options(renderer="blender", blender=render_options))
print(files["render"], files["blend"])

files = reconstruct("samples/chair.png", "outputs/chair", turntable=False,
                    renderer="blender", blender=render_options)
```

`BlenderOptions(executable=...)` can specify the executable directly. Its `timeout` setting
limits the Blender process to 600 seconds by default; increase it for larger renders or
turntables when needed.

## Outputs

| file | what |
|---|---|
| `depth.png` / `depth16.png` | colorized / 16-bit normalized disparity (white = near) |
| `points.ply` | colored point cloud (MeshLab, CloudCompare, Blender) |
| `mesh.glb` | vertex-colored mesh, depth discontinuities cut (Blender, three.js, Windows 3D Viewer) |
| `stereo_sbs.jpg` | left/right pair for VR or parallel viewing |
| `anaglyph.jpg` | red/cyan glasses |
| `wiggle.gif` | swaying-camera 3D-photo effect |
| `object.glb` / `object.obj` | full object mesh (object mode) |
| `object_input.png` | preprocessed input to TripoSR (object mode) |
| `object_turntable.gif` / `object_views.jpg` | object turntable and contact sheet, unless `--no-turntable` |
| `depth_render.png` / `depth_scene.blend` | Blender final image and editable scene (depth mode, `--renderer blender`) |
| `object_render.png` / `object_scene.blend` | Blender final image and editable scene (object mode, `--renderer blender`) |
| `depth_blender.log` / `object_blender.log` | Blender diagnostics, including failures and timeouts |

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
  object3d.py   full object reconstruction and native/Blender preview selection
  blender.py    Blender discovery, render options, and background render orchestration
  _blender_scene.py  scene setup and rendering inside Blender
  pipeline.py  orchestration (Options, convert)
  cli.py       command line
app.py         Gradio UI
tests/         model-free unit tests (pytest)
```

Run the model-free tests with `.\.venv\Scripts\python -m pytest -q`. To include actual Blender
rendering tests, set `BLENDER_TEST_PATH` to the Blender executable first. These tests use small
synthetic meshes and verify image orientation, colors, aspect ratio, turntables, and saved scenes.
Set `BLENDER_TEST_ENGINE=eevee` to run the same checks with Eevee instead of Cycles.

## Roadmap

1. **More object backends**: support alternatives to TripoSR, such as Hunyuan3D-2 / TRELLIS, for higher-quality reconstruction.
2. **Better disocclusion fill**: replace OpenCV inpainting with diffusion inpainting (LaMa / SD-inpaint) for larger camera moves.
3. **Layered depth / 3D Ken Burns video**: render camera paths to MP4.
4. **Textured mesh export**: a UV texture instead of vertex colors (sharper at low poly), with mesh decimation.
5. **Video input**: temporally consistent depth (Video Depth Anything) → stereo 3D video.
