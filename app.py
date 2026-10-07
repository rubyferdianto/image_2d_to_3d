"""Gradio web UI: .venv/Scripts/python app.py  ->  http://127.0.0.1:7860

Images are read from disk on the server side (no browser upload needed): pick one from
samples/ or type any local path. Results are also saved under outputs/<image name>/.
"""
import threading
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from img2three import Options, convert
from img2three.depth import MODELS

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
OUTPUTS = ROOT / "outputs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_samples() -> list[str]:
    return sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def resolve(choice: str | None) -> Path:
    """A samples/ file name, or any absolute/relative path typed by the user."""
    if not choice:
        raise gr.Error("Pick an image from the list or type a file path.")
    raw = choice.strip().strip('"')
    for p in (SAMPLES / raw, Path(raw), ROOT / raw):
        if p.is_file():
            return p
    raise gr.Error(f"File not found: {raw}")


def preview(choice):
    try:
        return Image.open(resolve(choice))
    except gr.Error:
        return None


def refresh():
    return gr.Dropdown(choices=list_samples())


def run_object(choice, mc_resolution, remove_bg, turntable):
    from img2three.object3d import reconstruct

    src = resolve(choice)
    out = OUTPUTS / src.stem
    t0 = time.perf_counter()
    f = reconstruct(src, out, int(mc_resolution), remove_bg=remove_bg, turntable=turntable)
    msg = f"Done in {time.perf_counter() - t0:.1f}s. Saved to: {out}"
    return str(f["mesh"]), str(f["turntable"]) if "turntable" in f else None, str(f["input"]), msg


def run(choice, model, fov, edge_threshold, stereo_strength, convergence):
    src = resolve(choice)
    out = OUTPUTS / src.stem
    opts = Options(model=model, fov=fov, edge_threshold=edge_threshold,
                   stereo_strength=stereo_strength, convergence=convergence)
    f = convert(src, out, opts)
    return str(f["depth"]), str(f["mesh"]), str(f["anaglyph"]), str(f["wiggle"]), str(f["sbs"]), \
        f"Saved to: {out}"


def source_picker():
    """Dropdown of samples/ images that also accepts a typed path, plus a preview."""
    with gr.Row():
        pick = gr.Dropdown(list_samples(), label="Image (from samples/ — or type/paste a full path)",
                           allow_custom_value=True, scale=4)
        gr.Button("↻ Refresh", scale=1, min_width=80).click(refresh, None, pick)
    shown = gr.Image(label="Preview", interactive=False, height=240)
    pick.change(preview, pick, shown)
    return pick


with gr.Blocks(title="2D → 3D") as demo:
    gr.Markdown("## 2D → 3D\nPut images in `samples/` (then press Refresh), or type a path like "
                "`C:\\Users\\me\\Pictures\\photo.jpg`. Results are saved under `outputs/`.")
    with gr.Tab("Full 3D object (TripoSR)"):
        gr.Markdown("One object per photo works best. The background is removed automatically; "
                    "the back and sides are predicted.")
        with gr.Row():
            with gr.Column(scale=1):
                o_pick = source_picker()
                o_res = gr.Slider(128, 384, 256, step=32, label="Mesh resolution")
                o_bg = gr.Checkbox(True, label="Remove background")
                o_tt = gr.Checkbox(True, label="360° turntable preview (+~5 s)")
                o_btn = gr.Button("Generate 3D", variant="primary")
                o_cut = gr.Image(label="What the model sees")
            with gr.Column(scale=2):
                o_mesh = gr.Model3D(label="3D model (drag to orbit)", clear_color=[0.9, 0.9, 0.9, 1.0])
                o_turn = gr.Image(label="360° turntable")
                o_saved = gr.Textbox(label="Output folder", interactive=False)
        o_btn.click(run_object, [o_pick, o_res, o_bg, o_tt], [o_mesh, o_turn, o_cut, o_saved])

    with gr.Tab("2.5D depth / stereo"):
        gr.Markdown("Monocular depth (Depth Anything V2) → mesh, point cloud, stereo & 3D photo.")
        with gr.Row():
            with gr.Column(scale=1):
                pick = source_picker()
                model = gr.Dropdown(list(MODELS), value="small", label="Depth model")
                fov = gr.Slider(20, 120, 55, step=1, label="Camera FOV (°)")
                edge = gr.Slider(0.005, 0.3, 0.04, step=0.005, label="Mesh edge cut threshold")
                strength = gr.Slider(0.0, 0.08, 0.03, step=0.005, label="Stereo strength")
                conv = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="Convergence (screen plane)")
                btn = gr.Button("Convert", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    depth = gr.Image(label="Depth")
                    wiggle = gr.Image(label="3D photo (wiggle)")
                mesh = gr.Model3D(label="Mesh (drag to orbit)", clear_color=[0.1, 0.1, 0.1, 1.0])
                with gr.Row():
                    ana = gr.Image(label="Anaglyph (red/cyan glasses)")
                    sbs = gr.Image(label="Side-by-side stereo")
                saved = gr.Textbox(label="Output folder", interactive=False)
        btn.click(run, [pick, model, fov, edge, strength, conv], [depth, mesh, ana, wiggle, sbs, saved])

if __name__ == "__main__":
    from img2three.object3d import warmup

    # Load TripoSR in the background while the UI starts, so the first "Generate 3D" isn't slow.
    threading.Thread(target=warmup, daemon=True).start()
    demo.launch(allowed_paths=[str(OUTPUTS), str(SAMPLES)])
