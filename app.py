"""Gradio web UI: .venv/Scripts/python app.py  ->  http://127.0.0.1:7860

Pick an image from samples/, upload/paste your own, or type any local path. Uploads are
copied to outputs/_uploads/; results are saved under outputs/<image name>/.
"""
import shutil
import threading
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from img2three import Options, convert
from img2three.blender import BlenderOptions
from img2three.depth import MODELS

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
OUTPUTS = ROOT / "outputs"
UPLOADS = OUTPUTS / "_uploads"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIEWER_BG = [0.07, 0.08, 0.1, 1.0]  # dark "studio" backdrop shared by both 3D viewers

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_background_fill="white",
    block_background_fill_dark="*neutral_900",
    block_border_color="*neutral_200",
    block_border_color_dark="*neutral_800",
    block_shadow="none",
    block_label_text_weight="500",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover_dark="*primary_500",
    button_primary_text_color="white",
    slider_color="*primary_500",
)

CSS = """
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; }
.app-header { display: flex; align-items: center; justify-content: space-between; gap: 16px;
              flex-wrap: wrap; padding: 8px 4px 4px; }
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
              color: white; background: linear-gradient(135deg, var(--primary-500), var(--secondary-600)); }
.brand h1 { margin: 0; font-size: 20px; font-weight: 650; letter-spacing: -0.01em; }
.brand p { margin: 2px 0 0; font-size: 13px; color: var(--body-text-color-subdued); }
.chip { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
        font-size: 12.5px; border: 1px solid var(--border-color-primary);
        background: var(--block-background-fill); color: var(--body-text-color); }
.chip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--neutral-400); }
.chip.cpu .dot { background: #f59e0b; }
.chip.gpu .dot { background: #22c55e; }
#mode-tabs { margin-top: 8px; }
.controls { background: var(--block-background-fill); border: 1px solid var(--block-border-color);
         border-radius: 16px; padding: 16px !important; gap: 14px !important; }
.controls .block { border: none !important; padding-left: 0 !important; padding-right: 0 !important; }
.controls .form { border: none !important; box-shadow: none !important; background: transparent !important; }
.step { display: flex; gap: 10px; align-items: flex-start; }
.step .num { flex: none; width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center;
             font-size: 12px; font-weight: 600; background: var(--primary-100); color: var(--primary-700); }
.dark .step .num { background: var(--primary-900); color: var(--primary-200); }
.step b { display: block; font-size: 14.5px; font-weight: 600; line-height: 24px; }
.step p { margin: 0; font-size: 12.5px; color: var(--body-text-color-subdued); }
.samples, .samples .gallery-container { height: auto !important; min-height: 0 !important; }
.samples .grid-wrap { min-height: 0 !important; height: auto !important; max-height: none !important;
                      padding: 0 !important; }
.samples .grid-container { grid-template-columns: repeat(5, minmax(0, 1fr)) !important; gap: 8px !important; }
.samples .thumbnail-item { width: 100% !important; height: auto !important; aspect-ratio: 1 / 1;
                           border-radius: 10px !important; }
.refresh-btn { align-self: flex-end; height: 42px; margin-bottom: 2px; }
.cta { min-height: 50px; font-size: 15.5px !important; font-weight: 600 !important; }
.stage { gap: 14px !important; }
.viewer { border-radius: 16px !important; overflow: hidden; background: rgb(18 20 26) !important; }
.app-footer { text-align: center; font-size: 12px; color: var(--body-text-color-subdued); padding: 8px 0 0; }
"""

CUBE_ICON = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="1.8" stroke-linejoin="round"><path d="M12 2 3 7v10l9 5 9-5V7z"/>'
             '<path d="m3 7 9 5 9-5M12 12v10"/></svg>')


def header_html(device: str = "Checking hardware…", kind: str = "") -> str:
    return (f'<header class="app-header"><div class="brand"><div class="brand-mark">{CUBE_ICON}</div>'
            '<div><h1>2D → 3D</h1><p>Turn a single photo into a 3D model, depth map or stereo image.</p></div>'
            f'</div><div class="chip {kind}"><span class="dot"></span>{device}</div></header>')


def device_status():
    """Header chip, and the turntable default: on CPU it takes ~5-10x longer than the mesh itself."""
    import torch

    if torch.cuda.is_available():
        return header_html(f"GPU · {torch.cuda.get_device_name(0)}", "gpu"), gr.skip()
    return (header_html("Running on CPU · expect minutes per model", "cpu"),
            gr.Checkbox(value=False, info="Off by default on CPU, where it can add 10+ minutes."))


def step(n: int, title: str, hint: str = "") -> None:
    sub = f"<p>{hint}</p>" if hint else ""
    gr.HTML(f'<div class="step"><span class="num">{n}</span><div><b>{title}</b>{sub}</div></div>',
            container=False, padding=False)


def list_samples() -> list[str]:
    return sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def sample_gallery() -> list[str]:
    return [str(SAMPLES / name) for name in list_samples()]


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
    return gr.Dropdown(choices=list_samples()), sample_gallery()


def pick_sample(evt: gr.SelectData):
    return list_samples()[evt.index]


def use_upload(path):
    """Copy an uploaded or pasted image to a stable path and select it like a sample."""
    if not path:
        return gr.skip()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / Path(path).name
    shutil.copyfile(path, dest)
    return str(dest)


def blender_options(renderer, executable, engine, resolution, samples):
    if renderer != "blender":
        return None
    return BlenderOptions(executable=(executable or "").strip().strip('"') or None, engine=engine,
                          resolution=int(resolution), samples=int(samples))


def run_object(choice, mc_resolution, remove_bg, turntable, renderer="native",
               blender_path="", blender_engine="CYCLES", render_size=768, render_samples=32):
    from img2three.object3d import reconstruct

    src = resolve(choice)
    out = OUTPUTS / src.stem
    t0 = time.perf_counter()
    try:
        opts = blender_options(renderer, blender_path, blender_engine, render_size, render_samples)
        f = reconstruct(src, out, int(mc_resolution), remove_bg=remove_bg, turntable=turntable,
                        renderer=renderer, blender=opts)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    msg = f"Done in {time.perf_counter() - t0:.1f}s. Saved to: {out}"
    return (str(f["mesh"]), str(f["turntable"]) if "turntable" in f else None,
            str(f["input"]), str(f["render"]) if "render" in f else None,
            str(f["blend"]) if "blend" in f else None, msg)


def run(choice, model, fov, edge_threshold, stereo_strength, convergence, renderer="native",
        blender_path="", blender_engine="CYCLES", render_size=768, render_samples=32):
    src = resolve(choice)
    out = OUTPUTS / src.stem
    try:
        render_opts = blender_options(renderer, blender_path, blender_engine, render_size, render_samples)
        opts = Options(model=model, fov=fov, edge_threshold=edge_threshold,
                       stereo_strength=stereo_strength, convergence=convergence,
                       renderer=renderer, blender=render_opts)
        f = convert(src, out, opts)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    return (str(f["depth"]), str(f["mesh"]), str(f["anaglyph"]), str(f["wiggle"]), str(f["sbs"]),
            str(f["render"]) if "render" in f else None,
            str(f["blend"]) if "blend" in f else None, f"Saved to: {out}")


def renderer_picker():
    with gr.Accordion("Advanced · final renderer", open=False):
        renderer = gr.Radio([("Native", "native"), ("Blender", "blender")], value="native",
                            label="Final renderer",
                            info="Blender 4.2+ (installed separately) adds a final PNG and an editable "
                                 "scene. Object turntables then also use Blender and can take minutes.")
        executable = gr.Textbox(label="Blender executable (optional)",
                                placeholder="Use BLENDER_PATH or Blender on PATH")
        engine = gr.Dropdown([("Cycles (CPU)", "CYCLES"), ("Eevee", "BLENDER_EEVEE_NEXT")],
                             value="CYCLES", label="Render engine")
        resolution = gr.Slider(128, 2048, 768, step=128, label="Render longest side (pixels)")
        samples = gr.Slider(1, 256, 32, step=1, label="Render samples")
    return [renderer, executable, engine, resolution, samples]


def source_picker():
    """Clickable sample thumbnails, a path box, and a preview that doubles as an upload drop zone."""
    gallery = gr.Gallery(sample_gallery(), show_label=False, columns=5, rows=1, height="auto",
                         allow_preview=False, object_fit="cover", buttons=[], elem_classes="samples")
    with gr.Row(equal_height=False):
        pick = gr.Dropdown(list_samples(), label="Image", allow_custom_value=True, scale=5,
                           info="Click a sample, drop your own image below, or paste a full path "
                                "such as C:\\Users\\me\\Pictures\\photo.jpg")
        reload = gr.Button("↻", scale=0, min_width=44, size="sm", elem_classes="refresh-btn")
    shown = gr.Image(label="Preview · drop, click or paste to upload", type="filepath",
                     sources=["upload", "clipboard"], height=220, buttons=[])
    gallery.select(pick_sample, None, pick, api_visibility="private")
    pick.change(preview, pick, shown, api_visibility="private")
    shown.upload(use_upload, shown, pick, api_visibility="private")
    shown.clear(lambda: None, None, pick, api_visibility="private")
    reload.click(refresh, None, [pick, gallery], api_visibility="private")
    return pick, shown


with gr.Blocks(title="2D → 3D") as demo:
    header = gr.HTML(header_html(), container=False, padding=False)

    with gr.Tabs(elem_id="mode-tabs"):
        with gr.Tab("Full 3D object"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=340, elem_classes="controls"):
                    step(1, "Choose an image", "One object per photo works best; the back and sides are predicted.")
                    o_pick, o_shown = source_picker()
                    step(2, "Settings")
                    o_res = gr.Slider(128, 384, 256, step=32, label="Mesh resolution",
                                      info="Higher is finer detail but slower.")
                    o_bg = gr.Checkbox(True, label="Remove background")
                    o_tt = gr.Checkbox(True, label="360° turntable preview",
                                       info="The slowest step without a GPU; turn off for a quick mesh.")
                    o_render_controls = renderer_picker()
                    o_btn = gr.Button("Generate 3D model", variant="primary", size="lg", elem_classes="cta")
                with gr.Column(scale=8, min_width=420, elem_classes="stage"):
                    o_mesh = gr.Model3D(label="3D model · drag to orbit", clear_color=VIEWER_BG, height=520,
                                        elem_classes="viewer")
                    o_saved = gr.Textbox(label="Status", interactive=False,
                                         placeholder="Results appear here and are saved under outputs/")
                    with gr.Tabs():
                        with gr.Tab("360° turntable"):
                            o_turn = gr.Image(show_label=False, interactive=False, height=320)
                        with gr.Tab("What the model sees"):
                            o_cut = gr.Image(show_label=False, interactive=False, height=320)
                        with gr.Tab("Blender"):
                            o_render = gr.Image(label="Final render", interactive=False, height=320)
                            o_blend = gr.File(label="Blender scene (.blend)", interactive=False)
            o_btn.click(run_object, [o_pick, o_res, o_bg, o_tt, *o_render_controls],
                        [o_mesh, o_turn, o_cut, o_render, o_blend, o_saved])

        with gr.Tab("2.5D depth & stereo"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=340, elem_classes="controls"):
                    step(1, "Choose an image", "Any photo works: scenes, people, landscapes.")
                    pick, shown = source_picker()
                    step(2, "Depth", "Depth Anything V2 estimates distance for every pixel.")
                    model = gr.Dropdown(list(MODELS), value="small", label="Depth model",
                                        info="small is fastest; metric-* give real-world meters.")
                    fov = gr.Slider(20, 120, 55, step=1, label="Camera FOV (°)")
                    edge = gr.Slider(0.005, 0.3, 0.04, step=0.005, label="Mesh edge cut threshold")
                    step(3, "Stereo")
                    strength = gr.Slider(0.0, 0.08, 0.03, step=0.005, label="Stereo strength")
                    conv = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="Convergence (screen plane)")
                    render_controls = renderer_picker()
                    btn = gr.Button("Convert", variant="primary", size="lg", elem_classes="cta")
                with gr.Column(scale=8, min_width=420, elem_classes="stage"):
                    mesh = gr.Model3D(label="Depth mesh · drag to orbit", clear_color=VIEWER_BG, height=520,
                                      elem_classes="viewer")
                    saved = gr.Textbox(label="Status", interactive=False,
                                       placeholder="Results appear here and are saved under outputs/")
                    with gr.Tabs():
                        with gr.Tab("3D photo (wiggle)"):
                            wiggle = gr.Image(show_label=False, interactive=False, height=320)
                        with gr.Tab("Depth map"):
                            depth = gr.Image(show_label=False, interactive=False, height=320)
                        with gr.Tab("Anaglyph"):
                            ana = gr.Image(label="Red/cyan glasses", interactive=False, height=320)
                        with gr.Tab("Side-by-side"):
                            sbs = gr.Image(show_label=False, interactive=False, height=320)
                        with gr.Tab("Blender"):
                            final_render = gr.Image(label="Final render", interactive=False, height=320)
                            blend_scene = gr.File(label="Blender scene (.blend)", interactive=False)
            btn.click(run, [pick, model, fov, edge, strength, conv, *render_controls],
                      [depth, mesh, ana, wiggle, sbs, final_render, blend_scene, saved])

    gr.HTML('<div class="app-footer">Results are saved under <code>outputs/&lt;image name&gt;/</code> · '
            'TripoSR (MIT) · Depth Anything V2 · Gradio</div>', container=False, padding=False)

    demo.load(device_status, None, [header, o_tt], show_progress="hidden", api_visibility="private")
    for picker, shown_image in ((o_pick, o_shown), (pick, shown)):  # fill previews for the preselected image
        demo.load(preview, picker, shown_image, show_progress="hidden", api_visibility="private")

if __name__ == "__main__":
    from img2three.object3d import warmup

    # Load TripoSR in the background while the UI starts, so the first "Generate 3D" isn't slow.
    threading.Thread(target=warmup, daemon=True).start()
    demo.launch(allowed_paths=[str(OUTPUTS), str(SAMPLES)], theme=THEME, css=CSS)
